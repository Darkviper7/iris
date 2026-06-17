#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Proton intra-kernel profiling demo for ``all_gather_matmul_hbm_buffer``.

Goal
----
Prove that Triton's Proton instrumentation backend can produce the same kind
of stacked Gantt chart that iris's built-in device tracing produces today
(``iris/host/tracing/core.py``), but via the lightweight tritonBLAS-style
``pl.scope`` markers compiled into the kernel — zero device-side atomics or
buffers when ``profile=False``.

What this script does
---------------------
1. Launches the example-local ``all_gather_matmul_hbm_buffer_proton`` wrapper
   with ``profile=True``,
   which gates a ``proton.start(..., data="trace", backend="instrumentation")``
   session around the kernel launch. Proton writes
   ``hbm_buffer_all_gather_matmul.chrome_trace`` (standard Chrome Trace Event
   Format JSON, Perfetto-loadable).
2. Reads that JSON back and renders a matplotlib ``broken_barh`` Gantt with
   one row per workgroup, bars colored by scope name. Saves
   ``hbm_buffer_gantt.png`` next to the script.
3. Also runs once with ``profile_format="tree"`` to emit
   ``hbm_buffer_all_gather_matmul.hatchet`` for ``proton-viewer``.

Run
---
::

    python examples/33_proton_all_gather_matmul_gantt/profile_and_plot_gantt.py \\
        --num_ranks 8

Uses ``torch.multiprocessing.spawn`` (same pattern as
``examples/21_gemm_one_shot_all_reduce_independent/benchmark.py``) so no
torchrun wrapper is needed.

The Gantt PNG and chrome_trace are written next to the script. The
chrome_trace is the apples-to-apples equivalent of iris's
``ctx.tracing.export(...)`` output — drop it into https://ui.perfetto.dev for
the same browser-rendered timeline iris users see today.

Add ``--tritonparse`` alongside ``--mode {proton,iris}`` to also capture
Triton compile-time IR (TTIR/TTGIR/LLIR/AMDGCN) and launch metadata for
viewing in https://meta-pytorch.org/tritonparse/.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import iris
from all_gather_matmul_hbm_buffer_proton import (
    all_gather_matmul_hbm_buffer as all_gather_matmul_hbm_buffer_proton,
    all_gather_matmul_hbm_buffer_preamble,
)
from iris.ops.all_gather_matmul_hbm_buffer import (
    all_gather_matmul_hbm_buffer as all_gather_matmul_hbm_buffer_iris,
)
from iris.ops.config import FusedConfig

# auto_config lives at benchmark/ops/all_gather_matmul/auto_config.py and the
# `benchmark/` tree isn't a regular package, so expose it via sys.path.
sys.path.insert(0, str(_REPO_ROOT / "benchmark" / "ops" / "all_gather_matmul"))
from auto_config import select_ag_mm_config  # noqa: E402


TRACE_BASENAME = "hbm_buffer_all_gather_matmul"

# Leaf scopes plotted on the per-WG Gantt and stacked on the CU activity chart.
LEAF_SCOPE_COLORS = {
    "all_gather_stages": "#1f77b4",   # blue   — communication (fetcher branch)
    "gemm_wait": "#d62728",           # red    — spin-wait on fetcher flags
    "gemm_dot": "#9467bd",            # purple — inner tl.dot accumulation
    "gemm_store_c": "#2ca02c",        # green  — epilogue HBM write
}

# Outer wrapper scopes plotted on a separate Gantt.
OUTER_SCOPE_COLORS = {
    "hbm_buffer_all_gather_matmul": "#888888",  # grey   — whole-kernel span
    "all_gather": "#1f77b4",                    # blue   — fetcher branch
    "gemm": "#ff7f0e",                          # orange — compute branch
}

# Combined view: outer all_gather (fetcher branch as one bar) + GEMM leaf phases.
COMBINED_SCOPE_COLORS = {
    "all_gather": "#1f77b4",       # blue   — whole communication branch
    "gemm_wait": "#d62728",        # red    — spin-wait on fetcher flags
    "gemm_dot": "#9467bd",         # purple — inner tl.dot accumulation
    "gemm_store_c": "#2ca02c",     # green  — epilogue HBM write
}

# iris built-in trace event names (from iris/ops/all_gather_matmul_hbm_buffer.py:
# fetch / compute / wait). Reuses the same plotting helpers — only the scope
# filter / color map changes.
IRIS_EVENT_COLORS = {
    "fetch":   "#1f77b4",  # blue   — fetcher WGs (one event per fetcher WG)
    "compute": "#ff7f0e",  # orange — GEMM WG outer span
    "wait":    "#d62728",  # red    — per-flag-group spin-wait inside GEMM
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-m", type=int, default=256, help="Rows of A")
    p.add_argument("-n", type=int, default=256, help="Columns of B")
    p.add_argument("--k_local", type=int, default=128, help="Columns of A per rank (K_local)")
    p.add_argument("--transpose", default="NN", choices=["NN", "NT", "TN", "TT"],
                   help="GEMM transpose mode passed to select_ag_mm_config")
    p.add_argument("--heap_size", type=int, default=1 << 30)
    p.add_argument("--datatype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--no_plot", action="store_true", help="Skip matplotlib Gantt (still writes trace)")
    p.add_argument("-r", "--num_ranks", type=int, default=8, help="Number of ranks/processes to spawn")
    p.add_argument("--mode", default="proton", choices=["proton", "iris"],
                   help="proton: pl.scope chrome_trace via Proton (rank 0 only). "
                        "iris: built-in ctx.tracing (all ranks, merged Perfetto JSON).")
    p.add_argument("--max_trace_events", type=int, default=1_000_000,
                   help="iris mode only: per-rank ring-buffer capacity.")
    p.add_argument("--tritonparse", action="store_true",
                   help="Also capture Triton compile-time IR + launch metadata "
                        "via tritonparse (works alongside --mode).")
    p.add_argument("--tritonparse_log_dir", type=Path,
                   default=SCRIPT_DIR / "tritonparse_logs",
                   help="Per-rank NDJSON output dir (tritonparse).")
    p.add_argument("--tritonparse_out_dir", type=Path,
                   default=SCRIPT_DIR / "tritonparse_output",
                   help="unified_parse() output dir (tritonparse).")
    return p.parse_args()


def _load_x_events(chrome_trace_path: Path, scope_filter):
    """Return list of (name, ts_rel, dur, (pid, tid)) for matching X events."""
    with open(chrome_trace_path) as f:
        trace = json.load(f)
    events = [
        e
        for e in trace.get("traceEvents", [])
        if e.get("ph") == "X" and e.get("name") in scope_filter
    ]
    if not events:
        return []
    t0 = min(int(e["ts"]) for e in events)
    return [
        (e["name"], int(e["ts"]) - t0, int(e.get("dur", 0)), (e.get("pid", 0), e.get("tid", 0)))
        for e in events
    ]


def _plot_gantt(chrome_trace_path: Path, out_png: Path, scope_colors: dict, title: str):
    """Render a Gantt PNG, one row per (pid, tid) lane, filtered to scope_colors."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print(f"[gantt] matplotlib not installed; skipping {out_png}")
        return

    events = _load_x_events(chrome_trace_path, scope_colors)
    if not events:
        print(f"[gantt] no matching events in {chrome_trace_path}; skipping {out_png.name}")
        return

    lanes_by_wg = defaultdict(list)
    for name, ts, dur, key in events:
        lanes_by_wg[key].append((name, ts, dur))

    # Sort WG rows by earliest start time so the chart reads top-down in
    # launch / scheduling order; ties broken by (pid, tid) for stability.
    wg_keys = sorted(
        lanes_by_wg.keys(),
        key=lambda k: (min(ts for _, ts, _ in lanes_by_wg[k]), k),
    )
    wg_label = {k: f"wg {i}" for i, k in enumerate(wg_keys)}

    height = max(2.0, 0.18 * len(wg_keys))
    fig, ax = plt.subplots(figsize=(14, height))

    scopes_seen = set()
    for row, key in enumerate(wg_keys):
        for name, ts, dur in lanes_by_wg[key]:
            ax.broken_barh(
                [(ts, max(dur, 1))], (row - 0.4, 0.8),
                facecolors=scope_colors.get(name, "#888888"), edgecolors="none",
            )
            scopes_seen.add(name)

    ax.set_yticks(range(len(wg_keys)))
    ax.set_yticklabels([wg_label[k] for k in wg_keys], fontsize=6)
    ax.set_xlabel("time (ns, relative to first event)")
    ax.set_title(title)
    ax.invert_yaxis()

    legend_handles = [Patch(facecolor=scope_colors[n], label=n) for n in sorted(scopes_seen)]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[gantt] wrote {out_png}")


def _plot_lines(chrome_trace_path: Path, out_png: Path, scope_colors: dict,
                sort_by: str = "start", mode: str = "proton"):
    """Same scopes as the combined Gantt, but each WG is a thin horizontal line.

    Lines instead of bars make dense traces readable: hundreds of WGs collapse
    into a stripe pattern where you can still see per-scope timing without
    bars overlapping into a solid block.

    sort_by:
      - "start" (default): earliest event ts. Reads top-down in execution order.
      - "pid":   raw Proton (pid, tid) tuple = kernel-launch order. Reveals
                 the interleaved [fetch | gemm | fetch | gemm | ...] stage
                 layout directly, since `stage_pid` and `my_stage` are
                 deterministic functions of the launch pid.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print(f"[lines] matplotlib not installed; skipping {out_png}")
        return

    events = _load_x_events(chrome_trace_path, scope_colors)
    if not events:
        print(f"[lines] no matching events in {chrome_trace_path}; skipping {out_png.name}")
        return

    lanes_by_wg = defaultdict(list)
    for name, ts, dur, key in events:
        lanes_by_wg[key].append((name, ts, dur))

    if sort_by == "pid":
        wg_keys = sorted(lanes_by_wg.keys())  # (pid, tid) tuple — launch order
        ylabel = "workgroup (sorted by launch pid → stage layout)"
        title_suffix = "sorted by stage/pid"
    else:
        wg_keys = sorted(
            lanes_by_wg.keys(),
            key=lambda k: (min(ts for _, ts, _ in lanes_by_wg[k]), k),
        )
        ylabel = "workgroup (sorted by start time)"
        title_suffix = "sorted by start time"

    num_rows = len(wg_keys)
    dpi = 120
    height_in = max(2.0, 0.05 * num_rows)
    fig, ax = plt.subplots(figsize=(14, height_in), dpi=dpi)
    # Lock axes geometry so we can compute an exact linewidth that makes
    # adjacent rows touch with zero gap.
    top, bottom = 0.94, 0.10
    fig.subplots_adjust(left=0.06, right=0.98, top=top, bottom=bottom)
    axes_h_in = height_in * (top - bottom)
    linewidth_pt = (axes_h_in / num_rows) * 72.0

    scopes_seen = set()
    for row, key in enumerate(wg_keys):
        for name, ts, dur in lanes_by_wg[key]:
            ax.hlines(
                row, ts, ts + max(dur, 1),
                colors=scope_colors.get(name, "#888888"),
                linewidth=linewidth_pt,
            )
            scopes_seen.add(name)

    ax.set_yticks(range(0, num_rows, max(1, num_rows // 20)))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (ns, relative to first event)")
    ax.set_title(f"{mode} combined line chart — {title_suffix} — {chrome_trace_path.name}")
    ax.set_ylim(num_rows - 0.5, -0.5)
    ax.margins(y=0)

    legend_handles = [Patch(facecolor=scope_colors[n], label=n) for n in sorted(scopes_seen)]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[lines] wrote {out_png}")


def _plot_cu(chrome_trace_path: Path, out_png: Path, scope_colors: dict,
             num_bins: int = 800, mode: str = "proton"):
    """Stacked CU-activity chart: at each time bin, count WGs in each leaf scope.

    Each Proton (pid, tid) lane corresponds to one workgroup, which AMD/HIP
    assigns to a single CU at a time. So the stacked height at time t equals
    the number of *active* CUs broken down by what they're doing. The gap
    between the stack and the launched-WG count = idle / unscheduled CUs
    (the 'quantization' loss you see when a wave finishes early).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"[cu] matplotlib/numpy not installed; skipping {out_png}")
        return

    events = _load_x_events(chrome_trace_path, scope_colors)
    if not events:
        print(f"[cu] no leaf events in {chrome_trace_path}; skipping {out_png.name}")
        return

    t_end = max(ts + dur for _, ts, dur, _ in events)
    if t_end <= 0:
        print(f"[cu] zero-duration trace; skipping {out_png.name}")
        return

    bin_edges = np.linspace(0, t_end, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    scope_names = list(scope_colors.keys())
    counts = {n: np.zeros(num_bins, dtype=np.int32) for n in scope_names}

    # For each event, increment all bins it covers. We use overlap-as-fraction:
    # a bin is "active" if the event's [ts, ts+dur] intersects it.
    for name, ts, dur, _ in events:
        lo = np.searchsorted(bin_edges, ts, side="right") - 1
        hi = np.searchsorted(bin_edges, ts + dur, side="left")
        lo = max(lo, 0)
        hi = min(hi, num_bins)
        if hi > lo:
            counts[name][lo:hi] += 1

    total_active = sum(counts.values())
    # Total launched lanes (= grid_size for this rank).
    launched_lanes = len({key for _, _, _, key in events})
    peak_active = int(total_active.max()) if len(total_active) else 0

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.stackplot(
        bin_centers,
        [counts[n] for n in scope_names],
        labels=scope_names,
        colors=[scope_colors[n] for n in scope_names],
        alpha=0.9,
        edgecolor="none",
    )
    ax.axhline(launched_lanes, color="black", linestyle="--", linewidth=0.8,
               label=f"launched WGs = {launched_lanes}")
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, max(launched_lanes, peak_active) * 1.05)
    ax.set_xlabel("time (ns, relative to first event)")
    ax.set_ylabel("active WGs (≈ active CUs)")
    ax.set_title(
        f"{mode} CU activity (quantization) — peak {peak_active} / {launched_lanes} launched WGs"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[cu] wrote {out_png}")


def _worker(local_rank: int, world_size: int, init_url: str, args):
    """Per-rank entry point launched by ``torch.multiprocessing.spawn``."""
    torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    # Initialize tritonparse BEFORE any Triton kernel JITs. The handler reads
    # torch.distributed.get_rank() to encode rank in each NDJSON filename.
    #
    # NOTE: enable_trace_launch is forced OFF because tritonparse installs a
    # single-object launch hook (`knobs.runtime.launch_enter_hook = LaunchHookImpl()`)
    # which clashes with Proton's `HookManager.register(...)` (Proton expects
    # the hook to be a set-like object exposing `.add(...)`, and crashes with
    # `'LaunchHookImpl' object has no attribute 'add'`). With launch tracing
    # off we still capture the full compile pipeline (TTIR/TTGIR/LLIR/AMDGCN
    # + source mapping), which is the primary value here.
    if args.tritonparse:
        import tritonparse.structured_logging
        tritonparse.structured_logging.init(
            str(args.tritonparse_log_dir),
            enable_trace_launch=False,
            enable_more_tensor_information=False,
        )

    ctx = iris.iris(heap_size=args.heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.datatype]
    M, K_local, N = args.m, args.k_local, args.n
    K = K_local * world_size

    torch.manual_seed(42 + rank)
    A_sharded = ctx.randn((M, K_local), dtype=dtype)
    torch.manual_seed(0)
    B = ctx.randn((K, N), dtype=dtype)
    output = ctx.zeros((M, N), dtype=dtype)

    # NOTE: auto_config / select_ag_mm_config disabled — its downstream
    # autotuning makes the demo stall. Use a small fixed config matching
    # tests/ops/test_all_gather_matmul.py instead.
    #
    # ag_mm = select_ag_mm_config(M, N, K, world_size=world_size, transpose=args.transpose)
    # config = ag_mm.to_fused_config()
    # hbm_params = ag_mm.hbm_buffer_params
    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    num_k_blocks = K // config.block_size_k
    k_per_flag = 32
    # while k_per_flag * 2 <= 8 and num_k_blocks % (k_per_flag * 2) == 0:
    #     k_per_flag *= 2
    hbm_params = {
        "k_per_flag": k_per_flag,
        "num_fetch_sms": 4,
        "num_fetch_stages": 1,
        "first_stage_fetch_sms": 16,
    }
    if rank == 0:
        print(f"[config] FusedConfig: {config}")
        print(f"[config] hbm_buffer_params: {hbm_params}")

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx, A_sharded, B, config=config, k_per_flag=hbm_params["k_per_flag"]
    )
    ctx.barrier()

    chdir_back = os.getcwd()
    os.chdir(SCRIPT_DIR)  # outputs land next to this script
    try:
        if args.mode == "proton":
            # --- run #1: chrome_trace (timeline / Gantt) ---
            all_gather_matmul_hbm_buffer_proton(
                ctx, output, A_sharded, B,
                config=config, workspace=workspace,
                k_per_flag=hbm_params.get("k_per_flag"),
                num_fetch_sms=hbm_params.get("num_fetch_sms"),
                num_fetch_stages=hbm_params.get("num_fetch_stages"),
                first_stage_fetch_sms=hbm_params.get("first_stage_fetch_sms"),
                profile=(rank == 0),
                profile_name=TRACE_BASENAME, profile_format="trace",
            )
            # --- run #2: hatchet (aggregate cycles per scope) ---
            all_gather_matmul_hbm_buffer_proton(
                ctx, output, A_sharded, B,
                config=config, workspace=workspace,
                k_per_flag=hbm_params.get("k_per_flag"),
                num_fetch_sms=hbm_params.get("num_fetch_sms"),
                num_fetch_stages=hbm_params.get("num_fetch_stages"),
                first_stage_fetch_sms=hbm_params.get("first_stage_fetch_sms"),
                profile=(rank == 0),
                profile_name=TRACE_BASENAME, profile_format="tree",
            )
        else:
            # iris built-in device tracing — enable on every rank, then a
            # single launch with trace=True, then merge-export to one JSON.
            ctx.tracing.enable(max_events=args.max_trace_events)
            ctx.barrier()
            ctx.tracing.reset()
            ctx.barrier()

            all_gather_matmul_hbm_buffer_iris(
                ctx, output, A_sharded, B,
                config=config, workspace=workspace,
                k_per_flag=hbm_params.get("k_per_flag"),
                num_fetch_sms=hbm_params.get("num_fetch_sms"),
                num_fetch_stages=hbm_params.get("num_fetch_stages"),
                first_stage_fetch_sms=hbm_params.get("first_stage_fetch_sms"),
                trace=True,
            )
            torch.cuda.synchronize()
            ctx.barrier()
            # merge=True → rank 0 collects every rank's events into one JSON.
            ctx.tracing.export(f"{TRACE_BASENAME}_iris.json", merge=True)
    finally:
        os.chdir(chdir_back)

    torch.cuda.synchronize()
    ctx.barrier()

    # Run tritonparse post-processing on rank 0 only; merges per-rank NDJSON
    # logs from all ranks into the unified_parse output directory.
    if args.tritonparse and rank == 0:
        import tritonparse.parse.utils
        tritonparse.parse.utils.unified_parse(
            source=str(args.tritonparse_log_dir),
            out=str(args.tritonparse_out_dir),
            all_ranks=True,
            overwrite=True,
        )
        print(f"[tritonparse] parsed traces: {args.tritonparse_out_dir}")
        print("[tritonparse] view at https://meta-pytorch.org/tritonparse/")

    if rank == 0:
        if args.mode == "proton":
            trace_file = SCRIPT_DIR / f"{TRACE_BASENAME}.chrome_trace"
            hatchet = SCRIPT_DIR / f"{TRACE_BASENAME}.hatchet"
            colors_leaf = LEAF_SCOPE_COLORS
            colors_outer = OUTER_SCOPE_COLORS
            colors_combined = COMBINED_SCOPE_COLORS
            prefix = "hbm_buffer"
        else:
            # Iris merge=True writes <name>_merged.json on rank 0.
            base = TRACE_BASENAME + "_iris"
            merged = SCRIPT_DIR / f"{base}_merged.json"
            per_rank0 = SCRIPT_DIR / f"{base}_rank0.json"
            trace_file = merged if merged.exists() else per_rank0
            hatchet = None
            # iris emits exactly three event types — reuse the same plotters.
            colors_leaf = IRIS_EVENT_COLORS
            colors_outer = IRIS_EVENT_COLORS
            colors_combined = IRIS_EVENT_COLORS
            prefix = "hbm_buffer_iris"

        gantt_leaf = SCRIPT_DIR / f"{prefix}_gantt_leaf.png"
        gantt_outer = SCRIPT_DIR / f"{prefix}_gantt_outer.png"
        gantt_combined = SCRIPT_DIR / f"{prefix}_gantt_combined.png"
        lines_combined = SCRIPT_DIR / f"{prefix}_lines_combined_kpf{hbm_params.get("k_per_flag")}_fs{hbm_params.get("num_fetch_sms")}_nfs{hbm_params.get("num_fetch_stages")}_fsf{hbm_params.get("first_stage_fetch_sms")}.png"
        lines_by_pid = SCRIPT_DIR / f"{prefix}_lines_by_pid.png"
        cu_activity = SCRIPT_DIR / f"{prefix}_cu_activity.png"

        print()
        print(f"{args.mode} outputs:")
        print(f"  trace : {trace_file}  ({'OK' if trace_file.exists() else 'MISSING'})")
        if hatchet is not None:
            print(f"  hatchet : {hatchet}  ({'OK' if hatchet.exists() else 'MISSING'})")

        if trace_file.exists() and not args.no_plot:
            _plot_gantt(trace_file, gantt_leaf, colors_leaf,
                        title=f"{args.mode} leaf-scope Gantt — {trace_file.name}")
            _plot_gantt(trace_file, gantt_outer, colors_outer,
                        title=f"{args.mode} outer-scope Gantt — {trace_file.name}")
            _plot_gantt(trace_file, gantt_combined, colors_combined,
                        title=f"{args.mode} combined Gantt — {trace_file.name}")
            _plot_lines(trace_file, lines_combined, colors_combined, sort_by="start", mode=args.mode)
            _plot_lines(trace_file, lines_by_pid, colors_combined, sort_by="pid", mode=args.mode)
            _plot_cu(trace_file, cu_activity, colors_combined, mode=args.mode)

        print()
        print("View options:")
        print(f"  • Open {trace_file.name} in https://ui.perfetto.dev")
        if hatchet is not None and hatchet.exists():
            print(f"  • proton-viewer -m normalized_cycles {hatchet}")
        print(f"  • {gantt_leaf.name}")
        print(f"  • {gantt_outer.name}")
        print(f"  • {gantt_combined.name}")
        print(f"  • {lines_combined.name}")
        print(f"  • {lines_by_pid.name}")
        print(f"  • {cu_activity.name}")
        if args.tritonparse:
            print(f"  • tritonparse: drop {args.tritonparse_out_dir} into "
                  "https://meta-pytorch.org/tritonparse/")

    ctx.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    init_url = "tcp://127.0.0.1:29500"
    mp.spawn(
        fn=_worker,
        args=(args.num_ranks, init_url, args),
        nprocs=args.num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
