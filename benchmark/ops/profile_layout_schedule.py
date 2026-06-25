#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Fine-grained Proton schedule analysis for all_gather_matmul_layout.

Drives the PRODUCTION layout kernel with profile_scopes=True (fine-grained leaf
scopes: fetch_gather / fetch_flag_set / gemm_wait / gemm_dot / gemm_store_c) and
renders the three views that make a producer/consumer schedule legible -- ported
from examples/33_proton_all_gather_matmul_gantt:

  *_lines_combined.png   one thin line per WG, sorted by start time: shows when
                         fetchers vs GEMM WGs run and how much red (gemm_wait/STALL)
                         each GEMM lane carries.
  *_lines_by_pid.png     same, sorted by launch pid -> reveals the role/XCD layout
                         directly (which pids fetch, which compute).
  *_cu_activity.png      stacked count of active WGs (~active CUs) per leaf scope
                         over time, vs launched-WG line: shows CU occupancy /
                         quantization and whether fetchers VACATE CUs for GEMM.

Which layout is profiled is chosen by env (defaults to the 4096^3 winner):
  IRIS_PROF_MNK / _M/_N/_K, IRIS_PROF_FK, IRIS_PROF_FXCD, IRIS_PROF_ORDER,
  IRIS_PROF_NFW, IRIS_PROF_NGW, IRIS_PROF_GM, IRIS_PROF_BM/_BN/_BK.

Run (rank 0 writes trace + PNGs):
  HIP_VISIBLE_DEVICES=0,1,2,3 HSA_NO_SCRATCH_RECLAIM=1 \\
    python tests/run_tests_distributed.py \\
      benchmark/ops/profile_layout_schedule.py --num_ranks 4 -s
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ops.all_gather_matmul_layout import (
    all_gather_matmul_layout,
    all_gather_matmul_layout_preamble,
)
from iris.ops.config import FusedConfig
from iris.ops.schedule_layout import make_layout

# Leaf scopes emitted by the kernel under profile_scopes=True.
LEAF_SCOPE_COLORS = {
    "fetch_gather": "#1f77b4",    # blue   — remote gather + staged store
    "fetch_flag_set": "#08306b",  # navy   — barrier + flag release
    "gemm_wait": "#d62728",       # red    — spin-wait on fetcher flag = STALL
    "gemm_dot": "#9467bd",        # purple — MFMA accumulation
    "gemm_store_c": "#2ca02c",    # green  — epilogue HBM write
}
_BASENAME = "ag_mm_schedule"


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else default


def _load_x_events(path, scope_filter):
    with open(path) as f:
        trace = json.load(f)
    events = [e for e in trace.get("traceEvents", [])
              if e.get("ph") == "X" and e.get("name") in scope_filter]
    if not events:
        return []
    t0 = min(int(e["ts"]) for e in events)
    return [(e["name"], int(e["ts"]) - t0, int(e.get("dur", 0)), (e.get("pid", 0), e.get("tid", 0)))
            for e in events]


def _plot_lines(trace_path, out_png, scope_colors, sort_by="start"):
    """One thin horizontal line per WG (dense traces stay readable as a stripe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    events = _load_x_events(trace_path, scope_colors)
    if not events:
        print(f"[lines] no matching events in {trace_path}; skipping {out_png.name}")
        return
    lanes = defaultdict(list)
    for name, ts, dur, key in events:
        lanes[key].append((name, ts, dur))

    if sort_by == "pid":
        keys = sorted(lanes.keys())  # (pid, tid) launch order -> role/XCD layout
        ylabel = "workgroup (launch pid -> role/XCD layout)"
        suffix = "by launch pid"
    else:
        keys = sorted(lanes.keys(), key=lambda k: (min(ts for _, ts, _ in lanes[k]), k))
        ylabel = "workgroup (sorted by start time)"
        suffix = "by start time"

    n = len(keys)
    height_in = max(2.0, 0.05 * n)
    fig, ax = plt.subplots(figsize=(14, height_in), dpi=120)
    top, bottom = 0.94, 0.10
    fig.subplots_adjust(left=0.06, right=0.98, top=top, bottom=bottom)
    lw = (height_in * (top - bottom) / max(n, 1)) * 72.0

    seen = set()
    for row, key in enumerate(keys):
        for name, ts, dur in lanes[key]:
            ax.hlines(row, ts, ts + max(dur, 1), colors=scope_colors[name], linewidth=lw)
            seen.add(name)
    ax.set_yticks(range(0, n, max(1, n // 20)))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (ns, relative to first event)")
    ax.set_title(f"layout schedule lines — {suffix} — {trace_path.name}")
    ax.set_ylim(n - 0.5, -0.5)
    ax.margins(y=0)
    ax.legend(handles=[Patch(facecolor=scope_colors[x], label=x) for x in sorted(seen)],
              loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[lines] wrote {out_png}  ({n} WG lanes)")


def _plot_cu(trace_path, out_png, scope_colors, num_bins=800):
    """Stacked active-WG (~active-CU) count per leaf scope over time, vs launched."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    events = _load_x_events(trace_path, scope_colors)
    if not events:
        print(f"[cu] no leaf events in {trace_path}; skipping {out_png.name}")
        return
    t_end = max(ts + dur for _, ts, dur, _ in events)
    if t_end <= 0:
        print(f"[cu] zero-duration trace; skipping {out_png.name}")
        return
    edges = np.linspace(0, t_end, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    names = list(scope_colors.keys())
    counts = {n: np.zeros(num_bins, dtype=np.int32) for n in names}
    for name, ts, dur, _ in events:
        lo = max(np.searchsorted(edges, ts, side="right") - 1, 0)
        hi = min(np.searchsorted(edges, ts + dur, side="left"), num_bins)
        if hi > lo:
            counts[name][lo:hi] += 1
    total = sum(counts.values())
    launched = len({key for _, _, _, key in events})
    peak = int(total.max()) if len(total) else 0

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.stackplot(centers, [counts[n] for n in names], labels=names,
                 colors=[scope_colors[n] for n in names], alpha=0.9, edgecolor="none")
    ax.axhline(launched, color="black", linestyle="--", linewidth=0.8,
               label=f"launched WGs = {launched}")
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, max(launched, peak) * 1.05)
    ax.set_xlabel("time (ns, relative to first event)")
    ax.set_ylabel("active WGs (~ active CUs)")
    ax.set_title(f"layout CU activity — peak {peak} / {launched} launched WGs")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[cu] wrote {out_png}")


def test_profile_layout_schedule():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    ctx = iris.iris(1 << 34)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    if world_size != 4:
        pytest.skip(f"profile fixed to 4 ranks, got {world_size}")

    mnk = _env_int("IRIS_PROF_MNK", 4096)
    M = _env_int("IRIS_PROF_M", mnk); N = _env_int("IRIS_PROF_N", mnk); K = _env_int("IRIS_PROF_K", mnk)
    bm = _env_int("IRIS_PROF_BM", 256); bn = _env_int("IRIS_PROF_BN", 256); bk = _env_int("IRIS_PROF_BK", 64)
    fk = _env_int("IRIS_PROF_FK", 16)
    gm = _env_int("IRIS_PROF_GM", 1)
    fxcd = _env_int("IRIS_PROF_FXCD", 2)
    order = os.environ.get("IRIS_PROF_ORDER", "mtile")
    nfw = _env_int("IRIS_PROF_NFW", 8)
    ngw = _env_int("IRIS_PROF_NGW", 32)
    num_xcds = 8
    K_local = K // world_size
    dtype = torch.float16

    layout, _ = make_layout(
        num_xcds=num_xcds, M=M, N=N, K=K, K_local=K_local, world_size=world_size,
        fetch_m=1, fetch_k=fk, group_m=gm, block_size_m=bm, block_size_n=bn,
        block_size_k=bk, order=order, fetch_xcds=(fxcd if fxcd > 0 else None),
        n_fetch_wg=nfw, n_gemm_wg=ngw,
    )
    config = FusedConfig(block_size_m=bm, block_size_n=bn, block_size_k=bk, num_xcds=num_xcds)
    A = ctx.randn((M, K_local), dtype=dtype, generator=torch.Generator("cuda").manual_seed(42 + rank))
    B = torch.randn((K, N), device="cuda", dtype=dtype, generator=torch.Generator("cuda").manual_seed(123))
    C = ctx.zeros((M, N), dtype=dtype)
    wsp = all_gather_matmul_layout_preamble(ctx, A, B, config=config, fetch_k=fk)
    ctx.barrier()

    for _ in range(3):  # warm up / compile
        all_gather_matmul_layout(ctx, C, A, B, config=config, workspace=wsp, layout=layout)
    torch.cuda.synchronize(); ctx.barrier()

    C.zero_(); wsp.locks.zero_()
    all_gather_matmul_layout(
        ctx, C, A, B, config=config, workspace=wsp, layout=layout,
        profile=(rank == 0), profile_name=_BASENAME, profile_format="trace",
        profile_scopes=True,
    )
    torch.cuda.synchronize(); ctx.barrier()

    if rank == 0:
        cands = [Path(f"{_BASENAME}.chrome_trace"), Path(f"{_BASENAME}.chrome_trace.json"),
                 Path(f"{_BASENAME}.json")]
        trace = next((p for p in cands if p.exists()), None)
        assert trace is not None, f"no proton trace; present: {list(Path('.').glob(_BASENAME + '*'))}"
        tag = f"fk{fk}_fx{fxcd}_{order}_nfw{nfw}_ngw{ngw}"
        _plot_lines(trace, Path(f"{_BASENAME}_lines_start_{tag}.png"), LEAF_SCOPE_COLORS, "start")
        _plot_lines(trace, Path(f"{_BASENAME}_lines_pid_{tag}.png"), LEAF_SCOPE_COLORS, "pid")
        _plot_cu(trace, Path(f"{_BASENAME}_cu_{tag}.png"), LEAF_SCOPE_COLORS)
        print(f"[schedule] trace: {trace.resolve()}")
    ctx.barrier()
