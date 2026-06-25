#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Per-winner schedule plots with fetcher WGs colored by GATHER SOURCE RANK.

For each tuned winner in benchmark/ops/tuned_configs.json, runs the production
layout kernel with profile_rank_scopes=True (each remote gather emits a scope
named fetch_gather_r{src_rank}) and renders, per shape:
  *_gantt.png   per-WG Gantt (broken_barh), fetch bars colored by source rank
  *_lines.png   per-WG thin lines sorted by start time (dense traces stay legible)
GEMM phases (wait/dot/store) are muted greys so the fetch->rank structure stands out.

Because a fetch WG strided-loops over cells from different k-flag-groups -- and
each flag-group's K-blocks live on ONE source rank -- a single fetcher lane is a
sequence of differently-colored segments (rank0, rank1, ...), so the gather-rank
schedule is directly legible.

Process model: one SUBPROCESS PER SHAPE (the iris symmetric heap never frees, so
accumulating all 5 shapes in one context exhausts it / triggers a peer-refresh
fatal). The orchestrator (no GPU) drives the subprocesses, then plots offline from
the saved chrome_traces. The worker is the same file run with IRIS_GANTT_MNK set.

Run:
  HIP_VISIBLE_DEVICES=0,1,2,3 HSA_NO_SCRATCH_RECLAIM=1 \\
    .venv/bin/python benchmark/ops/profile_winners_gantt_by_rank.py
  # plot only (reuse existing traces, no GPU):
  IRIS_GANTT_PLOT_ONLY=1 .venv/bin/python benchmark/ops/profile_winners_gantt_by_rank.py

Outputs (next to repo root / cwd), per shape:
  ag_mm_rank_gantt_<mnk>.chrome_trace
  ag_mm_rank_gantt_<mnk>.png   (Gantt bars)
  ag_mm_rank_lines_<mnk>.png   (lines, sorted by start time)
"""

import json
import os
import subprocess
import sys
from collections import defaultdict, Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# .../iris/benchmark/ops/<file>  -> parents[2] == the iris package root, which is
# the worker's cwd and where Proton writes its chrome_trace files.
RUN_DIR = Path(__file__).resolve().parents[2]
_CONFIGS_JSON = SCRIPT_DIR / "tuned_configs.json"
_NUM_XCDS = 8

# Distinct, high-contrast colors for gather source ranks (ws=4 -> r0..r3).
_RANK_COLORS = {
    "fetch_gather_r0": "#1f77b4",  # blue
    "fetch_gather_r1": "#ff7f0e",  # orange
    "fetch_gather_r2": "#2ca02c",  # green
    "fetch_gather_r3": "#d62728",  # red
}
# GEMM / flag phases kept but muted so the rank-colored fetch structure dominates.
_CONTEXT_COLORS = {
    "fetch_flag_set": "#08306b",  # navy   — barrier + flag release
    "gemm_wait": "#dddddd",       # light grey — spin-wait (STALL)
    "gemm_dot": "#aaaaaa",        # mid grey   — MFMA
    "gemm_store_c": "#777777",    # dark grey  — epilogue
}
_COLORS = {**_RANK_COLORS, **_CONTEXT_COLORS}
_LEGEND_ORDER = list(_RANK_COLORS) + list(_CONTEXT_COLORS)


# ----------------------------------------------------------------------------
# Plotting (offline, no GPU)
# ----------------------------------------------------------------------------
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


def _lanes_sorted(events):
    """Group events into per-WG lanes, sorted by each WG's CAUSAL start time.

    No fetch/gemm role grouping -- the bar colors already distinguish roles. The
    sort key is when a WG begins doing REAL work, not its first recorded event:
      - GEMM lanes: first ``gemm_dot`` (actual MFMA). A GEMM WG's earliest event is
        its ``gemm_wait`` spin, which fires at launch and just BLOCKS until the
        fetcher delivers -- sorting on that would rank a stalled WG as "early". Using
        first dot orders consumers by when they actually compute.
      - Fetch lanes (no gemm_dot): first event = first gather = when it starts
        delivering data.
    Ties broken by (pid,tid). This makes producers correctly precede the consumers
    that depend on them, instead of the ~4ns launch-jitter artifact of first-event."""
    lanes = defaultdict(list)
    for name, ts, dur, key in events:
        lanes[key].append((name, ts, dur))

    def causal_start(evts):
        dots = [ts for n, ts, _ in evts if n == "gemm_dot"]
        return min(dots) if dots else min(ts for _, ts, _ in evts)

    keys = sorted(lanes.keys(), key=lambda k: (causal_start(lanes[k]), k))
    return lanes, keys


def _rank_share(events):
    share = Counter()
    for name, _, dur, _ in events:
        if name.startswith("fetch_gather_r"):
            share[name] += dur
    tot = sum(share.values()) or 1
    return "  ".join(f"r{k.split('_r')[1]}:{100*v//tot}%" for k, v in sorted(share.items()))


def _plot_gantt(trace_path, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.collections import PolyCollection

    events = _load_x_events(trace_path, _COLORS)
    if not events:
        print(f"[gantt] no events in {trace_path.name}; skipping")
        return
    lanes, keys = _lanes_sorted(events)
    # Cap the figure height: at thousands of lanes a per-lane 0.14"/row figure
    # becomes hundreds of inches and stalls. Clamp to ~24". Draw ALL bars as a
    # single batched PolyCollection (one Python-level add) -- per-event
    # broken_barh over tens of thousands of events takes minutes; this is ~instant.
    n = len(keys)
    height = min(24.0, max(2.5, 0.14 * n))
    fig, ax = plt.subplots(figsize=(15, height))
    bar_h = 0.84 if n <= 170 else 1.0  # touch when compressed
    verts, facecolors = [], []
    seen = set()
    for row, key in enumerate(keys):
        y0, y1 = row - bar_h / 2, row + bar_h / 2
        for name, ts, dur in lanes[key]:
            x0, x1 = ts, ts + max(dur, 1)
            verts.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            facecolors.append(_COLORS.get(name, "#000000"))
            seen.add(name)
    ax.add_collection(PolyCollection(verts, facecolors=facecolors, edgecolors="none"))
    ax.set_xlim(0, max(ts + dur for _, ts, dur, _ in events))
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(0, n, max(1, n // 40)))
    ax.set_ylabel("workgroup (sorted by start time)")
    ax.set_xlabel("time (ns, relative to first event)")
    ax.set_title(title)
    ax.invert_yaxis()
    handles = [Patch(facecolor=_COLORS[n], label=n) for n in _LEGEND_ORDER if n in seen]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"[gantt] wrote {out_png.name}  ({len(keys)} lanes)  gather-by-rank: {_rank_share(events)}")


def _plot_lines(trace_path, out_png, title):
    """One thin horizontal line per WG, sorted by start time (dense-trace legible)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    events = _load_x_events(trace_path, _COLORS)
    if not events:
        print(f"[lines] no events in {trace_path.name}; skipping")
        return
    lanes, keys = _lanes_sorted(events)
    n = len(keys)
    height_in = max(2.0, 0.05 * n)
    fig, ax = plt.subplots(figsize=(15, height_in), dpi=120)
    top, bottom = 0.94, 0.10
    fig.subplots_adjust(left=0.06, right=0.98, top=top, bottom=bottom)
    lw = (height_in * (top - bottom) / max(n, 1)) * 72.0
    seen = set()
    for row, key in enumerate(keys):
        for name, ts, dur in lanes[key]:
            ax.hlines(row, ts, ts + max(dur, 1), colors=_COLORS.get(name, "#000000"), linewidth=lw)
            seen.add(name)
    ax.set_yticks(range(0, n, max(1, n // 20)))
    ax.set_ylabel("workgroup (sorted by start time)")
    ax.set_xlabel("time (ns, relative to first event)")
    ax.set_title(title)
    ax.set_ylim(n - 0.5, -0.5)
    ax.margins(y=0)
    handles = [Patch(facecolor=_COLORS[n], label=n) for n in _LEGEND_ORDER if n in seen]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"[lines] wrote {out_png.name}  ({n} lanes)")


def _plot_shape(mnk: int, winner: dict):
    base = RUN_DIR / f"ag_mm_rank_gantt_{mnk}"
    trace = next((p for p in [Path(f"{base}.chrome_trace"), Path(f"{base}.chrome_trace.json"),
                              Path(f"{base}.json")] if p.exists()), None)
    if trace is None:
        print(f"[plot] no trace for {mnk}; skipping")
        return
    lay = winner["layout"]
    role = "coloc" if lay["fetch_xcds"] is None else f"fx{lay['fetch_xcds']}"
    sub = f"{lay['order']}/{role}/fk{lay['fetch_k']}/nfw{lay['n_fetch_wg']}/ngw{lay['n_gemm_wg']}"
    # Lines first (cheap, the primary view); Gantt bars second (height-capped).
    _plot_lines(trace, RUN_DIR / f"ag_mm_rank_lines_{mnk}.png",
                f"{mnk}^3 ws4 winner ({sub}) — lines by start time, fetch by source rank")
    _plot_gantt(trace, RUN_DIR / f"ag_mm_rank_gantt_{mnk}.png",
                f"{mnk}^3 ws4 winner ({sub}) — Gantt, fetch bars by source rank")


# ----------------------------------------------------------------------------
# Worker: profile ONE shape (run in its own process; selected by IRIS_GANTT_MNK)
# ----------------------------------------------------------------------------
def _worker(mnk: int):
    import torch
    import torch.distributed as dist
    import iris
    from iris.ops.all_gather_matmul_layout import (
        all_gather_matmul_layout, all_gather_matmul_layout_preamble,
    )
    from iris.ops.config import FusedConfig
    from iris.ops.schedule_layout import make_layout

    dist.init_process_group("nccl")
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    ctx = iris.iris(1 << 34)
    rank = ctx.get_rank(); ws = ctx.get_num_ranks()
    assert ws == 4, f"winners tuned at ws=4, got {ws}"

    winner = next(w for w in json.loads(_CONFIGS_JSON.read_text()) if w["mnk"] == mnk)
    lk = {k: v for k, v in winner["layout"].items() if k not in ("num_warps", "num_stages")}
    nw = winner["layout"]["num_warps"]; ns = winner["layout"]["num_stages"]
    M = N = K = mnk; Kl = K // ws
    dt = torch.float16
    bm, bn, bk = lk["block_size_m"], lk["block_size_n"], lk["block_size_k"]

    A = ctx.randn((M, Kl), dtype=dt, generator=torch.Generator("cuda").manual_seed(42 + rank))
    B = torch.randn((K, N), device="cuda", dtype=dt, generator=torch.Generator("cuda").manual_seed(123))
    C = ctx.zeros((M, N), dtype=dt)
    cfg = FusedConfig(block_size_m=bm, block_size_n=bn, block_size_k=bk, num_xcds=_NUM_XCDS)
    lay, _ = make_layout(num_xcds=_NUM_XCDS, M=M, N=N, K=K, K_local=Kl, world_size=ws, **lk)
    wsp = all_gather_matmul_layout_preamble(ctx, A, B, config=cfg, fetch_k=lk["fetch_k"])
    ctx.barrier()
    for _ in range(3):
        all_gather_matmul_layout(ctx, C, A, B, config=cfg, workspace=wsp, layout=lay, num_warps=nw, num_stages=ns)
    torch.cuda.synchronize(); ctx.barrier()
    C.zero_(); wsp.locks.zero_()
    all_gather_matmul_layout(
        ctx, C, A, B, config=cfg, workspace=wsp, layout=lay, num_warps=nw, num_stages=ns,
        profile=(rank == 0), profile_name=f"ag_mm_rank_gantt_{mnk}", profile_format="trace",
        profile_rank_scopes=True,
    )
    torch.cuda.synchronize(); ctx.barrier()
    if rank == 0:
        print(f"[worker] {mnk}^3 trace written", flush=True)


# ----------------------------------------------------------------------------
# Orchestrator: subprocess per shape, then plot offline
# ----------------------------------------------------------------------------
def main():
    winners = json.loads(_CONFIGS_JSON.read_text())
    shapes = [int(s) for s in os.environ["IRIS_GANTT_SHAPES"].split(",")] if os.environ.get("IRIS_GANTT_SHAPES") \
        else [w["mnk"] for w in winners]
    plot_only = os.environ.get("IRIS_GANTT_PLOT_ONLY", "0") == "1"

    if not plot_only:
        for mnk in shapes:
            env = dict(os.environ)
            env["IRIS_GANTT_MNK"] = str(mnk)
            env.setdefault("HIP_VISIBLE_DEVICES", "0,1,2,3")
            env["HSA_NO_SCRATCH_RECLAIM"] = "1"
            print(f"\n=== profiling {mnk}^3 (subprocess) ===", flush=True)
            subprocess.run(
                ["torchrun", "--nproc_per_node=4", f"--master_port={29570 + (mnk % 100)}",
                 str(Path(__file__).resolve())],
                env=env, cwd=str(RUN_DIR), check=True,
            )

    for mnk in shapes:
        w = next((x for x in winners if x["mnk"] == mnk), None)
        if w:
            _plot_shape(mnk, w)


if __name__ == "__main__":
    _mnk = os.environ.get("IRIS_GANTT_MNK")
    if _mnk is not None:
        _worker(int(_mnk))   # we are inside torchrun (a per-shape worker)
    else:
        main()               # orchestrator
