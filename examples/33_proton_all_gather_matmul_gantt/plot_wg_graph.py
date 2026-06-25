#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fetcher↔GEMM dependency graph for all_gather_matmul_hbm_buffer.

Builds a directed graph where:
  - Nodes = every WG (label = pid), blue=fetcher, orange=GEMM
  - Edges = fetcher f → GEMM g when f signals a flag that g waits on
             (flag_idx = m_tile * NUM_FLAG_GROUPS_K + k_flag_group)
  - Partitions = stage bands (Y) × scheduling wave (X, wave = pid // num_cus)

No GPU, no kernel launch, no iris import needed.

Run (small shapes for legibility)
---
    # tiny contiguous
    python plot_wg_graph.py -m 256 -n 256 --k_local 128 --num_ranks 2 \
      --bm 128 --bn 128 --bk 64 --gm 1 --nfs 1 --fs 2 --fsf 4 --kpf 2 --num_cus 16

    # small interleaved
    python plot_wg_graph.py -m 256 -n 256 --k_local 128 --num_ranks 2 \
      --bm 128 --bn 128 --bk 64 --gm 2 --nfs 1 --fs 2 --fsf 4 --kpf 2 --num_cus 16 \
      --interleave_roles

    # two stages
    python plot_wg_graph.py -m 512 -n 256 --k_local 128 --num_ranks 2 \
      --bm 128 --bn 128 --bk 64 --gm 2 --nfs 2 --fs 2 --fsf 4 --kpf 2 --num_cus 16
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

SCRIPT_DIR = Path(__file__).resolve().parent


def ceil_div(a, b):
    return -(-a // b)


# ---------------------------------------------------------------------------
# Extended pid decode — returns full metadata per WG
# ---------------------------------------------------------------------------

class WGInfo:
    __slots__ = ("pid", "stage", "local_pid", "wave",
                 "is_fetch",
                 # fetchers
                 "produced_flags",          # set of (m_tile, k_flag_group)
                 # GEMM
                 "pid_m", "pid_n",
                 "consumed_flags")          # set of (m_tile, k_fg)

    def __init__(self, pid, stage, local_pid, wave, is_fetch):
        self.pid = pid
        self.stage = stage
        self.local_pid = local_pid
        self.wave = wave
        self.is_fetch = is_fetch
        self.produced_flags = set()
        self.consumed_flags = set()
        self.pid_m = self.pid_n = None


def build_graph(M, N, K, BM, BN, BK, GM, NFS, FS, FSF, KPF, interleave, num_cus):
    """
    Returns
    -------
    G         : nx.DiGraph  (fetcher → GEMM edges)
    wg_info   : dict[pid, WGInfo]
    stats     : dict
    """
    num_m_tiles          = M // BM
    num_tiles_n          = ceil_div(N, BN)
    num_k_blocks         = K // BK
    num_k_blocks_local   = (K // 2) // BK   # K_local per rank (for 2 ranks as default test)
    assert num_k_blocks % KPF == 0, f"KPF={KPF} must divide num_k_blocks={num_k_blocks}"
    num_flag_groups_k    = num_k_blocks // KPF

    m_per_stage          = ceil_div(num_m_tiles, NFS)
    gemm_tiles_per_stage = m_per_stage * num_tiles_n
    num_pid_in_group     = GM * num_tiles_n
    num_groups_per_stage = ceil_div(m_per_stage, GM)
    tiles_per_m_group    = num_flag_groups_k * GM

    if interleave:
        FIRST_STAGE_SIZE = num_groups_per_stage * (FSF + num_pid_in_group)
        REST_STAGE_SIZE  = num_groups_per_stage * (FS  + num_pid_in_group)
    else:
        FIRST_STAGE_SIZE = FSF + gemm_tiles_per_stage
        REST_STAGE_SIZE  = FS  + gemm_tiles_per_stage

    grid_size = FIRST_STAGE_SIZE + REST_STAGE_SIZE * max(0, NFS - 1)

    wg_info: dict[int, WGInfo] = {}

    # ---- pass 1: decode every pid → WGInfo ----
    for pid in range(grid_size):
        if pid < FIRST_STAGE_SIZE:
            stage, local_pid, fetch_thresh = 0, pid, FSF
        else:
            adj       = pid - FIRST_STAGE_SIZE
            stage     = 1 + adj // REST_STAGE_SIZE
            local_pid = adj %  REST_STAGE_SIZE
            fetch_thresh = FS

        wave = pid // num_cus

        if interleave:
            chunk_size     = fetch_thresh + num_pid_in_group
            group_in_stage = local_pid // chunk_size
            pos            = local_pid %  chunk_size
            is_fetch       = pos < fetch_thresh
            stage_pid      = pos          # index within the fetch pool
            gemm_within    = pos - fetch_thresh
        else:
            is_fetch       = local_pid < fetch_thresh
            group_in_stage = 0
            stage_pid      = local_pid
            gemm_local_id  = local_pid - fetch_thresh
            group_in_stage_gemm = gemm_local_id // num_pid_in_group if not is_fetch else 0
            gemm_within    = gemm_local_id % num_pid_in_group if not is_fetch else 0

        info = WGInfo(pid, stage, local_pid, wave, is_fetch)

        if is_fetch:
            # Enumerate which (m_tile, k_flag_group) pairs this fetcher produces
            stage_fetch_sms = FSF if stage == 0 else FS
            stage_m_start   = stage * m_per_stage
            stage_m_count   = min(m_per_stage, num_m_tiles - stage_m_start)
            total_fg_stage  = num_flag_groups_k * stage_m_count

            if interleave:
                fg_start = group_in_stage * tiles_per_m_group + stage_pid
                fg_end   = min((group_in_stage + 1) * tiles_per_m_group, total_fg_stage)
                step     = fetch_thresh
            else:
                fg_start = stage_pid
                fg_end   = total_fg_stage
                step     = stage_fetch_sms

            for fg_idx in range(fg_start, fg_end, step):
                m_group_l   = fg_idx // tiles_per_m_group
                within      = fg_idx %  tiles_per_m_group
                k_flag_grp  = within // GM
                m_in_group  = within %  GM
                m_tile      = stage_m_start + m_group_l * GM + m_in_group
                m_tile      = min(m_tile, num_m_tiles - 1)
                info.produced_flags.add((m_tile, k_flag_grp))

        else:
            # GEMM WG: decode pid_m / pid_n from gemm_within and group_in_stage
            if interleave:
                g_id = group_in_stage
                gw   = gemm_within
            else:
                gw   = gemm_local_id % num_pid_in_group
                g_id = gemm_local_id // num_pid_in_group

            stage_m_start = stage * m_per_stage
            first_pid_m   = stage_m_start + g_id * GM
            first_pid_m   = min(first_pid_m, num_m_tiles - 1)
            group_sz      = min(num_m_tiles - first_pid_m, GM)
            pid_m         = first_pid_m + (gw % group_sz)
            pid_n         = gw // group_sz
            pid_m         = min(pid_m, num_m_tiles - 1)

            info.pid_m = pid_m
            info.pid_n = pid_n
            info.consumed_flags = {(pid_m, k) for k in range(num_flag_groups_k)}

        wg_info[pid] = info

    # ---- pass 2: build flag index → producer/consumer lists ----
    flag_to_fetchers: dict = defaultdict(list)
    flag_to_gemm:     dict = defaultdict(list)

    for pid, info in wg_info.items():
        if info.is_fetch:
            for f in info.produced_flags:
                flag_to_fetchers[f].append(pid)
        else:
            for f in info.consumed_flags:
                flag_to_gemm[f].append(pid)

    # ---- pass 3: build DiGraph ----
    G = nx.DiGraph()
    for pid, info in wg_info.items():
        G.add_node(pid, **{
            "is_fetch": info.is_fetch,
            "stage": info.stage,
            "wave": info.wave,
            "pid_m": info.pid_m,
            "pid_n": info.pid_n,
        })

    edge_count = 0
    for flag_key, fetcher_pids in flag_to_fetchers.items():
        for fp in fetcher_pids:
            for gp in flag_to_gemm.get(flag_key, []):
                if not G.has_edge(fp, gp):
                    G.add_edge(fp, gp, flags=[flag_key])
                    edge_count += 1
                else:
                    G[fp][gp]["flags"].append(flag_key)

    stats = dict(
        grid_size=grid_size,
        total_fetch=sum(1 for i in wg_info.values() if i.is_fetch),
        total_gemm=sum(1 for i in wg_info.values() if not i.is_fetch),
        edge_count=edge_count,
        num_flag_groups_k=num_flag_groups_k,
        num_m_tiles=num_m_tiles,
        num_tiles_n=num_tiles_n,
        FIRST_STAGE_SIZE=FIRST_STAGE_SIZE,
        REST_STAGE_SIZE=REST_STAGE_SIZE,
        num_groups_per_stage=num_groups_per_stage,
    )
    return G, wg_info, stats


# ---------------------------------------------------------------------------
# Layout: hierarchical grid (stage × wave × role)
# ---------------------------------------------------------------------------

def make_positions(wg_info, stats, num_cus, NFS):
    """
    True bipartite layout:
      X = 0.0  for fetchers  |  X = 1.0  for GEMM
      Y = evenly spaced within each stage band, sorted by pid within role
      Wave marker stored separately for annotation (does not affect node position).

    Stage bands are separated vertically; within a band fetchers and GEMM nodes
    share the same Y range so edges are short and horizontal.
    """
    pos = {}

    # Group by (stage, is_fetch), sorted by pid within each bucket
    buckets = defaultdict(list)
    for pid, info in wg_info.items():
        buckets[(info.stage, info.is_fetch)].append(pid)
    for key in buckets:
        buckets[key].sort()

    # How tall is each stage band?  Use the larger of the two role counts.
    stage_heights = {}
    for stage in range(NFS):
        n_fetch = len(buckets.get((stage, True),  []))
        n_gemm  = len(buckets.get((stage, False), []))
        stage_heights[stage] = max(n_fetch, n_gemm, 1)

    # Cumulative Y offset per stage (stage 0 at top → largest Y values)
    stage_y_top = {}
    y_cursor = 0.0
    STAGE_GAP = 1.5   # extra gap between stages
    for stage in range(NFS):
        stage_y_top[stage] = y_cursor
        y_cursor += stage_heights[stage] + STAGE_GAP

    for (stage, is_fetch), pids in buckets.items():
        x = 0.0 if is_fetch else 1.0
        n = len(pids)
        h = stage_heights[stage]
        # Centre nodes vertically within the stage band
        y_start = stage_y_top[stage] + (h - n) / 2.0
        for i, pid in enumerate(pids):
            pos[pid] = (x, -(y_start + i))   # negate so stage 0 is at top

    return pos, stage_y_top, stage_heights


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def plot_graph(G, wg_info, stats, pos, stage_y_top, stage_heights,
               M, N, K, BM, BN, BK, GM, NFS, FS, FSF, KPF,
               interleave, num_cus, edge_alpha, show_labels, out_path):
    n_nodes = G.number_of_nodes()
    fig_w = max(10, min(20, 4 + n_nodes * 0.3))
    fig_h = max(6, NFS * 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Node colors and sizes
    node_colors = ["#1f77b4" if wg_info[p].is_fetch else "#ff7f0e" for p in G.nodes()]
    node_sizes  = [120 if wg_info[p].is_fetch else 80 for p in G.nodes()]

    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#888888", alpha=edge_alpha,
        arrows=True, arrowsize=8,
        connectionstyle="arc3,rad=0.1",
        width=0.6,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.5, edgecolors="#333333",
    )

    # Labels
    if show_labels:
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels={p: str(p) for p in G.nodes()},
            font_size=6, font_color="#111111",
        )

    # Stage horizontal bands — drawn in data coordinates after nodes are placed
    STAGE_GAP = 1.5
    for s in range(NFS):
        y_top_data = -(stage_y_top[s])
        y_bot_data = -(stage_y_top[s] + stage_heights[s])
        ax.axhspan(y_bot_data - 0.5, y_top_data + 0.5,
                   alpha=0.05, color="#0000ff" if s % 2 == 0 else "#ff8800",
                   zorder=0)
        ax.text(-0.15, (y_top_data + y_bot_data) / 2,
                f"stage {s}", fontsize=8, color="#555555",
                va="center", ha="right")

    # Column labels at top
    y_top_all = -(stage_y_top[0])
    ax.text(0.0, y_top_all + 0.8, "Fetchers", fontsize=9,
            color="#1f77b4", ha="center", fontweight="bold")
    ax.text(1.0, y_top_all + 0.8, "GEMM WGs", fontsize=9,
            color="#ff7f0e", ha="center", fontweight="bold")
    # Dividing vertical line between roles
    ax.axvline(0.5, color="#cccccc", linewidth=1.0, linestyle="--", zorder=0)

    layout_tag = "interleaved" if interleave else "contiguous"
    ax.set_title(
        f"WG dependency graph  —  {layout_tag}  |  "
        f"M={M} N={N} K={K}  BM={BM} BN={BN} BK={BK}  GM={GM}  "
        f"NFS={NFS} FS={FS} FSF={FSF} KPF={KPF}  num_cus={num_cus}",
        fontsize=8, pad=8,
    )

    legend_handles = [
        mpatches.Patch(facecolor="#1f77b4", label=f"Fetcher ({stats['total_fetch']} WGs)"),
        mpatches.Patch(facecolor="#ff7f0e", label=f"GEMM   ({stats['total_gemm']} WGs)"),
        mpatches.Patch(facecolor="#888888", alpha=0.5, label=f"flag-dep edges ({stats['edge_count']})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    ratio = stats["total_gemm"] / max(stats["total_fetch"], 1)
    edges_per_gemm = stats["edge_count"] / max(stats["total_gemm"], 1)
    info_text = (
        f"grid={stats['grid_size']}  fetch={stats['total_fetch']}  GEMM={stats['total_gemm']}\n"
        f"GEMM/fetch={ratio:.1f}×  edges={stats['edge_count']}  edges/GEMM={edges_per_gemm:.1f}\n"
        f"m_tiles={stats['num_m_tiles']}  n_tiles={stats['num_tiles_n']}  "
        f"flag_groups_k={stats['num_flag_groups_k']}"
    )
    ax.text(0.01, 0.01, info_text, transform=ax.transAxes, fontsize=7,
            verticalalignment="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[graph] wrote {out_path}")


def plot_simple(G, wg_info, stats, out_path, node_limit=200):
    """
    Bare dependency graph.

    Small graphs (total nodes ≤ node_limit): bipartite node-link diagram,
    fetchers left column, GEMM right column, PID labels on nodes.

    Large graphs (total nodes > node_limit): adjacency matrix heatmap —
    rows = fetcher PIDs, cols = GEMM PIDs, filled cell = dependency edge.
    Much more readable when there are thousands of WGs.
    """
    fetcher_nodes = sorted(p for p, i in wg_info.items() if i.is_fetch)
    gemm_nodes    = sorted(p for p, i in wg_info.items() if not i.is_fetch)
    n_nodes       = G.number_of_nodes()

    if n_nodes > node_limit:
        # ── adjacency matrix ──────────────────────────────────────────────
        import numpy as np

        fetch_idx = {p: i for i, p in enumerate(fetcher_nodes)}
        gemm_idx  = {p: i for i, p in enumerate(gemm_nodes)}
        nf, ng    = len(fetcher_nodes), len(gemm_nodes)
        mat       = np.zeros((nf, ng), dtype=np.uint8)

        for fp, gp in G.edges():
            if fp in fetch_idx and gp in gemm_idx:
                mat[fetch_idx[fp], gemm_idx[gp]] = 1

        # How many distinct flag-group "stripes" does each fetcher touch?
        # Colour by edge count per fetcher row to show load balance.
        row_counts = mat.sum(axis=1)

        fig_h = max(4, min(14, nf * 0.12 + 2))
        fig_w = max(10, min(20, ng * 0.01 + 4))
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                 gridspec_kw={"width_ratios": [20, 1]})
        ax, cax = axes

        im = ax.imshow(mat, aspect="auto", cmap="Blues",
                       interpolation="nearest", origin="upper")

        ax.set_xlabel(f"GEMM WG  (pid, {ng} total)", fontsize=9)
        ax.set_ylabel(f"Fetcher  (pid, {nf} total)", fontsize=9)

        # Y-tick every ~10 fetchers, X-tick every ~64 GEMM WGs
        y_step = max(1, nf  // 10)
        x_step = max(1, ng // 16)
        ax.set_yticks(range(0, nf, y_step))
        ax.set_yticklabels([str(fetcher_nodes[i]) for i in range(0, nf, y_step)], fontsize=6)
        ax.set_xticks(range(0, ng, x_step))
        ax.set_xticklabels([str(gemm_nodes[i]) for i in range(0, ng, x_step)],
                           fontsize=6, rotation=45, ha="right")

        ax.set_title(
            f"WG dependency adjacency matrix  —  "
            f"fetchers={nf}  GEMM={ng}  edges={stats['edge_count']}  "
            f"edges/GEMM={stats['edge_count']/max(ng,1):.1f}",
            fontsize=9, pad=6,
        )

        fig.colorbar(im, cax=cax, label="edge (1=dep)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[simple/matrix] wrote {out_path}")
        return

    # ── node-link bipartite diagram (small graphs) ────────────────────────
    pos = {}
    for rank, pids in [(0, fetcher_nodes), (1, gemm_nodes)]:
        n = len(pids)
        for i, pid in enumerate(pids):
            pos[pid] = (float(rank), (n - 1 - i) / max(n - 1, 1))

    fig_h = max(4, min(40, n_nodes * 0.35))
    fig, ax = plt.subplots(figsize=(8, fig_h))

    node_colors = ["#1f77b4" if wg_info[p].is_fetch else "#ff7f0e" for p in G.nodes()]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#888888", alpha=0.5,
        arrows=True, arrowsize=12,
        connectionstyle="arc3,rad=0.08",
        width=1.0,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=600,
        linewidths=1.0, edgecolors="#333333",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={p: str(p) for p in G.nodes()},
        font_size=8, font_color="white", font_weight="bold",
    )

    ax.text(0.0, 1.05, "Fetchers", transform=ax.transAxes,
            fontsize=10, color="#1f77b4", ha="left", fontweight="bold")
    ax.text(1.0, 1.05, "GEMM WGs", transform=ax.transAxes,
            fontsize=10, color="#ff7f0e", ha="right", fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor="#1f77b4", label=f"Fetcher ({stats['total_fetch']})"),
        mpatches.Patch(facecolor="#ff7f0e", label=f"GEMM ({stats['total_gemm']})"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              fontsize=8, framealpha=0.9, ncol=2)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[simple/graph] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="WG dependency graph for all_gather_matmul_hbm_buffer (no GPU needed)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-m",  type=int, default=256)
    p.add_argument("-n",  type=int, default=256)
    p.add_argument("--k_local",   type=int, default=128, help="K per rank (K = k_local * num_ranks)")
    p.add_argument("--num_ranks", type=int, default=2)
    p.add_argument("--bm",  type=int, default=128)
    p.add_argument("--bn",  type=int, default=128)
    p.add_argument("--bk",  type=int, default=64)
    p.add_argument("--gm",  type=int, default=1)
    p.add_argument("--nfs", type=int, default=1,  help="num_fetch_stages")
    p.add_argument("--fs",  type=int, default=2,  help="num_fetch_sms")
    p.add_argument("--fsf", type=int, default=4,  help="first_stage_fetch_sms")
    p.add_argument("--kpf", type=int, default=2,  help="k_per_flag")
    p.add_argument("--num_cus",    type=int, default=16,  help="CUs per wave (for wave partitioning)")
    p.add_argument("--edge_alpha", type=float, default=0.4)
    p.add_argument("--no_labels",  action="store_true", help="Suppress PID labels")
    p.add_argument("--interleave_roles", action="store_true")
    p.add_argument("--simple", action="store_true",
                   help="Bare dependency graph only — no stage/wave decoration")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    M, N   = args.m, args.n
    K      = args.k_local * args.num_ranks
    BM, BN, BK = args.bm, args.bn, args.bk
    GM, NFS, FS, FSF, KPF = args.gm, args.nfs, args.fs, args.fsf, args.kpf
    interleave = args.interleave_roles

    assert M % BM == 0, f"M={M} not divisible by BM={BM}"
    assert K % BK == 0, f"K={K} not divisible by BK={BK}"
    assert (K // BK) % KPF == 0, f"KPF={KPF} does not divide num_k_blocks={K//BK}"

    G, wg_info, stats = build_graph(
        M, N, K, BM, BN, BK, GM, NFS, FS, FSF, KPF, interleave, args.num_cus
    )

    print(f"\n{'='*60}")
    print(f"Layout         : {'interleaved' if interleave else 'contiguous'}")
    print(f"Shape          : M={M} N={N} K={K}")
    print(f"Config         : BM={BM} BN={BN} BK={BK} GM={GM} NFS={NFS} FS={FS} FSF={FSF} KPF={KPF}")
    print(f"{'─'*60}")
    print(f"grid_size      : {stats['grid_size']}")
    print(f"total fetch WGs: {stats['total_fetch']}")
    print(f"total GEMM WGs : {stats['total_gemm']}")
    print(f"flag_groups_k  : {stats['num_flag_groups_k']}")
    print(f"edges          : {stats['edge_count']}")
    print(f"edges/GEMM WG  : {stats['edge_count'] / max(stats['total_gemm'], 1):.1f}")
    print(f"{'='*60}\n")

    layout_tag = "interleave" if interleave else "contiguous"

    if args.simple:
        out_path = args.out or (
            SCRIPT_DIR / f"wg_graph_simple_{layout_tag}_M{M}N{N}K{K}_gm{GM}_nfs{NFS}.png"
        )
        plot_simple(G, wg_info, stats, out_path)
        return

    pos, stage_y_top, stage_heights = make_positions(wg_info, stats, args.num_cus, NFS)

    out_path = args.out or (
        SCRIPT_DIR / f"wg_graph_{layout_tag}_M{M}N{N}K{K}_gm{GM}_nfs{NFS}.png"
    )

    plot_graph(
        G, wg_info, stats, pos, stage_y_top, stage_heights,
        M, N, K, BM, BN, BK, GM, NFS, FS, FSF, KPF,
        interleave, args.num_cus, args.edge_alpha,
        show_labels=not args.no_labels,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
