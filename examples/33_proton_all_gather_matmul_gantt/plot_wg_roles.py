#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
WG-role grid visualizer for all_gather_matmul_hbm_buffer.

Reconstructs the kernel's pid→role mapping purely from host-side arithmetic
(no GPU, no kernel launch, no iris import). Renders a 2-D raster image:

    X-axis : local_pid within stage
    Y-axis : stage index (0 = top)
    color  : blue = fetcher, orange = GEMM compute, grey = unused padding

Run
---
    # contiguous layout
    python plot_wg_roles.py -m 8192 -n 8192 --k_local 1024 --num_ranks 8 --gm 16

    # interleaved layout
    python plot_wg_roles.py -m 8192 -n 8192 --k_local 1024 --num_ranks 8 --gm 16 --interleave_roles
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

ROLE_UNUSED = -1
ROLE_GEMM   =  0
ROLE_FETCH  =  1


def ceil_div(a, b):
    return -(-a // b)


def compute_grid(M, N, K, BM, BN, BK, GM, NFS, FS, FSF, KPF, interleave):
    """
    Pure-Python re-implementation of the kernel's pid-decode and host grid sizing.

    Returns
    -------
    grid_size          : int
    FIRST_STAGE_SIZE   : int
    REST_STAGE_SIZE    : int
    stage_sizes        : list[int]  — size of each stage (first may differ)
    roles              : np.ndarray shape (NFS, max_stage_size) of ROLE_*
    stats              : dict with summary counts
    """
    num_m_tiles          = M // BM
    num_tiles_n          = ceil_div(N, BN)
    m_per_stage          = ceil_div(num_m_tiles, NFS)
    gemm_tiles_per_stage = m_per_stage * num_tiles_n
    num_pid_in_group     = GM * num_tiles_n
    num_groups_per_stage = ceil_div(m_per_stage, GM)

    if interleave:
        FIRST_STAGE_SIZE = num_groups_per_stage * (FSF + num_pid_in_group)
        REST_STAGE_SIZE  = num_groups_per_stage * (FS  + num_pid_in_group)
    else:
        FIRST_STAGE_SIZE = FSF + gemm_tiles_per_stage
        REST_STAGE_SIZE  = FS  + gemm_tiles_per_stage

    grid_size = FIRST_STAGE_SIZE + REST_STAGE_SIZE * max(0, NFS - 1)

    stage_sizes = [FIRST_STAGE_SIZE] + [REST_STAGE_SIZE] * max(0, NFS - 1)
    max_stage_size = max(stage_sizes)

    roles = np.full((NFS, max_stage_size), ROLE_UNUSED, dtype=np.int8)

    total_fetch = 0
    total_gemm  = 0

    for pid in range(grid_size):
        if pid < FIRST_STAGE_SIZE:
            stage         = 0
            local_pid     = pid
            fetch_thresh  = FSF
        else:
            adj           = pid - FIRST_STAGE_SIZE
            stage         = 1 + adj // REST_STAGE_SIZE
            local_pid     = adj %  REST_STAGE_SIZE
            fetch_thresh  = FS

        if interleave:
            chunk_size = fetch_thresh + num_pid_in_group
            pos        = local_pid % chunk_size
            is_fetch   = pos < fetch_thresh
        else:
            is_fetch = local_pid < fetch_thresh

        roles[stage, local_pid] = ROLE_FETCH if is_fetch else ROLE_GEMM
        if is_fetch:
            total_fetch += 1
        else:
            total_gemm += 1

    stats = dict(
        grid_size=grid_size,
        total_fetch=total_fetch,
        total_gemm=total_gemm,
        num_m_tiles=num_m_tiles,
        num_tiles_n=num_tiles_n,
        m_per_stage=m_per_stage,
        gemm_tiles_per_stage=gemm_tiles_per_stage,
        num_pid_in_group=num_pid_in_group,
        num_groups_per_stage=num_groups_per_stage,
        FIRST_STAGE_SIZE=FIRST_STAGE_SIZE,
        REST_STAGE_SIZE=REST_STAGE_SIZE,
    )
    return grid_size, FIRST_STAGE_SIZE, REST_STAGE_SIZE, stage_sizes, roles, stats


def plot_roles(roles, stats, M, N, K, BM, BN, BK, GM, NFS, FS, FSF, KPF, interleave, out_path):
    NFS_actual, max_stage_size = roles.shape

    # Map ROLE values to floats for a 3-color colormap
    # ROLE_UNUSED=-1 → 0.0 (grey), ROLE_GEMM=0 → 0.5 (orange), ROLE_FETCH=1 → 1.0 (blue)
    display = np.where(roles == ROLE_UNUSED, 0.0,
              np.where(roles == ROLE_GEMM,   0.5, 1.0))

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#cccccc", "#ff7f0e", "#1f77b4"])  # grey, orange, blue
    norm = BoundaryNorm([-0.1, 0.25, 0.75, 1.1], cmap.N)

    # Adaptive figure height: at least 2 in, scale with NFS
    fig_h = max(2.0, 0.8 * NFS_actual + 1.5)
    fig_w = min(20, max(10, max_stage_size / 800))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(display, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest", origin="upper")

    ax.set_xlabel("local pid within stage", fontsize=9)
    ax.set_ylabel("stage index", fontsize=9)
    ax.set_yticks(range(NFS_actual))
    ax.set_yticklabels([f"stage {s}" for s in range(NFS_actual)], fontsize=8)

    layout_tag = "interleaved" if interleave else "contiguous"
    ax.set_title(
        f"WG role grid  —  {layout_tag}  |  "
        f"M={M} N={N} K={K}  BM={BM} BN={BN} BK={BK}  GM={GM}  "
        f"NFS={NFS} FS={FS} FSF={FSF} KPF={KPF}",
        fontsize=8, pad=6,
    )

    legend_handles = [
        mpatches.Patch(facecolor="#1f77b4", label=f"fetch  ({stats['total_fetch']:,} WGs)"),
        mpatches.Patch(facecolor="#ff7f0e", label=f"GEMM   ({stats['total_gemm']:,} WGs)"),
        mpatches.Patch(facecolor="#cccccc", label="unused (padding)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    # Summary text box
    ratio = stats["total_gemm"] / max(stats["total_fetch"], 1)
    info = (
        f"grid_size={stats['grid_size']:,}\n"
        f"fetch={stats['total_fetch']:,}  GEMM={stats['total_gemm']:,}\n"
        f"GEMM/fetch ratio={ratio:.1f}x\n"
        f"m_tiles={stats['num_m_tiles']}  n_tiles={stats['num_tiles_n']}\n"
        f"groups/stage={stats['num_groups_per_stage']}  pid_in_group={stats['num_pid_in_group']}"
    )
    ax.text(0.01, 0.02, info, transform=ax.transAxes, fontsize=7,
            verticalalignment="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[roles] wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize WG roles for all_gather_matmul_hbm_buffer (no GPU needed)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-m",  type=int, default=8192,  help="M (rows of A, full)")
    p.add_argument("-n",  type=int, default=8192,  help="N (cols of B)")
    p.add_argument("--k_local", type=int, default=1024, help="K_local per rank (K = k_local * num_ranks)")
    p.add_argument("--num_ranks", type=int, default=8)
    p.add_argument("--bm",  type=int, default=128)
    p.add_argument("--bn",  type=int, default=256)
    p.add_argument("--bk",  type=int, default=64)
    p.add_argument("--gm",  type=int, default=1,   help="group_size_m")
    p.add_argument("--nfs", type=int, default=1,   help="num_fetch_stages")
    p.add_argument("--fs",  type=int, default=16,  help="num_fetch_sms (steady-state)")
    p.add_argument("--fsf", type=int, default=64,  help="first_stage_fetch_sms")
    p.add_argument("--kpf", type=int, default=8,   help="k_per_flag (informational, not used in layout)")
    p.add_argument("--interleave_roles", action="store_true",
                   help="Use per-M-group interleaved [fetch|gemm] layout")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: next to this script)")
    return p.parse_args()


def main():
    args = parse_args()
    M   = args.m
    N   = args.n
    K   = args.k_local * args.num_ranks
    BM, BN, BK = args.bm, args.bn, args.bk
    GM  = args.gm
    NFS = args.nfs
    FS  = args.fs
    FSF = args.fsf
    KPF = args.kpf
    interleave = args.interleave_roles

    assert M % BM == 0, f"M={M} must be divisible by BM={BM}"
    assert K % BK == 0, f"K={K} must be divisible by BK={BK}"

    grid_size, FSZ, RSZ, stage_sizes, roles, stats = compute_grid(
        M, N, K, BM, BN, BK, GM, NFS, FS, FSF, KPF, interleave
    )

    layout_tag = "interleave" if interleave else "contiguous"
    print(f"\n{'='*60}")
    print(f"Layout         : {layout_tag}")
    print(f"Shape          : M={M} N={N} K={K}")
    print(f"Config         : BM={BM} BN={BN} BK={BK} GM={GM} NFS={NFS} FS={FS} FSF={FSF} KPF={KPF}")
    print(f"{'─'*60}")
    print(f"grid_size      : {grid_size:,}")
    print(f"FIRST_STAGE    : {FSZ:,}  (FSF={FSF} + {'chunks×' if interleave else ''}gemm={FSZ-FSF*(1 if not interleave else stats['num_groups_per_stage'])})")
    print(f"REST_STAGE     : {RSZ:,}")
    print(f"stage_sizes    : {stage_sizes}")
    print(f"{'─'*60}")
    print(f"total GEMM WGs : {stats['total_gemm']:,}  ({stats['num_m_tiles']} m-tiles × {stats['num_tiles_n']} n-tiles)")
    print(f"total fetch WGs: {stats['total_fetch']:,}")
    print(f"GEMM/fetch     : {stats['total_gemm']/max(stats['total_fetch'],1):.1f}×")
    print(f"groups/stage   : {stats['num_groups_per_stage']}  (m_per_stage={stats['m_per_stage']} / GM={GM})")
    print(f"{'='*60}\n")

    if args.out:
        out_path = args.out
    else:
        out_path = SCRIPT_DIR / f"wg_roles_{layout_tag}_M{M}N{N}K{K}_gm{GM}_nfs{NFS}.png"

    plot_roles(roles, stats, M, N, K, BM, BN, BK, GM, NFS, FS, FSF, KPF, interleave, out_path)


if __name__ == "__main__":
    main()
