#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Plot the NCCL_MAX_NCHANNELS x world-size RCCL all-gather sweep.

Reads nchannels_sweep.csv (from sweep_nchannels.py) and emits:

  bw_vs_nchannels.png  x=NCCL_MAX_NCHANNELS (log2), y=busbw, one curve/WS,
                       at the largest message size. Saturation knee annotated.
  bw_vs_cus.png        same, x = CUs = nchannels * CUS_PER_CHANNEL. Overlays the
                       predicted CU_knee = WS * base (base fit from smallest WS),
                       directly testing "#CU to saturate scales with WS".
  bw_vs_cus_fit.png    grid-free version: fits BW = BW_max*(1-exp(-CU/tau)) per WS
                       and reports the 95%-saturation CU = tau*ln(20) ~= 3*tau,
                       removing the dependence on which CU values were sampled.
  bw_vs_size.png       faceted by WS, x=size MB (log), y=busbw, one curve/nchannels.

CUS_PER_CHANNEL: CUs (workgroups) RCCL uses per channel on this GPU. Default 1;
override RCCL_CUS_PER_CHANNEL once the true mapping is known (NCCL_DEBUG=INFO).
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_IN = SCRIPT_DIR / "nchannels_sweep.csv"
CUS_PER_CHANNEL = int(os.environ.get("RCCL_CUS_PER_CHANNEL", "1"))
KNEE_FRAC = 0.95  # fraction of a curve's max BW that counts as "saturated"
# CU at which the fitted curve reaches KNEE_FRAC of BW_max:
#   BW_max*(1-exp(-CU/tau)) = KNEE_FRAC*BW_max  ->  CU = -tau*ln(1-KNEE_FRAC)
KNEE_TAU_MULT = -np.log(1.0 - KNEE_FRAC)  # = ln(20) ~= 3.0 for 0.95


def load():
    """rows -> dict keyed by world_size -> dict size_mb -> list[(nchannels, bw)]."""
    by_ws = defaultdict(lambda: defaultdict(list))
    with open(CSV_IN) as f:
        for row in csv.DictReader(f):
            ws = int(row["world_size"])
            size = int(row["size_mb"])
            n = int(row["nchannels"])
            bw = float(row["bandwidth_gbps"])
            by_ws[ws][size].append((n, bw))
    for ws in by_ws:
        for size in by_ws[ws]:
            by_ws[ws][size].sort()
    return by_ws


def knee(pairs):
    """First x (nchannels) reaching >=KNEE_FRAC of max BW. pairs sorted by x."""
    if not pairs:
        return None, None
    bw_max = max(b for _, b in pairs)
    thr = KNEE_FRAC * bw_max
    for x, b in pairs:
        if b >= thr:
            return x, bw_max
    return pairs[-1][0], bw_max


def fit_saturating(pairs):
    """Fit BW = BW_max*(1 - exp(-CU/tau)) to (cu, bw) pairs (dependency-free).

    For a fixed tau the model is linear in BW_max, so BW_max has a closed-form
    least-squares solution; we grid-search tau (log-spaced) and keep the best.
    Returns (bw_max, tau, cu95) where cu95 = tau*ln(20) is the fitted CU at which
    BW reaches 95% of BW_max. Returns None if there is too little data to fit."""
    if len(pairs) < 3:
        return None
    cu = np.array([c for c, _ in pairs], dtype=float)
    bw = np.array([b for _, b in pairs], dtype=float)
    # tau search range tied to the sampled CU span.
    taus = np.geomspace(max(cu.min(), 0.5), cu.max() * 2.0, 400)
    best = None
    for tau in taus:
        basis = 1.0 - np.exp(-cu / tau)
        denom = float(basis @ basis)
        if denom <= 0:
            continue
        bw_max = float((basis @ bw) / denom)  # closed-form LS for the scale
        resid = bw - bw_max * basis
        sse = float(resid @ resid)
        if best is None or sse < best[0]:
            best = (sse, bw_max, tau)
    if best is None:
        return None
    _, bw_max, tau = best
    return bw_max, tau, tau * KNEE_TAU_MULT


def plot_bw_vs_nchannels(by_ws, max_size):
    fig, ax = plt.subplots(figsize=(8, 6))
    for ws in sorted(by_ws):
        pairs = by_ws[ws].get(max_size, [])
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, marker="o", label=f"WS={ws}")
        kx, _ = knee(pairs)
        if kx is not None:
            ax.axvline(kx, color=ax.lines[-1].get_color(), ls=":", alpha=0.4)
            print(f"WS={ws}: knee at nchannels={kx}  (base = knee/WS = {kx/ws:.2f})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NCCL_MAX_NCHANNELS")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_title(f"RCCL all-gather BW vs MAX_NCHANNELS ({max_size} MB)\n"
                 "dotted line = saturation knee (>=95% of peak)")
    ax.legend(title="World size")
    ax.grid(True, alpha=0.3)
    out = SCRIPT_DIR / "bw_vs_nchannels.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_bw_vs_cus(by_ws, max_size):
    fig, ax = plt.subplots(figsize=(8, 6))
    base_cu = None  # CU knee of the smallest WS -> base CUs per link
    knees = {}
    for ws in sorted(by_ws):
        pairs = by_ws[ws].get(max_size, [])
        if not pairs:
            continue
        xs = [n * CUS_PER_CHANNEL for n, _ in pairs]
        ys = [b for _, b in pairs]
        ax.plot(xs, ys, marker="o", label=f"WS={ws}")
        kn_nch, _ = knee(pairs)
        kn_cu = kn_nch * CUS_PER_CHANNEL if kn_nch is not None else None
        knees[ws] = kn_cu
        if kn_cu is not None:
            ax.scatter([kn_cu], [max(ys)], color=ax.lines[-1].get_color(),
                       s=120, marker="*", zorder=5)
            if base_cu is None:
                base_cu = kn_cu / ws  # fit base from smallest WS
            print(f"WS={ws}: CU knee={kn_cu}  predicted WS*base={ws*base_cu:.1f}")

    # Predicted hypothesis line: CU_knee = WS * base, plotted as vertical markers.
    if base_cu is not None:
        for ws in sorted(knees):
            pred = ws * base_cu
            ax.axvline(pred, ls="--", alpha=0.3, color="gray")
        ax.plot([], [], ls="--", color="gray",
                label=f"predicted WS x {base_cu:.1f} CU/link")

    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"CUs used  (= NCCL_MAX_NCHANNELS x {CUS_PER_CHANNEL})")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_title(f"RCCL all-gather BW vs CUs ({max_size} MB)\n"
                 "star = saturation knee; dashed = WS x base prediction")
    ax.legend(title="World size")
    ax.grid(True, alpha=0.3)
    out = SCRIPT_DIR / "bw_vs_cus.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_bw_vs_cus_fit(by_ws, max_size):
    """Grid-free knee: fit a saturating curve per WS, report cu95 = tau*ln(20)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    base_cu = None  # cu95/WS of the smallest WS -> base CUs per link
    for ws in sorted(by_ws):
        pairs = by_ws[ws].get(max_size, [])
        if not pairs:
            continue
        cus = [(n * CUS_PER_CHANNEL, b) for n, b in pairs]
        xs = [c for c, _ in cus]
        ys = [b for _, b in cus]
        line, = ax.plot(xs, ys, marker="o", ls="", label=f"WS={ws} (data)")
        fit = fit_saturating(cus)
        if fit is None:
            continue
        bw_max, tau, cu95 = fit
        xx = np.geomspace(min(xs), max(xs), 200)
        ax.plot(xx, bw_max * (1.0 - np.exp(-xx / tau)),
                color=line.get_color(), alpha=0.8)
        ax.scatter([cu95], [KNEE_FRAC * bw_max], color=line.get_color(),
                   s=140, marker="*", zorder=5)
        if base_cu is None:
            base_cu = cu95 / ws
        print(f"WS={ws}: fit BW_max={bw_max:.1f} GB/s  tau={tau:.1f} CU  "
              f"cu95={cu95:.1f}  (cu95/WS={cu95/ws:.2f}, pred WS*base={ws*base_cu:.1f})")

    if base_cu is not None:
        ax.plot([], [], color="gray", ls="--",
                label=f"base ~= {base_cu:.1f} CU/link (from smallest WS)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"CUs used  (= NCCL_MAX_NCHANNELS x {CUS_PER_CHANNEL})")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_title(f"RCCL all-gather: fitted saturation ({max_size} MB)\n"
                 r"line = fit BW$_{max}$(1-e$^{-CU/\tau}$); star = 95% knee = $\tau$ln20")
    ax.legend(title="World size", fontsize=8)
    ax.grid(True, alpha=0.3)
    out = SCRIPT_DIR / "bw_vs_cus_fit.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_bw_vs_size(by_ws):
    wss = sorted(by_ws)
    fig, axes = plt.subplots(1, len(wss), figsize=(6 * len(wss), 5), squeeze=False)
    for ax, ws in zip(axes[0], wss):
        # regroup: nchannels -> list[(size, bw)]
        by_nch = defaultdict(list)
        for size, pairs in by_ws[ws].items():
            for n, bw in pairs:
                by_nch[n].append((size, bw))
        for n in sorted(by_nch):
            data = sorted(by_nch[n])
            xs, ys = zip(*data)
            ax.plot(xs, ys, marker="o", label=f"nch={n}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("message size (MB)")
        ax.set_ylabel("Bus bandwidth (GB/s)")
        ax.set_title(f"WS={ws}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("RCCL all-gather BW vs message size, per NCCL_MAX_NCHANNELS",
                 fontweight="bold")
    out = SCRIPT_DIR / "bw_vs_size.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    by_ws = load()
    if not by_ws:
        raise SystemExit(f"no data in {CSV_IN}")
    all_sizes = {s for ws in by_ws for s in by_ws[ws]}
    max_size = max(all_sizes)
    print(f"plotting headline at largest size = {max_size} MB; "
          f"CUS_PER_CHANNEL={CUS_PER_CHANNEL}")
    plot_bw_vs_nchannels(by_ws, max_size)
    plot_bw_vs_cus(by_ws, max_size)
    plot_bw_vs_cus_fit(by_ws, max_size)
    plot_bw_vs_size(by_ws)


if __name__ == "__main__":
    main()
