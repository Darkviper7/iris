#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Sweep NCCL_MAX_NCHANNELS x world-size for RCCL all-gather bandwidth.

NCCL_MAX_NCHANNELS is read by RCCL at communicator creation, so it must be fixed
for the lifetime of a process -- it cannot be a bench axis. This driver therefore
runs bench_rccl_all_gather.py ONCE PER nchannels value as a fresh subprocess with
the env var exported, then aggregates every (nchannels, world_size, size_mb,
bandwidth) row into nchannels_sweep.csv.

Resumable: a nchannels value whose raw JSON already parses is skipped. Set
RCCL_SWEEP_FORCE=1 to re-run everything.

Run:
  cd iris && ../.venv/bin/python benchmark/ccl/sweep_nchannels.py
  # smoke:
  RCCL_NCHANNELS=1,4 RCCL_SWEEP_SIZES_MB=1,16 \\
    ../.venv/bin/python benchmark/ccl/sweep_nchannels.py
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]  # iris/
BENCH = SCRIPT_DIR / "bench_rccl_all_gather.py"

NCHANNELS = [int(x) for x in os.environ.get("RCCL_NCHANNELS", "1,2,4,8,16,32").split(",")]
N_WARMUP = os.environ.get("RCCL_SWEEP_WARMUP", "10")
N_REPEAT = os.environ.get("RCCL_SWEEP_REPEAT", "50")

CSV_OUT = SCRIPT_DIR / "nchannels_sweep.csv"


def _run_nchannels(n: int) -> Path:
    """Run the inner bench with NCCL_MAX_NCHANNELS=n; return its raw JSON path."""
    out_json = SCRIPT_DIR / f"_raw_nch{n}.json"
    if out_json.exists() and os.environ.get("RCCL_SWEEP_FORCE", "0") != "1":
        try:
            json.loads(out_json.read_text())
            print(f"=== nchannels={n}: reusing {out_json.name} (resume) ===", flush=True)
            return out_json
        except (json.JSONDecodeError, OSError):
            print(f"=== nchannels={n}: {out_json.name} unreadable, re-running ===", flush=True)

    env = dict(os.environ)
    env["NCCL_MAX_NCHANNELS"] = str(n)
    env.setdefault("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    cmd = [
        sys.executable, str(BENCH),
        "--benchmark_filter", "rccl_all_gather",
        "--benchmark_format", "json",
        "--benchmark_out", str(out_json),
        "--n_warmup", N_WARMUP,
        "--n_repeat", N_REPEAT,
    ]
    print(f"\n=== NCCL_MAX_NCHANNELS={n} -> {out_json.name} ===", flush=True)
    subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), check=True)
    return out_json


def _rows_from_json(n: int, raw_json: Path):
    """Yield (nchannels, world_size, size_mb, bandwidth_gbps) for non-skipped rows."""
    for rec in json.loads(raw_json.read_text()):
        if rec.get("skipped") or rec.get("bandwidth_gbps") is None:
            continue
        size_mb = rec.get("params", {}).get("size_mb")
        yield (n, rec["world_size"], size_mb, rec["bandwidth_gbps"])


def main():
    all_rows = []
    for n in NCHANNELS:
        raw = _run_nchannels(n)
        all_rows.extend(_rows_from_json(n, raw))

    all_rows.sort(key=lambda r: (r[1], r[2], r[0]))  # ws, size, nchannels
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nchannels", "world_size", "size_mb", "bandwidth_gbps"])
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {CSV_OUT}", flush=True)


if __name__ == "__main__":
    main()
