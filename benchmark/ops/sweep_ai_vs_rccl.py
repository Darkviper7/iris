#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Arithmetic-intensity vs RCCL-speedup sweep for the fused AG+MM layout kernel.

Tunes the layout kernel per square shape M=N=K in {1024,2048,4096,8192,16384}
(ws=4, fp16) and plots compute arithmetic intensity (x) vs speedup over RCCL (y),
persisting the per-shape winning ScheduleLayout to JSON.

Design (see plan): the per-shape tuning REUSES the existing gated tune bench
(benchmark/ops/bench_all_gather_matmul_layout_tune.py, IRIS_TUNE_SPACE=schedule,
IRIS_TUNE_CHECK=1) run as ONE SUBPROCESS PER SHAPE -- the iris symmetric heap
never frees, so 8192/16384 buffers would overflow a single 16 GiB heap; a fresh
process per shape guarantees the heap frees between shapes. Each subprocess emits
the runner's own JSON (--benchmark_format json), from which we pick the best
correctness-passing layout_tune row and the rccl_reference row. Config is then
reconstructed with make_layout (pure Python) so the saved JSON is reloadable.

Run:
  HIP_VISIBLE_DEVICES=0,1,2,3 HSA_NO_SCRATCH_RECLAIM=1 \\
    .venv/bin/python benchmark/ops/sweep_ai_vs_rccl.py
  # quick smoke on the two cheap shapes:
  IRIS_SWEEP_SHAPES=1024,2048 .venv/bin/python benchmark/ops/sweep_ai_vs_rccl.py

Outputs (next to this script):
  tuned_configs.json        per-shape winner + AI + speedup (reloadable layout)
  ai_vs_rccl_speedup.png    x=compute AI (log), y=speedup vs RCCL
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = SCRIPT_DIR / "bench_all_gather_matmul_layout_tune.py"

# Square shapes M=N=K. Override with IRIS_SWEEP_SHAPES=1024,2048 for a smoke run.
_SHAPES = [int(s) for s in os.environ.get("IRIS_SWEEP_SHAPES", "1024,2048,4096,8192,16384").split(",")]
_WORLD_SIZE = 4
_NUM_XCDS = 8
_DTYPE = "fp16"
_ELT = 2  # fp16 bytes
_N_WARMUP = int(os.environ.get("IRIS_SWEEP_WARMUP", "8"))
_N_REPEAT = int(os.environ.get("IRIS_SWEEP_REPEAT", "25"))

_JSON_OUT = SCRIPT_DIR / "tuned_configs.json"
_PNG_OUT = SCRIPT_DIR / "ai_vs_rccl_speedup.png"


def _compute_ai(mnk: int) -> float:
    """Compute arithmetic intensity = 2*M*N*K / total-HBM-bytes (fp16, ws=4).

    Buffers the op touches: A(M x K_local) + B(K x N) + C(M x N) + staged_a(M x K).
    K_local = K / world_size. For square M=N=K=S this reduces to ~S/3.25.
    """
    m = n = k = mnk
    k_local = k // _WORLD_SIZE
    bytes_total = _ELT * (m * k_local + k * n + m * n + m * k)
    flops = 2 * m * n * k
    return flops / bytes_total


def _run_shape(mnk: int) -> Path:
    """Tune one shape in a fresh subprocess; return the path to its runner JSON.

    Resumable: if this shape's raw JSON already exists (and parses), reuse it and
    skip the (expensive) re-tune. Set IRIS_SWEEP_FORCE=1 to always re-run. This
    lets the sweep continue after an interruption (e.g. SLURM allocation ending)
    without redoing completed shapes."""
    out_json = SCRIPT_DIR / f"_tune_raw_{mnk}.json"
    if out_json.exists() and os.environ.get("IRIS_SWEEP_FORCE", "0") != "1":
        try:
            json.loads(out_json.read_text())
            print(f"\n=== {mnk}^3: reusing existing {out_json.name} (resume) ===", flush=True)
            return out_json
        except (json.JSONDecodeError, OSError):
            print(f"\n=== {mnk}^3: {out_json.name} unreadable, re-tuning ===", flush=True)
    env = dict(os.environ)
    env.update(
        IRIS_TUNE_MNK=str(mnk),
        IRIS_TUNE_SPACE="schedule",
        IRIS_TUNE_CHECK="1",
        HSA_NO_SCRATCH_RECLAIM="1",
    )
    env.setdefault("HIP_VISIBLE_DEVICES", "0,1,2,3")
    cmd = [
        sys.executable, str(BENCH),
        "--benchmark_format", "json",
        "--benchmark_out", str(out_json),
        "--n_warmup", str(_N_WARMUP),
        "--n_repeat", str(_N_REPEAT),
    ]
    print(f"\n=== tuning {mnk}^3 (ws={_WORLD_SIZE}) -> {out_json.name} ===", flush=True)
    subprocess.run(cmd, env=env, cwd=str(REPO_ROOT / "iris"), check=True)
    return out_json


def _pick_winner(raw_json: Path):
    """From the runner JSON, return (winner_counters, do_bench_tflops, rccl_tflops)."""
    records = json.loads(raw_json.read_text())
    layout_rows = [
        r for r in records
        if r.get("benchmark") == "layout_tune"
        and not r.get("skipped")
        and r.get("tflops") is not None
        and r.get("counters", {}).get("ok", 1) >= 0.5  # correctness-passing
    ]
    if not layout_rows:
        raise RuntimeError(f"no correctness-passing layout in {raw_json}")
    winner = max(layout_rows, key=lambda r: r["tflops"])
    rccl = next((r for r in records if r.get("benchmark") == "rccl_reference"), None)
    rccl_tflops = rccl["tflops"] if rccl and rccl.get("tflops") else None
    return winner["counters"], winner["tflops"], rccl_tflops


def _reconstruct_layout(mnk: int, c: dict) -> dict:
    """Rebuild the winning ScheduleLayout from runner counters; return a reloadable
    dict (layout knobs) plus its full constexprs contract.

    Counters carry numeric codes: ford 0=kfg/1=mtile; fxcd 0=co-located/else spatial.
    block_size_k, num_warps, num_stages are fixed by the `schedule` space (64/8/2).
    """
    sys.path.insert(0, str(REPO_ROOT / "iris"))
    from iris.ops.schedule_layout import make_layout  # noqa: E402

    knobs = dict(
        order="mtile" if int(round(c["ford"])) == 1 else "kfg",
        fetch_xcds=(None if int(round(c["fxcd"])) == 0 else int(round(c["fxcd"]))),
        fetch_k=int(round(c["fk"])),
        fetch_m=int(round(c["fm"])),
        group_m=int(round(c["gm"])),
        n_fetch_wg=int(round(c["nfw"])),
        n_gemm_wg=int(round(c["ngw"])),
        block_size_m=int(round(c["bm"])),
        block_size_n=int(round(c["bn"])),
        block_size_k=64,
        num_warps=8,
        num_stages=2,
    )
    k_local = mnk // _WORLD_SIZE
    layout, problem = make_layout(
        num_xcds=_NUM_XCDS, M=mnk, N=mnk, K=mnk, K_local=k_local,
        world_size=_WORLD_SIZE,
        fetch_m=knobs["fetch_m"], fetch_k=knobs["fetch_k"], group_m=knobs["group_m"],
        block_size_m=knobs["block_size_m"], block_size_n=knobs["block_size_n"],
        block_size_k=knobs["block_size_k"],
        order=knobs["order"], fetch_xcds=knobs["fetch_xcds"],
        n_fetch_wg=knobs["n_fetch_wg"], n_gemm_wg=knobs["n_gemm_wg"],
    )
    return knobs, {k: int(v) for k, v in layout.constexprs(problem).items()}


def _plot(entries: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = sorted(entries, key=lambda e: e["compute_ai"])
    xs = [e["compute_ai"] for e in entries]
    ys = [e["speedup"] for e in entries]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="RCCL parity (1.0x)")
    ax.plot(xs, ys, "-o", color="#1f77b4", markersize=8, linewidth=1.5,
            label="fused AG+MM (tuned)")
    for e in entries:
        lay = e["layout"]
        role = "coloc" if lay["fetch_xcds"] is None else f"fx{lay['fetch_xcds']}"
        ax.annotate(
            f"{e['mnk']}³\n{lay['order']},{role},fk{lay['fetch_k']}",
            (e["compute_ai"], e["speedup"]),
            textcoords="offset points", xytext=(8, 8), fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("compute arithmetic intensity  (2MNK / HBM bytes, FLOP/byte)")
    ax.set_ylabel("speedup over RCCL  (do_bench TFLOPS ratio)")
    ax.set_title(f"Fused all-gather+matmul vs RCCL — square shapes, ws={_WORLD_SIZE} {_DTYPE}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(_PNG_OUT, dpi=120)
    plt.close(fig)
    print(f"\n[plot] wrote {_PNG_OUT}")


def main():
    entries = []
    for mnk in _SHAPES:
        raw = _run_shape(mnk)
        counters, db_tflops, rccl_tflops = _pick_winner(raw)
        knobs, cexprs = _reconstruct_layout(mnk, counters)
        ai = _compute_ai(mnk)
        speedup = (db_tflops / rccl_tflops) if rccl_tflops else float("nan")
        entries.append(dict(
            mnk=mnk, world_size=_WORLD_SIZE, dtype=_DTYPE,
            compute_ai=ai, do_bench_tflops=db_tflops, rccl_tflops=rccl_tflops,
            speedup=speedup, layout=knobs, constexprs=cexprs,
        ))
        print(f"[{mnk}^3] AI={ai:.1f}  fused={db_tflops:.1f}  rccl={rccl_tflops:.1f}  "
              f"speedup={speedup:.3f}x  win={knobs['order']}/"
              f"{'coloc' if knobs['fetch_xcds'] is None else 'fx'+str(knobs['fetch_xcds'])}"
              f"/fk{knobs['fetch_k']}/nfw{knobs['n_fetch_wg']}/ngw{knobs['n_gemm_wg']}", flush=True)

    _JSON_OUT.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"\n[json] wrote {_JSON_OUT}  ({len(entries)} shapes)")
    _plot(entries)

    print("\n=== summary ===")
    print(f"{'shape':>8} {'AI':>8} {'fused':>8} {'rccl':>8} {'speedup':>8}  winner")
    for e in sorted(entries, key=lambda x: x["compute_ai"]):
        lay = e["layout"]
        role = "coloc" if lay["fetch_xcds"] is None else f"fx{lay['fetch_xcds']}"
        print(f"{e['mnk']:>7}³ {e['compute_ai']:>8.1f} {e['do_bench_tflops']:>8.1f} "
              f"{e['rccl_tflops']:>8.1f} {e['speedup']:>7.3f}x  "
              f"{lay['order']}/{role}/fk{lay['fetch_k']}/nfw{lay['n_fetch_wg']}/ngw{lay['n_gemm_wg']}")


if __name__ == "__main__":
    main()
