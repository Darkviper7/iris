# Example 33: Proton intra-kernel profiling (all-gather + GEMM)

Demonstrates [Triton Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton) instrumentation on the HBM-buffer all-gather + matmul kernel.

This example keeps a local copy with `pl.scope` markers:

- `all_gather_matmul_hbm_buffer_proton.py` — instrumented kernel + `profile=` wrapper
- `profile_and_plot_gantt.py` — launch, collect traces, plot Gantt charts

## Prerequisites

- Triton built from source with Proton enabled (`TRITON_BUILD_PROTON=ON`)
  - Pip/release builds often ship rocTracer only. A source build picks up upstream
    rocprofSDK support and falls back to rocTracer when rocprofSDK is unavailable.

## Usage

### Proton mode (default)

Uses the example-local instrumented kernel (`pl.scope` markers). Rank 0 runs two Proton
sessions: a chrome trace (timeline) and a hatchet tree (aggregate cycles).

```bash
python examples/33_proton_all_gather_matmul_gantt/profile_and_plot_gantt.py --num_ranks 8
```

**Outputs:** `hbm_buffer_all_gather_matmul.chrome_trace`, `.hatchet`, and matplotlib PNGs
(Gantt, line, and CU-activity plots).

View the trace in [Perfetto UI](https://ui.perfetto.dev). Inspect hatchet with:

```bash
proton-viewer -m normalized_cycles hbm_buffer_all_gather_matmul.hatchet
```

### Iris tracing mode

Uses the production `iris.ops.all_gather_matmul_hbm_buffer` with built-in device tracing
(all ranks, merged JSON).

```bash
python examples/33_proton_all_gather_matmul_gantt/profile_and_plot_gantt.py \
    --num_ranks 8 --mode iris
```

**Outputs:** `hbm_buffer_all_gather_matmul_iris_merged.json` (or `_rank0.json`) plus PNGs.

### Trace only (skip plots)

```bash
python examples/33_proton_all_gather_matmul_gantt/profile_and_plot_gantt.py \
    --num_ranks 8 --no_plot
```



All generated outputs (`*.chrome_trace`, `*.hatchet`, `*.json`, `*.png`) are written to this directory.
