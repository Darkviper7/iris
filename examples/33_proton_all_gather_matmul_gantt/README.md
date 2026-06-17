# Example 33: Proton intra-kernel profiling (all-gather + GEMM)

Demonstrates [Triton Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton) instrumentation on the HBM-buffer all-gather + matmul kernel.

This example keeps a local copy with `pl.scope` markers:

- `all_gather_matmul_hbm_buffer_proton.py` — instrumented kernel + `profile=` wrapper
- `profile_and_plot_gantt.py` — launch, collect traces, plot Gantt charts

## Prerequisites

- Iris with multi-GPU NCCL (8 GPUs by default)
- Triton built from source with Proton enabled (`TRITON_BUILD_PROTON=ON`)
  - Pip/release builds often ship rocTracer only. A source build picks up upstream
    rocprofSDK support and falls back to rocTracer when rocprofSDK is unavailable 
- `matplotlib` for plots (optional: `--no_plot` to skip)

## Run

```bash
# Proton chrome trace + hatchet + matplotlib plots
python examples/33_proton_all_gather_matmul_gantt/profile_and_plot_gantt.py --num_ranks 8

# Compare against iris built-in device tracing
python examples/33_proton_all_gather_matmul_gantt/profile_and_plot_gantt.py --num_ranks 8 --mode iris
```

Outputs land in this directory (gitignored): `*.chrome_trace`, `*.hatchet`, `*.png`.

Open `hbm_buffer_all_gather_matmul.chrome_trace` in [Perfetto UI](https://ui.perfetto.dev).
