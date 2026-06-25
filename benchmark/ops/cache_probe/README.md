# cache_probe — in-process L2 (TCC) counter tool

Reads per-kernel L2 cache counters (`TCC_HIT`/`TCC_MISS`, hit-rate) for the fused
all-gather+matmul kernel via the **rocprofiler-sdk loaded in-process**. This
sidesteps the external `rocprofv3` CLI, which aborts ("configuration outside valid
period") because PyTorch already loads `librocprofiler-register` — the in-process
tool registers as the process's own client and configures cleanly.

Used to establish the spatial-vs-co-located cache finding (spatial L2 hit-rate 0.68
vs co-located 0.48 at 4096³) — see `work-wiki/all_gather_matmul_layout/log.md`.

## Files
- `tcc_tool.cpp` — the SDK tool: dispatch-counting service, kernel-name filtered,
  CSV out. (idempotent init; handles the .so being loaded twice.)
- `Makefile` — one `g++ -shared` rule.
- `run_cache_probe.py` — per-rank driver (builds the layout for `CACHE_MODE`).
- `run.sh` — 4-rank launch + the env the SDK needs.

## Build (on a GPU node)
```
cd iris/benchmark/ops/cache_probe && make    # -> tcc_tool.so
```

## Use
`run.sh` wraps it for the spatial-vs-coloc probe:
```
bash run.sh      # prints TCC_HIT / TCC_MISS / hit_rate for spatial vs coloc
```
To point the tool at any run, load it in-process and set the env:
```
LD_LIBRARY_PATH=/opt/rocm-*/lib:<torch/lib> \
ROCPROFILER_LIBRARY_CTOR=1 \
LD_PRELOAD=/opt/rocm-*/lib/librocprofiler-sdk.so \
ROCP_TOOL_LIBRARIES=$PWD/tcc_tool.so \
TCC_OUT=out.csv TCC_KERNEL_SUB=all_gather_matmul TCC_COUNTERS=TCC_HIT \
  .venv/bin/python <your_kernel_script>
```
Tool env vars: `TCC_OUT` (csv path), `TCC_KERNEL_SUB` (kernel-name substring
filter), `TCC_COUNTERS` (comma list), `TCC_DUMP_COUNTERS` (list available counters).

## Gotchas
- Needs `ROCPROFILER_LIBRARY_CTOR=1` + `LD_PRELOAD` of `librocprofiler-sdk.so`, or
  `tool_init` never fires.
- `TCC_HIT`+`TCC_MISS` exceed one hardware pass on gfx942 — collect them in two
  separate runs (kernel is deterministic, so hit-rate = hit/(hit+miss) is valid).
- Counters are kernel-aggregate per GPU (fetch+GEMM blended); cannot isolate
  per-WG/per-role.
