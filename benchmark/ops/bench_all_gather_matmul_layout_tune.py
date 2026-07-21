#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tuning sweep for the hierarchical-layout AG+MM kernel, expressed as layouts.

Shape: square M=N=K, default 4096^3 on 4 ranks fp16; override the dim with
IRIS_TUNE_MNK (e.g. IRIS_TUNE_MNK=1024).

The tuning space IS a set of layouts
------------------------------------
``schedule_layout.enumerate_layouts(...)`` yields one ``LayoutCandidate`` per
valid point in the space -- each bundles a validated ``ScheduleLayout`` + its
``Problem`` (block sizes) + launch knobs (num_warps/num_stages). The space is
built from the shape's divisor structure, so every candidate is valid by
construction. The benchmark simply indexes that list with a single ``LID`` axis:
a tuning point = a layout. ALL config (order, fetch_xcds, fetch_k, pools, tiling,
warps/stages) comes from the enumeration -- there are no hand-built configs and no
side probe scripts.

Measurement gates (env; applied UNIFORMLY to every enumerated layout, in the
untimed setup phase -- they are MEASUREMENT, not schedule config):
  IRIS_TUNE_CHECK=1   assert each layout's output == torch all_gather@mm ref
                      before timing (adds maxdiff/ok columns; wrong layouts are
                      skipped, never reported as a perf number).
  IRIS_TUNE_WARM=1    also report a warm-cache TFLOPS column (no L2 clear),
                      alongside the framework's L2-cleared do_bench number, to
                      expose cache-residency sensitivity.
  IRIS_TUNE_ATOL / IRIS_TUNE_RTOL   tolerances for the correctness gate.

Each candidate's ``label`` (e.g. ``bm128_bn256_bk64_fk8_fm1_gm2_nw8_ns2``) names
the knobs, and is printed as a counter so the table maps LID -> layout.

Controlling the space (set IRIS_TUNE_SPACE env; it selects which layouts are
enumerated, NOT the LID axis):
  focused     bm=(128,256) bn=(128,256) bk=(64,) fk=(4,8,16,32) fm=(1,)
              full WG pools (one WG per cell/tile)            [default]
  full        all divisor values for every knob, full pools (largest space)
  blocks      sweep only block sizes (fk/fm/gm auto-derived)
  specialize  hold the champion tiling (bm256 bn256 fk4 fm1 gm2) and sweep the
              persistent producer/consumer WG-pool split (n_fetch_wg, n_gemm_wg)
              under the deadlock-safe cap n_fetch_wg+n_gemm_wg <= CUS_PER_XCD.
  schedule    the SCHEDULE campaign: sweep the two schedule-shaping layout knobs
              -- FetcherLayout.order (kfg vs m-tile-major fetch traversal) and
              XCDLayout.fetch_xcds (co-located vs spatial role-segregated XCDs) --
              plus fetch_k and the WG-pool split. Winner at 4096^3 ws4 (gated
              sweep, IRIS_TUNE_BM=256): order=mtile, fetch_xcds=2, fetch_k=16,
              gm1, nfw8/ngw32 -> 245.8 TFLOPS (do_bench, L2-cleared) / 287.7 warm;
              ties with fx1/nfw16/ngw32 (243.6/284.5). NOTE: time with async_op=True
              so do_bench's own barrier isn't doubled by the op's internal one.

Knobs ``nfw`` (n_fetch_wg) / ``ngw`` (n_gemm_wg) are the persistent WG-pool sizes
per XCD: that many fetch / gemm WGs each strided-loop over their share of cells /
tiles. Full pools = one WG per cell/tile (the original behavior). Smaller pools
dedicate CUs to a role (the specialization lever).

Run (4 GPUs):
  HIP_VISIBLE_DEVICES=0,1,2,3 IRIS_TUNE_SPACE=specialize .venv/bin/python \
      iris/benchmark/ops/bench_all_gather_matmul_layout_tune.py

A trailing ``rccl_reference`` row gives the baseline to beat at this shape.
"""

import os

import torch
import torch.distributed as dist
import iris
import iris.bench as bench
from iris.ops.all_gather_matmul_layout import (
    all_gather_matmul_layout,
    all_gather_matmul_layout_preamble,
)
from iris.ops.config import FusedConfig
from iris.ops.schedule_layout import enumerate_layouts

# Problem shape. Default 4096^3 (square via IRIS_TUNE_MNK, e.g. =1024). For
# non-square shapes (e.g. large-M / small-K, where fetcher-footprint economics
# differ) override per dim: IRIS_TUNE_M / IRIS_TUNE_N / IRIS_TUNE_K. Per-dim wins
# over the square default.
_MNK = int(os.environ.get("IRIS_TUNE_MNK", "4096"))
_M = int(os.environ.get("IRIS_TUNE_M", _MNK))
_N = int(os.environ.get("IRIS_TUNE_N", _MNK))
_K = int(os.environ.get("IRIS_TUNE_K", _MNK))
_RANKS = 4
_NUM_XCDS = 8
_K_LOCAL = _K // _RANKS
# CUs per XCD on MI300X ~= 304/8 = 38; persistent pools must fit (deadlock-safe).
_CUS_PER_XCD = 38


def _build_space():
    """Materialize the tuning space as a list of LayoutCandidate objects.

    Selectable via the IRIS_TUNE_SPACE env var so the LID axis stays a simple
    integer index the bench CLI can slice with --axis_LID.
    """
    space = os.environ.get("IRIS_TUNE_SPACE", "focused")
    common = dict(num_xcds=_NUM_XCDS)
    if space == "full":
        # all divisors for every tiling/handoff knob, full WG pools
        kw = dict(
            block_size_m=(128, 256), block_size_n=(128, 256), block_size_k=(64,),
        )
    elif space == "blocks":
        kw = dict(
            block_size_m=(128, 256), block_size_n=(128, 256), block_size_k=(64,),
            fetch_k=(8,), fetch_m=(1,), group_m=None,
        )
    elif space == "specialize":
        # Hold the champion tiling; sweep the persistent producer/consumer split.
        # n_gemm_wg=32 (= gemm_slots at bm256), n_fetch_wg in {2,4,6} keeps
        # n_fetch_wg+n_gemm_wg <= 38 (deadlock-safe). Also try smaller gemm pools.
        kw = dict(
            block_size_m=(256,), block_size_n=(256,), block_size_k=(64,),
            fetch_k=(4,), fetch_m=(1,), group_m=(2,),
            n_fetch_wg=(2, 4, 6), n_gemm_wg=(16, 24, 32),
            cus_per_xcd=_CUS_PER_XCD,
        )
    elif space == "schedule":
        # The SCHEDULE campaign: sweep the two schedule-shaping layout knobs that
        # decompose the perf gap, expressed as layouts:
        #   FetcherLayout.order  (kfg vs m-tile-major fetch traversal)
        #   XCDLayout.fetch_xcds (co-located vs spatial role-segregated XCDs)
        # plus fetch_k (handoff grain) and the WG-pool split. Co-located pools are
        # per-XCD (slab); spatial pools are per-role-XCD over the global space, so
        # the n_*_wg tuples cover both regimes -- enumerate_layouts range-caps each
        # and skips block/pool combos that don't divide the shape. Block sizes and
        # pool sizes are wide so this works across shapes (IRIS_TUNE_MNK); only
        # block_size_m giving num_m_tiles % num_xcds == 0 survives.
        #   4096^3: bm256 -> 16 m-tiles (slab 2). 1024^3: bm128 -> 8, bm64 -> 16.
        # IRIS_TUNE_BM (csv) narrows block_size_m to focus the sweep on a known-good
        # tiling (e.g. IRIS_TUNE_BM=256 at 4096^3, where bm256 dominates) without
        # leaving the enumeration -- it just restricts the axis.
        _bm = tuple(int(x) for x in os.environ.get("IRIS_TUNE_BM", "64,128,256").split(","))
        kw = dict(
            block_size_m=_bm, block_size_n=(256,), block_size_k=(64,),
            fetch_k=(4, 8, 16), fetch_m=(1,), group_m=(1, 2),
            order=("kfg", "mtile"),
            fetch_xcds=(None, 1, 2, 4),
            n_fetch_wg=(4, 8, 16, 32),
            n_gemm_wg=(4, 8, 16, 32),
            cus_per_xcd=_CUS_PER_XCD,
        )
    elif space == "fetchm":
        # fetch_m>1 across shapes: does fattening the fetcher footprint ever win?
        # Whether fm>1 helps is SHAPE-DEPENDENT -- large M amortizes per-cell
        # overhead over a fatter fetcher; small K shrinks the m-tile-major delay
        # penalty; launch-bound shapes want fewer/fatter fetch WGs. So this space
        # is deliberately WIDE on every other axis (block sizes, fetch_k, pools,
        # role split) and sweeps fetch_m {1,2,4,8} against all of it. The winning
        # NON-fm config is found per shape from the same enumeration, so the fm
        # comparison is always against that shape's own best -- not a 4k-pinned one.
        # IRIS_TUNE_BM narrows blocks (e.g. 256 at 4k, 64,128 at 1k) for speed.
        _bm = tuple(int(x) for x in os.environ.get("IRIS_TUNE_BM", "64,128,256").split(","))
        kw = dict(
            block_size_m=_bm, block_size_n=(256,), block_size_k=(64,),
            fetch_k=(4, 8, 16), fetch_m=(1, 2, 4, 8), group_m=(1, 2),
            order=("mtile",),
            fetch_xcds=(1, 2),
            n_fetch_wg=(8, 16, 32),
            n_gemm_wg=(16, 32),
            cus_per_xcd=_CUS_PER_XCD,
        )
    else:  # focused
        kw = dict(
            block_size_m=(128, 256), block_size_n=(128, 256), block_size_k=(64,),
            fetch_k=(4, 8, 16, 32), fetch_m=(1,),
        )
    return list(enumerate_layouts(_M, _N, _K, _K_LOCAL, _RANKS, **common, **kw))


_SPACE = _build_space()
_LIDS = list(range(len(_SPACE)))

# Print the LID -> layout label map once (rank 0 only) so the numeric table can
# be cross-referenced back to full layout descriptions.
if os.environ.get("RANK", "0") == "0":
    _space_name = os.environ.get("IRIS_TUNE_SPACE", "focused")
    print(f"[tune] space='{_space_name}' has {len(_SPACE)} layouts:")
    for _lid, _c in enumerate(_SPACE):
        print(f"[tune]   LID {_lid:3d}  {_c.label}  grid={_c.grid_size()}")


# Symmetric-heap allocations are NEVER freed, so allocating per-LID would exhaust
# the heap across a large space (OOM observed at ~hundreds of layouts). Cache and
# reuse: A_sharded/B/C are shape-fixed (allocate once per dtype); the workspace's
# flag count depends on (bm, bk, fetch_k) so cache one workspace per that key.
# This bounds heap use to a handful of allocations regardless of |space|.
_IO_CACHE = {}        # dtype -> (A_sharded, B, C)
_WS_CACHE = {}        # (bm, bk, fetch_k) -> FusedWorkspace
_REF_CACHE = {}       # dtype -> reference output tensor (torch all_gather @ mm)

# Optional measurement gates applied UNIFORMLY to every enumerated LayoutCandidate
# (these are measurement, NOT schedule config — config comes only from the layout):
#   IRIS_TUNE_CHECK=1   assert each layout's output matches the torch reference
#                       before timing; a wrong layout is skipped, never reported.
#   IRIS_TUNE_WARM=1    also report a warm-cache TFLOPS counter (no L2 clear),
#                       alongside the framework's L2-cleared do_bench number.
_CHECK = os.environ.get("IRIS_TUNE_CHECK", "0") == "1"
_WARM = os.environ.get("IRIS_TUNE_WARM", "0") == "1"
# Skip staging cur_rank's own shard (GEMM reads it from A_sharded) + compute it
# local-first. Auto-disabled per-shape when a flag-group straddles a rank boundary.
_SKIP_LOCAL = os.environ.get("IRIS_TUNE_SKIP_LOCAL", "0") == "1"
_ATOL = float(os.environ.get("IRIS_TUNE_ATOL", "1e-2"))
_RTOL = float(os.environ.get("IRIS_TUNE_RTOL", "1e-2"))


def _get_io(ctx, dtype):
    if dtype not in _IO_CACHE:
        rank = ctx.get_rank()
        A = ctx.randn((_M, _K_LOCAL), dtype=dtype, generator=torch.Generator("cuda").manual_seed(42 + rank))
        B = torch.randn((_K, _N), device="cuda", dtype=dtype, generator=torch.Generator("cuda").manual_seed(123))
        C = ctx.zeros((_M, _N), dtype=dtype)
        _IO_CACHE[dtype] = (A, B, C)
    return _IO_CACHE[dtype]


def _get_ws(ctx, A, B, config, fetch_k):
    key = (config.block_size_m, config.block_size_k, fetch_k)
    if key not in _WS_CACHE:
        _WS_CACHE[key] = all_gather_matmul_layout_preamble(ctx, A, B, config=config, fetch_k=fetch_k)
    return _WS_CACHE[key]


def _get_ref(ctx, A, B, dtype):
    """torch all_gather + mm reference for the cached IO (built once per dtype)."""
    if dtype not in _REF_CACHE:
        world_size = ctx.get_num_ranks()
        gathered = [torch.empty_like(A) for _ in range(world_size)]
        dist.all_gather(gathered, A)
        _REF_CACHE[dtype] = torch.matmul(torch.cat(gathered, dim=1), B)
        torch.cuda.synchronize()
    return _REF_CACHE[dtype]


def _warm_tflops(run, C, locks, flops, warmup=10, iters=30):
    """Warm-cache TFLOPS (no L2 clear): the producer/consumer overlap timed the
    way the standalone experiments measured it. Complements the framework's
    L2-cleared do_bench number, exposing the cache-residency sensitivity."""
    for _ in range(warmup):
        C.zero_(); locks.zero_(); run()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return flops / 1e12 / (ms * 1e-3)


@bench.register
@bench.axis("num_ranks", [_RANKS])
@bench.axis("LID", _LIDS)  # index into the enumerated layout space
@bench.axis("dtype", [torch.float16])
def layout_tune(state, ctx):
    """Run one enumerated LayoutCandidate (a tuning point = a layout)."""
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    if world_size != _RANKS:
        state.skip(f"tune fixed to {_RANKS} ranks, got {world_size}")
        return

    cand = _SPACE[state["LID"]]
    layout, problem = cand.layout, cand.problem
    M, N, K = problem.M, problem.N, problem.K
    K_local = problem.K_local

    config = FusedConfig(
        block_size_m=problem.block_size_m,
        block_size_n=problem.block_size_n,
        block_size_k=problem.block_size_k,
        num_xcds=_NUM_XCDS,
    )

    A_sharded, B, C = _get_io(ctx, dtype)
    workspace = _get_ws(ctx, A_sharded, B, config, cand.fetch_k)

    # Self-describing table: numeric knob columns (counters must be numeric; the
    # full label -> LID map is printed once to stdout, see _build_space()).
    ce = cand.constexprs()
    state.add_counter("bm", problem.block_size_m)
    state.add_counter("bn", problem.block_size_n)
    state.add_counter("fk", cand.fetch_k)
    state.add_counter("fm", layout.xcd.cu.fetcher.m)
    state.add_counter("gm", layout.consumer.group_m)
    state.add_counter("nfw", ce["N_FETCH_WG"])
    state.add_counter("ngw", ce["N_GEMM_WG"])
    state.add_counter("ford", ce["FETCH_ORDER"])  # 0=kfg, 1=mtile
    state.add_counter("fxcd", ce["FETCH_XCDS"])   # 0=co-located, else spatial fetch XCDs
    state.add_counter("slab_m", ce["SLAB_M"])
    state.add_counter("nfg_k", ce["NUM_FLAG_GROUPS_K"])
    state.add_counter("grid", cand.grid_size())

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())

    # async_op=True: do_bench already brackets each timed iteration with
    # ctx.barrier(); letting the op add its own internal barrier would charge an
    # extra cross-rank sync per iteration that the standalone experiments never
    # paid (and that is not part of the kernel's compute).
    run = lambda: all_gather_matmul_layout(
        ctx, C, A_sharded, B,
        config=config, workspace=workspace, layout=layout,
        num_warps=cand.num_warps, num_stages=cand.num_stages,
        skip_local_stage=_SKIP_LOCAL,
        async_op=True,
    )

    # Measurement gates applied UNIFORMLY to every enumerated layout (setup phase,
    # untimed). These replace the old throwaway probe scripts: correctness and the
    # warm-cache timer are now bench options over the SAME layout space, so the
    # enumeration stays the single source of truth for config.
    if _CHECK:
        C.zero_(); workspace.locks.zero_()
        run()
        torch.cuda.synchronize(); ctx.barrier()
        ref = _get_ref(ctx, A_sharded, B, dtype)
        max_diff = (C - ref).abs().max().item()
        ok = torch.allclose(C, ref, atol=_ATOL, rtol=_RTOL)
        state.add_counter("maxdiff", max_diff)
        state.add_counter("ok", 1 if ok else 0)
        if not ok:
            state.skip(f"incorrect: max_diff={max_diff:.4f} > atol={_ATOL}")
            return

    if _WARM:
        state.add_counter(
            "warm_tflops",
            _warm_tflops(run, C, workspace.locks, 2 * M * N * K),
        )

    state.exec(run, preamble_fn=lambda: (C.zero_(), workspace.locks.zero_()))


@bench.register
@bench.axis("num_ranks", [_RANKS])
@bench.axis("dtype", [torch.float16])
def rccl_reference(state, ctx):
    """RCCL all_gather + torch.mm reference at the swept shape (IRIS_TUNE_MNK)."""
    M, N, K = _M, _N, _K
    dtype = state["dtype"]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    K_local = K // world_size

    A_sharded = torch.randn(
        (M, K_local), device="cuda", dtype=dtype, generator=torch.Generator("cuda").manual_seed(42 + rank)
    )
    B = torch.randn((K, N), device="cuda", dtype=dtype, generator=torch.Generator("cuda").manual_seed(123))
    A_gathered_list = [torch.empty((M, K_local), device="cuda", dtype=dtype) for _ in range(world_size)]
    C = torch.empty((M, N), device="cuda", dtype=dtype)

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())

    def _run():
        dist.all_gather(A_gathered_list, A_sharded)
        A_gathered = torch.cat(A_gathered_list, dim=1)
        torch.mm(A_gathered, B, out=C)

    # Phase breakdown (untimed setup; same L2-cleared do_bench the harness uses).
    # RCCL runs comm-then-GEMM on one stream with NO overlap, so the two phase
    # times sum to the combined total measured by state.exec below. We time them
    # in isolation to show, on the slide, how much of RCCL's wall time is comm vs
    # the pure-hipBLASLt GEMM (fp16 torch.mm lowers to hipBLASLt on ROCm).
    A_gathered = torch.cat(A_gathered_list, dim=1)  # persistent buffer for GEMM-only

    def _comm():
        dist.all_gather(A_gathered_list, A_sharded)
        torch.cat(A_gathered_list, dim=1, out=A_gathered)

    def _gemm():
        torch.mm(A_gathered, B, out=C)

    comm_ms = iris.do_bench(_comm, barrier_fn=ctx.barrier)
    gemm_ms = iris.do_bench(_gemm, barrier_fn=ctx.barrier)
    state.add_counter("comm_ms", comm_ms)
    state.add_counter("gemm_ms", gemm_ms)

    state.exec(_run)


if __name__ == "__main__":
    bench.main()
