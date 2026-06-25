# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM driven by a hierarchical ScheduleLayout descriptor.

Dedicated fetcher (producer) and GEMM (consumer) workgroups communicate through a
local HBM staging buffer + spinlock flags. The ``pid`` -> tile decode is fed by
the flat constexprs of :class:`iris.ops.schedule_layout.ScheduleLayout`: each XCD
owns a full-K m-slab it both produces and consumes (co-located), the fetcher
footprint and CU spatial/temporal ordering pick producer cells, and ``group_m``
orders the consumers. Correctness is checked against a plain torch all-gather+mm
reference; there is no comparison to the legacy ``all_gather_matmul_hbm_buffer``.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import triton.profiler.language as pl
import iris

from iris.host.tracing.events import TraceEvent
from .config import FusedConfig
from .workspace import FusedWorkspace
from .schedule_layout import ScheduleLayout, Problem, default_layout, KERNEL_CONSTEXPR_KEYS
from .all_gather_matmul_hbm_buffer import _extract_wg_trace


@triton.jit
def _layout_all_gather_matmul_kernel(
    A_sharded,
    B,
    C,
    bias_ptr,
    staged_a,
    flags_ptr,
    M,
    N,
    K,
    K_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sa_m,  # staged_a stride in M dim
    stride_sa_k,  # staged_a stride in K dim
    stride_bias,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    TRACE: tl.constexpr,
    EXACT_TILES: tl.constexpr,  # M%BLOCK_M==0 and N%BLOCK_N==0 -> C store needs no mask
    PROFILE_SCOPES: tl.constexpr,  # emit fine-grained Proton leaf scopes (off=byte-identical codegen)
    PROFILE_RANK_SCOPES: tl.constexpr,  # name each gather scope by its source rank (fetch_gather_r{0..ws-1})
    # ---- ScheduleLayout.constexprs() decode contract (see KERNEL_CONSTEXPR_KEYS) ----
    NUM_XCDS: tl.constexpr,
    SLAB_M: tl.constexpr,
    RM: tl.constexpr,
    FETCH_M: tl.constexpr,
    FETCH_K: tl.constexpr,
    N_FETCH_WG: tl.constexpr,
    N_GEMM_WG: tl.constexpr,
    FETCH_SLOTS_PER_XCD: tl.constexpr,
    GEMM_SLOTS_PER_XCD: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    NUM_FLAG_GROUPS_K: tl.constexpr,
    FETCH_ORDER: tl.constexpr,         # 0 = kfg-major, 1 = m-tile-major
    FETCH_XCDS: tl.constexpr,          # 0 = co-located; >=1 = spatial fetch-only XCDs
    NUM_M_TILES: tl.constexpr,
    TOTAL_FETCH_CELLS: tl.constexpr,   # spatial: global fetch-cell count
    TOTAL_GEMM_TILES: tl.constexpr,    # spatial: global gemm-tile count
):
    pid = tl.program_id(0)
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    zero = tl.program_id(0) * 0

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size, tracing=TRACE)

    # ---- hierarchical decode (mirrors ScheduleLayout.decode verbatim) ----
    # Two role modes (compile-time branch on FETCH_XCDS):
    #   co-located (FETCH_XCDS==0): each XCD owns m-slab [xcd*SLAB_M,(xcd+1)*SLAB_M)
    #     across all K and both produces+consumes it; roles split by slot within XCD.
    #   spatial   (FETCH_XCDS>0): XCDs [0,FETCH_XCDS) are fetch-only and partition the
    #     GLOBAL fetch-cell space; the rest are gemm-only over the GLOBAL tile space.
    xcd = pid % NUM_XCDS
    slot = pid // NUM_XCDS
    if FETCH_XCDS == 0:
        m0 = xcd * SLAB_M
        do_fetch = slot < N_FETCH_WG
        fetch_start = slot
        fetch_stride = N_FETCH_WG
        do_gemm = slot >= N_FETCH_WG
        gemm_start = slot - N_FETCH_WG
        gemm_stride = N_GEMM_WG
    else:
        m0 = zero  # global m space
        do_fetch = (xcd < FETCH_XCDS) and (slot < N_FETCH_WG)
        fetch_start = xcd * N_FETCH_WG + slot
        fetch_stride = FETCH_XCDS * N_FETCH_WG
        do_gemm = (xcd >= FETCH_XCDS) and (slot < N_GEMM_WG)
        gemm_start = (xcd - FETCH_XCDS) * N_GEMM_WG + slot
        gemm_stride = (NUM_XCDS - FETCH_XCDS) * N_GEMM_WG

    if do_fetch:
        # ==============================================================
        # FETCHER pool member — strided-loop over this XCD's fetch cells.
        #   cells: slot, slot+N_FETCH_WG, ...  (covers [0, FETCH_SLOTS_PER_XCD)).
        #   Each cell = one flag-group; walks FETCH_M m-tiles x FETCH_K k-blocks.
        # ==============================================================
        pl.enter_scope("all_gather")  # Proton scope (free unless proton.start'd)
        if TRACE:
            _trace_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().fetch,
                target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1),
                pid_m=pid,
                pid_n=xcd,
            )

        src_view = iris.make_tensor_view(A_sharded, M, K_local, stride_am, stride_ak)

        # cell space + traversal differ by mode/order (compile-time constants):
        #   co-located: cells [0, FETCH_SLOTS_PER_XCD) within this XCD's slab
        #   spatial:    cells [0, TOTAL_FETCH_CELLS) over the global m space
        fetch_cells = TOTAL_FETCH_CELLS if FETCH_XCDS > 0 else FETCH_SLOTS_PER_XCD
        for cell in range(fetch_start, fetch_cells, fetch_stride):
            if FETCH_ORDER == 2:
                # local_first: m-tile-major, but rotate the flag-group by cur_rank so
                # this m-tile's LOCAL gather (kfg == cur_rank, an on-device copy) is
                # dispatched before the remote xGMI gathers. The flag INDEX below
                # uses k_flag_group directly, so the rotated value IS the true kfg and
                # the GEMM consumer contract is unchanged -- only dispatch order shifts.
                fp_m = cell // NUM_FLAG_GROUPS_K
                k_flag_group = (cell + cur_rank) % NUM_FLAG_GROUPS_K
            elif FETCH_ORDER == 1:
                fp_m = cell // NUM_FLAG_GROUPS_K   # m-tile advances slowest
                k_flag_group = cell % NUM_FLAG_GROUPS_K
            else:
                fp_m = cell % RM                   # flag-group advances slowest
                k_flag_group = cell // RM
            k_block_start = k_flag_group * FETCH_K

            for fi_m in range(FETCH_M):
                m_tile = m0 + fp_m * FETCH_M + fi_m

                rm = m_tile * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)

                if PROFILE_SCOPES and not PROFILE_RANK_SCOPES:
                    pl.enter_scope("fetch_gather")  # remote gather + .cg staged store
                for k_off in range(FETCH_K):
                    k_block_global = k_block_start + k_off

                    src_rank_idx = k_block_global // NUM_K_BLOCKS_LOCAL
                    k_block_local = k_block_global % NUM_K_BLOCKS_LOCAL

                    pid_m_t = zero + m_tile
                    tile_k_t = zero + k_block_local
                    k_tile = iris.TileView(pid_m_t, tile_k_t, BLOCK_SIZE_M, BLOCK_SIZE_K)

                    rk = k_block_global * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    staged_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k

                    for compile_rank in tl.static_range(world_size):
                        if src_rank_idx == compile_rank:
                            # When rank-coloring, name the scope by source rank so the
                            # Gantt can color each gather by which peer it pulled from.
                            # static_range makes compile_rank a true Python int, so the
                            # f-string resolves to a static literal per unrolled branch
                            # and enter/exit are balanced within this matched `if`.
                            if PROFILE_RANK_SCOPES:
                                pl.enter_scope(f"fetch_gather_r{compile_rank}")
                            a_tile = ctx.gather(k_tile, src_view, compile_rank, hint=(1, BLOCK_SIZE_K))
                            tl.store(staged_ptrs, a_tile, cache_modifier=".cg")
                            if PROFILE_RANK_SCOPES:
                                pl.exit_scope(f"fetch_gather_r{compile_rank}")
                if PROFILE_SCOPES and not PROFILE_RANK_SCOPES:
                    pl.exit_scope("fetch_gather")

                flag_idx = m_tile * NUM_FLAG_GROUPS_K + k_flag_group
                if PROFILE_SCOPES:
                    pl.enter_scope("fetch_flag_set")  # barrier + flag release
                tl.debug_barrier()  # ensure all per-block stores are visible before setting the flag
                tl.atomic_xchg(flags_ptr + flag_idx, 1, sem="release", scope="gpu")
                if PROFILE_SCOPES:
                    pl.exit_scope("fetch_flag_set")

        if TRACE:
            ctx.tracing.record_event_end(_trace_handle)
        pl.exit_scope("all_gather")

    if do_gemm:
        # ==============================================================
        # GEMM pool member — strided-loop over output tiles.
        #   co-located: tiles [0, GEMM_SLOTS_PER_XCD) within this XCD's slab.
        #   spatial:    tiles [0, TOTAL_GEMM_TILES) over the global tile space.
        #   Each tile waits on its m-tile's NUM_FLAG_GROUPS_K flags, then computes.
        # ==============================================================
        gemm_tiles = TOTAL_GEMM_TILES if FETCH_XCDS > 0 else GEMM_SLOTS_PER_XCD

        pl.enter_scope("compute")  # Proton scope (free unless proton.start'd)
        if TRACE:
            _trace_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().compute,
                target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1),
                pid_m=pid,
                pid_n=xcd,
            )

        for gtile in range(gemm_start, gemm_tiles, gemm_stride):
            num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
            group_id = gtile // num_pid_in_group
            within = gtile % num_pid_in_group
            pid_m = m0 + group_id * GROUP_SIZE_M + (within % GROUP_SIZE_M)
            pid_n = within // GROUP_SIZE_M

            rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

            for k_fg in range(NUM_FLAG_GROUPS_K):
                if TRACE:
                    _wait_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().wait,
                        target_rank=cur_rank,
                        address=flags_ptr + tl.arange(0, 1),
                        pid_m=pid,
                        pid_n=k_fg,
                    )

                flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
                if PROFILE_SCOPES:
                    pl.enter_scope("gemm_wait")  # spin-wait on the fetcher flag = STALL
                while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                    pass
                if PROFILE_SCOPES:
                    pl.exit_scope("gemm_wait")

                if TRACE:
                    ctx.tracing.record_event_end(_wait_handle)

                k_block_base = k_fg * FETCH_K
                if PROFILE_SCOPES:
                    pl.enter_scope("gemm_dot")  # MFMA accumulation over this flag-group
                for k_off in range(FETCH_K):
                    k_block = k_block_base + k_off
                    rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

                    a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                    a = tl.load(a_ptrs)

                    B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                    b = tl.load(B_ptrs)

                    if ALLOW_TF32:
                        acc = tl.dot(a, b, acc, allow_tf32=True)
                    else:
                        acc += tl.dot(a, b, allow_tf32=False)
                if PROFILE_SCOPES:
                    pl.exit_scope("gemm_dot")

            if BIAS:
                bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
                acc = acc + bias_val[:, None]

            c = acc.to(C.type.element_ty)
            C_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            if PROFILE_SCOPES:
                pl.enter_scope("gemm_store_c")  # epilogue HBM write
            if EXACT_TILES:
                # M,N divisible by the block -> every lane is in-bounds, so the
                # bounds mask is dead; drop it. Also drop the .wt write-through: C
                # is the final output, no other WG reads it back through the flag
                # protocol, so the store needn't bypass cache. (The store width
                # stays dwordx2 either way -- it's the MMA-accumulator per-lane
                # layout, not the mask/cache modifier -- but removing the dead mask
                # + write-through is a small measured win: ~246 -> ~251 do_bench.)
                tl.store(C_ptrs, c)
            else:
                c_mask = (rm[:, None] < M) & (rn[None, :] < N)
                tl.store(C_ptrs, c, mask=c_mask, cache_modifier=".wt")
            if PROFILE_SCOPES:
                pl.exit_scope("gemm_store_c")

        if TRACE:
            ctx.tracing.record_event_end(_trace_handle)
        pl.exit_scope("compute")


# ==========================================================================
# Python API
# ==========================================================================


def _default_config() -> FusedConfig:
    """Block sizes / tf32 / num_xcds for the layout kernel (no champion data)."""
    return FusedConfig(
        block_size_m=128,
        block_size_n=256,
        block_size_k=64,
        num_xcds=8,
    )


def all_gather_matmul_layout_preamble(
    ctx,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    fetch_k: int = 1,
    staged_a_layout: str = "k_contiguous",
) -> FusedWorkspace:
    """Allocate workspace for the layout-driven kernel (aux_buffer + flags).

    ``fetch_k`` is the k-blocks-per-flag handoff granularity (== FetcherLayout.k);
    it sets the flag count = num_m_tiles * (num_k_blocks // fetch_k)."""
    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = ctx.get_num_ranks()

    if config is None:
        config = _default_config()

    assert world_size * K_local == K
    assert K_local % config.block_size_k == 0
    assert K % config.block_size_k == 0
    assert M % config.block_size_m == 0

    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % fetch_k == 0
    num_flag_groups_k = num_k_blocks // fetch_k

    ws = FusedWorkspace(
        operation="all_gather_matmul_layout",
        shape=(M, N, K),
        dtype=A_sharded.dtype,
        world_size=world_size,
        variant=f"layout_{staged_a_layout}",
        prepared=True,
    )

    if staged_a_layout == "m_contiguous":
        storage = ctx.zeros((K, M), dtype=A_sharded.dtype)
        ws.aux_buffer = storage.T  # (M, K) view, M-contiguous
    else:
        ws.aux_buffer = ctx.zeros((M, K), dtype=A_sharded.dtype)

    ws.locks = ctx.zeros((num_m_tiles * num_flag_groups_k,), dtype=torch.int32)

    buffer_mb = M * K * A_sharded.element_size() / (1024**2)
    sa_stride_m, sa_stride_k = ws.aux_buffer.stride()
    ctx.info(
        f"Layout buffer: staged_a=({M},{K}) [{buffer_mb:.1f} MB] "
        f"layout={staged_a_layout} strides=({sa_stride_m},{sa_stride_k}), "
        f"flags={num_m_tiles}x{num_flag_groups_k}, fetch_k={fetch_k}"
    )

    ctx.barrier()
    return ws


def all_gather_matmul_layout(
    ctx,
    output_tensor: torch.Tensor,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    layout: Optional[ScheduleLayout] = None,
    fetch_k: Optional[int] = None,
    n_fetch_wg: Optional[int] = None,
    n_gemm_wg: Optional[int] = None,
    staged_a_layout: str = "k_contiguous",
    num_warps: Optional[int] = 8,
    num_stages: Optional[int] = 2,
    validate_layout: bool = False,
    trace: bool = False,
    profile: bool = False,
    profile_name: str = "ag_mm_layout",
    profile_format: str = "tree",
    profile_scopes: bool = False,
    profile_rank_scopes: bool = False,
) -> FusedWorkspace:
    """
    All-gather + matmul whose pid->tile decode is driven by a hierarchical
    ScheduleLayout (producer fetch + consumer GEMM, co-located per XCD).

    When ``layout`` is None, derives a valid layout from the shape via
    ``schedule_layout.default_layout`` (``fetch_k`` = k-blocks-per-flag handoff
    grain; auto-picked when None). ``n_fetch_wg`` / ``n_gemm_wg`` are the
    persistent producer/consumer WG-pool sizes per XCD; None = full pool (one WG
    per cell/tile = the original behavior). When an explicit ``layout`` is passed
    its pools are used directly; set ``validate_layout=True`` to re-run its
    coverage asserts (off by default — builders already validate, and
    ``validate()`` is O(grid_size) host work).
    """
    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = ctx.get_num_ranks()

    if config is None:
        config = _default_config()

    rank = ctx.get_rank()

    assert world_size * K_local == K
    assert output_tensor.shape == (M, N)
    assert M % config.block_size_m == 0
    assert K % config.block_size_k == 0
    assert K_local % config.block_size_k == 0

    if layout is None:
        layout, problem = default_layout(
            M=M,
            N=N,
            K=K,
            K_local=K_local,
            world_size=world_size,
            num_xcds=config.num_xcds if config.num_xcds and config.num_xcds > 1 else 8,
            block_size_m=config.block_size_m,
            block_size_n=config.block_size_n,
            block_size_k=config.block_size_k,
            fetch_k=fetch_k,
            n_fetch_wg=n_fetch_wg,
            n_gemm_wg=n_gemm_wg,
        )
    else:
        problem = Problem(
            M=M,
            N=N,
            K=K,
            K_local=K_local,
            block_size_m=config.block_size_m,
            block_size_n=config.block_size_n,
            block_size_k=config.block_size_k,
            world_size=world_size,
        )
        if validate_layout:
            layout.validate(problem)

    # flag granularity is owned by the layout (FetcherLayout.k)
    fetch_k = layout.xcd.cu.fetcher.k
    cexprs = layout.constexprs(problem)
    grid_size = layout.grid_size(problem)

    if workspace is None:
        workspace = all_gather_matmul_layout_preamble(ctx, A_sharded, B, config, fetch_k, staged_a_layout)

    workspace.locks.zero_()

    stride_am, stride_ak = A_sharded.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()
    stride_sa_m, stride_sa_k = workspace.aux_buffer.stride()

    if bias is not None:
        assert bias.shape[0] == M
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    num_k_blocks_local = K_local // config.block_size_k

    if trace:
        # Persistent WGs emit events inside strided loops, so grid_size*4 is far
        # too small. Budget exactly: each fetch WG -> 2 (start+end); each gemm WG
        # -> 2 (compute start+end) + tiles_per_wg * nfg_k * 2 (wait start+end).
        nfw = cexprs["N_FETCH_WG"]
        ngw = cexprs["N_GEMM_WG"]
        fetch_slots = cexprs["FETCH_SLOTS_PER_XCD"]
        gemm_slots = cexprs["GEMM_SLOTS_PER_XCD"]
        nfg = cexprs["NUM_FLAG_GROUPS_K"]
        gtiles_per_wg = -(-gemm_slots // ngw)  # ceil; max tiles any gemm WG walks
        per_fetch = 2
        per_gemm = 2 + gtiles_per_wg * nfg * 2
        max_trace_events = cexprs["NUM_XCDS"] * (nfw * per_fetch + ngw * per_gemm) + 64  # +headroom
        if not ctx.tracing.enabled:
            ctx.tracing.enable(max_events=max_trace_events)
        else:
            ctx.tracing.reset()

    launch_kwargs = {}
    if getattr(torch.version, "hip", None):
        launch_kwargs["matrix_instr_nonkdim"] = 16
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    if profile:
        # Proton instrumentation: scopes "all_gather"/"compute" in the kernel are
        # timed without iris device tracing (no get_xcc_id asm / event buffers).
        import triton.profiler as proton
        import triton.profiler.language as _pl
        from triton.profiler.mode import Default

        _pl.enable_semantic("triton")
        proton.start(profile_name, data=profile_format,
                     backend="instrumentation", mode=Default(buffer_type="global"))

    _layout_all_gather_matmul_kernel[(grid_size,)](
        A_sharded,
        B,
        output_tensor,
        bias_ptr,
        workspace.aux_buffer,
        workspace.locks,
        M,
        N,
        K,
        K_local,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_sa_m,
        stride_sa_k,
        stride_bias,
        ctx.get_device_context(),
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        use_bias,
        config.allow_tf32,
        num_k_blocks_local,
        trace,
        (M % config.block_size_m == 0) and (N % config.block_size_n == 0),  # EXACT_TILES
        profile_scopes or profile_rank_scopes,
        profile_rank_scopes,
        *(cexprs[k] for k in KERNEL_CONSTEXPR_KEYS),
        **launch_kwargs,
    )

    if profile:
        import triton.profiler as proton

        torch.cuda.synchronize()
        proton.finalize()

    if not async_op:
        ctx.barrier()

    if trace:
        torch.cuda.synchronize()
        workspace.trace_data = _extract_wg_trace(
            ctx,
            grid_size,
            num_xcds=cexprs["NUM_XCDS"],
            slab_m=cexprs["SLAB_M"],
            fetch_slots_per_xcd=cexprs["FETCH_SLOTS_PER_XCD"],
            gemm_slots_per_xcd=cexprs["GEMM_SLOTS_PER_XCD"],
            num_m_tiles=cexprs["NUM_M_TILES"],
            num_tiles_n=cexprs["NUM_TILES_N"],
        )

    return workspace
