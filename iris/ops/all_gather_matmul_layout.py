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
import os
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
def _ws_compute_tile(
    tile,
    A_sharded,
    B,
    C,
    bias_ptr,
    staged_a,
    flags_ptr,
    ctx,
    M,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sa_m,
    stride_sa_k,
    stride_bias,
    cur_rank: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    EXACT_TILES: tl.constexpr,
    A_LOAD_CACHE: tl.constexpr,
    B_LOAD_CACHE: tl.constexpr,
    SKIP_LOCAL_STAGE: tl.constexpr,
    GEMM_LOCAL_INTERLEAVE: tl.constexpr,
    FLAGS_PER_RANK: tl.constexpr,
    NUM_FLAG_GROUPS_K: tl.constexpr,
    FETCH_K: tl.constexpr,
    FETCH_XCDS: tl.constexpr,
    GEMM_SLOTS_PER_XCD: tl.constexpr,
    SLAB_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    TRACE: tl.constexpr,
    TRACE_GATHERS: tl.constexpr,
):
    """Compute ONE GEMM output tile given its GLOBAL tile index ``tile``. This is
    the shared per-tile compute used by the WORK_STEAL reserved-tail drain (a
    faithful copy of the non-WORK_STEAL ``do_gemm`` inner body: same
    tile->(pid_m,pid_n) decode, acquire-wait handshake, local-first / interleave /
    plain dot structure, C store).

    TRACE-gated iris events mirror the baseline ``do_gemm`` body EXACTLY (same
    TraceEvent ids -- wait / gemm_read / gemm_read_b / gemm_read_a_local / gemm_dot
    / gemm_store_c -- at the SAME points, all OUTSIDE the innermost k_off dot loop
    per the AMD-pipeliner constraint) so a drained tile renders as a GEMM bar. Every
    event is behind ``if TRACE:`` so TRACE=False (the benchmark default) emits
    nothing and is byte-identical to the benchmarked reserved-tail kernel. The outer
    per-tile ``compute`` span is emitted by the caller (_ws_drain_reserve).
    (``TRACE_GATHERS`` is threaded for signature parity with the baseline; the GEMM
    body does not use it, matching the baseline.) The non-WORK_STEAL do_gemm path is
    left untouched and does NOT call this helper."""
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # global tile index -> (pid_m, pid_n), preserving the XCD->slab pinning.
    if FETCH_XCDS == 0:
        xcd_t = tile // GEMM_SLOTS_PER_XCD
        local = tile % GEMM_SLOTS_PER_XCD
        m0 = xcd_t * SLAB_M
    else:
        local = tile
        m0 = tile * 0  # global m space (== 0), typed to match `local`
    num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
    group_id = local // num_pid_in_group
    within = local % num_pid_in_group
    pid_m = m0 + group_id * GROUP_SIZE_M + (within % GROUP_SIZE_M)
    pid_n = within // GROUP_SIZE_M

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    if SKIP_LOCAL_STAGE and GEMM_LOCAL_INTERLEAVE == 0:
        # LOCAL-FIRST: own shard from A_sharded (wait-free), then remote flag-groups.
        for lf in range(FLAGS_PER_RANK):
            # Per-flag-group anchors (outside k_off loop): gemm_read_a_local, gemm_read_b,
            # gemm_dot -- first-k-block addresses (mirrors baseline do_gemm local-first).
            if TRACE:
                _rkl0 = (lf * FETCH_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                _rkl0 = tl.max_contiguous(tl.multiple_of(_rkl0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                _al_ptrs = A_sharded + rm.to(tl.int64)[:, None] * stride_am + _rkl0[None, :] * stride_ak
                _reada_handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().gemm_read_a_local, target_rank=cur_rank,
                    address=_al_ptrs, pid_m=pid_m, pid_n=lf)
                _rkg0 = (cur_rank * NUM_K_BLOCKS_LOCAL + lf * FETCH_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                _rkg0 = tl.max_contiguous(tl.multiple_of(_rkg0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                _b_ptrs = B + _rkg0[:, None] * stride_bk + rn[None, :] * stride_bn
                _readb_handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                    address=_b_ptrs, pid_m=pid_m, pid_n=lf)
                _dot_handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                    address=_al_ptrs, pid_m=pid_m, pid_n=lf)
            for k_off in range(FETCH_K):
                kbl = lf * FETCH_K + k_off
                rk_l = kbl * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                rk_l = tl.max_contiguous(tl.multiple_of(rk_l, BLOCK_SIZE_K), BLOCK_SIZE_K)
                a_ptrs = A_sharded + rm.to(tl.int64)[:, None] * stride_am + rk_l[None, :] * stride_ak
                a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)
                rk_g = (cur_rank * NUM_K_BLOCKS_LOCAL + kbl) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                rk_g = tl.max_contiguous(tl.multiple_of(rk_g, BLOCK_SIZE_K), BLOCK_SIZE_K)
                B_ptrs = B + rk_g[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)
                if ALLOW_TF32:
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                else:
                    acc += tl.dot(a, b, allow_tf32=False)
            if TRACE:
                ctx.tracing.record_event_end(_dot_handle)
                ctx.tracing.record_event_end(_readb_handle)
                ctx.tracing.record_event_end(_reada_handle)
        for k_fg in range(NUM_FLAG_GROUPS_K):
            if k_fg // FLAGS_PER_RANK != cur_rank:
                flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
                if TRACE:
                    _wait_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().wait, target_rank=cur_rank,
                        address=flags_ptr + flag_idx + tl.arange(0, 1),
                        pid_m=pid_m, pid_n=k_fg)
                while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                    pass
                if TRACE:
                    ctx.tracing.record_event_end(_wait_handle)
                k_block_base = k_fg * FETCH_K
                if TRACE:
                    _read_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + (k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_sa_k
                    _read_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().gemm_read, target_rank=cur_rank,
                        address=_read_ptrs, pid_m=pid_m, pid_n=k_fg)
                    _rkb0 = k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    _rkb0 = tl.max_contiguous(tl.multiple_of(_rkb0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    _b_ptrs = B + _rkb0[:, None] * stride_bk + rn[None, :] * stride_bn
                    _readb_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                        address=_b_ptrs, pid_m=pid_m, pid_n=k_fg)
                    _dot_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                        address=flags_ptr + flag_idx + tl.arange(0, 1),
                        pid_m=pid_m, pid_n=k_fg)
                for k_off in range(FETCH_K):
                    k_block = k_block_base + k_off
                    rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                    a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)
                    B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                    b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)
                    if ALLOW_TF32:
                        acc = tl.dot(a, b, acc, allow_tf32=True)
                    else:
                        acc += tl.dot(a, b, allow_tf32=False)
                if TRACE:
                    ctx.tracing.record_event_end(_dot_handle)
                    ctx.tracing.record_event_end(_readb_handle)
                    ctx.tracing.record_event_end(_read_handle)
    elif SKIP_LOCAL_STAGE:
        # INTERLEAVE: local flag-groups spread evenly among remote ones.
        _lc = 0
        _rc = 0
        for _p in range(NUM_FLAG_GROUPS_K):
            _is_local = ((_p * FLAGS_PER_RANK) // NUM_FLAG_GROUPS_K
                         < ((_p + 1) * FLAGS_PER_RANK) // NUM_FLAG_GROUPS_K)
            if _is_local:
                k_fg = cur_rank * FLAGS_PER_RANK + _lc
                _lc += 1
            else:
                k_fg = _rc if _rc < cur_rank * FLAGS_PER_RANK else _rc + FLAGS_PER_RANK
                _rc += 1
            k_block_base = k_fg * FETCH_K
            if _is_local:
                lf = k_fg - cur_rank * FLAGS_PER_RANK
                if TRACE:
                    _rkl0 = (lf * FETCH_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    _rkl0 = tl.max_contiguous(tl.multiple_of(_rkl0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    _al_ptrs = A_sharded + rm.to(tl.int64)[:, None] * stride_am + _rkl0[None, :] * stride_ak
                    _reada_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().gemm_read_a_local, target_rank=cur_rank,
                        address=_al_ptrs, pid_m=pid_m, pid_n=k_fg)
                    _rkg0 = (cur_rank * NUM_K_BLOCKS_LOCAL + lf * FETCH_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    _b_ptrs = B + _rkg0[:, None] * stride_bk + rn[None, :] * stride_bn
                    _readb_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                        address=_b_ptrs, pid_m=pid_m, pid_n=k_fg)
                    _dot_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                        address=_al_ptrs, pid_m=pid_m, pid_n=k_fg)
                for k_off in range(FETCH_K):
                    kbl = lf * FETCH_K + k_off
                    rk_l = kbl * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    rk_l = tl.max_contiguous(tl.multiple_of(rk_l, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    a_ptrs = A_sharded + rm.to(tl.int64)[:, None] * stride_am + rk_l[None, :] * stride_ak
                    a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)
                    rk_g = (cur_rank * NUM_K_BLOCKS_LOCAL + kbl) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    rk_g = tl.max_contiguous(tl.multiple_of(rk_g, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    B_ptrs = B + rk_g[:, None] * stride_bk + rn[None, :] * stride_bn
                    b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)
                    if ALLOW_TF32:
                        acc = tl.dot(a, b, acc, allow_tf32=True)
                    else:
                        acc += tl.dot(a, b, allow_tf32=False)
                if TRACE:
                    ctx.tracing.record_event_end(_dot_handle)
                    ctx.tracing.record_event_end(_readb_handle)
                    ctx.tracing.record_event_end(_reada_handle)
            else:
                flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
                if TRACE:
                    _wait_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().wait, target_rank=cur_rank,
                        address=flags_ptr + flag_idx + tl.arange(0, 1),
                        pid_m=pid_m, pid_n=k_fg)
                while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                    pass
                if TRACE:
                    ctx.tracing.record_event_end(_wait_handle)
                    _read_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + (k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_sa_k
                    _read_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().gemm_read, target_rank=cur_rank,
                        address=_read_ptrs, pid_m=pid_m, pid_n=k_fg)
                    _rkb0 = k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    _rkb0 = tl.max_contiguous(tl.multiple_of(_rkb0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    _b_ptrs = B + _rkb0[:, None] * stride_bk + rn[None, :] * stride_bn
                    _readb_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                        address=_b_ptrs, pid_m=pid_m, pid_n=k_fg)
                    _dot_handle = ctx.tracing.record_event_start(
                        event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                        address=flags_ptr + flag_idx + tl.arange(0, 1),
                        pid_m=pid_m, pid_n=k_fg)
                for k_off in range(FETCH_K):
                    k_block = k_block_base + k_off
                    rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)
                    a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                    a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)
                    B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                    b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)
                    if ALLOW_TF32:
                        acc = tl.dot(a, b, acc, allow_tf32=True)
                    else:
                        acc += tl.dot(a, b, allow_tf32=False)
                if TRACE:
                    ctx.tracing.record_event_end(_dot_handle)
                    ctx.tracing.record_event_end(_readb_handle)
                    ctx.tracing.record_event_end(_read_handle)
    else:
        for k_fg in range(NUM_FLAG_GROUPS_K):
            flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
            if TRACE:
                _wait_handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().wait, target_rank=cur_rank,
                    address=flags_ptr + flag_idx + tl.arange(0, 1),
                    pid_m=pid_m, pid_n=k_fg)
            while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                pass
            if TRACE:
                ctx.tracing.record_event_end(_wait_handle)
            k_block_base = k_fg * FETCH_K
            # Per-flag-group anchors OUTSIDE the k_off loop (AMD-pipeliner constraint).
            if TRACE:
                _read_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + (k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_sa_k
                _read_handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().gemm_read, target_rank=cur_rank,
                    address=_read_ptrs, pid_m=pid_m, pid_n=k_fg)
                _rkb0 = k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                _rkb0 = tl.max_contiguous(tl.multiple_of(_rkb0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                _b_ptrs = B + _rkb0[:, None] * stride_bk + rn[None, :] * stride_bn
                _readb_handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                    address=_b_ptrs, pid_m=pid_m, pid_n=k_fg)
                _dot_handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                    address=flags_ptr + flag_idx + tl.arange(0, 1),
                    pid_m=pid_m, pid_n=k_fg)
            for k_off in range(FETCH_K):
                k_block = k_block_base + k_off
                rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)
                a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)
                B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)
                if ALLOW_TF32:
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                else:
                    acc += tl.dot(a, b, allow_tf32=False)
            if TRACE:
                ctx.tracing.record_event_end(_dot_handle)
                ctx.tracing.record_event_end(_readb_handle)
                ctx.tracing.record_event_end(_read_handle)

    if BIAS:
        bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
        acc = acc + bias_val[:, None]

    c = acc.to(C.type.element_ty)
    C_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    if TRACE:
        _storec_handle = ctx.tracing.record_event_start(
            event_id=TraceEvent().gemm_store_c, target_rank=cur_rank,
            address=C_ptrs, pid_m=pid_m, pid_n=pid_n)
    if EXACT_TILES:
        tl.store(C_ptrs, c)
    else:
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C_ptrs, c, mask=c_mask, cache_modifier=".wt")
    if TRACE:
        ctx.tracing.record_event_end(_storec_handle)


@triton.jit
def _ws_drain_reserve(
    bucket,
    reserve_base,
    reserve_count,
    steal_next,
    A_sharded,
    B,
    C,
    bias_ptr,
    staged_a,
    flags_ptr,
    ctx,
    pid,
    xcd,
    M,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sa_m,
    stride_sa_k,
    stride_bias,
    cur_rank: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    EXACT_TILES: tl.constexpr,
    A_LOAD_CACHE: tl.constexpr,
    B_LOAD_CACHE: tl.constexpr,
    SKIP_LOCAL_STAGE: tl.constexpr,
    GEMM_LOCAL_INTERLEAVE: tl.constexpr,
    FLAGS_PER_RANK: tl.constexpr,
    NUM_FLAG_GROUPS_K: tl.constexpr,
    FETCH_K: tl.constexpr,
    FETCH_XCDS: tl.constexpr,
    GEMM_SLOTS_PER_XCD: tl.constexpr,
    SLAB_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    TRACE: tl.constexpr,
    TRACE_GATHERS: tl.constexpr,
):
    """Reserved-tail drain: hand out the ``reserve_count`` tiles
    ``[reserve_base, reserve_base+reserve_count)`` one at a time via a single
    atomic counter ``steal_next[bucket]``, computing each with ONE
    ``_ws_compute_tile`` call.

    This is the ONLY dynamic GEMM body on the WORK_STEAL path -- the owner tiles
    run the untouched static strided baseline loop (full pipelining, fetch-aligned
    order), and this drain runs SEQUENTIALLY after it, so the compiler can reuse the
    owner body's registers (ON-path VGPR ~= baseline + one sequential drain body).

    Called by BOTH drained fetch WGs and finished GEMM WGs (same arg types -> one
    specialization -> one inlined compute body). No claim/CAS is needed: the reserve
    tiles [end-reserve, end) are DISJOINT from the owner tiles [start, end-reserve),
    and the atomic counter hands out each reserve tile exactly once across all
    drainers. bucket = xcd (co-located per-XCD tail) or 0 (spatial global tail).

    TRACE-gated: each drained tile is wrapped in a per-tile ``compute`` span
    (pid_m=pid, pid_n=xcd -> the draining WG lane) so a stolen tile renders as a
    GEMM bar attributed to the drainer, and _ws_compute_tile emits the same
    per-flag-group GEMM events the baseline do_gemm body does. All behind
    ``if TRACE:`` -> TRACE=False is byte-identical to the benchmarked kernel."""
    o = tl.atomic_add(steal_next + bucket, 1, sem="relaxed", scope="gpu")
    while o < reserve_count:
        if TRACE:
            _drain_compute_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().compute, target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1), pid_m=pid, pid_n=xcd)
        _ws_compute_tile(
            reserve_base + o, A_sharded, B, C, bias_ptr, staged_a, flags_ptr, ctx, M, N,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            stride_sa_m, stride_sa_k, stride_bias,
            cur_rank, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, BIAS, ALLOW_TF32,
            NUM_K_BLOCKS_LOCAL, EXACT_TILES, A_LOAD_CACHE, B_LOAD_CACHE,
            SKIP_LOCAL_STAGE, GEMM_LOCAL_INTERLEAVE, FLAGS_PER_RANK,
            NUM_FLAG_GROUPS_K, FETCH_K, FETCH_XCDS, GEMM_SLOTS_PER_XCD, SLAB_M,
            GROUP_SIZE_M, NUM_TILES_N, TRACE, TRACE_GATHERS,
        )
        if TRACE:
            ctx.tracing.record_event_end(_drain_compute_handle)
        o = tl.atomic_add(steal_next + bucket, 1, sem="relaxed", scope="gpu")


@triton.jit
def _layout_all_gather_matmul_kernel(
    A_sharded,
    B,
    C,
    bias_ptr,
    staged_a,
    flags_ptr,
    credit_produced,   # int32[NUM_XCDS]: tiles staged/XCD (credit window; unused if CREDIT_WINDOW==0)
    credit_consumed,   # int32[NUM_XCDS]: distinct tiles first-consumed/XCD
    first_seen,        # int32[num_flags]: 0/1 marker set on a staged tile's FIRST consume
    steal_next,        # int32[NUM_XCDS]: reserved-tail drain counter/XCD (WORK_STEAL only; spatial uses [0])
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
    TRACE_GATHERS: tl.constexpr,  # emit a per-gather trace event tagged by source rank (needs bigger buffer)
    EXACT_TILES: tl.constexpr,  # M%BLOCK_M==0 and N%BLOCK_N==0 -> C store needs no mask
    A_LOAD_CACHE: tl.constexpr,  # cache modifier for the GEMM staged-A load ("" = default/.ca)
    B_LOAD_CACHE: tl.constexpr,  # cache modifier for the GEMM B load ("" = default/.ca)
    SKIP_LOCAL_STAGE: tl.constexpr,  # skip staging cur_rank's own shard; GEMM reads it from A_sharded
    PHASE_ADAPTIVE: tl.constexpr,    # spatial only: drained fetch WGs also compute GEMM tiles
    PA_TAIL_START: tl.constexpr,     # first GEMM tile of the reserved fetch-joiner tail
    COMPACT_SPATIAL: tl.constexpr,   # spatial: pack pids [0,n_fetch)=fetch,[n_fetch,..)=gemm (no phantom slots)
    WORK_STEAL: tl.constexpr,        # UNIFIED DYNAMIC GEMM. OFF (False) = byte-identical to baseline (the
                                     # static owner loop runs, the dynamic loop is constexpr-dead). ON: BOTH
                                     # GEMM WGs and drained fetch WGs pull tile CHUNKS from one atomic counter
                                     # (steal_next[bucket]) covering the FULL tile space -- no static owner
                                     # range, no separate reserve. Bypasses PHASE_ADAPTIVE/COMPACT_SPATIAL.
    WS_CHUNK: tl.constexpr,          # tiles claimed per atomic_add on the unified path (locality/contention
                                     # knob; default GROUP_SIZE_M). Inner chunk walk is tl.range(num_stages=2).
                                     # Unused (constexpr-dead) when WORK_STEAL is off.

    PROFILE_SCOPES: tl.constexpr,  # emit fine-grained Proton leaf scopes (off=byte-identical codegen)
    PROFILE_RANK_SCOPES: tl.constexpr,  # name each gather scope by its source rank (fetch_gather_r{0..ws-1})
    CREDIT_WINDOW: tl.constexpr,  # TCP-style flow control: max A-tiles fetcher may run ahead of GEMM
                                  # (co-located only). 0 = OFF -> byte-identical codegen to the baseline.
    GEMM_LOCAL_INTERLEAVE: tl.constexpr,  # skip_local consumer ordering: 0 = LOCAL-FIRST (baseline),
                                  # 1 = INTERLEAVE the wait-free local flag-groups evenly among the
                                  # remote ones (fill remote-wait bubbles). 0 = byte-identical codegen.
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
    FETCH_WAVE: tl.constexpr,          # DEMAND-order (FETCH_ORDER==5) wave width (m-tiles)
):
    pid = tl.program_id(0)
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    zero = tl.program_id(0) * 0

    # SKIP_LOCAL_STAGE: cur_rank's own shard already lives in A_sharded (local HBM),
    # so staging it into staged_a + the flag handshake is redundant. We skip it when
    # a whole flag-group lies within one rank (NUM_K_BLOCKS_LOCAL % FETCH_K == 0, the
    # host gates this), giving FLAGS_PER_RANK flag-groups per rank; the local rank is
    # then flag-group range [cur_rank*FLAGS_PER_RANK, (cur_rank+1)*FLAGS_PER_RANK).
    FLAGS_PER_RANK: tl.constexpr = NUM_K_BLOCKS_LOCAL // FETCH_K

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
        # WORK_STEAL: no static owner range -- all GEMM tiles of this XCD's slab are
        # pulled from the dynamic counter below. gemm_end unused on that path (the
        # static loop is constexpr-gated off). OFF: exact baseline range.
        gemm_end = GEMM_SLOTS_PER_XCD
    elif COMPACT_SPATIAL:
        # COMPACT spatial: no rectangular NUM_XCDS*max(pool) grid (which spawns phantom
        # no-op slots that delay a whole fetch XCD's launch wave). Pack pids linearly:
        # [0, n_fetch_total) = fetchers, [n_fetch_total, +n_gemm_total) = GEMM. Grid is
        # exactly the active-WG count. Trades away the pid%NUM_XCDS on-die XCD isolation
        # (fetch no longer pinned to XCDs 0..FETCH_XCDS-1) to get every fetcher into the
        # first launch wave. m0=0 (global m space), same cell/tile partition as spatial.
        m0 = zero
        n_fetch_total = FETCH_XCDS * N_FETCH_WG
        n_native_gemm = (NUM_XCDS - FETCH_XCDS) * N_GEMM_WG
        do_fetch = pid < n_fetch_total
        fetch_start = pid
        fetch_stride = n_fetch_total
        do_gemm = pid >= n_fetch_total
        gemm_start = pid - n_fetch_total
        gemm_stride = n_native_gemm
        gemm_end = TOTAL_GEMM_TILES
    else:
        m0 = zero  # global m space
        do_fetch = (xcd < FETCH_XCDS) and (slot < N_FETCH_WG)
        fetch_start = xcd * N_FETCH_WG + slot
        fetch_stride = FETCH_XCDS * N_FETCH_WG
        native_gemm = (xcd >= FETCH_XCDS) and (slot < N_GEMM_WG)
        n_native_gemm = (NUM_XCDS - FETCH_XCDS) * N_GEMM_WG
        if PHASE_ADAPTIVE:
            # Drained fetch WGs ALSO compute GEMM, on a RESERVED TAIL of tiles so the
            # native GEMM WGs keep their EXACT baseline assignment (same start/stride ->
            # unchanged GROUP_SIZE_M L2 locality + fetch/compute interleave). Native WGs
            # cover [0, PA_TAIL_START); the n_join drained fetchers cover [PA_TAIL_START,
            # TOTAL_GEMM_TILES), offloading the latest, most-stalled iters to the freed
            # fetch CUs. Deadlock-free: fetchers are pure producers, enter GEMM only
            # after setting all their flags.
            n_join = FETCH_XCDS * N_FETCH_WG
            do_gemm = native_gemm or do_fetch
            if native_gemm:
                gemm_start = (xcd - FETCH_XCDS) * N_GEMM_WG + slot
                gemm_stride = n_native_gemm
                gemm_end = PA_TAIL_START
            else:
                gemm_start = PA_TAIL_START + xcd * N_FETCH_WG + slot
                gemm_stride = n_join
                gemm_end = TOTAL_GEMM_TILES
        else:
            do_gemm = native_gemm
            gemm_start = (xcd - FETCH_XCDS) * N_GEMM_WG + slot
            gemm_stride = n_native_gemm
            # WORK_STEAL: no static owner range -- GEMM tiles pulled from the global
            # dynamic counter below. gemm_end unused on that path. OFF: exact baseline.
            gemm_end = TOTAL_GEMM_TILES

    if do_fetch:
        # ==============================================================
        # FETCHER pool member — strided-loop over this XCD's fetch cells.
        #   cells: slot, slot+N_FETCH_WG, ...  (covers [0, FETCH_SLOTS_PER_XCD)).
        #   Each cell = one flag-group; walks FETCH_M m-tiles x FETCH_K k-blocks.
        # ==============================================================
        pl.enter_scope("all_gather")  # Proton scope (free unless proton.start'd)
        # Outer per-WG fetch span (target_rank=cur_rank). Suppressed under
        # TRACE_GATHERS so the only fetch events are the per-gather ones (tagged by
        # true source rank) -- otherwise this whole-lifetime span, tagged cur_rank,
        # would paint every fetcher as a bogus "gather from rank cur_rank" bar.
        if TRACE and not TRACE_GATHERS:
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
        # Unified fetch-cell walk. NUM_FETCHERS = size of this WG's pool; every WG runs
        # CELLS_PER_FETCHER iterations (guarded by cell < fetch_cells) so the loop trip
        # count is a compile-time constant regardless of order:
        #   orders 0/1/2 (strided round-robin):  cell = fetch_start + j*fetch_stride
        #   order 3 (pipelined, contiguous):     cell = fetch_start*CELLS_PER_FETCHER + j
        # The contiguous block gives each WG whole m-tiles (NUM_FLAG_GROUPS_K consecutive
        # cells each); combined with the m-tile-major (fp_m=cell//NFG, kfg=cell%NFG)
        # mapping below, one WG then walks a single m-tile's k-flag-groups IN ORDER,
        # flagging each as it completes -> the GEMM consumer (which waits kfg 0,1,2,..
        # in order) can dot kfg0 while this WG still fetches kfg1: real fetch/compute
        # pipelining, instead of all of a tile's kfg landing at once from different WGs.
        NUM_FETCHERS: tl.constexpr = (FETCH_XCDS * N_FETCH_WG) if FETCH_XCDS > 0 else N_FETCH_WG
        FETCH_CELLS_C: tl.constexpr = TOTAL_FETCH_CELLS if FETCH_XCDS > 0 else FETCH_SLOTS_PER_XCD
        CELLS_PER_FETCHER: tl.constexpr = (FETCH_CELLS_C + NUM_FETCHERS - 1) // NUM_FETCHERS
        # order 4 (coop): stride over the REMOTE-ONLY flag list (m-major, kfg-minor).
        # remote flags/m-tile = NUM_FLAG_GROUPS_K - FLAGS_PER_RANK (one rank's band is
        # local under SKIP_LOCAL_STAGE). Adjacent fetchers cover adjacent remote kfg of
        # the SAME m-tile, so ~NUM_FETCHERS/REMOTE_KFG m-tiles COMPLETE per wave -> the
        # producer delivers whole m-tiles at the width the GEMM consumes them, instead of
        # only NUM_FETCHERS/NUM_FLAG_GROUPS_K (mtile) which starves the 12-wide GEMM demand.
        REMOTE_KFG: tl.constexpr = NUM_FLAG_GROUPS_K - FLAGS_PER_RANK
        for j in range(CELLS_PER_FETCHER):
            if FETCH_ORDER == 3:
                cell = fetch_start * CELLS_PER_FETCHER + j
            else:
                cell = fetch_start + j * fetch_stride
            if FETCH_ORDER == 5:
                # DEMAND-order (wave-blocked kfg-major) over the dense REMOTE space.
                # Within a block of FETCH_WAVE m-tiles, stage remote-slice r=0 for ALL
                # of them first, then r=1, then r=2 -> matches the GEMM's flag-demand
                # order (every first-wave m-tile wants its first remote slice up front),
                # killing the k_fg=1 stall wave that m-major (mtile/coop) leaves. Uses a
                # remainder-safe bijection so RM need NOT be divisible by FETCH_WAVE.
                rc = cell
                cpw = FETCH_WAVE * REMOTE_KFG           # cells per full wave
                nfull = RM // FETCH_WAVE                 # number of full waves
                last = RM - nfull * FETCH_WAVE           # partial last-wave m-tiles (0..)
                full_cells = nfull * cpw
                if rc < full_cells:
                    w = rc // cpw
                    c = rc % cpw
                    r = c // FETCH_WAVE                  # remote slice advances slowest
                    m_in = c % FETCH_WAVE
                    fp_m = w * FETCH_WAVE + m_in
                else:
                    c = rc - full_cells
                    wsz = last if last > 0 else 1        # avoid /0 (unused when last==0)
                    r = c // wsz
                    m_in = c % wsz
                    fp_m = nfull * FETCH_WAVE + m_in
                k_flag_group = r + FLAGS_PER_RANK if r >= cur_rank * FLAGS_PER_RANK else r
            elif FETCH_ORDER == 4:
                # dense remote-cell -> (m, kfg), skipping the local kfg band
                rc = cell
                fp_m = rc // REMOTE_KFG
                r = rc % REMOTE_KFG
                # shift past this rank's local flag band [cur_rank*FPR, +FPR)
                k_flag_group = r + FLAGS_PER_RANK if r >= cur_rank * FLAGS_PER_RANK else r
            elif FETCH_ORDER == 2:
                # local_first: m-tile-major, but rotate the flag-group by cur_rank so
                # this m-tile's LOCAL gather (kfg == cur_rank, an on-device copy) is
                # dispatched before the remote xGMI gathers. The flag INDEX below
                # uses k_flag_group directly, so the rotated value IS the true kfg and
                # the GEMM consumer contract is unchanged -- only dispatch order shifts.
                fp_m = cell // NUM_FLAG_GROUPS_K
                k_flag_group = (cell + cur_rank) % NUM_FLAG_GROUPS_K
            elif FETCH_ORDER == 1 or FETCH_ORDER == 3:
                # m-tile-major. Order 3 (pipelined) uses the same mapping but a
                # CONTIGUOUS cell walk (set above), so a WG's consecutive cells are
                # kfg 0,1,2,.. of the SAME m-tile -> sequential per-tile k release.
                fp_m = cell // NUM_FLAG_GROUPS_K   # m-tile advances slowest
                k_flag_group = cell % NUM_FLAG_GROUPS_K
            else:
                fp_m = cell % RM                   # flag-group advances slowest
                k_flag_group = cell // RM
            k_block_start = k_flag_group * FETCH_K

            # Skip this cell entirely when it stages cur_rank's own shard: the GEMM
            # reads that data straight from A_sharded, so no staging store + no flag.
            # Also guard the padded tail of the walk: contiguous(3)/strided use the full
            # cell space (cell >= fetch_cells); coop(4) walks the DENSE REMOTE space where
            # fp_m is an m-GROUP index in [0, RM) (RM = num_m_tiles//FETCH_M), so its tail
            # guard is fp_m >= RM. Using NUM_M_TILES here is WRONG for FETCH_M>1 (RM<NMT):
            # the guard never fires, fetchers over-iterate, and flags get double-set ->
            # the GEMM consumer's handshake corrupts and it spins forever (NCCL timeout).
            tail = ((fp_m >= RM) if FETCH_ORDER == 4
                    else (cell >= RM * REMOTE_KFG) if FETCH_ORDER == 5
                    else (cell >= fetch_cells))
            skip_cell = (SKIP_LOCAL_STAGE and (k_flag_group // FLAGS_PER_RANK == cur_rank)) or tail

            if not skip_cell:
                if CREDIT_WINDOW > 0 and FETCH_XCDS == 0:
                    # TCP-style credit throttle (co-located): block until this XCD has
                    # fewer than CREDIT_WINDOW staged-but-not-yet-first-consumed tiles,
                    # so the fetcher cannot run far ahead of the GEMM and blow the tile
                    # out of L2 before it is read. produced/first_consumed are per-XCD
                    # monotonic counters (bumped below / by the consumer). Deadlock-free
                    # ONLY when fetcher + GEMM WGs co-reside (host enforces the pool cap).
                    while (tl.atomic_add(credit_produced + xcd, 0, sem="acquire", scope="gpu")
                           - tl.atomic_add(credit_consumed + xcd, 0, sem="acquire", scope="gpu")
                           ) >= CREDIT_WINDOW:
                        pass
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
                                # Per-gather iris trace event, tagged with the SOURCE rank
                                # (target_rank=compile_rank) so the gantt can color each
                                # individual gather by which peer it pulled from. Gated on
                                # TRACE_GATHERS to keep the default 2-events/fetcher budget.
                                if TRACE and TRACE_GATHERS:
                                    _g_handle = ctx.tracing.record_event_start(
                                        event_id=TraceEvent().fetch,
                                        target_rank=compile_rank,
                                        address=staged_ptrs,
                                        pid_m=m_tile,
                                        pid_n=xcd,
                                    )
                                a_tile = ctx.gather(k_tile, src_view, compile_rank, hint=(1, BLOCK_SIZE_K))
                                tl.store(staged_ptrs, a_tile, cache_modifier=".cg")
                                if TRACE and TRACE_GATHERS:
                                    ctx.tracing.record_event_end(_g_handle)
                                if PROFILE_RANK_SCOPES:
                                    pl.exit_scope(f"fetch_gather_r{compile_rank}")
                    if PROFILE_SCOPES and not PROFILE_RANK_SCOPES:
                        pl.exit_scope("fetch_gather")

                    flag_idx = m_tile * NUM_FLAG_GROUPS_K + k_flag_group
                    if PROFILE_SCOPES:
                        pl.enter_scope("fetch_flag_set")  # barrier + flag release
                    # Trace the flag-set (barrier + release) as its own event so the
                    # gantt can mark the handoff between flag-groups. event_id=atomic_xchg.
                    if TRACE and TRACE_GATHERS:
                        _f_handle = ctx.tracing.record_event_start(
                            event_id=TraceEvent().atomic_xchg,
                            target_rank=cur_rank,
                            address=flags_ptr + flag_idx + tl.arange(0, 1),
                            pid_m=m_tile,
                            pid_n=xcd,
                        )
                    tl.debug_barrier()  # ensure all per-block stores are visible before setting the flag
                    tl.atomic_xchg(flags_ptr + flag_idx, 1, sem="release", scope="gpu")
                    if CREDIT_WINDOW > 0 and FETCH_XCDS == 0:
                        # one more tile staged on this XCD (credit accounting)
                        tl.atomic_add(credit_produced + xcd, 1, sem="release", scope="gpu")
                    if TRACE and TRACE_GATHERS:
                        ctx.tracing.record_event_end(_f_handle)
                    if PROFILE_SCOPES:
                        pl.exit_scope("fetch_flag_set")

        if TRACE and not TRACE_GATHERS:
            ctx.tracing.record_event_end(_trace_handle)
        pl.exit_scope("all_gather")

    if do_gemm:
        # ==============================================================
        # GEMM pool member — strided-loop over output tiles.
        #   co-located: tiles [0, GEMM_SLOTS_PER_XCD) within this XCD's slab.
        #   spatial:    tiles [0, TOTAL_GEMM_TILES) over the global tile space.
        #   Each tile waits on its m-tile's NUM_FLAG_GROUPS_K flags, then computes.
        # ==============================================================
        pl.enter_scope("compute")  # Proton scope (free unless proton.start'd)
        if TRACE:
            _trace_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().compute,
                target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1),
                pid_m=pid,
                pid_n=xcd,
            )

        # OWNERS: the static strided baseline loop (full inter-tile pipelining +
        # fetch-aligned consumption order). Under WORK_STEAL the upper bound collapses
        # to gemm_start (empty range -> constexpr dead-code) so the dynamic counter
        # loop below owns all GEMM tiles; OFF keeps the exact baseline range, so this
        # loop is byte-identical to the original.
        gemm_static_end = gemm_start if WORK_STEAL else gemm_end
        for gtile in range(gemm_start, gemm_static_end, gemm_stride):
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

            # NOTE: the interleaved [wait k_fg; dot k_fg] structure is deliberate.
            # Draining all flags up front then running one flat k-loop was tried
            # (gemm_drain_first) and REGRESSED -13% at 4096^3 / -18% at 8192^3: it
            # serializes the whole tile behind the slowest (last remote) flag-group,
            # killing the compute/fetch overlap the interleaved form gets by computing
            # on early-ready k-groups while later ones are still landing. See log.md
            # 2026-06-25. Keep interleaved.
            if SKIP_LOCAL_STAGE and GEMM_LOCAL_INTERLEAVE == 0:
                # LOCAL-FIRST: cur_rank's own shard is in A_sharded and was never
                # staged, so compute it immediately (no flag wait) at tile entry.
                # This does useful MFMA work while the fetchers are still landing the
                # remote shards into staged_a -- the wait below then overlaps less.
                if PROFILE_SCOPES:
                    pl.enter_scope("gemm_dot")  # local flag-groups, read from A_sharded
                for lf in range(FLAGS_PER_RANK):
                    # Per-flag-group anchors (outside k_off loop): gemm_read_a_local (own
                    # shard from A_sharded, wait-free local-L2 pressure), gemm_read_b,
                    # gemm_dot span. First-k-block addresses.
                    if TRACE:
                        _rkl0 = (lf * FETCH_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        _rkl0 = tl.max_contiguous(tl.multiple_of(_rkl0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                        _al_ptrs = A_sharded + rm.to(tl.int64)[:, None] * stride_am + _rkl0[None, :] * stride_ak
                        _reada_handle = ctx.tracing.record_event_start(
                            event_id=TraceEvent().gemm_read_a_local, target_rank=cur_rank,
                            address=_al_ptrs, pid_m=pid_m, pid_n=lf)
                        _rkg0 = (cur_rank * NUM_K_BLOCKS_LOCAL + lf * FETCH_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        _rkg0 = tl.max_contiguous(tl.multiple_of(_rkg0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                        _b_ptrs = B + _rkg0[:, None] * stride_bk + rn[None, :] * stride_bn
                        _readb_handle = ctx.tracing.record_event_start(
                            event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                            address=_b_ptrs, pid_m=pid_m, pid_n=lf)
                        _dot_handle = ctx.tracing.record_event_start(
                            event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                            address=_al_ptrs, pid_m=pid_m, pid_n=lf)
                    for k_off in range(FETCH_K):
                        # local k-block index within A_sharded's K_local span
                        kbl = lf * FETCH_K + k_off
                        rk_l = kbl * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        rk_l = tl.max_contiguous(tl.multiple_of(rk_l, BLOCK_SIZE_K), BLOCK_SIZE_K)

                        a_ptrs = A_sharded + rm.to(tl.int64)[:, None] * stride_am + rk_l[None, :] * stride_ak
                        a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)

                        rk_g = (cur_rank * NUM_K_BLOCKS_LOCAL + kbl) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        rk_g = tl.max_contiguous(tl.multiple_of(rk_g, BLOCK_SIZE_K), BLOCK_SIZE_K)
                        B_ptrs = B + rk_g[:, None] * stride_bk + rn[None, :] * stride_bn
                        b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)

                        if ALLOW_TF32:
                            acc = tl.dot(a, b, acc, allow_tf32=True)
                        else:
                            acc += tl.dot(a, b, allow_tf32=False)
                    if TRACE:
                        ctx.tracing.record_event_end(_dot_handle)
                        ctx.tracing.record_event_end(_readb_handle)
                        ctx.tracing.record_event_end(_reada_handle)
                if PROFILE_SCOPES:
                    pl.exit_scope("gemm_dot")

                # Then the remote flag-groups: wait + dot from staged_a as usual.
                for k_fg in range(NUM_FLAG_GROUPS_K):
                    if k_fg // FLAGS_PER_RANK != cur_rank:
                        flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
                        if TRACE:
                            _wait_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().wait,
                                target_rank=cur_rank,
                                address=flags_ptr + flag_idx + tl.arange(0, 1),
                                pid_m=pid_m,
                                pid_n=k_fg,
                            )
                        if PROFILE_SCOPES:
                            pl.enter_scope("gemm_wait")
                        while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                            pass
                        if CREDIT_WINDOW > 0 and FETCH_XCDS == 0:
                            # first GEMM WG to read this staged tile frees a producer credit
                            if tl.atomic_xchg(first_seen + flag_idx, 1, sem="relaxed", scope="gpu") == 0:
                                tl.atomic_add(credit_consumed + xcd, 1, sem="release", scope="gpu")
                        if PROFILE_SCOPES:
                            pl.exit_scope("gemm_wait")
                        if TRACE:
                            ctx.tracing.record_event_end(_wait_handle)

                        k_block_base = k_fg * FETCH_K
                        # Per-flag-group anchors (outside k_off loop -- see non-skip path:
                        # per-k-block events between load/dot break the AMD pipeliner).
                        if TRACE:
                            _read_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + (k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_sa_k
                            _read_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().gemm_read, target_rank=cur_rank,
                                address=_read_ptrs, pid_m=pid_m, pid_n=k_fg)
                            _rkb0 = k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                            _rkb0 = tl.max_contiguous(tl.multiple_of(_rkb0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                            _b_ptrs = B + _rkb0[:, None] * stride_bk + rn[None, :] * stride_bn
                            _readb_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                                address=_b_ptrs, pid_m=pid_m, pid_n=k_fg)
                            _dot_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                                address=flags_ptr + flag_idx + tl.arange(0, 1),
                                pid_m=pid_m, pid_n=k_fg)
                        if PROFILE_SCOPES:
                            pl.enter_scope("gemm_dot")
                        for k_off in range(FETCH_K):
                            k_block = k_block_base + k_off
                            rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                            rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

                            a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                            a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)

                            B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                            b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)

                            if ALLOW_TF32:
                                acc = tl.dot(a, b, acc, allow_tf32=True)
                            else:
                                acc += tl.dot(a, b, allow_tf32=False)
                        if PROFILE_SCOPES:
                            pl.exit_scope("gemm_dot")
                        if TRACE:
                            ctx.tracing.record_event_end(_dot_handle)
                            ctx.tracing.record_event_end(_readb_handle)
                            ctx.tracing.record_event_end(_read_handle)
            elif SKIP_LOCAL_STAGE:
                # INTERLEAVE: same skip_local semantics (local read wait-free from
                # A_sharded, remote waited from staged_a) but the FLAGS_PER_RANK local
                # flag-groups are spread EVENLY among the remote ones instead of all
                # computed first -- so wait-free local MFMA fills the bubbles while the
                # later remote tiles are still landing. The order is a compile-time
                # (constexpr) schedule; matmul accumulation is order-independent so any
                # permutation of the k-flag-groups is numerically identical.
                _lc = 0
                _rc = 0
                for _p in range(NUM_FLAG_GROUPS_K):
                    # even-spread (Bresenham): exactly FLAGS_PER_RANK local slots.
                    _is_local = ((_p * FLAGS_PER_RANK) // NUM_FLAG_GROUPS_K
                                 < ((_p + 1) * FLAGS_PER_RANK) // NUM_FLAG_GROUPS_K)
                    if _is_local:
                        k_fg = cur_rank * FLAGS_PER_RANK + _lc
                        _lc += 1
                    else:
                        k_fg = _rc if _rc < cur_rank * FLAGS_PER_RANK else _rc + FLAGS_PER_RANK
                        _rc += 1
                    k_block_base = k_fg * FETCH_K
                    if _is_local:
                        # wait-free local flag-group: read from A_sharded (local K span).
                        lf = k_fg - cur_rank * FLAGS_PER_RANK
                        if TRACE:
                            _rkl0 = (lf * FETCH_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                            _rkl0 = tl.max_contiguous(tl.multiple_of(_rkl0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                            _al_ptrs = A_sharded + rm.to(tl.int64)[:, None] * stride_am + _rkl0[None, :] * stride_ak
                            _reada_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().gemm_read_a_local, target_rank=cur_rank,
                                address=_al_ptrs, pid_m=pid_m, pid_n=k_fg)
                            _rkg0 = (cur_rank * NUM_K_BLOCKS_LOCAL + lf * FETCH_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                            _b_ptrs = B + _rkg0[:, None] * stride_bk + rn[None, :] * stride_bn
                            _readb_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                                address=_b_ptrs, pid_m=pid_m, pid_n=k_fg)
                            _dot_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                                address=_al_ptrs, pid_m=pid_m, pid_n=k_fg)
                        if PROFILE_SCOPES:
                            pl.enter_scope("gemm_dot")
                        for k_off in range(FETCH_K):
                            kbl = lf * FETCH_K + k_off
                            rk_l = kbl * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                            rk_l = tl.max_contiguous(tl.multiple_of(rk_l, BLOCK_SIZE_K), BLOCK_SIZE_K)
                            a_ptrs = A_sharded + rm.to(tl.int64)[:, None] * stride_am + rk_l[None, :] * stride_ak
                            a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)
                            rk_g = (cur_rank * NUM_K_BLOCKS_LOCAL + kbl) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                            rk_g = tl.max_contiguous(tl.multiple_of(rk_g, BLOCK_SIZE_K), BLOCK_SIZE_K)
                            B_ptrs = B + rk_g[:, None] * stride_bk + rn[None, :] * stride_bn
                            b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)
                            if ALLOW_TF32:
                                acc = tl.dot(a, b, acc, allow_tf32=True)
                            else:
                                acc += tl.dot(a, b, allow_tf32=False)
                        if PROFILE_SCOPES:
                            pl.exit_scope("gemm_dot")
                        if TRACE:
                            ctx.tracing.record_event_end(_dot_handle)
                            ctx.tracing.record_event_end(_readb_handle)
                            ctx.tracing.record_event_end(_reada_handle)
                    else:
                        # remote flag-group: wait on the fetcher flag, read from staged_a.
                        flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
                        if TRACE:
                            _wait_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().wait, target_rank=cur_rank,
                                address=flags_ptr + flag_idx + tl.arange(0, 1),
                                pid_m=pid_m, pid_n=k_fg)
                        if PROFILE_SCOPES:
                            pl.enter_scope("gemm_wait")
                        while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                            pass
                        if CREDIT_WINDOW > 0 and FETCH_XCDS == 0:
                            if tl.atomic_xchg(first_seen + flag_idx, 1, sem="relaxed", scope="gpu") == 0:
                                tl.atomic_add(credit_consumed + xcd, 1, sem="release", scope="gpu")
                        if PROFILE_SCOPES:
                            pl.exit_scope("gemm_wait")
                        if TRACE:
                            ctx.tracing.record_event_end(_wait_handle)
                            _read_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + (k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_sa_k
                            _read_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().gemm_read, target_rank=cur_rank,
                                address=_read_ptrs, pid_m=pid_m, pid_n=k_fg)
                            _rkb0 = k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                            _rkb0 = tl.max_contiguous(tl.multiple_of(_rkb0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                            _b_ptrs = B + _rkb0[:, None] * stride_bk + rn[None, :] * stride_bn
                            _readb_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                                address=_b_ptrs, pid_m=pid_m, pid_n=k_fg)
                            _dot_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                                address=flags_ptr + flag_idx + tl.arange(0, 1),
                                pid_m=pid_m, pid_n=k_fg)
                        if PROFILE_SCOPES:
                            pl.enter_scope("gemm_dot")
                        for k_off in range(FETCH_K):
                            k_block = k_block_base + k_off
                            rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                            rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)
                            a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                            a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)
                            B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                            b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)
                            if ALLOW_TF32:
                                acc = tl.dot(a, b, acc, allow_tf32=True)
                            else:
                                acc += tl.dot(a, b, allow_tf32=False)
                        if PROFILE_SCOPES:
                            pl.exit_scope("gemm_dot")
                        if TRACE:
                            ctx.tracing.record_event_end(_dot_handle)
                            ctx.tracing.record_event_end(_readb_handle)
                            ctx.tracing.record_event_end(_read_handle)
            else:
                for k_fg in range(NUM_FLAG_GROUPS_K):
                    flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
                    if TRACE:
                        _wait_handle = ctx.tracing.record_event_start(
                            event_id=TraceEvent().wait,
                            target_rank=cur_rank,
                            address=flags_ptr + flag_idx + tl.arange(0, 1),
                            pid_m=pid_m,
                            pid_n=k_fg,
                        )

                    if PROFILE_SCOPES:
                        pl.enter_scope("gemm_wait")  # spin-wait on the fetcher flag = STALL
                    while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                        pass
                    if CREDIT_WINDOW > 0 and FETCH_XCDS == 0:
                        # first GEMM WG to read this staged tile frees a producer credit
                        if tl.atomic_xchg(first_seen + flag_idx, 1, sem="relaxed", scope="gpu") == 0:
                            tl.atomic_add(credit_consumed + xcd, 1, sem="release", scope="gpu")
                    if PROFILE_SCOPES:
                        pl.exit_scope("gemm_wait")

                    if TRACE:
                        ctx.tracing.record_event_end(_wait_handle)

                    k_block_base = k_fg * FETCH_K
                    # Per-flag-group anchors (one event, first k-block address): gemm_read
                    # (staged-A consume, reuse-distance join key), gemm_read_b (B load),
                    # gemm_dot (the flag-group's MFMA accumulation span). NOTE: these are
                    # deliberately OUTSIDE the k_off loop -- per-k-block events placed
                    # between tl.load and tl.dot break the AMD ttgir pipeliner
                    # (PassManager::run failed); instrumentation cannot interleave the MMA.
                    if TRACE:
                        _read_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + (k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_sa_k
                        _read_handle = ctx.tracing.record_event_start(
                            event_id=TraceEvent().gemm_read, target_rank=cur_rank,
                            address=_read_ptrs, pid_m=pid_m, pid_n=k_fg)
                        _rkb0 = k_block_base * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        _rkb0 = tl.max_contiguous(tl.multiple_of(_rkb0, BLOCK_SIZE_K), BLOCK_SIZE_K)
                        _b_ptrs = B + _rkb0[:, None] * stride_bk + rn[None, :] * stride_bn
                        _readb_handle = ctx.tracing.record_event_start(
                            event_id=TraceEvent().gemm_read_b, target_rank=cur_rank,
                            address=_b_ptrs, pid_m=pid_m, pid_n=k_fg)
                        _dot_handle = ctx.tracing.record_event_start(
                            event_id=TraceEvent().gemm_dot, target_rank=cur_rank,
                            address=flags_ptr + flag_idx + tl.arange(0, 1),
                            pid_m=pid_m, pid_n=k_fg)
                    if PROFILE_SCOPES:
                        pl.enter_scope("gemm_dot")  # MFMA accumulation over this flag-group
                    for k_off in range(FETCH_K):
                        k_block = k_block_base + k_off
                        rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

                        a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                        a = tl.load(a_ptrs, cache_modifier=A_LOAD_CACHE)

                        B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                        b = tl.load(B_ptrs, cache_modifier=B_LOAD_CACHE)

                        if ALLOW_TF32:
                            acc = tl.dot(a, b, acc, allow_tf32=True)
                        else:
                            acc += tl.dot(a, b, allow_tf32=False)
                    if PROFILE_SCOPES:
                        pl.exit_scope("gemm_dot")
                    if TRACE:
                        ctx.tracing.record_event_end(_dot_handle)
                        ctx.tracing.record_event_end(_readb_handle)
                        ctx.tracing.record_event_end(_read_handle)

            if BIAS:
                bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
                acc = acc + bias_val[:, None]

            c = acc.to(C.type.element_ty)
            C_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            if TRACE:
                _storec_handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().gemm_store_c,
                    target_rank=cur_rank,
                    address=C_ptrs,
                    pid_m=pid_m,
                    pid_n=pid_n,
                )
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
                ctx.tracing.record_event_end(_storec_handle)

        if TRACE:
            ctx.tracing.record_event_end(_trace_handle)
        pl.exit_scope("compute")

    if WORK_STEAL:
        # ---- UNIFIED DYNAMIC GEMM LOOP (single site, one inlined _ws_compute_tile) ----
        # Reached by every WG that did work: drained fetch WGs (all flags already set)
        # AND every GEMM WG (its static owner loop above ran an empty range). There is
        # NO static owner range and NO separate reserve -- ALL GEMM tiles are pulled
        # here from one atomic counter steal_next[bucket] in CHUNKS of WS_CHUNK tiles.
        # bucket = xcd (co-located: per-XCD counter over that XCD's [0,GEMM_SLOTS_PER_XCD)
        # slab -> a WG only computes tiles staged in its own XCD's L2) or 0 (spatial:
        # one global counter over [0,TOTAL_GEMM_TILES)). Work is balanced to true finish
        # times: whichever WG is free grabs the next chunk, so a freed fetcher joins the
        # SAME pool the GEMM WGs are draining -- no late serial tail. Exactly-once by the
        # monotonic counter (disjoint chunks, no CAS). Deadlock-free: each tile still
        # acquire-waits its own flags, and fetchers enter only after setting all flags.
        if do_fetch or do_gemm:
            if FETCH_XCDS == 0:
                _cur_bucket = xcd
                _tile_base = xcd * GEMM_SLOTS_PER_XCD
                _tile_count = zero + GEMM_SLOTS_PER_XCD
            else:
                _cur_bucket = zero
                _tile_base = zero
                _tile_count = zero + TOTAL_GEMM_TILES
            # Claim a chunk of WS_CHUNK consecutive tiles per atomic (locality + fewer
            # atomics); walk it with tl.range(num_stages=2) for inter-tile double-buffering.
            _c0 = tl.atomic_add(steal_next + _cur_bucket, WS_CHUNK, sem="relaxed", scope="gpu")
            while _c0 < _tile_count:
                for _ci in tl.range(WS_CHUNK, num_stages=2):
                    _local_tile = _c0 + _ci
                    if _local_tile < _tile_count:
                        if TRACE:
                            _dyn_handle = ctx.tracing.record_event_start(
                                event_id=TraceEvent().compute, target_rank=cur_rank,
                                address=flags_ptr + tl.arange(0, 1), pid_m=pid, pid_n=xcd)
                        _ws_compute_tile(
                            _tile_base + _local_tile,
                            A_sharded, B, C, bias_ptr, staged_a, flags_ptr, ctx, M, N,
                            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                            stride_sa_m, stride_sa_k, stride_bias,
                            cur_rank, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, BIAS, ALLOW_TF32,
                            NUM_K_BLOCKS_LOCAL, EXACT_TILES, A_LOAD_CACHE, B_LOAD_CACHE,
                            SKIP_LOCAL_STAGE, GEMM_LOCAL_INTERLEAVE, FLAGS_PER_RANK,
                            NUM_FLAG_GROUPS_K, FETCH_K, FETCH_XCDS, GEMM_SLOTS_PER_XCD, SLAB_M,
                            GROUP_SIZE_M, NUM_TILES_N, TRACE, TRACE_GATHERS,
                        )
                        if TRACE:
                            ctx.tracing.record_event_end(_dyn_handle)
                _c0 = tl.atomic_add(steal_next + _cur_bucket, WS_CHUNK, sem="relaxed", scope="gpu")


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

    # Credit-window flow control (opt-in). Tiny per-XCD counters + a per-flag
    # first-consume marker; always allocated (few KB) so the launch can pass real
    # pointers, but only touched by the kernel when CREDIT_WINDOW > 0.
    num_xcds = config.num_xcds if config.num_xcds and config.num_xcds > 1 else 8
    ws.credit_produced = ctx.zeros((num_xcds,), dtype=torch.int32)
    ws.credit_consumed = ctx.zeros((num_xcds,), dtype=torch.int32)
    ws.first_seen = ctx.zeros((num_m_tiles * num_flag_groups_k,), dtype=torch.int32)

    # Reserved-tail work-stealing counter (opt-in): one int32 per XCD, drained by
    # fetchers/finished-GEMM WGs when work_steal=True. Tiny; always allocated so the
    # launch has a valid pointer (inert -- never touched -- when work_steal=False).
    ws.steal_next = ctx.zeros((num_xcds,), dtype=torch.int32)

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
    a_load_cache: str = "",
    b_load_cache: str = "",
    skip_local_stage: bool = False,
    local_interleave: bool = False,
    phase_adaptive: bool = False,
    compact_spatial: bool = False,
    work_steal: bool = False,
    work_steal_chunk: Optional[int] = None,
    credit_window: Optional[int] = None,
    cus_per_xcd: int = 38,
    validate_layout: bool = False,
    trace: bool = False,
    trace_gathers: bool = False,
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

    ``work_steal`` (default False) turns on the UNIFIED DYNAMIC GEMM loop: there is
    no static owner range and no reserved tail -- ALL GEMM WGs and drained fetch WGs
    pull tile CHUNKS (``work_steal_chunk`` tiles, default ``group_m``) from ONE atomic
    counter (``steal_next``) that covers the full tile space (per-XCD counter when
    co-located -> stays L2-local; one global counter when spatial). Work is balanced
    to true finish times: a freed fetcher joins the same pool the GEMM WGs drain, so
    there is no late serial tail (the failure mode of the earlier reserved-tail form,
    which lost 26-34% on compute-bound shapes). Each chunk is walked with
    ``tl.range(num_stages=2)`` for inter-tile double-buffering; the inner k-block
    load/dot loops (which carry the MFMA pipelining) are unchanged. Exactly-once by the
    monotonic counter (disjoint chunks, no CAS). Default off is byte-identical to the
    baseline (WORK_STEAL constexpr dead; static owner loop runs its exact range); when
    on it bypasses ``phase_adaptive`` / ``compact_spatial`` and is mutually exclusive
    with ``credit_window``.
    """
    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = ctx.get_num_ranks()
    trace = trace or trace_gathers  # per-gather events require device tracing on

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

    # ---- credit-window (TCP-style producer run-ahead bound) validation ----
    # OFF by default (credit_window None/<=0) -> CREDIT_WINDOW=0 constexpr -> the kernel
    # compiles the exact baseline (all throttle/signal blocks are constexpr-dead).
    credit_w = int(credit_window) if credit_window else 0
    if credit_w > 0:
        # Co-located only: the throttle makes a fetcher wait on its OWN XCD's GEMM
        # consumer; in spatial layouts producer and consumer live on different XCDs, so
        # the per-XCD counters would never balance -> permanent stall.
        if cexprs["FETCH_XCDS"] != 0:
            raise ValueError(
                "credit_window is only supported for co-located layouts (FETCH_XCDS==0); "
                f"got FETCH_XCDS={cexprs['FETCH_XCDS']}. Use the co-located schedule "
                "(fetch_xcds=None / no spatial fetch XCDs)."
            )
        # Deadlock safety: a throttled fetcher spins holding its CU slot; the GEMM WG
        # that frees its credit MUST be able to run concurrently. Require every WG of an
        # XCD to co-reside at occupancy 1 (n_fetch_wg + n_gemm_wg <= cus_per_xcd). The
        # defaults are the FULL pools (n_gemm_wg == gemm_slots, often hundreds) -> the
        # caller MUST pass small explicit pools when enabling the window.
        nfw = cexprs["N_FETCH_WG"]; ngw = cexprs["N_GEMM_WG"]
        if nfw + ngw > cus_per_xcd:
            raise ValueError(
                f"credit_window needs fetcher+GEMM WGs co-resident per XCD to avoid "
                f"deadlock: n_fetch_wg + n_gemm_wg = {nfw}+{ngw}={nfw + ngw} > "
                f"cus_per_xcd={cus_per_xcd}. Pass smaller n_fetch_wg/n_gemm_wg (e.g. "
                f"n_fetch_wg=4, n_gemm_wg=16), or raise cus_per_xcd if your part has more."
            )

    # ---- work-stealing (dynamic GEMM rebalancing) validation ----
    # OFF by default -> WORK_STEAL=0 constexpr -> byte-identical baseline codegen.
    work_steal_on = bool(work_steal)
    if work_steal_on and credit_w > 0:
        # The credit-window fetcher throttle waits on GEMM consumers; work-steal turns
        # drained fetchers into GEMM consumers, so the two flow-control schemes would
        # interact in untested ways. Keep them mutually exclusive.
        raise ValueError("work_steal is incompatible with credit_window; enable only one.")

    if workspace is None:
        workspace = all_gather_matmul_layout_preamble(ctx, A_sharded, B, config, fetch_k, staged_a_layout)

    workspace.locks.zero_()
    if credit_w > 0:
        # a workspace built before the credit-window feature (or by a path that didn't
        # allocate them) may lack these; allocate on demand so reuse is safe.
        if workspace.credit_produced is None or workspace.first_seen is None:
            _nx = config.num_xcds if config.num_xcds and config.num_xcds > 1 else 8
            workspace.credit_produced = ctx.zeros((_nx,), dtype=torch.int32)
            workspace.credit_consumed = ctx.zeros((_nx,), dtype=torch.int32)
            workspace.first_seen = ctx.zeros((workspace.locks.numel(),), dtype=torch.int32)
        workspace.credit_produced.zero_()
        workspace.credit_consumed.zero_()
        workspace.first_seen.zero_()

    # ---- unified dynamic GEMM: chunk grain + per-launch counter init ----
    # WS_CHUNK = tiles claimed per atomic on the dynamic path (locality/contention knob).
    # Default = GROUP_SIZE_M so a WG's chunk stays within one B-column group (L2 reuse) and
    # matches the baseline tile-walk grain; overridable via the work_steal_chunk arg.
    # steal_next[NUM_XCDS] is the shared tile cursor: bucket=xcd (co-located) / [0] (spatial).
    _fx0 = cexprs["FETCH_XCDS"]
    _nx0 = cexprs["NUM_XCDS"]
    _gsm0 = cexprs["GROUP_SIZE_M"]
    ws_chunk = 1
    if work_steal_on:
        ws_chunk = int(work_steal_chunk) if work_steal_chunk else _gsm0
        ws_chunk = max(1, ws_chunk)
        # steal_next: one cursor per XCD (co-located buckets) / [0] (spatial); (re)alloc
        # on demand so a reused / pre-feature workspace is safe, then zero each launch.
        if workspace.steal_next is None or workspace.steal_next.numel() != _nx0:
            workspace.steal_next = ctx.zeros((_nx0,), dtype=torch.int32)
        workspace.steal_next.zero_()
    else:
        # Kernel never touches it when WORK_STEAL is off, but the launch needs a valid
        # pointer; ensure non-None (a pre-feature workspace may lack it).
        if workspace.steal_next is None:
            workspace.steal_next = ctx.zeros((_nx0,), dtype=torch.int32)

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

    # Skip staging cur_rank's own shard only when a flag-group lies wholly within one
    # rank (else a group straddles a rank boundary and can't be cleanly skipped).
    skip_local = skip_local_stage and (num_k_blocks_local % fetch_k == 0)

    # Phase-adaptive (spatial only): drained fetch WGs also compute GEMM tiles, so the
    # fetch XCDs' CUs aren't idle in the comm-bound back half. Co-located already has
    # every XCD doing both, so it's a no-op there. PA_TAIL_START splits the global tile
    # space so native GEMM WGs keep their exact baseline assignment on [0, TAIL) and the
    # joiners cover the tail; tail size ~ proportional to the joiners' CU share.
    # work_steal subsumes phase_adaptive (true dynamic rebalancing) -> gate it off so
    # the native GEMM WGs use their WORK_STEAL contiguous ranges, not the tail-split.
    phase_adapt = phase_adaptive and (cexprs["FETCH_XCDS"] > 0) and not work_steal_on
    _fx = cexprs["FETCH_XCDS"]
    _n_native = (cexprs["NUM_XCDS"] - _fx) * cexprs["N_GEMM_WG"]
    _n_join = _fx * cexprs["N_FETCH_WG"]
    _total_tiles = cexprs["TOTAL_GEMM_TILES"]
    pa_tail_start = (
        (_total_tiles * _n_native) // (_n_native + _n_join) if phase_adapt else _total_tiles
    )

    # Compact spatial: linear pid pack (no phantom slots). Grid shrinks to exactly the
    # active-WG count so every fetcher lands in the first launch wave. Spatial only;
    # incompatible with phase_adaptive (different tile decode) -> gate it off.
    # WORK_STEAL relies on the standard (non-compact) pid decode + XCD->slab pinning
    # for its native-WG global index and tile ranges, so compact packing is gated off.
    compact = compact_spatial and (_fx > 0) and not phase_adapt and not work_steal_on
    if compact:
        grid_size = _n_fetch_total = _fx * cexprs["N_FETCH_WG"]
        grid_size = _n_fetch_total + _n_native

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
        # With trace_gathers, each fetcher also emits 2 events per individual gather.
        # Upper bound: every fetch cell it can walk * FETCH_M * FETCH_K gathers * 2.
        if trace_gathers:
            fk = cexprs["FETCH_K"]; fm = cexprs["FETCH_M"]
            ftiles_per_wg = -(-fetch_slots // nfw)  # ceil; max cells any fetch WG walks
            # 2 events per gather (fm*fk) + 2 per flag-set (fm) per cell
            per_fetch += ftiles_per_wg * (fm * fk * 2 + fm * 2)
        # per gemm WG (per tile it walks), all PER FLAG-GROUP (x2 start/end):
        #   2 compute span + nfg*2 wait + nfg*2 gemm_dot + nfg*2 gemm_read
        #   + nfg*2 gemm_read_b + nfg*2 gemm_read_a_local (skip-local; nfg >= FLAGS_PER_RANK)
        #   + 2 gemm_store_c (per tile).
        per_gemm = (2 + gtiles_per_wg * (nfg * 2 * 5 + 2))
        max_trace_events = cexprs["NUM_XCDS"] * (nfw * per_fetch + ngw * per_gemm) + 64  # +headroom
        # WORK_STEAL unified dynamic loop: the static owner loop runs empty (per_gemm's
        # gtiles_per_wg term is then slack), and instead the FULL tile space is computed
        # once across all WGs via the dynamic counter. Each dynamic tile emits the SAME
        # per-tile GEMM events as an owner tile (nfg flag-groups * 2 start/end * 5 event
        # types + 2 gemm_store_c) PLUS a per-tile `compute` span (2). Budget the full
        # tile space (co-located: GEMM_SLOTS_PER_XCD*NUM_XCDS; spatial: TOTAL_GEMM_TILES)
        # on top of the (now-slack) static estimate -- a safe over-count.
        if work_steal_on:
            _dyn_tiles = (gemm_slots * cexprs["NUM_XCDS"]) if _fx0 == 0 else cexprs["TOTAL_GEMM_TILES"]
            max_trace_events += _dyn_tiles * (nfg * 2 * 5 + 2 + 2)
        # The per-WG event estimate assumes an even tile-per-WG split; rectangular shapes
        # (e.g. N=11008 -> 43 n-tiles) walk unevenly and can emit more events per WG than
        # budgeted, overrunning the buffer (illegal memory access). IRIS_TRACE_SAFETY scales
        # the budget (default 1.0 = unchanged); the buffer is a few MB so 2-4x is cheap.
        _safety = float(os.environ.get("IRIS_TRACE_SAFETY", "1.0"))
        if _safety > 1.0:
            max_trace_events = int(max_trace_events * _safety)
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
        workspace.credit_produced,
        workspace.credit_consumed,
        workspace.first_seen,
        workspace.steal_next,
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
        trace_gathers,
        (M % config.block_size_m == 0) and (N % config.block_size_n == 0),  # EXACT_TILES
        a_load_cache,
        b_load_cache,
        skip_local,
        phase_adapt,
        pa_tail_start,
        compact,
        work_steal_on,
        ws_chunk,    # WS_CHUNK (tiles/atomic on the dynamic path; 1 when work_steal off)
        profile_scopes or profile_rank_scopes,
        profile_rank_scopes,
        credit_w,  # CREDIT_WINDOW constexpr (0 = disabled -> baseline codegen)
        1 if local_interleave else 0,  # GEMM_LOCAL_INTERLEAVE (0 = local-first baseline)
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
