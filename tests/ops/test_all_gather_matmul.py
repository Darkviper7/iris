# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for fused all_gather + matmul operations.

Each rank has A_sharded (M x K_local), B is replicated.
The operation gathers A from all ranks and computes C = A_gathered @ B.
Covers both the baseline pull kernel and the HBM-buffered kernel.
"""

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ops.all_gather_matmul_hbm_buffer import (
    _auto_config,
    _CHAMPION_CONFIGS,
    all_gather_matmul_hbm_buffer,
    all_gather_matmul_hbm_buffer_preamble,
)
from iris.ops.all_gather_matmul_layout import (
    all_gather_matmul_layout,
    all_gather_matmul_layout_preamble,
)
from iris.ops.all_gather_matmul_fused_a8x8 import all_gather_matmul_fused_a8x8
from iris.ops.config import FusedConfig
from iris.ops.schedule_layout import make_layout


def _make_reference(rank, world_size, M, K_local, N, dtype):
    """Build a torch reference output for all_gather + matmul."""
    device = f"cuda:{rank}"
    K = K_local * world_size

    torch.manual_seed(42 + rank)
    A_sharded = torch.randn(M, K_local, dtype=dtype, device=device)

    torch.manual_seed(123)
    B = torch.randn(K, N, dtype=dtype, device=device)

    A_gathered_list = [torch.zeros(M, K_local, dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_gather(A_gathered_list, A_sharded)
    A_gathered_ref = torch.cat(A_gathered_list, dim=1)
    ref_output = torch.matmul(A_gathered_ref, B)
    torch.cuda.synchronize()
    return A_sharded, B, ref_output


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    [
        (128, 32, 64),
        (256, 64, 128),
    ],
)
def test_all_gather_matmul_baseline(dtype, atol, rtol, M, K_local, N):
    """Test baseline all_gather_matmul against torch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    min_block_size = 32
    if M < min_block_size or K_local < min_block_size or N < min_block_size:
        pytest.skip(f"Problem too small for min block size {min_block_size}")

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    device = f"cuda:{rank}"

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = (
        FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
        if M <= 256 or K_local <= 64 or N <= 128
        else FusedConfig()
    )

    assert M >= config.block_size_m
    assert K_local >= config.block_size_k
    assert N >= config.block_size_n

    ctx.ops.all_gather_matmul(output, A_sharded_shmem, B_shmem, config=config)

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol}"
    )


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    [
        (128, 32, 64),
        (256, 64, 128),
        (512, 64, 128),
    ],
)
@pytest.mark.parametrize(
    "staged_a_layout",
    [
        "k_contiguous",
        "m_contiguous",
    ],
)
def test_all_gather_matmul_hbm_buffer(dtype, atol, rtol, M, K_local, N, staged_a_layout):
    """Test all_gather_matmul_hbm_buffer against torch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)

    # k_per_flag must divide num_k_blocks = K // block_size_k; use 1 for small shapes
    num_k_blocks = K // config.block_size_k
    k_per_flag = 1
    while k_per_flag * 2 <= 8 and num_k_blocks % (k_per_flag * 2) == 0:
        k_per_flag *= 2

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx, A_sharded_shmem, B_shmem, config=config, staged_a_layout=staged_a_layout, k_per_flag=k_per_flag
    )

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=workspace,
        k_per_flag=k_per_flag,
        staged_a_layout=staged_a_layout,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} "
        f"(staged_a_layout={staged_a_layout}, M={M}, K_local={K_local}, N={N})"
    )


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    # num_m_tiles = M // block_size_m (64) must be divisible by world_size for the
    # co-located layout. These shapes work for world_size in {2,4,8}.
    "M,K_local,N",
    [
        (512, 64, 128),
        (1024, 128, 256),
    ],
)
@pytest.mark.parametrize(
    "staged_a_layout",
    [
        "k_contiguous",
        "m_contiguous",
    ],
)
def test_all_gather_matmul_layout(dtype, atol, rtol, M, K_local, N, staged_a_layout):
    """Hierarchical layout-driven kernel vs torch all_gather + matmul reference.

    The only comparison is functional correctness against torch; there is no
    comparison to the legacy hbm_buffer kernel.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    # num_xcds set to world_size so num_m_tiles is divisible by it for these shapes.
    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32, num_xcds=world_size)

    # layout=None -> default_layout derives a valid co-located hierarchical layout.
    all_gather_matmul_layout(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        staged_a_layout=staged_a_layout,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} "
        f"(staged_a_layout={staged_a_layout}, M={M}, K_local={K_local}, N={N})"
    )


@pytest.mark.parametrize("credit_window", [1, 2, 4])
def test_all_gather_matmul_layout_credit_window(credit_window):
    """TCP-style credit window (bounded producer run-ahead) must stay correct and
    NOT deadlock. Co-located layout with small co-resident pools (n_fetch_wg +
    n_gemm_wg <= cus_per_xcd) so the fetcher's throttle-wait on consumer credit can
    always make progress. Compared to the torch all_gather + matmul reference."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype, atol, rtol = torch.float16, 1e-2, 1e-2
    M, K_local, N = 1024, 128, 256
    ctx = iris.iris(2**33)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    K = K_local * world_size
    num_xcds = world_size  # num_m_tiles (16) divisible by num_xcds

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32, num_xcds=num_xcds)
    # small pools -> fetcher + GEMM WGs co-reside per XCD (deadlock-safe throttle).
    all_gather_matmul_layout(
        ctx, output, A_shmem, B_shmem,
        config=config,
        n_fetch_wg=2, n_gemm_wg=8,
        credit_window=credit_window,
        cus_per_xcd=38,
    )
    torch.cuda.synchronize(); ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: credit_window={credit_window} max diff {max_diff} >= {atol}"
    )


def test_all_gather_matmul_layout_credit_window_rejects_spatial():
    """credit_window must be rejected (clean error, no hang) for spatial layouts where
    producer and consumer live on different XCDs."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    world_size = iris.iris(2**33).get_num_ranks()
    if world_size < 2:
        pytest.skip("spatial layout needs world_size >= 2")

    dtype = torch.float16
    M, K_local, N = 1024, 128, 256
    ctx = iris.iris(2**33)
    rank = ctx.get_rank()
    K = K_local * world_size
    num_xcds = world_size

    A_sharded, B, _ = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    bm, bn, bk = 64, 64, 32
    config = FusedConfig(block_size_m=bm, block_size_n=bn, block_size_k=bk, num_xcds=num_xcds)
    layout, _ = make_layout(
        fetch_m=1, fetch_k=2, group_m=1, num_xcds=num_xcds,
        block_size_m=bm, block_size_n=bn, block_size_k=bk,
        M=M, N=N, K=K, K_local=K_local, world_size=world_size,
        order="mtile", fetch_xcds=1,
    )
    with pytest.raises(ValueError, match="co-located"):
        all_gather_matmul_layout(ctx, output, A_shmem, B_shmem, config=config,
                                 layout=layout, credit_window=2)


@pytest.mark.parametrize("order", ["kfg", "mtile"])
@pytest.mark.parametrize("fetch_xcds", [None, 1, 2])
def test_all_gather_matmul_layout_schedule_modes(order, fetch_xcds):
    """The new schedule knobs (FetcherLayout.order, XCDLayout.fetch_xcds) must each
    produce correct output vs the torch all_gather+mm reference. Co-located
    (fetch_xcds=None) and spatial role-segregated layouts are both checked."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype, atol, rtol = torch.float16, 1e-2, 1e-2
    M, K_local, N = 1024, 128, 256
    ctx = iris.iris(2**33)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    K = K_local * world_size
    num_xcds = world_size  # so num_m_tiles (16) is divisible by num_xcds

    if fetch_xcds is not None and not (1 <= fetch_xcds < num_xcds):
        pytest.skip(f"fetch_xcds={fetch_xcds} invalid for num_xcds={num_xcds}")

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    bm, bn, bk = 64, 64, 32
    config = FusedConfig(block_size_m=bm, block_size_n=bn, block_size_k=bk, num_xcds=num_xcds)
    layout, _ = make_layout(
        fetch_m=1, fetch_k=2, group_m=1, num_xcds=num_xcds,
        block_size_m=bm, block_size_n=bn, block_size_k=bk,
        M=M, N=N, K=K, K_local=K_local, world_size=world_size,
        order=order, fetch_xcds=fetch_xcds,
    )
    all_gather_matmul_layout(ctx, output, A_shmem, B_shmem, config=config, layout=layout)
    torch.cuda.synchronize(); ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: max_diff {max_diff} (order={order}, fetch_xcds={fetch_xcds})"
    )


@pytest.mark.parametrize("order", ["mtile", "coop"])
def test_all_gather_matmul_layout_local_interleave(order):
    """skip_local_stage + local_interleave: the wait-free local flag-groups are spread
    among the remote ones (new consumer ordering). Since matmul accumulation is
    order-independent, output must still match torch all_gather+mm for both mtile and
    coop fetch orders."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    dtype, atol, rtol = torch.float16, 1e-2, 1e-2
    M, K_local, N = 1024, 128, 256
    ctx = iris.iris(2**33)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    K = K_local * world_size
    num_xcds = world_size

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    bm, bn, bk = 64, 64, 32
    # fetch_k=2 with K_local/bk = 128/32 = 4 local k-blocks -> FLAGS_PER_RANK=2 local
    # flag-groups to interleave among the remote ones (num_k_blocks_local % fetch_k==0).
    config = FusedConfig(block_size_m=bm, block_size_n=bn, block_size_k=bk, num_xcds=num_xcds)
    # co-located (fetch_xcds=None) for both orders; coop strides the remote-only flag
    # list, which is exactly the skip_local regime (cf. the w4k_coloc coop configs).
    layout, _ = make_layout(
        fetch_m=1, fetch_k=2, group_m=1, num_xcds=num_xcds,
        block_size_m=bm, block_size_n=bn, block_size_k=bk,
        M=M, N=N, K=K, K_local=K_local, world_size=world_size,
        order=order, fetch_xcds=None,
    )
    all_gather_matmul_layout(ctx, output, A_shmem, B_shmem, config=config, layout=layout,
                             skip_local_stage=True, local_interleave=True)
    torch.cuda.synchronize(); ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: local_interleave max_diff {max_diff} (order={order})"
    )


# Exercise the in-kernel pid->tile DECODE arithmetic across every branch that the
# basic schedule-modes test above does not: non-full WG pools (strided loops),
# fetch_m>1 (the m-footprint multiply), local_first order, and both role modes.
# Output==ref only stays correct if the kernel decode covers each tile EXACTLY
# once, so any drift between the kernel decode (all_gather_matmul_layout.py) and the
# host ScheduleLayout.decode surfaces here as a wrong result. make_layout.validate()
# already checks host-side exactly-once coverage; this asserts the KERNEL agrees.
@pytest.mark.parametrize(
    "order,fetch_xcds,fetch_m,n_fetch_wg,n_gemm_wg",
    [
        # co-located, small pools (strided fetch + gemm loops)
        ("mtile", None, 1, 4, 16),
        ("kfg",   None, 1, 8, 8),
        # co-located, fetch_m>1 (footprint walks 2 m-tiles per cell)
        ("mtile", None, 2, 4, 16),
        # local_first order (rank-rotated kfg dispatch)
        ("local_first", None, 1, 8, 16),
        # spatial, small per-role pools (strided over the GLOBAL space)
        ("mtile", 2, 1, 8, 16),
        ("mtile", 1, 1, 16, 32),
        # spatial + fetch_m>1
        ("mtile", 2, 2, 8, 16),
    ],
)
def test_all_gather_matmul_layout_decode_coverage(order, fetch_xcds, fetch_m, n_fetch_wg, n_gemm_wg):
    """Kernel decode must match host ScheduleLayout for pools, fetch_m, and order
    variants — a wrong tile->WG mapping shows up as output != torch reference."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype, atol, rtol = torch.float16, 1e-2, 1e-2
    M, K_local, N = 1024, 128, 256
    ctx = iris.iris(2**33)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    K = K_local * world_size
    num_xcds = world_size  # num_m_tiles (16) divisible by num_xcds

    if fetch_xcds is not None and not (1 <= fetch_xcds < num_xcds):
        pytest.skip(f"fetch_xcds={fetch_xcds} invalid for num_xcds={num_xcds}")

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    bm, bn, bk = 64, 64, 32
    config = FusedConfig(block_size_m=bm, block_size_n=bn, block_size_k=bk, num_xcds=num_xcds)
    try:
        layout, _ = make_layout(
            fetch_m=fetch_m, fetch_k=2, group_m=1, num_xcds=num_xcds,
            block_size_m=bm, block_size_n=bn, block_size_k=bk,
            M=M, N=N, K=K, K_local=K_local, world_size=world_size,
            order=order, fetch_xcds=fetch_xcds,
            n_fetch_wg=n_fetch_wg, n_gemm_wg=n_gemm_wg,
        )
    except AssertionError as e:
        pytest.skip(f"invalid layout for this shape: {e}")

    all_gather_matmul_layout(ctx, output, A_shmem, B_shmem, config=config, layout=layout)
    torch.cuda.synchronize(); ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: max_diff {max_diff} "
        f"(order={order}, fetch_xcds={fetch_xcds}, fetch_m={fetch_m}, "
        f"nfw={n_fetch_wg}, ngw={n_gemm_wg})"
    )


# skip_local_stage: cur_rank's own shard is never staged; the GEMM reads it from
# A_sharded and computes it local-first (no flag wait). Output must still equal the
# torch reference. Also covers the auto-disable path (fetch_k straddling a rank).
@pytest.mark.parametrize(
    "order,fetch_xcds,fetch_k",
    [
        ("mtile", None, 2),   # co-located, flag-group within rank
        ("mtile", 2, 2),      # spatial fetch-only XCDs
        ("kfg", None, 2),     # kfg order
        ("mtile", None, 4),   # fetch_k == num_k_blocks_local -> 1 flag-group/rank
    ],
)
def test_all_gather_matmul_layout_skip_local_stage(order, fetch_xcds, fetch_k):
    """skip_local_stage must produce output == torch all_gather+mm reference."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype, atol, rtol = torch.float16, 1e-2, 1e-2
    M, K_local, N = 1024, 128, 256
    ctx = iris.iris(2**33)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    K = K_local * world_size
    num_xcds = world_size

    if fetch_xcds is not None and not (1 <= fetch_xcds < num_xcds):
        pytest.skip(f"fetch_xcds={fetch_xcds} invalid for num_xcds={num_xcds}")

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    bm, bn, bk = 64, 64, 32
    config = FusedConfig(block_size_m=bm, block_size_n=bn, block_size_k=bk, num_xcds=num_xcds)
    try:
        layout, _ = make_layout(
            fetch_m=1, fetch_k=fetch_k, group_m=1, num_xcds=num_xcds,
            block_size_m=bm, block_size_n=bn, block_size_k=bk,
            M=M, N=N, K=K, K_local=K_local, world_size=world_size,
            order=order, fetch_xcds=fetch_xcds,
        )
    except AssertionError as e:
        pytest.skip(f"invalid layout for this shape: {e}")

    all_gather_matmul_layout(
        ctx, output, A_shmem, B_shmem, config=config, layout=layout,
        skip_local_stage=True,
    )
    torch.cuda.synchronize(); ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: max_diff {max_diff} "
        f"(order={order}, fetch_xcds={fetch_xcds}, fetch_k={fetch_k})"
    )


# ──────────────────────────────────────────────────────────────────────
# Dynamic work-stealing GEMM (work_steal=True)
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "fetch_xcds,n_fetch_wg,n_gemm_wg,skip_local",
    [
        (None, 4, 16, False),  # co-located, small pools (owners + thieves)
        (None, 8, 8, False),   # co-located, larger fetch pool
        (None, 4, 16, True),   # co-located + skip_local_stage (local-first compute)
        (2, 8, 16, False),     # spatial: drained fetch XCDs steal cross-XCD
        (2, 8, 16, True),      # spatial + skip_local: cross-XCD thief on local-first path
    ],
)
def test_all_gather_matmul_layout_work_steal(fetch_xcds, n_fetch_wg, n_gemm_wg, skip_local):
    """Unified dynamic work-stealing GEMM must produce output == torch all_gather+mm.
    ALL GEMM WGs and drained fetch WGs pull tile chunks from one atomic counter over the
    full tile space (per-XCD co-located / global spatial); exactly-once by the monotonic
    counter (disjoint chunks). Covers co-located + spatial, small pools, and skip_local
    (incl. spatial+skip_local: cross-XCD drainer on the local-first path)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype, atol, rtol = torch.float16, 1e-2, 1e-2
    M, K_local, N = 1024, 128, 256
    ctx = iris.iris(2**33)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    K = K_local * world_size
    num_xcds = world_size  # num_m_tiles (16) divisible by num_xcds

    if fetch_xcds is not None and not (1 <= fetch_xcds < num_xcds):
        pytest.skip(f"fetch_xcds={fetch_xcds} invalid for num_xcds={num_xcds}")

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    bm, bn, bk = 64, 64, 32
    config = FusedConfig(block_size_m=bm, block_size_n=bn, block_size_k=bk, num_xcds=num_xcds)
    try:
        layout, _ = make_layout(
            fetch_m=1, fetch_k=2, group_m=1, num_xcds=num_xcds,
            block_size_m=bm, block_size_n=bn, block_size_k=bk,
            M=M, N=N, K=K, K_local=K_local, world_size=world_size,
            order="mtile", fetch_xcds=fetch_xcds,
            n_fetch_wg=n_fetch_wg, n_gemm_wg=n_gemm_wg,
        )
    except AssertionError as e:
        pytest.skip(f"invalid layout for this shape: {e}")

    all_gather_matmul_layout(
        ctx, output, A_shmem, B_shmem, config=config, layout=layout,
        work_steal=True, skip_local_stage=skip_local,
    )
    torch.cuda.synchronize(); ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: work_steal max_diff {max_diff} "
        f"(fetch_xcds={fetch_xcds}, nfw={n_fetch_wg}, ngw={n_gemm_wg}, skip_local={skip_local})"
    )


def test_all_gather_matmul_layout_work_steal_default_off_matches():
    """work_steal=False must give the SAME output as the baseline call with no
    work_steal kwarg (default-off invariance at the API level)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype, atol = torch.float16, 1e-2
    M, K_local, N = 1024, 128, 256
    ctx = iris.iris(2**33)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    K = K_local * world_size
    num_xcds = world_size

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    out_base = ctx.zeros((M, N), dtype=dtype)
    out_off = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32, num_xcds=num_xcds)
    all_gather_matmul_layout(ctx, out_base, A_shmem, B_shmem, config=config)
    all_gather_matmul_layout(ctx, out_off, A_shmem, B_shmem, config=config, work_steal=False)
    torch.cuda.synchronize(); ctx.barrier()

    assert torch.equal(out_base, out_off), (
        f"Rank {rank}: work_steal=False output differs from the no-kwarg baseline"
    )
    assert torch.allclose(out_off, ref_output, atol=atol, rtol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_all_gather_matmul_layout_work_steal_gate_shape(dtype):
    """Gate projection shape 4096x11008x4096 at ws4 with dynamic work-stealing on
    a co-located schedule. This is the compute-bound regime work_steal targets."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    ctx = iris.iris(2**34)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    if world_size != 4:
        pytest.skip("gate-shape work_steal test targets world_size == 4 (ws4)")

    atol, rtol = 1e-2, 1e-2
    M, N, K = 4096, 11008, 4096
    K_local = K // world_size  # 1024

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    # co-located default layout; small GEMM pool so the native owner ranges + steal
    # protocol are exercised (rather than one WG per tile).
    config = FusedConfig(block_size_m=128, block_size_n=256, block_size_k=64, num_xcds=world_size)
    all_gather_matmul_layout(
        ctx, output, A_shmem, B_shmem, config=config,
        n_fetch_wg=8, n_gemm_wg=32, work_steal=True,
    )
    torch.cuda.synchronize(); ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: gate-shape work_steal max_diff {max_diff}"
    )


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    [
        (128, 32, 64),
    ],
)
def test_all_gather_matmul_hbm_buffer_with_bias(dtype, atol, rtol, M, K_local, N):
    """Test all_gather_matmul_hbm_buffer with a bias vector."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    A_sharded, B, ref_output_no_bias = _make_reference(rank, world_size, M, K_local, N, dtype)
    device = f"cuda:{rank}"

    torch.manual_seed(77)
    bias = torch.randn(M, dtype=dtype, device=device)
    ref_output = ref_output_no_bias + bias[:, None]

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    bias_shmem = ctx.zeros((M,), dtype=dtype)
    bias_shmem.copy_(bias)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)

    # k_per_flag must divide num_k_blocks = K // block_size_k; use 1 for small shapes
    num_k_blocks = K // config.block_size_k
    k_per_flag = 1
    while k_per_flag * 2 <= 8 and num_k_blocks % (k_per_flag * 2) == 0:
        k_per_flag *= 2

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        bias=bias_shmem,
        config=config,
        k_per_flag=k_per_flag,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} (with bias)"
    )


def test_all_gather_matmul_hbm_buffer_auto_workspace():
    """Test all_gather_matmul_hbm_buffer with workspace=None (auto preamble)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, K_local, N = 128, 32, 64
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2

    K = K_local * world_size
    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    k_per_flag = 1

    # workspace=None triggers automatic preamble inside the kernel function
    ws = all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=None,
        k_per_flag=k_per_flag,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    assert ws is not None, "all_gather_matmul_hbm_buffer should return workspace"
    assert ws.aux_buffer is not None, "Workspace aux_buffer should be allocated"
    assert ws.locks is not None, "Workspace locks should be allocated"

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} (auto workspace)"
    )


def test_all_gather_matmul_hbm_buffer_workspace_reuse():
    """Test that workspace can be reused across multiple kernel calls."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, K_local, N = 128, 32, 64
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2

    K = K_local * world_size
    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output1 = ctx.zeros((M, N), dtype=dtype)
    output2 = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    k_per_flag = 1

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx, A_sharded_shmem, B_shmem, config=config, k_per_flag=k_per_flag
    )

    # First call
    all_gather_matmul_hbm_buffer(
        ctx, output1, A_sharded_shmem, B_shmem, config=config, workspace=workspace, k_per_flag=k_per_flag, trace=False
    )
    torch.cuda.synchronize()
    ctx.barrier()

    # Second call reusing workspace
    all_gather_matmul_hbm_buffer(
        ctx, output2, A_sharded_shmem, B_shmem, config=config, workspace=workspace, k_per_flag=k_per_flag, trace=False
    )
    torch.cuda.synchronize()
    ctx.barrier()

    max_diff1 = (output1 - ref_output).abs().max().item()
    max_diff2 = (output2 - ref_output).abs().max().item()
    assert torch.allclose(output1, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: First call max diff {max_diff1}, expected < {atol}"
    )
    assert torch.allclose(output2, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Second call (workspace reuse) max diff {max_diff2}, expected < {atol}"
    )
    assert torch.allclose(output1, output2), "Both calls should produce identical results"


def test_all_gather_matmul_hbm_buffer_trace():
    """Test that trace_data is None when trace=False (default)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, K_local, N = 128, 32, 64
    dtype = torch.float16

    K = K_local * world_size
    A_sharded, B, _ = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    k_per_flag = 1

    ws = all_gather_matmul_hbm_buffer_preamble(ctx, A_sharded_shmem, B_shmem, config=config, k_per_flag=k_per_flag)

    # With trace=False, trace_data should not be populated
    ws = all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=ws,
        k_per_flag=k_per_flag,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    assert not hasattr(ws, "trace_data") or ws.trace_data is None, (
        # FusedWorkspace is a dataclass; trace_data is set only when trace=True.
        # Both conditions handle the case where the attribute is absent (fresh workspace)
        # or explicitly set to None (workspace reused from a previous trace=False call).
        "trace_data should not be populated when trace=False"
    )


# ──────────────────────────────────────────────────────────────────────
# Unit tests for _auto_config (no distributed context required)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "M, N, K, world_size",
    [
        (1024, 256, 1024, 8),
        (4096, 3584, 8192, 8),
        (8192, 8192, 16384, 8),
        (16384, 3584, 8192, 4),
        (256, 256, 512, 2),
    ],
)
def test_auto_config_heuristic_validity(M, N, K, world_size):
    """Verify _auto_config returns valid configs where k_per_flag divides K//block_k."""
    config, kpf, fs, nfs, fsf = _auto_config(M, N, K, world_size)

    assert config.block_size_m > 0
    assert config.block_size_n > 0
    assert config.block_size_k > 0

    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % kpf == 0, (
        f"k_per_flag={kpf} does not divide num_k_blocks={num_k_blocks} for M={M},N={N},K={K}"
    )
    assert fs > 0, "num_fetch_sms must be positive"
    assert nfs > 0, "num_fetch_stages must be positive"
    assert fsf > 0, "first_stage_fetch_sms must be positive"


def test_auto_config_champion_shapes():
    """Verify that champion shapes are returned directly from _CHAMPION_CONFIGS."""
    for key in _CHAMPION_CONFIGS:
        M, N, K = key
        config, kpf, fs, nfs, fsf = _auto_config(M, N, K, world_size=8)
        c = _CHAMPION_CONFIGS[key]

        assert config.block_size_m == c["bm"]
        assert config.block_size_n == c["bn"]
        assert config.block_size_k == c["bk"]
        assert config.group_size_m == c["gm"]

        # kpf may be adjusted down by _auto_config when champion["kpf"] doesn't divide
        # num_k_blocks (e.g. different world_size changes K and therefore num_k_blocks).
        num_k_blocks = K // c["bk"]
        assert num_k_blocks % kpf == 0, f"Champion kpf={kpf} does not divide num_k_blocks={num_k_blocks} for {key}"


def test_auto_config_large_m_uses_block_256():
    """Verify _auto_config picks block_m=256 for large M (M >= 8192, M divisible by 256)."""
    config, *_ = _auto_config(8192, 3584, 8192, world_size=8)
    assert config.block_size_m == 256, f"Expected block_m=256 for large M, got {config.block_size_m}"


def test_auto_config_small_m_uses_block_128():
    """Verify _auto_config picks block_m=128 for small M (M < 8192)."""
    config, *_ = _auto_config(1024, 3584, 8192, world_size=8)
    assert config.block_size_m == 128, f"Expected block_m=128 for small M, got {config.block_size_m}"


def test_auto_config_block_n_always_256():
    """Verify _auto_config always selects block_n=256 (from sweep data)."""
    for M in [1024, 4096, 16384]:
        config, *_ = _auto_config(M, 3584, 8192, world_size=8)
        assert config.block_size_n == 256, f"Expected block_n=256 for M={M}, got {config.block_size_n}"


def test_auto_config_block_k_always_64():
    """Verify _auto_config always selects block_k=64 (exceeding LDS on MI300X with 128)."""
    for M in [1024, 4096, 16384]:
        config, *_ = _auto_config(M, 3584, 8192, world_size=8)
        assert config.block_size_k == 64, f"Expected block_k=64 for M={M}, got {config.block_size_k}"


# ──────────────────────────────────────────────────────────────────────
# Fully-fused 8x8 A-stationary schedule (all_gather_matmul_fused_a8x8)
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "M,K_local,N,bm,bn,cols",
    [
        # 2048x2048x2048 ws4: bm=128, bn=256, cols=8  (K_local=512)
        (2048, 512, 2048, 128, 256, 8),
        # 1024x1024x1024 ws4: bm=64, bn=128, cols=8  (K_local=256)
        (1024, 256, 1024, 64, 128, 8),
    ],
)
def test_all_gather_matmul_fused_a8x8(M, K_local, N, bm, bn, cols):
    """8x8 A-stationary fully-fused kernel vs torch all_gather -> cat -> mm."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    if not torch.cuda.is_available():
        pytest.skip("no GPU available")

    dtype, atol, rtol = torch.float16, 1e-2, 1e-2
    ctx = iris.iris(2**33)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    if world_size != 4:
        pytest.skip("fused a8x8 test targets world_size == 4 (ws4)")

    K = K_local * world_size
    assert N % (bn * cols) == 0, f"bn*cols ({bn * cols}) must tile N ({N})"

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    A_shmem = ctx.zeros((M, K_local), dtype=dtype); A_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype); B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)
    ctx.barrier()

    config = FusedConfig(block_size_m=bm, block_size_n=bn, block_size_k=64, num_xcds=8)
    all_gather_matmul_fused_a8x8(
        ctx, output, A_shmem, B_shmem, config=config, cols_per_xcd=cols, async_op=False
    )
    torch.cuda.synchronize(); ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: fused a8x8 max_diff {max_diff} "
        f"(M={M}, K_local={K_local}, N={N}, bm={bm}, bn={bn}, cols={cols})"
    )


if __name__ == "__main__":
    import sys

    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=2 tests/ops/test_all_gather_matmul.py")
        sys.exit(1)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Tests in this file require pytest + torchrun. See tests/run_tests_distributed.py")
