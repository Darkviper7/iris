#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Per-rank kernel driver for the L2 (TCC) cache probe (run under run.sh).

Builds the 4096^3 layout for CACHE_MODE in {spatial, coloc}, warms, then runs ONE
launch. The in-process rocprofiler-sdk tool (tcc_tool.so, loaded via the env set by
run.sh) counts the layout kernel's TCC_HIT or TCC_MISS (one counter per invocation;
they don't fit a single hardware pass) and writes a per-rank CSV.

Launched as 4 manual ranks (no torchrun) so the SDK can configure cleanly.
Env (set by run.sh): RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR/PORT, CACHE_MODE,
plus ROCPROFILER_LIBRARY_CTOR / LD_PRELOAD / ROCP_TOOL_LIBRARIES / TCC_* for the tool.
"""

import os
import torch
import torch.distributed as dist
import iris
from iris.ops.config import FusedConfig
from iris.ops.all_gather_matmul_layout import (
    all_gather_matmul_layout,
    all_gather_matmul_layout_preamble,
)
from iris.ops.schedule_layout import make_layout

M = N = K = 4096
NUM_XCDS = 8


def main():
    dist.init_process_group("nccl")
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    ctx = iris.iris(1 << 34)
    rank = ctx.get_rank()
    ws = ctx.get_num_ranks()
    Kl = K // ws
    dt = torch.float16

    A = ctx.randn((M, Kl), dtype=dt, generator=torch.Generator("cuda").manual_seed(42 + rank))
    B = torch.randn((K, N), device="cuda", dtype=dt, generator=torch.Generator("cuda").manual_seed(123))
    C = ctx.zeros((M, N), dtype=dt)
    cfg = FusedConfig(block_size_m=256, block_size_n=256, block_size_k=64, num_xcds=NUM_XCDS)
    cws = all_gather_matmul_layout_preamble(ctx, A, B, config=cfg, fetch_k=16)

    mode = os.environ["CACHE_MODE"]
    fx, nfw, ngw = (2, 8, 32) if mode == "spatial" else (None, 8, 16)
    layout, _ = make_layout(
        num_xcds=NUM_XCDS, M=M, N=N, K=K, K_local=Kl, world_size=ws,
        fetch_m=1, fetch_k=16, group_m=1,
        block_size_m=256, block_size_n=256, block_size_k=64,
        order="mtile", fetch_xcds=fx, n_fetch_wg=nfw, n_gemm_wg=ngw,
    )

    # warm/compile WITHOUT the counter (counter collection serializes; warm first)
    for _ in range(3):
        all_gather_matmul_layout(ctx, C, A, B, config=cfg, workspace=cws, layout=layout)
    torch.cuda.synchronize(); ctx.barrier()
    # the ONE measured launch (tool counts it)
    all_gather_matmul_layout(ctx, C, A, B, config=cfg, workspace=cws, layout=layout)
    torch.cuda.synchronize(); ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
