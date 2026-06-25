#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""RCCL all-gather bandwidth bench, swept by world size and message size.

This is the *inner* bench of the NCCL_MAX_NCHANNELS experiment. It is run once
per ``NCCL_MAX_NCHANNELS`` value by ``sweep_nchannels.py`` (the env var is read
by RCCL at communicator creation, so it cannot be a bench axis -- it must be
fixed for the whole process). Each run sweeps num_ranks and message size and
reports achieved bus bandwidth.

Bus bandwidth for all-gather: (W-1)/W of the gathered data, matching the
formula in bench_all_gather.py.
"""

import os
import torch
import torch.distributed as dist
import iris.bench as bench

# Per-rank input size in MiB. Override with RCCL_SWEEP_SIZES_MB=1,16 for smoke runs.
SIZES_MB = [int(s) for s in os.environ.get("RCCL_SWEEP_SIZES_MB", "1,4,16,64,256").split(",")]


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("size_mb", SIZES_MB)
@bench.axis("dtype", [torch.bfloat16])
def rccl_all_gather(state, ctx):
    size_mb, dtype = state["size_mb"], state["dtype"]
    world_size = ctx.get_num_ranks()
    rank = ctx.get_rank()

    elt = torch.tensor([], dtype=dtype).element_size()
    numel = (size_mb * (1 << 20)) // elt

    inp = torch.full((numel,), float(rank + 1), dtype=dtype, device="cuda")
    out_list = [torch.empty_like(inp) for _ in range(world_size)]

    # Bus bandwidth: each rank ultimately moves (W-1)/W of the total gathered data.
    state.set_bytes(int((world_size - 1) * numel * elt))

    state.exec(lambda: dist.all_gather(out_list, inp))


if __name__ == "__main__":
    bench.main()
