# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Workspace management for fused GEMM+CCL operations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class FusedWorkspace:
    """
    Workspace for fused GEMM+CCL operations.

    This class holds temporary buffers and metadata needed for fused operations.
    Buffers are allocated lazily and reused across calls when shapes match.

    Attributes:
        operation: Operation type ("matmul_all_reduce", "all_gather_matmul", etc.)
        shape: Problem dimensions (M, N, K)
        dtype: Data type of tensors
        world_size: Number of ranks in the communicator
        variant: Algorithm variant (for operations that support multiple variants)

        # Temporary buffers
        aux_buffer: Generic auxiliary buffer for intermediate results (gathered data, temp results, etc.)
        locks: Lock array for spinlock synchronization

        prepared: Whether workspace has been initialized for current operation
    """

    operation: str = ""
    shape: Tuple[int, int, int] = ()  # (M, N, K)
    dtype: Optional[torch.dtype] = None
    world_size: int = 1
    variant: str = ""

    # Temporary buffers (allocated as needed)
    aux_buffer: Optional[torch.Tensor] = None  # Generic buffer for intermediate results
    locks: Optional[torch.Tensor] = None  # Synchronization primitives

    # Credit-window flow control (opt-in; only used when credit_window > 0). Per-XCD
    # int32 counters bounding how far the fetcher may run ahead of the GEMM consumer,
    # plus a per-flag "first consume seen" marker. See all_gather_matmul_layout.
    credit_produced: Optional[torch.Tensor] = None  # int32[num_xcds]: staged tiles/XCD
    credit_consumed: Optional[torch.Tensor] = None  # int32[num_xcds]: first-consumed tiles/XCD
    first_seen: Optional[torch.Tensor] = None        # int32[num_flags]: 0/1 first-consume marker

    # Reserved-tail work-stealing GEMM (opt-in; only used when work_steal=True). See
    # all_gather_matmul_layout WORK_STEAL. A single per-XCD int32 counter that hands
    # out the reserved-tail tiles to drained fetchers/finished GEMM WGs; zeroed each
    # launch. co-located uses per-XCD buckets [xcd]; spatial uses only [0].
    steal_next: Optional[torch.Tensor] = None   # int32[NUM_XCDS]: reserved-tail drain counter/XCD

    prepared: bool = False

    def matches(
        self,
        operation: str,
        shape: Tuple[int, int, int],
        dtype: torch.dtype,
        world_size: int,
        variant: str = "",
    ) -> bool:
        """
        Check if workspace can be reused for the given parameters.

        Args:
            operation: Operation type
            shape: Problem dimensions (M, N, K)
            dtype: Data type
            world_size: Number of ranks
            variant: Algorithm variant

        Returns:
            True if workspace matches and can be reused
        """
        return (
            self.operation == operation
            and self.shape == shape
            and self.dtype == dtype
            and self.world_size == world_size
            and self.variant == variant
            and self.prepared
        )

    def reset(self):
        """Mark workspace as unprepared (buffers will be re-initialized next time)."""
        self.prepared = False

    def clear(self):
        """Free all allocated buffers."""
        self.aux_buffer = None
        self.locks = None
        self.credit_produced = None
        self.credit_consumed = None
        self.first_seen = None
        self.steal_next = None
        self.prepared = False
