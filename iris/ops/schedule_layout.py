# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Hierarchical scheduling-layout descriptor for fused all-gather + GEMM.

This module is **pure Python** (no torch, no triton, no GPU). It defines a
hierarchical layout that maps a workgroup id (``pid``) to the input-tile
coordinate it touches:

    producers (FETCH):  pid -> (m_tile, k_flag_group)  staged into the HBM buffer
    consumers (GEMM):   pid -> (m_tile, n_tile)        computed from staged A @ B

The layout is the single source of truth for the in-kernel ``pid`` decode. The
Triton kernel mirrors :meth:`ScheduleLayout.decode` verbatim using exactly the
flat integers returned by :meth:`ScheduleLayout.constexprs`. This file is the
host-side reference and is fully testable without a GPU.

Layout model (innermost -> outermost):

    FetcherLayout(m, k)               (m x k) flag-group footprint ONE fetcher WG
                                      walks temporally inside its CU
    CULayout(spatial, temporal, fch)  CUs within an XCD: ``spatial`` concurrent,
                                      ``temporal`` waves (intended placement)
    XCDLayout(x, y, cu)               partition over (M_tiles, K_flag_groups).
                                      Co-located preset: x=num_xcds, y=1 (each XCD
                                      owns a full-K m-slab it both produces and
                                      consumes)
    ConsumerLayout(group_m)           GEMM (m_tile, n_tile) ordering (grouped)
    ScheduleLayout(xcd, num_xcds, consumer)   top level; owns both schedules

Hardware grounding (MI300X): AMD round-robins consecutive workgroup ids across
XCDs, so the only placement the kernel controls is ``xcd = pid % NUM_XCDS``.
Spatial-CU / temporal-wave coordinates are the *intended* placement (shaping pid
order, verified post-hoc from the trace); correctness never depends on them.

Co-location: each physical XCD produces AND consumes the same m-slab, so every
flag a GEMM waits on was written by a fetcher on the same XCD -> on-die L2
handshake. Flags are still global (scope="gpu"), so co-location is a perf
property, not a correctness requirement.

------------------------------------------------------------------------------
In-kernel decode contract (pseudocode the kernel author mirrors verbatim)
------------------------------------------------------------------------------
All operands below are ``tl.constexpr`` ints from :meth:`constexprs` so Triton
unrolls the nested divmod at compile time.

    # producer space = flag-group cells (m_tile, k_flag_group),
    #   total NUM_M_TILES * NUM_FLAG_GROUPS_K. FETCH_K is k-blocks-per-flag (the
    #   handoff grain): one flag covers FETCH_K consecutive k-blocks, and
    #   NUM_FLAG_GROUPS_K = num_k_blocks // FETCH_K. There is no k_per_flag.
    # co-located 1-D partition: each XCD owns SLAB_M m-tiles across ALL K.
    #   SLAB_M = NUM_M_TILES // NUM_XCDS
    #   RM = SLAB_M // FETCH_M ; FETCH_SLOTS_PER_XCD == RM * NUM_FLAG_GROUPS_K
    #   GEMM_SLOTS_PER_XCD  == SLAB_M * NUM_TILES_N
    # PERSISTENT pools: each XCD launches N_FETCH_WG fetch WGs + N_GEMM_WG gemm
    #   WGs; each WG STRIDED-LOOPS over its share of cells/tiles. Full pools
    #   (N_FETCH_WG==FETCH_SLOTS_PER_XCD, N_GEMM_WG==GEMM_SLOTS_PER_XCD) -> one WG
    #   per cell/tile (the original non-persistent behavior). Smaller pools ->
    #   fewer WGs co-resident on more CUs each -> dedicate CUs to a role.

    xcd  = pid % NUM_XCDS
    slot = pid // NUM_XCDS
    m0   = xcd * SLAB_M

    if slot < N_FETCH_WG:
        # ---------------- PRODUCER pool member (strided loop) --------------
        for cell in range(slot, FETCH_SLOTS_PER_XCD, N_FETCH_WG):
            fp_m = cell %  RM ; kfg = cell // RM   # 0 .. NUM_FLAG_GROUPS_K-1
            k_block_start = kfg * FETCH_K
            for fi_m in range(FETCH_M):            # footprint walks FETCH_M m-tiles
                m_tile = m0 + fp_m*FETCH_M + fi_m
                for k_off in range(FETCH_K):       # gather + .cg store body
                    ...
                flag_idx = m_tile*NUM_FLAG_GROUPS_K + kfg
                debug_barrier; atomic_xchg(flags+flag_idx, 1, release)
    else:
        # ---------------- CONSUMER pool member (strided loop) -------------
        gwg = slot - N_FETCH_WG
        for gtile in range(gwg, GEMM_SLOTS_PER_XCD, N_GEMM_WG):
            group_id = gtile // (GROUP_SIZE_M*NUM_TILES_N)
            within   = gtile %  (GROUP_SIZE_M*NUM_TILES_N)
            pid_m    = m0 + group_id*GROUP_SIZE_M + (within % GROUP_SIZE_M)
            pid_n    = within // GROUP_SIZE_M
            for k_fg in range(NUM_FLAG_GROUPS_K):
                wait flags[pid_m*NUM_FLAG_GROUPS_K + k_fg] (acquire); dot ...

    grid_size = NUM_XCDS * (N_FETCH_WG + N_GEMM_WG)
"""

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# Role tags ------------------------------------------------------------------
ROLE_FETCH = "FETCH"
ROLE_GEMM = "GEMM"

# The subset of constexprs() the Triton kernel decode actually consumes, in the
# exact order the kernel's "decode contract" signature block / launch expect.
# NUM_K_BLOCKS_LOCAL is also used by the kernel but is threaded as a separate
# arg (not part of this contract block). The remaining constexprs() keys
# (RK, NUM_M_TILES) are host-only.
KERNEL_CONSTEXPR_KEYS = (
    "NUM_XCDS",
    "SLAB_M",
    "RM",
    "FETCH_M",
    "FETCH_K",
    "N_FETCH_WG",
    "N_GEMM_WG",
    "FETCH_SLOTS_PER_XCD",
    "GEMM_SLOTS_PER_XCD",
    "GROUP_SIZE_M",
    "NUM_TILES_N",
    "NUM_FLAG_GROUPS_K",
    # ---- schedule knobs (added) ----
    "FETCH_ORDER",          # 0 = kfg-major, 1 = m-tile-major fetch traversal
    "FETCH_XCDS",           # 0 = co-located; >=1 = spatial fetch-only XCD count
    "NUM_M_TILES",          # M // block_size_m (global m-tile count)
    "TOTAL_FETCH_CELLS",    # global fetch cells (spatial mode): NUM_M_TILES*NFG_K
    "TOTAL_GEMM_TILES",     # global gemm tiles (spatial mode): NUM_M_TILES*NUM_TILES_N
    "FETCH_WAVE",           # DEMAND-order (order 5) wave width in m-tiles (else ignored)
)

# Fetch traversal orders (FetcherLayout.order)
FETCH_ORDER_KFG = "kfg"      # flag-group advances slowest (stage kfg 0 for all m first)
FETCH_ORDER_MTILE = "mtile"  # m-tile advances slowest (stage one m-tile's full K first)
FETCH_ORDER_LOCAL_FIRST = "local_first"  # mtile traversal, but each m-tile's LOCAL-rank
# flag-group (kfg == cur_rank, an on-device copy) is dispatched first, then the
# remote xGMI gathers. A per-m_tile rotation of kfg by cur_rank: kfg = (cell +
# cur_rank) % nfg_k. The flag INDEX still uses the true kfg, so the GEMM consumer
# contract is unchanged; only producer dispatch order shifts. Coverage is identical
# (a permutation of each m-tile's flag-groups).
FETCH_ORDER_PIPELINED = "pipelined"  # each WG owns a CONTIGUOUS block of cells (whole
# m-tiles: kfg 0,1,2,.. of one m-tile in sequence), flagging each kfg as it completes.
# The GEMM consumer waits kfg in order, so it can dot kfg0 while the producer still
# fetches kfg1 -> real per-tile fetch/compute overlap. Uses m-tile-major mapping with a
# contiguous (not strided) cell walk; the in-kernel FETCH_ORDER==3 branch handles both.
FETCH_ORDER_COOP = "coop"  # stride over the REMOTE-ONLY flag list (m-major, kfg-minor):
# adjacent fetchers cooperatively complete adjacent remote kfg of the SAME m-tile, so
# ~NUM_FETCHERS/REMOTE_KFG m-tiles finish per wave. Delivers whole m-tiles at the width
# the GEMM consumes them (fixes the 4-wide supply vs 12-wide demand stall at 4k).
FETCH_ORDER_WAVE = "wave"  # DEMAND-order: wave-blocked kfg-major over the remote flags.
# Within a block of FETCH_WAVE m-tiles (= the # of m-tiles the GEMM WG pool runs
# concurrently, total_gemm_wg // num_tiles_n), stage remote-kfg=0 (first-needed slice)
# for ALL wave m-tiles, then kfg=1, then kfg=2, then the next wave. This matches the
# GEMM's actual flag-demand order (all first-wave m-tiles want their first remote slice
# up front), eliminating the k_fg=1 stall wave that coop/mtile leave.
_FETCH_ORDER_CODE = {FETCH_ORDER_KFG: 0, FETCH_ORDER_MTILE: 1, FETCH_ORDER_LOCAL_FIRST: 2,
                     FETCH_ORDER_PIPELINED: 3, FETCH_ORDER_COOP: 4, FETCH_ORDER_WAVE: 5}


@dataclass
class Problem:
    """Derived tiling counts for a single (M, N, K) fused all-gather + GEMM.

    Carries everything the layout needs in tile units. Block sizes are kept so
    the layout can report them, but the decode works purely in tile counts.

    Note: flag (producer->consumer handoff) granularity is NOT a property of the
    Problem -- it is a property of the layout (``FetcherLayout.k`` k-blocks per
    flag). The Problem only knows the raw block counts.
    """

    M: int
    N: int
    K: int
    K_local: int
    block_size_m: int
    block_size_n: int
    block_size_k: int
    world_size: int

    # derived (filled in __post_init__)
    num_m_tiles: int = field(init=False)
    num_tiles_n: int = field(init=False)
    num_k_blocks: int = field(init=False)
    num_k_blocks_local: int = field(init=False)
    total_gemm_tiles: int = field(init=False)
    total_gather_tiles: int = field(init=False)

    def __post_init__(self):
        assert self.M % self.block_size_m == 0, "M must be divisible by block_size_m"
        assert self.K % self.block_size_k == 0, "K must be divisible by block_size_k"
        assert self.K_local % self.block_size_k == 0, "K_local must be divisible by block_size_k"
        assert self.world_size * self.K_local == self.K, "world_size * K_local must == K"

        self.num_m_tiles = self.M // self.block_size_m
        self.num_tiles_n = ceil_div(self.N, self.block_size_n)
        self.num_k_blocks = self.K // self.block_size_k
        self.num_k_blocks_local = self.K_local // self.block_size_k
        self.total_gemm_tiles = self.num_m_tiles * self.num_tiles_n
        self.total_gather_tiles = self.num_m_tiles * self.num_k_blocks

    def num_flags(self, fetch_k: int) -> int:
        """Size of the flag/lock workspace given the layout's per-flag k-chunk.

        flags = num_m_tiles * (num_k_blocks // fetch_k). Flag granularity is a
        layout property (``FetcherLayout.k``), not a Problem property.
        """
        assert self.num_k_blocks % fetch_k == 0, "fetch_k must divide num_k_blocks"
        return self.num_m_tiles * (self.num_k_blocks // fetch_k)


@dataclass
class TileAssignment:
    """What one workgroup (pid) does. Carries enough to derive dependency distance."""

    pid: int
    role: str  # ROLE_FETCH | ROLE_GEMM

    # input-tile coordinate
    m_tile: int = -1
    k_flag_group: int = -1       # flag-group cell (FETCH only)
    k_blocks: Tuple[int, ...] = ()  # all k-blocks this fetcher visit stages (FETCH)
    n_tile: int = -1             # GEMM only

    # hardware coordinate (xcd is real; cu_slot/wave are intended placement)
    xcd: int = -1
    cu_slot: int = -1            # intended spatial CU within the XCD
    wave: int = -1               # intended temporal wave

    # producer -> consumer dependency
    flag_index: int = -1                 # flag a fetcher writes (FETCH)
    wait_flags: Tuple[int, ...] = ()     # flags a GEMM waits on (GEMM)

    def __repr__(self) -> str:
        if self.role == ROLE_FETCH:
            return (
                f"TileAssignment(pid={self.pid}, FETCH, m_tile={self.m_tile}, "
                f"kfg={self.k_flag_group}, k_blocks={self.k_blocks}, "
                f"flag={self.flag_index}, xcd={self.xcd}, cu={self.cu_slot}, wave={self.wave})"
            )
        return (
            f"TileAssignment(pid={self.pid}, GEMM, m_tile={self.m_tile}, "
            f"n_tile={self.n_tile}, waits={self.wait_flags}, xcd={self.xcd})"
        )


# ---------------------------------------------------------------------------
# Nested layout levels
# ---------------------------------------------------------------------------
@dataclass
class FetcherLayout:
    """The iteration footprint ONE fetcher WG walks temporally inside its CU.

    ``m`` = number of m-tiles the WG walks. ``k`` = number of ``BLOCK_SIZE_K``
    k-blocks staged per flag (the producer->consumer handoff granularity). One
    flag is raised per (m_tile, k-chunk-of-k-blocks); the total number of flag
    groups along K is ``num_k_blocks // k``. There is no separate ``k_per_flag``
    -- ``k`` IS it, expressed in the layout.

    ``order`` is the traversal a fetcher walks over its (m_tile, k_flag_group)
    cells. ``"kfg"`` (flag-group advances slowest) stages flag-group 0 for ALL
    m-tiles before flag-group 1 -> a GEMM consumer stalls waiting for its later
    flag-groups. ``"mtile"`` (m-tile advances slowest) stages one m-tile's FULL
    K (all flag-groups) before the next m-tile, so that m-tile becomes ready as a
    unit and a GEMM WG runs through it with no mid-k stall. Coverage is identical
    for both orders; only producer->consumer timing differs."""

    m: int
    k: int
    order: str = FETCH_ORDER_KFG


@dataclass
class CULayout:
    """Within one XCD: ``spatial`` = the persistent fetcher WG-pool size (how many
    fetch workgroups run concurrently, ideally ~one per CU), ``temporal`` = how
    many fetch cells each WG sequentially walks (the strided-loop trip count, =
    ceil(fetch_slots / spatial)). spatial*ceil gives full coverage. With
    spatial == fetch_slots, temporal == 1 -> one WG per cell (today's behavior).
    ``fetcher`` is the per-cell (m x k) iteration footprint."""

    spatial: int
    temporal: int
    fetcher: FetcherLayout


@dataclass
class XCDLayout:
    """Partition over the global (M_tiles, K_flag_groups) producer space.

    Two role-assignment modes:

    - ``fetch_xcds is None`` -> **co-located** preset: x = num_xcds, y = 1, each
      XCD owns a full-K m-slab it BOTH produces and consumes. Every flag a GEMM
      waits on was written by a fetcher on the same XCD (on-die L2 handshake).

    - ``fetch_xcds`` in [1, num_xcds-1] -> **spatial** mode: XCDs
      ``[0, fetch_xcds)`` are fetch-only (producers) and partition ALL fetch
      cells (num_m_tiles * nfg_k) across ``fetch_xcds * spatial`` WGs; XCDs
      ``[fetch_xcds, num_xcds)`` are GEMM-only (consumers) and partition ALL
      output tiles (num_m_tiles * num_tiles_n) across ``(num_xcds-fetch_xcds) *
      n_gemm_wg`` WGs. Flags stay global, so a GEMM on one XCD waits on a flag
      written by a fetcher on another XCD (cross-XCD flag traffic) in exchange
      for giving GEMM uncontended CUs."""

    x: int
    y: int
    cu: CULayout
    fetch_xcds: Optional[int] = None

    @property
    def spatial_mode(self) -> bool:
        return self.fetch_xcds is not None


@dataclass
class ConsumerLayout:
    """GEMM (m_tile, n_tile) ordering, grouped by ``group_m`` for L2 locality.

    ``n_gemm_wg`` is the persistent consumer WG-pool size per XCD: that many GEMM
    workgroups each loop (strided) over the XCD's output tiles. None = full pool
    (one WG per tile = today's behavior)."""

    group_m: int
    n_gemm_wg: Optional[int] = None


@dataclass
class ScheduleLayout:
    """Top-level schedule. Owns both the producer (fetch) and consumer (GEMM)
    schedules so per-input-tile dependency distance is a property of one object.

    The decode is driven by the hierarchy: ``xcd`` picks the m-slab; persistent
    fetch/gemm WG pools (``N_FETCH_WG`` / ``N_GEMM_WG``) strided-loop over the
    XCD's fetch cells / gemm tiles; the ``(fetch_m, fetch_k)`` footprint sets each
    fetch cell and ``group_m`` orders the consumer tiles over the same slab.
    """

    xcd: XCDLayout
    num_xcds: int
    consumer: ConsumerLayout

    # ---- derived geometry --------------------------------------------------
    def _geometry(self, problem: Problem) -> dict:
        num_m_tiles = problem.num_m_tiles
        num_k_blocks = problem.num_k_blocks
        fm = self.xcd.cu.fetcher.m
        fk = self.xcd.cu.fetcher.k  # k-blocks per flag
        gm = self.consumer.group_m
        ns = self.num_xcds

        # flag granularity is owned by the layout (FetcherLayout.k)
        assert num_k_blocks % fk == 0, f"num_k_blocks ({num_k_blocks}) % fetch_k ({fk}) != 0"
        nfg_k = num_k_blocks // fk

        if self.xcd.spatial_mode:
            # ---------------- SPATIAL: role-segregated XCDs ----------------
            fetch_xcds = self.xcd.fetch_xcds
            assert 1 <= fetch_xcds < ns, f"fetch_xcds ({fetch_xcds}) must be in [1, num_xcds-1]"
            gemm_xcds = ns - fetch_xcds
            # producer/consumer spaces are GLOBAL (not per-XCD slabs)
            assert num_m_tiles % fm == 0, f"num_m_tiles ({num_m_tiles}) % fetch_m ({fm}) != 0"
            rm = num_m_tiles // fm
            assert num_m_tiles % gm == 0, f"num_m_tiles ({num_m_tiles}) % group_m ({gm}) != 0"
            total_fetch_cells = rm * nfg_k
            total_gemm_tiles = num_m_tiles * problem.num_tiles_n
            # per-XCD pool sizes (each role-XCD runs this many WGs of its role)
            n_fetch_wg = self.xcd.cu.spatial
            n_gemm_wg = self.consumer.n_gemm_wg if self.consumer.n_gemm_wg is not None else total_gemm_tiles
            # Per-role-XCD pools start at contiguous global indices [0, role_total)
            # and strided-loop by role_total, so coverage is exactly-once for ANY
            # role_total >= 1 (over-provisioned WGs simply run zero iterations).
            n_fetch_total = fetch_xcds * n_fetch_wg
            n_gemm_total = gemm_xcds * n_gemm_wg
            assert n_fetch_wg >= 1 and n_gemm_wg >= 1, "spatial pools must be >= 1"
            return dict(
                spatial=True,
                fetch_xcds=fetch_xcds,
                gemm_xcds=gemm_xcds,
                slab_m=0,  # unused in spatial (m space is global; m0 = 0)
                rm=rm,
                rk=nfg_k,
                nfg_k=nfg_k,
                # loop bounds the kernel strides over are the GLOBAL totals
                fetch_slots_per_xcd=total_fetch_cells,
                gemm_slots_per_xcd=total_gemm_tiles,
                total_fetch_cells=total_fetch_cells,
                total_gemm_tiles=total_gemm_tiles,
                n_fetch_wg=n_fetch_wg,
                n_gemm_wg=n_gemm_wg,
                n_fetch_total=n_fetch_total,
                n_gemm_total=n_gemm_total,
            )

        # ---------------- CO-LOCATED: each XCD produces+consumes a slab ----
        assert num_m_tiles % ns == 0, f"num_m_tiles ({num_m_tiles}) % num_xcds ({ns}) != 0"
        slab_m = num_m_tiles // ns

        assert self.xcd.x == ns and self.xcd.y == 1, "co-located preset requires XCDLayout(x=num_xcds, y=1)"
        assert slab_m % fm == 0, f"slab_m ({slab_m}) % fetch_m ({fm}) != 0"
        rm = slab_m // fm
        rk = nfg_k  # each fetch CELL owns exactly one flag-group along K

        fetch_slots = rm * rk  # total fetch cells per XCD (loop-iteration count)
        assert slab_m % gm == 0, f"slab_m ({slab_m}) % group_m ({gm}) != 0"
        gemm_slots = slab_m * problem.num_tiles_n  # total gemm tiles per XCD

        # Persistent WG pools per XCD. spatial = fetch pool size; n_gemm_wg =
        # gemm pool size. Each WG strided-loops over its share of cells/tiles.
        # Pools must not exceed the work count (else idle WGs / empty loops).
        n_fetch_wg = self.xcd.cu.spatial
        n_gemm_wg = self.consumer.n_gemm_wg if self.consumer.n_gemm_wg is not None else gemm_slots
        assert 1 <= n_fetch_wg <= fetch_slots, (
            f"n_fetch_wg ({n_fetch_wg}) must be in [1, fetch_slots={fetch_slots}]"
        )
        assert 1 <= n_gemm_wg <= gemm_slots, (
            f"n_gemm_wg ({n_gemm_wg}) must be in [1, gemm_slots={gemm_slots}]"
        )

        return dict(
            spatial=False,
            fetch_xcds=0,
            gemm_xcds=0,
            slab_m=slab_m,
            rm=rm,
            rk=rk,
            nfg_k=nfg_k,
            fetch_slots_per_xcd=fetch_slots,
            gemm_slots_per_xcd=gemm_slots,
            total_fetch_cells=ns * fetch_slots,
            total_gemm_tiles=ns * gemm_slots,
            n_fetch_wg=n_fetch_wg,
            n_gemm_wg=n_gemm_wg,
            n_fetch_total=ns * n_fetch_wg,
            n_gemm_total=ns * n_gemm_wg,
        )

    def grid_size(self, problem: Problem) -> int:
        g = self._geometry(problem)
        if g["spatial"]:
            # round-robin pid layout: a given (xcd, slot) exists iff
            # slot < max(pool). WGs with slot >= their role's pool are idle.
            return self.num_xcds * max(g["n_fetch_wg"], g["n_gemm_wg"])
        return self.num_xcds * (g["n_fetch_wg"] + g["n_gemm_wg"])

    # ---- cell/tile decode helpers (a pool WG walks these in a strided loop) --
    def _fetch_cell(self, xcd: int, cell: int, g: dict, problem: Problem) -> TileAssignment:
        """Decode one fetch CELL within the role's cell space.

        Co-located: cell in [0, fetch_slots_per_xcd), within this XCD's slab
        (m0 = xcd*slab_m). Spatial: cell in [0, total_fetch_cells), global m
        space (m0 = 0). The ``cell -> (fp_m, kfg)`` mapping honors
        ``FetcherLayout.order``: ``kfg`` -> flag-group advances slowest;
        ``mtile`` -> m-tile advances slowest (one m-tile's full K staged first);
        ``local_first`` -> like ``mtile`` but kfg rotated by cur_rank in-kernel so
        the local-rank gather dispatches first. The host reference is rank-agnostic
        (cur_rank=0), so its cell->kfg mapping matches ``mtile`` and coverage is
        identical for every rank (the rotation is a producer-timing-only effect)."""
        rm = g["rm"]
        nfg_k = g["nfg_k"]
        fm = self.xcd.cu.fetcher.m
        fk = self.xcd.cu.fetcher.k
        m0 = 0 if g["spatial"] else xcd * g["slab_m"]
        if self.xcd.cu.fetcher.order in (FETCH_ORDER_MTILE, FETCH_ORDER_LOCAL_FIRST,
                                         FETCH_ORDER_PIPELINED, FETCH_ORDER_COOP,
                                         FETCH_ORDER_WAVE):
            # COOP shares the same host COVERAGE as mtile (m-major over the full cell
            # space); its remote-only dense delivery reordering is a KERNEL-only effect
            # (FETCH_ORDER==4) that doesn't change which flags get staged, only when.
            fp_m = cell // nfg_k   # m-tile advances slowest
            kfg = cell % nfg_k     # local_first rotates this by cur_rank in-kernel;
                                   # pipelined only changes the per-WG WALK, not this map
        else:
            fp_m = cell % rm       # flag-group advances slowest (kfg-major)
            kfg = cell // rm
        m_tile = m0 + fp_m * fm
        k_block_start = kfg * fk
        return TileAssignment(
            pid=-1, role=ROLE_FETCH, m_tile=m_tile, k_flag_group=kfg,
            k_blocks=tuple(range(k_block_start, k_block_start + fk)),
            xcd=xcd, flag_index=m_tile * nfg_k + kfg,
        )

    def _gemm_tile(self, xcd: int, gtile: int, g: dict, problem: Problem) -> TileAssignment:
        """Decode one GEMM tile. Co-located: gtile in [0, gemm_slots_per_xcd),
        within this XCD's slab (m0 = xcd*slab_m). Spatial: gtile in
        [0, total_gemm_tiles), global m space (m0 = 0)."""
        nfg_k = g["nfg_k"]
        gm = self.consumer.group_m
        num_tiles_n = problem.num_tiles_n
        m0 = 0 if g["spatial"] else xcd * g["slab_m"]
        npig = gm * num_tiles_n
        group_id = gtile // npig
        within = gtile % npig
        pid_m = m0 + group_id * gm + (within % gm)
        pid_n = within // gm
        return TileAssignment(
            pid=-1, role=ROLE_GEMM, m_tile=pid_m, n_tile=pid_n, xcd=xcd,
            wait_flags=tuple(pid_m * nfg_k + kf for kf in range(nfg_k)),
        )

    # ---- single-pid decode (mirror of the in-kernel decode) ----------------
    def decode(self, pid: int, problem: Problem) -> TileAssignment:
        """Decode the FIRST cell/tile a pool-member WG (pid) handles.

        With persistent pools a WG strided-loops over multiple cells/tiles;
        :meth:`decode` returns the first (loop start), enough for role + the
        primary assignment. Use :meth:`materialize` for the full per-WG coverage.
        """
        g = self._geometry(problem)
        xcd = pid % self.num_xcds
        slot = pid // self.num_xcds
        if g["spatial"]:
            # role by XCD: [0, fetch_xcds) fetch-only, rest gemm-only. The global
            # cell/tile index = role_xcd * pool + slot.
            if xcd < g["fetch_xcds"]:
                fwg = xcd * g["n_fetch_wg"] + slot
                if self.xcd.cu.fetcher.order == FETCH_ORDER_PIPELINED:
                    cpf = ceil_div(g["total_fetch_cells"], g["n_fetch_total"])
                    a = self._fetch_cell(xcd, fwg * cpf, g, problem)
                else:
                    a = self._fetch_cell(xcd, fwg, g, problem)
            else:
                gxcd = xcd - g["fetch_xcds"]
                gwg = gxcd * g["n_gemm_wg"] + slot
                a = self._gemm_tile(xcd, gwg, g, problem)
            a.pid = pid
            return a
        n_fetch_wg = g["n_fetch_wg"]
        if slot < n_fetch_wg:
            a = self._fetch_cell(xcd, slot, g, problem)  # first cell = fwg index
        else:
            a = self._gemm_tile(xcd, slot - n_fetch_wg, g, problem)  # first tile
        a.pid = pid
        return a

    def materialize(self, problem: Problem) -> List[TileAssignment]:
        """Full schedule: one TileAssignment per (WG, loop-iteration). A pool WG
        with index ``slot`` handles cells/tiles ``slot, slot+pool, slot+2*pool...``
        """
        g = self._geometry(problem)
        n_fetch_wg = g["n_fetch_wg"]
        n_gemm_wg = g["n_gemm_wg"]
        fetch_slots = g["fetch_slots_per_xcd"]
        gemm_slots = g["gemm_slots_per_xcd"]
        out: List[TileAssignment] = []
        if g["spatial"]:
            # Each role-XCD's WGs strided-loop over the GLOBAL cell/tile space
            # starting at their global WG index, stride = role's total WG count.
            for pid in range(self.grid_size(problem)):
                xcd = pid % self.num_xcds
                slot = pid // self.num_xcds
                if xcd < g["fetch_xcds"]:
                    if slot >= n_fetch_wg:
                        continue  # idle WG (pool smaller than max role pool)
                    fwg = xcd * n_fetch_wg + slot
                    if self.xcd.cu.fetcher.order == FETCH_ORDER_PIPELINED:
                        # contiguous block per fetcher (mirrors in-kernel FETCH_ORDER==3)
                        cpf = ceil_div(g["total_fetch_cells"], g["n_fetch_total"])
                        cells = range(fwg * cpf, min((fwg + 1) * cpf, g["total_fetch_cells"]))
                    else:
                        cells = range(fwg, g["total_fetch_cells"], g["n_fetch_total"])
                    for cell in cells:
                        a = self._fetch_cell(xcd, cell, g, problem)
                        a.pid = pid
                        out.append(a)
                else:
                    if slot >= n_gemm_wg:
                        continue  # idle WG
                    gxcd = xcd - g["fetch_xcds"]
                    gwg = gxcd * n_gemm_wg + slot
                    for gtile in range(gwg, g["total_gemm_tiles"], g["n_gemm_total"]):
                        a = self._gemm_tile(xcd, gtile, g, problem)
                        a.pid = pid
                        out.append(a)
            return out
        for pid in range(self.grid_size(problem)):
            xcd = pid % self.num_xcds
            slot = pid // self.num_xcds
            if slot < n_fetch_wg:
                for cell in range(slot, fetch_slots, n_fetch_wg):
                    a = self._fetch_cell(xcd, cell, g, problem)
                    a.pid = pid
                    out.append(a)
            else:
                gwg = slot - n_fetch_wg
                for gtile in range(gwg, gemm_slots, n_gemm_wg):
                    a = self._gemm_tile(xcd, gtile, g, problem)
                    a.pid = pid
                    out.append(a)
        return out

    # ---- nested (CuTe-style) composition view of the fetch schedule --------
    def compose(self, problem: Problem) -> dict:
        """Express the fetch schedule as an explicit NESTED tiling, inner->outer:

            FetcherLayout (m x k cells one WG walks)   -- the "value" tile
              tiled by CULayout.spatial (the CU pool)  -- the "thread"/CU dim
                placed by XCDLayout onto A's cell grid -- WHERE on A

        This is the composition ``(CULayout.spatial  X  FetcherLayout)`` read as a
        CuTe thread x value layout. It does NOT change the kernel or the flat
        constexpr contract; it is a GPU-free re-view of exactly the same cell
        coverage that :meth:`materialize` produces, usable for the slide, static
        analysis, and the cost model's dependency graph.

        Returns a dict:
            xcds       : list of fetch-XCD ids
            spatial    : CU-pool size per XCD (concurrent fetch WGs)
            temporal   : passes each CU-slot serializes (max over slots)
            fetcher    : (m, k) footprint of one cell
            tiles[(xcd, cu_slot)] = [ (m_tile, k_flag_group), ... ]   # in walk order

        The union over all (xcd, cu_slot) of these (m_tile, kfg) pairs is IDENTICAL
        to the fetch cells in :meth:`materialize` (asserted by the coverage test).
        """
        g = self._geometry(problem)
        spatial = g["n_fetch_wg"]                 # CU pool per XCD
        fk = self.xcd.cu.fetcher.k
        fm = self.xcd.cu.fetcher.m
        xcds = list(range(g["fetch_xcds"])) if g["spatial"] else list(range(self.num_xcds))

        tiles: dict = {}
        max_pass = 0
        for xcd in xcds:
            for cu_slot in range(spatial):
                # the flat cell indices this CU-slot walks (its "temporal" passes),
                # mirroring materialize()'s strided loop exactly.
                if g["spatial"]:
                    fwg = xcd * spatial + cu_slot
                    if self.xcd.cu.fetcher.order == FETCH_ORDER_PIPELINED:
                        cpf = ceil_div(g["total_fetch_cells"], g["n_fetch_total"])
                        cell_idxs = range(fwg * cpf,
                                          min((fwg + 1) * cpf, g["total_fetch_cells"]))
                    else:
                        cell_idxs = range(fwg, g["total_fetch_cells"], g["n_fetch_total"])
                else:
                    cell_idxs = range(cu_slot, g["fetch_slots_per_xcd"], spatial)
                walk = []
                n_cells = 0
                for cell in cell_idxs:
                    n_cells += 1
                    a = self._fetch_cell(xcd, cell, g, problem)
                    # a fetch cell footprint spans FETCH_M m-tiles x 1 flag-group;
                    # expand to the (m_tile, kfg) pairs it actually covers.
                    for fi_m in range(fm):
                        walk.append((a.m_tile + fi_m, a.k_flag_group))
                tiles[(xcd, cu_slot)] = walk
                max_pass = max(max_pass, n_cells)   # temporal = cells one CU-slot serializes
        return dict(xcds=xcds, spatial=spatial, temporal=max_pass,
                    fetcher=(fm, fk), tiles=tiles)

    # ---- the flat constexpr contract the kernel consumes -------------------
    def constexprs(self, problem: Problem) -> dict:
        """Flat named ints describing the schedule. A subset feeds the kernel
        decode (see :data:`KERNEL_CONSTEXPR_KEYS`); the rest are host-only
        (grid sizing, tracing, bench display).

        Keys (K = consumed by kernel decode, H = host-only):
          NUM_XCDS            K  hardware XCD count; xcd = pid % NUM_XCDS
          SLAB_M              K  m-tiles per XCD (NUM_M_TILES // NUM_XCDS)
          RM                  K  fetch cells over m within slab (SLAB_M//FETCH_M)
          RK                  H  fetch cells over k (== NUM_FLAG_GROUPS_K)
          FETCH_M             K  fetcher footprint m-extent (in m-tiles)
          FETCH_K             K  k-blocks staged per flag (handoff grain)
          N_FETCH_WG          K  persistent fetch WG-pool size per XCD (strided loop)
          N_GEMM_WG           K  persistent gemm WG-pool size per XCD (strided loop)
          FETCH_SLOTS_PER_XCD K  total fetch cells per XCD (fetch-loop bound)
          GEMM_SLOTS_PER_XCD  K  total gemm tiles per XCD (gemm-loop bound)
          GROUP_SIZE_M        K  consumer M-grouping for L2 locality
          NUM_M_TILES         H  M // block_size_m
          NUM_TILES_N         K  ceil(N / block_size_n)
          NUM_FLAG_GROUPS_K   K  num_k_blocks // FETCH_K (flags per m-tile)
          NUM_K_BLOCKS_LOCAL  K  K_local // block_size_k (src-rank split in fetch)
        """
        g = self._geometry(problem)
        return {
            "NUM_XCDS": self.num_xcds,
            "SLAB_M": g["slab_m"],
            "RM": g["rm"],
            "RK": g["rk"],
            "FETCH_M": self.xcd.cu.fetcher.m,
            "FETCH_K": self.xcd.cu.fetcher.k,
            "N_FETCH_WG": g["n_fetch_wg"],
            "N_GEMM_WG": g["n_gemm_wg"],
            "FETCH_SLOTS_PER_XCD": g["fetch_slots_per_xcd"],
            "GEMM_SLOTS_PER_XCD": g["gemm_slots_per_xcd"],
            "GROUP_SIZE_M": self.consumer.group_m,
            "NUM_M_TILES": problem.num_m_tiles,
            "NUM_TILES_N": problem.num_tiles_n,
            "NUM_FLAG_GROUPS_K": g["nfg_k"],
            "NUM_K_BLOCKS_LOCAL": problem.num_k_blocks_local,
            "FETCH_ORDER": _FETCH_ORDER_CODE[self.xcd.cu.fetcher.order],
            "FETCH_XCDS": g["fetch_xcds"],  # 0 = co-located
            "TOTAL_FETCH_CELLS": g["total_fetch_cells"],
            "TOTAL_GEMM_TILES": g["total_gemm_tiles"],
            # DEMAND-order (FETCH_ORDER_WAVE) wave width = # of m-tiles the GEMM WG pool
            # runs concurrently (total gemm WGs / N-tiles). Only used by order 5; other
            # orders ignore it. >=1.
            "FETCH_WAVE": max(1, (g["n_gemm_wg"] * (self.num_xcds - g["fetch_xcds"])
                                  if g["spatial"] else g["n_gemm_wg"] * self.num_xcds)
                             // max(1, problem.num_tiles_n)),
        }

    # ---- correctness invariants -------------------------------------------
    def validate(self, problem: Problem) -> None:
        """Assert the schedule covers each input tile exactly once and that the
        flag space matches the workspace sizing. Raises AssertionError otherwise.
        """
        g = self._geometry(problem)
        nfg_k = g["nfg_k"]
        fm = self.xcd.cu.fetcher.m
        fk = self.xcd.cu.fetcher.k  # k-blocks per flag
        num_m_tiles = problem.num_m_tiles

        gather_count: dict = {}
        flag_writers: dict = {}
        gemm_count: dict = {}

        for a in self.materialize(problem):
            if a.role == ROLE_FETCH:
                # walk the fm m-tiles this fetcher owns for its single flag-group
                for di_m in range(fm):
                    m_tile = a.m_tile + di_m
                    kfg = a.k_flag_group
                    flag_idx = m_tile * nfg_k + kfg
                    flag_writers[flag_idx] = flag_writers.get(flag_idx, 0) + 1
                    for k_off in range(fk):
                        kb = kfg * fk + k_off
                        gather_count[(m_tile, kb)] = gather_count.get((m_tile, kb), 0) + 1
            else:
                gemm_count[(a.m_tile, a.n_tile)] = gemm_count.get((a.m_tile, a.n_tile), 0) + 1
                expected = tuple(a.m_tile * nfg_k + kf for kf in range(nfg_k))
                assert a.wait_flags == expected, f"pid {a.pid} waits {a.wait_flags}, expected {expected}"

        # every (m_tile, k_block) staged exactly once
        for m_tile in range(num_m_tiles):
            for kb in range(problem.num_k_blocks):
                c = gather_count.get((m_tile, kb), 0)
                assert c == 1, f"gather tile ({m_tile},{kb}) staged {c} times (expected 1)"

        # flag space == num_m_tiles * nfg_k, each written once
        expected_flags = problem.num_flags(fk)
        assert len(flag_writers) == expected_flags, (
            f"flag space {len(flag_writers)} != expected {expected_flags}"
        )
        for fi, c in flag_writers.items():
            assert c == 1, f"flag {fi} written {c} times (expected 1)"

        # every (m_tile, n_tile) GEMM output computed exactly once
        for m_tile in range(num_m_tiles):
            for n_tile in range(problem.num_tiles_n):
                c = gemm_count.get((m_tile, n_tile), 0)
                assert c == 1, f"gemm tile ({m_tile},{n_tile}) computed {c} times (expected 1)"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def make_layout(
    *,
    fetch_m: int,
    fetch_k: int,
    group_m: int,
    num_xcds: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    M: int,
    N: int,
    K: int,
    K_local: int,
    world_size: int,
    n_fetch_wg: Optional[int] = None,
    n_gemm_wg: Optional[int] = None,
    order: str = FETCH_ORDER_KFG,
    fetch_xcds: Optional[int] = None,
) -> Tuple[ScheduleLayout, Problem]:
    """Build a hierarchical ScheduleLayout (+ Problem) and validate it.

    ``fetch_k`` is the k-blocks-per-flag handoff granularity. ``order`` is the
    fetch traversal (``"kfg"`` | ``"mtile"``; see :class:`FetcherLayout`).

    ``fetch_xcds`` selects the XCD role mode:
      - ``None`` -> co-located preset (x=num_xcds, y=1; each XCD produces AND
        consumes a full-K m-slab). ``n_fetch_wg``/``n_gemm_wg`` are PER-XCD pool
        sizes; None = full pool (one WG per cell/tile = original behavior).
      - int in [1, num_xcds-1] -> spatial mode: that many fetch-only XCDs + the
        rest GEMM-only, partitioning the GLOBAL cell/tile spaces. ``n_fetch_wg``
        is the per-fetch-XCD pool; ``n_gemm_wg`` the per-gemm-XCD pool (None =
        full = ceil(total/role_xcds)).
    """
    problem = Problem(
        M=M,
        N=N,
        K=K,
        K_local=K_local,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        world_size=world_size,
    )

    num_m_tiles = M // block_size_m
    num_k_blocks = K // block_size_k
    assert num_k_blocks % fetch_k == 0, (
        f"num_k_blocks ({num_k_blocks}) must be divisible by fetch_k ({fetch_k})"
    )
    nfg_k = num_k_blocks // fetch_k

    fetcher = FetcherLayout(m=fetch_m, k=fetch_k, order=order)

    if fetch_xcds is not None:
        # ---- spatial: pools sized against the GLOBAL cell/tile spaces ----
        assert 1 <= fetch_xcds < num_xcds, f"fetch_xcds ({fetch_xcds}) must be in [1, num_xcds-1]"
        assert num_m_tiles % fetch_m == 0, f"num_m_tiles ({num_m_tiles}) % fetch_m ({fetch_m})"
        assert num_m_tiles % group_m == 0, f"num_m_tiles ({num_m_tiles}) % group_m ({group_m})"
        gemm_xcds = num_xcds - fetch_xcds
        total_fetch_cells = (num_m_tiles // fetch_m) * nfg_k
        total_gemm_tiles = num_m_tiles * problem.num_tiles_n
        if n_fetch_wg is None:
            n_fetch_wg = ceil_div(total_fetch_cells, fetch_xcds)
        if n_gemm_wg is None:
            n_gemm_wg = ceil_div(total_gemm_tiles, gemm_xcds)
        cu = CULayout(spatial=n_fetch_wg, temporal=ceil_div(total_fetch_cells, fetch_xcds * n_fetch_wg), fetcher=fetcher)
        xcd = XCDLayout(x=num_xcds, y=1, cu=cu, fetch_xcds=fetch_xcds)
    else:
        # ---- co-located: per-XCD slab pools ----
        assert num_m_tiles % num_xcds == 0, (
            f"num_m_tiles ({num_m_tiles}) must be divisible by num_xcds ({num_xcds})"
        )
        slab_m = num_m_tiles // num_xcds
        assert slab_m % fetch_m == 0, f"slab_m ({slab_m}) must be divisible by fetch_m ({fetch_m})"
        fetch_slots = (slab_m // fetch_m) * nfg_k
        gemm_slots = slab_m * problem.num_tiles_n
        if n_fetch_wg is None:
            n_fetch_wg = fetch_slots
        if n_gemm_wg is None:
            n_gemm_wg = gemm_slots
        cu = CULayout(spatial=n_fetch_wg, temporal=ceil_div(fetch_slots, n_fetch_wg), fetcher=fetcher)
        xcd = XCDLayout(x=num_xcds, y=1, cu=cu)

    consumer = ConsumerLayout(group_m=group_m, n_gemm_wg=n_gemm_wg)
    layout = ScheduleLayout(xcd=xcd, num_xcds=num_xcds, consumer=consumer)
    layout.validate(problem)
    return layout, problem


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Largest divisor of n that is <= cap (>= 1)."""
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def default_layout(
    M: int,
    N: int,
    K: int,
    K_local: int,
    world_size: int,
    num_xcds: int = 8,
    block_size_m: int = 128,
    block_size_n: int = 256,
    block_size_k: int = 64,
    fetch_k: Optional[int] = None,
    fetch_m: int = 1,
    group_m: Optional[int] = None,
    n_fetch_wg: Optional[int] = None,
    n_gemm_wg: Optional[int] = None,
    order: str = FETCH_ORDER_MTILE,
    fetch_xcds: Optional[int] = None,
) -> Tuple[ScheduleLayout, Problem]:
    """Derive a valid layout from the shape alone (no champion data).

    Picks divisible knobs so all divisibility asserts hold. Defaults reflect the
    tuning campaign's findings (see work-wiki log/design):

    - ``order`` defaults to ``"mtile"`` (m-tile-major): a strict win over ``"kfg"``
      at every shape measured -- a GEMM tile gets its full K staged before it runs.
    - ``fetch_k`` (k-blocks per flag) defaults to the largest divisor of
      num_k_blocks ``<= 16`` (large-K shapes want fk16; small-K fall back to fk4/8).
    - ``fetch_xcds`` defaults to ``None`` (co-located): the safe, always-divisible
      general default. SPATIAL role-segregation (``fetch_xcds=2``) WINS at small/mid
      shapes (<= ~4096^3) and co-located wins at large -- the crossover is
      world-size/AI-dependent (AI-sweep, ws=4), so it is NOT baked in here. Pass
      ``fetch_xcds=2`` explicitly for small/mid shapes to get that win.

    ``n_fetch_wg`` / ``n_gemm_wg`` are the WG-pool sizes; None = full pool. This is a
    GOOD default (right order + handoff grain), not the fully-tuned winner (which
    also uses small pools + spatial); enumerate_layouts + a sweep finds that.
    """
    num_k_blocks = K // block_size_k
    if fetch_k is None:
        fetch_k = _largest_divisor_at_most(num_k_blocks, 16)
    assert num_k_blocks % fetch_k == 0, "fetch_k must divide num_k_blocks"

    num_m_tiles = M // block_size_m
    if fetch_xcds is not None:
        # spatial: m space is global; grouping/footprint divide num_m_tiles.
        assert num_m_tiles % fetch_m == 0, "fetch_m must divide num_m_tiles"
        if group_m is None:
            group_m = _largest_divisor_at_most(num_m_tiles, 8)
        assert num_m_tiles % group_m == 0, "group_m must divide num_m_tiles"
    else:
        assert num_m_tiles % num_xcds == 0, (
            f"num_m_tiles ({num_m_tiles}) must be divisible by num_xcds ({num_xcds})"
        )
        slab_m = num_m_tiles // num_xcds
        assert slab_m % fetch_m == 0, "fetch_m must divide slab_m"
        if group_m is None:
            group_m = _largest_divisor_at_most(slab_m, 8)
        assert slab_m % group_m == 0, "group_m must divide slab_m"

    return make_layout(
        fetch_m=fetch_m,
        fetch_k=fetch_k,
        group_m=group_m,
        num_xcds=num_xcds,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        M=M,
        N=N,
        K=K,
        K_local=K_local,
        world_size=world_size,
        n_fetch_wg=n_fetch_wg,
        n_gemm_wg=n_gemm_wg,
        order=order,
        fetch_xcds=fetch_xcds,
    )


# ---------------------------------------------------------------------------
# Tuning space, expressed as layouts
# ---------------------------------------------------------------------------
@dataclass
class LayoutCandidate:
    """One point in the tuning space = a fully-specified, validated schedule.

    Bundles everything needed to launch a run: the ``ScheduleLayout`` (which owns
    fetch_m / fetch_k / group_m / n_fetch_wg / n_gemm_wg), its ``Problem``
    (which carries the block sizes that set the tile counts), and the kernel
    launch knobs (``num_warps`` / ``num_stages``) that are not part of the
    schedule. ``label`` is a short, stable, human-readable id.
    """

    layout: ScheduleLayout
    problem: Problem
    num_warps: int = 8
    num_stages: int = 2
    label: str = ""

    def __post_init__(self):
        if not self.label:
            p = self.problem
            ce = self.layout.constexprs(p)
            fx = ce["FETCH_XCDS"]
            role = f"_fx{fx}" if fx else "_coloc"
            self.label = (
                f"bm{p.block_size_m}_bn{p.block_size_n}_bk{p.block_size_k}"
                f"_fk{self.layout.xcd.cu.fetcher.k}_fm{self.layout.xcd.cu.fetcher.m}"
                f"_gm{self.layout.consumer.group_m}_{self.layout.xcd.cu.fetcher.order}{role}"
                f"_nfw{ce['N_FETCH_WG']}_ngw{ce['N_GEMM_WG']}"
                f"_nw{self.num_warps}_ns{self.num_stages}"
            )

    @property
    def fetch_k(self) -> int:
        return self.layout.xcd.cu.fetcher.k

    def constexprs(self) -> dict:
        return self.layout.constexprs(self.problem)

    def grid_size(self) -> int:
        return self.layout.grid_size(self.problem)


def _divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _divisors_leq(n: int, cap: int) -> List[int]:
    return [d for d in range(1, min(cap, n) + 1) if n % d == 0]


def enumerate_layouts(
    M: int,
    N: int,
    K: int,
    K_local: int,
    world_size: int,
    *,
    num_xcds: int = 8,
    block_size_m: Tuple[int, ...] = (128, 256),
    block_size_n: Tuple[int, ...] = (128, 256),
    block_size_k: Tuple[int, ...] = (64,),
    fetch_k: Optional[Tuple[int, ...]] = None,
    fetch_m: Optional[Tuple[int, ...]] = None,
    group_m: Optional[Tuple[int, ...]] = None,
    n_fetch_wg: Optional[Tuple[int, ...]] = None,
    n_gemm_wg: Optional[Tuple[int, ...]] = None,
    cus_per_xcd: Optional[int] = None,
    order: Tuple[str, ...] = (FETCH_ORDER_KFG, FETCH_ORDER_MTILE),
    fetch_xcds: Tuple[Optional[int], ...] = (None, 1, 2, 3, 4),
    num_warps: Tuple[int, ...] = (8,),
    num_stages: Tuple[int, ...] = (2,),
) -> Iterator[LayoutCandidate]:
    """Yield every VALID :class:`LayoutCandidate` for this shape.

    The tuning space IS this set of layouts. Each knob axis accepts a tuple of
    candidate values; ``None`` means "all valid values derived from the shape's
    divisor structure" (so the space is built from divisibility, never guessed):

      block_size_m/n/k   tiling -> sets slab_m and num_k_blocks
      fetch_k            k-blocks per flag; divisors of num_k_blocks
      fetch_m            fetcher m-footprint; divisors of slab_m
      group_m            consumer grouping; divisors of slab_m
      n_fetch_wg         persistent fetch WG-pool size per XCD; None = full pool
                         (one WG per cell). Values capped to [1, fetch_slots].
      n_gemm_wg          persistent gemm WG-pool size per XCD; None = full pool.
                         Values capped to [1, gemm_slots].
      order              fetch traversal; defaults to BOTH (kfg, mtile).
      fetch_xcds         role mode; defaults to the FULL set (None=co-located plus
                         spatial 1..4). i.e. the defaults span the whole schedule
                         space (mtile + spatial), where the winners live.
      num_warps/stages   launch knobs (not part of the schedule)

    NOTE: the order/fetch_xcds defaults span the schedule AXES, but pools stay at
    full (n_fetch_wg/n_gemm_wg=None) by default -- the tuned winners use SMALL pools
    (e.g. nfw8/ngw32), so to reach them pass pool tuples + ``cus_per_xcd``. Flipping
    pool axes to defaults too would explode the space (~182 -> 1820 with just
    order/fetch_xcds at 4096^3; ~22k if pools are also swept).

    cus_per_xcd: if set, only emit pool splits with n_fetch_wg + n_gemm_wg <=
    cus_per_xcd (deadlock-safe co-residency at occupancy 1). MI300X ~38.

    Invalid block-size combinations (don't divide the shape, or num_m_tiles not
    divisible by num_xcds) are skipped silently; everything yielded is validated.
    """
    for bm in block_size_m:
        if M % bm:
            continue
        num_m_tiles = M // bm
        if num_m_tiles % num_xcds:
            continue
        slab_m = num_m_tiles // num_xcds
        fm_vals = fetch_m if fetch_m is not None else tuple(_divisors(slab_m))
        gm_vals = group_m if group_m is not None else tuple(_divisors(slab_m))
        for bk in block_size_k:
            if K % bk or K_local % bk:
                continue
            num_k_blocks = K // bk
            fk_vals = fetch_k if fetch_k is not None else tuple(_divisors(num_k_blocks))
            for bn in block_size_n:
                for fk in fk_vals:
                    if num_k_blocks % fk:
                        continue
                    for fm in fm_vals:
                        for gm in gm_vals:
                            for fx in fetch_xcds:
                                # role-mode-dependent cell/tile spaces + grouping divisibility
                                if fx is None:
                                    if slab_m % fm or slab_m % gm:
                                        continue
                                    fetch_slots = (slab_m // fm) * (num_k_blocks // fk)
                                    gemm_slots = slab_m * (-(-N // bn))  # ceil(N/bn)
                                else:
                                    if not (1 <= fx < num_xcds):
                                        continue
                                    if num_m_tiles % fm or num_m_tiles % gm:
                                        continue
                                    # spatial pools are PER role-XCD over the global space
                                    fetch_slots = ceil_div((num_m_tiles // fm) * (num_k_blocks // fk), fx)
                                    gemm_slots = ceil_div(num_m_tiles * (-(-N // bn)), num_xcds - fx)
                                nfw_vals = n_fetch_wg if n_fetch_wg is not None else (fetch_slots,)
                                ngw_vals = n_gemm_wg if n_gemm_wg is not None else (gemm_slots,)
                                for nfw in nfw_vals:
                                    if not (1 <= nfw <= fetch_slots):
                                        continue
                                    for ngw in ngw_vals:
                                        if not (1 <= ngw <= gemm_slots):
                                            continue
                                        # deadlock-safe co-residency cap: per-XCD WGs
                                        # that share CUs. Co-located: nfw+ngw on each
                                        # XCD. Spatial: roles on separate XCDs, so the
                                        # cap applies per role independently.
                                        if cus_per_xcd is not None:
                                            if fx is None and nfw + ngw > cus_per_xcd:
                                                continue
                                            if fx is not None and (nfw > cus_per_xcd or ngw > cus_per_xcd):
                                                continue
                                        for od in order:
                                            try:
                                                layout, problem = default_layout(
                                                    M=M, N=N, K=K, K_local=K_local,
                                                    world_size=world_size, num_xcds=num_xcds,
                                                    block_size_m=bm, block_size_n=bn, block_size_k=bk,
                                                    fetch_k=fk, fetch_m=fm, group_m=gm,
                                                    n_fetch_wg=nfw, n_gemm_wg=ngw,
                                                    order=od, fetch_xcds=fx,
                                                )
                                            except AssertionError:
                                                continue
                                            for nw in num_warps:
                                                for ns in num_stages:
                                                    yield LayoutCandidate(
                                                        layout=layout, problem=problem,
                                                        num_warps=nw, num_stages=ns,
                                                    )
