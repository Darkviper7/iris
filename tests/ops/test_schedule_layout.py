# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Pure-Python tests for the hierarchical scheduling-layout descriptor.

Loaded by file path so importing the iris package (which GPU-inits and hangs on
CPU-only / multi-GPU hosts) is avoided. No torch, no triton, no GPU.
"""

import importlib.util
import pathlib

import pytest

_SL_PATH = pathlib.Path(__file__).resolve().parents[2] / "iris" / "ops" / "schedule_layout.py"
_spec = importlib.util.spec_from_file_location("schedule_layout", _SL_PATH)
sl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sl)


# (M, N, K, K_local, world_size, num_xcds, bm, bn, bk, fetch_m, fetch_k, group_m,
#  [n_fetch_wg, n_gemm_wg]).  fetch_k = k-blocks per flag (handoff grain).
# Divisibility contract: nfg_k = num_k_blocks // fetch_k, rm = slab_m // fetch_m,
# fetch_slots = rm * nfg_k, gemm_slots = slab_m * ceil(N/bn), group_m | slab_m.
# n_fetch_wg/n_gemm_wg (persistent pools) default to full (one WG per cell/tile)
# when omitted; when given must be in [1, slots].
VALID_CONFIGS = [
    # full pools (one WG per cell/tile): slab=2, nfg_k=16 -> fetch_slots=32
    dict(M=2048, N=512, K=1024, K_local=128, world_size=8, num_xcds=8,
         bm=128, bn=256, bk=64, fetch_m=1, fetch_k=1, group_m=1),
    # fetch_k=2 -> nfg_k=8, fetch_slots=16; full pools
    dict(M=2048, N=512, K=1024, K_local=128, world_size=8, num_xcds=8,
         bm=128, bn=256, bk=64, fetch_m=1, fetch_k=2, group_m=2),
    # slab=4, nfg_k=16, fetch_m=2 -> fetch_slots=32; full pools
    dict(M=4096, N=1024, K=2048, K_local=256, world_size=8, num_xcds=8,
         bm=128, bn=256, bk=64, fetch_m=2, fetch_k=2, group_m=4),
    # num_xcds=4: slab=4, nfg_k=4, fetch_m=2 -> fetch_slots=8; full pools
    dict(M=2048, N=512, K=512, K_local=128, world_size=4, num_xcds=4,
         bm=128, bn=256, bk=64, fetch_m=2, fetch_k=2, group_m=2),
    # PERSISTENT: small pools. slab=2, nfg_k=16 -> fetch_slots=32, gemm_slots=2*2=4
    # n_fetch_wg=6 (6 WGs strided over 32 cells), n_gemm_wg=2 (2 over 4 tiles)
    dict(M=2048, N=512, K=1024, K_local=128, world_size=8, num_xcds=8,
         bm=128, bn=256, bk=64, fetch_m=1, fetch_k=1, group_m=1,
         n_fetch_wg=6, n_gemm_wg=2),
    # PERSISTENT champion-like: 4096^3 ws4 bm256 -> slab=2, nfg_k=16,
    # fetch_slots=32, gemm_slots=2*16=32; pools 4 fetch + 16 gemm
    dict(M=4096, N=4096, K=4096, K_local=1024, world_size=4, num_xcds=8,
         bm=256, bn=256, bk=64, fetch_m=1, fetch_k=4, group_m=2,
         n_fetch_wg=4, n_gemm_wg=16),
]


def _build(cfg):
    return sl.make_layout(
        fetch_m=cfg["fetch_m"], fetch_k=cfg["fetch_k"],
        group_m=cfg["group_m"],
        num_xcds=cfg["num_xcds"],
        block_size_m=cfg["bm"], block_size_n=cfg["bn"], block_size_k=cfg["bk"],
        M=cfg["M"], N=cfg["N"], K=cfg["K"], K_local=cfg["K_local"],
        world_size=cfg["world_size"],
        n_fetch_wg=cfg.get("n_fetch_wg"), n_gemm_wg=cfg.get("n_gemm_wg"),
    )


@pytest.mark.parametrize("cfg", VALID_CONFIGS)
def test_make_layout_validates(cfg):
    # make_layout runs validate() internally; reaching here means it passed.
    layout, problem = _build(cfg)
    assert layout.grid_size(problem) > 0


@pytest.mark.parametrize("cfg", VALID_CONFIGS)
def test_coverage_each_gather_tile_once(cfg):
    layout, problem = _build(cfg)
    fk = cfg["fetch_k"]  # k-blocks per flag
    gather = {}
    for a in layout.materialize(problem):
        if a.role != sl.ROLE_FETCH:
            continue
        # each fetcher owns fetch_m m-tiles for ONE flag-group, staging fk k-blocks
        for di_m in range(cfg["fetch_m"]):
            m_tile = a.m_tile + di_m
            kfg = a.k_flag_group
            for k_off in range(fk):
                kb = kfg * fk + k_off
                gather[(m_tile, kb)] = gather.get((m_tile, kb), 0) + 1
    for m in range(problem.num_m_tiles):
        for kb in range(problem.num_k_blocks):
            assert gather.get((m, kb), 0) == 1


@pytest.mark.parametrize("cfg", VALID_CONFIGS)
def test_coverage_each_gemm_tile_once(cfg):
    layout, problem = _build(cfg)
    gemm = {}
    for a in layout.materialize(problem):
        if a.role != sl.ROLE_GEMM:
            continue
        gemm[(a.m_tile, a.n_tile)] = gemm.get((a.m_tile, a.n_tile), 0) + 1
    for m in range(problem.num_m_tiles):
        for n in range(problem.num_tiles_n):
            assert gemm.get((m, n), 0) == 1


@pytest.mark.parametrize("cfg", VALID_CONFIGS)
def test_flag_space_sizing(cfg):
    layout, problem = _build(cfg)
    fk = cfg["fetch_k"]
    nfg_k = layout.constexprs(problem)["NUM_FLAG_GROUPS_K"]
    writers = set()
    for a in layout.materialize(problem):
        if a.role != sl.ROLE_FETCH:
            continue
        for di_m in range(cfg["fetch_m"]):
            writers.add((a.m_tile + di_m) * nfg_k + a.k_flag_group)
    assert len(writers) == problem.num_flags(fk)


@pytest.mark.parametrize("cfg", VALID_CONFIGS)
def test_colocation_consumer_on_producer_xcd(cfg):
    """Every flag a GEMM waits on was written by a fetcher on the SAME xcd."""
    layout, problem = _build(cfg)
    nfg_k = layout.constexprs(problem)["NUM_FLAG_GROUPS_K"]
    flag_to_xcd = {}
    gemm_assignments = []
    for a in layout.materialize(problem):
        if a.role == sl.ROLE_FETCH:
            for di_m in range(cfg["fetch_m"]):
                fi = (a.m_tile + di_m) * nfg_k + a.k_flag_group
                flag_to_xcd[fi] = a.xcd
        else:
            gemm_assignments.append(a)
    for a in gemm_assignments:
        for fi in a.wait_flags:
            assert flag_to_xcd[fi] == a.xcd, (
                f"GEMM pid {a.pid} on xcd {a.xcd} waits on flag {fi} produced by xcd {flag_to_xcd[fi]}"
            )


@pytest.mark.parametrize("cfg", VALID_CONFIGS)
def test_materialize_covers_all_cells_and_tiles(cfg):
    # With persistent pools, grid_size = #WGs (num_xcds * pools) while materialize
    # yields one entry per (WG, loop-iteration). Total assignments must equal the
    # total work: num_xcds * (fetch_slots + gemm_slots).
    layout, problem = _build(cfg)
    ce = layout.constexprs(problem)
    expected = ce["NUM_XCDS"] * (ce["FETCH_SLOTS_PER_XCD"] + ce["GEMM_SLOTS_PER_XCD"])
    assert len(layout.materialize(problem)) == expected


@pytest.mark.parametrize("cfg", VALID_CONFIGS)
def test_constexpr_identities(cfg):
    layout, problem = _build(cfg)
    ce = layout.constexprs(problem)
    assert ce["FETCH_SLOTS_PER_XCD"] == ce["RM"] * ce["RK"]
    assert ce["GEMM_SLOTS_PER_XCD"] == ce["SLAB_M"] * ce["NUM_TILES_N"]
    assert ce["SLAB_M"] == ce["NUM_M_TILES"] // ce["NUM_XCDS"]
    # pools are within [1, slots]
    assert 1 <= ce["N_FETCH_WG"] <= ce["FETCH_SLOTS_PER_XCD"]
    assert 1 <= ce["N_GEMM_WG"] <= ce["GEMM_SLOTS_PER_XCD"]
    # grid = num_xcds * (pool sum)
    assert layout.grid_size(problem) == ce["NUM_XCDS"] * (ce["N_FETCH_WG"] + ce["N_GEMM_WG"])


def test_full_pool_default_matches_one_wg_per_cell():
    """With pools omitted, N_FETCH_WG/N_GEMM_WG == slot counts and grid ==
    num_xcds*(fetch_slots+gemm_slots) -- the original non-persistent behavior."""
    layout, problem = sl.make_layout(
        fetch_m=1, fetch_k=4, group_m=2, num_xcds=8,
        block_size_m=256, block_size_n=256, block_size_k=64,
        M=4096, N=4096, K=4096, K_local=1024, world_size=4,
    )
    ce = layout.constexprs(problem)
    assert ce["N_FETCH_WG"] == ce["FETCH_SLOTS_PER_XCD"]
    assert ce["N_GEMM_WG"] == ce["GEMM_SLOTS_PER_XCD"]
    assert layout.grid_size(problem) == 8 * (32 + 32)  # the known champion grid


@pytest.mark.parametrize("n_fetch_wg,n_gemm_wg", [(1, 1), (2, 2), (4, 16), (6, 8), (32, 32)])
def test_persistent_pools_preserve_coverage(n_fetch_wg, n_gemm_wg):
    """Any pool sizes <= slot counts still cover every cell/tile exactly once via
    the strided loops (validate() checks coverage over the full materialization)."""
    layout, problem = sl.make_layout(
        fetch_m=1, fetch_k=4, group_m=2, num_xcds=8,
        block_size_m=256, block_size_n=256, block_size_k=64,
        M=4096, N=4096, K=4096, K_local=1024, world_size=4,
        n_fetch_wg=n_fetch_wg, n_gemm_wg=n_gemm_wg,
    )
    layout.validate(problem)  # raises if any cell/tile is missed or double-covered
    ce = layout.constexprs(problem)
    assert ce["N_FETCH_WG"] == n_fetch_wg and ce["N_GEMM_WG"] == n_gemm_wg


def test_default_layout_is_valid():
    layout, problem = sl.default_layout(
        M=4096, N=4096, K=4096, K_local=512, world_size=8, num_xcds=8,
        block_size_m=128, block_size_n=256, block_size_k=64,
    )
    # default_layout calls make_layout -> validate(); just confirm it produced a grid.
    assert layout.grid_size(problem) > 0


def test_bad_divisibility_raises():
    # num_m_tiles (3) not divisible by num_xcds (8)
    with pytest.raises(AssertionError):
        sl.make_layout(
            fetch_m=1, fetch_k=1, group_m=1,
            num_xcds=8, block_size_m=128, block_size_n=256, block_size_k=64,
            M=128 * 3, N=256, K=512, K_local=64, world_size=8,
        )


def test_bad_pool_too_large_raises():
    # n_fetch_wg (999) > fetch_slots -> invalid
    with pytest.raises(AssertionError):
        sl.make_layout(
            fetch_m=1, fetch_k=1, group_m=1, n_fetch_wg=999,
            num_xcds=8, block_size_m=128, block_size_n=256, block_size_k=64,
            M=2048, N=512, K=1024, K_local=128, world_size=8,
        )


# ---------------------------------------------------------------------------
# New schedule knobs: FetcherLayout.order and XCDLayout.fetch_xcds
# ---------------------------------------------------------------------------
def _champion_kwargs(**over):
    """4096^3 ws4 bm256 champion-ish base; override any knob."""
    base = dict(
        fetch_m=1, fetch_k=4, group_m=2, num_xcds=8,
        block_size_m=256, block_size_n=256, block_size_k=64,
        M=4096, N=4096, K=4096, K_local=1024, world_size=4,
    )
    base.update(over)
    return base


@pytest.mark.parametrize("order", [sl.FETCH_ORDER_KFG, sl.FETCH_ORDER_MTILE])
def test_order_preserves_coverage(order):
    """Both fetch orders cover every gather tile / flag / gemm tile exactly once
    (validate() runs inside make_layout); only producer->consumer timing differs."""
    layout, problem = sl.make_layout(**_champion_kwargs(order=order))
    layout.validate(problem)
    ce = layout.constexprs(problem)
    assert ce["FETCH_ORDER"] == (1 if order == sl.FETCH_ORDER_MTILE else 0)
    assert ce["FETCH_XCDS"] == 0  # still co-located


def test_order_changes_first_cell_mapping():
    """kfg-major vs m-tile-major must decode the SAME cell to different tiles
    (proving the order knob actually reorders the traversal)."""
    kf, p = sl.make_layout(**_champion_kwargs(order=sl.FETCH_ORDER_KFG))
    mt, _ = sl.make_layout(**_champion_kwargs(order=sl.FETCH_ORDER_MTILE))
    g_kf = kf._geometry(p)
    g_mt = mt._geometry(p)
    nfg_k = g_kf["nfg_k"]
    assert nfg_k > 1 and g_kf["rm"] > 1  # need a 2-D cell space for order to matter
    # cell index 1: kfg-major advances m first; m-tile-major advances kfg first
    a_kf = kf._fetch_cell(0, 1, g_kf, p)
    a_mt = mt._fetch_cell(0, 1, g_mt, p)
    assert (a_kf.m_tile, a_kf.k_flag_group) != (a_mt.m_tile, a_mt.k_flag_group)


@pytest.mark.parametrize("fetch_xcds", [1, 2, 3, 4])
@pytest.mark.parametrize("order", [sl.FETCH_ORDER_KFG, sl.FETCH_ORDER_MTILE])
def test_spatial_coverage(fetch_xcds, order):
    """Spatial role-segregated layouts cover the GLOBAL gather/flag/gemm spaces
    exactly once (validate() asserts this); constexprs report the split."""
    layout, problem = sl.make_layout(**_champion_kwargs(
        fetch_xcds=fetch_xcds, order=order, n_fetch_wg=None, n_gemm_wg=None))
    layout.validate(problem)
    ce = layout.constexprs(problem)
    assert ce["FETCH_XCDS"] == fetch_xcds
    assert ce["TOTAL_FETCH_CELLS"] == problem.num_m_tiles * ce["NUM_FLAG_GROUPS_K"]
    assert ce["TOTAL_GEMM_TILES"] == problem.num_m_tiles * problem.num_tiles_n


def test_spatial_roles_on_separate_xcds():
    """In spatial mode fetchers live only on [0,fetch_xcds), gemm only on the rest."""
    fetch_xcds = 2
    layout, problem = sl.make_layout(**_champion_kwargs(fetch_xcds=fetch_xcds))
    for a in layout.materialize(problem):
        if a.role == sl.ROLE_FETCH:
            assert a.xcd < fetch_xcds
        else:
            assert a.xcd >= fetch_xcds


def test_spatial_small_pools_preserve_coverage():
    """Per-role-XCD pools smaller than the role's share still cover the global
    space via strided loops."""
    layout, problem = sl.make_layout(**_champion_kwargs(
        fetch_xcds=2, n_fetch_wg=8, n_gemm_wg=16))
    layout.validate(problem)  # raises on miss / double-cover
    ce = layout.constexprs(problem)
    assert ce["N_FETCH_WG"] == 8 and ce["N_GEMM_WG"] == 16


def test_spatial_bad_fetch_xcds_raises():
    with pytest.raises(AssertionError):
        sl.make_layout(**_champion_kwargs(fetch_xcds=8))  # == num_xcds, no gemm XCDs


def test_enumerate_includes_order_and_spatial():
    """enumerate_layouts emits both orders and both role modes when asked."""
    cands = list(sl.enumerate_layouts(
        4096, 4096, 4096, 1024, 4, num_xcds=8,
        block_size_m=(256,), block_size_n=(256,), block_size_k=(64,),
        fetch_k=(4,), fetch_m=(1,), group_m=(2,),
        order=(sl.FETCH_ORDER_KFG, sl.FETCH_ORDER_MTILE),
        fetch_xcds=(None, 2),
    ))
    orders = {c.layout.xcd.cu.fetcher.order for c in cands}
    modes = {c.constexprs()["FETCH_XCDS"] for c in cands}
    assert orders == {sl.FETCH_ORDER_KFG, sl.FETCH_ORDER_MTILE}
    assert 0 in modes and 2 in modes
    for c in cands:  # every emitted candidate is valid
        c.layout.validate(c.problem)
