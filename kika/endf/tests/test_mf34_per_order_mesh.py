"""MF34 with a mesh per Legendre order.

ENDF-6 states the grids inside each (L, L1) sub-subsection, not once for the
section: LB=5 carries one grid for a diagonal block and LB=6 a row/column pair
off it. So an order whose coefficient is well determined need not be written on
the mesh the noisiest order requires, and a section that does this is ordinary
format, not an extension.

The gate that matters is the first one: asking for a mesh per order and then
handing every order the *same* mesh must reproduce the shared-mesh output byte
for byte. That is what makes the shared path safe to leave alone.
"""
import numpy as np
import pytest

from kika.endf.writers.mf34_writer import create_mf34_from_covariance

ZA, AWR, MAT, MT = 26056.0, 55.454, 2631, 2
L_MAX = 4


def _fixture(n_energies=12, l_max=L_MAX, seed=7):
    rng = np.random.default_rng(seed)
    grid = np.linspace(1.0e6, 4.0e6, n_energies + 1)
    size = n_energies * l_max
    a = rng.normal(size=(size, size))
    cov = (a @ a.T) / size * 1e-3
    return grid, cov


def _blocks_from_flat(cov, n_energies, l_max, rows_of=None):
    """Slice the shared-layout matrix the way the writer documents."""
    rows_of = rows_of or {l: np.arange(n_energies) for l in range(1, l_max + 1)}
    out = {}
    for l in range(1, l_max + 1):
        r = [i * l_max + (l - 1) for i in rows_of[l]]
        for l1 in range(l, l_max + 1):
            c = [i * l_max + (l1 - 1) for i in rows_of[l1]]
            out[(l, l1)] = cov[np.ix_(r, c)]
    return out


def test_per_order_meshes_all_equal_reproduce_the_shared_mesh_byte_for_byte():
    grid, cov = _fixture()
    n = len(grid) - 1
    shared = create_mf34_from_covariance(
        cov, grid, max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1)
    per_order = create_mf34_from_covariance(
        _blocks_from_flat(cov, n, L_MAX), {l: grid for l in range(1, L_MAX + 1)},
        max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1)
    assert str(per_order) == str(shared)


def test_per_order_meshes_all_equal_reproduce_the_cross_form_byte_for_byte():
    grid, cov = _fixture()
    n = len(grid) - 1
    coarse = np.linspace(grid[0], grid[-1], 4)
    rng = np.random.default_rng(11)
    cross = {l: rng.normal(size=(len(coarse) - 1, n)) * 1e-2
             for l in range(1, L_MAX + 1)}
    shared = create_mf34_from_covariance(
        cov, grid, max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1,
        cross_cov=cross, cross_energy_grid_ev=coarse)
    per_order = create_mf34_from_covariance(
        _blocks_from_flat(cov, n, L_MAX), {l: grid for l in range(1, L_MAX + 1)},
        max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1,
        cross_cov=cross, cross_energy_grid_ev=coarse)
    assert str(per_order) == str(shared)


def test_each_sub_subsection_carries_its_own_grid():
    grid, cov = _fixture()
    n = len(grid) - 1
    rows_of = {1: np.arange(n), 2: np.arange(n),
               3: np.arange(0, n, 2), 4: np.arange(0, n, 4)}
    grids = {l: np.append(grid[idx], grid[-1]) for l, idx in rows_of.items()}
    sec = create_mf34_from_covariance(
        _blocks_from_flat(cov, n, L_MAX, rows_of), grids,
        max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1)

    sizes = {l: len(grids[l]) - 1 for l in grids}
    seen = {(s.l, s.l1): s.records[0]
            for s in sec._subsections[0].sub_subsections}
    assert set(seen) == {(l, l1) for l in range(1, L_MAX + 1)
                         for l1 in range(l, L_MAX + 1)}
    for (l, l1), rec in seen.items():
        if l == l1:
            assert rec.lb == 5
            assert len(rec.energies) - 1 == sizes[l]
        else:
            assert rec.lb == 6
            assert len(rec.row_energies) - 1 == sizes[l]
            assert len(rec.col_energies) - 1 == sizes[l1]


def test_cross_blocks_follow_each_order_column_count():
    grid, cov = _fixture()
    n = len(grid) - 1
    rows_of = {1: np.arange(n), 2: np.arange(n),
               3: np.arange(0, n, 2), 4: np.arange(0, n, 4)}
    grids = {l: np.append(grid[idx], grid[-1]) for l, idx in rows_of.items()}
    coarse = np.linspace(grid[0], grid[-1], 4)
    rng = np.random.default_rng(3)
    cross = {l: rng.normal(size=(len(coarse) - 1, len(idx))) * 1e-2
             for l, idx in rows_of.items()}
    sec = create_mf34_from_covariance(
        _blocks_from_flat(cov, n, L_MAX, rows_of), grids,
        max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1,
        cross_cov=cross, cross_energy_grid_ev=coarse)
    assert sec._ltt == 3
    for s in sec._subsections[0].sub_subsections:
        if s.l == 0 and s.l1 >= 1:
            rec = s.records[0]
            assert len(rec.row_energies) - 1 == len(coarse) - 1
            assert len(rec.col_energies) - 1 == len(grids[s.l1]) - 1


def test_the_two_dict_forms_must_be_given_together():
    grid, cov = _fixture()
    with pytest.raises(ValueError, match="both a dict of grids and a dict"):
        create_mf34_from_covariance(
            cov, {l: grid for l in range(1, L_MAX + 1)},
            max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1)


def test_a_missing_block_is_named():
    grid, cov = _fixture()
    n = len(grid) - 1
    blocks = _blocks_from_flat(cov, n, L_MAX)
    del blocks[(2, 3)]
    with pytest.raises(ValueError, match=r"missing the \(2, 3\) block"):
        create_mf34_from_covariance(
            blocks, {l: grid for l in range(1, L_MAX + 1)},
            max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1)


def test_a_block_that_disagrees_with_its_meshes_is_named():
    grid, cov = _fixture()
    n = len(grid) - 1
    blocks = _blocks_from_flat(cov, n, L_MAX)
    blocks[(1, 2)] = blocks[(1, 2)][:, :-1]
    with pytest.raises(ValueError, match=r"cov_matrix\[\(1, 2\)\] shape"):
        create_mf34_from_covariance(
            blocks, {l: grid for l in range(1, L_MAX + 1)},
            max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1)


def test_ragged_meshes_refuse_the_stacked_cross_array():
    """It cannot hold blocks of different widths, so say so instead of
    broadcasting something wrong."""
    grid, cov = _fixture()
    n = len(grid) - 1
    rows_of = {1: np.arange(n), 2: np.arange(n),
               3: np.arange(0, n, 2), 4: np.arange(0, n, 4)}
    grids = {l: np.append(grid[idx], grid[-1]) for l, idx in rows_of.items()}
    coarse = np.linspace(grid[0], grid[-1], 4)
    stacked = np.zeros((L_MAX, len(coarse) - 1, n))
    with pytest.raises(ValueError, match="ragged"):
        create_mf34_from_covariance(
            _blocks_from_flat(cov, n, L_MAX, rows_of), grids,
            max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1,
            cross_cov=stacked, cross_energy_grid_ev=coarse)


def test_the_cross_grid_has_no_default_once_the_orders_differ():
    grid, cov = _fixture()
    n = len(grid) - 1
    rows_of = {1: np.arange(n), 2: np.arange(n),
               3: np.arange(0, n, 2), 4: np.arange(0, n, 4)}
    grids = {l: np.append(grid[idx], grid[-1]) for l, idx in rows_of.items()}
    cross = {l: np.zeros((3, len(idx))) for l, idx in rows_of.items()}
    with pytest.raises(ValueError, match="cross_energy_grid_ev is required"):
        create_mf34_from_covariance(
            _blocks_from_flat(cov, n, L_MAX, rows_of), grids,
            max_order=L_MAX, za=ZA, awr=AWR, mat=MAT, mt=MT, ltt=1,
            cross_cov=cross)
