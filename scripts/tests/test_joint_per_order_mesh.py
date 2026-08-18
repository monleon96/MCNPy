"""JointCov emitting a mesh per Legendre order.

The gate that matters is the identity one: asking for per-order meshes and then
handing every order the mesh the object already has must reproduce the shared
emission byte for byte. Everything else is a coarsening on top of that.
"""
import numpy as np
import pytest

from scripts.joint_covariance import JointCov

ZA, AWR, MAT, MT = 26056.0, 55.454, 2631, 2
L = 3


def _joint(n_sigma=4, n_a=12, l_max=L, seed=5, cross=True):
    rng = np.random.default_rng(seed)
    n = n_sigma + n_a * l_max
    a = rng.normal(size=(n, n))
    m = (a @ a.T) / n * 1e-3
    if not cross:
        m[:n_sigma, n_sigma:] = 0.0
        m[n_sigma:, :n_sigma] = 0.0
    return JointCov(
        grid_sigma_ev=np.linspace(1.0e6, 4.0e6, n_sigma + 1),
        grid_a_ev=np.linspace(1.0e6, 4.0e6, n_a + 1),
        l_max=l_max, matrix=m,
    )


def _identity_maps(j):
    eye = np.eye(j.n_a_bins)
    return ({l: eye for l in range(1, j.l_max + 1)},
            {l: j.grid_a_ev for l in range(1, j.l_max + 1)})


@pytest.mark.parametrize("cross", [True, False])
def test_identity_maps_reproduce_the_shared_emission_byte_for_byte(cross):
    j = _joint(cross=cross)
    s33, s34 = j.to_endf_sections(ZA, AWR, MAT, MT)
    w, g = _identity_maps(j)
    p33, p34 = j.to_endf_sections(ZA, AWR, MAT, MT,
                                  order_weights=w, order_grids_ev=g)
    assert str(p33) == str(s33)
    assert str(p34) == str(s34)


def test_collapse_is_a_congruence_so_it_cannot_invent_variance():
    """W C W^T on the full joint: the coarse object's spectrum is bounded by
    the fine one's, which is what makes the collapse safe to ship."""
    j = _joint()
    n_g = 4
    W = np.zeros((n_g, j.n_a_bins))
    for g in range(n_g):
        W[g, g * 3:(g + 1) * 3] = 1 / 3
    grids = {l: j.grid_a_ev[::3] for l in range(1, j.l_max + 1)}
    blocks, _ = j.collapse_orders({l: W for l in range(1, j.l_max + 1)}, grids)
    for l in range(1, j.l_max + 1):
        fine = j.c34[np.ix_(
            np.arange(j.n_a_bins) * j.l_max + (l - 1),
            np.arange(j.n_a_bins) * j.l_max + (l - 1))]
        coarse = blocks[(l, l)]
        assert np.allclose(coarse, coarse.T)
        assert np.linalg.eigvalsh(coarse).min() >= -1e-12
        assert np.linalg.eigvalsh(coarse).max() <= np.linalg.eigvalsh(fine).max() + 1e-12


def test_the_cross_columns_follow_the_same_map_as_the_shape_blocks():
    j = _joint()
    n_g = 4
    W = np.zeros((n_g, j.n_a_bins))
    for g in range(n_g):
        W[g, g * 3:(g + 1) * 3] = 1 / 3
    Ws = {1: W, 2: W[::2], 3: np.eye(j.n_a_bins)}
    grids = {1: j.grid_a_ev[::3], 2: j.grid_a_ev[::6], 3: j.grid_a_ev}
    blocks, cross = j.collapse_orders(Ws, grids)
    for l in range(1, j.l_max + 1):
        assert blocks[(l, l)].shape == (Ws[l].shape[0], Ws[l].shape[0])
        assert cross[l].shape == (j.n_sigma, Ws[l].shape[0])
    assert blocks[(1, 2)].shape == (Ws[1].shape[0], Ws[2].shape[0])


def test_ragged_meshes_reach_the_section_and_each_block_keeps_its_grid():
    j = _joint()
    W4 = np.zeros((4, j.n_a_bins))
    for g in range(4):
        W4[g, g * 3:(g + 1) * 3] = 1 / 3
    Ws = {1: np.eye(j.n_a_bins), 2: W4, 3: W4[::2]}
    grids = {1: j.grid_a_ev, 2: j.grid_a_ev[::3], 3: j.grid_a_ev[::6]}
    _, sec34 = j.to_endf_sections(ZA, AWR, MAT, MT,
                                  order_weights=Ws, order_grids_ev=grids)
    sizes = {l: len(grids[l]) - 1 for l in grids}
    for s in sec34._subsections[0].sub_subsections:
        rec = s.records[0]
        if s.l == 0 and s.l1 == 0:
            # the null magnitude self-block: LB=5, so one grid, the sigma one
            assert len(rec.energies) - 1 == j.n_sigma
        elif s.l == 0:
            assert len(rec.row_energies) - 1 == j.n_sigma
            assert len(rec.col_energies) - 1 == sizes[s.l1]
        elif s.l == s.l1:
            assert len(rec.energies) - 1 == sizes[s.l]
        else:
            assert len(rec.row_energies) - 1 == sizes[s.l]
            assert len(rec.col_energies) - 1 == sizes[s.l1]


def test_the_two_arguments_go_together():
    j = _joint()
    w, g = _identity_maps(j)
    with pytest.raises(ValueError, match="go together"):
        j.to_endf_sections(ZA, AWR, MAT, MT, order_weights=w)
    with pytest.raises(ValueError, match="go together"):
        j.to_endf_sections(ZA, AWR, MAT, MT, order_grids_ev=g)


def test_a_map_that_disagrees_with_its_grid_is_named():
    j = _joint()
    w, g = _identity_maps(j)
    g = dict(g); g[2] = g[2][:-1]
    with pytest.raises(ValueError, match=r"order_grids_ev\[2\] has"):
        j.to_endf_sections(ZA, AWR, MAT, MT, order_weights=w, order_grids_ev=g)


def test_a_map_with_the_wrong_number_of_fine_bins_is_named():
    j = _joint()
    w, g = _identity_maps(j)
    w = dict(w); w[3] = w[3][:, :-1]
    with pytest.raises(ValueError, match=r"order_weights\[3\] is"):
        j.to_endf_sections(ZA, AWR, MAT, MT, order_weights=w, order_grids_ev=g)


def test_a_missing_order_is_named():
    j = _joint()
    w, g = _identity_maps(j)
    w = dict(w); del w[2]
    with pytest.raises(ValueError, match=r"missing order\(s\) \[2\]"):
        j.to_endf_sections(ZA, AWR, MAT, MT, order_weights=w, order_grids_ev=g)
