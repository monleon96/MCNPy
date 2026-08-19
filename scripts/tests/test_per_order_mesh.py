"""The per-order MF34 mesh.

The gate that matters is the identity one: handing every order the mesh it
already has must reproduce the shared-mesh covariance exactly.  Everything else
is a coarsening on top of that.
"""
import numpy as np
import pytest

from scripts.per_order_mesh import (
    collapse_relative_per_order,
    order_aggregator,
    order_cut_indices,
    per_order_meshes,
    solve_order,
)

L = 3


def _case(n_g=10, l_max=L, seed=3, dead=()):
    """Relative covariance + means on a regular mesh, with chosen orders absent."""
    rng = np.random.default_rng(seed)
    n = n_g * l_max
    a = rng.normal(size=(n, n))
    cov = (a @ a.T) / n
    means = rng.normal(size=n) + 3.0
    for g, l in dead:
        means[g * l_max + (l - 1)] = 0.0
    edges = np.linspace(1.0e6, 4.0e6, n_g + 1)
    return edges, cov, means


# --------------------------------------------------------------------------- #
# equivalence: the same mesh must be a no-op
# --------------------------------------------------------------------------- #
def test_identity_mesh_reproduces_the_shared_collapse():
    edges, cov, means = _case()
    meshes = {l: edges for l in range(1, L + 1)}
    blocks, grids, W = collapse_relative_per_order(edges, cov, means, L, meshes)

    for l in range(1, L + 1):
        np.testing.assert_array_equal(grids[l], edges)
        np.testing.assert_allclose(W[l], np.eye(len(edges) - 1), atol=0, rtol=0)
    n_g = len(edges) - 1
    for l in range(1, L + 1):
        for l1 in range(l, L + 1):
            rows = np.arange(n_g) * L + (l - 1)
            cols = np.arange(n_g) * L + (l1 - 1)
            np.testing.assert_allclose(blocks[(l, l1)], cov[np.ix_(rows, cols)],
                                       atol=0, rtol=0)


def test_identity_aggregator_is_exactly_the_identity_even_where_the_mean_is_zero():
    """A dead group must not become a zero ROW of the identity map."""
    edges, _cov, means = _case(dead=[(2, 1), (5, 1)])
    sel = np.arange(len(edges) - 1) * L
    W = order_aggregator(edges, edges, means[sel])
    # w_i * m_i / (w_i * m_i) == 1 wherever m != 0; the dead rows are 0/0 -> 0
    expected = np.diag((means[sel] != 0).astype(float))
    np.testing.assert_allclose(W, expected, atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# the collapse is exact in relative space
# --------------------------------------------------------------------------- #
def test_relative_collapse_matches_the_absolute_route_where_means_are_nonzero():
    edges, cov, means = _case(n_g=8, seed=11)
    coarse = edges[::2]
    meshes = {l: coarse for l in range(1, L + 1)}
    blocks, _, W = collapse_relative_per_order(edges, cov, means, L, meshes)

    n_g = len(edges) - 1
    for l in range(1, L + 1):
        sel = np.arange(n_g) * L + (l - 1)
        m = means[sel]
        w = np.diff(edges)
        # absolute route: collapse abs with width weights, divide by the merged mean
        A = np.zeros((len(coarse) - 1, n_g))
        g = np.repeat(np.arange(len(coarse) - 1), 2)
        A[g, np.arange(n_g)] = w
        A /= A.sum(1, keepdims=True)
        abs_c = A @ (cov[np.ix_(sel, sel)] * np.outer(m, m)) @ A.T
        m_c = A @ m
        np.testing.assert_allclose(blocks[(l, l)], abs_c / np.outer(m_c, m_c),
                                   rtol=1e-12, atol=0)
        assert W[l].shape == (len(coarse) - 1, n_g)


# --------------------------------------------------------------------------- #
# what the mesh is allowed to touch
# --------------------------------------------------------------------------- #
def test_the_lower_window_edge_survives_when_nothing_is_merged():
    """The fork's rebuild dropped base[lo] unconditionally; that is the a_2 = 702 of 703."""
    n_g = 12
    w = np.ones(n_g)
    a_g = np.full(n_g, 10.0)
    B = np.eye(n_g)                       # sd 1 against mean 10: no group is degenerate
    live = np.ones(n_g, dtype=bool)
    idx = order_cut_indices(B, a_g, w, live, lo=3, hi=9)
    np.testing.assert_array_equal(idx, np.arange(n_g + 1))


def test_groups_outside_the_window_are_untouched():
    n_g = 12
    w = np.ones(n_g)
    a_g = np.full(n_g, 0.01)              # SNR far below 1 -> the DP wants to merge
    B = np.eye(n_g)
    live = np.ones(n_g, dtype=bool)
    idx = order_cut_indices(B, a_g, w, live, lo=4, hi=8)
    outside = [i for i in range(n_g + 1) if i <= 4 or i >= 8]
    assert set(outside) <= set(idx.tolist())


def test_absent_groups_keep_their_edges_and_split_the_runs():
    n_g = 9
    w = np.ones(n_g)
    a_g = np.full(n_g, 0.01)
    a_g[4] = 0.0
    B = np.eye(n_g)
    live = a_g != 0.0
    idx = order_cut_indices(B, a_g, w, live, lo=0, hi=n_g)
    assert 4 in idx and 5 in idx          # both edges of the dead group survive
    # and the merge on either side never crosses it
    assert not any(i < 4 < j for i, j in zip(idx[:-1], idx[1:]))


# --------------------------------------------------------------------------- #
# the DP itself
# --------------------------------------------------------------------------- #
def test_nothing_is_merged_when_every_group_is_already_resolved():
    n = 8
    cuts, (n_bad, loss, _) = solve_order(np.eye(n), np.full(n, 10.0), np.ones(n))
    np.testing.assert_array_equal(cuts, np.arange(n + 1))
    assert n_bad == 0 and loss == 0.0


def test_merging_happens_only_when_forced_and_fixes_what_it_can():
    n = 8
    B = np.full((n, n), 0.9) + 0.1 * np.eye(n)     # strongly correlated
    a_g = np.full(n, 0.5)                          # sd ~1 against mean 0.5: degenerate
    cuts, (n_bad, _loss, _) = solve_order(B, a_g, np.ones(n))
    assert len(cuts) - 1 < n                       # it did merge
    assert n_bad <= n                              # and never made things worse


def test_a_segment_that_only_reaches_snr_by_cancellation_is_refused():
    """Anti-correlated neighbours: averaging them shrinks sigma below independent.

    Singly the two are degenerate (SNR 0.5).  Merged they would reach SNR 7, but
    the gain comes entirely from cancellation -- Var(mean) = 0.005 against 0.5
    for independent averaging -- so the merge is refused and the degeneracy is
    declared instead of repaired.
    """
    B = np.array([[1.0, -0.99], [-0.99, 1.0]])
    a_g = np.array([0.5, 0.5])
    w = np.ones(2)
    cuts, (n_bad, _loss, _) = solve_order(B, a_g, w)
    np.testing.assert_array_equal(cuts, np.array([0, 1, 2]))   # left un-merged
    assert n_bad == 2                                          # and declared degenerate


def test_the_loss_is_extensive_so_a_single_span_is_never_free():
    """The fraction form collapsed a_6 to one group over 20 MeV; the extensive one cannot."""
    n = 10
    B = np.eye(n)
    a_g = np.full(n, 10.0)
    _, (_, loss_none, _) = solve_order(B, a_g, np.ones(n))
    T_all = solve_order(B, np.full(n, 10.0), np.ones(n))[0]
    assert loss_none == 0.0
    assert len(T_all) - 1 == n           # singletons cost 0, so they win outright


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #
def test_per_order_meshes_are_subsets_of_the_base_mesh():
    edges, cov, means = _case(n_g=14, seed=7, dead=[(3, 2), (9, 3)])
    meshes = per_order_meshes(edges, cov, means, L,
                              window_ev=(edges[2], edges[-3]))
    for l in range(1, L + 1):
        assert np.isin(meshes[l], edges).all()
        assert meshes[l][0] == edges[0] and meshes[l][-1] == edges[-1]


def test_the_driver_rejects_a_covariance_that_does_not_match_the_mesh():
    edges, cov, means = _case(n_g=6)
    with pytest.raises(ValueError, match="expected"):
        per_order_meshes(edges[:-1], cov, means, L)


def test_capping_every_sigma_at_snr_one_leaves_the_dp_nothing_to_do():
    """Run 97's defect, isolated: the criterion cannot fire on a capped object.

    `regularize_near_zero_relative_covariance` scales down every sigma whose SNR
    is below 1 until it is exactly 1. The DP merges only to repair SNR < 1, so
    running it after that step is running it on a matrix built to pass the test:
    it merges nothing, whatever the data said. The mesh must therefore be chosen
    on the covariance BEFORE the cap.
    """
    n_g, l_max = 12, 1
    edges = np.linspace(1.0e6, 4.0e6, n_g + 1)
    # Neighbours that agree, on a mean far below their own spread: every
    # singleton fails SNR, and averaging them is what rescues it.
    means = np.full(n_g, 0.2)
    rho = 0.995
    sd = np.full(n_g, 1.0)
    raw = np.outer(sd, sd) * (rho + (1.0 - rho) * np.eye(n_g))

    n_merged = len(per_order_meshes(edges, raw, means, l_max)[1]) - 1
    assert n_merged < n_g, "el objeto crudo tiene que fusionar algo"

    # The cap, applied as the pipeline applies it: rescale each row/column so no
    # diagonal sigma_rel exceeds 1.
    sd_cap = np.minimum(np.sqrt(np.diag(raw)), np.abs(means))
    s = sd_cap / np.sqrt(np.diag(raw))
    capped = raw * np.outer(s, s)
    assert np.all(np.sqrt(np.diag(capped)) <= np.abs(means) + 1e-12)

    n_capped = len(per_order_meshes(edges, capped, means, l_max)[1]) - 1
    assert n_capped == n_g, "sobre el objeto capado el DP no puede fusionar nada"
