"""Unit tests for the MF34 representation-study truncation helpers.

These guard the one thing the study depends on: that "ship fewer Legendre
orders" is implemented as a transformation of the library data and nothing
else, so that every other difference between modes is genuinely zero.
"""
import numpy as np
import pytest

from kika.cov import MF34CovMat
from scripts.precompute_chi2_representation import (
    truncate_mf4,
    truncate_mf34,
    truncate_library,
)


def _make_mf34(l_max: int, n_bins: int = 4) -> MF34CovMat:
    """Upper-triangular set of (l_r, l_c) blocks for l = 1..l_max."""
    cov = MF34CovMat()
    grid = list(np.linspace(1e6, 4e6, n_bins + 1))
    for l_r in range(1, l_max + 1):
        for l_c in range(l_r, l_max + 1):
            cov.add_matrix(
                isotope_row=26056, reaction_row=2, l_row=l_r,
                isotope_col=26056, reaction_col=2, l_col=l_c,
                energy_grid=grid,
                matrix=np.full((n_bins, n_bins), 0.01 * l_r * l_c),
                is_relative=True,
                frame="same-as-MF4",
            )
    return cov


def _make_library(l_max: int = 6, n_e: int = 3) -> dict:
    return {
        "energies_mf4_mev": np.linspace(1.0, 4.0, n_e),
        "coefficients": [
            np.arange(1, l_max + 1, dtype=float) * (0.1 * (i + 1))
            for i in range(n_e)
        ],
        "mf34": _make_mf34(l_max),
    }


# ── MF4 ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("l_max", [3, 4, 5])
def test_truncate_mf4_slices_every_energy(l_max):
    lib = _make_library(6)
    out = truncate_mf4(lib, l_max)
    assert all(len(c) == l_max for c in out["coefficients"])
    # surviving orders are untouched
    for before, after in zip(lib["coefficients"], out["coefficients"]):
        np.testing.assert_allclose(after, before[:l_max])


def test_truncate_mf4_is_a_noop_at_or_above_max_order():
    lib = _make_library(6)
    assert truncate_mf4(lib, 6) is lib
    assert truncate_mf4(lib, 9) is lib


def test_truncate_mf4_does_not_mutate_the_input():
    lib = _make_library(6)
    original = [c.copy() for c in lib["coefficients"]]
    truncate_mf4(lib, 3)
    for before, after in zip(original, lib["coefficients"]):
        np.testing.assert_array_equal(before, after)


def test_truncated_mf4_zero_pads_through_the_interpolator():
    """The forward model asks for 6 orders regardless; the dropped ones must
    come back as exact zeros, not as a shorter array."""
    from scripts.precompute_chi2_library_c0 import interp_a_l_to_energy

    lib = truncate_mf4(_make_library(6), 3)
    a = interp_a_l_to_energy(lib, 2.5, 6)
    assert a.shape == (6,)
    assert np.all(a[3:] == 0.0)
    assert np.any(a[:3] != 0.0)


# ── MF34 ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("l_max", [3, 4, 5])
def test_truncate_mf34_keeps_only_low_order_blocks(l_max):
    cov = _make_mf34(6)
    out = truncate_mf34(cov, l_max)
    assert len(out.matrices) == l_max * (l_max + 1) // 2
    assert max(out.l_rows) <= l_max
    assert max(out.l_cols) <= l_max
    # every surviving block is bit-identical
    kept = [
        i for i in range(len(cov.matrices))
        if cov.l_rows[i] <= l_max and cov.l_cols[i] <= l_max
    ]
    for j, i in enumerate(kept):
        np.testing.assert_array_equal(out.matrices[j], cov.matrices[i])
        assert out.is_relative[j] == cov.is_relative[i]
        assert out.energy_grids[j] == cov.energy_grids[i]


def test_truncate_mf34_drops_cross_order_blocks_touching_a_dropped_order():
    """(3,5) must go when l_max=4 — it is covariance *with* a dropped order."""
    out = truncate_mf34(_make_mf34(6), 4)
    pairs = set(zip(map(int, out.l_rows), map(int, out.l_cols)))
    assert (3, 4) in pairs
    assert (3, 5) not in pairs
    assert (5, 6) not in pairs


def test_truncate_mf34_noop_and_none():
    cov = _make_mf34(6)
    assert truncate_mf34(cov, 6) is cov
    assert truncate_mf34(None, 3) is None


def test_truncate_mf34_preserves_energy_unit():
    cov = _make_mf34(6)
    cov.energy_unit = "MeV"
    assert truncate_mf34(cov, 3).energy_unit == "MeV"


# ── Combined, and the sandwich ────────────────────────────────────────────────

def test_truncate_library_applies_both():
    lib = truncate_library(_make_library(6), 4, 3)
    assert all(len(c) == 4 for c in lib["coefficients"])
    assert max(lib["mf34"].l_rows) <= 3


def test_truncate_library_never_mutates_the_base():
    """Correctness-critical for the load-once sweep.

    Each ENDF is parsed once and the same library dict backs every mode, so a
    truncation that leaked back into the base would silently make mode N+1 a
    truncation of mode N. Re-truncating the same base to different depths must
    give the same answer every time, in any order.
    """
    base = _make_library(6)
    base_orders = [len(c) for c in base["coefficients"]]
    base_blocks = len(base["mf34"].matrices)

    a = truncate_library(base, 3, 3)
    b = truncate_library(base, 6, 6)     # full, after a deep truncation
    c = truncate_library(base, 5, 4)

    # the base is untouched
    assert [len(x) for x in base["coefficients"]] == base_orders
    assert len(base["mf34"].matrices) == base_blocks

    # and every view is what it asked for, independent of the others
    assert all(len(x) == 3 for x in a["coefficients"])
    assert max(a["mf34"].l_rows) <= 3
    assert all(len(x) == 6 for x in b["coefficients"])
    assert len(b["mf34"].matrices) == base_blocks
    assert all(len(x) == 5 for x in c["coefficients"])
    assert max(c["mf34"].l_rows) <= 4

    # repeating the first one reproduces it exactly
    a2 = truncate_library(base, 3, 3)
    for x, y in zip(a["coefficients"], a2["coefficients"]):
        np.testing.assert_array_equal(x, y)
    assert len(a2["mf34"].matrices) == len(a["mf34"].matrices)


def test_truncated_views_share_matrices_with_the_base():
    """The views must not copy the matrices — the fine MF34 is ~0.5 GB parsed
    and one copy per mode would defeat the point of loading once."""
    base = _make_library(6)
    view = truncate_library(base, 6, 4)
    kept = [
        i for i in range(len(base["mf34"].matrices))
        if base["mf34"].l_rows[i] <= 4 and base["mf34"].l_cols[i] <= 4
    ]
    for j, i in enumerate(kept):
        assert view["mf34"].matrices[j] is base["mf34"].matrices[i]


def test_dropped_orders_contribute_nothing_to_the_sandwich():
    """The point of the whole study: truncating MF34 must remove exactly the
    dropped orders' contribution to Sigma_eval and leave the rest alone."""
    from scripts.eval_covariance import build_mf34_block

    n = 5
    rng = np.random.default_rng(0)
    e_mev = np.linspace(1.5, 3.5, n)
    mu = np.linspace(-0.9, 0.9, n)
    c0 = np.full(n, 0.5)
    a_l = rng.normal(0.2, 0.05, size=(n, 6))

    full = _make_mf34(6)
    full_block = build_mf34_block(full, e_mev, mu, c0, a_l)
    trunc_block = build_mf34_block(truncate_mf34(full, 3), e_mev, mu, c0, a_l)

    # Truncating changes the answer — otherwise the study has nothing to measure.
    assert not np.allclose(trunc_block, full_block)

    # The property that matters: dropping is exactly not-having. A truncated
    # MF34 must give the same Sigma_eval as an MF34 that never carried the high
    # orders, so a `cov{L}` run is a faithful stand-in for shipping a smaller
    # file rather than an artifact of the filtering.
    native = build_mf34_block(_make_mf34(3), e_mev, mu, c0, a_l)
    np.testing.assert_allclose(trunc_block, native, rtol=0, atol=0)


def test_truncation_is_not_monotonic_in_the_propagated_variance():
    """Dropping high orders can RAISE the propagated DCS variance.

    Sigma_eval carries cross-order blocks (l_r != l_c) whose contribution is
    sens_r ⊗ block ⊗ sens_c, and P_l(mu) changes sign with l — so a cross term
    between a low and a high order can be negative and be *cancelling*
    variance. Remove the high order and the cancellation goes with it.

    Consequence for the study: "drop orders 4-6 to shrink the file" is not
    automatically a conservative choice, and a lower chi^2 after truncation is
    not evidence the orders were useless. The fine-vs-truncated comparison has
    to be read in both directions.
    """
    from scripts.eval_covariance import build_mf34_block

    n = 5
    rng = np.random.default_rng(0)
    e_mev = np.linspace(1.5, 3.5, n)
    mu = np.linspace(-0.9, 0.9, n)
    c0 = np.full(n, 0.5)
    a_l = rng.normal(0.2, 0.05, size=(n, 6))

    full = np.diag(build_mf34_block(_make_mf34(6), e_mev, mu, c0, a_l))
    trunc = np.diag(build_mf34_block(_make_mf34(3), e_mev, mu, c0, a_l))

    assert np.any(trunc > full), "expected at least one point to gain variance"
    assert np.any(trunc < full), "expected at least one point to lose variance"


def test_zeroing_the_central_kills_relative_high_order_covariance():
    """Justifies rejecting MF34_L_MAX > MF4_L_MAX: with a relative MF34 the
    sandwich scales each block by a_l, so covariance above the central
    truncation is inert."""
    from scripts.eval_covariance import build_mf34_block

    n = 4
    e_mev = np.linspace(1.5, 3.5, n)
    mu = np.linspace(-0.8, 0.8, n)
    c0 = np.full(n, 0.5)

    a_full = np.full((n, 6), 0.2)
    a_trunc = a_full.copy()
    a_trunc[:, 3:] = 0.0  # MF4 truncated to l <= 3

    cov6 = _make_mf34(6)
    with_zeroed_central = build_mf34_block(cov6, e_mev, mu, c0, a_trunc)
    cov3_only = build_mf34_block(truncate_mf34(cov6, 3), e_mev, mu, c0, a_trunc)
    np.testing.assert_allclose(with_zeroed_central, cov3_only, rtol=1e-12, atol=1e-18)
