"""Tests for the MF33 build path: folding, adaptive collapse and the NaN guards.

These cover the four failure modes that let run 81 write no MF33 at all:
a zero File-3 central inside the resolved resonance range, the sanitized matrix
being discarded by ``collapse_mf33_covariance_to_grid``, the resulting NaNs
reaching ``np.linalg.eigvalsh`` as an opaque LAPACK error, and the box-average
denominator being the wrong quantity to divide by in the first place.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.mf33_build import fold_xs_over_bins  # noqa: E402
from scripts.mf33_diagnostics import bin_average_xs  # noqa: E402
from scripts.multigroup_collapse import (  # noqa: E402
    collapse_mf33_covariance_to_grid,
    perform_mf33_multigroup_collapse,
)


@dataclass
class _Bin:
    """Minimal EnergyBinInfo stand-in (only the fields the folding needs)."""
    index: int
    energy_mev: float
    sigma_E_mev: float
    bin_lower_mev: float
    bin_upper_mev: float


def _bins(energies_mev, sigma_E_mev, width_mev=0.001):
    half = 0.5 * width_mev
    return [
        _Bin(i, float(e), float(s), float(e) - half, float(e) + half)
        for i, (e, s) in enumerate(zip(energies_mev, np.broadcast_to(sigma_E_mev, np.shape(energies_mev))))
    ]


# --- folding ---------------------------------------------------------------

def test_fold_over_bins_flat_xs_is_identity():
    """A flat cross section folds to itself whatever the resolution."""
    e = np.linspace(0.5e6, 5.0e6, 400)
    xs = np.full_like(e, 2.5)
    out = fold_xs_over_bins(e, xs, _bins([1.0, 2.0, 3.0], 0.02))
    np.testing.assert_allclose(out, 2.5, rtol=1e-8)


def test_fold_over_bins_matches_gauss_hermite_on_a_gaussian_free_case():
    """A linear cross section folds to its value at the centroid (odd moments cancel)."""
    e = np.linspace(0.5e6, 5.0e6, 4000)
    xs = 1.0 + 3.0e-7 * e          # linear in E
    out = fold_xs_over_bins(e, xs, _bins([2.0], 0.05))
    np.testing.assert_allclose(out, 1.0 + 3.0e-7 * 2.0e6, rtol=1e-6)


def test_fold_over_bins_smooths_a_narrow_resonance():
    """A resonance narrower than sigma_E is broadened, not sampled at the peak."""
    e = np.linspace(1.9e6, 2.1e6, 20001)
    xs = 1.0 + 50.0 * np.exp(-0.5 * ((e - 2.0e6) / 2.0e3) ** 2)  # 2 keV wide
    peak = fold_xs_over_bins(e, xs, _bins([2.0], 0.0005))   # 0.5 keV kernel
    broad = fold_xs_over_bins(e, xs, _bins([2.0], 0.030))   # 30 keV kernel
    assert peak[0] > broad[0] > 1.0
    assert peak[0] > 40.0        # narrow kernel still sees the resonance
    assert broad[0] < 5.0        # wide kernel washes it out


def test_fold_over_bins_falls_back_to_box_average_at_zero_resolution():
    """sigma_E <= 0 must give a box average, not a point sample.

    fold_xs_over_resolution degenerates to np.interp at zero width, which on a
    resonant cross section is strictly worse than averaging over the bin.
    """
    e = np.linspace(1.99e6, 2.01e6, 4001)
    xs = 1.0 + 50.0 * np.exp(-0.5 * ((e - 2.0005e6) / 3.0e2) ** 2)
    b = _Bin(0, 2.0, 0.0, 1.9995, 2.0005)
    got = fold_xs_over_bins(e, xs, [b])
    expected = bin_average_xs(e, xs, np.array([1.9995e6, 2.0005e6]))
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    assert not np.isclose(got[0], np.interp(2.0e6, e, xs))


def test_fold_over_bins_reports_fallback_count():
    class _Rec:
        def __init__(self):
            self.msgs = []

        def warning(self, m, **_kw):
            self.msgs.append(m)

        def info(self, m, **_kw):
            pass

    e = np.linspace(0.5e6, 5.0e6, 200)
    xs = np.full_like(e, 1.0)
    rec = _Rec()
    fold_xs_over_bins(e, xs, _bins([1.0, 2.0], [0.0, 0.02]), logger=rec)
    assert any("sigma_E <= 0" in m for m in rec.msgs)


# --- zero-central guards ---------------------------------------------------

def _block_corr_cov(n, block, rho, sigma):
    """Absolute covariance with rho inside contiguous blocks, 0 across them."""
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = 1.0 if i == j else (rho if i // block == j // block else 0.0)
    s = np.broadcast_to(np.asarray(sigma, dtype=float), (n,))
    return corr * np.outer(s, s)


def test_collapse_to_grid_zeroes_rows_with_no_central_and_stays_finite():
    """A target group whose native means are zero must not yield NaN.

    This is the exact shape of the run-81 failure: File 3 is zero inside the
    resolved resonance range, so the first groups have no central to divide by.
    """
    n = 8
    native = np.linspace(0.0, 8.0, n + 1) * 1e5
    cov = _block_corr_cov(n, 4, 0.9, 0.1)
    means = np.array([0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    target = native[::2]

    out = collapse_mf33_covariance_to_grid(
        native_grid_ev=native, native_cov=cov, target_grid_ev=target,
        native_means=means, is_relative=True,
    )
    assert np.all(np.isfinite(out)), "zero-central groups must not produce NaN"
    assert np.allclose(out[0], 0.0) and np.allclose(out[:, 0], 0.0)
    assert np.any(out[1:, 1:] != 0.0), "groups with a central must survive"


def test_collapse_to_grid_returns_sanitized_matrix():
    """Non-finite input must not survive the PSD check into the return value."""
    n = 6
    native = np.linspace(0.0, 6.0, n + 1) * 1e5
    cov = _block_corr_cov(n, 3, 0.8, 0.1)
    cov[2, :] = np.nan
    cov[:, 2] = np.nan
    out = collapse_mf33_covariance_to_grid(
        native_grid_ev=native, native_cov=cov, target_grid_ev=native[::2],
        native_means=None, is_relative=False,
    )
    assert np.all(np.isfinite(out))


# The merge-side guard lives with the other merge tests, in
# kika/endf/tests/test_mf33_writer.py — it needs a host that actually carries an
# MF33 MT2 section, otherwise merge_mf33_covariance_into_host short-circuits to
# create_mf33_from_covariance and that function's own guard fires first.


# --- adaptive collapse -----------------------------------------------------

def test_mf33_collapse_groups_follow_rho_min():
    """A looser correlation gate must merge at least as aggressively."""
    n = 12
    cov = _block_corr_cov(n, 4, 0.95, 0.1)
    bins = _bins(np.linspace(1.0, 1.011, n), 0.005)
    means = np.full(n, 2.0)

    tight = perform_mf33_multigroup_collapse(
        cov_abs_fine=cov, means_fine=means, energy_bins=bins, rho_min=0.99,
    )
    loose = perform_mf33_multigroup_collapse(
        cov_abs_fine=cov, means_fine=means, energy_bins=bins, rho_min=0.90,
    )
    assert len(loose.groups) <= len(tight.groups)
    assert len(tight.groups) == n, "rho_min above the block correlation merges nothing"
    assert len(loose.groups) == 3, "rho_min below it recovers the three blocks"


def test_mf33_collapse_aggregator_is_width_weighted_mean():
    """A0 @ means must reproduce the width-weighted average per group."""
    n = 6
    cov = _block_corr_cov(n, 3, 0.99, 0.1)
    bins = _bins(np.linspace(1.0, 1.005, n), 0.005)
    means = np.arange(1.0, 1.0 + n)

    res = perform_mf33_multigroup_collapse(
        cov_abs_fine=cov, means_fine=means, energy_bins=bins, rho_min=0.90,
    )
    for g, idxs in enumerate(res.groups):
        np.testing.assert_allclose(res.means_grouped[g], np.mean(means[idxs]), rtol=1e-9)
    np.testing.assert_allclose(res.aggregation_matrix.sum(axis=1), 1.0, rtol=1e-12)


def test_mf33_collapse_relative_is_absolute_over_outer_means():
    n = 6
    cov = _block_corr_cov(n, 3, 0.99, 0.1)
    bins = _bins(np.linspace(1.0, 1.005, n), 0.005)
    means = np.full(n, 3.0)
    res = perform_mf33_multigroup_collapse(
        cov_abs_fine=cov, means_fine=means, energy_bins=bins, rho_min=0.90,
    )
    expected = res.cov_abs_grouped / np.outer(res.means_grouped, res.means_grouped)
    np.testing.assert_allclose(res.cov_rel_grouped, expected, rtol=1e-9, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(res.cov_rel_grouped)) > -1e-10


def test_mf33_collapse_boundaries_are_contiguous_and_span_the_grid():
    n = 9
    cov = _block_corr_cov(n, 3, 0.99, 0.1)
    bins = _bins(np.linspace(1.0, 1.008, n), 0.005)
    res = perform_mf33_multigroup_collapse(
        cov_abs_fine=cov, means_fine=np.full(n, 2.0), energy_bins=bins, rho_min=0.90,
    )
    b = res.group_boundaries_mev
    assert len(b) == len(res.groups) + 1
    assert np.all(np.diff(b) > 0)
    assert b[0] == pytest.approx(bins[0].bin_lower_mev)
    assert b[-1] == pytest.approx(bins[-1].bin_upper_mev)


def test_mf33_collapse_rejects_shape_mismatch():
    bins = _bins(np.linspace(1.0, 1.004, 5), 0.005)
    with pytest.raises(ValueError, match="doesn't match"):
        perform_mf33_multigroup_collapse(
            cov_abs_fine=np.eye(4), means_fine=np.ones(5), energy_bins=bins,
        )
    with pytest.raises(ValueError, match="doesn't match"):
        perform_mf33_multigroup_collapse(
            cov_abs_fine=np.eye(5), means_fine=np.ones(4), energy_bins=bins,
        )


# --- MT1 rebuild from the partials ------------------------------------------

def _stub_partial_sources(monkeypatch, xs_by_mt, cov_by_mt, grid_ev):
    """Stub the PENDF and host-MF33 reads `build_mt1_from_partials` depends on."""
    import scripts.mf33_build as mb

    @dataclass
    class _Sec:
        energies: np.ndarray
        cross_sections: np.ndarray

    def _sections(_path):
        # Piecewise-constant per group, sampled just inside each edge so the
        # group average reproduces the requested value exactly.
        out = {}
        for mt, vals in xs_by_mt.items():
            e, x = [], []
            for i, v in enumerate(vals):
                e += [grid_ev[i], grid_ev[i + 1]]
                x += [float(v), float(v)]
            out[mt] = _Sec(np.asarray(e, float), np.asarray(x, float))
        return out

    class _Decoded:
        def __init__(self, m, g):
            self.matrices, self.energy_grids = [m], [g]

    class _Section:
        def __init__(self, mt):
            self._mt, self._mtl = mt, 0

        def to_xs_covmat(self):
            return _Decoded(cov_by_mt[self._mt], grid_ev)

    class _File:
        sections = {mt: _Section(mt) for mt in cov_by_mt}

    class _Endf:
        def get_file(self, _n):
            return _File()

    monkeypatch.setattr(mb, "read_pendf_mf3_sections", _sections)
    monkeypatch.setattr("kika.endf.read_endf", lambda *a, **k: _Endf())


def test_mt1_rebuild_with_a_single_partial_returns_the_partial(monkeypatch):
    """If elastic IS the total, the rebuilt relative total is the elastic one."""
    from scripts.mf33_build import build_mt1_from_partials

    grid = np.array([1.0e6, 2.0e6, 3.0e6])
    elastic = np.array([[0.04, 0.01], [0.01, 0.09]])
    _stub_partial_sources(
        monkeypatch,
        xs_by_mt={1: [3.0, 4.0], 2: [3.0, 4.0]},
        cov_by_mt={2: elastic},
        grid_ev=grid,
    )
    cov, diag = build_mt1_from_partials(
        "unused.endf", elastic_rel=elastic, grid_ev=grid, pendf_path="unused.pendf",
    )
    np.testing.assert_allclose(cov, elastic, atol=1e-12)
    assert diag["partials"] == [2]


def test_mt1_rebuild_sums_partial_variances_with_zero_cross_terms(monkeypatch):
    """Var_tot = (s_el^2 V_el + s_4^2 V_4) / s_tot^2, exactly — no cross terms."""
    from scripts.mf33_build import build_mt1_from_partials

    grid = np.array([1.0e6, 2.0e6, 3.0e6])
    s_el, s_4 = np.array([3.0, 3.0]), np.array([1.0, 1.0])
    elastic = np.diag([0.04, 0.04])
    inelastic = np.diag([0.25, 0.25])
    _stub_partial_sources(
        monkeypatch,
        xs_by_mt={1: s_el + s_4, 2: s_el, 4: s_4},
        cov_by_mt={2: elastic, 4: inelastic},
        grid_ev=grid,
    )
    cov, diag = build_mt1_from_partials(
        "unused.endf", elastic_rel=elastic, grid_ev=grid, pendf_path="unused.pendf",
    )
    expected = (s_el ** 2 * 0.04 + s_4 ** 2 * 0.25) / (s_el + s_4) ** 2
    np.testing.assert_allclose(np.diag(cov), expected, rtol=1e-12)
    assert diag["partials"] == [2, 4]
    assert diag["budget_max_rel_residual"] < 1e-12


def test_mt1_rebuild_is_psd_when_the_partials_are(monkeypatch):
    """A sum of PSD blocks scaled by positive diagonals cannot lose PSD."""
    from scripts.mf33_build import build_mt1_from_partials

    n = 12
    grid = np.linspace(1.0e6, 4.0e6, n + 1)
    rng = np.random.default_rng(7)
    a = rng.normal(size=(n, n))
    elastic = (a @ a.T) / n * 0.01
    b = rng.normal(size=(n, n))
    inelastic = (b @ b.T) / n * 0.04
    s_el = np.full(n, 3.0)
    s_4 = np.linspace(0.5, 1.5, n)
    _stub_partial_sources(
        monkeypatch,
        xs_by_mt={1: s_el + s_4, 2: s_el, 4: s_4},
        cov_by_mt={2: elastic, 4: inelastic},
        grid_ev=grid,
    )
    cov, _ = build_mt1_from_partials(
        "unused.endf", elastic_rel=elastic, grid_ev=grid, pendf_path="unused.pendf",
    )
    assert np.allclose(cov, cov.T, atol=1e-14)
    assert np.linalg.eigvalsh(cov).min() >= -1e-12 * np.diag(cov).max()


def test_project_lb5_is_exact_on_a_refinement():
    """LB=5 is piecewise-constant, so refining a grid must copy values."""
    from scripts.mf33_build import _project_lb5

    src = np.array([1.0e6, 2.0e6, 3.0e6])
    dst = np.array([1.0e6, 1.5e6, 2.0e6, 2.5e6, 3.0e6])
    m = np.array([[1.0, 2.0], [2.0, 4.0]])
    out = _project_lb5(m, src, dst)
    expected = np.array([
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [2.0, 2.0, 4.0, 4.0],
        [2.0, 2.0, 4.0, 4.0],
    ])
    np.testing.assert_allclose(out, expected)
