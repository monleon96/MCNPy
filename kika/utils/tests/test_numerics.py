"""Tests for the shared numerical primitives and the unified TOF resolution.

These pin the two properties the folding/TOF unification was done for:

* every entry point computes the *same* sigma_E, under both conventions;
* the Gaussian fold integrates the interpolant, so its answer does not depend
  on how densely the input happens to be sampled — which is exactly what the
  old point-weighted implementation got wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika._constants import FWHM_TO_SIGMA
from kika.utils.numerics import (
    average_over_intervals,
    fold_tabulated,
    gauss_hermite_nodes,
)
from kika.utils.energy_folding import tof_energy_resolution


# --- fold_tabulated --------------------------------------------------------

def test_fold_flat_is_identity():
    x = np.linspace(0.0, 10.0, 101)
    y = np.full_like(x, 3.0)
    assert fold_tabulated(x, y, 5.0, 1.0) == pytest.approx(3.0, rel=1e-12)


def test_fold_linear_returns_value_at_centroid():
    """Odd moments of a symmetric kernel vanish, so a line folds to y(x0)."""
    x = np.linspace(0.0, 10.0, 1001)
    y = 2.0 + 0.5 * x
    assert fold_tabulated(x, y, 5.0, 0.7) == pytest.approx(2.0 + 0.5 * 5.0, rel=1e-9)


def test_fold_quadratic_matches_analytic_second_moment():
    """For y = x^2 the exact fold is x0^2 + sigma^2."""
    x = np.linspace(-20.0, 20.0, 40001)
    y = x ** 2
    got = fold_tabulated(x, y, 1.0, 2.0, n_nodes=20)
    assert got == pytest.approx(1.0 ** 2 + 2.0 ** 2, rel=1e-4)


def test_fold_is_insensitive_to_input_sampling_density():
    """The regression the unification exists to prevent.

    A Gaussian-weighted average of tabulated *points* changes when the same
    function is resampled more densely near its peak.  Integrating the
    interpolant does not.
    """
    peak = lambda t: 1.0 + 40.0 * np.exp(-0.5 * ((t - 5.0) / 0.10) ** 2)
    uniform = np.linspace(0.0, 10.0, 4001)
    # Same function, but heavily oversampled around the peak.
    clustered = np.unique(np.concatenate([uniform, np.linspace(4.5, 5.5, 8000)]))

    def point_weighted(t):
        w = np.exp(-0.5 * ((t - 5.0) / 1.0) ** 2)
        return float(np.sum(w * peak(t)) / np.sum(w))

    quad_a = fold_tabulated(uniform, peak(uniform), 5.0, 1.0)
    quad_b = fold_tabulated(clustered, peak(clustered), 5.0, 1.0)
    pw_a, pw_b = point_weighted(uniform), point_weighted(clustered)

    quad_shift = abs(quad_b - quad_a) / quad_a
    pw_shift = abs(pw_b - pw_a) / pw_a

    # The quadrature is not perfectly invariant — the interpolant itself
    # improves with sampling — but it is orders of magnitude steadier than
    # weighting the points, which is the property that matters.
    assert quad_shift < 1e-4
    assert pw_shift > 0.05
    assert pw_shift / quad_shift > 1000


def test_fold_zero_sigma_is_interpolation():
    x = np.linspace(0.0, 10.0, 101)
    y = x ** 2
    assert fold_tabulated(x, y, 3.3, 0.0) == pytest.approx(np.interp(3.3, x, y))


def test_fold_vectorises_over_centroids():
    x = np.linspace(0.0, 10.0, 501)
    y = np.sin(x)
    centroids = np.array([2.0, 4.0, 6.0])
    got = fold_tabulated(x, y, centroids, 0.3)
    assert got.shape == (3,)
    for i, c in enumerate(centroids):
        assert got[i] == pytest.approx(fold_tabulated(x, y, float(c), 0.3))


# --- gauss_hermite_nodes ---------------------------------------------------

def test_nodes_weights_sum_to_one_and_centre_on_x0():
    pts, w = gauss_hermite_nodes(5.0, 2.0, n_nodes=16)
    assert w.sum() == pytest.approx(1.0, rel=1e-12)
    assert float(np.sum(w * pts)) == pytest.approx(5.0, rel=1e-9)
    # Second central moment recovers sigma^2.
    assert float(np.sum(w * (pts - 5.0) ** 2)) == pytest.approx(4.0, rel=1e-9)


def test_nodes_degenerate_at_zero_sigma():
    pts, w = gauss_hermite_nodes(1.5, 0.0)
    np.testing.assert_allclose(pts, [1.5])
    np.testing.assert_allclose(w, [1.0])


# --- average_over_intervals ------------------------------------------------

def test_interval_average_of_a_line_is_the_midpoint_value():
    x = np.linspace(0.0, 10.0, 1001)
    y = 3.0 * x + 1.0
    edges = np.array([0.0, 2.0, 5.0, 10.0])
    got = average_over_intervals(x, y, edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    np.testing.assert_allclose(got, 3.0 * mids + 1.0, rtol=1e-9)


def test_interval_average_does_not_step_over_a_narrow_feature():
    """The native points are unioned into the sub-grid, so spikes survive."""
    x = np.unique(np.concatenate([
        np.linspace(0.0, 10.0, 101), np.linspace(4.999, 5.001, 201),
    ]))
    y = np.where(np.abs(x - 5.0) < 0.001, 100.0, 1.0)
    got = average_over_intervals(x, y, np.array([0.0, 10.0]), n_sub=5)
    assert got[0] > 1.0, "narrow spike was skipped by the uniform sub-grid"


# --- TOF resolution --------------------------------------------------------

def test_fwhm_and_sigma_conventions_differ_by_the_gaussian_factor():
    kw = dict(flight_path_m=27.037, delta_t_ns=5.0)
    as_fwhm = tof_energy_resolution(2.0, delta_t_is_fwhm=True, **kw)
    as_sigma = tof_energy_resolution(2.0, delta_t_is_fwhm=False, **kw)
    assert as_sigma / as_fwhm == pytest.approx(FWHM_TO_SIGMA, rel=1e-12)


def test_resolution_scales_as_sqrt_energy():
    """sigma_E / E = 2 dt / t and t ~ 1/sqrt(E), so sigma_E ~ E^{3/2}."""
    kw = dict(flight_path_m=27.037, delta_t_ns=5.0, delta_t_is_fwhm=True)
    s1 = tof_energy_resolution(1.0, **kw)
    s4 = tof_energy_resolution(4.0, **kw)
    assert s4 / s1 == pytest.approx(4.0 ** 1.5, rel=1e-9)


def test_all_entry_points_agree():
    """One formula, five call sites — the point of the unification."""
    from kika.ace.classes.angular_distribution.container import (
        _compute_energy_resolution_tof as ace_fn,
    )
    from scripts.resample_AD import compute_energy_resolution_tof as pipeline_fn
    from scripts.tof_parameters import compute_sigma_E_direct

    for energy in (0.847, 2.0, 4.0):
        for fwhm in (True, False):
            core = tof_energy_resolution(
                energy, flight_path_m=27.037, delta_t_ns=5.0, delta_t_is_fwhm=fwhm,
            )
            assert ace_fn(energy, 27.037, 5.0, delta_t_is_fwhm=fwhm) == core
            assert pipeline_fn(
                E_mev=energy, delta_t_ns=5.0, flight_path_m=27.037,
                delta_t_is_fwhm=fwhm,
            ) == core
            # compute_sigma_E_direct applies a 1 keV floor; compare above it.
            assert compute_sigma_E_direct(
                energy, 27.037, 5.0, delta_t_is_fwhm=fwhm,
            ) == pytest.approx(max(core, 1e-3))


def test_zero_energy_is_zero():
    assert tof_energy_resolution(0.0, flight_path_m=27.0, delta_t_ns=5.0) == 0.0
