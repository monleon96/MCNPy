"""Unit tests for :mod:`kika.processing.group_averaging`.

These are pure-numerical tests that need no external data or NJOY
install, so they run on every CI machine.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.energy_grids import equal_lethargy_grid
from kika.processing import resonance_group_average


# ---------------------------------------------------------------------------
# equal_lethargy_grid
# ---------------------------------------------------------------------------

def test_equal_lethargy_grid_edges_and_count():
    edges = equal_lethargy_grid(1.0, 1000.0, 3)
    assert edges.shape == (4,)
    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(1000.0)
    # Constant ratio between adjacent edges
    ratios = edges[1:] / edges[:-1]
    assert np.allclose(ratios, ratios[0])
    assert ratios[0] == pytest.approx(10.0)


def test_equal_lethargy_grid_rejects_bad_inputs():
    with pytest.raises(ValueError):
        equal_lethargy_grid(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        equal_lethargy_grid(-1.0, 1.0, 5)
    with pytest.raises(ValueError):
        equal_lethargy_grid(1.0, 1.0, 5)
    with pytest.raises(ValueError):
        equal_lethargy_grid(1.0, 10.0, 0)


# ---------------------------------------------------------------------------
# resonance_group_average
# ---------------------------------------------------------------------------

def test_constant_xs_returns_same_value_lethargy():
    """Constant sigma(E) = c must give group-averaged c regardless of phi."""
    energies = np.logspace(0, 4, 2000)  # 1 eV -> 10 keV
    sigma = np.full_like(energies, 3.14)
    bins = equal_lethargy_grid(1.0, 1e4, 20)

    _, avgs = resonance_group_average(energies, sigma, bins, weighting="lethargy")
    assert np.allclose(avgs, 3.14, rtol=1e-6)


def test_constant_xs_returns_same_value_constant():
    energies = np.logspace(0, 4, 2000)
    sigma = np.full_like(energies, 7.0)
    bins = equal_lethargy_grid(1.0, 1e4, 10)

    _, avgs = resonance_group_average(energies, sigma, bins, weighting="constant")
    assert np.allclose(avgs, 7.0, rtol=1e-6)


def test_sigma_equals_E_with_lethargy_weight_matches_analytic():
    """For sigma(E) = E and phi(E) = 1/E, the group average over [a, b] is
    (b - a) / ln(b/a)."""
    energies = np.logspace(0, 4, 5000)
    sigma = energies.copy()
    # Use a coarse grid so we exercise the group-by-group path
    bins = np.array([1.0, 10.0, 100.0, 1000.0, 1e4])

    _, avgs = resonance_group_average(energies, sigma, bins, weighting="lethargy")

    expected = np.array([
        (bins[g + 1] - bins[g]) / np.log(bins[g + 1] / bins[g])
        for g in range(bins.size - 1)
    ])
    # Trapezoidal error on a piecewise-linear integrand is small but not zero
    # at this grid density; 1e-3 relative is comfortable.
    assert np.allclose(avgs, expected, rtol=1e-3)


def test_sigma_equals_E_with_constant_weight_matches_analytic():
    """For sigma(E) = E and phi = 1, the average over [a, b] is (a + b) / 2."""
    energies = np.linspace(1.0, 1000.0, 5000)
    sigma = energies.copy()
    bins = np.array([1.0, 100.0, 500.0, 1000.0])

    _, avgs = resonance_group_average(energies, sigma, bins, weighting="constant")
    expected = 0.5 * (bins[:-1] + bins[1:])
    assert np.allclose(avgs, expected, rtol=1e-6)


def test_group_outside_pointwise_range_is_nan():
    energies = np.array([1.0, 10.0, 100.0])
    sigma = np.array([2.0, 2.0, 2.0])
    bins = np.array([0.001, 0.01, 1.0, 100.0, 1000.0])

    _, avgs = resonance_group_average(energies, sigma, bins, weighting="lethargy")
    # First two bins end at or below 1.0 (the pointwise start).
    # The first bin [0.001, 0.01] is entirely below, so it's NaN.
    # The second bin [0.01, 1.0] clips to [1.0, 1.0] which is empty → NaN.
    assert np.isnan(avgs[0])
    assert np.isnan(avgs[1])
    # Bin [1, 100] is fully covered.
    assert avgs[2] == pytest.approx(2.0, rel=1e-6)
    # Last bin [100, 1000] clips to a single point → NaN (empty integration width).
    assert np.isnan(avgs[3])


def test_returns_input_edges_unchanged():
    energies = np.linspace(1.0, 100.0, 100)
    sigma = np.full_like(energies, 1.0)
    bins = np.array([1.0, 10.0, 100.0])

    edges, _ = resonance_group_average(energies, sigma, bins)
    assert np.array_equal(edges, bins)


def test_rejects_non_monotonic_inputs():
    energies = np.array([1.0, 10.0, 5.0, 100.0])
    sigma = np.array([1.0, 1.0, 1.0, 1.0])
    bins = np.array([1.0, 100.0])
    with pytest.raises(ValueError):
        resonance_group_average(energies, sigma, bins)

    energies = np.array([1.0, 10.0, 100.0])
    bins = np.array([1.0, 50.0, 30.0])
    with pytest.raises(ValueError):
        resonance_group_average(energies, np.ones_like(energies), bins)


def test_rejects_unknown_weighting():
    energies = np.array([1.0, 10.0])
    sigma = np.array([1.0, 1.0])
    bins = np.array([1.0, 10.0])
    with pytest.raises(ValueError):
        resonance_group_average(energies, sigma, bins, weighting="maxwellian")


# ---------------------------------------------------------------------------
# detect_resonance_bounds
# ---------------------------------------------------------------------------

class _FakeResonance:
    def __init__(self):
        self.energy = 1.0


class _FakeLBlock:
    def __init__(self, with_resonances=True):
        self.resonances = [_FakeResonance()] if with_resonances else []
        self.j_states = []


class _FakeParameters:
    def __init__(self, with_data=True):
        self.l_values = [_FakeLBlock(with_resonances=with_data)]


class _FakeRange:
    def __init__(self, el, eh, lru=1, with_data=True):
        self.el = el
        self.eh = eh
        self.lru = lru
        self.parameters = _FakeParameters(with_data=with_data) if with_data else None


class _FakeIsotope:
    def __init__(self, ranges):
        self.energy_ranges = ranges


class _FakeMT151:
    def __init__(self, isotopes):
        self.isotopes = isotopes


def test_detect_bounds_spans_multiple_ranges():
    from kika.endf.processing import detect_resonance_bounds

    mt151 = _FakeMT151([
        _FakeIsotope([
            _FakeRange(1e-5, 100.0, lru=1),
            _FakeRange(100.0, 2.25e3, lru=2),
        ]),
    ])
    bounds = detect_resonance_bounds(mt151)
    assert bounds == (1e-5, 2.25e3)


def test_detect_bounds_skips_scattering_only_and_stubs():
    from kika.endf.processing import detect_resonance_bounds

    mt151 = _FakeMT151([
        _FakeIsotope([
            _FakeRange(1e-5, 1e6, lru=0, with_data=False),  # scattering radius only
            _FakeRange(1.0, 100.0, lru=1, with_data=False),  # stub, no data
        ]),
    ])
    assert detect_resonance_bounds(mt151) is None


def test_detect_bounds_handles_none():
    from kika.endf.processing import detect_resonance_bounds
    assert detect_resonance_bounds(None) is None
