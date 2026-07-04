"""Regression test for the MF34->MG Legendre-coefficient cache.

The shared ``_coeffs_cache`` in ``compute_base_cell_means`` is keyed only by
``(id(mf4_data), cm_to_lab_alpha)`` -- not by the requested order. A first call
for a low order populates the cache with ``extract_order == max_order`` (non-lab
path), so a later call for a higher order used to read the under-populated cache
and silently return zeros for every missing order. That zeroed the base-cell
means ``A_{l,i}`` used by the a_l-weighted relative-covariance collapse, killing
all Legendre orders above the first.

See ``MF34_to_MG`` / ``collapse_relative_covariance`` in
``kika/cov/multigroup/collapse.py``.
"""
import numpy as np

from kika.cov.multigroup.collapse import compute_base_cell_means, WeightingFunction


class _StubMF4:
    """Minimal MF4 stand-in returning a constant coefficient per Legendre order."""

    def __init__(self, energies, coeff_per_order):
        self.legendre_energies = np.asarray(energies, dtype=float)
        self._coeff_per_order = coeff_per_order

    def extract_legendre_coefficients(self, energies, max_legendre_order,
                                      out_of_range="zero", **kwargs):
        energies = np.asarray(energies, dtype=float)
        return {
            l: np.full(energies.shape, self._coeff_per_order.get(l, 0.0))
            for l in range(max_legendre_order + 1)
        }


def _means(mf4, grid, orders, cache):
    return compute_base_cell_means(
        grid, mf4, orders,
        WeightingFunction.lethargy, WeightingFunction.lethargy_antiderivative,
        _coeffs_cache=cache,
    )


def test_shared_cache_preserves_higher_orders():
    """A low-order call must not poison a later higher-order call sharing the cache."""
    E = np.array([1e3, 1e4, 1e5, 1e6, 2e7], dtype=float)
    grid = np.array([1e3, 1e5, 2e7], dtype=float)  # two multigroup cells
    coeffs = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
    mf4 = _StubMF4(E, coeffs)

    cache = {}
    m1 = _means(mf4, grid, [1], cache)   # populates cache at extract_order == 1
    m3 = _means(mf4, grid, [3], cache)   # reuses the SAME (formerly stale) cache

    # Order 1 is correct either way.
    assert np.allclose(m1[1], 0.8, atol=1e-9)
    # With the bug, order 3 came back identically zero. With the fix it must
    # recover the true constant coefficient (0.4).
    assert np.allclose(m3[3], 0.4, atol=1e-9), f"higher order zeroed: {m3[3]}"


def test_single_call_all_orders_consistent_with_incremental():
    """Requesting all orders at once must match requesting them incrementally."""
    E = np.array([1e3, 1e4, 1e5, 1e6, 2e7], dtype=float)
    grid = np.array([1e3, 1e5, 2e7], dtype=float)
    coeffs = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
    mf4 = _StubMF4(E, coeffs)

    all_at_once = _means(mf4, grid, [1, 2, 3, 4], {})
    cache = {}
    incremental = {}
    for l in (1, 2, 3, 4):
        incremental[l] = _means(mf4, grid, [l], cache)[l]

    for l in (1, 2, 3, 4):
        assert np.allclose(all_at_once[l], incremental[l], atol=1e-9), l
