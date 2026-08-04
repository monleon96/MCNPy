"""The multigroup collapse must honour the caller's valid-order rule.

WHY THIS FILE EXISTS. Phase 3 replaced the winner-take-all parameter mask
(``min(frozen_degree, max_order)``) with one keyed on the inclusion probability
``q_l``, and wired it through every covariance path on the *fine* grid. But
``perform_adaptive_multigroup_collapse`` rebuilt the legacy mask internally from
``nr.frozen_degree``, so the multigroup product silently kept winner-take-all.

Measured on the shipped files 2026-08-03 (roadmap §8.4), dead-parameter fraction
at l=6, run 84 -> run 85:

    fine MF34        92.3 %  ->   1.0 %     (the mixture arrived)
    multigroup MF34  81.6 %  ->  81.7 %     (it did not)

Since the chi2 scores the multigroup file, run 85 measured the mixture's
*absence*. These tests pin the fix so it cannot regress into silence again:
nothing here asserts a chi2 value, only that the caller's rule is the rule that
gets used, and that the default is still the legacy one so v2 stays frozen.
"""
import numpy as np
import pytest

from scripts.multigroup_collapse import (
    idx,
    perform_adaptive_multigroup_collapse,
)


class _NR:
    """Minimal stand-in for NominalFitResult."""

    def __init__(self, energy_index, frozen_degree, coeffs):
        self.energy_index = energy_index
        self.frozen_degree = frozen_degree
        self.nominal_coeffs = np.asarray(coeffs, dtype=float)
        self.interpolated = False
        self.has_data = True


class _EB:
    def __init__(self, e_mev, lo, hi):
        self.energy_mev = e_mev
        self.sigma_E_mev = 0.001
        self.bin_lower_mev = lo
        self.bin_upper_mev = hi


MAX_ORDER = 6
N_BINS = 12
# Every bin wins at degree 2, so the legacy rule keeps orders 1-2 and zeroes
# 3-6. That is exactly the situation Phase 3 exists to fix.
WINNER_DEGREE = 2


def _scenario():
    nominal, bins = [], []
    for i in range(N_BINS):
        lo = 1.0 + 0.01 * i
        nominal.append(_NR(i, WINNER_DEGREE, [0.5] + [0.1 / (l + 1) for l in range(MAX_ORDER)]))
        bins.append(_EB(lo + 0.005, lo, lo + 0.01))

    size = N_BINS * MAX_ORDER
    rng = np.random.default_rng(0)
    a = rng.normal(size=(size, size))
    cov = a @ a.T / size * 1e-4
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return cov, corr, nominal, bins


def _collapse(**kw):
    cov, corr, nominal, bins = _scenario()
    return perform_adaptive_multigroup_collapse(
        cov_matrix=cov, corr_matrix=corr, nominal_results=nominal,
        energy_bins=bins, max_order=MAX_ORDER, rho_min=0.90, **kw,
    )


def _orders_present(result):
    """Which Legendre orders survive as valid parameters in the collapsed mask."""
    vm = result.valid_mask_grouped
    n_groups = len(result.groups)
    return {
        l for l in range(1, MAX_ORDER + 1)
        if any(vm[idx(g, l, MAX_ORDER)] for g in range(n_groups))
    }


def test_default_is_the_legacy_winner_take_all_rule():
    """No `valid_orders_fn` => v2 behaviour, unchanged. This keeps v2 frozen."""
    got = _orders_present(_collapse())
    assert got == {1, 2}, (
        f"default rule must keep only orders <= frozen_degree ({WINNER_DEGREE}); got {got}"
    )


def test_caller_rule_overrides_frozen_degree():
    """The Phase-3 case: q_l keeps orders the winner never fitted."""
    got = _orders_present(_collapse(valid_orders_fn=lambda nr, mo: mo))
    assert got == {1, 2, 3, 4, 5, 6}, (
        f"caller rule asked for all {MAX_ORDER} orders; collapse kept {got}. "
        "This is the §8.4 regression: the mixture never reaches the shipped "
        "multigroup MF34."
    )


def test_caller_rule_is_clamped_to_max_order():
    """A rule returning more than max_order must not overflow the mask."""
    r = _collapse(valid_orders_fn=lambda nr, mo: mo + 5)
    assert _orders_present(r) == set(range(1, MAX_ORDER + 1))
    assert len(r.valid_mask_grouped) == len(r.groups) * MAX_ORDER


def test_caller_rule_can_be_more_restrictive_than_the_winner():
    """The override is a replacement, not a union — it must be able to shrink."""
    got = _orders_present(_collapse(valid_orders_fn=lambda nr, mo: 1))
    assert got == {1}, f"expected only order 1; got {got}"


def test_the_two_rules_actually_differ_on_this_scenario():
    """Guards the tests above: a scenario where both agree would prove nothing."""
    legacy = _orders_present(_collapse())
    phase3 = _orders_present(_collapse(valid_orders_fn=lambda nr, mo: mo))
    assert legacy != phase3, (
        "scenario is degenerate — the legacy and Phase-3 rules coincide, so "
        "these tests could not detect the regression they exist for"
    )


def test_valid_mask_grouped_shape_is_independent_of_the_rule():
    """The rule changes which slots are valid, never the grid."""
    a, b = _collapse(), _collapse(valid_orders_fn=lambda nr, mo: mo)
    assert len(a.groups) == len(b.groups)
    assert a.valid_mask_grouped.shape == b.valid_mask_grouped.shape
