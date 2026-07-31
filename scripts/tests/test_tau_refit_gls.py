"""Tests for ``tau_refit_use_gls`` — the solver used by the τ-IRLS refit.

Background. ``use_gls_kernel`` puts the IC model-selection scan, the initial
nominal fit and the post-τ rescan on the block-correlated kernel
Σ = D + u uᵀ + v vᵀ. The coefficient refit *inside* the τ-IRLS loop was left
on diagonal WLS, fed ``sigma_refit = √(σ_eff² + σ_sys²)`` — σ_sys folded into
the fit weights as if it were uncorrelated. That loop's ``coeffs0`` is the
shipped nominal (the post-τ rescan replaces it only when the winning *degree*
changes), so the historical default ships a WLS central whose degree was
chosen by GLS scores.

``tau_refit_use_gls`` makes the refit use the same kernel as the scan. It
defaults to False so the frozen v2 pipeline is bit-unchanged.

The load-bearing correctness property is the one tested hardest here: the GLS
branch must receive ``sigma_eff`` (= τ·σ_stat) and carry σ_sys as rank-1
structure. Handing it ``sigma_refit`` would count σ_sys twice — once on the
diagonal and once in the rank-1 blocks — which is silent and would only show
up as mysteriously over-inflated uncertainties downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.legendre import legvander

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import scripts.resample_AD as rad  # noqa: E402
from scripts.resample_AD import sample_legendre_coefficients  # noqa: E402


def _two_experiment_frame(seed: int = 7,
                          sys_rel: float = 0.08,
                          norm_offset: float = 0.12) -> pd.DataFrame:
    """Two experiments over the same angular range, offset in normalisation
    and with enough scatter that the band-discrepancy τ actually fires.

    A normalisation disagreement is exactly the situation the rank-1 block
    exists for, so this is the configuration where WLS and GLS refits differ.
    """
    rng = np.random.default_rng(seed)
    mu = np.concatenate([np.linspace(-0.95, 0.95, 15),
                         np.linspace(-0.9, 0.9, 15)])
    true_c = np.array([1.0, 0.35, 0.45, 0.12, 0.05])
    y = legvander(mu, 4) @ true_c
    exp_id = np.array(['A'] * 15 + ['B'] * 15)
    # Experiment B sits high by norm_offset; both carry real scatter.
    y = y * np.where(exp_id == 'A', 1.0, 1.0 + norm_offset)
    y = y + rng.normal(0, 0.05, len(mu))
    sigma_stat = np.full(len(mu), 0.02)
    return pd.DataFrame({
        'value': y, 'unc': sigma_stat, 'mu': mu,
        'theta_deg': np.rad2deg(np.arccos(mu)),
        'entry': np.where(exp_id == 'A', '10571', '20743'),
        'subentry': np.where(exp_id == 'A', '002', '003'),
        'sigma_sys_relative':       np.full(len(mu), sys_rel),
        'sigma_sys_indep_relative': np.full(len(mu), sys_rel),
        'sigma_sys_dep_relative':   np.zeros(len(mu)),
        '_sigma_sys_abs':           np.abs(y) * sys_rel,
    })


def _fit(df: pd.DataFrame, **kw):
    base = dict(
        value_col='value', unc_col='unc', sys_unc_col='_sigma_sys_abs',
        degree=None, max_degree=5, select_degree='aicc',
        ridge_lambda=1e-4, ridge_power=4, df_method='hat',
        n_samples=1, use_gls_kernel=True,
        use_band_discrepancy=True, min_points_per_band=3,
        max_band_scale=3.0, tau_irls_max_iters=6, tau_irls_tol=1e-4,
    )
    base.update(kw)
    return sample_legendre_coefficients(df, **base)


def _coeffs(coef_df) -> np.ndarray:
    """The shipped nominal is row 0 of ``coef_df`` — ``info`` carries no
    top-level 'coeffs' key."""
    return coef_df.iloc[0].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# v2 safety: the default must not change anything
# --------------------------------------------------------------------------

def test_default_is_false() -> None:
    """v2 relies on the library default. If this flips, the frozen thesis
    pipeline silently changes its central values."""
    import inspect
    sig = inspect.signature(sample_legendre_coefficients)
    assert sig.parameters['tau_refit_use_gls'].default is False


def test_explicit_false_matches_omitting_the_kwarg() -> None:
    """Passing False must be indistinguishable from the pre-change call."""
    df = _two_experiment_frame()
    cd_omitted, info_omitted = _fit(df)
    cd_false, info_false = _fit(df, tau_refit_use_gls=False)
    np.testing.assert_array_equal(_coeffs(cd_omitted), _coeffs(cd_false))
    assert info_omitted['degree'] == info_false['degree']
    for band in ('tau_F', 'tau_M', 'tau_B'):
        assert info_omitted['tau_info'][band] == info_false['tau_info'][band]


def test_no_op_when_gls_kernel_is_off() -> None:
    """``tau_refit_use_gls`` is gated on ``use_gls_kernel``. With the kernel
    off there is no GLS anywhere, so the flag must do nothing."""
    df = _two_experiment_frame()
    cd_off, _ = _fit(df, use_gls_kernel=False, tau_refit_use_gls=False)
    cd_on, _ = _fit(df, use_gls_kernel=False, tau_refit_use_gls=True)
    np.testing.assert_array_equal(_coeffs(cd_off), _coeffs(cd_on))


# --------------------------------------------------------------------------
# The double-counting guard — the reason this is not a one-line copy of the
# rescan branch
# --------------------------------------------------------------------------

def test_gls_never_receives_sigma_with_sys_already_folded_in() -> None:
    """Every GLS call must get a diagonal that carries σ_stat (τ-inflated or
    not) and NOT √(σ_stat² + σ_sys²). σ_sys enters through the rank-1 columns.

    The check: σ_sys here is ~8 % of |y| while σ_stat is 0.02 absolute, so
    sigma_refit is several times sigma_eff. Any call whose diagonal exceeds
    the largest legitimate σ_eff (σ_stat · max_band_scale) is folding σ_sys in.
    """
    df = _two_experiment_frame()
    seen: list[np.ndarray] = []
    original = rad._weighted_ridge_fit_gls

    def spy(mu, y, sigma_stat, *a, **kw):
        seen.append(np.asarray(sigma_stat).copy())
        return original(mu, y, sigma_stat, *a, **kw)

    rad._weighted_ridge_fit_gls = spy
    try:
        _fit(df, tau_refit_use_gls=True)
    finally:
        rad._weighted_ridge_fit_gls = original

    assert seen, "GLS was never called — the test is not exercising the branch"
    sigma_stat = df['unc'].to_numpy()
    sigma_sys = df['_sigma_sys_abs'].to_numpy()
    # Loosest admissible diagonal: σ_stat scaled by the τ ceiling.
    ceiling = sigma_stat * 3.0 * (1.0 + 1e-9)
    sigma_refit_floor = np.sqrt(sigma_stat ** 2 + sigma_sys ** 2)
    assert np.min(sigma_refit_floor) > np.max(ceiling), (
        "test data is not discriminating: sigma_refit must be clearly above "
        "the tau ceiling for this assertion to mean anything"
    )
    for i, s in enumerate(seen):
        assert np.all(s <= ceiling), (
            f"GLS call {i} received a diagonal above tau*sigma_stat — "
            f"sigma_sys has been folded into the diagonal and is also "
            f"carried rank-1, i.e. counted twice. max ratio to ceiling: "
            f"{np.max(s / ceiling):.3f}"
        )


def test_flag_actually_changes_the_solver_call_pattern() -> None:
    """With the rescan disabled, turning the flag on must add GLS calls —
    otherwise the branch is dead and every other test here is vacuous."""
    df = _two_experiment_frame()

    def count(flag: bool) -> int:
        n = 0
        original = rad._weighted_ridge_fit_gls

        def spy(*a, **kw):
            nonlocal n
            n += 1
            return original(*a, **kw)

        rad._weighted_ridge_fit_gls = spy
        try:
            _fit(df, tau_refit_use_gls=flag, rerun_aicc_post_tau=False)
        finally:
            rad._weighted_ridge_fit_gls = original
        return n

    assert count(True) > count(False)


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------

def test_reduces_to_wls_when_no_correlated_systematics() -> None:
    """With σ_sys_indep = σ_sys_dep = 0 the rank-1 blocks vanish, so the GLS
    refit and the WLS refit solve the same system and must agree."""
    df = _two_experiment_frame(sys_rel=0.0)
    df['_sigma_sys_abs'] = np.zeros(len(df))
    cd_wls, wls = _fit(df, tau_refit_use_gls=False)
    cd_gls, gls = _fit(df, tau_refit_use_gls=True)
    assert wls['degree'] == gls['degree']
    np.testing.assert_allclose(_coeffs(cd_wls), _coeffs(cd_gls),
                               rtol=1e-8, atol=1e-10)


def test_correlated_systematics_move_the_central_value() -> None:
    """The whole point: with a real normalisation disagreement the two
    solvers disagree, and the difference lands in the shipped coefficients."""
    df = _two_experiment_frame()
    cd_wls, wls = _fit(df, tau_refit_use_gls=False)
    cd_gls, gls = _fit(df, tau_refit_use_gls=True)
    cw, cg = _coeffs(cd_wls), _coeffs(cd_gls)
    n = min(len(cw), len(cg))
    if wls['degree'] == gls['degree']:
        assert not np.allclose(cw[:n], cg[:n], rtol=1e-6, atol=1e-9), (
            "GLS and WLS tau refits gave identical coefficients despite an "
            "8 % correlated normalisation error — the branch is not firing"
        )


@pytest.mark.parametrize("seed", range(6))
def test_gls_refit_stays_finite_and_well_formed(seed: int) -> None:
    """Guard against the GLS path returning NaN/inf or a wrong-length vector
    across a spread of random bins."""
    df = _two_experiment_frame(seed=seed, norm_offset=0.05 + 0.05 * seed)
    coef_df, info = _fit(df, tau_refit_use_gls=True)
    c = _coeffs(coef_df)
    assert np.all(np.isfinite(c))
    assert len(c) == info['degree'] + 1
    assert np.isfinite(info['chi2_red'])
    for band in ('tau_F', 'tau_M', 'tau_B'):
        assert 1.0 <= info['tau_info'][band] <= 3.0 + 1e-9
