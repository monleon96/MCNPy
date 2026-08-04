"""Tests for the Phase-2 fixed-shape c0 (elastic magnitude) channel.

The magnitude channel records, per MC sample and bin, the closed-form scale of
the perturbed data against the *frozen* nominal shape::

    s = sum(w * Y * y_nom) / sum(w * y_nom**2)          c0 = s * c0_nom

Because it only reads the perturbed data ``Y`` (never re-solves the shape fit)
the MF4/MF34 outputs stay bit-identical; these tests pin the estimator itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legval

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402

from scripts.resample_AD import (  # noqa: E402
    combine_c0_covariance,
    fixed_shape_c0_scale,
    sample_legendre_coefficients,
)
from scripts.exfor_utils import build_mf33_channel, stack_c0_samples  # noqa: E402
from scripts.multigroup_collapse import collapse_mf33_covariance_to_grid  # noqa: E402
from kika.endf.writers.mf33_writer import (  # noqa: E402
    create_mf33_from_covariance,
    write_mf33_to_file,
)
from kika.endf.parsers.parse_mf33 import parse_mf33_mt  # noqa: E402
from kika.endf.utils import (  # noqa: E402
    ENDF_FORMAT_INT,
    format_endf_data_line,
    format_endf_fend_record,
    format_endf_mend_record,
    format_endf_send_record,
)


NOMINAL = np.array([2.5, 0.4, 0.15])  # c0=2.5, c1, c2
MU = np.linspace(-0.9, 0.9, 11)


def test_pure_scaling_recovers_scale_1d():
    """If Y is exactly alpha*y_nom, s == alpha for any weights."""
    y_nom = legval(MU, NOMINAL)
    alpha = 1.37
    Y = alpha * y_nom
    s, c0 = fixed_shape_c0_scale(Y, MU, NOMINAL)
    assert np.isclose(s, alpha)
    assert np.isclose(c0, alpha * NOMINAL[0])
    # 0-D outputs for a 1-D input.
    assert np.ndim(s) == 0 and np.ndim(c0) == 0


def test_pure_scaling_invariant_to_weights():
    """s = alpha regardless of external_weights / sigma_for_fit when Y=alpha*y_nom."""
    y_nom = legval(MU, NOMINAL)
    alpha = 0.82
    Y = alpha * y_nom
    rng = np.random.default_rng(0)
    g = rng.uniform(0.2, 2.0, size=MU.size)
    sig = rng.uniform(0.05, 0.5, size=MU.size)
    s, _ = fixed_shape_c0_scale(Y, MU, NOMINAL, external_weights=g, sigma_for_fit=sig)
    assert np.isclose(s, alpha)


def test_batch_recovers_per_row_scales():
    """2-D Y -> one scale per row, shape (n_draws,)."""
    y_nom = legval(MU, NOMINAL)
    alphas = np.array([0.5, 1.0, 1.5, 2.2])
    Y = alphas[:, None] * y_nom[None, :]
    s, c0 = fixed_shape_c0_scale(Y, MU, NOMINAL)
    assert s.shape == (4,)
    assert np.allclose(s, alphas)
    assert np.allclose(c0, alphas * NOMINAL[0])


def test_matches_explicit_wls_formula():
    """Non-exact Y: s equals the hand-computed weighted ratio."""
    y_nom = legval(MU, NOMINAL)
    rng = np.random.default_rng(1)
    Y = y_nom + rng.normal(0, 0.1, size=MU.size)
    g = rng.uniform(0.3, 1.5, size=MU.size)
    sig = rng.uniform(0.05, 0.4, size=MU.size)
    w = g / sig ** 2
    expected = np.sum(w * Y * y_nom) / np.sum(w * y_nom ** 2)
    s, _ = fixed_shape_c0_scale(Y, MU, NOMINAL, external_weights=g, sigma_for_fit=sig)
    assert np.isclose(s, expected)


def test_nonpositive_sigma_points_dropped():
    """Points with sigma<=0 or non-finite are excluded (weight 0), not NaN-poisoned."""
    y_nom = legval(MU, NOMINAL)
    alpha = 1.1
    Y = alpha * y_nom
    sig = np.full(MU.size, 0.2)
    sig[0] = 0.0
    sig[1] = -1.0
    sig[2] = np.inf
    s, _ = fixed_shape_c0_scale(Y, MU, NOMINAL, sigma_for_fit=sig)
    assert np.isclose(s, alpha)  # dropped points don't change an exact scaling


def test_zero_nominal_curve_raises():
    """A degenerate all-zero nominal shape has no scale to fit."""
    zero_nom = np.zeros(3)
    try:
        fixed_shape_c0_scale(legval(MU, NOMINAL), MU, zero_nom)
    except ValueError as exc:
        assert "non-positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a zero nominal curve")


# --- Pass-2 opt-in recording in sample_legendre_coefficients ----------------

def _ad_df(n_per_pt: int = 1):
    """A small single-experiment angular-distribution frame around NOMINAL."""
    y = legval(MU, NOMINAL)
    return pd.DataFrame({
        "mu": MU,
        "value": y,
        "unc": np.full(MU.size, 0.05),
        "entry": ["12345"] * MU.size,
    })


def test_record_c0_off_is_bit_identical():
    """Default (record_c0_scale=False): coef_df and info identical, no new key."""
    df = _ad_df()
    kw = dict(
        value_col="value", unc_col="unc", mu_col="mu", degree=2, max_degree=2,
        n_samples=64, stochastic=True, random_state=7, freeze_c0=True,
        rescale_unc_by_chi2=False,
    )
    coef_a, info_a = sample_legendre_coefficients(df, **kw)
    coef_b, info_b = sample_legendre_coefficients(df, **kw)
    pd.testing.assert_frame_equal(coef_a, coef_b)
    assert "c0_samples" not in info_a
    assert set(info_a) == set(info_b)


def test_record_c0_does_not_change_coef_df():
    """Turning the recorder on must not perturb the fitted coefficients."""
    df = _ad_df()
    kw = dict(
        value_col="value", unc_col="unc", mu_col="mu", degree=2, max_degree=2,
        n_samples=64, stochastic=True, random_state=11, freeze_c0=True,
        rescale_unc_by_chi2=False,
    )
    coef_off, info_off = sample_legendre_coefficients(df, **kw)
    coef_on, info_on = sample_legendre_coefficients(
        df, **kw, record_c0_scale=True, c0_scale_ref_coeffs=NOMINAL,
    )
    pd.testing.assert_frame_equal(coef_off, coef_on)
    assert "c0_samples" in info_on
    assert info_on["c0_samples"].shape == (64,)


def test_recorded_c0_scatters_around_nominal():
    """Recorded c0 samples center on c0_nom with a sensible spread."""
    df = _ad_df()
    _, info = sample_legendre_coefficients(
        df, value_col="value", unc_col="unc", mu_col="mu", degree=2,
        max_degree=2, n_samples=4000, stochastic=True, random_state=3,
        freeze_c0=True, rescale_unc_by_chi2=False,
        record_c0_scale=True, c0_scale_ref_coeffs=NOMINAL,
    )
    c0 = info["c0_samples"]
    assert np.isclose(np.mean(c0), NOMINAL[0], rtol=0.02)
    assert 0.0 < np.std(c0) < 0.5 * NOMINAL[0]


# --- Two-pass congruence combine -------------------------------------------

def _two_pass_samples(n1=6000, n2=6000, seed=0):
    rng = np.random.default_rng(seed)
    c0_nom = np.array([2.0, 2.5, 3.0])
    # Pass 1: a shared multiplicative factor across bins → strong + correlation.
    z = rng.normal(0.0, 0.10, (n1, 1))
    p1 = c0_nom[None, :] * (1.0 + z) + rng.normal(0.0, 0.01, (n1, 3))
    # Pass 2: independent per-bin scatter with known marginal std.
    std_true = np.array([0.20, 0.25, 0.30])
    p2 = c0_nom[None, :] + rng.normal(0.0, 1.0, (n2, 3)) * std_true[None, :]
    return p1, p2, c0_nom, std_true


def test_combine_variances_from_pass2():
    """Diagonal (absolute) variance is set by the Pass-2 marginal std."""
    p1, p2, c0_nom, std_true = _two_pass_samples()
    rel, cov = combine_c0_covariance(p1, p2, c0_nom)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), std_true, rtol=0.05)
    np.testing.assert_allclose(np.diag(rel), (std_true / c0_nom) ** 2, rtol=0.05)


def test_combine_correlations_from_pass1():
    """Off-diagonal correlation comes from the Pass-1 shared draws (high +)."""
    p1, p2, c0_nom, _ = _two_pass_samples()
    rel, cov = combine_c0_covariance(p1, p2, c0_nom)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    assert corr[0, 1] > 0.9 and corr[0, 2] > 0.9 and corr[1, 2] > 0.9


def test_combine_symmetric_and_psd():
    p1, p2, c0_nom, _ = _two_pass_samples()
    rel, cov = combine_c0_covariance(p1, p2, c0_nom)
    np.testing.assert_allclose(rel, rel.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(rel)) > -1e-10


def test_combine_tolerates_nan_and_zero_nominal():
    """NaN entries and a zero nominal bin don't crash or poison the matrix."""
    p1, p2, c0_nom, _ = _two_pass_samples(n1=2000, n2=2000, seed=5)
    p1[0, 0] = np.nan
    p1[3, 2] = np.nan
    c0_nom = c0_nom.copy()
    c0_nom[1] = 0.0  # degenerate bin
    rel, cov = combine_c0_covariance(p1, p2, c0_nom)
    assert np.all(np.isfinite(rel))
    assert np.allclose(rel[1, :], 0.0) and np.allclose(rel[:, 1], 0.0)


# --- Sidecar assembly from the per-sample dicts ----------------------------

def test_stack_c0_samples_fills_missing_with_nan():
    samples = {0: {5: 2.0, 7: 3.0}, 1: {5: 2.1}}  # sample 1 missing bin 7
    m = stack_c0_samples(samples, energy_indices=[5, 7], n_samples=2)
    assert m.shape == (2, 2)
    assert m[0, 0] == 2.0 and m[0, 1] == 3.0 and m[1, 0] == 2.1
    assert np.isnan(m[1, 1])


def test_build_mf33_channel_shapes_and_frame():
    """End-to-end assembly from two per-sample dicts → cov + long sample frame."""
    n_samples = 500
    energy_indices = [3, 4, 5]
    c0_nom = np.array([2.0, 2.5, 3.0])
    rng = np.random.default_rng(9)
    z = rng.normal(0.0, 0.08, (n_samples, 1))
    p1 = c0_nom[None, :] * (1.0 + z)
    p2 = c0_nom[None, :] + rng.normal(0.0, 0.2, (n_samples, 3))
    kw = {s: {e: p1[s, k] for k, e in enumerate(energy_indices)} for s in range(n_samples)}
    pb = {s: {e: p2[s, k] for k, e in enumerate(energy_indices)} for s in range(n_samples)}

    rel, cov, df, diag = build_mf33_channel(kw, pb, energy_indices, c0_nom, n_samples)

    assert rel.shape == (3, 3) and cov.shape == (3, 3)
    np.testing.assert_allclose(rel, rel.T, atol=1e-12)
    # Long frame: both passes, all (sample, bin) rows, expected columns.
    assert set(df.columns) == {"sample_idx", "energy_index", "pass", "c0"}
    assert len(df) == 2 * n_samples * len(energy_indices)
    assert set(df["pass"].unique()) == {"pass1", "pass2"}
    assert set(df["energy_index"].unique()) == set(energy_indices)
    # Completeness/PSD diagnostics: full samples → full counts, PSD corr.
    np.testing.assert_array_equal(diag["p1_finite_per_bin"], [n_samples] * 3)
    np.testing.assert_array_equal(diag["p2_finite_per_bin"], [n_samples] * 3)
    assert diag["corr_pass1_min_eig"] > -1e-10


def test_build_mf33_channel_sidecar_roundtrip(tmp_path):
    """The saved npy sidecars round-trip to the in-memory matrices."""
    n_samples = 200
    energy_indices = [0, 1]
    c0_nom = np.array([1.8, 2.2])
    rng = np.random.default_rng(2)
    p1 = c0_nom[None, :] * (1.0 + rng.normal(0, 0.05, (n_samples, 1)))
    p2 = c0_nom[None, :] + rng.normal(0, 0.15, (n_samples, 2))
    kw = {s: {e: p1[s, k] for k, e in enumerate(energy_indices)} for s in range(n_samples)}
    pb = {s: {e: p2[s, k] for k, e in enumerate(energy_indices)} for s in range(n_samples)}
    rel, cov, df, _diag = build_mf33_channel(kw, pb, energy_indices, c0_nom, n_samples)

    np.save(tmp_path / "mf33_relative_covariance.npy", rel)
    np.save(tmp_path / "mf33_c0_nominal.npy", c0_nom)
    df.to_parquet(tmp_path / "mf33_c0_samples.parquet", engine="pyarrow", index=False)

    np.testing.assert_array_equal(np.load(tmp_path / "mf33_relative_covariance.npy"), rel)
    np.testing.assert_array_equal(np.load(tmp_path / "mf33_c0_nominal.npy"), c0_nom)
    df_back = pd.read_parquet(tmp_path / "mf33_c0_samples.parquet")
    assert len(df_back) == len(df)


# --- Write path: collapse → create_mf33 → write → re-parse -----------------

ZA, AWR, MAT, MT = 26056.0, 55.454, 2631, 2


def _mf33_template(path):
    """Tiny ENDF template lacking MF33 (one MF3/MT2 stub + FEND + MEND)."""
    def line(vals, mat, mf, mt):
        return format_endf_data_line(vals, mat, mf, mt, 0, formats=[ENDF_FORMAT_INT] * 6)
    lines = [
        line([26056, 0, 0, 0, 0, 0], MAT, 3, MT),
        format_endf_send_record(MAT, 3),
        format_endf_fend_record(MAT),
        format_endf_mend_record(),
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_mf33_write_path_collapse_create_write_roundtrip(tmp_path):
    """The pipeline's collapse→create→write chain round-trips on the coarse grid."""
    # Fine relative covariance over 4 bins (fully correlated, 10% rel std).
    fine_grid = np.array([0.8e6, 1.0e6, 1.2e6, 1.4e6, 1.6e6])
    rel_std = 0.10
    fine_rel = np.full((4, 4), rel_std ** 2)  # rho=1 block
    c0_nom = np.array([2.0, 2.1, 2.2, 2.3])
    sigma_el = 4.0 * np.pi * c0_nom

    # Collapse onto a 2-group coarse grid (merge pairs).
    coarse_grid = np.array([0.8e6, 1.2e6, 1.6e6])
    rel_mg = collapse_mf33_covariance_to_grid(
        native_grid_ev=fine_grid, native_cov=fine_rel, target_grid_ev=coarse_grid,
        native_means=sigma_el, is_relative=True,
    )
    assert rel_mg.shape == (2, 2)
    # Fully-correlated 10% input collapses to ~10% rel std per group.
    np.testing.assert_allclose(np.sqrt(np.diag(rel_mg)), rel_std, rtol=0.05)

    mf33 = create_mf33_from_covariance(rel_mg, coarse_grid, ZA, AWR, MAT, MT)
    src = _mf33_template(tmp_path / "t.endf")
    out = tmp_path / "with_mf33.endf"
    write_mf33_to_file(str(src), mf33, str(out), update_directory=False)

    mf33_lines = [ln for ln in out.read_text().splitlines()
                  if len(ln) >= 72 and ln[70:72] == "33"]
    parsed = parse_mf33_mt(mf33_lines, MT)
    M = np.asarray(parsed.to_xs_covmat().matrices[0])
    np.testing.assert_allclose(M, rel_mg, atol=1e-6)


# --- Phase-2 pre-run fixes: recentring, contiguity ---------------------------

import pytest  # noqa: E402

from scripts.exfor_utils import (  # noqa: E402
    contiguous_grid_from_bins,
    recentre_relative_covariance,
)


class _Bin:
    def __init__(self, lo, hi):
        self.bin_lower_mev = lo
        self.bin_upper_mev = hi


def test_contiguous_grid_from_bins_builds_grid():
    bins = [_Bin(0.8, 0.9), _Bin(0.9, 1.1), _Bin(1.1, 1.5)]
    grid = contiguous_grid_from_bins(bins)
    np.testing.assert_allclose(grid, np.array([0.8, 0.9, 1.1, 1.5]) * 1e6)


def test_contiguous_grid_from_bins_raises_on_gap():
    bins = [_Bin(0.8, 0.9), _Bin(1.0, 1.1)]  # gap 0.9 -> 1.0
    with pytest.raises(ValueError, match="gap"):
        contiguous_grid_from_bins(bins)


def test_contiguous_grid_from_bins_empty_raises():
    with pytest.raises(ValueError, match="no bins"):
        contiguous_grid_from_bins([])


def test_recentre_relative_covariance_known_values():
    """rel = C_abs / outer(ref); non-positive reference rows/cols zeroed."""
    cov_abs = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 9.0, 3.0],
        [1.0, 3.0, 16.0],
    ])
    ref = np.array([2.0, 3.0, 0.0])  # third bin: host sigma <= 0
    rel = recentre_relative_covariance(cov_abs, ref)
    np.testing.assert_allclose(rel[0, 0], 1.0)      # 4 / (2*2)
    np.testing.assert_allclose(rel[1, 1], 1.0)      # 9 / (3*3)
    np.testing.assert_allclose(rel[0, 1], 2.0 / 6.0)
    assert np.all(rel[2, :] == 0.0) and np.all(rel[:, 2] == 0.0)
    np.testing.assert_allclose(rel, rel.T)


def test_recentre_preserves_absolute_claim():
    """Recentring on different means keeps rel * outer(means) == C_abs."""
    rng = np.random.default_rng(5)
    A = rng.normal(size=(4, 4))
    cov_abs = A @ A.T
    host = np.array([1.5, 2.5, 3.5, 4.5])
    rel = recentre_relative_covariance(cov_abs, host)
    np.testing.assert_allclose(rel * np.outer(host, host), cov_abs, atol=1e-12)
