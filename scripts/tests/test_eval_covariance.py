"""Regression tests for the dense per-experiment evaluation covariance.

Two independent invariants are checked:

1. When MF34 has only diagonal energy entries (mat[k, k] only) and a single
   Legendre order, build_mf34_block must reduce to the same per-row variance
   that the legacy `propagate_mf34_to_dy` produced on the diagonal.
2. With a purely diagonal Sigma_eval, the new Cholesky-based per-experiment
   chi^2 must equal the closed-form Sherman-Morrison value computed inline.
3. Whitened residuals satisfy sum(z^2) == chi^2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts importable from a clean test environment.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from numpy.polynomial.legendre import legval
from kika.cov.legendre_covariance import LegendreCovariance

from scripts.eval_covariance import build_mf34_block, build_mf33_block
from scripts.chi2_metrics import (
    mahalanobis_chi2_per_experiment,
    whitened_residuals_per_experiment,
)


# ── Synthetic MF34 builder ────────────────────────────────────────────────────

def _diagonal_mf34(grid_ev, var_per_bin, l_order=1, is_relative=True):
    """One-block MF34 with only the energy-diagonal populated (off-diagonals = 0)."""
    cov = LegendreCovariance()
    M = len(grid_ev) - 1
    mat = np.diag(np.asarray(var_per_bin, dtype=float)[:M])
    cov.add_matrix(
        isotope_row=26056, reaction_row=2, l_row=l_order,
        isotope_col=26056, reaction_col=2, l_col=l_order,
        matrix=mat, energy_grid=list(grid_ev),
        is_relative=is_relative, frame="LAB",
    )
    return cov


def _legacy_propagate_diag(mf34, e_mev, mu, a_l, c0):
    """Reimplemented legacy diagonal-only MF34 propagation, for regression."""
    L = len(a_l)
    e_ev = e_mev * 1e6
    cov = np.zeros((L, L), dtype=float)
    for idx in range(len(mf34.matrices)):
        l_r = mf34.l_rows[idx]
        l_c = mf34.l_cols[idx]
        if l_r < 1 or l_c < 1 or l_r > L or l_c > L:
            continue
        grid = np.asarray(mf34.energy_grids[idx], dtype=float)
        mat = np.asarray(mf34.matrices[idx], dtype=float)
        k = int(np.clip(np.searchsorted(grid, e_ev, side="right") - 1,
                        0, mat.shape[0] - 1))
        v = float(mat[k, k])
        if mf34.is_relative[idx]:
            v *= a_l[l_r - 1] * a_l[l_c - 1]
        cov[l_r - 1, l_c - 1] = v
        if l_r != l_c:
            cov[l_c - 1, l_r - 1] = v
    S = np.zeros((L, len(mu)), dtype=float)
    for l in range(1, L + 1):
        coef = np.zeros(l + 1); coef[l] = 1.0
        S[l - 1, :] = c0 * (2 * l + 1) * legval(mu, coef)
    return np.einsum("li,lk,ki->i", S, cov, S)  # per-row variance (the diagonal)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_diagonal_mf34_matches_legacy_per_row():
    """With diagonal-only MF34, diag(build_mf34_block) must equal the legacy
    per-row variance from `propagate_mf34_to_dy`."""
    grid_ev = np.array([0.5e6, 1.5e6, 2.5e6, 3.5e6, 4.5e6])
    var_per_bin = np.array([0.04, 0.05, 0.06, 0.07])  # rel-var, so a_L^2 sandwiched
    mf34 = _diagonal_mf34(grid_ev, var_per_bin, l_order=1, is_relative=True)

    # Two data points in different MF34 bins, two angles each.
    e_mev = np.array([1.0, 2.0, 3.0, 4.0])
    mu = np.array([-0.5, 0.0, 0.3, 0.8])
    c0 = np.full(4, 0.7)
    a_l_per_pt = np.array([[0.6], [0.5], [0.4], [0.3]])  # L=1 only

    sigma = build_mf34_block(mf34, e_mev, mu, c0, a_l_per_pt)
    assert sigma.shape == (4, 4)

    # All off-diagonal entries must be zero (since each point is in a different
    # MF34 bin and the bin-to-bin block is zero).
    off = sigma - np.diag(np.diag(sigma))
    assert np.allclose(off, 0.0, atol=1e-12)

    # Diagonal must equal legacy per-row variance.
    legacy_var = np.array([
        _legacy_propagate_diag(
            mf34, e_mev[j], mu[j:j+1], a_l_per_pt[j], c0[j],
        )[0] for j in range(4)
    ])
    np.testing.assert_allclose(np.diag(sigma), legacy_var, rtol=1e-12, atol=1e-15)


def test_intra_bin_perfect_correlation():
    """Two data points sharing the same MF34 bin must have block[j,k] = block[j,j]
    (perfect intra-bin correlation per ENDF piecewise-constant semantics)."""
    grid_ev = np.array([0.5e6, 4.5e6])  # one big bin covering the whole range
    var_per_bin = np.array([0.04])
    mf34 = _diagonal_mf34(grid_ev, var_per_bin, l_order=1, is_relative=False)

    e_mev = np.array([1.0, 2.0])
    mu = np.array([0.5, 0.5])  # same angle, same a_L scale (absolute), so same sens
    c0 = np.array([0.8, 0.8])
    a_l_per_pt = np.array([[0.4], [0.4]])

    sigma = build_mf34_block(mf34, e_mev, mu, c0, a_l_per_pt)
    # Should be a rank-1 outer-product-like block (all four entries equal).
    assert np.isclose(sigma[0, 0], sigma[1, 1])
    assert np.isclose(sigma[0, 1], sigma[0, 0])
    assert np.isclose(sigma[1, 0], sigma[0, 0])


def test_chi2_diagonal_matches_rank2_woodbury():
    """With a diagonal Sigma_eval and σ_dep promoted to a second rank-1 mode,
    the Cholesky chi^2 must match the Woodbury closed form on
    Σ = D + uu^T + vv^T + diag(sigma_eval^2) (D excludes σ_dep)."""
    rng = np.random.default_rng(0)
    n = 12
    y_exp = rng.uniform(0.5, 2.0, size=n)
    y_eval = y_exp + rng.normal(scale=0.05, size=n)
    sigma_stat = rng.uniform(0.02, 0.05, size=n)
    sigma_indep = 0.04
    sigma_dep_rel = rng.uniform(0.01, 0.04, size=n)
    sigma_eval_diag = rng.uniform(0.01, 0.03, size=n)

    df = pd.DataFrame({
        "library": "L",
        "experiment_id": "E",
        "y_exp": y_exp,
        "y_eval": y_eval,
        "sigma_exp_stat": sigma_stat,
        "sigma_sys_indep_rel": sigma_indep,
        "sigma_sys_dep_rel": sigma_dep_rel,
    })

    eval_cov = {("L", "E"): np.diag(sigma_eval_diag ** 2)}

    out = mahalanobis_chi2_per_experiment(df, eval_cov)
    chi2_new = float(out["chi2"].iloc[0])

    # Brute-force reference via explicit inverse: Σ = D + uuᵀ + vvᵀ + diag(σ_eval²)
    # where D = diag(σ_stat²) and σ_dep enters as the second rank-1 column.
    D = sigma_stat ** 2 + sigma_eval_diag ** 2
    u = sigma_indep * y_exp
    v = sigma_dep_rel * y_exp
    r = y_exp - y_eval
    Sigma = np.diag(D) + np.outer(u, u) + np.outer(v, v)
    chi2_ref = float(r @ np.linalg.solve(Sigma, r))

    np.testing.assert_allclose(chi2_new, chi2_ref, rtol=1e-10, atol=1e-12)


def test_whitening_identity_with_dense_eval_cov():
    """sum(z^2) over each (library, experiment) group must equal chi^2."""
    rng = np.random.default_rng(7)
    n = 20

    df = pd.DataFrame({
        "library": ["L"] * n,
        "experiment_id": ["E1"] * (n // 2) + ["E2"] * (n - n // 2),
        "y_exp": rng.uniform(0.5, 2.0, size=n),
        "y_eval": rng.uniform(0.5, 2.0, size=n),
        "sigma_exp_stat": rng.uniform(0.02, 0.05, size=n),
        "sigma_sys_indep_rel": 0.04,
        "sigma_sys_dep_rel": 0.02,
    })

    # Construct two PSD dense blocks via random low-rank construction.
    def _rand_psd(m):
        a = rng.normal(size=(m, m + 2))
        return a @ a.T * 0.001

    eval_cov = {
        ("L", "E1"): _rand_psd(n // 2),
        ("L", "E2"): _rand_psd(n - n // 2),
    }

    chi2 = mahalanobis_chi2_per_experiment(df, eval_cov)
    z = whitened_residuals_per_experiment(df, eval_cov)

    z_sums = (z ** 2).groupby(
        [df["library"], df["experiment_id"]], observed=True,
    ).sum()
    chi2_indexed = chi2.set_index(["library", "experiment_id"])["chi2"]
    aligned = chi2_indexed.reindex(z_sums.index)
    np.testing.assert_allclose(z_sums.to_numpy(), aligned.to_numpy(),
                               rtol=1e-8, atol=1e-10)


def test_mf33_block_diagonal_matches_diagonal_average():
    """build_mf33_block diag must equal y_eval^2 * length-weighted-avg(rel_var)."""
    # Two MF33 bins fully inside one MF4 bin → length-weighted avg is the
    # length-weighted average of the diagonal rel-var.
    mf33_grid_ev = np.array([1.0e6, 1.5e6, 2.0e6])
    rel_cov = np.array([
        [0.04, 0.0],
        [0.0,  0.09],
    ])
    energies_mf4_mev = np.array([0.5, 2.5])  # one bin spanning [0.5, 2.5] MeV
    e_mev = np.array([1.2])  # any point in that MF4 bin
    y_eval = np.array([0.7])

    sigma = build_mf33_block(
        mf33_grid_ev, rel_cov, energies_mf4_mev, e_mev, y_eval,
    )
    # The MF4 bin spans [0.5e6, 2.5e6] eV; overlap with MF33 bin0 = 0.5e6,
    # overlap with bin1 = 0.5e6, so weights = [0.5, 0.5] after normalization.
    # rho = 0.5*0.5*0.04 + 0.5*0.5*0.09 + 0 + 0 = 0.0325
    expected = y_eval[0] ** 2 * 0.0325
    np.testing.assert_allclose(sigma[0, 0], expected, rtol=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
