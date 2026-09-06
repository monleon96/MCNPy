"""Regression tests for per-block EXFOR-systematic correlation in chi2_metrics.

The chi^2 covariance is Sigma = diag(D) + u uᵀ + v vᵀ + Sigma_eval, where u, v are
the EXFOR rank-1 systematic modes (normalization, shape). With c_0 fit
independently per incident energy (the exfor_c0 methodology), a single
experiment-wide u/v mode penalises energy-to-energy level differences the
per-energy fit already removed. Passing `systematic_block_col="energy_mev"`
masks u/v to block-diagonal per energy, removing that penalty.

Invariants checked:
1. `systematic_block_col=None` reproduces the original (global) numbers exactly.
2. With per-energy blocking and no Sigma_eval, the experiment chi^2 equals the
   sum of independent per-energy chi^2 (block-diagonal identity), across the
   variant API (V2/V3) and the dense API (mahalanobis_chi2_per_experiment).
3. V1 (diagonal) is unaffected by the blocking flag.
4. A correlated overall offset across energies inflates the global chi^2 but not
   the per-energy chi^2 (the bug this fix addresses).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.chi2_metrics import (
    chi2_per_experiment_variants,
    mahalanobis_chi2_per_experiment,
)


def _synthetic_frame(n_energies=6, n_angles=8, seed=0):
    """One experiment, many energies × angles, with correlated systematics."""
    rng = np.random.default_rng(seed)
    rows = []
    for e_idx in range(n_energies):
        energy = 1.0 + 0.5 * e_idx
        # A per-energy overall level offset (what per-energy c_0 fitting removes):
        # data sits a few % off the eval, coherently across all angles at this E.
        level = 1.0 + 0.04 * rng.standard_normal()
        for a_idx in range(n_angles):
            y_eval = 0.5 + 0.3 * np.cos(np.pi * a_idx / n_angles)
            y_exp = level * y_eval * (1.0 + 0.01 * rng.standard_normal())
            rows.append({
                "library": "L",
                "experiment_id": "E1",
                "energy_mev": energy,
                "y_exp": y_exp,
                "y_eval": y_eval,
                "sigma_exp_stat": 0.02 * y_exp,
                "sigma_sys_indep_rel": 0.06,   # 6% correlated normalization
                "sigma_sys_dep_rel": 0.05,     # 5% correlated shape
            })
    return pd.DataFrame(rows)


def test_none_reproduces_global():
    """systematic_block_col=None must equal the default (global) behaviour."""
    df = _synthetic_frame()
    a = chi2_per_experiment_variants(df, None, systematic_block_col=None)
    b = chi2_per_experiment_variants(df, None)  # default
    for col in ("chi2_v1", "chi2_v2", "chi2_v3"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_per_energy_equals_sum_of_independent_blocks():
    """Per-energy blocking ⇔ summing independent per-energy chi^2 (no Sigma_eval)."""
    df = _synthetic_frame()
    blocked = chi2_per_experiment_variants(
        df, None, systematic_block_col="energy_mev",
    )

    # Reference: treat each energy as its own 'experiment' under the GLOBAL metric.
    ref = chi2_per_experiment_variants(
        df, None, group_cols=("library", "experiment_id", "energy_mev"),
    )
    for col in ("chi2_v2", "chi2_v3"):
        assert np.isclose(blocked[col].iloc[0], ref[col].sum(), rtol=1e-9)

    # Dense API agrees with the variant API's V4-less path (no eval_cov ⇒ V2 == V4).
    dense = mahalanobis_chi2_per_experiment(
        df, None, systematic_block_col="energy_mev",
    )
    assert np.isclose(dense["chi2"].iloc[0], blocked["chi2_v2"].iloc[0], rtol=1e-9)


def test_v1_unaffected_by_blocking():
    """V1 is purely diagonal: the blocking flag must not change it."""
    df = _synthetic_frame()
    g = chi2_per_experiment_variants(df, None, systematic_block_col=None)
    b = chi2_per_experiment_variants(df, None, systematic_block_col="energy_mev")
    assert np.allclose(g["chi2_v1"].to_numpy(), b["chi2_v1"].to_numpy())


def test_global_inflates_relative_to_per_energy():
    """Per-energy level scatter inflates the global chi^2 but not the per-energy one."""
    df = _synthetic_frame()
    g = chi2_per_experiment_variants(df, None, systematic_block_col=None)
    b = chi2_per_experiment_variants(df, None, systematic_block_col="energy_mev")
    # The fix can only lower (or equal) the rank-2 chi^2; the global mode cannot
    # absorb energy-to-energy level differences, so it scores higher.
    assert b["chi2_v2"].iloc[0] <= g["chi2_v2"].iloc[0] + 1e-9
    assert b["chi2_v3"].iloc[0] <= g["chi2_v3"].iloc[0] + 1e-9
