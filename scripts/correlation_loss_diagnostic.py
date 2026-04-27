"""Loss-of-correlation diagnostic: MI ratio vs Gaussian-equivalent MI.

The Pearson covariance shipped to ENDF MF34 captures only the linear part of
joint dependence. If the underlying multi-bin MC samples are non-Gaussian
(heavy-tailed, asymmetric, or with monotonic non-linearity), Pearson misses
some of the dependence — and downstream first-order error propagation
(``sigma_y^2 = J Sigma J^T``) inherits that loss.

This module computes, per active-parameter pair, the ratio between the
empirical mutual information (estimated from samples) and the
Gaussian-equivalent MI implied by the Pearson correlation:

    mi_gauss(rho) = -0.5 * log(1 - rho^2)        # bivariate-Gaussian MI in nats
    mi_ratio      = mi_actual / mi_gauss

Pairs with ``mi_ratio > 1.5`` carry substantial non-Gaussian dependence that
ENDF MF34 cannot preserve. TMC consumers should sample from the full-MC
parquet for those pairs instead of regenerating from MF34.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _mi_2d_histogram(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Histogram-based mutual information estimate (nats).

    Matches the simple estimator used in
    ``scripts/correlation_analysis_legendre.ipynb`` Cell 12 so that diagnostic
    values are directly comparable.
    """
    H, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = H / H.sum() if H.sum() > 0 else H
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    if not np.any(nz):
        return 0.0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))


def compute_correlation_loss_diagnostic(
    sample_matrix: np.ndarray,
    param_labels: List[Tuple[int, int]],
    energy_mev_lookup: Dict[int, float],
    active_mask: np.ndarray,
    logger,
    rho_min: float = 0.1,
    mi_ratio_flag: float = 1.5,
    mi_bins: int = 20,
    n_top_log: int = 50,
) -> dict:
    """Compute MI-ratio diagnostic and emit flagged pairs into the run log.

    Parameters
    ----------
    sample_matrix : np.ndarray (n_samples, n_params)
        Raw multi-bin MC samples (Pass-1 KW samples).
    param_labels : list of (energy_index, order)
        Same ordering as columns of ``sample_matrix``.
    energy_mev_lookup : dict
        ``{energy_index: energy_mev}`` for log labelling.
    active_mask : np.ndarray (n_params,) bool
        Restrict diagnostic to these parameters.
    logger
        Logger instance — diagnostic output is written here only.
    rho_min : float
        Skip pairs with ``|rho_pearson| < rho_min`` (MI is dominated by
        histogram noise at low correlation; ratio is meaningless).
    mi_ratio_flag : float
        Flag pairs where ``mi_actual / mi_gauss`` exceeds this threshold.
    mi_bins : int
        Histogram bins per axis for the empirical MI estimator.
    n_top_log : int
        Number of top-ranked flagged pairs to dump into the log.

    Returns
    -------
    dict with summary statistics and the top-ranked flagged pairs.
    """
    n_samples, n_params = sample_matrix.shape
    active_idx = np.where(active_mask)[0]
    n_active = int(len(active_idx))

    if n_active < 2:
        logger.info(
            f"  [Loss diag] Skipping: only {n_active} active parameters"
        )
        return {'n_active': n_active, 'n_evaluated': 0, 'n_flagged': 0}

    # Pearson correlation among active parameters
    sm_active = sample_matrix[:, active_idx]
    corr_pearson = np.corrcoef(sm_active, rowvar=False)

    # Iterate upper-triangle, screening on |rho_pearson| >= rho_min
    rows = []
    n_evaluated = 0
    for ii in range(n_active):
        rho_row = corr_pearson[ii]
        candidates = np.where(np.abs(rho_row[ii + 1:]) >= rho_min)[0] + (ii + 1)
        if len(candidates) == 0:
            continue
        x = sm_active[:, ii]
        for jj in candidates:
            rho = float(rho_row[jj])
            if not np.isfinite(rho) or abs(rho) >= 1.0 - 1e-12:
                continue
            mi_g = -0.5 * np.log(1.0 - rho * rho)
            if mi_g <= 0:
                continue
            mi_a = _mi_2d_histogram(x, sm_active[:, jj], bins=mi_bins)
            mi_ratio = mi_a / mi_g
            n_evaluated += 1
            if mi_ratio > mi_ratio_flag:
                gi = active_idx[ii]
                gj = active_idx[jj]
                e_i, l_i = param_labels[gi]
                e_j, l_j = param_labels[gj]
                rows.append({
                    'param_i': int(gi),
                    'param_j': int(gj),
                    'energy_i_index': int(e_i),
                    'energy_j_index': int(e_j),
                    'energy_i_mev': float(energy_mev_lookup.get(e_i, np.nan)),
                    'energy_j_mev': float(energy_mev_lookup.get(e_j, np.nan)),
                    'order_i': int(l_i),
                    'order_j': int(l_j),
                    'rho_pearson': rho,
                    'mi_actual': float(mi_a),
                    'mi_gauss': float(mi_g),
                    'mi_ratio': float(mi_ratio),
                })

    n_flagged = len(rows)
    rows.sort(key=lambda r: r['mi_ratio'], reverse=True)
    rows_top = rows[:n_top_log]

    logger.info("")
    logger.info("#-- Loss-of-correlation diagnostic ----------------------------------------")
    logger.info(
        f"  [Loss diag] active params: {n_active}; "
        f"pairs evaluated (|rho|>={rho_min}): {n_evaluated}; "
        f"flagged (mi_ratio>{mi_ratio_flag}): {n_flagged}"
    )
    if n_evaluated > 0:
        logger.info(
            f"  [Loss diag] flagged fraction of evaluated: "
            f"{100.0 * n_flagged / n_evaluated:.2f}%"
        )

    if n_flagged > 0:
        n_show = len(rows_top)
        truncated = " (truncated)" if n_flagged > n_show else ""
        logger.info(
            f"  [Loss diag] Top-{n_show} flagged pairs by mi_ratio{truncated}:"
        )
        logger.info(
            "    rank  rho_pearson  mi_actual  mi_gauss  mi_ratio  "
            "(E_i,L_i) -- (E_j,L_j)  E_i [MeV] -- E_j [MeV]"
        )
        for k, r in enumerate(rows_top, 1):
            logger.info(
                f"    {k:>4d}  {r['rho_pearson']:>+.3f}      "
                f"{r['mi_actual']:.3e}  {r['mi_gauss']:.3e}  "
                f"{r['mi_ratio']:>6.2f}    "
                f"({r['energy_i_index']},{r['order_i']}) -- "
                f"({r['energy_j_index']},{r['order_j']})  "
                f"{r['energy_i_mev']:.3f} -- {r['energy_j_mev']:.3f}"
            )
        logger.info(
            "  [Loss diag] TMC consumers should sample from "
            "legendre_coefficients_full_correlations_calibrated.parquet "
            "for the flagged pairs above; MF34 captures only the Pearson-"
            "equivalent dependence."
        )
    logger.info("#-- END Loss diag ----------------------------------------------------------")

    return {
        'n_active': n_active,
        'n_evaluated': n_evaluated,
        'n_flagged': n_flagged,
        'top_pairs': rows_top,
    }
