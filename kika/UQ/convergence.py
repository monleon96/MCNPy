"""
Convergence analysis for Monte Carlo uncertainty quantification.

Computes running statistics (mean, standard deviation, uncertainty)
with bootstrap confidence intervals as a function of the number of
samples to assess convergence behavior.
"""

import numpy as np
from typing import List, Dict, Optional


def convergence_analysis(
    y_val: np.ndarray,
    y_sigma: Optional[np.ndarray] = None,
    sampling_technique: str = "random",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Optional[List[Dict]]:
    """
    Compute running mean and running uncertainty with bootstrap confidence
    intervals as a function of sample count.

    Parameters
    ----------
    y_val : array-like
        Sample results (N values).
    y_sigma : array-like, optional
        Relative 1-sigma uncertainties per sample (fractions).
        If provided, nuclear-data uncertainty decomposition is also computed.
    sampling_technique : str
        One of 'random', 'lhs', 'sobol'.
        - 'random': compute at every sample count (log-spaced if N > 200)
        - 'sobol': compute only at powers of 2
        - 'lhs': returns None (convergence analysis not meaningful)
    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals.
    ci_level : float
        Confidence level for bootstrap CIs (default 0.95 = 95%).
    random_seed : int, optional
        Seed for reproducible bootstrap resampling.

    Returns
    -------
    list of dict or None
        Each dict has keys: n, running_mean, running_std, running_unc_pct,
        mean_ci_lower, mean_ci_upper, unc_ci_lower, unc_ci_upper.
        If y_sigma is provided, also includes: running_stat_unc_pct,
        running_nd_unc_pct, nd_unc_ci_lower, nd_unc_ci_upper.
        Returns None for LHS sampling.
    """
    y_val = np.asarray(y_val, dtype=float)
    n_total = len(y_val)

    if n_total < 2:
        return []

    technique = sampling_technique.lower().strip()
    if technique == "lhs":
        return None

    # Determine which sample counts to evaluate
    if technique == "sobol":
        sample_points = []
        k = 1
        while 2**k <= n_total:
            sample_points.append(2**k)
            k += 1
    else:
        # Random: every point if N <= 200, else ~200 log-spaced points
        if n_total <= 200:
            sample_points = list(range(2, n_total + 1))
        else:
            pts = np.geomspace(2, n_total, 200)
            pts = np.unique(np.round(pts).astype(int))
            # Ensure endpoints are included
            pts = np.union1d(pts, [2, n_total])
            sample_points = sorted(pts.tolist())

    if not sample_points:
        return []

    has_sigma = y_sigma is not None
    if has_sigma:
        y_sigma = np.asarray(y_sigma, dtype=float)

    rng = np.random.default_rng(random_seed)
    alpha = 1.0 - ci_level
    lo_p = 100.0 * (alpha / 2.0)
    hi_p = 100.0 * (1.0 - alpha / 2.0)

    results = []
    for n in sample_points:
        y_sub = y_val[:n]

        # --- Point estimates ---
        mean_val = y_sub.mean()
        var_obs = np.var(y_sub, ddof=1) if n > 1 else 0.0
        std_obs = np.sqrt(var_obs)
        abs_mean = abs(mean_val)
        unc_pct = (std_obs / abs_mean * 100.0) if abs_mean > 0 else 0.0

        # --- Vectorized bootstrap: shape (n_bootstrap, n) ---
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        y_bs = y_sub[idx]

        bs_means = y_bs.mean(axis=1)
        bs_var = y_bs.var(axis=1, ddof=1)
        bs_std = np.sqrt(bs_var)
        abs_bs_means = np.abs(bs_means)
        bs_unc_pct = np.where(
            abs_bs_means > 0,
            bs_std / abs_bs_means * 100.0,
            0.0,
        )

        mean_ci = np.percentile(bs_means, [lo_p, hi_p])
        unc_ci = np.percentile(bs_unc_pct, [lo_p, hi_p])

        point = {
            "n": int(n),
            "running_mean": float(mean_val),
            "running_std": float(std_obs),
            "running_unc_pct": float(unc_pct),
            "mean_ci_lower": float(mean_ci[0]),
            "mean_ci_upper": float(mean_ci[1]),
            "unc_ci_lower": float(unc_ci[0]),
            "unc_ci_upper": float(unc_ci[1]),
        }

        if has_sigma:
            sig_sub = y_sigma[:n]

            # Point estimates for decomposition
            abs_err = sig_sub * y_sub
            mean_var_within = np.mean(abs_err**2)
            var_nd = max(var_obs - mean_var_within, 0.0)
            stat_unc_pct = (np.sqrt(mean_var_within) / abs_mean * 100.0) if abs_mean > 0 else 0.0
            nd_unc_pct = (np.sqrt(var_nd) / abs_mean * 100.0) if abs_mean > 0 else 0.0

            # Bootstrap decomposition
            sig_bs = sig_sub[idx]
            abs_err_bs = sig_bs * y_bs
            var_within_bs = (abs_err_bs**2).mean(axis=1)
            var_nd_bs = np.maximum(bs_var - var_within_bs, 0.0)
            std_nd_bs = np.sqrt(var_nd_bs)
            bs_nd_unc_pct = np.where(
                abs_bs_means > 0,
                std_nd_bs / abs_bs_means * 100.0,
                0.0,
            )

            nd_unc_ci = np.percentile(bs_nd_unc_pct, [lo_p, hi_p])

            point["running_stat_unc_pct"] = float(stat_unc_pct)
            point["running_nd_unc_pct"] = float(nd_unc_pct)
            point["nd_unc_ci_lower"] = float(nd_unc_ci[0])
            point["nd_unc_ci_upper"] = float(nd_unc_ci[1])

        results.append(point)

    return results
