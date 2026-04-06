"""
Normality testing and diagnostic data for Monte Carlo sample distributions.

Provides statistical tests and data for histogram / Q-Q plot generation.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Optional


def normality_tests(y_val: np.ndarray, alpha: float = 0.05) -> List[Dict]:
    """
    Run normality tests on sample data.

    Parameters
    ----------
    y_val : array-like
        Sample values.
    alpha : float
        Significance level for the is_normal decision (default 0.05).

    Returns
    -------
    list of dict
        Each dict has: test_name, statistic, p_value, is_normal.
        Tests included:
        - Shapiro-Wilk (N <= 5000, subsampled otherwise)
        - Anderson-Darling
        - D'Agostino-Pearson K² (N >= 20)
    """
    y_val = np.asarray(y_val, dtype=float)
    # Remove NaN/inf
    y_val = y_val[np.isfinite(y_val)]
    n = len(y_val)

    if n < 8:
        return []

    results = []

    # Shapiro-Wilk (limit to 5000 samples)
    if n <= 5000:
        sw_stat, sw_p = stats.shapiro(y_val)
    else:
        rng = np.random.default_rng(42)
        subsample = rng.choice(y_val, size=5000, replace=False)
        sw_stat, sw_p = stats.shapiro(subsample)
    results.append({
        "test_name": "Shapiro-Wilk",
        "statistic": float(sw_stat),
        "p_value": float(sw_p),
        "is_normal": bool(sw_p > alpha),
    })

    # Anderson-Darling
    ad_result = stats.anderson(y_val, dist="norm")
    # Use the 5% significance level critical value
    ad_critical_idx = list(ad_result.significance_level).index(5.0) if 5.0 in ad_result.significance_level else 2
    ad_is_normal = ad_result.statistic < ad_result.critical_values[ad_critical_idx]
    # Anderson-Darling doesn't return a p-value directly, approximate
    # Using the statistic vs critical value at 5%
    results.append({
        "test_name": "Anderson-Darling",
        "statistic": float(ad_result.statistic),
        "p_value": float(ad_result.critical_values[ad_critical_idx]),  # critical value at 5%
        "is_normal": bool(ad_is_normal),
    })

    # D'Agostino-Pearson K² (requires N >= 20)
    if n >= 20:
        try:
            k2_stat, k2_p = stats.normaltest(y_val)
            results.append({
                "test_name": "D'Agostino-Pearson",
                "statistic": float(k2_stat),
                "p_value": float(k2_p),
                "is_normal": bool(k2_p > alpha),
            })
        except Exception:
            pass

    return results


def histogram_data(y_val: np.ndarray, n_bins: int = 30) -> Dict:
    """
    Compute histogram bins/counts and a normal PDF overlay.

    Parameters
    ----------
    y_val : array-like
        Sample values.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict with keys:
        bin_edges: list of float (n_bins + 1 values)
        bin_centers: list of float (n_bins values)
        counts: list of float (normalized density, n_bins values)
        normal_pdf: list of float (normal PDF at bin centers, n_bins values)
    """
    y_val = np.asarray(y_val, dtype=float)
    y_val = y_val[np.isfinite(y_val)]

    counts, bin_edges = np.histogram(y_val, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mu = np.mean(y_val)
    sigma = np.std(y_val)
    if sigma > 0:
        normal_pdf = stats.norm.pdf(bin_centers, mu, sigma)
    else:
        normal_pdf = np.zeros_like(bin_centers)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "normal_pdf": normal_pdf.tolist(),
    }


def qq_plot_data(y_val: np.ndarray) -> Dict:
    """
    Compute Q-Q plot data (theoretical vs sample quantiles).

    Parameters
    ----------
    y_val : array-like
        Sample values.

    Returns
    -------
    dict with keys:
        theoretical: list of float (theoretical normal quantiles)
        sample: list of float (sorted sample quantiles)
    """
    y_val = np.asarray(y_val, dtype=float)
    y_val = y_val[np.isfinite(y_val)]
    y_sorted = np.sort(y_val)
    n = len(y_sorted)

    # Theoretical quantiles from standard normal
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probabilities)

    # Scale sample to standard normal for comparison
    mu = np.mean(y_sorted)
    sigma = np.std(y_sorted)
    if sigma > 0:
        sample_standardized = (y_sorted - mu) / sigma
    else:
        sample_standardized = np.zeros(n)

    return {
        "theoretical": theoretical.tolist(),
        "sample": sample_standardized.tolist(),
    }
