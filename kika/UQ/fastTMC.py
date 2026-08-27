"""
Fast Total Monte Carlo (FTMC) uncertainty propagation analysis.

This module provides functions for decomposing uncertainty in Monte Carlo calculations
into statistical and nuclear data components using the Fast TMC methodology.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional, List


def fastTMC(
    y_val: Optional[Union[np.ndarray, list]] = None,
    y_sigma: Optional[Union[np.ndarray, list]] = None,
    results_df: Optional[pd.DataFrame] = None,
    uncertainties_df: Optional[pd.DataFrame] = None,
    columns: Optional[Union[List[str], List[int], str, int]] = None,
    n_samples: Optional[int] = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = None,
    verbose: bool = True,
    reliability_threshold: float = 0.5
) -> Union[Dict, pd.DataFrame]:
    """
    Perform Fast Total Monte Carlo decomposition with bootstrap confidence intervals.
    
    This unified function can handle both single-tally analysis (using arrays/lists) 
    and multi-tally analysis (using DataFrames). Only one input mode should be used.
    
    Parameters
    ----------
    y_val : array-like, optional
        N sample results for single tally analysis. Cannot be used with DataFrames.
    y_sigma : array-like, optional
        N relative 1-σ uncertainties for single tally analysis (as fractions).
        If None, only total observed uncertainty is computed (no decomposition).
    results_df : pd.DataFrame, optional
        DataFrame with results for each tally (columns) and sample (rows).
    uncertainties_df : pd.DataFrame, optional
        DataFrame with uncertainties for each tally (columns) and sample (rows).
    columns : list or str or int, optional
        Specific columns to analyze. Can be column names, indices, or mix.
        If None, analyzes all columns in DataFrame mode.
    n_samples : int, optional
        Number of samples to use from DataFrames. If None, uses all available.
    n_bootstrap : int, optional
        Number of bootstrap replicates for confidence intervals. Default is 1000.
    ci_level : float, optional
        Desired confidence level (e.g., 0.95 for 95% CI). Default is 0.95.
    random_seed : int, optional
        Seed for reproducible bootstrap resampling. Default is None.
    verbose : bool, optional
        Whether to print detailed results. Default is True.
    reliability_threshold : float, optional
        Maximum ratio of statistical_unc / observed_unc for the decomposition to be 
        considered reliable. Above this threshold, statistical uncertainty dominates and 
        observed uncertainty should be used instead. Default is 0.5 (statistical 
        uncertainty must be less than 50% of observed uncertainty for reliable decomposition).
    
    Returns
    -------
    dict or pd.DataFrame
        For single tally: Dict with uncertainty decomposition results including:
            - 'mean_tally': Mean value across samples
            - 'percent_unc_observed': Total observed uncertainty (%)
            - 'percent_unc_statistical': Statistical uncertainty component (%)
            - 'percent_unc_nuclear_data': Nuclear data uncertainty component (%)
            - 'is_reliable': Boolean flag indicating if decomposition is reliable
            - 'bootstrap': Bootstrap resampling results
            - 'ci': Confidence intervals for key quantities
        For multiple tallies: DataFrame with summary results including reliability flags.
        
        Note: When is_reliable is False, use percent_unc_observed instead of 
        percent_unc_nuclear_data for reporting uncertainties.
    
    Raises
    ------
    ValueError
        If both single-tally and DataFrame inputs are provided, or if neither is provided.
    
    Examples
    --------
    # Single tally analysis
    >>> result = fastTMC(y_val=[1.05, 0.98, 1.02], y_sigma=[0.02, 0.025, 0.018])
    
    # Multi-tally analysis
    >>> summary = fastTMC(results_df=df_results, uncertainties_df=df_uncertainties)
    
    # Specific columns only
    >>> summary = fastTMC(results_df=df_results, uncertainties_df=df_uncertainties, 
    ...                   columns=['F4_flux', 'F6_heating'])
    """
    # Input validation
    single_mode = y_val is not None or y_sigma is not None
    dataframe_mode = results_df is not None or uncertainties_df is not None
    
    if single_mode and dataframe_mode:
        raise ValueError(
            "Cannot use both single-tally mode (y_val, y_sigma) and DataFrame mode "
            "(results_df, uncertainties_df) simultaneously."
        )
    
    if not single_mode and not dataframe_mode:
        raise ValueError(
            "Must provide either single-tally inputs (y_val, y_sigma) or "
            "DataFrame inputs (results_df, uncertainties_df)."
        )
    
    if single_mode:
        if y_val is None:
            raise ValueError("y_val must be provided for single-tally analysis.")

        if y_sigma is not None:
            return _single_tally_analysis(
                y_val, y_sigma, n_bootstrap, ci_level, random_seed, verbose, reliability_threshold
            )
        else:
            # Values-only analysis (deterministic results, no uncertainty decomposition)
            return _values_only_analysis(
                y_val, n_bootstrap, ci_level, random_seed, verbose
            )
    
    else:
        # DataFrame analysis
        if results_df is None or uncertainties_df is None:
            raise ValueError("Both results_df and uncertainties_df must be provided for DataFrame analysis.")
        
        return _dataframe_analysis(
            results_df, uncertainties_df, columns, n_samples, 
            n_bootstrap, ci_level, random_seed, verbose, reliability_threshold
        )


def create_summary_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a summary table from Fast TMC analysis results with separate CI columns.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of FTMC results from fastTMC function.
    
    Returns
    -------
    pd.DataFrame
        Summary table with key uncertainty metrics and separate CI columns.
    """
    summary_data = []
    
    for tally_name, result in results_dict.items():
        if result['mean_tally'] == 0:
            continue
        
        summary_data.append({
            'Mean_Tally': result['mean_tally'],
            'Total_Unc_%': result['percent_unc_observed'],
            'Statistical_%': result['percent_unc_statistical'],
            'Nuclear_Data_%': result['percent_unc_nuclear_data'],
            'Is_Reliable': result['is_reliable'],
            'Observed_CI_Lower_%': result['ci']['percent_unc_observed'][0],
            'Observed_CI_Upper_%': result['ci']['percent_unc_observed'][1],
            'Nuclear_Data_CI_Lower_%': result['ci']['percent_unc_nuclear_data'][0],
            'Nuclear_Data_CI_Upper_%': result['ci']['percent_unc_nuclear_data'][1],
            'Mean_Tally_CI_Lower': result['ci']['mean_tally'][0],
            'Mean_Tally_CI_Upper': result['ci']['mean_tally'][1]
        })
    
    summary_df = pd.DataFrame(summary_data, index=list(results_dict.keys()))
    summary_df.index.name = 'Tally'
    
    return summary_df


def _single_tally_analysis(
    y_val: Union[np.ndarray, list],
    y_sigma: Union[np.ndarray, list],
    n_bootstrap: int,
    ci_level: float,
    random_seed: Optional[int],
    verbose: bool,
    reliability_threshold: float
) -> Dict:
    """Core Fast TMC analysis for a single tally."""
    # Convert to numpy arrays
    y = np.asarray(y_val, dtype=float)
    sig = np.asarray(y_sigma, dtype=float).copy()
    N = y.size
    
    # Validate input
    if N != sig.size:
        raise ValueError("y_val and y_sigma must have the same length")
    
    if sig.mean() > 1.0:
        raise ValueError(
            "Relative uncertainties should be in fraction form (e.g., 0.05 for 5%). "
            "If your uncertainties are in percent, divide by 100."
        )
    
    # Core FTMC calculation
    mean_tally = y.mean()
    abs_err = sig * y
    mean_var_within = np.mean(abs_err**2)
    var_observed = np.var(y, ddof=1)
    var_nuclear = max(var_observed - mean_var_within, 0.0)
    
    # Standard deviations
    std_obs = np.sqrt(var_observed)
    std_stat = np.sqrt(mean_var_within)
    std_nuc = np.sqrt(var_nuclear)
    
    # Convert to percentages
    pct_obs = std_obs / mean_tally * 100.0
    pct_stat = std_stat / mean_tally * 100.0
    pct_nuc = std_nuc / mean_tally * 100.0
    
    # Reliability check: statistical uncertainty must be LESS than reliability_threshold * observed
    # to ensure reliable decomposition (i.e., statistical doesn't dominate)
    # Default threshold is 0.5, meaning statistical must be < 50% of observed
    if pct_obs > 0:
        stat_to_obs_ratio = pct_stat / pct_obs
        is_reliable = stat_to_obs_ratio < reliability_threshold
    else:
        stat_to_obs_ratio = 0.0
        is_reliable = False
    
    # Bootstrap analysis
    bootstrap_results = _bootstrap_analysis(
        y, sig, n_bootstrap, random_seed, verbose
    )
    
    # Print summary (always show warning if unreliable, other details only if verbose)
    _print_summary(mean_tally, pct_obs, pct_stat, pct_nuc, is_reliable, 
                  stat_to_obs_ratio, reliability_threshold, verbose)
    
    # Compute confidence intervals
    ci = _compute_confidence_intervals(bootstrap_results, ci_level, verbose)
    
    return {
        'mean_tally': mean_tally,
        'percent_unc_observed': pct_obs,
        'percent_unc_statistical': pct_stat,
        'percent_unc_nuclear_data': pct_nuc,
        'is_reliable': is_reliable,
        'stat_to_obs_ratio': stat_to_obs_ratio,
        'bootstrap': bootstrap_results,
        'ci': ci
    }


def _values_only_analysis(
    y_val: Union[np.ndarray, list],
    n_bootstrap: int,
    ci_level: float,
    random_seed: Optional[int],
    verbose: bool,
) -> Dict:
    """Analysis for values-only data (no per-sample uncertainties).

    Computes total observed uncertainty with bootstrap CIs but cannot
    decompose into statistical vs nuclear-data components.
    """
    y = np.asarray(y_val, dtype=float)
    N = y.size

    mean_tally = y.mean()
    var_observed = np.var(y, ddof=1)
    std_obs = np.sqrt(var_observed)
    pct_obs = std_obs / mean_tally * 100.0 if mean_tally != 0 else 0.0

    # Bootstrap for CIs on mean and observed uncertainty (vectorized, see
    # _bootstrap_analysis — same RNG stream as the old per-replicate loop)
    rng = np.random.default_rng(random_seed)
    bs_mean = np.empty(n_bootstrap)
    bs_obs = np.empty(n_bootstrap)

    for start, block in _bootstrap_blocks(n_bootstrap, N):
        stop = start + block
        idx = rng.integers(0, N, size=(block, N))
        y_bs = y[idx]
        means = y_bs.mean(axis=1)
        bs_mean[start:stop] = means
        std_obs_bs = np.sqrt(y_bs.var(axis=1, ddof=1))
        bs_obs[start:stop] = np.where(means != 0, std_obs_bs / means * 100.0, 0.0)

    alpha = 1 - ci_level
    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    ci = {
        'mean_tally': tuple(np.percentile(bs_mean, [lo, hi])),
        'percent_unc_observed': tuple(np.percentile(bs_obs, [lo, hi])),
        'percent_unc_nuclear_data': (0.0, 0.0),
    }

    if verbose:
        print(f"Values-only analysis (no uncertainty decomposition)")
        print(f"  Mean tally value      : {mean_tally:.5e}")
        print(f"  Observed uncertainty  : {pct_obs:.2f}%")
        print(f"\nBootstrap {int(ci_level*100)}% CIs:")
        print(f"  • Mean tally          : {ci['mean_tally'][0]:.5e} – {ci['mean_tally'][1]:.5e}")
        print(f"  • Observed unc.       : {ci['percent_unc_observed'][0]:.2f}% – "
              f"{ci['percent_unc_observed'][1]:.2f}%")

    return {
        'mean_tally': mean_tally,
        'percent_unc_observed': pct_obs,
        'percent_unc_statistical': 0.0,
        'percent_unc_nuclear_data': 0.0,
        'is_reliable': True,  # Not applicable — no decomposition to judge
        'stat_to_obs_ratio': 0.0,
        'bootstrap': {
            'mean_tally': bs_mean,
            'percent_unc_nuclear_data': np.zeros(n_bootstrap),
            'percent_unc_observed': bs_obs,
        },
        'ci': ci,
    }


def _dataframe_analysis(
    results_df: pd.DataFrame,
    uncertainties_df: pd.DataFrame,
    columns: Optional[Union[List[str], List[int], str, int]],
    n_samples: Optional[int],
    n_bootstrap: int,
    ci_level: float,
    random_seed: Optional[int],
    verbose: bool,
    reliability_threshold: float
) -> pd.DataFrame:
    """Core Fast TMC analysis for multiple tallies from DataFrames."""
    # Validate inputs
    if not all(col in uncertainties_df.columns for col in results_df.columns):
        raise ValueError("Column mismatch between results and uncertainties DataFrames")
    
    # Determine columns to analyze
    if columns is None:
        columns_to_analyze = list(results_df.columns)
    else:
        if isinstance(columns, (str, int)):
            columns = [columns]
        
        columns_to_analyze = []
        for col in columns:
            if isinstance(col, int):
                # Convert index to column name
                if col < len(results_df.columns):
                    columns_to_analyze.append(results_df.columns[col])
                else:
                    raise ValueError(f"Column index {col} out of range")
            else:
                # Assume it's a column name
                if col in results_df.columns:
                    columns_to_analyze.append(col)
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Determine sample size
    if n_samples is None:
        n_samples = len(results_df)
    else:
        n_samples = min(n_samples, len(results_df))
    
    all_results = {}
    
    # Always show we're starting analysis
    print(f"\nAnalyzing {len(columns_to_analyze)} tallies with Fast TMC ({n_samples} samples)...")
    if verbose:
        print("=" * 70)
    
    for tally_name in columns_to_analyze:
        # Always show which tally we're processing
        print(f"\n{tally_name}: ", end="")
        
        # Extract data
        y_res = results_df[tally_name].iloc[:n_samples].values
        y_sig = uncertainties_df[tally_name].iloc[:n_samples].values
        
        # Skip if mean is zero
        if np.mean(y_res) == 0:
            print(f"SKIPPED (mean is zero)")
            continue
        
        # Perform analysis
        try:
            result = _single_tally_analysis(
                y_res, y_sig, n_bootstrap, ci_level, random_seed, verbose, reliability_threshold
            )
            all_results[tally_name] = result
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Print summary of reliability for all analyzed tallies
    if len(all_results) > 0:
        reliable_tallies = [name for name, res in all_results.items() if res['is_reliable']]
        unreliable_tallies = [name for name, res in all_results.items() if not res['is_reliable']]
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total tallies analyzed: {len(all_results)}")
        print(f"OK (use Nuclear Data): {len(reliable_tallies)}/{len(all_results)}")
        print(f"NOT OK (use Observed): {len(unreliable_tallies)}/{len(all_results)}")
        
        if len(unreliable_tallies) > 0:
            print(f"\nNOT OK tallies (statistical uncertainty too high):")
            for tally in unreliable_tallies:
                res = all_results[tally]
                print(f"   • {tally}: Statistical is {res['stat_to_obs_ratio']*100:.0f}% of observed")
        
        print("=" * 70)
    
    # Return as summary DataFrame
    return create_summary_table(all_results)


def _print_summary(mean_tally: float, pct_obs: float, pct_stat: float, pct_nuc: float,
                  is_reliable: bool, stat_to_obs_ratio: float, 
                  reliability_threshold: float, verbose: bool):
    """Print formatted summary of FTMC results."""
    # ALWAYS show the result - user-friendly and concise
    if is_reliable:
        print(f"✓ OK - Use Nuclear Data uncertainty ({pct_nuc:.2f}%)")
    else:
        print(f"✗ NOT OK - Use Observed uncertainty ({pct_obs:.2f}%) instead")
        print(f"           Statistical uncertainty is {stat_to_obs_ratio*100:.0f}% of observed (threshold: <{reliability_threshold*100:.0f}%)")
        print(f"           → Reduce statistical uncertainty from your results or use observed uncertainty")
    
    # Detailed breakdown only if verbose
    if verbose:
        print(f"\n  Details:")
        print(f"    Mean tally value         : {mean_tally:.5e}")
        print(f"    Observed uncertainty     : {pct_obs:.2f}%")
        print(f"    Statistical component    : {pct_stat:.2f}%")
        print(f"    Nuclear-data component   : {pct_nuc:.2f}%")
        print(f"    Statistical/Observed ratio: {stat_to_obs_ratio*100:.1f}%")


# Cap on the number of (replicate x sample) floats held in memory at once by
# the vectorized bootstrap. 4e6 elements is ~32 MB per intermediate array,
# which keeps peak usage bounded for large N or large n_bootstrap while still
# letting the common case (N ~ 500, n_bootstrap ~ 1000) run in a single block.
_BOOTSTRAP_BLOCK_ELEMENTS = 4_000_000


def _bootstrap_blocks(n_bootstrap: int, n_samples: int):
    """Yield block sizes that partition ``n_bootstrap`` replicates.

    Drawing the resampling indices block-by-block consumes the RNG stream in
    exactly the same order as drawing them one replicate at a time, so the
    blocked/vectorized bootstrap reproduces the old per-replicate loop
    bit-for-bit for any given ``random_seed``.
    """
    per_block = max(1, _BOOTSTRAP_BLOCK_ELEMENTS // max(n_samples, 1))
    done = 0
    while done < n_bootstrap:
        block = min(per_block, n_bootstrap - done)
        yield done, block
        done += block


def _bootstrap_analysis(
    y: np.ndarray,
    sig: np.ndarray,
    n_bootstrap: int,
    random_seed: Optional[int],
    verbose: bool
) -> Dict[str, np.ndarray]:
    """Perform bootstrap analysis for confidence intervals.

    Vectorized over bootstrap replicates: each block resamples ``block x N``
    indices in one draw instead of looping in Python. Same numbers, roughly
    an order of magnitude faster — which matters because a Fast TMC run calls
    this once per tally, and a full MCNP tally set is hundreds of tallies.
    """
    rng = np.random.default_rng(random_seed)
    N = len(y)

    bs_mean = np.empty(n_bootstrap)
    bs_nuc = np.empty(n_bootstrap)
    bs_obs = np.empty(n_bootstrap)

    # (sig * y)**2 is a per-sample quantity, so it can be squared once here
    # and merely re-indexed per replicate instead of re-multiplied and
    # re-squared over the whole (block x N) resample. Element-for-element the
    # same products, so the bootstrap draws stay bit-identical.
    abs_err_sq = (sig * y) ** 2

    for start, block in _bootstrap_blocks(n_bootstrap, N):
        stop = start + block
        idx = rng.integers(0, N, size=(block, N))
        y_bs = y[idx]

        # Mean for each bootstrap replicate
        means = y_bs.mean(axis=1)
        bs_mean[start:stop] = means

        # Observed (total) uncertainty for each replicate
        var_obs_bs = y_bs.var(axis=1, ddof=1)
        bs_obs[start:stop] = np.sqrt(var_obs_bs) / means * 100.0

        # Nuclear data uncertainty for each replicate
        var_within_bs = abs_err_sq[idx].mean(axis=1)
        var_nuc_bs = np.maximum(var_obs_bs - var_within_bs, 0.0)
        bs_nuc[start:stop] = np.sqrt(var_nuc_bs) / means * 100.0

    return {
        'mean_tally': bs_mean,
        'percent_unc_nuclear_data': bs_nuc,
        'percent_unc_observed': bs_obs
    }


def _compute_confidence_intervals(
    bootstrap_results: Dict[str, np.ndarray],
    ci_level: float,
    verbose: bool
) -> Dict[str, Tuple[float, float]]:
    """Compute percentile confidence intervals from bootstrap results."""
    alpha = 1 - ci_level
    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    
    ci = {
        'mean_tally': tuple(np.percentile(bootstrap_results['mean_tally'], [lo, hi])),
        'percent_unc_nuclear_data': tuple(
            np.percentile(bootstrap_results['percent_unc_nuclear_data'], [lo, hi])
        ),
        'percent_unc_observed': tuple(
            np.percentile(bootstrap_results['percent_unc_observed'], [lo, hi])
        ),
    }
    
    if verbose:
        print(f"\nBootstrap {int(ci_level*100)}% CIs:")
        print(f"  • Mean tally                : {ci['mean_tally'][0]:.5e} – {ci['mean_tally'][1]:.5e}")
        print(f"  • Observed unc.             : {ci['percent_unc_observed'][0]:.2f}% – "
              f"{ci['percent_unc_observed'][1]:.2f}%")
        print(f"  • Nuclear-data unc.         : {ci['percent_unc_nuclear_data'][0]:.2f}% – "
              f"{ci['percent_unc_nuclear_data'][1]:.2f}%")
    
    return ci
