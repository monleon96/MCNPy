import math
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

# Import shared logger getter
from kika.sampling.utils import _get_logger

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _empirical_cov(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)

def _relative_frobenius(A: np.ndarray, B: np.ndarray) -> float:
    """Return 100 * ||A-B||_F / ||B||_F as percent (NaN-safe)."""
    finite_mask = np.isfinite(A) & np.isfinite(B)
    A_clean = np.where(finite_mask, A, 0.0)
    B_clean = np.where(finite_mask, B, 0.0)
    denom = np.linalg.norm(B_clean, ord='fro')
    if denom == 0.0:
        return np.inf
    return (np.linalg.norm(A_clean - B_clean, ord='fro') / denom) * 100.0

# ----------------------------------------------------------------------
#  Linear-space sample diagnostics
# ----------------------------------------------------------------------
def _diagnostics_samples_linear(
    samples: np.ndarray,
    cov_lin: np.ndarray,
    param_pairs: List[Tuple[int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    z_limit: float = 3.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Linear space sample diagnostics for standard covariance data.
    Returns consolidated diagnostic results.
    """
    results = {
        'n_samples': samples.shape[0],
        'n_parameters': samples.shape[1],
        'negative_warnings': [],
        'mean_deviation_warnings': [],
        'frobenius_error_pct': None,
        'space': 'linear',
        'overall_status': 'PASS'
    }

    if not verbose:
        return results

    # Get logger from shared utils
    logger = _get_logger()

    separator = "-" * 60
    log_msg = f"\n[SAMPLING] [LINEAR SPACE DIAGNOSTICS]\n{separator}"
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    means_f = samples.mean(axis=0)
    p, n = cov_lin.shape[0], samples.shape[0]
    results['n_samples'] = n
    results['n_parameters'] = p

    # Check for negative values (problematic for cross-sections)
    neg_mask = samples < 0
    if np.any(neg_mask):
        frac_neg = np.mean(neg_mask, axis=0)
        for dim, frac in enumerate(frac_neg):
            if frac == 0.0:
                continue
            pair_idx, grp_idx = divmod(dim, num_groups)
            if pair_idx < len(param_pairs):
                zaid, mt = param_pairs[pair_idx]

                if bins is not None and grp_idx < len(bins) - 1:
                    lo, hi = bins[grp_idx], bins[grp_idx + 1]
                    param_desc = (f"(ZAID={zaid}, MT={mt}), "
                                 f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
                else:
                    param_desc = f"(ZAID={zaid}, MT={mt}), G={grp_idx}"
            else:
                param_desc = f"Parameter {dim}"

            warning_msg = f"Negative values in {param_desc}: {frac:.2%} of samples"
            results['negative_warnings'].append(warning_msg)

    # Check means significantly off 1
    for dim in range(p):
        var_lin = cov_lin[dim, dim]
        if var_lin == 0.0:
            continue
        
        se_f = np.sqrt(var_lin / n)
        z = (means_f[dim] - 1.0) / se_f
        if abs(z) <= z_limit:
            continue

        pair_idx, grp_idx = divmod(dim, num_groups)
        if pair_idx < len(param_pairs):
            zaid, mt = param_pairs[pair_idx]

            if bins is not None and grp_idx < len(bins) - 1:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                param_desc = (f"(ZAID={zaid}, MT={mt}), "
                             f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                param_desc = f"(ZAID={zaid}, MT={mt}), G={grp_idx}"
        else:
            param_desc = f"Parameter {dim}"

        warning_msg = f"Mean deviation in {param_desc}: |z|={abs(z):.2f}>{z_limit} (mean={means_f[dim]:+.3e})"
        results['mean_deviation_warnings'].append(warning_msg)

    # Full-matrix reproduction
    emp_cov = _empirical_cov(samples)
    frob_rel = _relative_frobenius(emp_cov, cov_lin)
    results['frobenius_error_pct'] = frob_rel

    # Determine overall status
    n_neg_warnings = len(results['negative_warnings'])
    n_mean_warnings = len(results['mean_deviation_warnings'])
    
    if frob_rel > 15.0 or n_neg_warnings > p * 0.05 or n_mean_warnings > p * 0.05:
        results['overall_status'] = 'FAIL'
    elif frob_rel > 5.0 or n_neg_warnings > 0 or n_mean_warnings > 0:
        results['overall_status'] = 'WARN'

    # Consolidated technical summary
    tech_msg = f"  Technical Summary:"
    if logger:
        logger.info(tech_msg)
    else:
        print(tech_msg)
        
    frob_msg = f"    • Covariance reproduction (Frobenius): {frob_rel:.2f}%"
    neg_msg = f"    • Negative values: {n_neg_warnings}/{p} parameters affected"
    mean_msg = f"    • Mean deviations: {n_mean_warnings}/{p} parameters outside z±{z_limit}"
    
    for msg in [frob_msg, neg_msg, mean_msg]:
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # User-friendly status
    status_tag = {"PASS": "[GOOD]", "WARN": "[WARN]", "FAIL": "[POOR]"}
    status_msg = f"  {status_tag.get(results['overall_status'], '?')} Overall Status: {results['overall_status']}"
    if logger:
        logger.info(status_msg)
    else:
        print(status_msg)

    # Detailed warnings if any
    all_warnings = results['negative_warnings'] + results['mean_deviation_warnings']
    if all_warnings:
        detail_msg = f"  [WARN] Detailed warnings ({len(all_warnings)} total):"
        if logger:
            logger.info(detail_msg)
        else:
            print(detail_msg)
        for w in all_warnings:
            warning_detail = f"    • {w}"
            if logger:
                logger.info(warning_detail)
            else:
                print(warning_detail)
    else:
        ok_detail = "  ✓ All sample dimensions within acceptable thresholds."
        if logger:
            logger.info(ok_detail)
        else:
            print(ok_detail)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)
        
    return results


def _diagnostics_samples_log(
    samples: np.ndarray,
    cov_log: np.ndarray,
    param_pairs: List[Tuple[int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    z_limit: float = 3.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Log space sample diagnostics for standard covariance data.
    Returns consolidated diagnostic results.
    """
    results = {
        'n_samples': samples.shape[0],
        'n_parameters': samples.shape[1],
        'mean_deviation_warnings': [],
        'frobenius_error_pct': None,
        'space': 'log',
        'overall_status': 'PASS'
    }

    if not verbose:
        return results

    # Get logger from shared utils
    logger = _get_logger()

    separator = "-" * 60
    log_msg = f"\n[SAMPLING] [LOG SPACE DIAGNOSTICS]\n{separator}"
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    means_f = samples.mean(axis=0)
    p, n = cov_log.shape[0], samples.shape[0]
    results['n_samples'] = n
    results['n_parameters'] = p

    # Check for mean deviations in log space
    for dim in range(p):
        var_log = cov_log[dim, dim]
        if var_log == 0.0:
            continue

        mean_th = 1.0
        se_f = np.sqrt((np.exp(var_log) - 1.0) / n)
        z = (means_f[dim] - mean_th) / se_f
        if abs(z) <= z_limit:
            continue

        pair_idx, grp_idx = divmod(dim, num_groups)
        if pair_idx < len(param_pairs):
            zaid, mt = param_pairs[pair_idx]

            if bins is not None and grp_idx < len(bins) - 1:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                param_desc = (f"(ZAID={zaid}, MT={mt}), "
                             f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                param_desc = f"(ZAID={zaid}, MT={mt}), G={grp_idx}"
        else:
            param_desc = f"Parameter {dim}"

        warning_msg = f"Mean deviation in {param_desc}: |z|={abs(z):.2f}>{z_limit} (mean={means_f[dim]:+.3e})"
        results['mean_deviation_warnings'].append(warning_msg)

    # Log space covariance reproduction (clip to avoid -inf from float32 zeros)
    log_samples = np.log(np.maximum(samples, np.finfo(samples.dtype).tiny))
    emp_cov = _empirical_cov(log_samples)
    frob_rel = _relative_frobenius(emp_cov, cov_log)
    results['frobenius_error_pct'] = frob_rel

    # Determine overall status
    n_warnings = len(results['mean_deviation_warnings'])
    if frob_rel > 15.0 or n_warnings > p * 0.05:
        results['overall_status'] = 'FAIL'
    elif frob_rel > 5.0 or n_warnings > 0:
        results['overall_status'] = 'WARN'

    # Consolidated technical summary
    tech_msg = f"  Technical Summary:"
    if logger:
        logger.info(tech_msg)
    else:
        print(tech_msg)
        
    frob_msg = f"    • Log-space covariance reproduction (Frobenius): {frob_rel:.2f}%"
    mean_msg = f"    • Mean deviations: {n_warnings}/{p} parameters outside z±{z_limit}"
    
    for msg in [frob_msg, mean_msg]:
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # User-friendly status
    status_tag = {"PASS": "[GOOD]", "WARN": "[WARN]", "FAIL": "[POOR]"}
    status_msg = f"  {status_tag.get(results['overall_status'], '?')} Overall Status: {results['overall_status']}"
    if logger:
        logger.info(status_msg)
    else:
        print(status_msg)

    # Detailed warnings if any
    if results['mean_deviation_warnings']:
        detail_msg = f"  [WARN] Detailed mean deviation warnings:"
        if logger:
            logger.info(detail_msg)
        else:
            print(detail_msg)
        for w in results['mean_deviation_warnings']:
            warning_detail = f"    • {w}"
            if logger:
                logger.info(warning_detail)
            else:
                print(warning_detail)
    else:
        ok_detail = "  ✓ All sample dimensions within acceptable thresholds."
        if logger:
            logger.info(ok_detail)
        else:
            print(ok_detail)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)
        
    return results


def _diagnostics_covariance(
    cov_lin: np.ndarray,
    param_pairs: List[Tuple[int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    high_var_thr: float = 2.0,
    check_spd: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Diagnostic function for standard covariance matrices.
    Returns a dictionary with diagnostic results for consolidated reporting.
    """
    results = {
        'matrix_size': cov_lin.shape[0],
        'high_variance_warnings': [],
        'spd_status': 'UNKNOWN',
        'min_eigenvalue': None,
        'max_variance': float(np.max(np.diag(cov_lin))),
        'overall_status': 'PASS'
    }
    
    if not verbose:
        return results

    separator = "-" * 60
    log_msg = f"\n[COVARIANCE] [STRUCTURE DIAGNOSTICS]\n{separator}"
    
    # Get logger from shared utils
    logger = _get_logger()
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    diag = np.diag(cov_lin)
    p = cov_lin.shape[0]

    # Check for large variances
    for dim, var in enumerate(diag):
        if var <= high_var_thr:
            continue
        
        pair_idx, grp_idx = divmod(dim, num_groups)
        if pair_idx < len(param_pairs):
            zaid, mt = param_pairs[pair_idx]

            if bins is not None and grp_idx < len(bins) - 1:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                param_desc = (f"(ZAID={zaid}, MT={mt}), "
                             f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                param_desc = f"(ZAID={zaid}, MT={mt}), G={grp_idx}"
        else:
            param_desc = f"Parameter {dim}"

        warning_msg = f"High variance in {param_desc}: σ²={var:.2f}>{high_var_thr}"
        results['high_variance_warnings'].append(warning_msg)

    # SPD test (optional)
    if check_spd:
        try:
            eigenvals = np.linalg.eigvalsh(cov_lin)
            lam_min = float(np.min(eigenvals))
            results['min_eigenvalue'] = lam_min
            
            if lam_min <= 0.0:
                results['spd_status'] = 'FAIL'
                spd_warning = f"Covariance matrix is not SPD (λ_min={lam_min:.3e})"
                results['high_variance_warnings'].append(spd_warning)
            else:
                results['spd_status'] = 'PASS'
        except Exception as e:
            results['spd_status'] = 'ERROR'
            error_msg = f"Could not check eigenvalues: {e}"
            results['high_variance_warnings'].append(error_msg)

    # Determine overall status
    if results['spd_status'] == 'FAIL' or len(results['high_variance_warnings']) > p * 0.1:
        results['overall_status'] = 'WARN'
    elif results['spd_status'] == 'ERROR':
        results['overall_status'] = 'FAIL'

    # Report findings
    if results['high_variance_warnings']:
        warning_msg = f"  {len(results['high_variance_warnings'])} covariance issues detected:"
        if logger:
            logger.info(warning_msg)
        else:
            print(warning_msg)
        for w in results['high_variance_warnings']:
            detail_msg = f"    • {w}"
            if logger:
                logger.info(detail_msg)
            else:
                print(detail_msg)
    else:
        ok_msg = f"  ✓ Covariance matrix structure: {results['overall_status']} (max σ²={results['max_variance']:.2f})"
        if logger:
            logger.info(ok_msg)
        else:
            print(ok_msg)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)
        
    return results

def _diagnostics_endf_covariance(
    cov_lin: np.ndarray,
    param_triplets: List[Tuple[int, int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    high_var_thr: float = 2.0,
    verbose: bool = True,
    logger = None,
) -> Dict[str, Any]:
    """
    Diagnostic function for ENDF covariance matrices.
    Returns a dictionary with diagnostic results for consolidated reporting.
    """
    results = {
        'matrix_size': cov_lin.shape[0],
        'high_variance_warnings': [],
        'spd_status': 'UNKNOWN',
        'min_eigenvalue': None,
        'max_variance': float(np.max(np.diag(cov_lin))),
        'overall_status': 'PASS'
    }
    
    if not verbose:
        return results

    separator = "-" * 60
    log_msg = f"\n[ENDF COVARIANCE] [STRUCTURE DIAGNOSTICS]\n{separator}"
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    diag = np.diag(cov_lin)
    p = cov_lin.shape[0]

    # Check for large variances
    for dim, var in enumerate(diag):
        if var <= high_var_thr:
            continue
        
        # Fixed logic: For ENDF MF34 data, parameters are organized as:
        # [triplet0_group0, triplet0_group1, ..., triplet1_group0, triplet1_group1, ...]
        # So we need to find which triplet and which group within that triplet
        if len(param_triplets) > 0 and num_groups > 0:
            triplet_idx = dim // num_groups
            grp_idx = dim % num_groups
        else:
            # Fallback for edge cases
            triplet_idx = 0
            grp_idx = dim
        
        if triplet_idx < len(param_triplets):
            isotope, mt, legendre = param_triplets[triplet_idx]
            
            if bins is not None and grp_idx < len(bins) - 1:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                param_desc = (f"(ISO={isotope}, MT={mt}, L={legendre}), "
                             f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                param_desc = f"(ISO={isotope}, MT={mt}, L={legendre}), G={grp_idx}"
        else:
            param_desc = f"Parameter {dim}"

        warning_msg = f"High variance in {param_desc}: σ²={var:.2f}>{high_var_thr}"
        results['high_variance_warnings'].append(warning_msg)

    # Check if matrix is positive semi-definite
    try:
        eigenvals = np.linalg.eigvalsh(cov_lin)
        lam_min = float(np.min(eigenvals))
        results['min_eigenvalue'] = lam_min
        
        if lam_min <= 0.0:
            results['spd_status'] = 'FAIL'
            spd_warning = f"Covariance matrix is not SPD (λ_min={lam_min:.3e})"
            results['high_variance_warnings'].append(spd_warning)
        else:
            results['spd_status'] = 'PASS'
    except Exception as e:
        results['spd_status'] = 'ERROR'
        error_msg = f"Could not check eigenvalues: {e}"
        results['high_variance_warnings'].append(error_msg)

    # Determine overall status
    if results['spd_status'] == 'FAIL' or len(results['high_variance_warnings']) > p * 0.1:
        results['overall_status'] = 'WARN'
    elif results['spd_status'] == 'ERROR':
        results['overall_status'] = 'FAIL'

    # Report findings
    if results['high_variance_warnings']:
        warning_msg = f"  {len(results['high_variance_warnings'])} covariance issues detected:"
        if logger:
            logger.info(warning_msg)
        else:
            print(warning_msg)
        for w in results['high_variance_warnings']:
            detail_msg = f"    • {w}"
            if logger:
                logger.info(detail_msg)
            else:
                print(detail_msg)
    else:
        ok_msg = f"  ✓ Covariance matrix structure: {results['overall_status']} (max σ²={results['max_variance']:.2f})"
        if logger:
            logger.info(ok_msg)
        else:
            print(ok_msg)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)
        
    return results


def _diagnostics_endf_samples_linear(
    samples: np.ndarray,
    cov_lin: np.ndarray,
    param_triplets: List[Tuple[int, int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    z_limit: float = 3.0,
    verbose: bool = True,
    logger = None,
) -> Dict[str, Any]:
    """
    Linear space sample diagnostics for ENDF data.
    Returns consolidated diagnostic results.
    """
    results = {
        'n_samples': samples.shape[0],
        'n_parameters': samples.shape[1],
        'mean_deviation_warnings': [],
        'frobenius_error_pct': None,
        'space': 'linear',
        'overall_status': 'PASS'
    }
    
    if not verbose:
        # Still compute basic metrics for downstream use
        try:
            emp_cov = _empirical_cov(samples)
            results['frobenius_error_pct'] = _relative_frobenius(emp_cov, cov_lin)
            metrics = _diagnostics_whiten_mahalanobis(
                samples, cov_lin, space="linear",
                z_limit=z_limit, coverage_alpha=0.95, verbose=False, logger=logger
            )
            results.update(metrics)
        except:
            pass
        return results

    # Skip verbose intermediate logging - only show final quality summary
    means_f = samples.mean(axis=0)
    p, n = cov_lin.shape[0], samples.shape[0]
    results['n_samples'] = n
    results['n_parameters'] = p

    # NOTE: For ENDF Legendre coefficients, negative factors are physically valid
    # so we skip the negative factor check that is used for cross-sections

    # Check means significantly off 1
    for dim in range(p):
        var_lin = cov_lin[dim, dim]
        if var_lin == 0.0:
            continue
        
        se_f = np.sqrt(var_lin / n)
        z = (means_f[dim] - 1.0) / se_f
        if abs(z) <= z_limit:
            continue

        # Fixed logic: Consistent with covariance diagnostics
        if len(param_triplets) > 0 and num_groups > 0:
            triplet_idx = dim // num_groups
            grp_idx = dim % num_groups
        else:
            triplet_idx = 0
            grp_idx = dim
        
        if triplet_idx < len(param_triplets):
            isotope, mt, legendre = param_triplets[triplet_idx]
            
            if bins is not None and grp_idx < len(bins) - 1:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                param_desc = (f"(ISO={isotope}, MT={mt}, L={legendre}), "
                             f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                param_desc = f"(ISO={isotope}, MT={mt}, L={legendre}), G={grp_idx}"
        else:
            param_desc = f"Parameter {dim}"

        warning_msg = f"Mean deviation in {param_desc}: |z|={abs(z):.2f}>{z_limit} (mean={means_f[dim]:+.3e})"
        results['mean_deviation_warnings'].append(warning_msg)

    # Full-matrix reproduction check
    try:
        emp_cov = _empirical_cov(samples)
    except Exception:
        emp_cov = np.cov(samples, rowvar=False, ddof=1)

    frob_rel = _relative_frobenius(emp_cov, cov_lin)
    results['frobenius_error_pct'] = frob_rel

    # Collect whitening/Mahalanobis metrics
    metrics = _diagnostics_whiten_mahalanobis(
        samples, cov_lin, space="linear",
        z_limit=z_limit, coverage_alpha=0.95, verbose=verbose, logger=logger
    )
    results.update(metrics)

    # Determine overall status
    n_warnings = len(results['mean_deviation_warnings'])
    if (frob_rel > 15.0 or n_warnings > p * 0.05 or 
        metrics.get('coverage_status', 'PASS') == 'FAIL'):
        results['overall_status'] = 'FAIL'
    elif (frob_rel > 5.0 or n_warnings > 0 or 
          metrics.get('coverage_status', 'PASS') == 'WARN'):
        results['overall_status'] = 'WARN'

    # Skip intermediate technical summary - only show final quality assessment
    comprehensive_status = _log_fast_qc_summary(
        space="linear",
        frob_cov_pct=frob_rel,
        mean_dev_warns=n_warnings,
        total_dims=p,
        metrics=metrics,
        logger=logger
    )
    
    # Store the comprehensive overall status (EXCELLENT/GOOD/ACCEPTABLE/POOR/CRITICAL)
    # This overrides the basic status (PASS/WARN/FAIL) with the comprehensive assessment
    if comprehensive_status:
        results['overall_status'] = comprehensive_status
    
    return results


def _diagnostics_endf_samples_log(
    samples: np.ndarray,
    cov_log: np.ndarray,
    param_triplets: List[Tuple[int, int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    z_limit: float = 3.0,
    verbose: bool = True,
    logger = None,
) -> Dict[str, Any]:
    """
    Log space sample diagnostics for ENDF data.
    Returns consolidated diagnostic results.
    """
    results = {
        'n_samples': samples.shape[0],
        'n_parameters': samples.shape[1],
        'mean_deviation_warnings': [],
        'frobenius_error_pct': None,
        'space': 'log',
        'overall_status': 'PASS'
    }
    
    if not verbose:
        # Still compute basic metrics for downstream use
        try:
            emp_cov = _empirical_cov(np.log(samples))
            results['frobenius_error_pct'] = _relative_frobenius(emp_cov, cov_log)
            metrics = _diagnostics_whiten_mahalanobis(
                samples, cov_log, space="log",
                z_limit=z_limit, coverage_alpha=0.95, verbose=False, logger=logger
            )
            results.update(metrics)
        except:
            pass
        return results

    # Skip verbose intermediate logging - only show final quality summary
    means_f = samples.mean(axis=0)
    p, n = cov_log.shape[0], samples.shape[0]
    results['n_samples'] = n
    results['n_parameters'] = p

    # Check for mean deviations in log space
    for dim in range(p):
        var_log = cov_log[dim, dim]
        if var_log == 0.0:
            continue

        mean_th = 1.0
        se_f = np.sqrt((np.exp(var_log) - 1.0) / n)
        z = (means_f[dim] - mean_th) / se_f
        if abs(z) <= z_limit:
            continue

        # Fixed logic: Consistent with other ENDF functions
        if len(param_triplets) > 0 and num_groups > 0:
            triplet_idx = dim // num_groups
            grp_idx = dim % num_groups
        else:
            triplet_idx = 0
            grp_idx = dim
        
        if triplet_idx < len(param_triplets):
            isotope, mt, legendre = param_triplets[triplet_idx]
            
            if bins is not None and grp_idx < len(bins) - 1:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                param_desc = (f"(ISO={isotope}, MT={mt}, L={legendre}), "
                             f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                param_desc = f"(ISO={isotope}, MT={mt}, L={legendre}), G={grp_idx}"
        else:
            param_desc = f"Parameter {dim}"

        warning_msg = f"Mean deviation in {param_desc}: |z|={abs(z):.2f}>{z_limit} (mean={means_f[dim]:+.3e})"
        results['mean_deviation_warnings'].append(warning_msg)

    # Log space covariance reproduction
    try:
        emp_cov = _empirical_cov(np.log(samples))
    except Exception:
        emp_cov = np.cov(np.log(samples), rowvar=False, ddof=1)

    frob_rel = _relative_frobenius(emp_cov, cov_log)
    results['frobenius_error_pct'] = frob_rel

    # Collect whitening/Mahalanobis metrics
    metrics = _diagnostics_whiten_mahalanobis(
        samples, cov_log, space="log",
        z_limit=z_limit, coverage_alpha=0.95, verbose=verbose, logger=logger
    )
    results.update(metrics)

    # Determine overall status
    n_warnings = len(results['mean_deviation_warnings'])
    if (frob_rel > 15.0 or n_warnings > p * 0.05 or 
        metrics.get('coverage_status', 'PASS') == 'FAIL'):
        results['overall_status'] = 'FAIL'
    elif (frob_rel > 5.0 or n_warnings > 0 or 
          metrics.get('coverage_status', 'PASS') == 'WARN'):
        results['overall_status'] = 'WARN'

    # Skip intermediate technical summary - only show final quality assessment
    comprehensive_status = _log_fast_qc_summary(
        space="log",
        frob_cov_pct=frob_rel,
        mean_dev_warns=n_warnings,
        total_dims=p,
        metrics=metrics,
        logger=logger
    )
    
    # Store the comprehensive overall status (EXCELLENT/GOOD/ACCEPTABLE/POOR/CRITICAL)
    # This overrides the basic status (PASS/WARN/FAIL) with the comprehensive assessment
    if comprehensive_status:
        results['overall_status'] = comprehensive_status
    
    return results


def _whitening_matrix_from_cov(
    cov: np.ndarray,
    *,
    clip_tol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, int]:
    """
    Build effective whitening transformation via eigen-decomposition with safe clipping.
    
    Returns
    -------
    Tuple containing:
        - V_eff: (p, k_eff) eigenvectors for effective subspace  
        - lam_eff: (k_eff,) eigenvalues for effective subspace
        - L_eff_inv: (k_eff, k_eff) inverse sqrt of effective eigenvalues
        - k_eff: number of effective dimensions
        - lam_min: minimum eigenvalue before clipping
        - n_clipped: number of clipped eigenvalues
    """
    # Symmetric PSD expected; use eigh
    lam, V = np.linalg.eigh(cov)
    lam_max = float(np.max(lam)) if lam.size else 0.0
    if clip_tol is None:
        # Tolerance proportional to scale; keeps things stable for near-singular Σ
        clip_tol = max(1e-12, 1e-12 * lam_max)

    # Count/eject tiny or negative eigenvalues
    mask = lam > clip_tol
    k_eff = int(np.sum(mask))
    n_clipped = int(np.size(lam) - k_eff)
    lam_min = float(np.min(lam)) if lam.size else np.nan

    if k_eff == 0:
        # Degenerate; return zeros so downstream warns loudly
        return (np.zeros((cov.shape[0], 0)), np.array([]), np.zeros((0, 0)), 
                0, lam_min, n_clipped)

    # Return components for effective subspace whitening
    V_eff = V[:, mask]  # (p, k_eff)
    lam_eff = lam[mask]  # (k_eff,)
    L_eff_inv = np.diag(1.0 / np.sqrt(lam_eff))  # (k_eff, k_eff)
    
    return V_eff, lam_eff, L_eff_inv, k_eff, lam_min, n_clipped

# ---------- Normal quantile (Acklam) ----------
def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF for p in (0,1). ~1e-9 rel. acc."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    # Coefficients for Acklam's rational approximation
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow  = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2*math.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return -num/den
    elif p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return num/den
    else:
        q = p - 0.5
        r = q*q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q
        den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
        return num/den

# ---------- Regularized lower incomplete gamma P(a, x) ----------
def _gammainc_lower_reg(a: float, x: float, eps: float = 1e-12, maxiter: int = 10000) -> float:
    """P(a,x) = γ(a,x)/Γ(a) using series (x<a+1) or continued fraction (x≥a+1)."""
    if x <= 0:
        return 0.0
    if a <= 0:
        raise ValueError("a must be > 0")
    lg = math.lgamma(a)
    # Series expansion
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(maxiter):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * eps:
                break
        return summ * math.exp(-x + a*math.log(x) - lg)
    # Continued fraction (Lentz)
    b = x + 1.0 - a
    c = 1e308
    d = 1.0 / max(b, 1e-308)
    h = d
    for i in range(1, maxiter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-308: d = 1e-308
        c = b + an / c
        if abs(c) < 1e-308: c = 1e-308
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return 1.0 - h * math.exp(-x + a*math.log(x) - lg)

# ---------- Chi-square PDF/CDF ----------
def _chi2_pdf(x: float, k: int) -> float:
    if x <= 0 or k <= 0: return 0.0
    a = 0.5 * k
    # log form for stability
    logf = (a - 1.0) * math.log(x) - 0.5*x - a*math.log(2.0) - math.lgamma(a)
    return math.exp(logf)

def _chi2_cdf(x: float, k: int) -> float:
    if x <= 0: return 0.0
    a = 0.5 * k
    return _gammainc_lower_reg(a, 0.5 * x)

# ---------- Chi-square quantile ----------
def _chi2_ppf(alpha: float, k: int) -> float:
    """
    Proper χ² quantile Q with P[X<=Q]=alpha, dof=k.
    Tries SciPy if available; otherwise Newton refinement from Wilson–Hilferty
    with accurate CDF/PDF above. Works well for all practical k.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if k <= 0:
        return 0.0
    # Fast path: SciPy if present
    try:
        from scipy.stats import chi2 as _chi2
        return float(_chi2.ppf(alpha, k))
    except Exception:
        pass
    # Initial guess: Wilson–Hilferty
    z = _norm_ppf(alpha)
    c = 2.0 / (9.0 * k)
    x = k * (1.0 - c + z * math.sqrt(c))**3
    x = max(x, 1e-12)
    # Newton iterations on F(x)-alpha=0
    for _ in range(8):
        f  = _chi2_cdf(x, k) - alpha
        fp = _chi2_pdf(x, k)
        if fp <= 0:
            break
        step = f / fp
        x_new = x - step
        if x_new <= 0:
            x_new = x / 2.0
        if abs(step) <= 1e-10 * max(1.0, x_new):
            x = x_new
            break
        x = x_new
    return float(max(x, 0.0))

# ---------- Backward compatibility name (now proper) ----------
def _wilson_hilferty_chi2_q(alpha: float, k: int) -> float:
    """
    Kept for compatibility: now delegates to proper χ² quantile.
    """
    return _chi2_ppf(alpha, k)


def _binomial_three_sigma_band(n: int, p: float = 0.95) -> float:
    """Return ±3σ band (in percentage points) for binomial(n,p)."""
    if n <= 0:
        return 100.0
    var = p * (1 - p) / n
    return 3.0 * np.sqrt(var) * 100.0


def _diagnostics_whiten_mahalanobis(
    factors: np.ndarray,
    cov: np.ndarray,
    *,
    space: str,
    z_limit: float = 3.0,
    coverage_alpha: float = 0.95,
    verbose: bool = True,
    logger = None,
) -> dict:
    """
    Core quality checks with proper effective subspace whitening:
      1) Whitening: u_eff = Σ^{-1/2}_eff δ  ~ N(0, I_k_eff) in effective subspace
      2) Mahalanobis: m² = ||u_eff||²  ~ χ²_k_eff
    Returns a metrics dict for FAST CHECK summary.
    """
    if not verbose:
        # still compute metrics silently
        pass

    # Skip intermediate verbose logging - only compute metrics for final summary
    n, p = factors.shape

    # Build delta from multiplicative factors
    if space.lower() == "linear":
        delta = factors - 1.0
    else:
        m = -0.5 * np.diag(cov)
        delta = np.log(factors) - m  # equals Y

    # Get effective subspace components
    V_eff, lam_eff, L_eff_inv, k_eff, lam_min, n_clipped = _whitening_matrix_from_cov(cov)

    if k_eff == 0:
        # Skip verbose error logging
        return {
            "n": int(n), "p": int(p), "k_eff": 0, "lambda_min": lam_min,
            "n_clipped": int(n_clipped), "whiten_rel_frob": np.inf,
            "max_offdiag_corr": np.inf, "worst_z_mean_u": np.inf,
            "m2_mean": np.nan, "m2_var": np.nan, "chi2_q": np.nan,
            "coverage_alpha": coverage_alpha, "covered_pct": 0.0,
            "covered_band_pp": 100.0, "coverage_status": "FAIL"
        }

    # Skip verbose eigenvalue clipping notifications

    # CORRECTED: Work in effective k_eff-dimensional subspace only
    # u_eff = delta @ V_eff @ L_eff_inv  -> (n, k_eff)
    U_eff = delta @ V_eff @ L_eff_inv

    # Whitening diagnostics: Compare Cov(U_eff) to I_k_eff (not I_p!)
    emp_cov_u_eff = _empirical_cov(U_eff)
    I_eff = np.eye(k_eff, dtype=emp_cov_u_eff.dtype)
    frob_rel_u = _relative_frobenius(emp_cov_u_eff, I_eff)  # percent

    # Correlation analysis in effective subspace
    std_u_eff = np.sqrt(np.clip(np.diag(emp_cov_u_eff), 1e-300, None))
    corr_u_eff = emp_cov_u_eff / np.outer(std_u_eff, std_u_eff)
    np.fill_diagonal(corr_u_eff, 0.0)
    max_off = float(np.nanmax(np.abs(corr_u_eff))) if k_eff > 1 else 0.0

    # Mean analysis in effective subspace
    means_u_eff = U_eff.mean(axis=0)
    se_mean = 1.0 / np.sqrt(n)
    z_means = np.abs(means_u_eff) / se_mean
    worst_z = float(np.max(z_means)) if k_eff > 0 else 0.0

    # Skip verbose whitening and Mahalanobis logging
    
    # Mahalanobis distance: ||u_eff||² ~ χ²_k_eff
    m2 = np.einsum('ij,ij->i', U_eff, U_eff)
    mean_m2 = float(np.mean(m2))
    var_m2 = float(np.var(m2, ddof=1))
    k = int(k_eff)
    exp_mean, exp_var = float(k), float(2*k)
    q = _chi2_ppf(coverage_alpha, k)
    covered = float(np.mean(m2 <= q)) * 100.0
    band = _binomial_three_sigma_band(n, p=coverage_alpha)
    low, high = 100*(coverage_alpha) - band, 100*(coverage_alpha) + band
    status = "PASS" if (covered >= low and covered <= high) else "WARN"

    # Skip verbose Mahalanobis logging

    return {
        "n": int(n), "p": int(p), "k_eff": k, "lambda_min": lam_min,
        "n_clipped": int(n_clipped), "whiten_rel_frob": float(frob_rel_u),
        "max_offdiag_corr": float(max_off), "worst_z_mean_u": float(worst_z),
        "m2_mean": mean_m2, "m2_var": var_m2, "chi2_q": float(q),
        "coverage_alpha": float(coverage_alpha), "covered_pct": float(covered),
        "covered_band_pp": float(band), "coverage_status": status
    }


def _classify_sampling_regime(n: int, p: int, k_eff: int) -> dict:
    """Classify sampling regime based on sample-to-parameter ratio."""
    ratio = n / max(p, 1)
    expected_max_corr = math.sqrt(2 * math.log(max(k_eff, 2)) / max(n, 2))
    if n >= 10 * p:
        regime = "well_sampled"
        label = f"Well-sampled (n/p = {ratio:.0f})"
    elif n >= p:
        regime = "moderate"
        label = f"Moderately sampled (n/p = {ratio:.1f})"
    else:
        regime = "high_dimensional"
        label = f"High-dimensional (n={n} < p={p})"
    return {
        "regime": regime, "label": label, "ratio": ratio,
        "expected_max_corr": expected_max_corr,
        "n": n, "p": p, "k_eff": k_eff,
    }


def _log_fast_qc_summary(
    *,
    space: str,
    frob_cov_pct: float,
    mean_dev_warns: int,
    total_dims: int,
    metrics: dict,
    logger=None
):
    """
    Provide a clear, user-friendly sampling quality summary.

    The assessment distinguishes between:
      - Tier 1 (Sampling Correctness): metrics that verify whether the
        sampling procedure produced correct draws from the target
        distribution. These are valid regardless of sample size.
      - Tier 2 (Covariance Reconstruction): metrics that measure how well
        the original covariance can be recovered from the samples. These
        require n >> p to be meaningful and do NOT indicate sampling errors
        when n < p.

    The overall verdict is driven by Tier 1. Tier 2 is informational.
    """
    Z_LIMIT = 3.0

    n = metrics["n"]
    p = metrics["p"]
    k_eff = metrics["k_eff"]
    wht = metrics["whiten_rel_frob"]
    off = metrics["max_offdiag_corr"]
    wz = metrics["worst_z_mean_u"]
    covered_pct = metrics["covered_pct"]
    expected_coverage = metrics["coverage_alpha"] * 100
    coverage_band = metrics["covered_band_pp"]

    regime_info = _classify_sampling_regime(n, p, k_eff)
    regime = regime_info["regime"]

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _emit(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    def evaluate_metric(value, excellent, good, warn, poor):
        if value <= excellent:
            return "EXCELLENT"
        elif value <= good:
            return "GOOD"
        elif value <= warn:
            return "ACCEPTABLE"
        elif value <= poor:
            return "POOR"
        else:
            return "CRITICAL"

    quality_tags = {
        "EXCELLENT": "[EXCELLENT]",
        "GOOD": "[GOOD]",
        "ACCEPTABLE": "[WARN]",
        "POOR": "[POOR]",
        "CRITICAL": "[CRITICAL]",
    }

    # ------------------------------------------------------------------
    # Tier 1 — Sampling Correctness (always meaningful)
    # ------------------------------------------------------------------
    # Coverage
    in_range = abs(covered_pct - expected_coverage) <= coverage_band
    cover_quality = "GOOD" if in_range else "POOR"

    # Z-score
    z_quality = "GOOD" if wz <= Z_LIMIT else ("ACCEPTABLE" if wz <= Z_LIMIT + 1.0 else "POOR")

    # Mean deviation
    mean_dev_rate = mean_dev_warns / max(total_dims, 1) * 100
    if mean_dev_rate == 0:
        mean_quality = "EXCELLENT"
    elif mean_dev_rate <= 1:
        mean_quality = "GOOD"
    elif mean_dev_rate <= 5:
        mean_quality = "ACCEPTABLE"
    else:
        mean_quality = "POOR"

    tier1 = [cover_quality, z_quality, mean_quality]
    quality_scores = {
        "EXCELLENT": 5, "GOOD": 4, "ACCEPTABLE": 3, "POOR": 2, "CRITICAL": 1
    }
    tier1_score = np.mean([quality_scores[q] for q in tier1])

    # ------------------------------------------------------------------
    # Tier 2 — Covariance Reconstruction (informational when n < p)
    # ------------------------------------------------------------------
    # Thresholds depend on regime
    if regime == "well_sampled":
        TH_FROB = (1.0, 5.0, 15.0, 30.0)
        TH_WHT = (2.0, 5.0, 15.0, 30.0)
        TH_OFF = (0.05, 0.15, 0.30, 0.60)
    elif regime == "moderate":
        TH_FROB = (3.0, 10.0, 25.0, 50.0)
        TH_WHT = (3.0, 8.0, 20.0, 40.0)
        TH_OFF = (0.10, 0.25, 0.45, 0.80)
    else:  # high_dimensional
        TH_FROB = None  # not meaningful when n < p
        TH_WHT = (5.0, 15.0, 30.0, 60.0)
        # Scale off-diagonal thresholds by expected spurious baseline
        base = regime_info["expected_max_corr"]
        TH_OFF = tuple(
            max(t, base * m)
            for t, m in zip((0.05, 0.15, 0.30, 0.60), (2, 3, 4, 6))
        )

    cov_quality = evaluate_metric(frob_cov_pct, *TH_FROB) if TH_FROB else None
    wht_quality = evaluate_metric(wht, *TH_WHT)
    off_quality = evaluate_metric(off, *TH_OFF)

    tier2 = [q for q in [cov_quality, wht_quality, off_quality] if q is not None]
    tier2_score = np.mean([quality_scores[q] for q in tier2]) if tier2 else None

    # ------------------------------------------------------------------
    # Overall verdict — driven by Tier 1
    # In the well-sampled regime, Tier 2 can also downgrade.
    # ------------------------------------------------------------------
    if regime == "well_sampled":
        all_q = tier1 + tier2
        avg = np.mean([quality_scores[q] for q in all_q])
        crit = sum(1 for q in all_q if q == "CRITICAL")
        poor = sum(1 for q in all_q if q in ("CRITICAL", "POOR"))
    else:
        avg = tier1_score
        crit = sum(1 for q in tier1 if q == "CRITICAL")
        poor = sum(1 for q in tier1 if q in ("CRITICAL", "POOR"))

    if crit > 0 or avg < 2.5:
        overall_status = "CRITICAL"
    elif poor >= 2 or avg < 3.0:
        overall_status = "POOR"
    elif avg < 3.5:
        overall_status = "ACCEPTABLE"
    elif avg < 4.5:
        overall_status = "GOOD"
    else:
        overall_status = "EXCELLENT"

    # ------------------------------------------------------------------
    # Log output
    # ------------------------------------------------------------------
    sep = "=" * 60

    _emit(f"\n{sep}")
    _emit(f"SAMPLING QUALITY ASSESSMENT ({space.upper()} SPACE)")
    _emit(sep)
    _emit(f"  Samples: {n}  |  Parameters: {p}  |  Effective rank: {k_eff}")
    _emit(f"  Regime: {regime_info['label']}")

    # --- Tier 1: Sampling Correctness ---
    _emit(f"\n  SAMPLING CORRECTNESS: {overall_status}")

    if overall_status in ("GOOD", "EXCELLENT"):
        _emit("    Samples are valid draws from the target distribution.")
    elif overall_status == "ACCEPTABLE":
        _emit("    Sampling appears correct with minor statistical deviations.")
    elif overall_status == "POOR":
        _emit("    Some correctness metrics show significant deviations.")
        _emit("    Results should be interpreted with care. Consider increasing sample size.")
    else:
        _emit("    Correctness metrics indicate a problem with the sampling procedure.")
        _emit("    Check the covariance matrix and decomposition. Increase sample size.")

    _emit(f"    - Mean Consistency:     {quality_tags[z_quality]} {z_quality} (max |z|={wz:.1f})")
    _emit(f"    - Parameter Deviations: {quality_tags[mean_quality]} {mean_quality} ({mean_dev_warns}/{total_dims} outside 3-sigma)")
    _emit(f"    - Chi-squared Coverage: {quality_tags[cover_quality]} {cover_quality} ({covered_pct:.1f}% coverage, expected ~{expected_coverage:.0f}%)")

    # --- Tier 2: Covariance Reconstruction ---
    if regime == "high_dimensional":
        _emit(f"\n  COVARIANCE RECONSTRUCTION FROM SAMPLES:")
        _emit(f"    These metrics measure whether the original covariance matrix can")
        _emit(f"    be recovered by computing the sample covariance from the generated")
        _emit(f"    samples. This is NOT the same as whether the samples are correct.")
        _emit(f"")
        _emit(f"    The samples were generated FORWARD: the Cholesky decomposition of")
        _emit(f"    the target covariance was used to transform independent random draws")
        _emit(f"    into correlated samples. This guarantees each sample is a correct")
        _emit(f"    draw from the target distribution.")
        _emit(f"")
        _emit(f"    Reconstructing the covariance BACKWARD (from samples) is a harder")
        _emit(f"    statistical problem that requires n >> p. With {n} samples and")
        _emit(f"    {p} parameters, the sample covariance has rank <= {min(n-1, p)} and")
        _emit(f"    cannot reproduce all {p}x{p} entries. This does NOT affect the")
        _emit(f"    validity of the samples for uncertainty propagation.")
        _emit(f"")
        if cov_quality is None:
            _emit(f"    - Covariance Reproduction: {frob_cov_pct:.1f}% Frobenius error (not scored, rank-deficient)")
        else:
            _emit(f"    - Covariance Reproduction: {quality_tags[cov_quality]} {cov_quality} ({frob_cov_pct:.1f}%)")
        _emit(f"    - Statistical Whitening:   {wht:.1f}% in {k_eff}-dim subspace (elevated, expected for n < p)")
        _emit(f"    - Correlation Structure:   max |r|={off:.3f} (expected baseline ~{regime_info['expected_max_corr']:.2f} for n={n}, k={k_eff})")
    else:
        _emit(f"\n  COVARIANCE RECONSTRUCTION FROM SAMPLES:")
        if cov_quality is not None:
            _emit(f"    - Covariance Reproduction: {quality_tags[cov_quality]} {cov_quality} ({frob_cov_pct:.1f}%)")
        _emit(f"    - Statistical Whitening:   {quality_tags[wht_quality]} {wht_quality} ({wht:.1f}%)")
        _emit(f"    - Correlation Structure:   {quality_tags[off_quality]} {off_quality} (max |r|={off:.3f})")

    # --- Recommendation ---
    _emit(f"\n  RECOMMENDATION:")
    if overall_status in ("GOOD", "EXCELLENT"):
        _emit(f"    These samples are suitable for uncertainty propagation.")
        if regime == "high_dimensional":
            _emit(f"    Individual parameter uncertainties are well captured. To also")
            _emit(f"    verify the full correlation structure from the samples, increase")
            _emit(f"    sample count to at least {p} (n >= p).")
        elif regime == "moderate":
            _emit(f"    To improve covariance reconstruction fidelity, increase samples.")
    elif overall_status == "ACCEPTABLE":
        _emit(f"    Samples are usable for uncertainty propagation with minor caveats.")
        _emit(f"    Increase sample size if higher statistical confidence is needed.")
    elif overall_status == "POOR":
        _emit(f"    Use with caution. Some correctness metrics deviate significantly.")
        _emit(f"    Consider increasing the sample size or checking the input covariance.")
    else:
        _emit(f"    Sampling correctness could not be verified. Check the covariance")
        _emit(f"    matrix, decomposition method, and sampling parameters before using")
        _emit(f"    these samples. Increase sample size.")

    _emit(sep)

    return overall_status
