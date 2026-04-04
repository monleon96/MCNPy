import numpy as np
import os
import shutil
import logging
import pandas as pd
import json
import time
from typing import List, Union, Optional, Dict, Tuple
from multiprocessing import Pool
from datetime import datetime

from kika.sampling.generators import generate_samples
from kika.ace.parsers import read_ace
from kika.ace.writers.write_ace import write_ace
from kika._constants import MT_GROUPS
from kika._utils import MeV_to_kelvin
from kika.ace.xsdir import create_xsdir_files_for_ace
from kika.sampling.utils import (
    DualLogger, 
    _get_logger, 
    _set_logger,
    load_covariance,
    _initialize_master_perturbation_matrix,
    _update_master_perturbation_matrix,
    _finalize_master_perturbation_matrix
)


def _process_sample(
    ace_file: str,
    sample: np.ndarray,
    sample_index: int,
    energy_grid: List[float],
    mt_numbers: List[int],
    output_dir: str,
    xsdir_file: Optional[str],
):
    # — read & perturb ACE —
    ace = read_ace(ace_file)
    apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers, False)  # Set verbose=False to reduce output

    # — write perturbed ACE —
    base, ext = os.path.splitext(os.path.basename(ace_file))
    sample_str = f"{sample_index+1:04d}"
    
    # Convert temperature from MeV to Kelvin for directory structure
    hdr = ace.header
    temp_K = MeV_to_kelvin(hdr.temperature)
    
    # Use exact temperature formatting to match ENDF perturbation directory structure
    temp_str = str(temp_K).rstrip('0').rstrip('.') if '.' in str(temp_K) else str(temp_K)
    
    # Create directory structure: output_dir/ace/temp/zaid/sample_num/
    sample_dir = os.path.join(output_dir, "ace", temp_str, str(ace.zaid), sample_str)
    os.makedirs(sample_dir, exist_ok=True)
    out_ace = os.path.join(sample_dir, f"{base}_{sample_str}{ext}")
    
    # Recalculate cross sections but don't print
    ace.update_cross_sections()
    
    # Write ACE file without verbose output
    write_ace(ace, out_ace, overwrite=True)

    # — write xsdir files using shared function —
    has_ptable = bool(getattr(ace.unresolved_resonance, 'has_data', False))
    
    create_xsdir_files_for_ace(
        ace_file_path=out_ace,
        zaid=hdr.zaid,
        awr=hdr.atomic_weight_ratio,
        xss_len=hdr.nxs_array[1],
        temperature_mev=hdr.temperature,
        sample_index=sample_index,
        output_dir=output_dir,
        master_xsdir_file=xsdir_file,
        has_ptable=has_ptable,
    )


def perturb_ACE_files(
    ace_files: Union[str, List[str]],
    cov_files: Union[str, List[str]],
    mt_list: List[int],
    num_samples: int,
    space: str = "log",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    output_dir: str = '.',
    xsdir_file: Optional[str] = None,
    seed: Optional[int] = None,
    nprocs: int = 1,
    dry_run: bool = False,
    autofix: Optional[str] = None, 
    high_val_thresh: float = 1.0,
    accept_tol: float = -1.0e-4,
    remove_blocks: Optional[Dict[int, Union[Tuple[int, int], List[Tuple[int, int]]]]] = None,
    verbose: bool = True,
    energy_ranges: Optional[List[Tuple[float, float]]] = None,
    psd_method: str = "higham",
    max_relative_std: Optional[float] = 3.0,
):
    """
    Perturb ACE nuclear data files using covariance matrices.

    This function generates perturbed ACE files by sampling perturbation factors
    from multivariate normal distributions derived from covariance matrices.

    Parameters
    ----------
    ace_files : Union[str, List[str]]
        Path(s) to ACE file(s) to be perturbed
    cov_files : Union[str, List[str]]
        Path(s) to covariance matrix file(s) (SCALE or NJOY format).
        Can be a single file (used for all ACE files) or one file per ACE file.
    mt_list : List[int]
        List of MT reaction numbers to perturb. Empty list means all available MTs
    num_samples : int
        Number of perturbed ACE files to generate
    space : str, default "log"
        Sampling space: "linear" (factors = 1 + X) or "log" (factors = exp(Y))
    decomposition_method : str, default "svd"
        Matrix decomposition method: "svd", "cholesky", "eigen", or "pca"
    sampling_method : str, default "sobol"
        Sampling method: "sobol", "lhs", or "random"
    output_dir : str, default "."
        Output directory for perturbed files
    xsdir_file : Optional[str], default None
        Path to master XSDIR file to be modified for each sample
    seed : Optional[int], default None
        Random seed for reproducible sampling
    nprocs : int, default 1
        Number of parallel processes (currently unused)
    dry_run : bool, default False
        If True, only generate perturbation factors without creating ACE files
    autofix : Optional[str], default None
        Covariance matrix fixing level:
        - None: No autofix - use covariance matrix as-is
        - "soft": Clamp diagonal variances only
        - "medium": Clamp variances then remove worst block pairs
        - "hard": Clamp variances then remove worst reactions entirely
    high_val_thresh : float, default 1.0
        Threshold for identifying problematic covariance values during autofix
    accept_tol : float, default -1.0e-4
        Minimum eigenvalue threshold for accepting the covariance matrix
    remove_blocks : Optional[Dict[int, Union[Tuple[int, int], List[Tuple[int, int]]]]], default None
        Manual specification of covariance blocks to remove by isotope
    verbose : bool, default True
        Enable verbose logging output
    energy_ranges : Optional[List[Tuple[float, float]]], default None
        List of (emin, emax) energy ranges in eV. Only energy bins overlapping
    max_relative_std : float or None, default 3.0
        Maximum relative standard deviation (e.g. 3.0 = 300%). Variances
        exceeding this cap are reduced via a correlation-preserving congruence
        transform. Common near threshold reactions. Set to None to disable.
        these ranges will be perturbed; bins outside are set to neutral (1.0 for
        linear space, 0.0 for log space). None means perturb all energies.

    Notes
    -----
    When autofix=None, the covariance matrix is used exactly as read from the file,
    without any modifications. This may result in decomposition failures if the
    matrix is not positive semi-definite, but preserves the original uncertainties
    and correlations exactly as specified in the nuclear data evaluation.
    """
    global _logger
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'ace_perturbation_{timestamp}.log')
    _logger = DualLogger(log_file)
    _set_logger(_logger)
    
    # Console: Basic start message
    _logger.info(f"[INFO] [ACE] Starting ACE perturbation job", console=True)
    _logger.info(f"[INFO] [ACE] Log file: {log_file}", console=True)
    _logger.info(f"[INFO] [ACE] Output directory: {os.path.abspath(output_dir)}", console=True)

    # Print run parameters as metadata TO LOG FILE
    _logger.info(f"\n#== CONFIG ============================================================")
    
    # Format and print input files
    if isinstance(ace_files, str):
        ace_files = [ace_files]
        formatted_ace = ace_files[0]
    else:
        formatted_ace = f"{len(ace_files)} files"
        if len(ace_files) <= 3:
            formatted_ace = ", ".join(os.path.basename(f) for f in ace_files)
    
    if isinstance(cov_files, str):
        cov_files = [cov_files]
        formatted_cov = cov_files[0]
    else:
        formatted_cov = f"{len(cov_files)} files"
        if len(cov_files) <= 3:
            formatted_cov = ", ".join(os.path.basename(f) for f in cov_files)
    
    # Format MT list
    if len(mt_list) == 0:
        formatted_mt = "All available MTs"
    elif len(mt_list) <= 10:
        formatted_mt = ", ".join(str(mt) for mt in mt_list)
    else:
        formatted_mt = f"{len(mt_list)} MTs: {', '.join(str(mt) for mt in mt_list[:5])}..."
    
    # Print all parameters TO LOG FILE
    _logger.info(f"  ACE_FILES = {formatted_ace}")
    _logger.info(f"  COV_FILES = {formatted_cov}")
    _logger.info(f"  MT_NUMBERS = {formatted_mt}")
    _logger.info(f"  NUM_SAMPLES = {num_samples}")
    _logger.info(f"  SPACE = {space}")
    _logger.info(f"  DECOMPOSITION_METHOD = {decomposition_method}")
    _logger.info(f"  SAMPLING_METHOD = {sampling_method}")
    _logger.info(f"  OUTPUT_DIR = {os.path.abspath(output_dir)}")
    _logger.info(f"  XSDIR_FILE = {xsdir_file if xsdir_file else 'None'}")
    _logger.info(f"  SEED = {seed if seed is not None else 'Random'}")
    _logger.info(f"  NPROCS = {nprocs}")
    _logger.info(f"  MODE = {'Dry run (factors only)' if dry_run else 'Full ACE generation'}")
    _logger.info(f"  AUTOFIX = {autofix if autofix is not None else 'None (no autofix)'}")
    _logger.info(f"  HIGH_VAL_THRESH = {high_val_thresh}")
    _logger.info(f"  ACCEPT_TOL = {accept_tol}")
    _logger.info(f"  REMOVE_BLOCKS = {remove_blocks if remove_blocks else 'None'}")
    _logger.info(f"  VERBOSE = {verbose}")
    if max_relative_std is not None:
        _logger.info(f"  MAX_RELATIVE_STD = {max_relative_std} ({max_relative_std*100:.0f}% cap)")
    else:
        _logger.info(f"  MAX_RELATIVE_STD = None (disabled)")
    if energy_ranges is not None:
        formatted_ranges = ", ".join(f"[{lo:.4e}, {hi:.4e}] eV" for lo, hi in energy_ranges)
        _logger.info(f"  ENERGY_RANGES = {formatted_ranges}")
    else:
        _logger.info(f"  ENERGY_RANGES = None (all energies)")

    # Print timestamp
    _logger.info(f"  TIMESTAMP = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _logger.info(f"#== END CONFIG ========================================================")

    # normalize inputs (already done in the prints, but needed for the rest of the function)
    if isinstance(ace_files, str):
        ace_files = [ace_files]
    if isinstance(cov_files, str):
        cov_files = [cov_files]

    # Validate input compatibility
    if len(cov_files) != 1 and len(cov_files) != len(ace_files):
        raise ValueError(
            f"Number of covariance files ({len(cov_files)}) must be either 1 "
            f"(to be used for all ACE files) or equal to the number of ACE files ({len(ace_files)})"
        )

    # Initialize a dictionary to collect summary information for each isotope
    summary_data = {}

    # Track skipped isotopes due to missing or invalid covariance
    skipped_isotopes = {}

    # Initialize dictionary to collect all perturbation factors for matrix generation
    all_factors_data = {}

    # Initialize master perturbation matrix directory for incremental updates
    matrix_dir = _initialize_master_perturbation_matrix(output_dir, timestamp, num_samples)
    _logger.info(f"  [INFO] [ACE] Initialized matrix directory: {os.path.basename(matrix_dir)}")

    # Handle case where there's only one covariance file for multiple ACE files
    if len(cov_files) == 1 and len(ace_files) > 1:
        # Use the same covariance file for all ACE files
        cov_files = cov_files * len(ace_files)
        _logger.info(f"  [INFO] [ACE] Using single covariance file for all {len(ace_files)} isotopes: {os.path.basename(cov_files[0])}")

    job_t0 = time.time()
    warning_counts = {"covariance_load_failed": 0, "file_not_found": 0, "autofix_warnings": 0, "negative_factors": 0}

    for i, (ace_file, cov_file) in enumerate(zip(ace_files, cov_files)):

        # ====== Start of ACE file processing ======
        step_t0 = time.time()
        step_num = i + 1
        _logger.info(f"\n#-- STEP {step_num}: Process isotope from {os.path.basename(ace_file)} ------------------------------------------------")

        # Check if ACE file exists
        if not os.path.exists(ace_file):
            _logger.error(f"  [ERROR] [ACE] ACE file not found: {ace_file}")
            warning_counts["file_not_found"] += 1
            zaid = os.path.splitext(os.path.basename(ace_file))[0]  # Use filename without extension as ZAID
            summary_data[zaid] = {
                "ace_file": os.path.basename(ace_file),
                "cov_file": os.path.basename(cov_file),
                "mt_in_ace": [],
                "mt_in_cov": [],
                "mt_perturbed": [],
                "removed_mts": {},
                "autofix_info": {
                    "level": autofix,
                    "removed_pairs": [],
                    "removed_mts": [],
                    "removed_correlations": [],
                    "converged": False,
                    "soft_threshold_warning": False
                },
                "warnings": [f"ACE file not found: {os.path.basename(ace_file)}"]
            }
            skipped_isotopes[zaid] = "ACE file not found"
            _logger.info(f"#-- END STEP {step_num} (elapsed: {time.time() - step_t0:.1f}s) -------------------------------------")
            continue

        # Read ACE file first to get ZAID
        ace = read_ace(ace_file)
        zaid = ace.zaid
        base, _ = os.path.splitext(os.path.basename(ace_file))

        # Initialize summary information for this isotope
        summary_data[zaid] = {
            "ace_file": os.path.basename(ace_file),
            "cov_file": os.path.basename(cov_file),
            "mt_in_ace": sorted(set(ace.mt_numbers)),
            "mt_in_cov": [],
            "mt_perturbed": [],
            "removed_mts": {},  # Will store MT: reason pairs
            "autofix_info": {  # Track autofix-specific information
                "level": autofix,
                "removed_pairs": [],
                "removed_mts": [],
                "removed_correlations": [],  # Track off-diagonal block removals
                "converged": True,
                "soft_threshold_warning": False  # Track if soft autofix didn't meet threshold
            },
            "warnings": []
        }
        
        cov = load_covariance(cov_file)
        if cov is None:
            _logger.error(f"  [ERROR] [ACE] Unable to load a valid covariance matrix for {cov_file}")
            warning_counts["covariance_load_failed"] += 1
            summary_data[zaid]["warnings"].append(f"No valid covariance file found: {os.path.basename(cov_file)}")
            skipped_isotopes[zaid] = "No valid covariance file found"
            _logger.info(f"#-- END STEP {step_num} (elapsed: {time.time() - step_t0:.1f}s) -------------------------------------")
            continue

        _logger.info(f"  [INFO] [ACE] Covariance file: {cov_file}")
        _logger.info(f"  [INFO] [ACE] Isotope: {zaid}")

        # Check if the covariance matrix is empty (no data)
        if cov.num_matrices == 0:
            _logger.info(f"  [WARN] [ACE] No covariance found in {zaid}. Skipping.")
            summary_data[zaid]["warnings"].append("No covariance data found in matrix")
            skipped_isotopes[zaid] = "No covariance data found in matrix"
            _logger.info(f"#-- END STEP {step_num} (elapsed: {time.time() - step_t0:.1f}s) -------------------------------------")
            continue

        # Apply user-specified block removals before clean_cov
        if remove_blocks and zaid in remove_blocks:
            _logger.info(f"  [INFO] [ACE] Applying user-specified block removals for isotope {zaid}")
            
            # Normalize the removal specification to a list of tuples
            removal_spec = remove_blocks[zaid]
            if isinstance(removal_spec, tuple):
                # Single tuple: convert to list
                blocks_to_remove = [removal_spec]
            else:
                # Should be a list of tuples
                blocks_to_remove = list(removal_spec)
            
            _logger.info(f"  Blocks to remove: {blocks_to_remove}")
            
            # Get reactions before removal for comparison
            reactions_before = cov.reactions_by_isotope(zaid)
            _logger.info(f"  Reactions before removal: {sorted(reactions_before)}")
            
            # Apply the removal
            try:
                cov = cov.remove_matrix(isotope=zaid, reaction_pairs=blocks_to_remove)
                
                # Get reactions after removal
                reactions_after = cov.reactions_by_isotope(zaid)
                _logger.info(f"  Reactions after removal:  {sorted(reactions_after)}")
                
                # Track removed reactions for the summary
                removed_reactions = set(reactions_before) - set(reactions_after)
                for mt in removed_reactions:
                    summary_data[zaid]["removed_mts"][mt] = f"User-specified block removal: {blocks_to_remove}"
                
                if removed_reactions:
                    _logger.info(f"  Successfully removed reactions: {sorted(removed_reactions)}")
                else:
                    _logger.info(f"  No reactions were removed (blocks may not have existed)")
                    
            except Exception as e:
                _logger.error(f"  [ERROR] [ACE] Failed to remove blocks {blocks_to_remove}: {str(e)}")
                summary_data[zaid]["warnings"].append(f"Failed to remove user-specified blocks: {str(e)}")
            
        cov = cov.clean_cov(zaid)

        mt_in_cov = cov.reactions_by_isotope(zaid)
        summary_data[zaid]["mt_in_cov"] = sorted(mt_in_cov)
        
        mt_in_ace = sorted(set(ace.mt_numbers))
        
        # Separate positive and negative MTs from the input list
        mt_positive = [mt for mt in mt_list if mt > 0]
        mt_negative = [abs(mt) for mt in mt_list if mt < 0]  # Store as positive for easier comparison
        
        # Expand negative MTs to include their associated ranges
        # If a summary MT (like 4, 103, 107) is excluded, also exclude its range
        expanded_mt_negative = set(mt_negative)
        for excluded_mt in mt_negative:
            for single, rng in MT_GROUPS:
                if single == excluded_mt:
                    # Add all MTs in the associated range to the exclusion list
                    expanded_mt_negative.update(rng)
                    break
        
        # Determine requested MTs based on positive list
        if len(mt_positive) == 0:
            # No positive MTs specified: start with all MTs in ACE
            mt_request = mt_in_ace
        else:
            # Positive MTs specified: use only those
            mt_request = sorted(mt_positive)
        
        # Apply exclusions (negative MTs and their associated ranges)
        if mt_negative:
            if len(expanded_mt_negative) > len(mt_negative):
                # Log both the explicitly excluded MTs and the ranges they affect
                _logger.info(f"  [INFO] [ACE] Excluding MTs: {sorted(mt_negative)}")
                additional = sorted(expanded_mt_negative - set(mt_negative))
                if additional:
                    _logger.info(f"  [INFO] [ACE] Also excluding associated range MTs: {additional}")
            else:
                _logger.info(f"  [INFO] [ACE] Excluding MTs: {sorted(mt_negative)}")
            mt_request = [mt for mt in mt_request if mt not in expanded_mt_negative]

        # Save original MT list before processing 
        original_mt_perturb = set(mt_in_cov) & set(mt_request) & set(mt_in_ace)
        
        mt_perturb = set(mt_in_cov) & set(mt_request) & set(mt_in_ace)

        # Track removed MTs during group processing
        group_removed_mts = {}

        for single, rng in MT_GROUPS:
            cov_has_single  = single in mt_in_cov
            cov_in_range    = [mt for mt in mt_in_cov if mt in rng]
            list_has_single = single in mt_request
            list_in_range   = [mt for mt in mt_request if mt in rng]
            ace_in_range    = [mt for mt in mt_in_ace if mt in rng]

            # --- Logic for single element (e.g., 4, 103, 104, etc.) ---
            # Check if the single MT was explicitly excluded
            single_is_excluded = single in expanded_mt_negative
            
            if cov_has_single and (list_has_single or (list_in_range and ace_in_range)) and not single_is_excluded:
                mt_perturb.add(single)
            else:
                if single in mt_perturb:
                    if single_is_excluded:
                        if single in mt_negative:
                            group_removed_mts[single] = f"MT={single} explicitly excluded via negative MT specification"
                        else:
                            group_removed_mts[single] = f"MT={single} excluded because summary MT was negatively specified"
                    elif not (cov_has_single and (list_has_single or (list_in_range and ace_in_range))):
                        group_removed_mts[single] = f"Missing data in covariance or ACE file for MT={single} or its associated range"
                mt_perturb.discard(single)
                if cov_has_single:
                    cov = cov.remove_matrix(zaid, [(single, 0)])

            # --- Logic for associated range (51-91, 600-649, etc.) ---
            if cov_in_range:
                if (list_has_single or list_in_range) and ace_in_range and not single_is_excluded:
                    # Keep only the triple intersection in this range
                    triple = set(cov_in_range) & set(mt_request) & set(mt_in_ace)
                    removed_range = set(cov_in_range) - triple
                    for mt in removed_range:
                        if mt in original_mt_perturb:
                            if mt in expanded_mt_negative:
                                group_removed_mts[mt] = f"MT={mt} excluded because summary MT={single} was negatively specified"
                            else:
                                group_removed_mts[mt] = f"MT not in triple intersection for range associated with MT={single}"
                    
                    mt_perturb.update(triple)
                    # Remove those outside the triple intersection from cov
                    to_remove = [(mt, 0) for mt in cov_in_range if mt not in triple]
                    if to_remove:
                        cov = cov.remove_matrix(zaid, to_remove)
                else:
                    # Remove the entire range from cov and perturb
                    to_remove = [(mt, 0) for mt in cov_in_range]
                    for mt in cov_in_range:
                        if mt in original_mt_perturb:
                            if single_is_excluded and mt in expanded_mt_negative:
                                group_removed_mts[mt] = f"MT={mt} excluded because summary MT={single} was negatively specified"
                            else:
                                group_removed_mts[mt] = f"MT range removed because required data missing in ACE or request list"
                    
                    cov = cov.remove_matrix(zaid, to_remove)
                    for mt in cov_in_range:
                        mt_perturb.discard(mt)

        # Update summary with group processing results
        for mt, reason in group_removed_mts.items():
            summary_data[zaid]["removed_mts"][mt] = reason

        # Final sorted list
        mt_perturb = sorted(mt_perturb)

        remaining = cov.reactions_by_isotope(zaid)
        to_remove = [(mt, 0) for mt in remaining if mt not in mt_perturb]
        if to_remove:
            cov = cov.remove_matrix(zaid, to_remove)

        # Print available MT numbers TO LOG FILE
        _logger.info(f"  [INFO] [ACE] MT selection:")
        _logger.info(f"    MTs in ACE file:          {mt_in_ace}")
        _logger.info(f"    MTs in covariance matrix: {mt_in_cov}")
        _logger.info(f"    MTs to be perturbed:      {mt_perturb}")
        _logger.info(f">> mt_count = {len(mt_perturb)}")

        # Convert energy grid from eV to MeV for ACE (ACE energies are in MeV)
        # CrossSectionCovariance now stores energies in eV by default
        energy_grid_eV = cov.energy_grid
        if cov.energy_unit == 'eV':
            energy_grid = [e / 1e6 for e in energy_grid_eV]  # Convert eV to MeV
            _logger.info(f"  [INFO] [ACE] Converted energy grid from eV to MeV for ACE compatibility")
        elif cov.energy_unit == 'MeV':
            energy_grid = energy_grid_eV
        else:
            raise ValueError(f"Unknown energy unit '{cov.energy_unit}' in covariance matrix")

        # Save pre-autofix MT list
        pre_autofix_mts = list(mt_perturb)
        
        try:
            factors, mt_perturb_final, fix_info = generate_samples(
                cov                  = cov,
                space                = space,
                n_samples            = num_samples,
                decomposition_method = decomposition_method,
                sampling_method      = sampling_method,
                seed                 = None if seed is None else seed + zaid,
                mt_numbers           = mt_perturb,
                energy_grid          = energy_grid,
                autofix              = autofix,
                high_val_thresh      = high_val_thresh,
                accept_tol           = accept_tol,
                psd_method           = psd_method,
                max_relative_std     = max_relative_std,
                verbose              = verbose,
                label                = str(zaid),
            )
        except Exception as e:
            # Import the exception classes to check for them
            from kika.sampling.generators import CovarianceFixError, SoftAutofixWarning
            
            if isinstance(e, SoftAutofixWarning):
                # Soft autofix failed threshold but decomposition also failed
                _logger.error(f"  [ERROR] [ACE] Soft autofix warning for isotope {zaid}")
                _logger.error(f"    Error details: {str(e)}")
                warning_counts["autofix_warnings"] += 1

                summary_data[zaid]["warnings"].append(f"Soft autofix failed to meet eigenvalue threshold and decomposition failed")
                summary_data[zaid]["autofix_info"]["converged"] = False
                summary_data[zaid]["autofix_info"]["soft_threshold_warning"] = True
                skipped_isotopes[zaid] = str(e)

                _logger.info(f"#-- END STEP {step_num} (elapsed: {time.time() - step_t0:.1f}s) -------------------------------------")
                continue
            elif isinstance(e, CovarianceFixError):
                _logger.error(f"  [ERROR] [ACE] Covariance matrix fixing failed for isotope {zaid}")
                _logger.error(f"    Error details: {str(e)}")
                _logger.error(f"    Suggestion: Try processing isotope {zaid} separately with autofix='medium' or 'hard'")
                warning_counts["autofix_warnings"] += 1

                summary_data[zaid]["warnings"].append(f"Covariance matrix could not be fixed to meet eigenvalue threshold")
                summary_data[zaid]["autofix_info"]["converged"] = False
                skipped_isotopes[zaid] = str(e)

                _logger.info(f"#-- END STEP {step_num} (elapsed: {time.time() - step_t0:.1f}s) -------------------------------------")
                continue
            else:
                # Re-raise other exceptions
                raise e
        
        # Update summary with autofix results - now get detailed removal information
        if autofix is not None and fix_info is not None:
            # Check for soft autofix threshold warning
            if fix_info.get("soft_autofix_failed", False):
                summary_data[zaid]["autofix_info"]["soft_threshold_warning"] = True
                min_eigenvalue = fix_info.get("min_eigenvalue", float('nan'))
                summary_data[zaid]["warnings"].append(
                    f"Soft autofix failed to meet eigenvalue threshold (λ_min={min_eigenvalue:.4e} < {accept_tol:.4e}) but decomposition succeeded"
                )
                
            removed_in_autofix = set(pre_autofix_mts) - set(mt_perturb_final if mt_perturb_final else [])
            for mt in removed_in_autofix:
                summary_data[zaid]["removed_mts"][mt] = f"Removed during covariance autofix (level='{autofix}')"
                summary_data[zaid]["autofix_info"]["removed_mts"].append(mt)
            
            # Extract correlation removals from fix_info
            if fix_info.get("removed_correlations"):
                summary_data[zaid]["autofix_info"]["removed_correlations"] = fix_info["removed_correlations"]
            
            # Also store the removed pairs for completeness
            if fix_info.get("removed_pairs"):
                summary_data[zaid]["autofix_info"]["removed_pairs"] = fix_info["removed_pairs"]

        # Apply energy range masking if requested
        if energy_ranges is not None and factors is not None and mt_perturb_final:
            factors = _mask_factors_by_energy_range_ace(
                factors, mt_perturb_final, energy_grid, energy_ranges, space
            )
            _logger.info(f"  [INFO] [ACE] Applied energy range masking for {zaid}")

        # Store final perturbed MTs
        summary_data[zaid]["mt_perturbed"] = mt_perturb_final if mt_perturb_final else []
        if factors is not None:
            _logger.info(f">> factors_shape = {factors.shape}")

        # Store factors data for master matrix generation
        if mt_perturb_final and factors is not None:
            all_factors_data[zaid] = {
                'factors': factors.astype(np.float32),  # Convert to float32 for consistency
                'mt_numbers': mt_perturb_final,
                'energy_grid': energy_grid
            }
            
            # Update master perturbation matrix incrementally
            _update_master_perturbation_matrix(
                matrix_dir, zaid, factors, mt_perturb_final, energy_grid, verbose
            )

        # =====================================================================
        #  DRY-RUN
        # =====================================================================
        if dry_run:
            _logger.info(f"  [INFO] [ACE] Dry-run: generating only perturbation factors (no ACE files will be written)")
            _logger.info(f"  [INFO] [ACE] Perturbation factors stored in master matrix")
            _logger.info(f"#-- END STEP {step_num} (elapsed: {time.time() - step_t0:.1f}s) -------------------------------------")
            continue

        # =====================================================================
        #  FULL processing (ACE rewrite)
        # =====================================================================
        _logger.info(f"  [INFO] [ACE] Creating {num_samples} perturbed ACE files")
        _logger.info(f"    Output directory: {os.path.abspath(output_dir)}")
        if nprocs > 1:
            _logger.info(f"    Using {nprocs} parallel processes")
            
        # Create progress tracking variables
        report_interval = max(1, min(100, num_samples // 10))  # Report at most 10 times
        
        tasks = [(ace_file, factors[j], j, energy_grid, mt_perturb_final, output_dir, xsdir_file) for j in range(num_samples)]

        if nprocs > 1:
            # For parallel processing, just show start and end messages
            _logger.info(f"  [INFO] [ACE] Starting parallel sample generation (progress updates disabled in parallel mode)")

            with Pool(processes=nprocs) as pool:
                for args in tasks:
                    pool.apply_async(_process_sample, args=args)
                pool.close()
                pool.join()

            _logger.info(f"  [INFO] [ACE] Completed generating {num_samples} samples")
        else:
            # For sequential processing, show periodic progress updates
            if verbose:
                _logger.info(f"  [INFO] [ACE] Generating samples (progress updates every {report_interval} samples)")

            for i, args in enumerate(tasks):
                _process_sample(*args)

                # Report progress periodically TO LOG FILE
                if verbose and ((i + 1) % report_interval == 0 or i + 1 == num_samples):
                    progress = (i + 1) / num_samples * 100
                    _logger.info(f"  [INFO] [ACE] Progress: {i + 1}/{num_samples} samples ({progress:.1f}%)")

        # ====== End of ACE file processing ======
        _logger.info(f">> samples_generated = {num_samples}")
        _logger.info(f"#-- END STEP {step_num} (elapsed: {time.time() - step_t0:.1f}s) -------------------------------------")

    # =====================================================================
    #  Print final summary for all isotopes TO LOG FILE
    # =====================================================================
    _logger.info(f"\n#== SUMMARY ===========================================================")
    
    if not summary_data:
        _logger.info("  No isotopes were processed.")
    else:
        # Custom sorting function: numeric ZAIDs first (ascending), then string ZAIDs (alphabetically)
        def sort_key(item):
            zaid = item[0]
            try:
                # Try to convert to int - if successful, return (0, numeric_value) for primary sort
                return (0, int(zaid))
            except (ValueError, TypeError):
                # If not numeric, return (1, string) to sort after numeric ones
                return (1, str(zaid))
        
        # Report isotopes that were successfully processed (had some MTs perturbed)
        successfully_processed = {zaid: data for zaid, data in sorted(summary_data.items(), key=sort_key) 
                                if zaid not in skipped_isotopes and data['mt_perturbed']}
        
        # Report isotopes that were processed but had no MTs perturbed
        processed_no_perturbation = {zaid: data for zaid, data in sorted(summary_data.items(), key=sort_key) 
                                   if zaid not in skipped_isotopes and not data['mt_perturbed']}
        
        if successfully_processed:
            _logger.info("\n  SUCCESSFULLY PROCESSED ISOTOPES:")

            for zaid, data in successfully_processed.items():
                # Only show filename in parentheses if ZAID is not numeric (i.e., it's a filename)
                try:
                    int(zaid)  # Test if ZAID is numeric
                    isotope_display = f"{zaid}"
                except (ValueError, TypeError):
                    isotope_display = f"{zaid} ({data['ace_file']})"

                _logger.info(f"\n  Isotope: {isotope_display}")

                # MT numbers that were perturbed
                _logger.info(f"    Perturbed MT numbers: {', '.join(map(str, data['mt_perturbed']))}")

                # Autofix information (for medium/hard levels)
                autofix_info = data.get('autofix_info', {})
                if autofix_info.get('level') in ['medium', 'hard']:
                    autofix_lines = []
                    if autofix_info.get('removed_mts'):
                        autofix_lines.append(f"removed MTs {', '.join(map(str, sorted(autofix_info['removed_mts'])))}")
                    if autofix_info.get('removed_correlations'):
                        corr_pairs = autofix_info['removed_correlations']
                        if len(corr_pairs) <= 3:
                            corr_str = ', '.join(f"({a},{b})" for a, b in corr_pairs)
                        else:
                            corr_str = f"({corr_pairs[0][0]},{corr_pairs[0][1]}), ({corr_pairs[1][0]},{corr_pairs[1][1]}), ... (+{len(corr_pairs)-2} more)"
                        autofix_lines.append(f"removed correlations {corr_str}")

                    if autofix_lines:
                        _logger.info(f"    Autofix (level='{autofix_info['level']}'): {'; '.join(autofix_lines)}")

                # MT numbers that were removed and why (excluding autofix info already shown above)
                other_removed = {mt: reason for mt, reason in data['removed_mts'].items()
                               if not reason.startswith("Removed during covariance autofix")}
                if other_removed:
                    _logger.info(f"    Other removed MT numbers:")
                    for mt, reason in sorted(other_removed.items()):
                        _logger.info(f"      MT={mt}: {reason}")

                # Warnings if any
                if data['warnings']:
                    _logger.info(f"    Warnings:")
                    for warning in data['warnings']:
                        _logger.info(f"      {warning}")

        # Report isotopes that were processed but had no perturbation
        if processed_no_perturbation or skipped_isotopes:
            _logger.info("\n  ISOTOPES WITH NO PERTURBATION:")

            # First report explicitly skipped isotopes - use same sorting and display logic
            for zaid, reason in sorted(skipped_isotopes.items(), key=lambda x: sort_key((x[0], None))):
                if zaid in summary_data:
                    try:
                        int(zaid)  # Test if ZAID is numeric
                        isotope_display = f"{zaid}"
                    except (ValueError, TypeError):
                        isotope_display = f"{zaid} ({summary_data[zaid]['ace_file']})"

                    _logger.info(f"  Isotope: {isotope_display}")
                    _logger.info(f"    Reason: {reason}")

            # Then report isotopes that were processed but had no MTs perturbed
            for zaid, data in processed_no_perturbation.items():
                try:
                    int(zaid)  # Test if ZAID is numeric
                    isotope_display = f"{zaid}"
                except (ValueError, TypeError):
                    isotope_display = f"{zaid} ({data['ace_file']})"

                _logger.info(f"  Isotope: {isotope_display}")
                if data['warnings']:
                    for warning in data['warnings']:
                        _logger.info(f"    Reason: {warning}")
                else:
                    _logger.info(f"    Reason: No eligible MT numbers to perturb")

    # Emit metric lines for key results
    processed_count = len([zaid for zaid in summary_data.keys() if zaid not in skipped_isotopes and summary_data[zaid]['mt_perturbed']])
    skipped_count = len(skipped_isotopes)
    _logger.info(f"\n>> success_count = {processed_count}")
    _logger.info(f">> fail_count = {skipped_count}")
    _logger.info(f">> total_elapsed_s = {time.time() - job_t0:.1f}s")

    _logger.info(f"#== END SUMMARY ========================================================")

    # Convert individual files to master parquet and clean up
    final_parquet_path = _finalize_master_perturbation_matrix(matrix_dir, verbose)
    _logger.info(f"  [INFO] [ACE] Master perturbation matrix finalized")
    _logger.info(f"    Final matrix file: {os.path.basename(final_parquet_path)}")

    # WARNINGS section
    _logger.info(f"\n#== WARNINGS ==========================================================")
    total_warnings = sum(warning_counts.values())
    if total_warnings == 0:
        _logger.info(f"  No warnings.")
    else:
        for wkey, wcount in warning_counts.items():
            if wcount > 0:
                _logger.info(f"  {wkey} = {wcount}")
    _logger.info(f">> total_warnings = {total_warnings}")
    _logger.info(f"#== END WARNINGS ======================================================")

    # Console: Final summary
    _logger.info(f"[INFO] [ACE] Job completed", console=True)
    _logger.info(f"[INFO] [ACE] Processed: {processed_count} isotope(s)", console=True)
    _logger.info(f"[INFO] [ACE] Skipped: {skipped_count} isotope(s)", console=True)
    _logger.info(f"[INFO] [ACE] Detailed log saved to: {log_file}", console=True)
    _logger.info(f"[INFO] [ACE] Master matrix file: {os.path.basename(final_parquet_path)}", console=True)


def _mask_factors_by_energy_range_ace(
    factors: np.ndarray,
    mt_numbers: List[int],
    energy_grid_mev: List[float],
    energy_ranges_ev: List[Tuple[float, float]],
    space: str = "log",
) -> np.ndarray:
    """
    Mask perturbation factors for energy bins outside the requested ranges.

    Parameters
    ----------
    factors : np.ndarray
        Perturbation factors, shape (n_samples, n_mts * n_groups)
    mt_numbers : List[int]
        MT reaction numbers corresponding to the factor columns
    energy_grid_mev : List[float]
        Energy bin boundaries in MeV
    energy_ranges_ev : List[Tuple[float, float]]
        List of (emin, emax) energy ranges in eV
    space : str
        "linear" or "log"

    Returns
    -------
    np.ndarray
        Copy of factors with masked columns set to neutral value
    """
    neutral = 1.0 if space == "linear" else 0.0
    masked = factors.copy()
    n_groups = len(energy_grid_mev) - 1

    # Convert ranges from eV to MeV
    ranges_mev = [(lo / 1e6, hi / 1e6) for lo, hi in energy_ranges_ev]

    for g in range(n_groups):
        bin_low = energy_grid_mev[g]
        bin_high = energy_grid_mev[g + 1]

        overlaps = any(bin_low < hi and bin_high > lo for lo, hi in ranges_mev)
        if not overlaps:
            for mt_idx in range(len(mt_numbers)):
                masked[:, mt_idx * n_groups + g] = neutral

    return masked


def apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers, verbose=True):
    """Apply per-group perturbation factors to ACE, and list which MTs were actually perturbed."""
    logger = _get_logger()
    
    # Convert sample to float32
    sample = sample.astype(np.float32)
    
    n_groups = len(energy_grid) - 1
    if sample.shape[0] != len(mt_numbers) * n_groups:
        raise ValueError(f"sample length {sample.shape[0]} ≠ {len(mt_numbers)}×{n_groups}")

    boundaries = np.asarray(energy_grid)
    perturbed_mts = []  # collect the actual MT numbers

    for mt_idx, mt in enumerate(mt_numbers):
        start = mt_idx * n_groups
        end   = start + n_groups
        factors = sample[start:end]

        if mt == 4:
            for mt_inelastic in range(51, 92):
                if mt_inelastic in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_inelastic, factors, boundaries, verbose)
                    perturbed_mts.append(mt_inelastic)

        elif mt == 103:
            for mt_proton in range(600, 650):
                if mt_proton in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_proton, factors, boundaries, verbose)
                    perturbed_mts.append(mt_proton)

        elif mt == 104:
            for mt_H2 in range(650, 700):
                if mt_H2 in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_H2, factors, boundaries, verbose)
                    perturbed_mts.append(mt_H2)

        elif mt == 105:
            for mt_H3 in range(700, 750):
                if mt_H3 in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_H3, factors, boundaries, verbose)
                    perturbed_mts.append(mt_H3)

        elif mt == 106:
            for mt_He3 in range(750, 800):
                if mt_He3 in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_He3, factors, boundaries, verbose)
                    perturbed_mts.append(mt_He3)

        elif mt == 107:
            for mt_He4 in range(800, 850):
                if mt_He4 in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_He4, factors, boundaries, verbose)
                    perturbed_mts.append(mt_He4)

        else:
            # direct mt
            if mt in ace.mt_numbers:
                _apply_factors_to_mt(ace, mt, factors, boundaries, verbose)
                perturbed_mts.append(mt)

    # remove duplicates and sort for neatness
    perturbed_mts = sorted(set(perturbed_mts))

    if verbose and logger:
        logger.info(f"  [INFO] [ACE] Applying factors for sample #{sample_index+1:04d}")
        logger.info(f"    Perturbed MT numbers: {perturbed_mts}")


def _apply_factors_to_mt(ace, mt, factors, boundaries, verbose=True):
    """Multiply each entry.value by factors[group] for this mt,
    **unless any factor is non-positive**.
    """
    logger = _get_logger()
    
    # Convert factors to float32 if not already
    if factors.dtype != np.float32:
        factors = factors.astype(np.float32)
    
    if (factors <= 0).any():
        bad = ", ".join(f"{f:+.3e}" for f in factors if f <= 0)
        if verbose and logger:
            logger.warning(f"  [WARN] [ACE] MT={mt}: negative or zero factors detected ({bad}). Reaction not perturbed.")
        return  # leave this reaction unperturbed

    reac       = ace.cross_section.reaction[mt]
    energies   = np.asarray(reac.energies)
    xs_entries = reac._xs_entries
    bin_idx    = np.digitize(energies, boundaries) - 1

    for i, entry in enumerate(xs_entries):
        grp = bin_idx[i]
        if 0 <= grp < len(factors):
            entry.value *= float(factors[grp])  # Convert to float for multiplication