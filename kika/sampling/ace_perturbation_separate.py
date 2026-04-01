"""
ACE Perturbation Module for Existing Files

This module provides functionality to apply perturbations to existing ACE files
that have been created by the ENDF perturbation process. Unlike the original
version that created new ACE files, this version:

1. Works with existing ACE files in a predefined directory structure
2. Applies each perturbation sample to a different existing ACE file
3. Replaces files in-place without creating new directories
4. Skips samples where ACE files don't exist
5. Does not create XSDIR files (assumes they already exist)
6. Generates only parquet files with perturbation data (no text summaries)

Key Changes from Original:
- Function signature changed to take root_dir, temperatures, zaids instead of ace_files
- New _process_sample_inplace function for in-place file modification
- Directory structure follows: root_dir/ace/tempK/zaid/sample_num/
- No XSDIR file generation
- Enhanced logging for missing files and skipped samples
- Only parquet output files are generated (no text summary files)

Usage:
    from kika.sampling.ace_perturbation_separate import perturb_seprate_ACE_files
    
    perturb_seprate_ACE_files(
        root_dir="/path/to/output",
        temperatures=[300.0, 600.0],
        zaids=[92235, 92238],
        cov_files="covariance.cov",
        mt_list=[1, 2, 18],
        num_samples=100
    )
"""

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
from kika._utils import temperature_to_suffix
from kika.sampling.utils import (
    DualLogger, 
    _get_logger, 
    _set_logger,
    load_covariance,
    _initialize_master_perturbation_matrix,
    _update_master_perturbation_matrix,
    _finalize_master_perturbation_matrix
)


def _process_sample_inplace(
    ace_file: str,
    sample: np.ndarray,
    sample_index: int,
    energy_grid: List[float],
    mt_numbers: List[int],
    temperature: float,
):
    """
    Apply perturbation to an existing ACE file and replace it in-place.
    
    Parameters
    ----------
    ace_file : str
        Path to the existing ACE file to perturb
    sample : np.ndarray
        Perturbation factors to apply
    sample_index : int
        Sample index (for logging purposes)
    energy_grid : List[float]
        Energy grid boundaries
    mt_numbers : List[int]
        MT numbers to perturb
    temperature : float
        Temperature of the ACE file (for logging/organization)
    """
    # Read & perturb ACE
    ace = read_ace(ace_file)
    apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers, False)  # Set verbose=False to reduce output

    # Recalculate cross sections
    ace.update_cross_sections()
    
    # Write ACE file back in-place (overwrite the original)
    write_ace(ace, ace_file, overwrite=True)


def perturb_seprate_ACE_files(
    root_dir: str,
    temperatures: Union[float, List[float]],
    zaids: List[int],
    cov_files: Union[str, List[str]],
    mt_list: List[int],
    num_samples: int,
    space: str = "log",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    nprocs: int = 1,
    dry_run: bool = False,
    autofix: Optional[str] = None, 
    high_val_thresh: float = 1.0,
    accept_tol: float = -1.0e-4,
    remove_blocks: Optional[Dict[int, Union[Tuple[int, int], List[Tuple[int, int]]]]] = None,  # Add remove_blocks parameter
    verbose: bool = True,
    energy_ranges: Optional[List[Tuple[float, float]]] = None,
):
    """
    Perturb existing ACE nuclear data files using covariance matrices.

    This function applies perturbation factors to existing ACE files that are organized
    in a specific directory structure (from ENDF perturbation output). Each perturbation
    sample is applied to a different existing ACE file, replacing it in-place.

    Parameters
    ----------
    root_dir : str
        Root directory containing the ACE files in the structure:
        root_dir/ace/tempK/zaid/sample_num/
    temperatures : Union[float, List[float]]
        Temperature(s) (in Kelvin) for which ACE files exist
    zaids : List[int]
        List of ZAID numbers to process
    cov_files : Union[str, List[str]]
        Path(s) to covariance matrix file(s) (SCALE or NJOY format).
        Can be a single file (used for all ZAIDs) or one file per ZAID.
    mt_list : List[int]
        List of MT reaction numbers to perturb. Empty list means all available MTs
    num_samples : int
        Number of perturbation samples to apply (must match existing ACE files)
    space : str, default "log"
        Sampling space: "linear" (factors = 1 + X) or "log" (factors = exp(Y))
    decomposition_method : str, default "svd"
        Matrix decomposition method: "svd", "cholesky", "eigen", or "pca"
    sampling_method : str, default "sobol"
        Sampling method: "sobol", "lhs", or "random"
    seed : Optional[int], default None
        Random seed for reproducible sampling
    nprocs : int, default 1
        Number of parallel processes (currently unused)
    dry_run : bool, default False
        If True, only generate perturbation factors without modifying ACE files
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
        these ranges will be perturbed; bins outside are set to neutral (1.0 for
        linear space, 0.0 for log space). None means perturb all energies.

    Notes
    -----
    This function expects ACE files to exist in the directory structure created by
    perturb_ENDF_files. Each perturbation sample will be applied to the corresponding
    existing ACE file and replace it in-place. No new XSDIR files are created as
    they should already exist from the original ENDF perturbation process.
    """
    global _logger
    
    # Normalize temperatures to list
    if isinstance(temperatures, (int, float)):
        temperatures = [float(temperatures)]
    elif isinstance(temperatures, list):
        temperatures = [float(t) for t in temperatures]
    else:
        raise ValueError("temperatures must be a float or list of floats")
    
    # Create output directory for logs and matrices (use root_dir, not temperature subdirectory)
    # We'll create master files in root_dir and then copy them to each temperature directory
    output_dir = root_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'ace_perturbation_{timestamp}.log')
    _logger = DualLogger(log_file)
    _set_logger(_logger)
    
    # Console: Basic start message
    _logger.info(f"[INFO] [ACE] Starting ACE perturbation job", console=True)
    _logger.info(f"[INFO] [ACE] Log file: {log_file}", console=True)
    _logger.info(f"[INFO] [ACE] Root directory: {os.path.abspath(root_dir)}", console=True)

    # Format input values for config dump
    formatted_temps = ", ".join(f"{t:.1f}K" for t in temperatures)

    if isinstance(cov_files, str):
        cov_files = [cov_files]
        formatted_cov = cov_files[0]
    else:
        formatted_cov = f"{len(cov_files)} files"
        if len(cov_files) <= 3:
            formatted_cov = ", ".join(os.path.basename(f) for f in cov_files)

    # Format ZAID list
    if len(zaids) <= 10:
        formatted_zaids = ", ".join(str(zaid) for zaid in zaids)
    else:
        formatted_zaids = f"{len(zaids)} ZAIDs: {', '.join(str(zaid) for zaid in zaids[:5])}..."

    # Format MT list
    if len(mt_list) == 0:
        formatted_mt = "All available MTs"
    elif len(mt_list) <= 10:
        formatted_mt = ", ".join(str(mt) for mt in mt_list)
    else:
        formatted_mt = f"{len(mt_list)} MTs: {', '.join(str(mt) for mt in mt_list[:5])}..."

    # CONFIG section
    _logger.info(f"#== CONFIG ============================================================")
    _logger.info(f"  ROOT_DIR = {os.path.abspath(root_dir)}")
    _logger.info(f"  TEMPERATURES = {formatted_temps}")
    _logger.info(f"  ZAIDS = {formatted_zaids}")
    _logger.info(f"  COV_FILES = {formatted_cov}")
    _logger.info(f"  MT_NUMBERS = {formatted_mt}")
    _logger.info(f"  NUM_SAMPLES = {num_samples}")
    _logger.info(f"  SPACE = {space}")
    _logger.info(f"  DECOMPOSITION_METHOD = {decomposition_method}")
    _logger.info(f"  SAMPLING_METHOD = {sampling_method}")
    _logger.info(f"  SEED = {seed if seed is not None else 'Random'}")
    _logger.info(f"  NPROCS = {nprocs}")
    _logger.info(f"  MODE = {'Dry run (factors only)' if dry_run else 'Full ACE perturbation'}")
    _logger.info(f"  AUTOFIX = {autofix if autofix is not None else 'None'}")
    _logger.info(f"  HIGH_VAL_THRESH = {high_val_thresh}")
    _logger.info(f"  ACCEPT_TOL = {accept_tol}")
    _logger.info(f"  REMOVE_BLOCKS = {remove_blocks if remove_blocks else 'None'}")
    _logger.info(f"  VERBOSE = {verbose}")
    if energy_ranges is not None:
        formatted_ranges = ", ".join(f"[{lo:.4e}, {hi:.4e}] eV" for lo, hi in energy_ranges)
        _logger.info(f"  ENERGY_RANGES = {formatted_ranges}")
    else:
        _logger.info(f"  ENERGY_RANGES = None (all energies)")
    _logger.info(f"  TIMESTAMP = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _logger.info(f"#== END CONFIG ========================================================")

    # normalize inputs (already done for cov_files, need for zaids)
    if isinstance(cov_files, str):
        cov_files = [cov_files]

    # Validate input compatibility
    if len(cov_files) != 1 and len(cov_files) != len(zaids):
        raise ValueError(
            f"Number of covariance files ({len(cov_files)}) must be either 1 "
            f"(to be used for all ZAIDs) or equal to the number of ZAIDs ({len(zaids)})"
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

    # Console: Show progress
    _logger.info(f"[INFO] [ACE] Processing {len(zaids)} isotope(s)", console=True)
    _logger.info(f"[INFO] [ACE] Matrix directory: {os.path.basename(matrix_dir)}", console=True)

    # Track total job time
    _job_t0 = time.time()

    # Handle case where there's only one covariance file for multiple ZAIDs
    if len(cov_files) == 1 and len(zaids) > 1:
        # Use the same covariance file for all ZAIDs
        cov_files = cov_files * len(zaids)
        _logger.info(f"  [INFO] [ACE] Using single covariance file for all {len(zaids)} isotopes: {os.path.basename(cov_files[0])}")

    _warning_counts = {}  # Collect warnings for WARNINGS section

    for i, (zaid, cov_file) in enumerate(zip(zaids, cov_files)):

        # ====== Start of ZAID processing ======
        _step_num = i + 1
        _step_t0 = time.time()
        _logger.info(f"#-- STEP {_step_num}: Process ZAID {zaid} ------------------------------------------------")

        # Console: Basic progress
        _logger.info(f"[INFO] [ACE] Processing isotope {_step_num}/{len(zaids)}: ZAID {zaid}", console=True)

        # Find a representative ACE file to read structure and MT numbers
        # Look for any sample file to get the ACE structure
        ace_file = None
        for temp in temperatures:
            # Use exact temperature formatting to match ENDF perturbation directory structure
            temp_str = str(temp).rstrip('0').rstrip('.') if '.' in str(temp) else str(temp)
            sample_dir = os.path.join(root_dir, "ace", temp_str, str(zaid), "0001")
            if os.path.exists(sample_dir):
                # Get the expected file extension for this temperature
                temp_suffix = temperature_to_suffix(temp)
                expected_ext = f"{temp_suffix}c"
                # Find ACE file in this directory with correct extension
                for filename in os.listdir(sample_dir):
                    if filename.endswith(expected_ext):
                        ace_file = os.path.join(sample_dir, filename)
                        break
                if ace_file:
                    break
        
        if ace_file is None:
            _logger.error(f"  [ERROR] [ACE] No representative ACE file found for ZAID {zaid}")
            # Show the exact temperature directories that were searched
            searched_paths = []
            for temp in temperatures:
                temp_str = str(temp).rstrip('0').rstrip('.') if '.' in str(temp) else str(temp)
                searched_paths.append(f"{root_dir}/ace/{temp_str}/{zaid}/0001/")
            _logger.error(f"  [ERROR] [ACE] Looked in: {searched_paths}")
            summary_data[zaid] = {
                "representative_ace": "Not found",
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
                "warnings": [f"No representative ACE file found"]
            }
            skipped_isotopes[zaid] = "No representative ACE file found"
            _warning_counts[f"ZAID {zaid}: no ACE file"] = 1
            _step_elapsed = time.time() - _step_t0
            _logger.info(f"#-- END STEP {_step_num} (elapsed: {_step_elapsed:.1f}s) -------------------------------------")
            _logger.info(f"  [ERROR] [ACE] No ACE files found for ZAID {zaid}", console=True)
            continue

        # Check if ACE file exists
        if not os.path.exists(ace_file):
            _logger.error(f"  [ERROR] [ACE] Representative ACE file not found: {ace_file}")
            summary_data[zaid] = {
                "representative_ace": os.path.basename(ace_file),
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
                "warnings": [f"Representative ACE file not found: {os.path.basename(ace_file)}"]
            }
            skipped_isotopes[zaid] = f"Representative ACE file not found: {os.path.basename(ace_file)}"
            _warning_counts[f"ZAID {zaid}: ACE file not found"] = 1
            _step_elapsed = time.time() - _step_t0
            _logger.info(f"#-- END STEP {_step_num} (elapsed: {_step_elapsed:.1f}s) -------------------------------------")
            _logger.info(f"  [ERROR] [ACE] Representative ACE file not found for ZAID {zaid}", console=True)
            continue

        # Read ACE file first to get ZAID and structure
        ace = read_ace(ace_file)
        actual_zaid = ace.zaid
        base, _ = os.path.splitext(os.path.basename(ace_file))

        # Verify ZAID matches
        if actual_zaid != zaid:
            _logger.warning(f"  [WARN] [ACE] ZAID mismatch: requested {zaid}, found {actual_zaid} in {ace_file}")
            _warning_counts[f"ZAID {zaid}: mismatch"] = _warning_counts.get(f"ZAID {zaid}: mismatch", 0) + 1

        # Initialize summary information for this isotope
        summary_data[zaid] = {
            "representative_ace": os.path.basename(ace_file),
            "cov_file": os.path.basename(cov_file),
            "mt_in_ace": sorted(set(ace.mt_numbers)),
            "mt_in_cov": [],
            "mt_perturbed": [],
            "removed_mts": {},
            "autofix_info": {
                "level": autofix,
                "removed_pairs": [],
                "removed_mts": [],
                "removed_correlations": [],
                "converged": True,
                "soft_threshold_warning": False
            },
            "warnings": []
        }
        
        cov = load_covariance(cov_file)
        if cov is None:
            _logger.error(f"  [ERROR] [ACE] Unable to load a valid covariance matrix for {cov_file}")
            summary_data[zaid]["warnings"].append(f"No valid covariance file found: {os.path.basename(cov_file)}")
            skipped_isotopes[zaid] = "No valid covariance file found"
            _warning_counts[f"ZAID {zaid}: no covariance"] = 1
            _step_elapsed = time.time() - _step_t0
            _logger.info(f"#-- END STEP {_step_num} (elapsed: {_step_elapsed:.1f}s) -------------------------------------")
            _logger.info(f"  [ERROR] [ACE] Failed to load covariance for {os.path.basename(ace_file)}", console=True)
            continue

        _logger.info(f"  [INFO] [ACE] Covariance file: {cov_file}")
        _logger.info(f"  [INFO] [ACE] Isotope: {zaid}")

        # Check if the covariance matrix is empty (no data)
        if cov.num_matrices == 0:
            _logger.info(f"  [WARN] [ACE] No covariance found in {zaid}. Skipping.")
            summary_data[zaid]["warnings"].append("No covariance data found in matrix")
            skipped_isotopes[zaid] = "No covariance data found in matrix"
            _warning_counts[f"ZAID {zaid}: empty covariance"] = 1
            _step_elapsed = time.time() - _step_t0
            _logger.info(f"#-- END STEP {_step_num} (elapsed: {_step_elapsed:.1f}s) -------------------------------------")
            _logger.info(f"  [WARN] [ACE] No covariance data for {os.path.basename(ace_file)}", console=True)
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

            _logger.info(f"  [INFO] [ACE] Blocks to remove: {blocks_to_remove}")

            # Get reactions before removal for comparison
            reactions_before = cov.reactions_by_isotope(zaid)
            _logger.info(f"  [INFO] [ACE] Reactions before removal: {sorted(reactions_before)}")

            # Apply the removal
            try:
                cov = cov.remove_matrix(isotope=zaid, reaction_pairs=blocks_to_remove)

                # Get reactions after removal
                reactions_after = cov.reactions_by_isotope(zaid)
                _logger.info(f"  [INFO] [ACE] Reactions after removal:  {sorted(reactions_after)}")

                # Track removed reactions for the summary
                removed_reactions = set(reactions_before) - set(reactions_after)
                for mt in removed_reactions:
                    summary_data[zaid]["removed_mts"][mt] = f"User-specified block removal: {blocks_to_remove}"

                if removed_reactions:
                    _logger.info(f"  [INFO] [ACE] Successfully removed reactions: {sorted(removed_reactions)}")
                else:
                    _logger.info(f"  [INFO] [ACE] No reactions were removed (blocks may not have existed)")

            except Exception as e:
                _logger.error(f"  [ERROR] [ACE] Failed to remove blocks {blocks_to_remove}: {str(e)}")
                summary_data[zaid]["warnings"].append(f"Failed to remove user-specified blocks: {str(e)}")

        cov = cov.clean_cov(zaid)

        mt_in_cov = cov.reactions_by_isotope(zaid)
        summary_data[zaid]["mt_in_cov"] = sorted(mt_in_cov)
        
        mt_in_ace = sorted(set(ace.mt_numbers))
        if len(mt_list) == 0:
            mt_request = mt_in_ace
        else:
            mt_request = sorted(mt_list)

        # Save original MT list before processing 
        original_mt_perturb = set(mt_in_cov) & set(mt_request) & set(mt_in_ace)
        
        mt_perturb = set(mt_in_cov) & set(mt_request) & set(mt_in_ace)
        groups = [
            (4,   range(51,  92)),
            (103, range(600, 650)),
            (104, range(650, 700)),
            (105, range(700, 750)),
            (106, range(750, 800)),
            (107, range(800, 850)),
        ]

        # Track removed MTs during group processing
        group_removed_mts = {}

        for single, rng in groups:
            cov_has_single  = single in mt_in_cov
            cov_in_range    = [mt for mt in mt_in_cov if mt in rng]
            list_has_single = single in mt_request
            list_in_range   = [mt for mt in mt_request if mt in rng]
            ace_in_range    = [mt for mt in mt_in_ace if mt in rng]

            # --- Logic for single element (e.g., 4, 103, 104, etc.) ---
            if cov_has_single and (list_has_single or (list_in_range and ace_in_range)):
                mt_perturb.add(single)
            else:
                if single in mt_perturb:
                    if not (cov_has_single and (list_has_single or (list_in_range and ace_in_range))):
                        group_removed_mts[single] = f"Missing data in covariance or ACE file for MT={single} or its associated range"
                mt_perturb.discard(single)
                if cov_has_single:
                    cov = cov.remove_matrix(zaid, [(single, 0)])

            # --- Logic for associated range (51-91, 600-649, etc.) ---
            if cov_in_range:
                if (list_has_single or list_in_range) and ace_in_range:
                    # Keep only the triple intersection in this range
                    triple = set(cov_in_range) & set(mt_request) & set(mt_in_ace)
                    removed_range = set(cov_in_range) - triple
                    for mt in removed_range:
                        if mt in original_mt_perturb:
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
        _logger.info(f"  [INFO] [ACE]   MTs in ACE file:          {mt_in_ace}")
        _logger.info(f"  [INFO] [ACE]   MTs in covariance matrix: {mt_in_cov}")
        _logger.info(f"  [INFO] [ACE]   MTs to be perturbed:      {mt_perturb}")
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
        
        # Console: Show sample generation start
        _logger.info(f"  [INFO] [ACE] Generating {num_samples} samples for ZAID {zaid}", console=True)
        
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
                verbose              = verbose,
            )
        except Exception as e:
            # Import the exception classes to check for them
            from kika.sampling.generators import CovarianceFixError, SoftAutofixWarning
            
            if isinstance(e, SoftAutofixWarning):
                # Soft autofix failed threshold but decomposition also failed
                _logger.error(f"  [ERROR] [ACE] Soft autofix warning for isotope {zaid}")
                _logger.error(f"  [ERROR] [ACE] Error details: {str(e)}")

                summary_data[zaid]["warnings"].append(f"Soft autofix failed to meet eigenvalue threshold and decomposition failed")
                summary_data[zaid]["autofix_info"]["converged"] = False
                summary_data[zaid]["autofix_info"]["soft_threshold_warning"] = True
                skipped_isotopes[zaid] = str(e)
                _warning_counts[f"ZAID {zaid}: soft autofix failed"] = 1

                _step_elapsed = time.time() - _step_t0
                _logger.info(f"#-- END STEP {_step_num} (elapsed: {_step_elapsed:.1f}s) -------------------------------------")
                _logger.info(f"  [ERROR] [ACE] Soft autofix and decomposition failed for {os.path.basename(ace_file)}", console=True)
                continue
            elif isinstance(e, CovarianceFixError):
                _logger.error(f"  [ERROR] [ACE] Covariance matrix fixing failed for isotope {zaid}")
                _logger.error(f"  [ERROR] [ACE] Error details: {str(e)}")
                _logger.error(f"  [ERROR] [ACE] Suggestion: Try processing isotope {zaid} separately with autofix='medium' or 'hard'")

                summary_data[zaid]["warnings"].append(f"Covariance matrix could not be fixed to meet eigenvalue threshold")
                summary_data[zaid]["autofix_info"]["converged"] = False
                skipped_isotopes[zaid] = str(e)
                _warning_counts[f"ZAID {zaid}: covariance fix failed"] = 1

                _step_elapsed = time.time() - _step_t0
                _logger.info(f"#-- END STEP {_step_num} (elapsed: {_step_elapsed:.1f}s) -------------------------------------")
                _logger.info(f"  [ERROR] [ACE] Covariance fix failed for {os.path.basename(ace_file)}", console=True)
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

        # Store factors data for master matrix generation
        if mt_perturb_final and factors is not None:
            _logger.info(f">> factors_shape = {factors.shape}")
            _logger.info(f">> samples_generated = {factors.shape[0]}")
            all_factors_data[zaid] = {
                'factors': factors.astype(np.float32),  # Convert to float32 for consistency
                'mt_numbers': mt_perturb_final,
                'energy_grid': energy_grid
            }

            # Update master perturbation matrix incrementally
            _update_master_perturbation_matrix(
                matrix_dir, zaid, factors, mt_perturb_final, energy_grid, verbose
            )

        # DRY-RUN
        if dry_run:
            _logger.info(f"  [INFO] [ACE] Dry-run: generating only perturbation factors (no ACE files will be written)")

            # For dry run, create summary for all samples across all temperatures
            processed_samples_by_temp_dry = {temp: list(range(num_samples)) for temp in temperatures}

            # Note: Summary text files are no longer generated - parquet file contains all data

            _step_elapsed = time.time() - _step_t0
            _logger.info(f"#-- END STEP {_step_num} (elapsed: {_step_elapsed:.1f}s) -------------------------------------")
            _logger.info(f"  [INFO] [ACE] Completed dry run for ZAID {zaid}", console=True)
            continue

        # FULL processing (ACE perturbation in-place)
        _logger.info(f"  [INFO] [ACE] Applying perturbations to {num_samples} existing ACE files")
        _logger.info(f"  [INFO] [ACE] Root directory: {os.path.abspath(root_dir)}")
        if nprocs > 1:
            _logger.info(f"  [INFO] [ACE] Using {nprocs} parallel processes")
            
        # Create progress tracking variables
        report_interval = max(1, min(100, num_samples // 10))  # Report at most 10 times
        
        # Build list of ACE files to process and their corresponding perturbation factors
        tasks = []
        skipped_samples = []
        processed_samples_by_temp = {temp: [] for temp in temperatures}
        
        for j in range(num_samples):
            sample_str = f"{j+1:04d}"
            
            # Try to find ACE file for this sample across all temperatures
            for temp in temperatures:
                # Use exact temperature formatting to match ENDF perturbation directory structure
                temp_str = str(temp).rstrip('0').rstrip('.') if '.' in str(temp) else str(temp)
                sample_dir = os.path.join(root_dir, "ace", temp_str, str(zaid), sample_str)
                
                if os.path.exists(sample_dir):
                    # Get the expected file extension for this temperature
                    temp_suffix = temperature_to_suffix(temp)
                    expected_ext = f"{temp_suffix}c"
                    
                    # Find ACE file in this directory with correct extension
                    for filename in os.listdir(sample_dir):
                        if filename.endswith(expected_ext):
                            sample_ace_file = os.path.join(sample_dir, filename)
                            tasks.append((sample_ace_file, factors[j], j, energy_grid, mt_perturb_final, temp))
                            processed_samples_by_temp[temp].append(j)
                            break
            
            # Check if sample was found in any temperature
            if not any(j in processed_samples_by_temp[temp] for temp in temperatures):
                skipped_samples.append(j+1)
        
        if skipped_samples:
            _logger.warning(f"  [WARN] [ACE] Skipped {len(skipped_samples)} samples - ACE files not found: {skipped_samples[:10]}{'...' if len(skipped_samples) > 10 else ''}")
            _warning_counts[f"ZAID {zaid}: skipped samples"] = len(skipped_samples)

        if not tasks:
            _logger.error(f"  [ERROR] [ACE] No ACE files found for any samples of ZAID {zaid}")
            _step_elapsed = time.time() - _step_t0
            _logger.info(f"#-- END STEP {_step_num} (elapsed: {_step_elapsed:.1f}s) -------------------------------------")
            _logger.info(f"  [ERROR] [ACE] No ACE files found for ZAID {zaid}", console=True)
            continue

        _logger.info(f"  [INFO] [ACE] Found {len(tasks)} ACE files to process out of {num_samples} requested samples")

        if nprocs > 1:
            # For parallel processing, just show start and end messages
            _logger.info(f"  [INFO] [ACE] Starting parallel sample processing... (progress updates disabled in parallel mode)")

            with Pool(processes=nprocs) as pool:
                for args in tasks:
                    pool.apply_async(_process_sample_inplace, args=args)
                pool.close()
                pool.join()

            _logger.info(f"  [INFO] [ACE] Completed processing {len(tasks)} samples")
            _logger.info(f"  [INFO] [ACE] Completed ACE perturbation for ZAID {zaid}", console=True)
        else:
            # For sequential processing, show periodic progress updates
            _logger.info(f"  [INFO] [ACE] Processing samples (progress updates every {report_interval} samples)")

            for task_i, args in enumerate(tasks):
                _process_sample_inplace(*args)

                # Report progress periodically TO LOG FILE (verbose-gated)
                if verbose and ((task_i + 1) % report_interval == 0 or task_i + 1 == len(tasks)):
                    progress = (task_i + 1) / len(tasks) * 100
                    _logger.info(f"  [INFO] [ACE] Progress: {task_i + 1}/{len(tasks)} samples ({progress:.1f}%)")

            _logger.info(f"  [INFO] [ACE] Completed ACE perturbation for ZAID {zaid}", console=True)

        # ====== End of ZAID processing ======
        _step_elapsed = time.time() - _step_t0
        _logger.info(f">> success_count = {len(tasks)}")
        _logger.info(f">> fail_count = {len(skipped_samples) if skipped_samples else 0}")
        _logger.info(f"#-- END STEP {_step_num} (elapsed: {_step_elapsed:.1f}s) -------------------------------------")

        # Copy parquet data to each temperature directory (summary files generation removed)
        # Note: Summary text files are no longer generated - parquet file contains all data

    # SUMMARY section
    _logger.info(f"#== SUMMARY ============================================================")

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
            _logger.info("  SUCCESSFULLY PROCESSED ISOTOPES:")

            for zaid, data in successfully_processed.items():
                # Only show filename in parentheses if ZAID is not numeric (i.e., it's a filename)
                try:
                    int(zaid)  # Test if ZAID is numeric
                    isotope_display = f"{zaid}"
                except (ValueError, TypeError):
                    isotope_display = f"{zaid} ({data['ace_file']})"

                _logger.info(f"    Isotope: {isotope_display}")

                # MT numbers that were perturbed
                _logger.info(f"      Perturbed MT numbers: {', '.join(map(str, data['mt_perturbed']))}")

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
                        _logger.info(f"      Autofix (level='{autofix_info['level']}'): {'; '.join(autofix_lines)}")

                # MT numbers that were removed and why (excluding autofix info already shown above)
                other_removed = {mt: reason for mt, reason in data['removed_mts'].items()
                               if not reason.startswith("Removed during covariance autofix")}
                if other_removed:
                    _logger.info(f"      Other removed MT numbers:")
                    for mt, reason in sorted(other_removed.items()):
                        _logger.info(f"        MT={mt}: {reason}")

                # Warnings if any
                if data['warnings']:
                    _logger.info(f"      Warnings:")
                    for warning in data['warnings']:
                        _logger.info(f"        {warning}")

        # Report isotopes that were processed but had no perturbation
        if processed_no_perturbation or skipped_isotopes:
            _logger.info("  ISOTOPES WITH NO PERTURBATION:")

            # First report explicitly skipped isotopes - use same sorting and display logic
            for zaid, reason in sorted(skipped_isotopes.items(), key=lambda x: sort_key((x[0], None))):
                if zaid in summary_data:
                    try:
                        int(zaid)  # Test if ZAID is numeric
                        isotope_display = f"{zaid}"
                    except (ValueError, TypeError):
                        isotope_display = f"{zaid} ({summary_data[zaid]['ace_file']})"

                    _logger.info(f"    Isotope: {isotope_display}")
                    _logger.info(f"      Reason: {reason}")

            # Then report isotopes that were processed but had no MTs perturbed
            for zaid, data in processed_no_perturbation.items():
                try:
                    int(zaid)  # Test if ZAID is numeric
                    isotope_display = f"{zaid}"
                except (ValueError, TypeError):
                    isotope_display = f"{zaid} ({data['ace_file']})"

                _logger.info(f"    Isotope: {isotope_display}")
                if data['warnings']:
                    for warning in data['warnings']:
                        _logger.info(f"      Reason: {warning}")
                else:
                    _logger.info(f"      Reason: No eligible MT numbers to perturb")

    # Emit summary metrics
    processed_count = len([zaid for zaid in summary_data.keys() if zaid not in skipped_isotopes])
    skipped_count = len(skipped_isotopes)
    _logger.info(f">> processed_isotopes = {processed_count}")
    _logger.info(f">> skipped_isotopes = {skipped_count}")
    _job_elapsed = time.time() - _job_t0
    _logger.info(f">> total_elapsed_s = {_job_elapsed:.1f}")
    _logger.info(f"#== END SUMMARY ========================================================")
    
    # Convert individual files to master parquet and clean up
    final_parquet_path = _finalize_master_perturbation_matrix(matrix_dir, verbose)
    if final_parquet_path is not None:
        _logger.info(f"  [INFO] [ACE] Master perturbation matrix finalized")
        _logger.info(f"  [INFO] [ACE] Final matrix file: {os.path.basename(final_parquet_path)}")

        # Copy the master parquet file to each temperature directory
        _copy_master_files_to_temperature_directories(root_dir, temperatures, final_parquet_path, log_file, verbose)

        # Clean up master files from root directory after copying
        if final_parquet_path and os.path.exists(final_parquet_path):
            os.remove(final_parquet_path)
            _logger.info(f"  [INFO] [ACE] Removed master parquet from root directory")

        matrix_file_msg = f"Master matrix file copied to temperature directories"
    else:
        _logger.info(f"  [INFO] [ACE] No master perturbation matrix created (no valid data)")
        matrix_file_msg = "No master matrix file created (no valid perturbation data)"

    # WARNINGS section
    _logger.info(f"#== WARNINGS ===========================================================")
    if _warning_counts:
        for wkey, wcount in _warning_counts.items():
            _logger.info(f"  {wkey}: {wcount}")
        _logger.info(f">> total_warnings = {sum(_warning_counts.values())}")
    else:
        _logger.info(f"  No warnings.")
        _logger.info(f">> total_warnings = 0")
    _logger.info(f"#== END WARNINGS =======================================================")

    # Console: Final summary
    _logger.info(f"[INFO] [ACE] Job completed", console=True)
    _logger.info(f"[INFO] [ACE] Processed: {processed_count} isotope(s)", console=True)
    _logger.info(f"[INFO] [ACE] Skipped: {skipped_count} isotope(s)", console=True)
    _logger.info(f"[INFO] [ACE] Detailed log saved to: {log_file}", console=True)
    _logger.info(f"[INFO] [ACE] {matrix_file_msg}", console=True)
    
    # Clean up root log file after all console output is complete
    # The log file has been copied to temperature directories
    try:
        if os.path.exists(log_file):
            os.remove(log_file)
            # Note: Can't log this removal since we just deleted the log file
    except Exception as e:
        # Silently ignore cleanup errors
        pass
    
    # Clean up any empty ZAID directories that might have been created in root
    for zaid in zaids:
        zaid_dir = os.path.join(root_dir, str(zaid))
        try:
            if os.path.exists(zaid_dir) and os.path.isdir(zaid_dir):
                # Only remove if directory is empty
                if not os.listdir(zaid_dir):
                    os.rmdir(zaid_dir)
        except Exception as e:
            # Silently ignore cleanup errors
            pass


def _copy_master_files_to_temperature_directories(
    root_dir: str,
    temperatures: List[float],
    master_parquet_path: Optional[str],
    log_file: str,
    verbose: bool = True
):
    """
    Copy master parquet file and log file to each temperature directory.
    
    Parameters
    ----------
    root_dir : str
        Root directory
    temperatures : List[float]
        List of temperatures
    master_parquet_path : Optional[str]
        Path to the master parquet file (can be None if no file was created)
    log_file : str
        Path to the log file
    verbose : bool
        Whether to log progress
    """
    logger = _get_logger()
    
    if verbose and logger:
        logger.info(f"  [INFO] [ACE] Copying master files to temperature directories")
    
    import shutil
    
    for temp in temperatures:
        # Use exact temperature formatting to match ENDF perturbation directory structure
        temp_str = str(temp).rstrip('0').rstrip('.') if '.' in str(temp) else str(temp)
        temp_ace_dir = os.path.join(root_dir, "ace", temp_str)
        os.makedirs(temp_ace_dir, exist_ok=True)
        
        # Copy parquet file (only if it exists and is not already in this location)
        if master_parquet_path is not None and os.path.exists(master_parquet_path):
            dest_parquet = os.path.join(temp_ace_dir, os.path.basename(master_parquet_path))
            # Only copy if source and destination are different
            if os.path.abspath(master_parquet_path) != os.path.abspath(dest_parquet):
                shutil.copy2(master_parquet_path, dest_parquet)
                if verbose and logger:
                    logger.info(f"  [INFO] [ACE] Copied parquet to {temp_str}/")
            elif verbose and logger:
                logger.info(f"  [INFO] [ACE] Parquet already exists in {temp_str}/")
        elif verbose and logger:
            logger.info(f"  [INFO] [ACE] No parquet file to copy to {temp_str}/")

        # Copy log file (only if it's not already in this location)
        if os.path.exists(log_file):
            dest_log = os.path.join(temp_ace_dir, os.path.basename(log_file))
            # Only copy if source and destination are different
            if os.path.abspath(log_file) != os.path.abspath(dest_log):
                shutil.copy2(log_file, dest_log)
                if verbose and logger:
                    logger.info(f"  [INFO] [ACE] Copied log to {temp_str}/")
            elif verbose and logger:
                logger.info(f"  [INFO] [ACE] Log already exists in {temp_str}/")


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
        logger.info(f"  [INFO] [ACE]   Perturbed MT numbers: {perturbed_mts}")


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

