"""
ENDF file perturbation module for angular distribution covariance data.

This module provides functionality to perturb ENDF MF4 angular distribution data
using MF34 covariance matrices. It is designed to be scalable and maintainable
for future extension to other MF sections.
"""
from typing import List, Union, Optional, Dict, Tuple, Any
from multiprocessing import Pool
from datetime import datetime
import os
import time
import numpy as np
import pandas as pd
import shutil
import tempfile

from kika.sampling.core import draw_samples
from kika.sampling.generators import generate_endf_samples
from kika.sampling.model_blocks import (
    legendre_covariance_blocks,
    legendre_covariance_index,
)
from kika.cov.legendre_covariance import LegendreCovariance
from kika.endf.parsers.parse_endf import parse_endf_file
from kika.endf.writers.endf_writer import ENDFWriter
from kika.endf.classes.mf4.polynomial import MF4MTLegendre
from kika.endf.classes.mf4.mixed import MF4MTMixed
from kika.endf.classes.mf import MF
from kika._utils import zaid_to_symbol, temperature_to_suffix
from kika.njoy.run_njoy import run_njoy
from kika.endf.read_endf import read_endf
from kika.ace.xsdir import create_xsdir_files_for_ace, _rewrite_xsdir_for_zaids


# Reuse the DualLogger and utilities from sampling.utils
from kika.sampling.utils import (
    _format_energy_group_name,
    DualLogger,
    _get_logger,
    _set_logger,
    make_ace_sample_paths,
    normalize_mt_list,
    resolve_signed_request,
)
from kika._constants import MT_GROUPS

# Import NJOY runner for ACE generation
from kika.njoy.run_njoy import run_njoy

# Global logger instance
_logger = None


def _process_sample(
    endf_file: str,
    sample: np.ndarray,
    sample_index: int,
    energy_grids: Dict[Tuple[int, int, int], List[float]],  # (isotope, mt, l) -> energy_grid
    param_mapping: List[Tuple[int, int, int, int]],  # List of (isotope, mt, l, energy_bin) tuples
    output_dir: str,
    dry_run: bool = False,
    generate_ace: bool = False,
    njoy_exe: Optional[str] = None,
    temperatures: Optional[List[float]] = None,
    extensions: Optional[List[str]] = None,
    library_name: Optional[str] = None,
    njoy_version: str = "NJOY 2016.78",
    xsdir_file: Optional[str] = None,
    enforce_positivity: bool = False,
    positivity_check_points: int = 101,
):
    """
    Process a single perturbation sample for ENDF files.
    
    Parameters
    ----------
    endf_file : str
        Path to the original ENDF file
    sample : np.ndarray
        Perturbation factors for this sample
    sample_index : int
        Index of this sample (0-based)
    energy_grids : Dict[Tuple[int, int, int], List[float]]
        Energy grids for each (isotope, mt, l) combination
    param_mapping : List[Tuple[int, int, int]]
        Mapping of sample indices to (isotope, mt, l) parameters
    output_dir : str
        Output directory for results
    dry_run : bool
        If True, only generate factors without creating ENDF files
    generate_ace : bool
        If True, generate ACE files using NJOY for each perturbed ENDF
    njoy_exe : Optional[str]
        Path to NJOY executable
    temperatures : Optional[Union[float, List[float]]]
        Temperature(s) for ACE generation (can be single float or list)
    library_name : Optional[str]
        Nuclear data library name
    njoy_version : str
        NJOY version string
    """
    if dry_run:
        # Dry-run is a fast preview: write the factor summary, do not
        # apply factors and do not run the positivity check. For a
        # positivity audit, do a real run with a small num_samples.
        endf = parse_endf_file(endf_file)
        base = os.path.splitext(os.path.basename(endf_file))[0]
        sample_str = f"{sample_index+1:04d}"
        sample_dir = os.path.join(output_dir, "endf", str(endf.zaid or "unknown"), sample_str)
        os.makedirs(sample_dir, exist_ok=True)

        _write_sample_summary(
            sample=sample,
            sample_index=sample_index,
            energy_grids=energy_grids,
            param_mapping=param_mapping,
            sample_dir=sample_dir,
            base=base,
        )
        return {
            "success": True,
            "temperatures_processed": [],
            "errors": [],
            "warnings": [],
            "positivity_events": [],
            "dry_run": True,
        }
    
    # Compute the expected output path before doing any heavy work. If the
    # perturbed ENDF already exists, reuse it and skip parse + perturbation +
    # write — this lets re-runs cheaply add ACE files at a new temperature.
    # A targeted MF1 parse is enough to recover the ZAID; the full parse only
    # happens on the write path below. That was the intent all along, but
    # read_endf only set `mat` on the full-parse path, so `zaid` came back None
    # and every sample landed in endf/unknown/ while stage B looked for it
    # under endf/<zaid>/ and paired nothing.
    base, ext = os.path.splitext(os.path.basename(endf_file))
    sample_str = f"{sample_index+1:04d}"
    try:
        zaid_dir = str(read_endf(endf_file, mf_numbers=[1]).zaid or "unknown")
    except Exception:
        zaid_dir = "unknown"
    sample_dir = os.path.join(output_dir, "endf", zaid_dir, sample_str)
    out_endf = os.path.join(sample_dir, f"{base}_{sample_str}{ext}")

    if os.path.exists(out_endf):
        reused = True
        positivity_events: List[Tuple[int, float, float, float]] = []
    else:
        reused = False
        endf = parse_endf_file(endf_file)
        perturbed_params, positivity_events = apply_perturbation_factors_to_endf(
            endf, sample, sample_index, energy_grids, param_mapping, verbose=False,
            enforce_positivity=enforce_positivity,
            positivity_check_points=positivity_check_points,
        )
        os.makedirs(sample_dir, exist_ok=True)
        writer = ENDFWriter(endf_file)
        if 4 in endf.files:
            success = writer.replace_mf_section(endf.files[4], out_endf)
            if not success:
                if _get_logger():
                    _get_logger().error(f"  [ERROR] [ENDF] Failed to write perturbed ENDF file: {out_endf}")
                return {"success": False, "positivity_events": positivity_events,
                        "temperatures_processed": [], "errors": [], "warnings": [],
                        "reused": False}
        _write_sample_summary(
            sample=sample,
            sample_index=sample_index,
            energy_grids=energy_grids,
            param_mapping=param_mapping,
            sample_dir=sample_dir,
            base=base,
        )

    # Generate ACE files using NJOY if requested
    if generate_ace and not dry_run:
        njoy_result = _process_njoy_for_sample(
            out_endf=out_endf,
            sample_index=sample_index,
            njoy_exe=njoy_exe,
            temperatures=temperatures,
            extensions=extensions,
            library_name=library_name,
            njoy_version=njoy_version,
            output_dir=output_dir,
            xsdir_file=xsdir_file,
        )
        njoy_result["positivity_events"] = positivity_events
        njoy_result["reused"] = reused
        return njoy_result

    return {"success": True, "temperatures_processed": [], "errors": [],
            "warnings": [], "positivity_events": positivity_events,
            "reused": reused}


def _process_njoy_for_sample(
    out_endf: str,
    sample_index: int,
    njoy_exe: str,
    temperatures: List[float],
    library_name: str,
    njoy_version: str,
    output_dir: str,
    xsdir_file: Optional[str] = None,
    extensions: Optional[List[str]] = None,
):
    """
    Process a perturbed ENDF file through NJOY to generate ACE files.
    
    Parameters
    ----------
    out_endf : str
        Path to the perturbed ENDF file
    sample_index : int
        Sample index (0-based)
    njoy_exe : str
        Path to NJOY executable
    temperatures : List[float]
        List of temperatures for ACE generation
    library_name : str
        Nuclear data library name
    njoy_version : str
        NJOY version string
    output_dir : str
        Base output directory
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with success status and any error messages
    """
    
    logger = _get_logger()
    sample_str = f"{sample_index+1:04d}"
    
    # Parse ENDF to get ZAID for directory organization
    try:
        endf_data = read_endf(out_endf)
        zaid = endf_data.zaid or "unknown"
    except Exception as e:
        if logger:
            logger.error(f"  [ERROR] [NJOY] Sample {sample_str}: Failed to parse ENDF for ZAID - {e}")
        return {"success": False, "error": f"Failed to parse ENDF for ZAID: {e}"}
    
    results = {"success": True, "temperatures_processed": [], "errors": [], "warnings": []}

    # New layout: per-sample directory, all temperatures + isotopes co-located,
    # ZAID.<ext> filenames disambiguate them. xsdir snippets land in a
    # `xsdir/` subdirectory of the sample directory.
    ace_sample_dir, xsdir_sample_dir, _ = make_ace_sample_paths(output_dir, sample_index)
    njoy_sample_dir = os.path.join(output_dir, "njoy_files", sample_str)
    os.makedirs(ace_sample_dir, exist_ok=True)
    os.makedirs(xsdir_sample_dir, exist_ok=True)
    os.makedirs(njoy_sample_dir, exist_ok=True)

    if extensions is not None and len(extensions) != len(temperatures):
        msg = (
            f"extensions length ({len(extensions)}) must match temperatures length "
            f"({len(temperatures)}); ignoring extensions."
        )
        if logger:
            logger.warning(f"  [WARN] [NJOY] Sample {sample_str}: {msg}")
        extensions = None

    for t_idx, temp in enumerate(temperatures):
        try:
            ext_str = extensions[t_idx] if extensions is not None else None

            # Run NJOY directly into the per-sample dirs. The runner places
            # ACE at ace_sample_dir/<zaid>.<ext> and xsdir at
            # xsdir_sample_dir/<zaid>.<ext>.xsdir when `extension` is given.
            with tempfile.TemporaryDirectory(prefix="njoy_temp_", dir=output_dir) as temp_dir:
                result = run_njoy(
                    njoy_exe=njoy_exe,
                    endf_path=out_endf,
                    temperature=temp,
                    library_name=library_name,
                    output_dir=temp_dir,
                    njoy_version=njoy_version,
                    additional_suffix=sample_str if ext_str is None else None,
                    extension=ext_str,
                    ace_dir=ace_sample_dir if ext_str is not None else None,
                    xsdir_dir=xsdir_sample_dir if ext_str is not None else None,
                    njoy_files_dir=njoy_sample_dir if ext_str is not None else None,
                )

                # Always move NJOY auxiliary files (input/output logs) for debugging
                aux_files = ["njoy_input", "njoy_output", "viewr_output"]
                for aux_file in aux_files:
                    src = result.get(aux_file)
                    if src and os.path.exists(src):
                        if os.path.dirname(src) != njoy_sample_dir:
                            dest_aux = os.path.join(njoy_sample_dir, os.path.basename(src))
                            try:
                                shutil.move(src, dest_aux)
                                if aux_file == "njoy_output":
                                    results["njoy_output_path"] = dest_aux
                            except Exception as move_err:
                                warning_msg = f"Could not move {aux_file} at {temp}K: {move_err}"
                                results["warnings"].append(warning_msg)
                                if logger:
                                    logger.warning(f"  [WARN] [NJOY] Sample {sample_str}: {warning_msg}")
                        elif aux_file == "njoy_output":
                            results["njoy_output_path"] = src

                if result["returncode"] == 0:
                    results["temperatures_processed"].append(temp)

                    # When the legacy layout is in use (no extension), the
                    # runner put the ACE/xsdir under a library/<temp>K dir
                    # inside temp_dir — relocate to the sample dir manually.
                    if ext_str is None and result.get("ace_file") and os.path.exists(result["ace_file"]):
                        ace_filename = os.path.basename(result["ace_file"])
                        dest_ace = os.path.join(ace_sample_dir, ace_filename)
                        shutil.move(result["ace_file"], dest_ace)
                        result["ace_file"] = dest_ace
                    if ext_str is None and result.get("xsdir_file") and os.path.exists(result["xsdir_file"]):
                        xsd_dest = os.path.join(xsdir_sample_dir, os.path.basename(result["xsdir_file"]))
                        shutil.move(result["xsdir_file"], xsd_dest)
                        result["xsdir_file"] = xsd_dest

                    # Master-xsdir merge when requested.
                    if xsdir_file is not None and result.get("ace_file") and os.path.exists(result["ace_file"]):
                        try:
                            from kika.ace.parsers import read_ace
                            ace_data = read_ace(result["ace_file"])
                            hdr = ace_data.header
                            has_ptable = bool(getattr(ace_data.unresolved_resonance, 'has_data', False))
                            create_xsdir_files_for_ace(
                                ace_file_path=result["ace_file"],
                                zaid=hdr.zaid,
                                awr=hdr.atomic_weight_ratio,
                                xss_len=hdr.nxs_array[1],
                                temperature_mev=hdr.temperature,
                                sample_index=sample_index,
                                output_dir=output_dir,
                                master_xsdir_file=xsdir_file,
                                has_ptable=has_ptable,
                                per_sample_xsdir_path=result.get("xsdir_file"),
                            )
                        except Exception as xsdir_err:
                            warning_msg = f"Failed to update master XSDIR at {temp}K: {xsdir_err}"
                            results["warnings"].append(warning_msg)
                            if logger:
                                logger.warning(f"[NJOY] Sample {sample_str} at {temp}K: {warning_msg}")
                else:
                    error_msg = f"NJOY failed at {temp}K with return code {result['returncode']}"
                    results["errors"].append(error_msg)
                    results["success"] = False
                    if logger:
                        error_count = len(results["errors"])
                        if error_count <= 3:
                            logger.error(f"  [ERROR] [NJOY] Sample {sample_str}: {error_msg}")
                        elif error_count == 4:
                            logger.error(f"  [ERROR] [NJOY] Sample {sample_str}: Suppressing further identical errors (see batch summary)")

        except Exception as e:
            error_msg = f"Exception at {temp}K: {e}"
            results["errors"].append(error_msg)
            results["success"] = False
            if logger:
                logger.error(f"  [ERROR] [NJOY] Sample {sample_str}: {error_msg}")

    return results


def _log_positivity_events(
    events_by_sample: List[Tuple[int, List[Tuple[int, float, float, float]]]],
    file_key: str,
    num_samples: int,
    verbose: bool,
    logger,
):
    """
    Log MF4 positivity projection activity.

    Verbose mode: one line per projected (sample, MT, energy) tuple.
    Both modes: per-MT summary plus a cross-MT total.

    Parameters
    ----------
    events_by_sample : list of (sample_index, list of (mt, energy, sigma_min, max_delta))
        Per-sample projection records emitted by _apply_factors_to_mf4_legendre.
    file_key : str
        Short identifier of the current ENDF file for log context.
    num_samples : int
        Total samples processed for this file (denominator in the summary).
    verbose : bool
        If True, emit per-event lines; the summary is always emitted.
    logger : DualLogger
    """
    if logger is None:
        return
    per_mt_count: Dict[int, int] = {}
    per_mt_max_delta: Dict[int, float] = {}
    per_mt_max_sigma_below_zero: Dict[int, float] = {}
    n_events_total = 0

    for sample_index, events in events_by_sample:
        sample_str = f"{sample_index + 1:04d}"
        for mt, energy, sigma_min, max_delta in events:
            n_events_total += 1
            per_mt_count[mt] = per_mt_count.get(mt, 0) + 1
            per_mt_max_delta[mt] = max(per_mt_max_delta.get(mt, 0.0), max_delta)
            # sigma_min is negative here; track the most-negative value
            per_mt_max_sigma_below_zero[mt] = min(
                per_mt_max_sigma_below_zero.get(mt, 0.0), sigma_min
            )
            if verbose:
                logger.info(
                    f"  [INFO] [POSITIVITY] {file_key} sample={sample_str} "
                    f"MT={mt} E={energy:.6e} eV: f(μ)<0 (min={sigma_min:.3e}), "
                    f"projected (max |Δa_l|={max_delta:.3e})"
                )

    if n_events_total == 0:
        logger.info(
            f"  [INFO] [POSITIVITY] {file_key}: all {num_samples} samples already physical "
            f"(no projection needed)"
        )
        return

    for mt in sorted(per_mt_count):
        logger.info(
            f"  [INFO] [POSITIVITY] {file_key} MT={mt}: projected {per_mt_count[mt]} "
            f"(sample, energy) entries / {num_samples} samples; "
            f"max |Δa_l|={per_mt_max_delta[mt]:.3e}; "
            f"worst pre-projection σ(μ)={per_mt_max_sigma_below_zero[mt]:.3e}"
        )
    logger.info(
        f"  [INFO] [POSITIVITY] {file_key}: total {n_events_total} (sample, MT, energy) "
        f"entries projected across {num_samples} samples"
    )


def _log_njoy_batch_results(njoy_results, file_key, file_index, temperatures, summary_data):
    """
    Log NJOY processing results in batch to reduce log verbosity.
    
    Parameters
    ----------
    njoy_results : List[Tuple[int, Dict]]
        List of (sample_index, result_dict) tuples
    file_key : str
        File identifier for summary tracking
    file_index : int
        1-based file index
    temperatures : List[float]
        List of temperatures processed
    summary_data : Dict
        Summary data dictionary to update
    """
    logger = _get_logger()
    if not logger:
        return
    
    total_samples = len(njoy_results)
    successful_samples = sum(1 for _, result in njoy_results if result.get("success", False))
    failed_samples = total_samples - successful_samples
    
    # Count warnings and errors across all samples
    all_warnings = []
    all_errors = []
    temp_success_count = {temp: 0 for temp in temperatures}
    
    for sample_idx, result in njoy_results:
        if result.get("warnings"):
            all_warnings.extend(result["warnings"])
        if result.get("errors"):
            all_errors.extend(result["errors"])
        
        # Count successful temperatures
        for temp in result.get("temperatures_processed", []):
            temp_success_count[temp] += 1
    
    # Update summary data
    if file_key in summary_data:
        summary_data[file_key]['ace_generation']['successful_samples'] = successful_samples
        summary_data[file_key]['ace_generation']['failed_samples'] = failed_samples
    
    # Log batch summary
    logger.info(f"  [INFO] [NJOY] File {file_index}: Batch processing complete - {successful_samples}/{total_samples} samples successful", console=True)

    # Log temperature-specific results
    for temp in temperatures:
        success_count = temp_success_count[temp]
        temp_str = str(temp).rstrip('0').rstrip('.') if '.' in str(temp) else str(temp)
        logger.info(f"  [INFO] [NJOY] File {file_index}: Temperature {temp_str} - {success_count}/{total_samples} samples successful", console=True)

    # Log warnings (if any) - aggregate similar warnings
    if all_warnings:
        warning_counts = {}
        for warning in all_warnings:
            # Extract the core warning message (remove temperature-specific parts)
            core_warning = warning.split(" at ")[0] if " at " in warning else warning
            warning_counts[core_warning] = warning_counts.get(core_warning, 0) + 1

        logger.warning(f"  [WARN] [NJOY] File {file_index}: {len(all_warnings)} warnings occurred:")
        for warning, count in warning_counts.items():
            if count > 1:
                logger.warning(f"  [WARN] [NJOY] File {file_index}:   {warning} (occurred {count} times)")
            else:
                logger.warning(f"  [WARN] [NJOY] File {file_index}:   {warning}")

    # Log errors (if any) - aggregate similar errors
    if all_errors:
        error_counts = {}
        for error in all_errors:
            # Extract the core error message (remove temperature-specific parts)
            core_error = error.split(" at ")[0] if " at " in error else error
            error_counts[core_error] = error_counts.get(core_error, 0) + 1

        logger.error(f"  [ERROR] [NJOY] File {file_index}: {len(all_errors)} errors occurred:")
        for error, count in error_counts.items():
            if count > 1:
                logger.error(f"  [ERROR] [NJOY] File {file_index}:   {error} (occurred {count} times)")
            else:
                logger.error(f"  [ERROR] [NJOY] File {file_index}:   {error}")

        # Log NJOY output excerpt from first failure
        for sample_idx, result in njoy_results:
            if result.get("njoy_output_path") and result.get("errors"):
                try:
                    output_path = result["njoy_output_path"]
                    if os.path.exists(output_path):
                        with open(output_path, 'r') as f:
                            njoy_lines = f.readlines()
                        # Show last 10 non-empty lines
                        tail = [l.rstrip() for l in njoy_lines if l.strip()][-10:]
                        if tail:
                            logger.error(f"  [ERROR] [NJOY] File {file_index}: NJOY output excerpt (sample {sample_idx+1:04d}):")
                            for line in tail:
                                logger.error(f"  [ERROR] [NJOY] File {file_index}:   {line}")
                except Exception:
                    pass
                break  # Only show first failure

    if failed_samples == total_samples:
        logger.error(f"  [ERROR] [NJOY] File {file_index}: ALL {total_samples} samples failed -- systemic issue", console=True)
        logger.error(f"  [ERROR] [NJOY] File {file_index}: Check NJOY output excerpt above and sampling quality", console=True)


def load_mf34_covariance(path: str) -> Optional[LegendreCovariance]:
    """
    Load MF34 covariance matrix from ENDF file containing MF34 section.
    
    Parameters
    ----------
    path : str
        Path to the ENDF file containing MF34 covariance data
        
    Returns
    -------
    Optional[LegendreCovariance]
        Loaded LegendreCovariance object or None if loading failed
    """
    if not os.path.exists(path):
        if _get_logger():
            _get_logger().error(f"  [ERROR] [ENDF] ENDF file not found: {path}")
        return None
    
    try:
        logger = _get_logger()
        if logger:
            logger.info(f"  [INFO] [ENDF] Loading MF34 covariance from file: {path}")
        
        # Parse the ENDF file
        endf = parse_endf_file(path)
        
        # Get MF34 section
        mf34 = endf.get_file(34)
        if mf34 is None:
            if logger:
                logger.error(f"  [ERROR] [ENDF] No MF34 section found in file: {path}")
            return None
        
        # Convert all MT sections to LegendreCovariance and combine them
        combined_mf34_cov = LegendreCovariance()
        
        for mt_number, mt_data in mf34.sections.items():
            if hasattr(mt_data, 'to_ang_covmat'):
                if logger:
                    logger.info(f"  [INFO] [ENDF] Converting MT{mt_number} to angular covariance matrix")
                
                mt_covmat = mt_data.to_ang_covmat()
                
                # Add all matrices from this MT to the combined object
                for i in range(mt_covmat.num_matrices):
                    # Check if matrix is relative (required for sampling)
                    if not mt_covmat.is_relative[i]:
                        if logger:
                            logger.warning(f"  [WARN] [ENDF] Skipping absolute covariance matrix for isotope {mt_covmat.isotope_rows[i]} "
                                         f"MT{mt_covmat.reaction_rows[i]} L={mt_covmat.l_rows[i]} <-> "
                                         f"isotope {mt_covmat.isotope_cols[i]} MT{mt_covmat.reaction_cols[i]} L={mt_covmat.l_cols[i]}. "
                                         f"Only relative covariance matrices are supported for sampling.")
                        continue
                    
                    # Check if frame is compatible (should be "same-as-MF4")
                    if mt_covmat.frame[i] != "same-as-MF4":
                        if logger:
                            logger.warning(f"  [WARN] [ENDF] Covariance matrix for isotope {mt_covmat.isotope_rows[i]} "
                                         f"MT{mt_covmat.reaction_rows[i]} L={mt_covmat.l_rows[i]} <-> "
                                         f"isotope {mt_covmat.isotope_cols[i]} MT{mt_covmat.reaction_cols[i]} L={mt_covmat.l_cols[i]} "
                                         f"has reference frame '{mt_covmat.frame[i]}' instead of 'same-as-MF4'. "
                                         f"This may cause inconsistencies in perturbation.")
                    
                    combined_mf34_cov.add_matrix(
                        isotope_row=mt_covmat.isotope_rows[i],
                        reaction_row=mt_covmat.reaction_rows[i],
                        l_row=mt_covmat.l_rows[i],
                        isotope_col=mt_covmat.isotope_cols[i],
                        reaction_col=mt_covmat.reaction_cols[i],
                        l_col=mt_covmat.l_cols[i],
                        matrix=mt_covmat.matrices[i],
                        energy_grid=mt_covmat.energy_grids[i],
                        is_relative=mt_covmat.is_relative[i],
                        frame=mt_covmat.frame[i]
                    )
        
        if logger:
            logger.info(f"  [INFO] [ENDF] Successfully loaded covariance with {combined_mf34_cov.num_matrices} matrices")
            logger.info(f"  [INFO] [ENDF] Available isotopes: {sorted(combined_mf34_cov.isotopes)}")
            logger.info(f"  [INFO] [ENDF] Available reactions: {sorted(combined_mf34_cov.reactions)}")
            logger.info(f"  [INFO] [ENDF] Available Legendre indices: {sorted(combined_mf34_cov.legendre_indices)}")
        
        return combined_mf34_cov
        
    except Exception as e:
        if _get_logger():
            _get_logger().error(f"  [ERROR] [ENDF] Failed to load MF34 covariance: {e}")
        return None


def load_mf34_suite(path: str):
    """Load *path*'s covariance as the format-agnostic ``CovarianceSuite``.

    The replacement for :func:`load_mf34_covariance` on the sampling path, and
    the reason this module now reaches the GNDS adapter at all. Same contract:
    ``None`` on failure, with the reason logged, so the caller's per-file
    bookkeeping is unchanged.

    **The import is function-scope on purpose.** ``kika.endf.model_adapter``
    pulls in the whole GNDS model, and ``kika.sampling`` is imported on every
    cluster run; at module scope this would wake the model for callers that
    never sample MF34. ``test_only_the_facades_and_the_front_door_import_the_adapter``
    is the ratchet that holds this, and this module is on its allowlist.
    """
    from kika.endf.model_adapter import decodeCovarianceSuite

    logger = _get_logger()
    try:
        if logger:
            logger.info(f"  [INFO] [ENDF] Loading MF34 covariance from file: {path}")

        endf = read_endf(path)
        if endf.get_file(34) is None:
            if logger:
                logger.error(f"  [ERROR] [ENDF] No MF34 section found in file: {path}")
            return None

        suite, report = decodeCovarianceSuite(endf)
        if logger:
            for line in getattr(report, "unsupported", ()) or ():
                logger.info(f"  [INFO] [ENDF] {line}")
        return suite

    except Exception as e:
        if logger:
            logger.error(f"  [ERROR] [ENDF] Failed to load MF34 covariance: {e}")
        return None


def _mf34_available(suite) -> Tuple[List[int], List[int]]:
    """The MTs and Legendre orders *suite*'s MF34 sections mention.

    What ``resolve_signed_request`` needs, and the suite-side replacement for
    ``LegendreCovariance.reactions`` / ``.legendre_indices``. Read off the index
    rather than by walking sections, so it agrees with what would actually be
    assembled — including a component a file mentions only on the *column* of a
    cross block.
    """
    index = legendre_covariance_index(suite)
    if not index:
        return [], []
    triplets = next(iter(index.values()))["triplets"]
    return (sorted({mt for _za, mt, _l in triplets}),
            sorted({l for _za, _mt, l in triplets}))


def _parameter_mapping_from_index(entry) -> Tuple[
    List[Tuple[int, int, int, int]], Dict[Tuple[int, int, int], List[float]]
]:
    """A :func:`legendre_covariance_index` entry → what the appliers read.

    The successor of ``_create_parameter_mapping``, and deliberately a pure
    reshaping of what the index already says: ``triplets`` in row order, each
    occupying ``stride`` rows, and the per-triplet union grids. Both facts are
    the index's, so the flat layout can no longer disagree with the matrix the
    draw came from — which it could when one was derived from a
    ``LegendreCovariance`` and the other from the suite.

    Note the uniform stride is preserved, padding included: a triplet whose own
    grid is shorter than ``stride`` still gets ``stride`` entries. Those columns
    are inert (their rows in the joint are all zero), and the appliers already
    skip an ``energy_bin`` past the end of its grid.
    """
    stride = entry["stride"]
    param_mapping = [
        (za, mt, l_coeff, energy_bin)
        for (za, mt, l_coeff) in entry["triplets"]
        for energy_bin in range(stride)
    ]
    energy_grids = {
        triplet: np.asarray(entry["grids"][triplet], dtype=float).tolist()
        for triplet in entry["triplets"]
    }
    return param_mapping, energy_grids


def perturb_ENDF_files(
    endf_files: Union[str, List[str]],
    mt_list: Union[List[int], List[List[int]]],
    legendre_coeffs: List[int],
    num_samples: int,
    mf34_cov_files: Optional[Union[str, List[str]]] = None,
    space: str = "linear",
    decomposition_method: str = "svd",
    psd_method: str = "none",
    higham_projection: bool = False,
    enforce_positivity: bool = True,
    positivity_check_points: int = 101,
    sampling_method: str = "sobol",
    output_dir: str = '.',
    seed: Optional[int] = None,
    nprocs: int = 1,
    dry_run: bool = False,
    verbose: bool = True,
    generate_ace: bool = False,
    njoy_exe: Optional[str] = None,
    temperatures: Optional[Union[float, List[float]]] = None,
    extensions: Optional[List[str]] = None,
    library_name: Optional[str] = None,
    njoy_version: str = "NJOY 2016.78",
    xsdir_file: Optional[str] = None,
    energy_ranges: Optional[List[Tuple[float, float]]] = None,
    only_xsdir: bool = False,
    log_file: Optional[str] = None,
):
    """
    Perturb ENDF nuclear data files using MF34 angular covariance matrices.

    This function generates perturbed ENDF files by sampling perturbation factors
    from multivariate normal distributions derived from MF34 covariance matrices.
    Optionally, it can also generate ACE files for each perturbed ENDF using NJOY.
    
    Parameters
    ----------
    endf_files : Union[str, List[str]]
        Path(s) to ENDF file(s) to be perturbed
    mt_list : List[int] or List[List[int]]
        MT reactions to perturb. Either a flat list (applied to every ENDF file)
        or a list-of-lists with one entry per ENDF file. An empty inner list
        means "perturb all available MTs" for that file.
    legendre_coeffs : List[int]
        List of Legendre coefficient indices to perturb (e.g., [0, 1, 2])
    num_samples : int
        Number of perturbed ENDF files to generate
    mf34_cov_files : Optional[Union[str, List[str]]], default None
        Path(s) to MF34 covariance matrix file(s). If None, covariance data will be
        read from the ENDF files themselves (MF34 section). If provided, can be a 
        single file (used for all ENDF files) or one file per ENDF file.
    space : str, default "linear"
        Sampling space: "linear"/"lin" (factors = 1 + X) or "log" (factors = exp(Y))
    decomposition_method : str, default "svd"
        Matrix decomposition method: "svd", "cholesky", "eigen", or "pca"
    psd_method : str, default "none"
        How to handle a non-PSD covariance during decomposition:
            "none"   — do not modify the matrix (default); SVD silently folds
                       negative eigenvalues to their absolute value. A WARN
                       line is logged when |λ_min|/λ_max exceeds 1e-8.
            "auto"   — clip mild negatives, Higham for severe ones.
            "clip"   — zero out negative eigenvalues.
            "higham" — nearest-PSD Higham projection.
        Use `higham_projection=True` as a shortcut for "higham".
    higham_projection : bool, default False
        Convenience flag — when True, force `psd_method="higham"`. Raises if
        the user also passes a non-default `psd_method`.
    enforce_positivity : bool, default True
        After each MF4 sample is assembled (factor × baseline), check that
        f(μ) = (1/2) Σ (2l+1) a_l P_l(μ) ≥ 0 on a fine μ-grid. If not,
        project to the L²-nearest non-negative coefficient set via SLSQP.
        Applies during the perturbed-ENDF write path. **Skipped in
        dry_run** — dry_run only writes the factor parquet, it does not
        apply factors or check positivity. To audit positivity, just do
        a real run with a small `num_samples`.
    positivity_check_points : int, default 101
        Number of μ points used for the positivity check / projection.
    sampling_method : str, default "sobol"
        Sampling method: "sobol", "lhs", or "random"
    extensions : Optional[List[str]], default None
        ACE extensions to assign per temperature, e.g. ['06c'] for 600 K.
        Must have the same length as `temperatures`. When None, NJOY's
        internal K_TO_SUFFIX lookup is used (legacy behavior).
    output_dir : str, default "."
        Output directory for perturbed files
    seed : Optional[int], default None
        Random seed for reproducible sampling
    nprocs : int, default 1
        Number of parallel processes for sample processing
    dry_run : bool, default False
        If True, only generate perturbation factors without creating ENDF files
    verbose : bool, default True
        Enable verbose logging output
    generate_ace : bool, default False
        If True, generate ACE files for each perturbed ENDF using NJOY
    njoy_exe : Optional[str], default None
        Path to NJOY executable. Required if generate_ace is True
    temperatures : Optional[Union[float, List[float]]], default None
        Temperature(s) (in Kelvin) for ACE generation. Can be a single float or list of floats.
        Required if generate_ace is True.
    library_name : Optional[str], default None
        Nuclear data library name (e.g., 'endfb81', 'jeff40'). Required if generate_ace is True
    njoy_version : str, default "NJOY 2016.78"
        NJOY version string for metadata and titles
    xsdir_file : Optional[str], default None
        Path to master XSDIR file to be modified for each generated ACE file.
        Only used when generate_ace=True.
    energy_ranges : Optional[List[Tuple[float, float]]], default None
        List of (emin, emax) energy ranges in eV where perturbations are applied.
        Bins outside these ranges are left unperturbed (factor = 1.0).
        None means perturb all energies. Examples:
        - [(0, 0.85e6), (4e6, 20e6)] : perturb up to 0.85 MeV and from 4 MeV onward
        - [(1e6, 4e6)] : perturb only between 1-4 MeV

    Notes
    -----
    Default behavior samples directly from the MF34 covariance with no
    repair (`psd_method="none"`). Mild negative eigenvalues from numerical
    noise are silently folded to their magnitude by SVD; a WARN line is
    logged when the negative spectrum is non-trivial
    (|λ_min|/λ_max > 1e-8). To repair the matrix, set
    `higham_projection=True` (Higham projection) or pass an explicit
    `psd_method` ("auto", "clip", or "higham"). The two are mutually
    exclusive.
    
    When generate_ace=True, the output directory structure will include:
    - output_dir/endf/zaid/sample_num/ : perturbed ENDF files
    - output_dir/ace/temp/zaid/sample_num/ : ACE files organized by temperature (temp uses exact input format)
    - output_dir/njoy_files/temp/zaid/ : NJOY auxiliary files
    - output_dir/xsdir/ : modified xsdir files (if xsdir_file provided)
    - output_dir/*.log and *.parquet : log and master perturbation files
    """
    global _logger

    # Normalize space parameter: support both 'lin' and 'linear'
    if space.lower() == 'lin':
        space = 'linear'

    # The two settings `generate_endf_samples` accepted and `draw_samples` does
    # not mean the same thing by. Refused by name rather than quietly drawing
    # something else, which is what carrying them over would have done.
    #
    # `space="log"`: the old path drew from `LegendreCovariance`'s *separately
    # computed* log covariance. `draw_samples` decomposes the matrix it is
    # handed and moment-matches against that same matrix, so asking it for a log
    # draw of a linear covariance gives a distribution neither function
    # describes. The suite has no log form to hand it, and MF34 stores Legendre
    # coefficients that are routinely negative -- a log-space multiplicative
    # factor is not obviously meaningful for them -- so this is not plugged with
    # a transform until someone argues for what it should mean.
    if space.lower() != "linear":
        raise NotImplementedError(
            f"space={space!r} is not available on the MF34 path. It drew from "
            f"LegendreCovariance.log_covariance_matrix, which the covariance "
            f"suite has no equivalent of; draw_samples would moment-match "
            f"against the linear matrix instead and silently draw a third "
            f"thing. Use space='linear'."
        )
    # `pca` truncates at 99.9 % of the variance and `cholesky` is refused
    # outright by draw_samples: these covariances are rank deficient by
    # construction, so a Cholesky factor is either a failure or a fiction.
    if decomposition_method.lower() not in ("svd", "eigen"):
        raise NotImplementedError(
            f"decomposition_method={decomposition_method!r} is not available on "
            f"the MF34 path; draw_samples offers 'svd' (the shipped setting) "
            f"and 'eigen'."
        )

    # Resolve higham_projection shorthand into psd_method. The two are
    # mutually exclusive — flagging both is a user error worth surfacing
    # loudly rather than silently picking one.
    if higham_projection:
        if psd_method != "none":
            raise ValueError(
                "higham_projection=True conflicts with an explicit "
                f"psd_method={psd_method!r}. Pass one or the other."
            )
        psd_method = "higham"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging — when ``log_file`` is provided (e.g. by the combined
    # orchestrator), append this stage's banner+log to that shared file.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if log_file is None:
        log_file = os.path.join(output_dir, f'endf_perturbation_{timestamp}.log')
        _logger = DualLogger(log_file, mode='w')
    else:
        _logger = DualLogger(log_file, mode='a')
    _set_logger(_logger)
    
    # Console: Basic start message
    _logger.info("[INFO] [ENDF] Starting ENDF perturbation job", console=True)
    _logger.info(f"[INFO] [ENDF] Log file: {log_file}", console=True)
    _logger.info(f"[INFO] [ENDF] Output directory: {os.path.abspath(output_dir)}", console=True)
    if generate_ace:
        _logger.info("[INFO] [ENDF] ACE generation enabled", console=True)

    if only_xsdir:
        if dry_run:
            raise ValueError("only_xsdir and dry_run cannot both be True")
        _logger.info(
            "[INFO] [ENDF] only_xsdir=True — skipping ENDF perturbation and NJOY; "
            "rewrapping xsdir entries for the requested isotopes only.",
            console=True,
        )
        if isinstance(endf_files, str):
            endf_files_list = [endf_files]
        else:
            endf_files_list = list(endf_files)
        zaids = []
        for ef in endf_files_list:
            try:
                zaids.append(read_endf(ef).zaid)
            except Exception as e:
                _logger.error(f"  [ERROR] [ENDF] Could not read zaid from {ef}: {e}")
        count = _rewrite_xsdir_for_zaids(
            zaids=zaids,
            output_dir=output_dir,
            xsdir_file=xsdir_file,
            logger=_logger,
        )
        _logger.info(f"[INFO] [ENDF] only_xsdir complete: {count} file(s) modified", console=True)
        return
    
    # Validate NJOY parameters if ACE generation is requested
    if generate_ace:
        if njoy_exe is None:
            raise ValueError("njoy_exe must be provided when generate_ace=True")
        if temperatures is None:
            raise ValueError("temperatures must be provided when generate_ace=True")
        
        # Convert temperature to list if it's a single float
        if isinstance(temperatures, (int, float)):
            temperatures = [float(temperatures)]
        elif isinstance(temperatures, list):
            if len(temperatures) == 0:
                raise ValueError("temperatures list cannot be empty when generate_ace=True")
            temperatures = [float(t) for t in temperatures]
        else:
            raise ValueError("temperatures must be a float or list of floats")
            
        if library_name is None:
            raise ValueError("library_name must be provided when generate_ace=True")
        if not os.path.exists(njoy_exe):
            raise FileNotFoundError(f"NJOY executable not found: {njoy_exe}")
    
    # Print run parameters as metadata TO LOG FILE
    _logger.info("#== CONFIG ============================================================")

    # Format and print input files
    if isinstance(endf_files, str):
        endf_files = [endf_files]
        _logger.info(f"  ENDF_FILES = {endf_files[0]}")
    else:
        _logger.info(f"  ENDF_FILES = ({len(endf_files)} files)")
        for i, f in enumerate(endf_files):
            _logger.info(f"    [{i+1}] {f}")

    # Handle MF34 covariance files parameter
    if mf34_cov_files is None:
        _logger.info("  MF34_COV_FILES = None (will read from ENDF files)")
        # Use ENDF files themselves as covariance sources
        mf34_cov_files = endf_files[:]
    elif isinstance(mf34_cov_files, str):
        mf34_cov_files = [mf34_cov_files]
        _logger.info(f"  MF34_COV_FILES = {mf34_cov_files[0]}")
    else:
        _logger.info(f"  MF34_COV_FILES = ({len(mf34_cov_files)} files)")
        for i, f in enumerate(mf34_cov_files):
            _logger.info(f"    [{i+1}] {f}")

    # Normalize MT input to a per-file list-of-lists (broadcasts a flat list).
    mt_lists = normalize_mt_list(mt_list, len(endf_files), unit_label="ENDF files")
    unique_mt_lists = {tuple(sub) for sub in mt_lists}
    if len(unique_mt_lists) <= 1:
        _logger.info(f"  MT_REACTIONS = {mt_lists[0] if mt_lists[0] else 'All available'}")
    else:
        _logger.info("  MT_REACTIONS = per-file (see below)")
        for endf_path, mts in zip(endf_files, mt_lists):
            display = mts if mts else "All available"
            _logger.info(f"    - {os.path.basename(endf_path)}: {display}")
    _logger.info(f"  LEGENDRE_COEFFS = {legendre_coeffs}")
    _logger.info(f"  NUM_SAMPLES = {num_samples}")
    _logger.info(f"  SAMPLING_SPACE = {space}")
    _logger.info(f"  DECOMPOSITION_METHOD = {decomposition_method}")
    _logger.info(f"  SAMPLING_METHOD = {sampling_method}")
    _logger.info(f"  PSD_METHOD = {psd_method}")
    _logger.info(f"  RANDOM_SEED = {seed}")
    _logger.info(f"  NPROCS = {nprocs}")
    _logger.info(f"  DRY_RUN = {dry_run}")
    _logger.info(f"  GENERATE_ACE = {generate_ace}")
    if energy_ranges is not None:
        ranges_mev = [(e1/1e6, e2/1e6) for e1, e2 in energy_ranges]
        _logger.info(f"  ENERGY_RANGES_MEV = {ranges_mev}")
    else:
        _logger.info("  ENERGY_RANGES = all")
    if generate_ace:
        _logger.info(f"  NJOY_EXE = {njoy_exe}")
        _logger.info(f"  TEMPERATURES = {temperatures}")
        _logger.info(f"  LIBRARY_NAME = {library_name}")
        _logger.info(f"  NJOY_VERSION = {njoy_version}")
        _logger.info(f"  XSDIR_FILE = {xsdir_file if xsdir_file else 'None'}")
    _logger.info("#== END CONFIG ========================================================")
    
    # Validate inputs
    if mf34_cov_files is not None and len(mf34_cov_files) != 1 and len(mf34_cov_files) != len(endf_files):
        raise ValueError("Number of MF34 covariance files must be 1 or equal to number of ENDF files")
    
    # Initialize summary tracking
    summary_data = {}
    failed_files_details = {}
    
    # Process each ENDF file
    all_factors = []
    all_param_mappings = []
    all_energy_grids = []
    processed_files = 0
    failed_files = 0
    
    total_warnings_count = 0

    for i, endf_file in enumerate(endf_files):
        # Pick the MT list for this specific ENDF file (per-file or broadcast).
        mt_list = mt_lists[i]
        step_num = i + 1
        step_t0 = time.time()
        _logger.info(f"#-- STEP {step_num}: Process {os.path.basename(endf_file)} ({step_num}/{len(endf_files)}) ------------------------------------------------")
        try:
            
            # Initialize summary data for this file
            file_key = os.path.basename(endf_file)
            summary_data[file_key] = {
                'file_path': endf_file,
                'file_index': i + 1,
                'perturbed_mts': [],
                'perturbed_l_coeffs': [],
                'num_samples': num_samples,
                'warnings': [],
                'sampling_diagnostics': None,  # Will be populated if diagnostics are available
                'ace_generation': {
                    'enabled': generate_ace,
                    'successful_samples': 0,
                    'failed_samples': 0,
                    'temperatures': temperatures if generate_ace else None,
                    'xsdir_created': False
                }
            }
            
            # Determine which covariance file to use
            cov_file = mf34_cov_files[0] if len(mf34_cov_files) == 1 else mf34_cov_files[i]
            
            # Load covariance matrix
            suite = load_mf34_suite(cov_file)
            if suite is None:
                _logger.error(f"  [ERROR] [ENDF] File {step_num}: Failed to load covariance matrix from {cov_file}")
                failed_files_details[file_key] = "Failed to load MF34 covariance matrix"
                failed_files += 1
                step_elapsed = time.time() - step_t0
                _logger.info(f"#-- END STEP {step_num} (elapsed: {step_elapsed:.1f}s) -------------------------------------")
                continue

            # Resolve positive/negative MT and L semantics against what's
            # actually in the MF34 file (typically only MT2 has data).
            available_mts_in_cov, available_ls_in_cov = _mf34_available(suite)
            mt_resolved, mt_log = resolve_signed_request(
                list(mt_list or []), available_mts_in_cov, group_map=MT_GROUPS,
            )
            for line in mt_log:
                _logger.info(
                    f"  [INFO] [ENDF] File {step_num}: "
                    f"{line.replace('Excluding entries:', 'Excluding MTs:').replace('group expansion', 'composite expansion')}"
                )
            requested_mts_missing = (
                sorted(set(mt_resolved) - set(available_mts_in_cov)) if mt_resolved else []
            )
            if requested_mts_missing:
                _logger.warning(
                    f"  [WARN] [ENDF] File {step_num}: MTs {requested_mts_missing} "
                    f"not in MF34 (MF34 has {available_mts_in_cov}); silently dropped."
                )
            l_resolved, l_log = resolve_signed_request(
                list(legendre_coeffs or []), available_ls_in_cov, group_map=None,
            )
            for line in l_log:
                _logger.info(
                    f"  [INFO] [ENDF] File {step_num}: "
                    f"{line.replace('Excluding entries:', 'Excluding L:')}"
                )

            # Assemble the one covariance the file's MF34 sections partition,
            # filtered to the resolved MTs and Legendre orders. The order filter
            # is what keeps L=0 out, and on an `_a0cross` tape a0 is the section
            # carrying the MF33 cross term on the magnitude grid -- so this is
            # also what keeps two incompatible energy grids from meeting.
            # `relative=True` reproduces `load_mf34_covariance`, which has always
            # dropped absolute MF34 sections: the appliers multiply by what comes
            # back, and an absolute covariance does not describe a factor.
            blocks = legendre_covariance_blocks(
                suite, mt=mt_resolved, orders=l_resolved, relative=True,
            )
            index = legendre_covariance_index(
                suite, mt=mt_resolved, orders=l_resolved, relative=True,
            )

            if not blocks:
                _logger.warning(f"  [WARN] [ENDF] File {step_num}: No covariance data found for requested MTs {mt_list} and L coefficients {legendre_coeffs}")
                failed_files_details[file_key] = f"No covariance data for requested MTs {mt_list} and L coefficients {legendre_coeffs}"
                failed_files += 1
                step_elapsed = time.time() - step_t0
                _logger.info(f"#-- END STEP {step_num} (elapsed: {step_elapsed:.1f}s) -------------------------------------")
                continue

            (block_key, joint), = blocks
            triplets = index[block_key]["triplets"]

            # Store which MTs and L coefficients will be perturbed
            summary_data[file_key]['perturbed_mts'] = sorted({mt for _za, mt, _l in triplets})
            summary_data[file_key]['perturbed_l_coeffs'] = sorted({l for _za, _mt, l in triplets})

            # Create parameter mapping and energy grids
            param_mapping, energy_grids = _parameter_mapping_from_index(index[block_key])

            # Generate perturbation factors
            if verbose:
                _logger.info(f"  [INFO] [ENDF] File {step_num}: Generating {num_samples} perturbation samples...")
                _logger.info(
                    f"  [INFO] [ENDF] File {step_num}: joint covariance "
                    f"{joint.shape[0]}x{joint.shape[0]} over {len(triplets)} "
                    f"triplet(s), stride {index[block_key]['stride']}"
                )

            try:
                samples, diagnostic_results = draw_samples(
                    blocks,
                    num_samples,
                    space=space,
                    returns="factors",
                    decomposition_method=decomposition_method,
                    sampling_method=sampling_method,
                    seed=seed,
                    psd_method=psd_method,
                    # Retain the null directions. Truncating to the retained rank
                    # is `draw_samples`' default and is the better draw, but it
                    # changes the QMC dimension and so moves every drawn column
                    # -- 4218 -> 3603 on the shipped Fe-56 multigroup joint. It
                    # is turned on in its own commit, against a measurement,
                    # rather than as a side effect of changing which object the
                    # covariance arrives in.
                    null_tol=None,
                    dtype=np.float32,
                    verbose=verbose,
                )
                factors = samples[block_key]

                _logger.info(f"  [INFO] [ENDF] File {step_num}: Successfully generated perturbation factors: shape {factors.shape}")
                _logger.info(f">> factors_shape = {factors.shape}")

                # Apply energy range masking if specified
                if energy_ranges is not None:
                    factors = _mask_factors_by_energy_range(
                        factors, param_mapping, energy_grids, energy_ranges, space
                    )
                    n_masked = np.sum(factors[0] == (1.0 if space == "linear" else 0.0))
                    _logger.info(f"  [INFO] [ENDF] File {step_num}: Energy range filter applied -- {factors.shape[1] - n_masked}/{factors.shape[1]} parameters perturbed")

                # Store diagnostic results in summary data for later reporting
                if diagnostic_results:
                    summary_data[file_key]['sampling_diagnostics'] = diagnostic_results
                
            except Exception as e:
                _logger.error(f"  [ERROR] [ENDF] File {step_num}: Failed to generate perturbation factors: {e}")
                failed_files_details[file_key] = f"Perturbation generation failed: {str(e)}"
                failed_files += 1
                step_elapsed = time.time() - step_t0
                _logger.info(f"#-- END STEP {step_num} (elapsed: {step_elapsed:.1f}s) -------------------------------------")
                continue
            
            # Store factors and mapping for master file
            all_factors.append(factors)
            all_param_mappings.append(param_mapping)
            all_energy_grids.append(energy_grids)
            
            # Process samples
            if not dry_run:
                if verbose:
                    _logger.info(f"  [INFO] [ENDF] File {step_num}: Applying perturbations to {num_samples} samples...")
            else:
                if verbose:
                    _logger.info(f"  [INFO] [ENDF] File {step_num}: Dry run: generating factor summaries for {num_samples} samples...")
            
            # Process samples with optional parallelization
            njoy_results = []  # Track NJOY results for batch logging
            # (sample_index, [(mt, energy, sigma_min, max_delta_a_l), ...])
            positivity_events_by_sample: List[Tuple[int, List[Tuple[int, float, float, float]]]] = []
            # Mutable container so nested _absorb_result can mutate it without nonlocal.
            reused_count = [0]

            def _absorb_result(sample_idx, result):
                if not isinstance(result, dict):
                    return
                events = result.get("positivity_events") or []
                if events:
                    positivity_events_by_sample.append((sample_idx, events))
                if result.get("reused"):
                    reused_count[0] += 1
                if generate_ace and not dry_run:
                    njoy_results.append((sample_idx, result))

            if nprocs > 1 and num_samples > 1:
                if verbose:
                    _logger.info(f"  [INFO] [ENDF] File {step_num}: Using {nprocs} processes for parallel processing")

                try:
                    with Pool(processes=nprocs) as pool:
                        futures = []
                        for sample_idx in range(num_samples):
                            args = (
                                endf_file, factors[sample_idx], sample_idx, energy_grids,
                                param_mapping, output_dir, dry_run, generate_ace, njoy_exe,
                                temperatures, extensions, library_name, njoy_version, xsdir_file,
                                enforce_positivity, positivity_check_points,
                            )
                            future = pool.apply_async(_process_sample, args=args)
                            futures.append(future)

                        # Wait for all processes to complete
                        pool.close()
                        pool.join()

                        # Collect results and check for any exceptions
                        for j, future in enumerate(futures):
                            try:
                                result = future.get()  # This will raise any exception that occurred
                                _absorb_result(j, result)
                            except Exception as e:
                                _logger.error(f"  [ERROR] [ENDF] File {step_num}: Sample {j+1:04d} processing failed: {e}")

                except Exception as e:
                    _logger.error(f"  [ERROR] [ENDF] File {step_num}: Parallel processing failed, falling back to serial: {e}")
                    # Fall back to serial processing
                    njoy_results = []
                    positivity_events_by_sample = []
                    for sample_idx in range(num_samples):
                        try:
                            result = _process_sample(
                                endf_file=endf_file,
                                sample=factors[sample_idx],
                                sample_index=sample_idx,
                                energy_grids=energy_grids,
                                param_mapping=param_mapping,
                                output_dir=output_dir,
                                dry_run=dry_run,
                                generate_ace=generate_ace,
                                njoy_exe=njoy_exe,
                                temperatures=temperatures,
                                extensions=extensions,
                                library_name=library_name,
                                njoy_version=njoy_version,
                                xsdir_file=xsdir_file,
                                enforce_positivity=enforce_positivity,
                                positivity_check_points=positivity_check_points,
                            )
                            _absorb_result(sample_idx, result)
                        except Exception as sample_e:
                            _logger.error(f"  [ERROR] [ENDF] File {step_num}: Sample {sample_idx+1:04d} processing failed: {sample_e}")
                            continue
            else:
                # Serial processing
                for sample_idx in range(num_samples):
                    try:
                        result = _process_sample(
                            endf_file=endf_file,
                            sample=factors[sample_idx],
                            sample_index=sample_idx,
                            energy_grids=energy_grids,
                            param_mapping=param_mapping,
                            output_dir=output_dir,
                            dry_run=dry_run,
                            generate_ace=generate_ace,
                            njoy_exe=njoy_exe,
                            temperatures=temperatures,
                            extensions=extensions,
                            library_name=library_name,
                            njoy_version=njoy_version,
                            xsdir_file=xsdir_file,
                            enforce_positivity=enforce_positivity,
                            positivity_check_points=positivity_check_points,
                        )
                        _absorb_result(sample_idx, result)
                    except Exception as e:
                        _logger.error(f"  [ERROR] [ENDF] File {step_num}: Sample {sample_idx+1:04d} processing failed: {e}")
                        continue

            # Report reuse of existing perturbed ENDFs (re-runs at new temperature).
            if reused_count[0] > 0:
                _logger.info(
                    f"  [INFO] [ENDF] File {step_num}: Reused {reused_count[0]}/{num_samples} "
                    f"existing perturbed ENDFs on disk; skipped perturbation+write for those samples."
                )

            # Positivity reporting: per-event in verbose, summary always.
            if enforce_positivity:
                _log_positivity_events(
                    positivity_events_by_sample, file_key, num_samples,
                    verbose=verbose, logger=_logger,
                )

            # Batch logging for NJOY results
            if generate_ace and not dry_run and njoy_results:
                _log_njoy_batch_results(njoy_results, file_key, i+1, temperatures, summary_data)
            
            processed_files += 1
            if generate_ace and not dry_run:
                ace_info = summary_data.get(file_key, {}).get('ace_generation', {})
                ace_success = ace_info.get('successful_samples', 0)
                if ace_success == 0:
                    _logger.warning(f"  [WARN] [ENDF] File {step_num}: ENDF perturbation completed but ALL ACE generations failed (0/{num_samples})")
                elif ace_success < num_samples:
                    _logger.info(f"  [INFO] [ENDF] File {step_num}: ENDF perturbation completed, ACE: {ace_success}/{num_samples} succeeded")
                else:
                    _logger.info(f"  [INFO] [ENDF] File {step_num}: Successfully processed all {num_samples} samples")
            else:
                _logger.info(f"  [INFO] [ENDF] File {step_num}: Successfully generated {num_samples} perturbed ENDF files")
            _logger.info(f">> samples_generated = {num_samples}")
            step_elapsed = time.time() - step_t0
            _logger.info(f"#-- END STEP {step_num} (elapsed: {step_elapsed:.1f}s) -------------------------------------")

        except Exception as file_error:
            _logger.error(f"  [ERROR] [ENDF] File {step_num}: Critical error during processing: {file_error}")
            failed_files_details[file_key] = f"Critical processing error: {str(file_error)}"
            failed_files += 1
            step_elapsed = time.time() - step_t0
            _logger.info(f"#-- END STEP {step_num} (elapsed: {step_elapsed:.1f}s) -------------------------------------")
            continue
    
    # Write master perturbation factors file
    master_file = _write_master_perturbation_file(
        all_factors, all_param_mappings, all_energy_grids, output_dir, timestamp, verbose
    )
    
    # Final comprehensive summary
    _logger.info("#== SUMMARY ============================================================")

    if not summary_data and not failed_files_details:
        _logger.info("  No ENDF files were processed.")
    else:
        # Report files that were successfully processed
        successfully_processed = {k: v for k, v in summary_data.items() if k not in failed_files_details}

        if successfully_processed:
            _logger.info("  SUCCESSFULLY PROCESSED ENDF FILES:")

            for file_key, data in successfully_processed.items():
                _logger.info(f"  File: {file_key}")

                # MTs and Legendre coefficients that were perturbed
                if data['perturbed_mts']:
                    _logger.info(f"    Perturbed MT numbers: {', '.join(map(str, sorted(data['perturbed_mts'])))}")
                if data['perturbed_l_coeffs']:
                    _logger.info(f"    Perturbed Legendre coefficients: {', '.join(map(str, sorted(data['perturbed_l_coeffs'])))}")

                _logger.info(f"    Number of samples generated: {data['num_samples']}")

                # Sampling quality assessment. `draw_samples` reports per block,
                # and MF34 is one joint block -- so this reads the block out
                # rather than looking for the old `overall_status` verdict, which
                # no longer exists. What replaces it is not a grade but the
                # numbers a grade was standing in for: how much of the matrix was
                # actually spanned, and how well the draw reproduced it.
                diagnostics = data.get('sampling_diagnostics') or {}
                block = next(iter(diagnostics.values()), None)
                if isinstance(block, dict) and 'rank' in block:
                    _logger.info(
                        f"    Sampling: rank {block['rank']} of {block['n']} "
                        f"({block['n_null']} null direction(s)), seed {block['seed']}"
                    )
                    error = block.get('realised_covariance_error')
                    if error is not None and np.isfinite(error):
                        _logger.info(
                            f"      realised covariance error on the retained "
                            f"subspace: {error:.3e}"
                        )
                    if block['n_null']:
                        _logger.info(
                            f"      -> null directions were drawn in, not truncated "
                            f"(null_tol=None)"
                        )
                else:
                    _logger.info("    Sampling Quality: [INFO] Not available (diagnostics disabled)")

                # ACE generation information
                ace_info = data['ace_generation']
                if ace_info['enabled']:
                    _logger.info("    ACE generation: ENABLED")
                    if ace_info['temperatures']:
                        temps_str = ', '.join(str(t).rstrip('0').rstrip('.') if '.' in str(t) else str(t) for t in ace_info['temperatures'])
                        _logger.info(f"      Temperatures: {temps_str}")

                    if xsdir_file:
                        _logger.info(f"      XSDIR files: Generated from master file {os.path.basename(xsdir_file)}")
                else:
                    _logger.info("    ACE generation: DISABLED")

                # Warnings if any
                if data['warnings']:
                    total_warnings_count += len(data['warnings'])
                    _logger.info("    Warnings:")
                    for warning in data['warnings']:
                        _logger.info(f"      {warning}")

        # Report files that failed processing
        if failed_files_details:
            _logger.info("  FAILED ENDF FILES:")

            for file_key, reason in failed_files_details.items():
                _logger.info(f"  File: {file_key}")
                _logger.info(f"    Reason: {reason}")

    # Metric lines
    processed_count = len(successfully_processed) if 'successfully_processed' in locals() else processed_files
    failed_count = len(failed_files_details)
    _logger.info(f">> endf_success_count = {processed_count}")
    _logger.info(f">> endf_fail_count = {failed_count}")
    if master_file:
        _logger.info(f">> master_file = {os.path.basename(master_file)}")
    if generate_ace:
        ace_total = sum(
            sd['ace_generation'].get('successful_samples', 0)
            for sd in summary_data.values()
        )
        _logger.info(f">> ace_generated = {ace_total}")
        ace_failed = sum(
            sd['ace_generation'].get('failed_samples', 0)
            for sd in summary_data.values()
        )
        if ace_failed > 0:
            _logger.info(f">> ace_failed = {ace_failed}")

    _logger.info("#== END SUMMARY ========================================================")

    # Aggregated warnings section
    _logger.info(f"#== WARNINGS ============================================================")
    _logger.info(f">> total_warnings = {total_warnings_count}")
    if total_warnings_count == 0:
        _logger.info("  No warnings.")
    _logger.info(f"#== END WARNINGS ========================================================")

    # Console: Final summary
    _logger.info(f"[INFO] [ENDF] ENDF perturbation job completed", console=True)
    _logger.info(f"[INFO] [ENDF] Processed: {processed_count} file(s)", console=True)
    if failed_count > 0:
        _logger.info(f"[WARN] [ENDF] Failed: {failed_count} file(s)", console=True)
    _logger.info(f"[INFO] [ENDF] Detailed log saved to: {log_file}", console=True)

    if master_file:
        _logger.info(f"[INFO] [ENDF] Master matrix file: {os.path.basename(master_file)}", console=True)

    if generate_ace:
        _logger.info(f"[INFO] [ENDF] ACE generation: ENABLED", console=True)
        if xsdir_file:
            _logger.info(f"[INFO] [ENDF] XSDIR files generated in: xsdir/ subdirectory", console=True)


def apply_perturbation_factors_to_endf(
    endf,
    sample: np.ndarray,
    sample_index: int,
    energy_grids: Dict[Tuple[int, int, int], List[float]],
    param_mapping: List[Tuple[int, int, int, int]],
    verbose: bool = True,
    enforce_positivity: bool = False,
    positivity_check_points: int = 101,
):
    """
    Apply perturbation factors to ENDF MF4 angular distribution data.
    
    Parameters
    ----------
    endf : ENDF
        ENDF object with parsed data
    sample : np.ndarray
        Array of perturbation factors
    sample_index : int
        Index of this sample
    energy_grids : Dict[Tuple[int, int, int], List[float]]
        Energy grids for each parameter combination
    param_mapping : List[Tuple[int, int, int, int]]
        Mapping of sample indices to (isotope, mt, l, energy_bin) parameters
    verbose : bool
        Whether to log perturbation details
        
    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of perturbed parameter combinations
    """
    if verbose and _get_logger():
        _get_logger().info(f"  [INFO] [ENDF] Applying perturbations to sample {sample_index + 1}")
    
    perturbed_params = []
    positivity_events: List[Tuple[int, float, float, float]] = []

    # Get MF4 data
    mf4 = endf.get_file(4)
    if mf4 is None:
        if _get_logger():
            _get_logger().warning("  [WARN] [ENDF] No MF4 data found in ENDF file")
        return perturbed_params, positivity_events

    # Apply perturbations to each MT section
    for mt_number, mt_data in mf4.sections.items():
        # Check for both MF4MTLegendre and MF4MTMixed (both have Legendre coefficients)
        if isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
            # Apply perturbations to Legendre coefficients
            events = _apply_factors_to_mf4_legendre(
                mt_data, sample, param_mapping, energy_grids, verbose,
                enforce_positivity=enforce_positivity,
                positivity_check_points=positivity_check_points,
            )
            positivity_events.extend(events)
            # Add all perturbed parameters for this MT
            for isotope, mt, l_coeff, energy_bin in param_mapping:
                if mt == mt_number:
                    perturbed_params.append((isotope, mt, l_coeff, energy_bin))

    return perturbed_params, positivity_events


def _apply_factors_to_mf4_legendre(
    mt_data: Union[MF4MTLegendre, MF4MTMixed],
    factors: np.ndarray,
    param_mapping: List[Tuple[int, int, int, int]],
    energy_grids: Dict[Tuple[int, int, int], List[float]],
    verbose: bool = True,
    enforce_positivity: bool = False,
    positivity_check_points: int = 101,
    mf3_magnitude_sink: Optional[Dict[Tuple[int, int, int], float]] = None,
) -> List[Tuple[int, float, float, float]]:
    """
    Apply perturbation factors to MF4 Legendre coefficient data with proper discontinuity handling.
    
    This function implements the ENDF-6 standard for representing discontinuities by duplicating
    energy points at bin boundaries to encode proper jumps in the angular distributions.
    
    Uses tolerant boundary checking with np.isclose to avoid double-scaling at nearly-identical
    energy values. Union energy grids ensure consistent alignment across all covariance matrices.
    
    Works with both MF4MTLegendre (LTT=1) and MF4MTMixed (LTT=3) types.
    
    Parameters
    ----------
    mt_data : Union[MF4MTLegendre, MF4MTMixed]
        MF4 MT section with Legendre coefficients
    factors : np.ndarray
        Perturbation factors
    param_mapping : List[Tuple[int, int, int, int]]
        Mapping of factor indices to parameters: (isotope, mt, l_coeff, energy_bin)
    energy_grids : Dict[Tuple[int, int, int], List[float]]
        Union energy grids for each parameter combination
    verbose : bool
        Whether to log details
    mf3_magnitude_sink : dict, optional
        Dormant hook for a future joint MF3+MF4 (TMC) path.  In MF4 the
        isotropic term a_0 is identically 1 — the cross-section *magnitude*
        lives in MF3 sigma(E), not here — so an L=0 (magnitude) perturbation
        factor cannot be applied to the Legendre coefficients.  When this dict
        is provided, any L=0 entry in ``param_mapping`` is recorded as
        ``{(isotope, mt, energy_bin): factor}`` for the caller to apply to the
        MF3 cross section; when ``None`` (all current callers) L=0 entries are
        skipped.  L>=1 (shape) factors are applied to MF4 exactly as before, so
        this changes nothing for existing MF34-only callers (whose
        ``param_mapping`` never contains L=0).

    Notes
    -----
    Key improvements implemented:
    - Uses union energy grids for consistent parameter mapping
    - Tolerant boundary detection with np.isclose (atol=1e-10)
    - Prevents double-scaling at nearly-identical energies
    - Ensures discontinuities are inserted exactly once per boundary
    """
    # Get the current coefficients
    current_coeffs = mt_data.legendre_coefficients
    energies = mt_data.legendre_energies
    
    if not current_coeffs or not energies:
        if verbose and _get_logger():
            _get_logger().warning(f"  [WARN] [ENDF] MT{mt_data.number}: No Legendre coefficients found")
        return
    
    # Step 0: Make baseline copies before any scaling
    E0 = mt_data._energies[:]  # Original energy grid
    A0 = [c[:] for c in mt_data._legendre_coeffs]  # Original coefficients (deep copy)
    
    if verbose and _get_logger():
        _get_logger().debug(f"  [INFO] [ENDF] MT{mt_data.number}: Baseline: {len(E0)} energy points, max coeffs per point: {max(len(c) for c in A0) if A0 else 0}")
    
    # Step 1: Gather boundaries from all energy grids for this MT
    boundaries = set()
    
    for factor_idx, (isotope, mt, l_coeff, energy_bin) in enumerate(param_mapping):
        if mt != mt_data.number:
            continue
            
        triplet = (isotope, mt, l_coeff)
        energy_grid = energy_grids.get(triplet, [])
        
        if len(energy_grid) < 2:
            continue
            
        # Add internal bin boundaries (not the first and last)
        for i in range(1, len(energy_grid) - 1):
            boundary = energy_grid[i]
            # Only include if within MF4 energy span
            if E0[0] <= boundary <= E0[-1]:
                boundaries.add(boundary)
    
    boundaries = np.asarray(sorted(boundaries), dtype=float)
    
    def _on_boundary(x, atol=1e-10):
        """Check if energy x is on any boundary within tolerance."""
        return np.any(np.isclose(boundaries, x, rtol=0.0, atol=atol))
    
    if verbose and _get_logger():
        _get_logger().debug(f"  [INFO] [ENDF] MT{mt_data.number}: Found {len(boundaries)} bin boundaries: {boundaries}")
    
    # Step 2: Scale interior points (not at boundaries)
    applied_count = 0
    
    for factor_idx, (isotope, mt, l_coeff, energy_bin) in enumerate(param_mapping):
        if mt != mt_data.number:
            continue
        
        if factor_idx >= len(factors):
            continue
        
        factor = factors[factor_idx]
        
        # Get the energy grid for this parameter triplet
        triplet = (isotope, mt, l_coeff)
        energy_grid = energy_grids.get(triplet, [])
        
        if len(energy_grid) < 2:
            continue
        
        # Get energy bounds for this bin
        if energy_bin >= len(energy_grid) - 1:
            continue
            
        energy_low = energy_grid[energy_bin]
        energy_high = energy_grid[energy_bin + 1]

        # L=0 is the cross-section magnitude (a_0 ≡ 1 in MF4; the magnitude
        # lives in MF3 sigma(E)).  It must never scale the Legendre shape
        # coefficients.  Route it once per bin to the optional MF3 sink for a
        # future joint MF3+MF4 (TMC) path; skip otherwise (current behavior).
        if l_coeff == 0:
            if mf3_magnitude_sink is not None:
                mf3_magnitude_sink[(isotope, mt, energy_bin)] = float(factor)
                if verbose and _get_logger():
                    _get_logger().debug(
                        f"  [INFO] [ENDF] MAGNITUDE MT{mt} L0 bin {energy_bin}: "
                        f"factor {factor:.6f} routed to MF3 sink"
                    )
            continue

        # Apply factor to coefficients strictly inside this bin (not at boundaries)
        for energy_idx, energy in enumerate(mt_data._energies):
            if energy_low <= energy < energy_high and not _on_boundary(energy):
                # Check if this L coefficient exists at this energy
                coeff_index = l_coeff - 1  # Convert L=1,2,3... to 0-indexed
                if (coeff_index >= 0 and energy_idx < len(mt_data._legendre_coeffs) and
                    coeff_index < len(mt_data._legendre_coeffs[energy_idx])):
                    
                    old_value = mt_data._legendre_coeffs[energy_idx][coeff_index]
                    mt_data._legendre_coeffs[energy_idx][coeff_index] *= factor
                    applied_count += 1
                    
                    if verbose and _get_logger():
                        _get_logger().debug(
                            f"  [INFO] [ENDF] INTERIOR MT{mt} L{l_coeff} at {energy:.3e} MeV: "
                            f"factor {factor:.6f}, {old_value:.3e} -> {mt_data._legendre_coeffs[energy_idx][coeff_index]:.3e}"
                        )
    
    # Step 3 & 4: Handle discontinuities at boundaries
    insertions_made = 0
    
    for boundary_energy in boundaries:
        if verbose and _get_logger():
            _get_logger().debug(f"  [INFO] [ENDF] BOUNDARY: Processing boundary at {boundary_energy:.3e} MeV")
        
        # Interpolate baseline coefficients at this boundary
        baseline_coeffs = _interpolate_legendre_coefficients(boundary_energy, E0, A0)
        
        if baseline_coeffs is None:
            if verbose and _get_logger():
                _get_logger().warning(f"  [WARN] [ENDF] BOUNDARY: Could not interpolate at {boundary_energy:.3e} MeV")
            continue
        
        # Find the factors for lower and upper bins at this boundary
        lower_factors = {}  # l_coeff -> factor for the bin below this boundary
        upper_factors = {}  # l_coeff -> factor for the bin above this boundary
        
        for factor_idx, (isotope, mt, l_coeff, energy_bin) in enumerate(param_mapping):
            if mt != mt_data.number:
                continue

            # L=0 is magnitude (MF3), not an MF4 shape order — it creates no
            # angular-distribution discontinuity, so it must not enter the
            # boundary factor sets (else an L=0-only boundary would insert a
            # redundant energy point). It is routed to the MF3 sink in the
            # interior pass above.
            if l_coeff == 0:
                continue

            triplet = (isotope, mt, l_coeff)
            energy_grid = energy_grids.get(triplet, [])

            if len(energy_grid) < 2 or energy_bin >= len(energy_grid) - 1:
                continue

            factor = factors[factor_idx]
            energy_low = energy_grid[energy_bin]
            energy_high = energy_grid[energy_bin + 1]
            
            # Check which side of the boundary this bin is on
            if np.isclose(energy_high, boundary_energy, rtol=0.0, atol=1e-10):  # This bin ends at the boundary
                lower_factors[l_coeff] = factor  # This factor applies to the lower side
            if np.isclose(energy_low, boundary_energy, rtol=0.0, atol=1e-10):  # This bin starts at the boundary  
                upper_factors[l_coeff] = factor  # This factor applies to the upper side
        
        # If no factors found, skip this boundary
        if not lower_factors and not upper_factors:
            continue
        
        # Apply factors to create lower and upper coefficient vectors
        coeffs_minus = baseline_coeffs[:]
        coeffs_plus = baseline_coeffs[:]
        
        # L=0 (coeff_index == -1) is intentionally excluded by the guards below:
        # magnitude creates no MF4 shape discontinuity and is handled in the
        # interior pass via mf3_magnitude_sink.
        for l_coeff, factor in lower_factors.items():
            coeff_index = l_coeff - 1
            if 0 <= coeff_index < len(coeffs_minus):
                coeffs_minus[coeff_index] *= factor

        for l_coeff, factor in upper_factors.items():
            coeff_index = l_coeff - 1
            if 0 <= coeff_index < len(coeffs_plus):
                coeffs_plus[coeff_index] *= factor
        
        # Insert or replace the boundary points
        _insert_boundary_discontinuity(
            mt_data, boundary_energy, coeffs_minus, coeffs_plus, verbose
        )
        insertions_made += 1
    
    # Step 5: Update bookkeeping for ENDF output
    mt_data._ne = len(mt_data._energies)
    if mt_data._ne > 0:
        mt_data._interpolation = [(mt_data._ne, 2)]  # One region, linear-linear
        mt_data._nr = 1

    # For MF4MTMixed, also update Legendre-specific attributes
    if isinstance(mt_data, MF4MTMixed):
        mt_data._ne1 = mt_data._ne  # Number of Legendre energy points
        # Note: _ne2 and _nr_tab are for the tabulated part and should remain unchanged

    # Step 6: Positivity enforcement.
    # f(mu) = (1/2) sum_l (2l+1) a_l P_l(mu), with a_0 = 1 implicit.
    # Pin the high-l tail (outside the perturbed band) to baseline so the
    # projection only redistributes within the perturbed orders.
    positivity_events: List[Tuple[int, float, float, float]] = []
    if enforce_positivity and mt_data._ne > 0:
        from kika.sampling.mf4_positivity import (
            check_mf4_positivity,
            project_mf4_to_positive,
        )

        perturbed_l = {
            l_coeff for (_iso, mt, l_coeff, _bin) in param_mapping
            if mt == mt_data.number
        }
        if perturbed_l:
            max_perturbed_l = max(perturbed_l)
        else:
            max_perturbed_l = 0

        for energy_idx, energy in enumerate(mt_data._energies):
            a_endf_tail = np.asarray(
                mt_data._legendre_coeffs[energy_idx], dtype=float
            )
            if a_endf_tail.size == 0:
                continue
            a_full = np.concatenate(([1.0], a_endf_tail))
            is_pos, sigma_min = check_mf4_positivity(
                a_full, n_points=positivity_check_points
            )
            if is_pos:
                continue

            # Pin a_l beyond the perturbed band to their current (baseline) values.
            frozen = {
                idx: float(a_full[idx])
                for idx in range(max_perturbed_l + 1, len(a_full))
            }
            a_projected = project_mf4_to_positive(
                a_full,
                n_points=positivity_check_points,
                frozen_indices=frozen,
            )
            delta = np.abs(a_projected - a_full)
            max_delta = float(delta.max()) if delta.size else 0.0
            positivity_events.append(
                (mt_data.number, float(energy), float(sigma_min), max_delta)
            )

            # Write the projected a_l back (drop a_0).
            mt_data._legendre_coeffs[energy_idx] = list(a_projected[1:])

    if verbose and _get_logger():
        _get_logger().info(
            f"  [INFO] [ENDF] MT{mt_data.number}: Applied {applied_count} interior factors and "
            f"{insertions_made} boundary discontinuities. Final: {mt_data._ne} energy points"
        )

    return positivity_events


def _interpolate_legendre_coefficients(
    energy: float, 
    energy_grid: List[float], 
    coeff_grid: List[List[float]]
) -> Optional[List[float]]:
    """
    Interpolate Legendre coefficients at a given energy using linear-linear interpolation.
    
    Parameters
    ----------
    energy : float
        Energy at which to interpolate
    energy_grid : List[float]
        Original energy grid
    coeff_grid : List[List[float]]
        Coefficient arrays for each energy point
        
    Returns
    -------
    Optional[List[float]]
        Interpolated coefficients, or None if interpolation fails
    """
    if len(energy_grid) != len(coeff_grid) or len(energy_grid) < 2:
        return None
    
    # Find bracketing energies
    if energy <= energy_grid[0]:
        return coeff_grid[0][:]  # Return copy of first point
    if energy >= energy_grid[-1]:
        return coeff_grid[-1][:]  # Return copy of last point
    
    # Find the interval containing this energy
    for i in range(len(energy_grid) - 1):
        if energy_grid[i] <= energy <= energy_grid[i + 1]:
            e1, e2 = energy_grid[i], energy_grid[i + 1]
            c1, c2 = coeff_grid[i], coeff_grid[i + 1]
            
            # Linear interpolation factor
            if abs(e2 - e1) < 1e-15:  # Avoid division by zero
                return c1[:]
            
            t = (energy - e1) / (e2 - e1)
            
            # Interpolate each coefficient
            max_coeffs = max(len(c1), len(c2))
            result = []
            
            for j in range(max_coeffs):
                v1 = c1[j] if j < len(c1) else 0.0
                v2 = c2[j] if j < len(c2) else 0.0
                result.append(v1 + t * (v2 - v1))
            
            return result
    
    return None


def _insert_boundary_discontinuity(
    mt_data: Union[MF4MTLegendre, MF4MTMixed],
    boundary_energy: float,
    coeffs_minus: List[float],
    coeffs_plus: List[float],
    verbose: bool = True
):
    """
    Insert a discontinuity at a boundary energy by duplicating the energy point.
    
    Works with both MF4MTLegendre and MF4MTMixed types.
    
    Parameters
    ----------
    mt_data : Union[MF4MTLegendre, MF4MTMixed]
        MF4 section to modify
    boundary_energy : float
        Energy at which to insert discontinuity
    coeffs_minus : List[float]
        Coefficients for the "minus" side (lower bin)
    coeffs_plus : List[float]
        Coefficients for the "plus" side (upper bin)
    verbose : bool
        Whether to log details
    """
    # Find if this energy already exists
    existing_indices = [i for i, e in enumerate(mt_data._energies) if abs(e - boundary_energy) < 1e-10]
    
    if existing_indices:
        # Replace existing point with two consecutive points
        idx = existing_indices[0]
        
        if verbose and _get_logger():
            _get_logger().debug(f"  [INFO] [ENDF] BOUNDARY: Replacing existing point at {boundary_energy:.3e} MeV")
        
        # Replace the existing point with minus side
        mt_data._energies[idx] = boundary_energy
        mt_data._legendre_coeffs[idx] = coeffs_minus[:]
        
        # Insert plus side right after
        mt_data._energies.insert(idx + 1, boundary_energy)
        mt_data._legendre_coeffs.insert(idx + 1, coeffs_plus[:])
        
    else:
        # Find insertion point to maintain sorted order
        insert_idx = 0
        for i, e in enumerate(mt_data._energies):
            if e < boundary_energy:
                insert_idx = i + 1
            else:
                break
        
        if verbose and _get_logger():
            _get_logger().debug(f"  [INFO] [ENDF] BOUNDARY: Inserting new points at {boundary_energy:.3e} MeV at index {insert_idx}")
        
        # Insert minus side first
        mt_data._energies.insert(insert_idx, boundary_energy)
        mt_data._legendre_coeffs.insert(insert_idx, coeffs_minus[:])
        
        # Insert plus side right after
        mt_data._energies.insert(insert_idx + 1, boundary_energy)
        mt_data._legendre_coeffs.insert(insert_idx + 1, coeffs_plus[:])



def _filter_mf34_covariance(
    mf34_cov: LegendreCovariance, 
    mt_list: List[int], 
    legendre_coeffs: List[int]
) -> LegendreCovariance:
    """
    Filter MF34 covariance data by MT reactions and Legendre coefficients.
    
    Parameters
    ----------
    mf34_cov : LegendreCovariance
        Original MF34 covariance data
    mt_list : List[int]
        List of MT numbers to include (empty means all)
    legendre_coeffs : List[int]
        List of Legendre coefficient indices to include
        
    Returns
    -------
    LegendreCovariance
        Filtered covariance data
    """
    if _get_logger():
        _get_logger().info(f"  [INFO] [ENDF] Filtering MF34 covariance data: MT={mt_list}, L={legendre_coeffs}")
        if not legendre_coeffs or legendre_coeffs == [-1]:
            _get_logger().info(f"  [INFO] [ENDF] Using all available Legendre coefficients: {sorted(mf34_cov.legendre_indices)}")
    
    filtered_cov = LegendreCovariance()
    
    # If mt_list is empty, use all available MTs
    available_mts = mt_list if mt_list else list(mf34_cov.reactions)
    
    # If legendre_coeffs is empty or contains only -1, use all available L coefficients
    if not legendre_coeffs or legendre_coeffs == [-1]:
        available_ls = list(mf34_cov.legendre_indices)
        if _get_logger():
            _get_logger().info(f"  [INFO] [ENDF] Available L coefficients: {available_ls}")
    else:
        available_ls = legendre_coeffs
    
    for i in range(mf34_cov.num_matrices):
        mt_row = mf34_cov.reaction_rows[i]
        mt_col = mf34_cov.reaction_cols[i]
        l_row = mf34_cov.l_rows[i]
        l_col = mf34_cov.l_cols[i]
        
        # Check if this matrix should be included
        include_mt = (mt_row in available_mts and mt_col in available_mts)
        include_l = (l_row in available_ls and l_col in available_ls)
        
        if include_mt and include_l:
            filtered_cov.add_matrix(
                isotope_row=mf34_cov.isotope_rows[i],
                reaction_row=mt_row,
                l_row=l_row,
                isotope_col=mf34_cov.isotope_cols[i],
                reaction_col=mt_col,
                l_col=l_col,
                matrix=mf34_cov.matrices[i],
                energy_grid=mf34_cov.energy_grids[i],
                is_relative=mf34_cov.is_relative[i],
                frame=mf34_cov.frame[i]
            )
    
    if _get_logger():
        _get_logger().info(f"  [INFO] [ENDF] Filtered to {filtered_cov.num_matrices} matrices")
    
    return filtered_cov


def _mask_factors_by_energy_range(
    factors: np.ndarray,
    param_mapping: List[Tuple[int, int, int, int]],
    energy_grids: Dict[Tuple[int, int, int], List[float]],
    energy_ranges: List[Tuple[float, float]],
    space: str = "linear",
) -> np.ndarray:
    """
    Mask perturbation factors for energy bins outside the requested ranges.

    Bins that do not overlap any of the specified energy ranges are set to
    the neutral value (1.0 for linear space, 0.0 for log space).

    Parameters
    ----------
    factors : np.ndarray
        Perturbation factors, shape (n_samples, n_parameters)
    param_mapping : List[Tuple[int, int, int, int]]
        List of (isotope, mt, l_coeff, energy_bin) tuples
    energy_grids : Dict[Tuple[int, int, int], List[float]]
        Energy grids (eV) for each (isotope, mt, l) triplet
    energy_ranges : List[Tuple[float, float]]
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

    for col_idx, (iso, mt, l, energy_bin) in enumerate(param_mapping):
        triplet = (iso, mt, l)
        grid = energy_grids.get(triplet, [])
        if energy_bin >= len(grid) - 1:
            # Padding bin — leave as neutral
            masked[:, col_idx] = neutral
            continue

        bin_low = grid[energy_bin]
        bin_high = grid[energy_bin + 1]

        # Check if this bin overlaps any requested range
        overlaps = any(bin_low < emax and bin_high > emin
                       for emin, emax in energy_ranges)

        if not overlaps:
            masked[:, col_idx] = neutral

    return masked


def _create_parameter_mapping(mf34_cov):
    """
    Create parameter mapping using union energy grids for consistent alignment.
    
    This function replaces the original approach that used individual matrix energy grids
    with a union-based approach that ensures all covariance matrices are properly aligned.
    
    Parameters
    ----------
    mf34_cov : LegendreCovariance
        MF34 covariance matrix object with union grids support
        
    Returns
    -------
    Tuple[List[Tuple[int, int, int, int]], Dict[Tuple[int, int, int], List[float]]]
        - param_mapping: List of (isotope, mt, l_coeff, energy_bin) tuples
        - energy_grids: Dictionary mapping triplets to union energy grids
        
    Notes
    -----
    Key improvements:
    - Uses union grids instead of raw matrix grids
    - Ensures consistent max_G across all parameter triplets  
    - Proper alignment for covariance matrix construction
    """
    param_triplets = mf34_cov._get_param_triplets()
    union_grids = mf34_cov.get_union_energy_grids()
    energy_grids = {t: union_grids[t].tolist() for t in param_triplets}
    max_G = max(len(g)-1 for g in energy_grids.values()) if energy_grids else 0

    param_mapping = []
    for triplet in param_triplets:
        iso, mt, L = triplet
        for energy_bin in range(max_G):
            param_mapping.append((iso, mt, L, energy_bin))
    return param_mapping, energy_grids


def _write_sample_summary(
    sample: np.ndarray,
    sample_index: int,
    energy_grids: Dict[Tuple[int, int, int], List[float]],
    param_mapping: List[Tuple[int, int, int, int]],
    sample_dir: str,
    base: str,
):
    """
    Write a summary file for a single sample.
    
    Parameters
    ----------
    sample : np.ndarray
        Perturbation factors for this sample
    sample_index : int
        Sample index
    energy_grids : Dict[Tuple[int, int, int], List[float]]
        Energy grids for each parameter
    param_mapping : List[Tuple[int, int, int, int]]
        Parameter mapping with energy bins
    sample_dir : str
        Sample output directory
    base : str
        Base filename
    """
    sample_str = f"{sample_index+1:04d}"
    summary_file = os.path.join(sample_dir, f"{base}_{sample_str}_summary.txt")
    
    with open(summary_file, 'w') as f:        
        for i, factor in enumerate(sample):
            if i < len(param_mapping):
                isotope, mt, l_coeff, energy_bin = param_mapping[i]
                
                # Get energy bounds for this parameter
                triplet = (isotope, mt, l_coeff)
                energy_grid = energy_grids.get(triplet, [])
                
                # Skip padding bins - only write actual energy bins
                if energy_bin < len(energy_grid) - 1:
                    energy_low = energy_grid[energy_bin]
                    energy_high = energy_grid[energy_bin + 1]
                    
                    # Write with proper alignment for negative numbers
                    # Format: MT(6) L_coeff(6) Energy_Low(15) Energy_High(15) Factor(15)
                    f.write(f"{mt:6d}{l_coeff:6d}{energy_low:15.6e}{energy_high:15.6e}{factor:15.6e}\n")


def _write_master_perturbation_file(
    all_factors: List[np.ndarray],
    all_param_mappings: List[List[Tuple[int, int, int, int]]],
    all_energy_grids: List[Dict[Tuple[int, int, int], List[float]]],
    output_dir: str,
    timestamp: str,
    verbose: bool = True
):
    """
    Write master perturbation factors file in Parquet format.
    
    The format matches ACE perturbation: each row is a sample, each column is a parameter.
    Column names follow the format: Fe56_MT2_L1_E0-E1
    
    Parameters
    ----------
    all_factors : List[np.ndarray]
        List of factor arrays for each ENDF file
    all_param_mappings : List[List[Tuple[int, int, int, int]]]
        List of parameter mappings for each ENDF file with energy bins
    all_energy_grids : List[Dict[Tuple[int, int, int], List[float]]]
        List of energy grids for each ENDF file
    output_dir : str
        Output directory
    timestamp : str
        Timestamp for filename
    verbose : bool
        Whether to log progress
    """
    if not all_factors:
        if verbose and _get_logger():
            _get_logger().info("  [INFO] [ENDF] No factors to write to master file")
        return
    
    # Determine the maximum number of samples across all files
    max_samples = max(factors.shape[0] for factors in all_factors)
    
    # Initialize the master DataFrame with Sample_ID column
    master_data = {'Sample_ID': np.arange(1, max_samples + 1, dtype='int32')}
    
    # Process each ENDF file
    for file_idx, (factors, param_mapping, energy_grids) in enumerate(zip(all_factors, all_param_mappings, all_energy_grids)):
        # Convert factors to float32 for consistency
        factors = factors.astype(np.float32)
        
        # Create columns for each parameter in this file
        for param_idx, (isotope, mt, l_coeff, energy_bin) in enumerate(param_mapping):
            # Get the energy grid for this (isotope, mt, l_coeff) combination
            energy_grid_key = (isotope, mt, l_coeff)
            if energy_grid_key in energy_grids:
                energy_grid = energy_grids[energy_grid_key]
                
                # Skip padding bins - only process actual energy bins
                if energy_bin >= len(energy_grid) - 1:
                    continue
                    
                # Column name format: Fe56_MT2_L1_1.000e-05_1.234e-02 (actual energy boundaries)
                energy_group_name = _format_energy_group_name(energy_grid, energy_bin)
            else:
                # Fallback to old format if energy grid not found
                energy_group_name = f"E{energy_bin}-E{energy_bin+1}"
            
            # Get element symbol from ZAID
            symbol = zaid_to_symbol(isotope)
            col_name = f"{symbol}_MT{mt}_L{l_coeff}_{energy_group_name}"
            
            # Create column data, padding with NaN if this file has fewer samples
            column_data = np.full(max_samples, np.nan, dtype=np.float32)
            if factors.shape[0] > 0:
                column_data[:factors.shape[0]] = factors[:, param_idx]
            
            master_data[col_name] = column_data
    
    # Create DataFrame and save as Parquet
    df = pd.DataFrame(master_data)
    master_file = os.path.join(output_dir, f'endf_perturbation_factors_{timestamp}.parquet')
    df.to_parquet(master_file, index=False, compression='zstd', compression_level=3)
    
    if verbose and _get_logger():
        _get_logger().info(f"  [INFO] [ENDF] Master perturbation factors saved to: {master_file}")
        _get_logger().info(f"  [INFO] [ENDF] DataFrame shape: {df.shape}")
        _get_logger().info(f"  [INFO] [ENDF] Columns: {len(df.columns)} (including Sample_ID)")
        _get_logger().info(f"  [INFO] [ENDF] Samples: {max_samples}")
    
    return master_file
