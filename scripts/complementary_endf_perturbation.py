"""
Complementary ENDF perturbation script.

Applies MF34 covariance-based perturbations to the energy range that is
*complementary* to the EXFOR-fitted region processed by Pipeline 1
(``exfor_to_endf_sampling_v2.py``).

Pipeline 1 produces N ENDF sample files where MF4 Legendre coefficients in
the experimental energy range are sampled from EXFOR data.  Outside that range
the original evaluation coefficients are left untouched.  This script fills
the gap by applying evaluation-based (MF34) perturbations to *every energy
bin outside* the EXFOR range, for each Pipeline 1 sample independently.

The result: ENDF samples with EXFOR-based uncertainty in the experimental
range AND evaluation-based (MF34) uncertainty everywhere else.
"""

from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool
from datetime import datetime
import glob
import os
import re
import sys
import numpy as np

from kika.sampling.generators import generate_endf_samples
from kika.endf.parsers.parse_endf import parse_endf_file
from kika.endf.writers.endf_writer import ENDFWriter
from kika.sampling.endf_perturbation import (
    load_mf34_covariance,
    _filter_mf34_covariance,
    _create_parameter_mapping,
    _mask_factors_by_energy_range,
    apply_perturbation_factors_to_endf,
    _write_master_perturbation_file,
    _write_sample_summary,
    _process_njoy_for_sample,
)
from kika.sampling.utils import DualLogger, _get_logger, _set_logger


# ============================================================================
#  CONFIGURATION CONSTANTS
# ============================================================================

# Pipeline 1 output directory (contains endf/XXXX/*.endf)
SAMPLES_DIR = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/NEW_EXFOR_without_CAP3/"

# ENDF file with MF34 covariance for the full energy range
REFERENCE_COV_FILE = "/share_snc/snc/JuanMonleon/jeff40_with_MF4_from_jeff33/26-Fe-56g.txt"

# EXFOR energy range (eV) — bins inside this interval are left unchanged
EXFOR_ENERGY_MIN_EV = 0.847e6   # 0.847 MeV
EXFOR_ENERGY_MAX_EV = 4.0e6     # 4.0 MeV

# MF34 filter settings
MT_LIST = [2]                    # MT reactions to perturb (empty list = all)
LEGENDRE_COEFFS = [-1]          # L coefficients ([-1] = all available)

# Sampling settings
SPACE = "linear"                 # "linear" or "log"
DECOMPOSITION_METHOD = "svd"     # "svd", "cholesky", "eigen", "pca"
SAMPLING_METHOD = "random"        # "sobol", "lhs", "random"
SEED = 33                        # RNG seed (different from Pipeline 1!)

# Output
OUTPUT_DIR = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/COMBINED_SAMPLES/"

# Parallelism
N_PROCS = 24

# Dry run — if True, generate and log factors only (no ENDF files written)
DRY_RUN = False

# ACE generation (optional)
GENERATE_ACE = True
ACE_TEMPERATURES = [293.6]
ACE_NJOY_EXE = "/soft_snc/NJOY/2016.78/bin/njoy"
ACE_LIBRARY_NAME = "jeff40"
ACE_NJOY_VERSION = "NJOY 2016.78"
ACE_XSDIR_FILE = "/share_snc/snc/JuanMonleon/xsdir_MCNPy/xsdir40-irdff2"


# ============================================================================
#  Global logger
# ============================================================================

_logger = None


# ============================================================================
#  Helper functions
# ============================================================================

def _discover_pipeline1_samples(samples_dir: str) -> List[str]:
    """Find all Pipeline 1 ENDF sample files, sorted by sample number."""
    pattern = os.path.join(samples_dir, "endf", "????", "*.endf")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Pipeline 1 sample files found matching {pattern}\n"
            f"Expected directory layout: {{SAMPLES_DIR}}/endf/XXXX/*.endf"
        )
    return files


def _extract_sample_number(path: str) -> int:
    """Extract the 4-digit sample number from a Pipeline 1 path.

    Expects …/endf/XXXX/base_XXXX.endf  where XXXX is the sample number.
    Falls back to the directory name if the filename suffix doesn't match.
    """
    dirname = os.path.basename(os.path.dirname(path))
    m = re.search(r"(\d{4})", dirname)
    if m:
        return int(m.group(1))
    # fallback: try filename
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_(\d{4})$", base)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot determine sample number from path: {path}")


def _process_complementary_sample(args):
    """Process a single sample — designed for Pool.map (top-level, picklable).

    Parameters (passed as a single tuple)
    ----------
    sample_file : str
        Path to the Pipeline 1 ENDF sample file.
    sample_factors : np.ndarray
        1-D perturbation factors for this sample.
    sample_index : int
        0-based sample index.
    energy_grids : dict
        Energy grids keyed by (isotope, mt, l).
    param_mapping : list
        Parameter mapping list.
    output_dir : str
        Root output directory.
    dry_run : bool
    generate_ace : bool
    njoy_exe : str or None
    temperatures : list or None
    library_name : str or None
    njoy_version : str
    xsdir_file : str or None
    """
    (
        sample_file,
        sample_factors,
        sample_index,
        energy_grids,
        param_mapping,
        output_dir,
        dry_run,
        generate_ace,
        njoy_exe,
        temperatures,
        library_name,
        njoy_version,
        xsdir_file,
    ) = args

    logger = _get_logger()
    sample_str = f"{sample_index + 1:04d}"

    try:
        # Parse the Pipeline 1 sample
        endf = parse_endf_file(sample_file)

        # Build output path following Pipeline 1 convention:
        #   {output_dir}/endf/{sample_str}/{base}.endf
        # We keep the *original* basename (already has _XXXX suffix from P1).
        base_full = os.path.basename(sample_file)            # e.g. 26-Fe-56g_0001.endf
        base_no_ext, ext = os.path.splitext(base_full)

        sample_dir = os.path.join(output_dir, "endf", sample_str)
        os.makedirs(sample_dir, exist_ok=True)
        out_endf = os.path.join(sample_dir, base_full)

        if dry_run:
            _write_sample_summary(
                sample=sample_factors,
                sample_index=sample_index,
                energy_grids=energy_grids,
                param_mapping=param_mapping,
                sample_dir=sample_dir,
                base=base_no_ext,
            )
            return {"success": True, "dry_run": True}

        # Apply complementary perturbation factors to the parsed ENDF
        apply_perturbation_factors_to_endf(
            endf, sample_factors, sample_index,
            energy_grids, param_mapping, verbose=False,
        )

        # Write the perturbed ENDF file
        writer = ENDFWriter(sample_file)
        if 4 in endf.files:
            success = writer.replace_mf_section(endf.files[4], out_endf)
            if not success:
                msg = f"Sample {sample_str}: failed to write {out_endf}"
                if logger:
                    logger.error(msg)
                return {"success": False, "error": msg}
        else:
            # No MF4 section — just copy the file unchanged
            import shutil
            shutil.copy2(sample_file, out_endf)

        # Write factor summary
        _write_sample_summary(
            sample=sample_factors,
            sample_index=sample_index,
            energy_grids=energy_grids,
            param_mapping=param_mapping,
            sample_dir=sample_dir,
            base=base_no_ext,
        )

        # Optional ACE generation
        if generate_ace:
            njoy_result = _process_njoy_for_sample(
                out_endf=out_endf,
                sample_index=sample_index,
                njoy_exe=njoy_exe,
                temperatures=temperatures,
                library_name=library_name,
                njoy_version=njoy_version,
                output_dir=output_dir,
                xsdir_file=xsdir_file,
            )
            return njoy_result

        return {"success": True, "temperatures_processed": [], "errors": [], "warnings": []}

    except Exception as exc:
        msg = f"Sample {sample_str}: {exc}"
        if logger:
            logger.error(msg)
        return {"success": False, "error": msg}


# ============================================================================
#  Main entry point
# ============================================================================

def run_complementary_perturbation(
    samples_dir: str = SAMPLES_DIR,
    reference_cov_file: str = REFERENCE_COV_FILE,
    exfor_energy_min_ev: float = EXFOR_ENERGY_MIN_EV,
    exfor_energy_max_ev: float = EXFOR_ENERGY_MAX_EV,
    mt_list: List[int] = MT_LIST,
    legendre_coeffs: List[int] = LEGENDRE_COEFFS,
    space: str = SPACE,
    decomposition_method: str = DECOMPOSITION_METHOD,
    sampling_method: str = SAMPLING_METHOD,
    seed: int = SEED,
    output_dir: str = OUTPUT_DIR,
    n_procs: int = N_PROCS,
    dry_run: bool = DRY_RUN,
    generate_ace: bool = GENERATE_ACE,
    ace_temperatures: Optional[List[float]] = None,
    ace_njoy_exe: Optional[str] = None,
    ace_library_name: Optional[str] = None,
    ace_njoy_version: str = ACE_NJOY_VERSION,
    ace_xsdir_file: Optional[str] = None,
):
    """Apply MF34 complementary perturbations to Pipeline 1 ENDF samples.

    Parameters
    ----------
    samples_dir : str
        Pipeline 1 output root (contains ``endf/XXXX/``).
    reference_cov_file : str
        ENDF file that contains MF34 covariance for the full energy range.
    exfor_energy_min_ev, exfor_energy_max_ev : float
        Bounds (eV) of the Pipeline 1 EXFOR-fitted range.  Bins inside this
        interval keep the factors set to neutral (no MF34 perturbation).
    mt_list : list of int
        MT reactions to perturb (empty = all).
    legendre_coeffs : list of int
        L coefficients ([-1] = all available).
    space : str
        Sampling space: ``"linear"`` or ``"log"``.
    decomposition_method : str
        Matrix decomposition method.
    sampling_method : str
        Sampling method.
    seed : int
        RNG seed (should differ from Pipeline 1).
    output_dir : str
        Root directory for combined output.
    n_procs : int
        Number of parallel worker processes.
    dry_run : bool
        If True, only generate factors and summaries — no ENDF files written.
    generate_ace : bool
        Whether to run NJOY/ACE after writing perturbed ENDF files.
    ace_temperatures, ace_njoy_exe, ace_library_name, ace_njoy_version,
    ace_xsdir_file
        ACE generation parameters (required when ``generate_ace=True``).
    """
    global _logger

    # Defaults from module constants for ACE
    if ace_temperatures is None:
        ace_temperatures = ACE_TEMPERATURES
    if ace_njoy_exe is None:
        ace_njoy_exe = ACE_NJOY_EXE
    if ace_library_name is None:
        ace_library_name = ACE_LIBRARY_NAME
    if ace_xsdir_file is None:
        ace_xsdir_file = ACE_XSDIR_FILE

    # Normalise space
    if space.lower() == "lin":
        space = "linear"

    # ------------------------------------------------------------------
    #  Setup
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"complementary_perturbation_{timestamp}.log")
    _logger = DualLogger(log_file)
    _set_logger(_logger)

    separator = "=" * 80
    _logger.info(f"\n{separator}")
    _logger.info("[COMPLEMENTARY] Run Configuration")
    _logger.info(separator)
    _logger.info(f"Pipeline 1 samples directory : {samples_dir}")
    _logger.info(f"Reference MF34 covariance    : {reference_cov_file}")
    _logger.info(f"EXFOR energy range (eV)      : [{exfor_energy_min_ev:.6e}, {exfor_energy_max_ev:.6e}]")
    _logger.info(f"MT reactions                  : {mt_list}")
    _logger.info(f"Legendre coefficients         : {legendre_coeffs}")
    _logger.info(f"Sampling space                : {space}")
    _logger.info(f"Decomposition method          : {decomposition_method}")
    _logger.info(f"Sampling method               : {sampling_method}")
    _logger.info(f"RNG seed                      : {seed}")
    _logger.info(f"Output directory              : {os.path.abspath(output_dir)}")
    _logger.info(f"Parallel processes            : {n_procs}")
    _logger.info(f"Dry run                       : {dry_run}")
    _logger.info(f"Generate ACE                  : {generate_ace}")
    if generate_ace:
        _logger.info(f"  NJOY executable             : {ace_njoy_exe}")
        _logger.info(f"  Temperatures                : {ace_temperatures}")
        _logger.info(f"  Library name                : {ace_library_name}")
        _logger.info(f"  NJOY version                : {ace_njoy_version}")
        _logger.info(f"  XSDIR file                  : {ace_xsdir_file}")
    _logger.info(separator)

    print(f"[INFO] Starting complementary ENDF perturbation")
    print(f"[INFO] Log file: {log_file}")
    print(f"[INFO] Output directory: {os.path.abspath(output_dir)}")

    # Validate ACE parameters early
    if generate_ace:
        if ace_njoy_exe is None:
            raise ValueError("ace_njoy_exe must be provided when generate_ace=True")
        if ace_temperatures is None or len(ace_temperatures) == 0:
            raise ValueError("ace_temperatures must be provided when generate_ace=True")
        if ace_library_name is None:
            raise ValueError("ace_library_name must be provided when generate_ace=True")
        if not os.path.exists(ace_njoy_exe):
            raise FileNotFoundError(f"NJOY executable not found: {ace_njoy_exe}")
        ace_temperatures = [float(t) for t in ace_temperatures]

    # ------------------------------------------------------------------
    #  Step 1: Discover Pipeline 1 samples
    # ------------------------------------------------------------------
    sample_files = _discover_pipeline1_samples(samples_dir)
    n_samples = len(sample_files)
    _logger.info(f"\n[COMPLEMENTARY] Found {n_samples} Pipeline 1 sample file(s)")
    _logger.info(f"  First: {sample_files[0]}")
    _logger.info(f"  Last : {sample_files[-1]}")
    print(f"[INFO] Found {n_samples} Pipeline 1 sample files")

    # ------------------------------------------------------------------
    #  Step 2: Load and filter reference covariance
    # ------------------------------------------------------------------
    _logger.info(f"\n[COMPLEMENTARY] Loading reference MF34 covariance ...")
    mf34_cov = load_mf34_covariance(reference_cov_file)
    if mf34_cov is None:
        raise RuntimeError(
            f"Failed to load MF34 covariance from {reference_cov_file}. "
            "Ensure the file contains a valid MF34 section."
        )

    filtered_cov = _filter_mf34_covariance(mf34_cov, mt_list, legendre_coeffs)
    if filtered_cov.num_matrices == 0:
        available_mts = sorted(mf34_cov.reactions)
        available_ls = sorted(mf34_cov.legendre_indices)
        raise RuntimeError(
            f"No covariance data for requested MTs {mt_list} and L {legendre_coeffs}.\n"
            f"Available MTs: {available_mts}\n"
            f"Available L  : {available_ls}"
        )

    param_mapping, energy_grids = _create_parameter_mapping(filtered_cov)
    _logger.info(f"[COMPLEMENTARY] Parameter mapping: {len(param_mapping)} entries "
                 f"({len(set((p[0],p[1],p[2]) for p in param_mapping))} triplets)")

    # ------------------------------------------------------------------
    #  Step 3: Generate perturbation factors
    # ------------------------------------------------------------------
    _logger.info(f"\n[COMPLEMENTARY] Generating {n_samples} perturbation samples ...")
    factors, _, diagnostics = generate_endf_samples(
        filtered_cov,
        n_samples,
        space=space,
        decomposition_method=decomposition_method,
        sampling_method=sampling_method,
        seed=seed,
        mt_numbers=mt_list,
        verbose=True,
    )
    _logger.info(f"[COMPLEMENTARY] Factors shape: {factors.shape}")

    # ------------------------------------------------------------------
    #  Step 4: Mask the EXFOR range (keep only complementary energies)
    # ------------------------------------------------------------------
    complementary_ranges: List[Tuple[float, float]] = [
        (0.0, exfor_energy_min_ev),
        (exfor_energy_max_ev, 200e6),
    ]
    factors = _mask_factors_by_energy_range(
        factors, param_mapping, energy_grids, complementary_ranges, space,
    )

    # Log masking statistics
    neutral = 1.0 if space == "linear" else 0.0
    n_neutral = int(np.sum(factors[0] == neutral))
    n_active = factors.shape[1] - n_neutral
    _logger.info(f"[COMPLEMENTARY] Energy masking applied:")
    _logger.info(f"  Complementary ranges (eV) : {complementary_ranges}")
    _logger.info(f"  Active parameters         : {n_active}/{factors.shape[1]}")
    _logger.info(f"  Neutral (EXFOR range)     : {n_neutral}/{factors.shape[1]}")
    print(f"[INFO] Masking: {n_active} active / {n_neutral} neutral parameters "
          f"(EXFOR range [{exfor_energy_min_ev:.3e}, {exfor_energy_max_ev:.3e}] eV excluded)")

    # ------------------------------------------------------------------
    #  Step 5: Process each sample
    # ------------------------------------------------------------------
    _logger.info(f"\n[COMPLEMENTARY] Processing {n_samples} samples "
                 f"({'dry run' if dry_run else 'writing ENDF files'}) ...")

    worker_args = [
        (
            sample_files[i],
            factors[i],
            i,
            energy_grids,
            param_mapping,
            output_dir,
            dry_run,
            generate_ace,
            ace_njoy_exe,
            ace_temperatures,
            ace_library_name,
            ace_njoy_version,
            ace_xsdir_file,
        )
        for i in range(n_samples)
    ]

    results = []
    n_success = 0
    n_fail = 0

    if n_procs > 1 and n_samples > 1:
        _logger.info(f"[COMPLEMENTARY] Using {n_procs} parallel workers")
        try:
            with Pool(processes=n_procs) as pool:
                results = pool.map(_process_complementary_sample, worker_args)
        except Exception as exc:
            _logger.error(f"[COMPLEMENTARY] Parallel processing failed, falling back to serial: {exc}")
            results = []
            for args in worker_args:
                results.append(_process_complementary_sample(args))
    else:
        for args in worker_args:
            results.append(_process_complementary_sample(args))

    for i, res in enumerate(results):
        if res and res.get("success"):
            n_success += 1
        else:
            n_fail += 1
            err = res.get("error", "unknown error") if res else "no result"
            _logger.error(f"[COMPLEMENTARY] Sample {i+1:04d} failed: {err}")

    _logger.info(f"\n[COMPLEMENTARY] Sample processing complete: "
                 f"{n_success} succeeded, {n_fail} failed out of {n_samples}")

    # ------------------------------------------------------------------
    #  Step 6: Write master perturbation file and summary
    # ------------------------------------------------------------------
    master_file = _write_master_perturbation_file(
        [factors], [param_mapping], [energy_grids],
        output_dir, timestamp, verbose=True,
    )

    # Final summary
    _logger.info(f"\n{separator}")
    _logger.info("[COMPLEMENTARY] [SUMMARY] Processing Results")
    _logger.info(separator)
    _logger.info(f"  Pipeline 1 samples input  : {n_samples}")
    _logger.info(f"  Successfully processed    : {n_success}")
    _logger.info(f"  Failed                    : {n_fail}")
    _logger.info(f"  EXFOR range excluded (eV) : [{exfor_energy_min_ev:.6e}, {exfor_energy_max_ev:.6e}]")
    _logger.info(f"  Active / total parameters : {n_active} / {factors.shape[1]}")
    _logger.info(f"  Sampling space            : {space}")
    _logger.info(f"  RNG seed                  : {seed}")
    if master_file:
        _logger.info(f"  Master factors file       : {master_file}")
    _logger.info(f"  Output directory          : {os.path.abspath(output_dir)}")
    if generate_ace:
        _logger.info(f"  ACE generation            : ENABLED")
    _logger.info(separator)

    print(f"\n[INFO] Complementary perturbation complete!")
    print(f"[INFO] {n_success}/{n_samples} samples processed successfully")
    if n_fail > 0:
        print(f"[WARNING] {n_fail} sample(s) failed — see log for details")
    print(f"[INFO] Log: {log_file}")
    if master_file:
        print(f"[INFO] Master factors: {os.path.basename(master_file)}")


# ============================================================================
#  Script entry point
# ============================================================================

if __name__ == "__main__":
    run_complementary_perturbation()
