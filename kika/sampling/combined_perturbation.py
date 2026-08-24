"""
Combined MF3 (σ) + MF4 (angular) ENDF perturbation pipeline.

Drives the two existing single-physics pipelines independently and then runs
ONE NJOY ACER pass per sample using the paired (perturbed-ENDF, perturbed-
PENDF). Sample i from each pipeline is matched by index to form the input pair.

⚠ THIS PIPELINE ASSUMES THE TWO FAMILIES ARE INDEPENDENT, AND SAYS SO HERE
RATHER THAN IN A COMMENT NOBODY READS.

That assumption is right for every *released* evaluation — ENDF-6 has a home
for a magnitude/shape covariance (MF34's a₀ blocks, §34.1) but no released
library fills it — and it is **wrong for a tape that does**. The Fe-56
deliverable states ``Cov(sigma, a_l)`` in exactly those blocks. For such a tape
use :func:`kika.sampling.joint_perturbation.perturb_joint_mf33_mf34`, which
draws both families from ONE covariance.

⚠ And there is a second, independent defect in the pairing itself, which does
NOT need a cross term to bite: each pipeline draws a *balanced* Sobol set over
its own cube, and matching them by replica index correlates the two. Measured
on the PFNS driver, the mixture came out 18-21 % narrower than independence.
The joint path is immune by construction — there is no second set to pair — so
it is fixed there and left here, because changing this function's numbers is a
decision with its own measurement and not a side effect of adding another one.
The earlier version of this docstring asserted the cross-correlations were
"structurally zero"; that was a statement about our own methodology at the
time, not about the format or about this file's inputs.

Pipeline stages
---------------
  Stage A.MF34 :  ENDF → ``perturb_ENDF_files(generate_ace=False)``  →
                  perturbed ENDF tape per sample under
                  ``output_dir/endf/<zaid>/<NNNN>/<base>_<NNNN>.endf``.
  Stage A.MF33 :  ENDF → ``perturb_PENDF_files(output_formats=('pendf',))``
                  → perturbed PENDF tape per sample under
                  ``output_dir/pendf/<zaid>/<NNNN>/<base>_<NNNN>.pendf``.
  Stage B      :  pair sample i from each, call
                  :func:`pendf_perturbation._stage_b_run_njoy_for_pair`
                  → ACE + xsdir at ``output_dir/ace/<temp>/...``.

Output layout matches the ACE/ENDF pipelines: a single flat ``output_dir``
holding ``endf/``, ``pendf/``, ``ace/``, ``njoy_files/``, ``xsdir/``, plus
one parquet per perturbation kind (σ factors and Legendre-coefficient
factors) and one log per stage.
"""
from __future__ import annotations

from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import time

from kika.endf import read_endf
from kika.sampling.endf_perturbation import perturb_ENDF_files
from kika.sampling.pendf_perturbation import (
    _stage_b_run_njoy_for_pair,
    perturb_PENDF_files,
)
from kika.sampling.utils import (
    DualLogger,
    _get_logger,
    _set_logger,
    normalize_mt_list,
)


# ============================================================================
# PIPELINE CONFIGURATION CONSTANTS
# ============================================================================

# --- Output --------------------------------------------------------------
GENERATE_ACE: bool = True              # Stage B (paired NJOY → ACE) toggle
KEEP_NJOY_IO: bool = True              # retain njoy input/output per sample

# --- PENDF cache ---------------------------------------------------------
PENDF_CACHE_DIR: Optional[str] = None  # None → DEFAULT_PENDF_CACHE_DIR
PENDF_TOLERANCE: float = 1.0e-3
PENDF_TIMEOUT_S: float = 600.0

# --- MF33 (σ) sampling ---------------------------------------------------
SAMPLING_SPACE_MF33: str = "log"
DECOMPOSITION_METHOD_MF33: str = "svd"
SAMPLING_METHOD_MF33: str = "sobol"
PSD_METHOD_MF33: str = "auto"
AUTOFIX_MF33: Optional[str] = None
HIGH_VAL_THRESH_MF33: float = 1.0
ACCEPT_TOL_MF33: float = -1.0e-4
MAX_RELATIVE_STD_MF33: Optional[float] = 3.0
PIN_THRESHOLD_BINS_MF33: bool = True

# --- MF34 (angular) sampling ---------------------------------------------
SAMPLING_SPACE_MF34: str = "linear"
DECOMPOSITION_METHOD_MF34: str = "svd"
SAMPLING_METHOD_MF34: str = "sobol"
PSD_METHOD_MF34: str = "none"                # do nothing to the matrix by default
HIGHAM_PROJECTION_MF34: bool = False         # opt-in Higham repair shorthand
ENFORCE_POSITIVITY_MF34: bool = True         # project unphysical samples to f(μ)≥0
POSITIVITY_CHECK_POINTS_MF34: int = 101      # μ-grid resolution for check/projection

# --- NJOY post-pair (Stage B) -------------------------------------------
ACE_TEMPERATURES: List[float] = [293.6]
ACE_EXTENSIONS: Optional[List[str]] = None   # one entry per temperature, e.g. ['02c']
ACE_LIBRARY_NAME: str = "endfb81"
ACE_NJOY_VERSION: str = "NJOY 2016.78"

# --- Runtime ------------------------------------------------------------
N_PROCS: int = 1
VERBOSE_DIAGNOSTICS: int = 1


_logger: Optional[DualLogger] = None


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def perturb_ENDF_PENDF_combined(
    endf_files: Union[str, List[str]],
    mt_list_mf33: Union[List[int], List[List[int]]],
    mt_list_mf34: Union[List[int], List[List[int]]],
    legendre_coeffs: List[int],
    num_samples: int,
    *,
    generate_ace: bool = GENERATE_ACE,
    keep_njoy_io: bool = KEEP_NJOY_IO,
    pendf_cache_dir: Optional[str] = PENDF_CACHE_DIR,
    pendf_tolerance: float = PENDF_TOLERANCE,
    pendf_timeout_s: float = PENDF_TIMEOUT_S,
    space_mf33: str = SAMPLING_SPACE_MF33,
    decomposition_method_mf33: str = DECOMPOSITION_METHOD_MF33,
    sampling_method_mf33: str = SAMPLING_METHOD_MF33,
    psd_method_mf33: str = PSD_METHOD_MF33,
    autofix_mf33: Optional[str] = AUTOFIX_MF33,
    high_val_thresh_mf33: float = HIGH_VAL_THRESH_MF33,
    accept_tol_mf33: float = ACCEPT_TOL_MF33,
    max_relative_std_mf33: Optional[float] = MAX_RELATIVE_STD_MF33,
    pin_threshold_bins_mf33: bool = PIN_THRESHOLD_BINS_MF33,
    space_mf34: str = SAMPLING_SPACE_MF34,
    decomposition_method_mf34: str = DECOMPOSITION_METHOD_MF34,
    sampling_method_mf34: str = SAMPLING_METHOD_MF34,
    psd_method_mf34: str = PSD_METHOD_MF34,
    higham_projection_mf34: bool = HIGHAM_PROJECTION_MF34,
    enforce_positivity_mf34: bool = ENFORCE_POSITIVITY_MF34,
    positivity_check_points_mf34: int = POSITIVITY_CHECK_POINTS_MF34,
    energy_ranges: Optional[List[Tuple[float, float]]] = None,
    njoy_exe: Optional[str] = None,
    ace_temperatures: Optional[Sequence[float]] = None,
    ace_extensions: Optional[Sequence[str]] = None,
    ace_library_name: Optional[str] = None,
    ace_njoy_version: str = ACE_NJOY_VERSION,
    xsdir_file: Optional[str] = None,
    output_dir: str = ".",
    seed: Optional[int] = None,
    nprocs: int = N_PROCS,
    dry_run: bool = False,
    verbose_diagnostics: int = VERBOSE_DIAGNOSTICS,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run MF33 (σ) and MF34 (angular) perturbation independently, then pair-NJOY.

    Parameters
    ----------
    endf_files
        Single path or list of ENDF tape paths.
    mt_list_mf33
        MTs to perturb in MF33 (cross sections, written to PENDF). Flat list
        broadcasts across all input ENDFs, list-of-lists is per-file.
    mt_list_mf34
        MTs to perturb in MF34 (angular distributions, written to ENDF MF4).
        Same broadcast/per-file convention.
    legendre_coeffs
        Legendre coefficient indices to perturb (e.g. ``[1, 2, 3]``).
    num_samples
        Number of paired (ENDF, PENDF) realisations to produce.
    seed
        Master seed. Shared seed pattern: MF33 uses ``seed``, MF34 uses
        ``seed + 1`` so a given ``seed`` reproduces the same paired set.
    generate_ace
        If True (default), Stage B runs NJOY against each (perturbed ENDF,
        perturbed PENDF) pair and writes ACE + xsdir.

    Output layout
    -------------
    ``output_dir/``
        ├── ``combined_perturbation_<ts>.log``       orchestrator log
        ├── ``endf_perturbation_<ts>.log``           MF34 sub-step log
        ├── ``pendf_perturbation_<ts>.log``          MF33 sub-step log
        ├── ``endf_perturbation_factors_<ts>.parquet``    Legendre factors
        ├── ``perturbation_matrix_<ts>_master.parquet``   σ factors
        ├── ``endf/<zaid>/<NNNN>/<base>_<NNNN>.endf``
        ├── ``pendf/<zaid>/<NNNN>/<base>_<NNNN>.pendf``
        ├── ``ace/<temp_str>/<zaid>/<NNNN>/...``          paired ACEs (Stage B)
        ├── ``njoy_files/recon/<zaid>/<base>_recon.{input,output}``
        │                                                 RECONR deck + stdout
        │                                                 (only if keep_njoy_io)
        ├── ``njoy_files/<temp_str>/<zaid>/<NNNN>/<base>_<NNNN>.{input,output}``
        │                                                 acer-chain I/O
        │                                                 (only if keep_njoy_io)
        └── ``xsdir/<temp_str>/...``                      xsdir entries
                                                          (only if xsdir_file)
    """
    global _logger

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_file is None:
        log_file = os.path.join(output_dir, f"combined_perturbation_{timestamp}.log")
    # Orchestrator opens fresh; sub-pipelines below append to the same file.
    _logger = DualLogger(log_file, mode="w")
    _set_logger(_logger)

    if isinstance(endf_files, str):
        endf_files = [endf_files]
    endf_files = [str(p) for p in endf_files]

    if generate_ace and njoy_exe is None:
        raise ValueError("njoy_exe must be provided when generate_ace=True")
    if generate_ace:
        if ace_temperatures is None:
            ace_temperatures = list(ACE_TEMPERATURES)
        if ace_library_name is None:
            ace_library_name = ACE_LIBRARY_NAME
        ace_temperatures = [float(t) for t in ace_temperatures]
        if ace_extensions is not None:
            ace_extensions = [str(e) for e in ace_extensions]
            if len(ace_extensions) != len(ace_temperatures):
                raise ValueError(
                    f"ace_extensions length ({len(ace_extensions)}) must match "
                    f"ace_temperatures length ({len(ace_temperatures)})"
                )

    mt_lists_mf33 = normalize_mt_list(mt_list_mf33, len(endf_files), unit_label="ENDF files")
    mt_lists_mf34 = normalize_mt_list(mt_list_mf34, len(endf_files), unit_label="ENDF files")

    seed_mf33 = seed
    seed_mf34 = (seed + 1) if seed is not None else None

    _logger.info("[INFO] [COMB] Starting combined MF3+MF4 perturbation job", console=True)
    _logger.info(f"[INFO] [COMB] Log file: {log_file}", console=True)
    _logger.info(f"[INFO] [COMB] Output directory: {os.path.abspath(output_dir)}", console=True)
    _logger.info(f"[INFO] [COMB] generate_ace={generate_ace}, num_samples={num_samples}", console=True)

    _logger.info("#== CONFIG ============================================================")
    _logger.info(f"  ENDF_FILES = ({len(endf_files)} files)")
    for j, f in enumerate(endf_files):
        _logger.info(f"    [{j+1}] {f}")
    _logger.info(f"  MT_LIST_MF33 = {mt_lists_mf33}")
    _logger.info(f"  MT_LIST_MF34 = {mt_lists_mf34}")
    _logger.info(f"  LEGENDRE_COEFFS = {legendre_coeffs}")
    _logger.info(f"  NUM_SAMPLES = {num_samples}")
    _logger.info(f"  GENERATE_ACE = {generate_ace}")
    _logger.info(f"  KEEP_NJOY_IO = {keep_njoy_io}")
    _logger.info(f"  PENDF_CACHE_DIR = {pendf_cache_dir}")
    _logger.info(f"  PENDF_TOLERANCE = {pendf_tolerance}")
    _logger.info(f"  SAMPLING_SPACE_MF33 = {space_mf33}")
    _logger.info(f"  SAMPLING_SPACE_MF34 = {space_mf34}")
    _logger.info(f"  DECOMPOSITION_MF33 = {decomposition_method_mf33}")
    _logger.info(f"  DECOMPOSITION_MF34 = {decomposition_method_mf34}")
    _logger.info(f"  SAMPLING_METHOD_MF33 = {sampling_method_mf33}")
    _logger.info(f"  SAMPLING_METHOD_MF34 = {sampling_method_mf34}")
    _logger.info(f"  PSD_METHOD_MF33 = {psd_method_mf33}")
    _logger.info(f"  PSD_METHOD_MF34 = {psd_method_mf34}")
    _logger.info(f"  AUTOFIX_MF33 = {autofix_mf33}")
    _logger.info(f"  MAX_RELATIVE_STD_MF33 = {max_relative_std_mf33}")
    _logger.info(f"  PIN_THRESHOLD_BINS_MF33 = {pin_threshold_bins_mf33}")
    _logger.info(f"  SEED = {seed}  (mf33={seed_mf33}, mf34={seed_mf34})")
    _logger.info(f"  NPROCS = {nprocs}")
    _logger.info(f"  DRY_RUN = {dry_run}")
    _logger.info(f"  VERBOSE_DIAGNOSTICS = {verbose_diagnostics}")
    if generate_ace:
        _logger.info(f"  NJOY_EXE = {njoy_exe}")
        _logger.info(f"  ACE_TEMPERATURES = {ace_temperatures}")
        _logger.info(f"  ACE_LIBRARY_NAME = {ace_library_name}")
        _logger.info(f"  ACE_NJOY_VERSION = {ace_njoy_version}")
        _logger.info(f"  XSDIR_FILE = {xsdir_file}")
        from kika._constants import NDLIBRARY_TO_SUFFIX
        _lib_key = ace_library_name.lower().replace('-', '').replace('/', '').replace('.', '')
        if _lib_key not in NDLIBRARY_TO_SUFFIX:
            _logger.warning(
                f"  [WARN] ace_library_name={ace_library_name!r} not in "
                f"NDLIBRARY_TO_SUFFIX; ACE files will use fallback suffix '00'. "
                f"Known names: {sorted(NDLIBRARY_TO_SUFFIX.keys())}",
                console=True,
            )
    _logger.info("#== END CONFIG ========================================================")

    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "log_file": log_file,
        "output_dir": os.path.abspath(output_dir),
        "generate_ace": bool(generate_ace),
        "num_samples": int(num_samples),
        "isotopes": {},
    }

    # ----------------------------------------------------------------
    # Stage A.MF34: produce N perturbed ENDFs (no NJOY)
    # ----------------------------------------------------------------
    t0 = time.time()
    _logger.info("#-- STAGE A.MF34: perturbing ENDF angular data (MF4) ------------------")
    try:
        perturb_ENDF_files(
            endf_files=endf_files,
            mt_list=mt_lists_mf34,
            legendre_coeffs=legendre_coeffs,
            num_samples=num_samples,
            space=space_mf34,
            decomposition_method=decomposition_method_mf34,
            psd_method=psd_method_mf34,
            higham_projection=higham_projection_mf34,
            enforce_positivity=enforce_positivity_mf34,
            positivity_check_points=positivity_check_points_mf34,
            sampling_method=sampling_method_mf34,
            output_dir=output_dir,
            seed=seed_mf34,
            nprocs=nprocs,
            dry_run=dry_run,
            verbose=verbose_diagnostics > 0,
            generate_ace=False,
            energy_ranges=energy_ranges,
            log_file=log_file,
        )
    except Exception as e:
        _logger.error(f"  [ERROR] [COMB] Stage A.MF34 failed: {e}", console=True)
        summary["mf34_error"] = str(e)
        return summary
    finally:
        _logger.info(f"#-- STAGE A.MF34 done (elapsed {time.time() - t0:.1f}s) ----------")
    # The MF34 pipeline replaces the global logger; restore ours.
    _set_logger(_logger)

    # ----------------------------------------------------------------
    # Stage A.MF33: produce N perturbed PENDFs (no NJOY)
    # ----------------------------------------------------------------
    t0 = time.time()
    _logger.info("#-- STAGE A.MF33: perturbing PENDF cross sections (MF3) ---------------")
    try:
        pendf_summary = perturb_PENDF_files(
            endf_files=endf_files,
            mt_list=mt_lists_mf33,
            num_samples=num_samples,
            output_formats=("pendf",),
            keep_njoy_io=keep_njoy_io,
            pendf_cache_dir=pendf_cache_dir,
            pendf_tolerance=pendf_tolerance,
            pendf_timeout_s=pendf_timeout_s,
            space=space_mf33,
            decomposition_method=decomposition_method_mf33,
            sampling_method=sampling_method_mf33,
            psd_method=psd_method_mf33,
            autofix=autofix_mf33,
            high_val_thresh=high_val_thresh_mf33,
            accept_tol=accept_tol_mf33,
            max_relative_std=max_relative_std_mf33,
            pin_threshold_bins=pin_threshold_bins_mf33,
            energy_ranges=energy_ranges,
            njoy_exe=str(njoy_exe) if njoy_exe is not None else None,
            output_dir=output_dir,
            seed=seed_mf33,
            nprocs=nprocs,
            dry_run=dry_run,
            verbose_diagnostics=verbose_diagnostics,
            log_file=log_file,
        )
        summary["mf33_pendf_summary"] = pendf_summary
    except Exception as e:
        _logger.error(f"  [ERROR] [COMB] Stage A.MF33 failed: {e}", console=True)
        summary["mf33_error"] = str(e)
        return summary
    finally:
        _logger.info(f"#-- STAGE A.MF33 done (elapsed {time.time() - t0:.1f}s) ----------")
    _set_logger(_logger)

    if dry_run or not generate_ace:
        _logger.info(
            "[INFO] [COMB] Stage B skipped "
            f"({'dry_run' if dry_run else 'generate_ace=False'})",
            console=True,
        )
        return summary

    # ----------------------------------------------------------------
    # Stage B: pair sample i from MF34 ENDF + MF33 PENDF, run NJOY
    # ----------------------------------------------------------------
    t0 = time.time()
    _logger.info("#-- STAGE B: paired NJOY ACER (MF4 ENDF + MF3 PENDF) ------------------")

    for file_idx, endf_file in enumerate(endf_files, start=1):
        try:
            zaid = int(read_endf(endf_file).zaid)
        except Exception as e:
            _logger.error(
                f"  [ERROR] [COMB] {os.path.basename(endf_file)}: cannot read ZAID — {e}",
                console=True,
            )
            summary["isotopes"][os.path.basename(endf_file)] = {"error": str(e)}
            continue

        base, ext = os.path.splitext(os.path.basename(endf_file))
        endf_root = os.path.join(output_dir, "endf", str(zaid))
        pendf_root = os.path.join(output_dir, "pendf", str(zaid))

        pair_args: List[Dict[str, Any]] = []
        missing: List[Tuple[int, str]] = []
        for s in range(num_samples):
            sample_str = f"{s+1:04d}"
            endf_path = os.path.join(endf_root, sample_str, f"{base}_{sample_str}{ext}")
            pendf_path = os.path.join(pendf_root, sample_str, f"{base}_{sample_str}.pendf")
            if not os.path.exists(endf_path):
                missing.append((s, f"perturbed ENDF missing: {endf_path}"))
                continue
            if not os.path.exists(pendf_path):
                missing.append((s, f"perturbed PENDF missing: {pendf_path}"))
                continue
            pair_args.append({
                "endf_path": endf_path,
                "pendf_path": pendf_path,
                "sample_index": s,
                "zaid": zaid,
                "njoy_exe": str(njoy_exe),
                "ace_temperatures": list(ace_temperatures),
                "ace_extensions": list(ace_extensions) if ace_extensions is not None else None,
                "ace_library_name": ace_library_name,
                "ace_njoy_version": ace_njoy_version,
                "xsdir_file": xsdir_file,
                "output_dir": output_dir,
                "keep_njoy_io": keep_njoy_io,
            })

        if missing:
            for s, msg in missing[:5]:
                _logger.warning(f"  [WARN] [COMB] zaid={zaid} sample {s+1:04d}: {msg}")
            if len(missing) > 5:
                _logger.warning(
                    f"  [WARN] [COMB] zaid={zaid}: {len(missing)} samples missing inputs (first 5 above)"
                )

        if not pair_args:
            _logger.warning(
                f"  [WARN] [COMB] zaid={zaid}: no complete (ENDF, PENDF) pairs to run"
            )
            summary["isotopes"][str(zaid)] = {
                "n_attempted": 0, "n_ace_ok": 0, "n_missing_inputs": len(missing),
            }
            continue

        _logger.info(
            f"  [INFO] [COMB] zaid={zaid}: running NJOY on {len(pair_args)}/{num_samples} pairs"
            + (f" with {min(nprocs, len(pair_args))} workers" if nprocs > 1 else "")
        )

        if nprocs > 1 and len(pair_args) > 1:
            with Pool(processes=min(nprocs, len(pair_args))) as pool:
                pair_results = pool.map(_run_pair_b, pair_args, chunksize=1)
        else:
            pair_results = [_run_pair_b(args) for args in pair_args]

        n_ok = sum(1 for r in pair_results if r.get("success"))
        _logger.info(
            f"  [INFO] [COMB] zaid={zaid}: Stage B {n_ok}/{len(pair_args)} ACEs generated"
        )

        # Aggregate per-sample errors
        error_counts: Dict[str, int] = {}
        for r in pair_results:
            for err in r.get("errors", []):
                error_counts[err] = error_counts.get(err, 0) + 1
        if error_counts:
            _logger.warning(
                f"  [WARN] [COMB] zaid={zaid}: {sum(error_counts.values())} Stage B errors "
                f"across {len(error_counts)} distinct messages:"
            )
            for err, count in sorted(error_counts.items(), key=lambda kv: -kv[1])[:10]:
                _logger.warning(f"  [WARN] [COMB] zaid={zaid}:   ({count}×) {err}")

        summary["isotopes"][str(zaid)] = {
            "n_attempted": len(pair_args),
            "n_ace_ok": n_ok,
            "n_missing_inputs": len(missing),
        }

    _logger.info(f"#-- STAGE B done (elapsed {time.time() - t0:.1f}s) ----------------")
    _logger.info(
        f"[INFO] [COMB] Combined run complete; see {os.path.basename(log_file)}",
        console=True,
    )
    return summary


# ---------------------------------------------------------------------------
# Top-level worker (Pool pickling)
# ---------------------------------------------------------------------------

def _run_pair_b(args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker that drives ``_stage_b_run_njoy_for_pair`` for a single pair."""
    try:
        return _stage_b_run_njoy_for_pair(
            endf_path=args["endf_path"],
            pendf_path=args["pendf_path"],
            sample_index=args["sample_index"],
            zaid=args["zaid"],
            njoy_exe=args["njoy_exe"],
            ace_temperatures=args["ace_temperatures"],
            ace_extensions=args.get("ace_extensions"),
            ace_library_name=args["ace_library_name"],
            ace_njoy_version=args["ace_njoy_version"],
            xsdir_file=args["xsdir_file"],
            output_dir=args["output_dir"],
            keep_njoy_io=args["keep_njoy_io"],
        )
    except Exception as e:
        return {
            "success": False,
            "errors": [f"_stage_b_run_njoy_for_pair raised: {e}"],
            "warnings": [],
            "temperatures_processed": [],
        }
