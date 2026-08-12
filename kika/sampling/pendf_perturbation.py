"""
PENDF cross-section perturbation pipeline driven by MF33 covariance.

Sister module to ``ace_perturbation.py`` (multigroup ACE rewrite) and
``endf_perturbation.py`` (MF34 angular-distribution rewrite). Perturbs MF3
σ(E) directly on the MF33 native energy grid — no multigroup processing.

Pipeline stages (decoupled by design so the future MF3+MF4 combined pipeline
can call Stage B with paired perturbed inputs):

  Stage A  ENDF  →  RECONR-cached PENDF  →  MF33 cov on native grid  →
            generate_samples  →  perturbed PENDF tape on disk
  Stage B  (ENDF, perturbed PENDF)  →  NJOY (skip RECONR)  →  ACE + xsdir

Public entry: ``perturb_PENDF_files``.
"""
from __future__ import annotations

from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import time

import numpy as np

from kika.ace.xsdir import create_xsdir_files_for_ace
from kika._constants import MT_GROUPS
from kika.endf import read_endf
from kika._utils import MeV_to_kelvin
from kika.njoy.run_njoy import run_njoy_with_pendf
from kika.processing.njoy_pendf_cache import (
    DEFAULT_PENDF_CACHE_DIR,
    get_or_create_pendf,
)
from kika.sampling.carrier_blocks import (
    cross_section_carrier_blocks,
    cross_section_carrier_index,
)
from kika.sampling.mf33_sampling import (
    apply_factors_to_pendf_mf3,
    extract_mt_param_blocks,
    load_mf33_covariance,
)
from kika.sampling.multigroup_draw import (
    apply_legacy_autofix,
    draw_relative_factors,
)
from kika.sampling.utils import (
    DualLogger,
    _finalize_master_perturbation_matrix,
    _get_logger,
    _initialize_master_perturbation_matrix,
    _merge_isotope_metadata,
    _set_logger,
    _write_isotope_parquet,
    normalize_mt_list,
    resolve_signed_request,
)


# ============================================================================
# PIPELINE CONFIGURATION CONSTANTS
# ============================================================================

# --- I/O & output formats ---------------------------------------------------
OUTPUT_FORMATS: Tuple[str, ...] = ("pendf", "ace")   # subset of {"pendf", "ace"}
KEEP_NJOY_IO: bool = True                       # retain njoy input/output per sample

# --- PENDF cache ------------------------------------------------------------
PENDF_CACHE_DIR: Optional[str] = None           # None → DEFAULT_PENDF_CACHE_DIR (portable tempdir)
PENDF_TOLERANCE: float = 1.0e-3                 # NJOY reconr err
PENDF_TIMEOUT_S: float = 600.0

# --- Sampling ---------------------------------------------------------------
SAMPLING_SPACE: str = "log"                     # log preserves σ > 0
DECOMPOSITION_METHOD: str = "svd"
SAMPLING_METHOD: str = "sobol"
PSD_METHOD: str = "auto"
AUTOFIX: Optional[str] = None                   # None | "soft" | "medium" | "hard"
HIGH_VAL_THRESH: float = 1.0
ACCEPT_TOL: float = -1.0e-4
MAX_RELATIVE_STD: Optional[float] = 3.0         # 300% cap; None to disable
PIN_THRESHOLD_BINS: bool = True

# --- NJOY post-PENDF (Stage B) ---------------------------------------------
ACE_TEMPERATURES: List[float] = [293.6]
ACE_LIBRARY_NAME: str = "endfb81"
ACE_NJOY_VERSION: str = "NJOY 2016.78"

# --- Runtime ---------------------------------------------------------------
N_PROCS: int = 1
VERBOSE_DIAGNOSTICS: int = 1                    # 0=off, 1=summaries, 2=per-MT


# Module-global logger; mirrored into kika.sampling.utils via _set_logger.
_logger: Optional[DualLogger] = None


def derive_mt_thresholds(mf3_sections: Dict[int, Any],
                         mts: Sequence[int]) -> Dict[int, float]:
    """First energy at which each MT has a positive cross section.

    ⚑ **This feeds the draw, not a plot.** The map goes to
    ``draw_relative_factors``' ``mt_thresholds``, which pins the bin containing a
    reaction's threshold so a perturbation cannot be applied where the cross
    section is still zero. Move these numbers and the sampled factors move with
    them.

    It is a named function rather than nine lines inline because of what reads
    ``mf3_sections``. The map is built from ``.energies`` and
    ``.cross_sections`` — ENDF's spelling — while the canonical ``CrossSection``
    spells the second one ``.values``. The GNDS roadmap's phase 4 P2 ("one σ-source
    shape") changes what ``read_pendf_mf3_sections`` returns, and this is the
    place where that change reaches sampled numbers instead of a diagnostic.
    ``kika/sampling/tests/test_mt_thresholds.py`` freezes it first, so that
    increment can say by how much rather than hope it was nothing.

    Parameters
    ----------
    mf3_sections : Dict[int, Any]
        MT → section, from :func:`kika.processing.read_pendf_mf3_sections`.
    mts : Sequence[int]
        The MTs to look up. An MT absent from *mf3_sections*, or one whose cross
        section is nowhere positive, is simply absent from the result — the
        caller then applies no threshold pinning for it.
    """
    thresholds: Dict[int, float] = {}
    for mt in mts:
        if mt in mf3_sections:
            e = np.asarray(mf3_sections[mt].energies, dtype=float)
            xs = np.asarray(mf3_sections[mt].cross_sections, dtype=float)
            positive = e[xs > 0]
            if positive.size:
                thresholds[mt] = float(positive[0])
    return thresholds


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def perturb_PENDF_files(
    endf_files: Union[str, List[str]],
    mt_list: Union[List[int], List[List[int]]],
    num_samples: int,
    *,
    output_formats: Sequence[str] = OUTPUT_FORMATS,
    keep_njoy_io: bool = KEEP_NJOY_IO,
    pendf_cache_dir: Optional[str] = PENDF_CACHE_DIR,
    pendf_tolerance: float = PENDF_TOLERANCE,
    pendf_timeout_s: float = PENDF_TIMEOUT_S,
    space: str = SAMPLING_SPACE,
    decomposition_method: str = DECOMPOSITION_METHOD,
    sampling_method: str = SAMPLING_METHOD,
    psd_method: str = PSD_METHOD,
    autofix: Optional[str] = AUTOFIX,
    high_val_thresh: float = HIGH_VAL_THRESH,
    accept_tol: float = ACCEPT_TOL,
    max_relative_std: Optional[float] = MAX_RELATIVE_STD,
    pin_threshold_bins: bool = PIN_THRESHOLD_BINS,
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
    """Perturb PENDF cross sections directly from MF33 covariance on the native grid.

    See module docstring for the pipeline overview. Returns a per-ZAID summary
    dict with sample counts, NJOY successes/failures, and threshold-pin/coverage
    diagnostics.

    ``log_file`` defaults to a freshly-named ``pendf_perturbation_<ts>.log``
    inside ``output_dir``. Pass an existing path (e.g. from the combined
    orchestrator) to append this run's log block to that file instead.
    """
    global _logger

    # ------------------------------------------------------------------
    # 1. Setup: output dir, logger, normalize inputs
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_file is None:
        log_file = os.path.join(output_dir, f"pendf_perturbation_{timestamp}.log")
        _logger = DualLogger(log_file, mode="w")
    else:
        _logger = DualLogger(log_file, mode="a")
    _set_logger(_logger)

    if isinstance(endf_files, str):
        endf_files = [endf_files]
    endf_files = [str(p) for p in endf_files]

    output_formats = tuple(s.lower() for s in output_formats)
    valid_formats = {"pendf", "ace"}
    bad = set(output_formats) - valid_formats
    if bad:
        raise ValueError(f"output_formats: unknown values {bad}; allowed: {valid_formats}")
    if not output_formats:
        raise ValueError("output_formats cannot be empty")

    cache_dir = Path(pendf_cache_dir).expanduser() if pendf_cache_dir else DEFAULT_PENDF_CACHE_DIR

    if "ace" in output_formats and njoy_exe is None:
        raise ValueError("njoy_exe must be provided when 'ace' is in output_formats")
    if "ace" in output_formats:
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

    mt_lists = normalize_mt_list(mt_list, len(endf_files), unit_label="ENDF files")

    # ------------------------------------------------------------------
    # 2. Banner + config dump (file only; console gets the headline)
    _logger.info("[INFO] [PENDF] Starting PENDF perturbation job", console=True)
    _logger.info(f"[INFO] [PENDF] Log file: {log_file}", console=True)
    _logger.info(f"[INFO] [PENDF] Output directory: {os.path.abspath(output_dir)}", console=True)
    _logger.info(f"[INFO] [PENDF] Output formats: {output_formats}", console=True)
    _logger.info("#== CONFIG ============================================================")
    _logger.info(f"  ENDF_FILES = ({len(endf_files)} files)")
    for j, f in enumerate(endf_files):
        _logger.info(f"    [{j+1}] {f}")
    _logger.info(f"  MT_LISTS = {mt_lists}")
    _logger.info(f"  NUM_SAMPLES = {num_samples}")
    _logger.info(f"  OUTPUT_FORMATS = {output_formats}")
    _logger.info(f"  KEEP_NJOY_IO = {keep_njoy_io}")
    _logger.info(f"  PENDF_CACHE_DIR = {cache_dir}")
    _logger.info(f"  PENDF_TOLERANCE = {pendf_tolerance}")
    _logger.info(f"  SAMPLING_SPACE = {space}")
    _logger.info(f"  DECOMPOSITION_METHOD = {decomposition_method}")
    _logger.info(f"  SAMPLING_METHOD = {sampling_method}")
    _logger.info(f"  PSD_METHOD = {psd_method}")
    _logger.info(f"  AUTOFIX = {autofix}")
    _logger.info(f"  MAX_RELATIVE_STD = {max_relative_std}")
    _logger.info(f"  PIN_THRESHOLD_BINS = {pin_threshold_bins}")
    _logger.info(f"  RANDOM_SEED = {seed}")
    _logger.info(f"  NPROCS = {nprocs}")
    _logger.info(f"  DRY_RUN = {dry_run}")
    _logger.info(f"  VERBOSE_DIAGNOSTICS = {verbose_diagnostics}")
    if "ace" in output_formats:
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

    # ------------------------------------------------------------------
    # 3. Initialize incremental matrix dir
    matrix_dir = _initialize_master_perturbation_matrix(output_dir, timestamp, num_samples)

    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "log_file": log_file,
        "output_dir": os.path.abspath(output_dir),
        "output_formats": list(output_formats),
        "isotopes": {},
    }
    metadata_fragments: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 4. Per-ENDF processing
    for file_idx, (endf_file, mt_list_for_file) in enumerate(zip(endf_files, mt_lists), start=1):
        step_t0 = time.time()
        _logger.info(
            f"#-- STEP {file_idx}: {os.path.basename(endf_file)} ({file_idx}/{len(endf_files)}) "
            f"--------------------"
        )
        try:
            zaid = int(read_endf(endf_file).zaid)
        except Exception as e:
            _logger.error(f"  [ERROR] [PENDF] Could not read ZAID from {endf_file}: {e}", console=True)
            continue

        try:
            iso_summary = _process_one_isotope(
                endf_file=endf_file,
                zaid=zaid,
                mt_list_for_file=mt_list_for_file,
                num_samples=num_samples,
                output_formats=output_formats,
                keep_njoy_io=keep_njoy_io,
                cache_dir=cache_dir,
                pendf_tolerance=pendf_tolerance,
                pendf_timeout_s=pendf_timeout_s,
                space=space,
                decomposition_method=decomposition_method,
                sampling_method=sampling_method,
                psd_method=psd_method,
                autofix=autofix,
                high_val_thresh=high_val_thresh,
                accept_tol=accept_tol,
                max_relative_std=max_relative_std,
                pin_threshold_bins=pin_threshold_bins,
                energy_ranges=energy_ranges,
                njoy_exe=njoy_exe,
                ace_temperatures=ace_temperatures,
                ace_extensions=ace_extensions,
                ace_library_name=ace_library_name,
                ace_njoy_version=ace_njoy_version,
                xsdir_file=xsdir_file,
                output_dir=output_dir,
                matrix_dir=matrix_dir,
                seed=seed,
                nprocs=nprocs,
                dry_run=dry_run,
                verbose_diagnostics=verbose_diagnostics,
                metadata_fragments=metadata_fragments,
            )
            summary["isotopes"][str(zaid)] = iso_summary
        except Exception as e:
            _logger.error(
                f"  [ERROR] [PENDF] STEP {file_idx} ({os.path.basename(endf_file)}): {e}",
                console=True,
            )
            summary["isotopes"][str(zaid)] = {"error": str(e)}
        finally:
            elapsed = time.time() - step_t0
            _logger.info(f"#-- END STEP {file_idx} (elapsed {elapsed:.1f}s) --------------------")

    # ------------------------------------------------------------------
    # 5. Finalize parquet matrix + run summary
    _merge_isotope_metadata(matrix_dir, metadata_fragments)
    master = _finalize_master_perturbation_matrix(matrix_dir, verbose=verbose_diagnostics > 0)
    summary["master_matrix"] = master

    n_iso = len(summary["isotopes"])
    n_failed = sum(1 for v in summary["isotopes"].values() if isinstance(v, dict) and "error" in v)
    _logger.info(
        f"[INFO] [PENDF] Done — {n_iso - n_failed}/{n_iso} isotopes processed successfully",
        console=True,
    )
    return summary


# ---------------------------------------------------------------------------
# Per-isotope orchestration
# ---------------------------------------------------------------------------

def _process_one_isotope(
    *,
    endf_file: str,
    zaid: int,
    mt_list_for_file: Sequence[int],
    num_samples: int,
    output_formats: Tuple[str, ...],
    keep_njoy_io: bool,
    cache_dir: Path,
    pendf_tolerance: float,
    pendf_timeout_s: float,
    space: str,
    decomposition_method: str,
    sampling_method: str,
    psd_method: str,
    autofix: Optional[str],
    high_val_thresh: float,
    accept_tol: float,
    max_relative_std: Optional[float],
    pin_threshold_bins: bool,
    energy_ranges: Optional[List[Tuple[float, float]]],
    njoy_exe: Optional[str],
    ace_temperatures: Optional[List[float]],
    ace_extensions: Optional[List[str]],
    ace_library_name: Optional[str],
    ace_njoy_version: str,
    xsdir_file: Optional[str],
    output_dir: str,
    matrix_dir: str,
    seed: Optional[int],
    nprocs: int,
    dry_run: bool,
    verbose_diagnostics: int,
    metadata_fragments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Stage A then optional Stage B for a single (ENDF, ZAID)."""
    log = _get_logger()

    # --- Stage A.1: cached PENDF via NJOY RECONR ----------------------
    recon_io_dir = (
        os.path.join(output_dir, "njoy_files", "recon", str(zaid))
        if keep_njoy_io else None
    )
    pendf_path = get_or_create_pendf(
        endf_file,
        tolerance=pendf_tolerance,
        njoy_exe=str(njoy_exe) if njoy_exe is not None else "njoy",
        cache_dir=cache_dir,
        timeout_s=pendf_timeout_s,
        keep_njoy_io_dir=recon_io_dir,
    )
    log.info(f"  [INFO] [PENDF] zaid={zaid}: PENDF at {pendf_path}")

    # --- Stage A.2: resolve user MT request (negatives, composites) ---
    all_mts_in_file = _all_mts_in_mf33(endf_file)
    mts_requested, mt_log_lines = _resolve_mt_request(
        list(mt_list_for_file or []), all_mts_in_file,
    )
    for line in mt_log_lines:
        log.info(f"  [INFO] [PENDF] zaid={zaid}: {line}")
    log.info(
        f"  [INFO] [PENDF] zaid={zaid}: MTs requested after resolve = {mts_requested}; "
        f"MF33 contains {all_mts_in_file}"
    )

    # --- Stage A.3: assemble MF33 covariance on native union grid ----
    cov, mf3_sections, union_grid, mts_present = load_mf33_covariance(
        endf_path=endf_file,
        pendf_path=str(pendf_path),
        mt_list=mts_requested,
        energy_unit="eV",
        logger=log,
    )
    log.info(
        f"  [INFO] [PENDF] zaid={zaid}: MF33 cov assembled — "
        f"MTs={mts_present}, native bins={len(union_grid) - 1}"
    )

    # --- Stage A.4: extract per-MT thresholds from PENDF (for pinning) -
    mt_thresholds: Optional[Dict[int, float]] = None
    if pin_threshold_bins:
        mt_thresholds = derive_mt_thresholds(mf3_sections, mts_present)

    # --- Stage A.5: sample perturbation factors -----------------------
    # The repairs are applied ahead of the draw, as a plan, rather than inside
    # it -- `draw_relative_factors` runs `generate_samples`' sequence in
    # `generate_samples`' order and is gated bit-for-bit against it.
    #
    # ``cov`` is rebound to the fixed covariance, and that is the one deliberate
    # behaviour change in this migration. ``generate_samples`` fixed a copy it
    # kept to itself, so ``extract_mt_param_blocks(cov)`` below read the
    # *unfixed* carrier -- and under ``autofix='hard'``, which drops whole
    # reactions, that gave slices for a layout the factors no longer had.
    # Unreachable from kika (``AUTOFIX`` is None everywhere) and only reachable
    # from kika-app at 'soft', which drops nothing. The comment below has always
    # said "from the *fixed* cov"; now it is true.
    cov, mts_after_fix, fix_info = apply_legacy_autofix(
        cov, autofix,
        mt_numbers=mts_present,
        high_val_thresh=high_val_thresh,
        accept_tol=accept_tol,
        verbose=verbose_diagnostics > 0,
        logger=log,
    )
    (block_key, joint), = cross_section_carrier_blocks(cov)
    (_key, block_index), = cross_section_carrier_index(cov).items()
    factors, draw_info = draw_relative_factors(
        joint,
        num_samples,
        key=block_key,
        pairs=block_index["pairs"],
        stride=block_index["stride"],
        bins=union_grid,
        space=space,
        decomposition_method=decomposition_method,
        sampling_method=sampling_method,
        seed=seed,
        psd_method=psd_method,
        max_relative_std=max_relative_std,
        mt_thresholds=mt_thresholds,
        verbose=verbose_diagnostics > 0,
        logger=log,
        label=str(zaid),
    )
    log.info(f"  [INFO] [PENDF] zaid={zaid}: sampled factors shape={factors.shape}")
    log.info(
        f"  [INFO] [PENDF] zaid={zaid}: conditioning plan = "
        f"{[s.remedy for s in draw_info['plan'].steps] or ['none']}, "
        f"{draw_info['n_inert_dropped']} inert bin(s) dropped"
    )

    if mts_after_fix is not None and list(mts_after_fix) != list(mts_present):
        # autofix removed some MTs; rebuild mt_to_param_block from the *fixed* cov
        log.info(
            f"  [INFO] [PENDF] zaid={zaid}: autofix dropped MTs "
            f"{sorted(set(mts_present) - set(mts_after_fix))}"
        )
        mts_present = list(mts_after_fix)

    mt_to_param_block = extract_mt_param_blocks(cov)
    # Restrict to surviving MTs and re-sync mts_present to the MTs that
    # actually have factor blocks. Cov assembly silently drops blocks that
    # cannot be made relative (e.g. absolute MF33 with no matching PENDF
    # MF3 for that MT), so the post-cov MT set can be a strict subset of
    # the MF33-section MT set returned by ``load_mf33_covariance``.
    mt_to_param_block = {mt: sl for mt, sl in mt_to_param_block.items() if mt in mts_present}
    mts_present = list(mt_to_param_block.keys())

    # --- Stage A.6: optional energy-range mask on factors -------------
    if energy_ranges:
        factors = _mask_factors_by_energy_range(
            factors, mt_to_param_block, union_grid, energy_ranges
        )

    # --- Stage A.7: persist isotope parquet ---------------------------
    fragment = _write_isotope_parquet(
        matrix_dir, zaid, factors, list(mts_present), list(union_grid),
        verbose=verbose_diagnostics > 0, logger=log,
    )
    if fragment is not None:
        metadata_fragments.append(fragment)

    if dry_run:
        return {
            "mts": list(mts_present),
            "n_native_bins": len(union_grid) - 1,
            "n_samples": int(factors.shape[0]),
            "dry_run": True,
        }

    # --- Stage A.8: per-sample PENDF rewrite (+ Stage B if requested) -
    base = os.path.splitext(os.path.basename(endf_file))[0]
    pendf_out_root = os.path.join(output_dir, "pendf", str(zaid))
    os.makedirs(pendf_out_root, exist_ok=True)

    sample_args = []
    for s in range(int(factors.shape[0])):
        sample_args.append({
            "endf_file": endf_file,
            "pendf_path": str(pendf_path),
            "out_pendf_dir": pendf_out_root,
            "base": base,
            "sample_index": s,
            "factors_one_sample": factors[s],
            "mt_to_param_block": mt_to_param_block,
            "union_grid": union_grid,
            "output_formats": output_formats,
            "keep_njoy_io": keep_njoy_io,
            "njoy_exe": njoy_exe,
            "ace_temperatures": ace_temperatures,
            "ace_extensions": ace_extensions,
            "ace_library_name": ace_library_name,
            "ace_njoy_version": ace_njoy_version,
            "xsdir_file": xsdir_file,
            "output_dir": output_dir,
            "zaid": zaid,
        })

    # ACE generation dominates runtime (~10-60s per sample) — parallelize
    # eagerly. PENDF-only is millisecond-cheap so fork overhead dominates
    # below ~100 samples.
    parallelize = nprocs > 1 and (
        ("ace" in output_formats and num_samples >= 2) or num_samples >= 100
    )
    if parallelize:
        log.info(
            f"  [INFO] [PENDF] zaid={zaid}: running {num_samples} samples on "
            f"{min(nprocs, num_samples)} workers"
        )
        with Pool(processes=min(nprocs, num_samples)) as pool:
            sample_results = pool.map(_process_one_sample, sample_args, chunksize=1)
    else:
        sample_results = [_process_one_sample(args) for args in sample_args]

    n_pendf_ok = sum(1 for r in sample_results if r.get("pendf_ok"))
    n_ace_ok = sum(1 for r in sample_results if r.get("ace_ok"))
    log.info(
        f"  [INFO] [PENDF] zaid={zaid}: {n_pendf_ok}/{num_samples} PENDFs written"
        + (f", {n_ace_ok}/{num_samples} ACEs generated" if "ace" in output_formats else "")
    )

    # Aggregate per-sample errors across all samples (de-duplicate identical messages)
    error_counts: Dict[str, int] = {}
    for r in sample_results:
        for err in r.get("errors", []):
            error_counts[err] = error_counts.get(err, 0) + 1
    if error_counts:
        log.warning(
            f"  [WARN] [PENDF] zaid={zaid}: {sum(error_counts.values())} sample errors "
            f"across {len(error_counts)} distinct messages:"
        )
        for err, count in sorted(error_counts.items(), key=lambda kv: -kv[1])[:10]:
            log.warning(f"  [WARN] [PENDF] zaid={zaid}:   ({count}×) {err}")

    return {
        "mts": list(mts_present),
        "n_native_bins": len(union_grid) - 1,
        "n_samples": int(factors.shape[0]),
        "n_pendf_ok": n_pendf_ok,
        "n_ace_ok": n_ace_ok,
        "fix_info": fix_info,
    }


# ---------------------------------------------------------------------------
# Per-sample worker (top-level for Pool pickling)
# ---------------------------------------------------------------------------

def _process_one_sample(args: Dict[str, Any]) -> Dict[str, Any]:
    """Stage A.i: rewrite PENDF MF3 with sampled factors. Optionally Stage B."""
    sample_str = f"{args['sample_index'] + 1:04d}"
    sample_dir = os.path.join(args["out_pendf_dir"], sample_str)
    os.makedirs(sample_dir, exist_ok=True)
    out_pendf = os.path.join(sample_dir, f"{args['base']}_{sample_str}.pendf")

    result: Dict[str, Any] = {
        "sample_index": args["sample_index"],
        "out_pendf": out_pendf,
        "pendf_ok": False,
        "ace_ok": False,
        "errors": [],
    }
    try:
        # Need MF3 sections at runtime — re-read from cached PENDF.
        # (Sending the dict through Pool would pickle large arrays; re-read is cheap.)
        from kika.processing.njoy_pendf_cache import read_pendf_mf3_sections
        mf3_sections = read_pendf_mf3_sections(args["pendf_path"])

        diag = apply_factors_to_pendf_mf3(
            pendf_path=args["pendf_path"],
            out_pendf_path=out_pendf,
            mf3_sections=mf3_sections,
            factors_one_sample=args["factors_one_sample"],
            mt_to_param_block=args["mt_to_param_block"],
            bins_native=args["union_grid"],
        )
        result["pendf_ok"] = True
        result["mt_diagnostics"] = diag
    except Exception as e:
        result["errors"].append(f"Stage A: {e}")
        return result

    if "ace" in args["output_formats"]:
        try:
            ace_result = _stage_b_run_njoy_for_pair(
                endf_path=args["endf_file"],
                pendf_path=out_pendf,
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
            result["ace_ok"] = ace_result.get("success", False)
            result["ace"] = ace_result
            for err in ace_result.get("errors", []):
                result["errors"].append(f"Stage B: {err}")
        except NotImplementedError as e:
            result["errors"].append(f"Stage B: {e}")
        except Exception as e:
            result["errors"].append(f"Stage B: {e}")

    return result


# ---------------------------------------------------------------------------
# Stage B — to be wired in tasks #4–#5
# ---------------------------------------------------------------------------

def _stage_b_run_njoy_for_pair(
    *,
    endf_path: str,
    pendf_path: str,
    sample_index: int,
    zaid: int,
    njoy_exe: str,
    ace_temperatures: List[float],
    ace_extensions: Optional[List[str]],
    ace_library_name: str,
    ace_njoy_version: str,
    xsdir_file: Optional[str],
    output_dir: str,
    keep_njoy_io: bool,
) -> Dict[str, Any]:
    """(ENDF, perturbed PENDF) → NJOY chain (RECONR-skipped) → ACE + xsdir.

    Iterates over ``ace_temperatures`` paired with ``ace_extensions``; per
    temperature runs :func:`kika.njoy.run_njoy.run_njoy_with_pendf` and
    deposits the ACE / xsdir directly under the new per-sample layout
    (``ace/<NNNN>/<ZAID>.<ext>`` and ``ace/<NNNN>/xsdir/<ZAID>.<ext>.xsdir``).
    Errors at one temperature do not abort the others. The signature is
    stable across pipelines: the combined MF3+MF4 pipeline calls this with
    a *perturbed* ENDF instead of the original.
    """
    import shutil
    import tempfile
    from kika.sampling.utils import make_ace_sample_paths

    sample_str = f"{sample_index + 1:04d}"
    results: Dict[str, Any] = {
        "success": True,
        "temperatures_processed": [],
        "errors": [],
        "warnings": [],
    }

    ace_sample_dir, xsdir_sample_dir, _ = make_ace_sample_paths(output_dir, sample_index)
    os.makedirs(ace_sample_dir, exist_ok=True)
    os.makedirs(xsdir_sample_dir, exist_ok=True)
    njoy_sample_dir = os.path.join(output_dir, "njoy_files", sample_str)
    if keep_njoy_io:
        os.makedirs(njoy_sample_dir, exist_ok=True)

    if ace_extensions is not None and len(ace_extensions) != len(ace_temperatures):
        results["warnings"].append(
            f"ace_extensions length ({len(ace_extensions)}) must match "
            f"ace_temperatures length ({len(ace_temperatures)}); ignoring extensions."
        )
        ace_extensions = None

    for t_idx, temp in enumerate(ace_temperatures):
        ext_str = ace_extensions[t_idx] if ace_extensions is not None else None

        try:
            with tempfile.TemporaryDirectory(
                prefix="njoy_pendf_", dir=output_dir,
            ) as temp_workdir:
                njoy_result = run_njoy_with_pendf(
                    njoy_exe=njoy_exe,
                    endf_path=endf_path,
                    pendf_path=pendf_path,
                    temperature=temp,
                    library_name=ace_library_name,
                    output_dir=temp_workdir,
                    njoy_version=ace_njoy_version,
                    additional_suffix=sample_str if ext_str is None else None,
                    extension=ext_str,
                    ace_dir=ace_sample_dir if ext_str is not None else None,
                    xsdir_dir=xsdir_sample_dir if ext_str is not None else None,
                    njoy_files_dir=njoy_sample_dir if (ext_str is not None and keep_njoy_io) else None,
                )

                if keep_njoy_io:
                    for aux in ("njoy_input", "njoy_output"):
                        path = njoy_result.get(aux)
                        if path and os.path.exists(path) and os.path.dirname(path) != njoy_sample_dir:
                            try:
                                shutil.move(path, os.path.join(
                                    njoy_sample_dir, os.path.basename(path),
                                ))
                            except Exception as move_err:
                                results["warnings"].append(
                                    f"Could not move {aux} at {temp}K: {move_err}"
                                )

                if njoy_result["returncode"] == 0:
                    # Legacy path: NJOY put the ACE under a library/<temp>K subdir
                    # inside temp_workdir — relocate to the per-sample layout.
                    if ext_str is None and njoy_result.get("ace_file") and os.path.exists(njoy_result["ace_file"]):
                        dest_ace = os.path.join(ace_sample_dir, os.path.basename(njoy_result["ace_file"]))
                        shutil.move(njoy_result["ace_file"], dest_ace)
                        njoy_result["ace_file"] = dest_ace
                    if ext_str is None and njoy_result.get("xsdir_file") and os.path.exists(njoy_result["xsdir_file"]):
                        xsd_dest = os.path.join(xsdir_sample_dir, os.path.basename(njoy_result["xsdir_file"]))
                        shutil.move(njoy_result["xsdir_file"], xsd_dest)
                        njoy_result["xsdir_file"] = xsd_dest

                    ace_path = njoy_result.get("ace_file")
                    if ace_path and os.path.exists(ace_path):
                        results.setdefault("ace_files", []).append(ace_path)

                        if xsdir_file is not None:
                            try:
                                from kika.ace.parsers import read_ace
                                ace_data = read_ace(ace_path)
                                hdr = ace_data.header
                                has_ptable = bool(getattr(
                                    ace_data.unresolved_resonance, "has_data", False,
                                ))
                                create_xsdir_files_for_ace(
                                    ace_file_path=ace_path,
                                    zaid=hdr.zaid,
                                    awr=hdr.atomic_weight_ratio,
                                    xss_len=hdr.nxs_array[1],
                                    temperature_mev=hdr.temperature,
                                    sample_index=sample_index,
                                    output_dir=output_dir,
                                    master_xsdir_file=xsdir_file,
                                    has_ptable=has_ptable,
                                    per_sample_xsdir_path=njoy_result.get("xsdir_file"),
                                )
                            except Exception as xsdir_err:
                                results["warnings"].append(
                                    f"XSDIR write failed at {temp}K: {xsdir_err}"
                                )

                    results["temperatures_processed"].append(temp)
                else:
                    error_msg = (
                        f"NJOY failed at {temp}K with return code "
                        f"{njoy_result['returncode']}"
                    )
                    results["errors"].append(error_msg)
                    results["success"] = False

        except Exception as e:
            results["errors"].append(f"Exception at {temp}K: {e}")
            results["success"] = False

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_mts_in_mf33(endf_path: str) -> List[int]:
    """Return all MT numbers present in MF33 of the ENDF tape."""
    from kika.endf.parsers.parse_endf import parse_endf_file
    mf33 = parse_endf_file(endf_path).get_file(33)
    if mf33 is None or not getattr(mf33, "sections", None):
        return []
    return sorted(int(mt) for mt in mf33.sections.keys())


def _resolve_mt_request(
    mt_list_for_file: Sequence[int],
    all_mts_in_mf33: Sequence[int],
) -> Tuple[List[int], List[str]]:
    """Resolve positive/negative MT semantics for the PENDF pipeline.

    Thin wrapper around :func:`kika.sampling.utils.resolve_signed_request`
    pinned to ``kika._constants.MT_GROUPS`` so the user can write
    ``[-4]`` to drop both MT4 and the inelastic-level partials (MT51–91).
    """
    resolved, log = resolve_signed_request(
        list(mt_list_for_file or []), list(all_mts_in_mf33), group_map=MT_GROUPS,
    )
    # Re-label generic helper output as MT-specific for readable logs.
    log = [line.replace("Excluding entries:", "Excluding MTs:")
              .replace("group expansion", "composite expansion")
           for line in log]
    return resolved, log


def _mask_factors_by_energy_range(
    factors: np.ndarray,
    mt_to_param_block: Dict[int, slice],
    union_grid: Sequence[float],
    energy_ranges: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Force factor=1.0 in bins whose centre lies outside any (emin,emax) range."""
    bins = np.asarray(union_grid, dtype=float)
    centers = 0.5 * (bins[:-1] + bins[1:])
    in_range = np.zeros(centers.size, dtype=bool)
    for emin, emax in energy_ranges:
        in_range |= (centers >= emin) & (centers <= emax)
    out = factors.copy()
    for mt, sl in mt_to_param_block.items():
        block = out[:, sl]
        block[:, ~in_range] = 1.0
        out[:, sl] = block
    return out
