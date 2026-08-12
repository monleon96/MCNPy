"""
nu-bar (MF31) perturbation pipeline.

Sister module to ``pendf_perturbation.py`` (MF33 → PENDF MF3) and
``endf_perturbation.py`` (MF34 → ENDF MF4). Samples nu-bar covariance from
MF31 and rewrites the MF1 nu-bar tables (MT452/455/456) directly in the ENDF
tape, then runs the standard NJOY chain to regenerate ACE.

Pipeline:
  ENDF  →  MF31 cov (central values from MF1 nu-bar)  →  generate_samples  →
  per sample: perturb MF1 nu-bar family (sum rule enforced)  →  perturbed ENDF
  →  NJOY  →  ACE + xsdir

Public entry: ``perturb_nubar_files``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import shutil
import time

import numpy as np

from kika._constants import NUBAR_TOTAL_MT, NUBAR_COMPONENT_MTS
from kika.endf.parsers.parse_endf import parse_endf_file
from kika.endf.writers.endf_writer import ENDFWriter, update_mf1_directory
from kika.sampling.carrier_blocks import (
    cross_section_carrier_blocks,
    cross_section_carrier_index,
)
from kika.sampling.multigroup_draw import (
    apply_legacy_autofix,
    draw_relative_factors,
)
from kika.sampling.mf31_sampling import (
    build_mf31_covariance,
    perturb_nubar_family,
    sum_rule_residual,
    extract_mt_param_blocks,
)
from kika.sampling.endf_perturbation import _process_njoy_for_sample
from kika.sampling.utils import (
    DualLogger,
    _finalize_master_perturbation_matrix,
    _get_logger,
    _initialize_master_perturbation_matrix,
    _merge_isotope_metadata,
    _set_logger,
    _write_isotope_parquet,
)

# ============================================================================
# PIPELINE CONFIGURATION CONSTANTS
# ============================================================================

# --- Output ----------------------------------------------------------------
GENERATE_ACE: bool = True               # run NJOY on each perturbed ENDF

# --- Sampling --------------------------------------------------------------
SAMPLING_SPACE: str = "log"             # log preserves nu-bar > 0
DECOMPOSITION_METHOD: str = "svd"
SAMPLING_METHOD: str = "sobol"
PSD_METHOD: str = "auto"
AUTOFIX: Optional[str] = None           # None | "soft" | "medium" | "hard"
MAX_RELATIVE_STD: Optional[float] = 3.0
DERIVED_MT: Optional[int] = None        # which family member is redundant (default 452)

# --- NJOY ------------------------------------------------------------------
ACE_TEMPERATURES: List[float] = [293.6]
ACE_LIBRARY_NAME: str = "endfb81"
ACE_NJOY_VERSION: str = "NJOY 2016.78"

# --- Runtime ---------------------------------------------------------------
VERBOSE_DIAGNOSTICS: int = 1            # 0=off, 1=summaries, 2=per-MT


_logger: Optional[DualLogger] = None


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def perturb_nubar_files(
    endf_files: Union[str, List[str]],
    num_samples: int,
    *,
    mt_list: Optional[Union[List[int], List[List[int]]]] = None,
    generate_ace: bool = GENERATE_ACE,
    space: str = SAMPLING_SPACE,
    decomposition_method: str = DECOMPOSITION_METHOD,
    sampling_method: str = SAMPLING_METHOD,
    psd_method: str = PSD_METHOD,
    autofix: Optional[str] = AUTOFIX,
    max_relative_std: Optional[float] = MAX_RELATIVE_STD,
    derived_mt: Optional[int] = DERIVED_MT,
    njoy_exe: Optional[str] = None,
    ace_temperatures: Optional[Sequence[float]] = None,
    ace_extensions: Optional[Sequence[str]] = None,
    ace_library_name: Optional[str] = None,
    ace_njoy_version: str = ACE_NJOY_VERSION,
    xsdir_file: Optional[str] = None,
    output_dir: str = ".",
    seed: Optional[int] = None,
    dry_run: bool = False,
    verbose_diagnostics: int = VERBOSE_DIAGNOSTICS,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Perturb MF1 nu-bar from MF31 covariance, optionally regenerating ACE.

    Returns a per-ZAID summary dict (sample counts, NJOY successes, the nu-bar
    MTs perturbed, and the derived MT recomputed from the sum rule).
    """
    global _logger

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_file is None:
        log_file = os.path.join(output_dir, f"nubar_perturbation_{timestamp}.log")
        _logger = DualLogger(log_file, mode="w")
    else:
        _logger = DualLogger(log_file, mode="a")
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

    # Per-file MT request (None → all MF31 MTs present in each file).
    mt_lists = _normalize_optional_mt_list(mt_list, len(endf_files))

    _logger.info("[INFO] [NUBAR] Starting nu-bar (MF31) perturbation job", console=True)
    _logger.info(f"[INFO] [NUBAR] Log file: {log_file}", console=True)
    _logger.info(f"[INFO] [NUBAR] Output directory: {os.path.abspath(output_dir)}", console=True)
    _logger.info("#== CONFIG ============================================================")
    _logger.info(f"  ENDF_FILES = ({len(endf_files)} files)")
    for j, f in enumerate(endf_files):
        _logger.info(f"    [{j+1}] {f}")
    _logger.info(f"  MT_LISTS = {mt_lists}")
    _logger.info(f"  NUM_SAMPLES = {num_samples}")
    _logger.info(f"  GENERATE_ACE = {generate_ace}")
    _logger.info(f"  SAMPLING_SPACE = {space}")
    _logger.info(f"  DECOMPOSITION_METHOD = {decomposition_method}")
    _logger.info(f"  SAMPLING_METHOD = {sampling_method}")
    _logger.info(f"  PSD_METHOD = {psd_method}")
    _logger.info(f"  AUTOFIX = {autofix}")
    _logger.info(f"  MAX_RELATIVE_STD = {max_relative_std}")
    _logger.info(f"  DERIVED_MT = {derived_mt}")
    _logger.info(f"  RANDOM_SEED = {seed}")
    _logger.info(f"  DRY_RUN = {dry_run}")
    _logger.info(f"  VERBOSE_DIAGNOSTICS = {verbose_diagnostics}")
    if generate_ace:
        _logger.info(f"  NJOY_EXE = {njoy_exe}")
        _logger.info(f"  ACE_TEMPERATURES = {ace_temperatures}")
        _logger.info(f"  ACE_LIBRARY_NAME = {ace_library_name}")
    _logger.info("#== END CONFIG ========================================================")

    matrix_dir = _initialize_master_perturbation_matrix(output_dir, timestamp, num_samples)

    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "log_file": log_file,
        "output_dir": os.path.abspath(output_dir),
        "isotopes": {},
    }
    metadata_fragments: List[Dict[str, Any]] = []

    for file_idx, (endf_file, mts_req) in enumerate(zip(endf_files, mt_lists), start=1):
        step_t0 = time.time()
        _logger.info(
            f"#-- STEP {file_idx}: {os.path.basename(endf_file)} "
            f"({file_idx}/{len(endf_files)}) --------------------"
        )
        try:
            # Parsed once here and handed down: the covariance assembly and the
            # ZAID lookup both need the whole tape, and parsing a JEFF-4.0
            # actinide is not cheap.
            endf_obj = parse_endf_file(endf_file)
            zaid = _zaid_of(endf_obj)
        except Exception as e:
            _logger.error(f"  [ERROR] [NUBAR] Could not read ZAID from {endf_file}: {e}", console=True)
            continue

        try:
            iso_summary = _process_one_isotope(
                endf_file=endf_file,
                endf_obj=endf_obj,
                zaid=zaid,
                mts_req=mts_req,
                num_samples=num_samples,
                generate_ace=generate_ace,
                space=space,
                decomposition_method=decomposition_method,
                sampling_method=sampling_method,
                psd_method=psd_method,
                autofix=autofix,
                max_relative_std=max_relative_std,
                derived_mt=derived_mt,
                njoy_exe=njoy_exe,
                ace_temperatures=ace_temperatures,
                ace_extensions=list(ace_extensions) if ace_extensions else None,
                ace_library_name=ace_library_name,
                ace_njoy_version=ace_njoy_version,
                xsdir_file=xsdir_file,
                output_dir=output_dir,
                matrix_dir=matrix_dir,
                seed=seed,
                dry_run=dry_run,
                verbose_diagnostics=verbose_diagnostics,
                metadata_fragments=metadata_fragments,
            )
            summary["isotopes"][str(zaid)] = iso_summary
        except Exception as e:
            _logger.error(
                f"  [ERROR] [NUBAR] STEP {file_idx} ({os.path.basename(endf_file)}): {e}",
                console=True,
            )
            summary["isotopes"][str(zaid)] = {"error": str(e)}
        finally:
            _logger.info(
                f"#-- END STEP {file_idx} (elapsed {time.time() - step_t0:.1f}s) "
                f"--------------------"
            )

    _merge_isotope_metadata(matrix_dir, metadata_fragments)
    summary["master_matrix"] = _finalize_master_perturbation_matrix(
        matrix_dir, verbose=verbose_diagnostics > 0
    )

    n_iso = len(summary["isotopes"])
    n_failed = sum(1 for v in summary["isotopes"].values() if isinstance(v, dict) and "error" in v)
    _logger.info(
        f"[INFO] [NUBAR] Done — {n_iso - n_failed}/{n_iso} isotopes processed successfully",
        console=True,
    )
    return summary


# ---------------------------------------------------------------------------
# Per-isotope orchestration
# ---------------------------------------------------------------------------

def _process_one_isotope(
    *,
    endf_file: str,
    endf_obj,
    zaid: int,
    mts_req: Optional[Sequence[int]],
    num_samples: int,
    generate_ace: bool,
    space: str,
    decomposition_method: str,
    sampling_method: str,
    psd_method: str,
    autofix: Optional[str],
    max_relative_std: Optional[float],
    derived_mt: Optional[int],
    njoy_exe: Optional[str],
    ace_temperatures: Optional[List[float]],
    ace_extensions: Optional[List[str]],
    ace_library_name: Optional[str],
    ace_njoy_version: str,
    xsdir_file: Optional[str],
    output_dir: str,
    matrix_dir: str,
    seed: Optional[int],
    dry_run: bool,
    verbose_diagnostics: int,
    metadata_fragments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    log = _get_logger()

    # --- assemble MF31 covariance + MF1 nu-bar sections -------------------
    cov, nubar_sections, union_grid, mts_present = build_mf31_covariance(
        endf_obj, mt_list=mts_req, energy_unit="eV", logger=log,
    )
    log.info(
        f"  [INFO] [NUBAR] zaid={zaid}: MF31 cov assembled — "
        f"MTs={mts_present}, native bins={len(union_grid) - 1}, "
        f"MF1 nu-bar sections={sorted(nubar_sections)}"
    )

    # --- sample perturbation factors --------------------------------------
    # No ``mt_thresholds`` on this path: nu-bar has no reaction threshold to
    # straddle, so the targeted rescale never applies and only the outlier
    # rescale and the cap do.
    cov, mts_after_fix, fix_info = apply_legacy_autofix(
        cov, autofix,
        mt_numbers=mts_present,
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
        verbose=verbose_diagnostics > 0,
        logger=log,
        label=str(zaid),
    )
    log.info(
        f"  [INFO] [NUBAR] zaid={zaid}: conditioning plan = "
        f"{[s.remedy for s in draw_info['plan'].steps] or ['none']}, "
        f"{draw_info['n_inert_dropped']} inert bin(s) dropped"
    )
    if mts_after_fix is not None and list(mts_after_fix) != list(mts_present):
        log.info(
            f"  [INFO] [NUBAR] zaid={zaid}: autofix dropped MTs "
            f"{sorted(set(mts_present) - set(mts_after_fix))}"
        )
        mts_present = list(mts_after_fix)
    log.info(f"  [INFO] [NUBAR] zaid={zaid}: sampled factors shape={factors.shape}")

    mt_to_param_block = extract_mt_param_blocks(cov)
    mt_to_param_block = {mt: sl for mt, sl in mt_to_param_block.items() if mt in mts_present}

    # --- which MTs actually receive their own factor block -----------------
    # The family member that ``perturb_nubar_family`` derives from the sum rule
    # never gets its own block applied. It is still sampled (MF31 gives it a
    # covariance and the flatten order has to match), so its columns sit in the
    # parquet looking like free parameters. Say so, here and in the metadata.
    derived_used = _derived_family_mt(derived_mt, nubar_sections)
    mts_applied = [mt for mt in mts_present if mt != derived_used]
    if derived_used is not None and derived_used in mts_present:
        log.info(
            f"  [INFO] [NUBAR] zaid={zaid}: MT{derived_used} factors sampled but "
            f"discarded — it is derived from the sum rule over "
            f"{[mt for mt in NUBAR_COMPONENT_MTS if mt in nubar_sections]}"
        )

    # --- how far the input file itself is from the sum rule ---------------
    residual = _report_sum_rule_residual(
        nubar_sections, union_grid, cov, mts_present, zaid, derived_used, log
    )

    fragment = _write_isotope_parquet(
        matrix_dir, zaid, factors, list(mts_present), list(union_grid),
        verbose=verbose_diagnostics > 0, logger=log,
        extra_metadata={
            "derived_mt": derived_used,
            "mts_sampled": list(mts_present),
            "mts_applied": list(mts_applied),
        },
    )
    if fragment is not None:
        metadata_fragments.append(fragment)

    if dry_run:
        return {
            "mts_perturbed": list(mts_present),
            "mts_applied": list(mts_applied),
            "derived_mt": derived_used,
            "sum_rule_residual": residual,
            "n_native_bins": len(union_grid) - 1,
            "n_samples": int(factors.shape[0]),
            "dry_run": True,
        }

    # --- per-sample MF1 rewrite (+ optional NJOY) -------------------------
    base, ext = os.path.splitext(os.path.basename(endf_file))
    endf_out_root = os.path.join(output_dir, "endf", str(zaid))
    os.makedirs(endf_out_root, exist_ok=True)

    n_endf_ok = 0
    n_ace_ok = 0
    error_counts: Dict[str, int] = {}
    diag_acc: Dict[int, Dict[str, float]] = {}

    for s in range(int(factors.shape[0])):
        sample_str = f"{s + 1:04d}"
        sample_dir = os.path.join(endf_out_root, sample_str)
        os.makedirs(sample_dir, exist_ok=True)
        out_endf = os.path.join(sample_dir, f"{base}_{sample_str}{ext}")

        try:
            fblocks = {mt: factors[s][sl] for mt, sl in mt_to_param_block.items()}
            out_sections, diags = perturb_nubar_family(
                nubar_sections, fblocks, union_grid,
                derived_mt=derived_mt, logger=log if verbose_diagnostics > 1 else None,
            )
            _accumulate_diags(diag_acc, diags)
            _write_perturbed_nubar_endf(endf_file, out_endf, out_sections)
            n_endf_ok += 1
        except Exception as e:
            error_counts[f"Stage A: {e}"] = error_counts.get(f"Stage A: {e}", 0) + 1
            continue

        if generate_ace:
            try:
                njoy_result = _process_njoy_for_sample(
                    out_endf=out_endf,
                    sample_index=s,
                    njoy_exe=njoy_exe,
                    temperatures=ace_temperatures,
                    extensions=ace_extensions,
                    library_name=ace_library_name,
                    njoy_version=ace_njoy_version,
                    output_dir=output_dir,
                    xsdir_file=xsdir_file,
                )
                if njoy_result.get("success"):
                    n_ace_ok += 1
                for err in njoy_result.get("errors", []):
                    error_counts[f"Stage B: {err}"] = error_counts.get(f"Stage B: {err}", 0) + 1
            except Exception as e:
                error_counts[f"Stage B: {e}"] = error_counts.get(f"Stage B: {e}", 0) + 1

    log.info(
        f"  [INFO] [NUBAR] zaid={zaid}: {n_endf_ok}/{factors.shape[0]} ENDFs written"
        + (f", {n_ace_ok}/{factors.shape[0]} ACEs generated" if generate_ace else "")
    )
    if error_counts:
        log.warning(
            f"  [WARN] [NUBAR] zaid={zaid}: {sum(error_counts.values())} sample errors "
            f"across {len(error_counts)} distinct messages:"
        )
        for err, count in sorted(error_counts.items(), key=lambda kv: -kv[1])[:10]:
            log.warning(f"  [WARN] [NUBAR] zaid={zaid}:   ({count}×) {err}")

    for mt, d in sorted(diag_acc.items()):
        log.info(
            f"  [INFO] [NUBAR] zaid={zaid}: MT{mt} factors applied in "
            f"[{d['min_factor']:.4f}, {d['max_factor']:.4f}], "
            f"{d['frac_out_of_coverage']:.1%} of table points outside MF31 coverage, "
            f"+{d['n_inserted']:.0f} points inserted"
        )

    return {
        "mts_perturbed": list(mts_present),
        "mts_applied": list(mts_applied),
        "derived_mt": derived_used,
        "sum_rule_residual": residual,
        "factor_diagnostics": diag_acc,
        "n_native_bins": len(union_grid) - 1,
        "n_samples": int(factors.shape[0]),
        "n_endf_ok": n_endf_ok,
        "n_ace_ok": n_ace_ok,
        "fix_info": fix_info,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derived_family_mt(
    derived_mt: Optional[int], nubar_sections: Dict[int, object]
) -> Optional[int]:
    """Which family member ``perturb_nubar_family`` will derive, if any.

    Mirrors that function's own rule: nothing is derived unless the whole
    452/455/456 family is present, since an incomplete family cannot be closed
    by the sum rule.
    """
    present = set(int(mt) for mt in nubar_sections)
    if not {NUBAR_TOTAL_MT, *NUBAR_COMPONENT_MTS}.issubset(present):
        return None
    return NUBAR_TOTAL_MT if derived_mt is None else int(derived_mt)


def _report_sum_rule_residual(
    nubar_sections, union_grid, cov, mts_present, zaid, derived_used, log,
) -> Optional[Dict[str, Any]]:
    """Measure and log how far the input file is from nu_452 = nu_455 + nu_456.

    Rebuilding the derived member repairs that residual, which moves the central
    value by something the perturbation never asked for. Expressed against the
    MF31 1σ so it reads as "the repair is worth this fraction of the stated
    uncertainty": the derived MT's own σ when MF31 covers it, otherwise the
    largest σ among the MTs that do — those are what end up driving it.
    """
    if derived_used is None:
        return None
    res = sum_rule_residual(nubar_sections, union_grid)
    if res is None:
        return None

    per_bin = np.asarray(res["per_bin_rel"], dtype=float)
    blocks = extract_mt_param_blocks(cov)
    diag = np.sqrt(np.clip(np.diag(cov.covariance_matrix), 0.0, None))
    if derived_used in mts_present and blocks.get(int(derived_used)) is not None:
        sig = diag[blocks[int(derived_used)]]
    else:
        sig = np.zeros_like(per_bin)
        for mt in mts_present:
            sl = blocks.get(int(mt))
            if sl is not None:
                sig = np.maximum(sig, diag[sl])
    ratio = np.where(sig > 0, per_bin / np.where(sig > 0, sig, 1.0), np.nan)
    max_over_sigma = (float(np.nanmax(ratio))
                      if np.isfinite(ratio).any() else 0.0)

    out = {
        "max_bin_rel": float(res["max_bin_rel"]),
        "argmax_bin": int(res["argmax_bin"]),
        "max_pointwise_rel": float(res["max_pointwise_rel"]),
        "max_bin_over_sigma": max_over_sigma,
        "n_bins_scored": int(res["n_bins_scored"]),
        "n_bins_uncovered": int(res["n_bins_uncovered"]),
    }
    msg = (
        f"  [INFO] [NUBAR] zaid={zaid}: input sum-rule residual "
        f"|452-(455+456)|/452 = {out['max_bin_rel']:.2e} at bin "
        f"{out['argmax_bin']} ({out['max_bin_over_sigma']:.2f}× the MF31 1σ "
        f"there) over {out['n_bins_scored']} bins the whole family spans"
        + (f", {out['n_bins_uncovered']} not spanned by all three"
           if out["n_bins_uncovered"] else "")
        + f"; rebuilding MT{derived_used} removes it."
    )
    if out["max_bin_over_sigma"] > 0.25:
        log.warning(msg.replace("[INFO]", "[WARN]", 1))
    else:
        log.info(msg)
    return out


def _accumulate_diags(
    acc: Dict[int, Dict[str, float]], diags: Dict[int, dict]
) -> None:
    """Fold one sample's per-MT factor diagnostics into the running summary."""
    for mt, d in (diags or {}).items():
        cur = acc.setdefault(int(mt), {
            "min_factor": float("inf"), "max_factor": float("-inf"),
            "frac_out_of_coverage": 0.0, "n_inserted": 0.0,
        })
        cur["min_factor"] = min(cur["min_factor"], float(d["min_factor"]))
        cur["max_factor"] = max(cur["max_factor"], float(d["max_factor"]))
        cur["frac_out_of_coverage"] = max(
            cur["frac_out_of_coverage"], float(d["frac_out_of_coverage"])
        )
        cur["n_inserted"] = max(cur["n_inserted"], float(d["n_inserted"]))


def _zaid_of(endf_obj) -> int:
    """ZAID from the MF1 nu-bar ZA field of an already-parsed tape."""
    mf1 = endf_obj.get_file(1)
    if mf1 is not None and getattr(mf1, "sections", None):
        for mt in (452, 456, 455, 451):
            sec = mf1.sections.get(mt)
            if sec is not None and getattr(sec, "_za", None) is not None:
                return int(round(sec._za))
    raise RuntimeError("no MF1 ZA field found")


def _write_perturbed_nubar_endf(
    endf_file: str, out_endf: str, out_sections: Dict[int, object]
) -> None:
    """Write ``out_endf`` from ``endf_file`` with MF1 nu-bar MTs replaced.

    Each MT section is replaced sequentially; the writer is re-instantiated per
    MT so it reads the freshly-updated tape. The MF1/MT451 directory is updated
    once at the end (record counts changed by the perturbation).
    """
    shutil.copyfile(endf_file, out_endf)
    for mt in sorted(out_sections):
        writer = ENDFWriter(out_endf)
        ok = writer.replace_mt_section(
            out_sections[mt], mf_number=1,
            output_filepath=out_endf, update_directory=False,
        )
        if not ok:
            raise RuntimeError(f"Failed to replace MF1/MT{mt} in {out_endf}")
    update_mf1_directory(out_endf)


def _normalize_optional_mt_list(
    mt_list: Optional[Union[List[int], List[List[int]]]], n_files: int
) -> List[Optional[List[int]]]:
    """Broadcast an optional MT request to one entry per file (None = all MTs)."""
    if mt_list is None:
        return [None] * n_files
    if mt_list and isinstance(mt_list[0], (list, tuple)):
        if len(mt_list) != n_files:
            raise ValueError(
                f"mt_list has {len(mt_list)} sub-lists but there are {n_files} files"
            )
        return [list(x) for x in mt_list]
    return [list(mt_list)] * n_files
