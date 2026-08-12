import numpy as np
import os
import shutil
import logging
import pandas as pd
import json
import time
import threading
from typing import List, Union, Optional, Dict, Tuple
from multiprocessing import Pool, Manager
from datetime import datetime

from kika.sampling.carrier_blocks import (
    cross_section_carrier_blocks,
    cross_section_carrier_index,
)
from kika.sampling.multigroup_draw import (
    SoftAutofixWarning,
    apply_legacy_autofix,
    draw_relative_factors,
    mark_soft_autofix_survived,
    soft_autofix_missed,
)
from kika.ace.parsers import read_ace
from kika.ace.writers.write_ace import write_ace
from kika._constants import MT_GROUPS
from kika._utils import MeV_to_kelvin
from kika.ace.xsdir import create_xsdir_files_for_ace, _rewrite_xsdir_for_zaids, _read_ace_zaid
from kika.sampling.utils import (
    DualLogger,
    _get_logger,
    _set_logger,
    load_covariance,
    normalize_mt_list,
    _initialize_master_perturbation_matrix,
    _update_master_perturbation_matrix,
    _write_isotope_parquet,
    _merge_isotope_metadata,
    _finalize_master_perturbation_matrix,
)


class QueueLogger:
    """Streaming drop-in for ``DualLogger`` used by parallel dry-run workers.

    Each call pushes a ``(zaid_or_step, level, msg)`` tuple onto a shared
    ``multiprocessing.Queue`` that a daemon thread in the parent drains and
    forwards to the real logger immediately — so progress on long-running
    Higham iterations is visible without waiting for the worker to exit.

    The optional tag (zaid or step number) is stamped onto each record so
    the parent can prefix isotope identity when isotope blocks interleave.
    Pickleable (only stores the queue + a small string tag).
    """

    __slots__ = ("queue", "tag")

    def __init__(self, queue, tag: str = ""):
        self.queue = queue
        self.tag = tag

    def _emit(self, level: str, msg: str):
        try:
            self.queue.put((self.tag, level, str(msg)))
        except Exception:
            # If the queue is closed (parent died), don't take down the worker.
            pass

    def info(self, msg, *args, **kwargs):
        self._emit("info", msg)

    def warning(self, msg, *args, **kwargs):
        self._emit("warning", msg)

    def error(self, msg, *args, **kwargs):
        self._emit("error", msg)

    def debug(self, msg, *args, **kwargs):
        self._emit("debug", msg)


# Module-global slot used by the Pool initializer to hand the shared log
# queue to forked workers. Set in ``_dryrun_worker_init`` and read by
# ``_process_one_isotope_dryrun`` when constructing its QueueLogger.
_DRYRUN_LOG_QUEUE = None
_DRYRUN_BLAS_THREADS = 1


def _dryrun_worker_init(queue, blas_threads: int):
    """Pool initializer — runs once per worker process at startup.

    Stores the shared log queue and the BLAS thread budget into module
    globals so the per-task entry point can pick them up without having
    to pickle the queue with every task.
    """
    global _DRYRUN_LOG_QUEUE, _DRYRUN_BLAS_THREADS
    _DRYRUN_LOG_QUEUE = queue
    _DRYRUN_BLAS_THREADS = max(1, int(blas_threads))


def _drain_log_queue(queue, real_logger, stop_sentinel) -> None:
    """Parent-side daemon thread: drain queue and forward to real logger.

    Runs until it sees ``stop_sentinel`` posted by the parent after the
    Pool has joined. The sentinel goes through Manager-queue pickling,
    so we compare by equality (not identity).
    """
    while True:
        try:
            item = queue.get()
        except (EOFError, OSError):
            return
        if item == stop_sentinel:
            return
        try:
            _tag, level, msg = item
            getattr(real_logger, level, real_logger.info)(msg)
        except Exception:
            # Never let the listener thread die from a malformed record.
            try:
                real_logger.warning(f"[ACE] Malformed log record dropped: {item!r}")
            except Exception:
                pass


def _process_sample(
    ace_file: str,
    sample: np.ndarray,
    sample_index: int,
    energy_grid: List[float],
    mt_numbers: List[int],
    output_dir: str,
    xsdir_file: Optional[str],
    mt_request: Optional[List[int]] = None,
):
    # — read & perturb ACE —
    ace = read_ace(ace_file)
    apply_perturbation_factor_to_ace(
        ace, sample, sample_index, energy_grid, mt_numbers, False,
        mt_request=mt_request,
    )  # Set verbose=False to reduce output

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


def _do_isotope_work(args: dict, log) -> dict:
    """Per-isotope cov-load → autofix → factor sample → parquet write.

    The body is a near-verbatim lift of the per-isotope loop in
    ``perturb_ACE_files`` (lines roughly 320–700 of the legacy sequential
    path), with shared-state writes replaced by local accumulators that
    the caller merges.

    All logging goes through ``log`` — pass a ``BufferedLogger`` for
    parallel workers (records replayed in the parent), or the module
    ``DualLogger`` for the sequential path.

    Returns a dict with keys: ``zaid``, ``summary_fragment``,
    ``skip_reason``, ``warning_increments``, ``metadata_fragment``,
    ``factors_data``. ``skip_reason`` is non-None when the isotope was
    skipped (missing ACE, missing covariance, autofix failure).
    """
    from kika.sampling.multigroup_draw import CovarianceFixError

    ace_file = args["ace_file"]
    cov_file = args["cov_file"]
    mt_list = args["mt_list"]
    num_samples = args["num_samples"]
    space = args["space"]
    decomposition_method = args["decomposition_method"]
    sampling_method = args["sampling_method"]
    psd_method = args["psd_method"]
    autofix = args["autofix"]
    high_val_thresh = args["high_val_thresh"]
    accept_tol = args["accept_tol"]
    max_relative_std = args["max_relative_std"]
    energy_ranges = args["energy_ranges"]
    matrix_dir = args["matrix_dir"]
    remove_blocks = args["remove_blocks"]
    seed = args["seed"]
    verbose = args["verbose"]

    warning_increments = {
        "covariance_load_failed": 0,
        "file_not_found": 0,
        "autofix_warnings": 0,
        "negative_factors": 0,
    }

    if not os.path.exists(ace_file):
        log.error(f"  [ERROR] [ACE] ACE file not found: {ace_file}")
        warning_increments["file_not_found"] += 1
        zaid = os.path.splitext(os.path.basename(ace_file))[0]
        summary_fragment = {
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
                "soft_threshold_warning": False,
            },
            "warnings": [f"ACE file not found: {os.path.basename(ace_file)}"],
        }
        return {
            "zaid": zaid,
            "summary_fragment": summary_fragment,
            "skip_reason": "ACE file not found",
            "warning_increments": warning_increments,
            "metadata_fragment": None,
            "factors_data": None,
        }

    ace = read_ace(ace_file)
    zaid = ace.zaid

    summary_fragment = {
        "ace_file": os.path.basename(ace_file),
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
            "soft_threshold_warning": False,
        },
        "warnings": [],
    }

    cov = load_covariance(cov_file)
    if cov is None:
        log.error(f"  [ERROR] [ACE] Unable to load a valid covariance matrix for {cov_file}")
        warning_increments["covariance_load_failed"] += 1
        summary_fragment["warnings"].append(
            f"No valid covariance file found: {os.path.basename(cov_file)}"
        )
        return {
            "zaid": zaid,
            "summary_fragment": summary_fragment,
            "skip_reason": "No valid covariance file found",
            "warning_increments": warning_increments,
            "metadata_fragment": None,
            "factors_data": None,
        }

    log.info(f"  [INFO] [ACE] Covariance file: {cov_file}")
    log.info(f"  [INFO] [ACE] Isotope: {zaid}")

    if cov.num_matrices == 0:
        log.info(f"  [WARN] [ACE] No covariance found in {zaid}. Skipping.")
        summary_fragment["warnings"].append("No covariance data found in matrix")
        return {
            "zaid": zaid,
            "summary_fragment": summary_fragment,
            "skip_reason": "No covariance data found in matrix",
            "warning_increments": warning_increments,
            "metadata_fragment": None,
            "factors_data": None,
        }

    if remove_blocks and zaid in remove_blocks:
        log.info(f"  [INFO] [ACE] Applying user-specified block removals for isotope {zaid}")
        removal_spec = remove_blocks[zaid]
        if isinstance(removal_spec, tuple):
            blocks_to_remove = [removal_spec]
        else:
            blocks_to_remove = list(removal_spec)
        log.info(f"  Blocks to remove: {blocks_to_remove}")
        reactions_before = cov.reactions_by_isotope(zaid)
        log.info(f"  Reactions before removal: {sorted(reactions_before)}")
        try:
            cov = cov.remove_matrix(isotope=zaid, reaction_pairs=blocks_to_remove)
            reactions_after = cov.reactions_by_isotope(zaid)
            log.info(f"  Reactions after removal:  {sorted(reactions_after)}")
            removed_reactions = set(reactions_before) - set(reactions_after)
            for mt in removed_reactions:
                summary_fragment["removed_mts"][mt] = (
                    f"User-specified block removal: {blocks_to_remove}"
                )
            if removed_reactions:
                log.info(f"  Successfully removed reactions: {sorted(removed_reactions)}")
            else:
                log.info("  No reactions were removed (blocks may not have existed)")
        except Exception as e:
            log.error(f"  [ERROR] [ACE] Failed to remove blocks {blocks_to_remove}: {str(e)}")
            summary_fragment["warnings"].append(
                f"Failed to remove user-specified blocks: {str(e)}"
            )

    cov = cov.clean_cov(zaid)
    mt_in_cov = cov.reactions_by_isotope(zaid)
    summary_fragment["mt_in_cov"] = sorted(mt_in_cov)
    mt_in_ace = sorted(set(ace.mt_numbers))

    mt_positive = [mt for mt in mt_list if mt > 0]
    mt_negative = [abs(mt) for mt in mt_list if mt < 0]

    expanded_mt_negative = set(mt_negative)
    for excluded_mt in mt_negative:
        for single, rng in MT_GROUPS:
            if single == excluded_mt:
                expanded_mt_negative.update(rng)
                break

    if len(mt_positive) == 0:
        mt_request = mt_in_ace
    else:
        mt_request = sorted(mt_positive)

    if mt_negative:
        if len(expanded_mt_negative) > len(mt_negative):
            log.info(f"  [INFO] [ACE] Excluding MTs: {sorted(mt_negative)}")
            additional = sorted(expanded_mt_negative - set(mt_negative))
            if additional:
                log.info(f"  [INFO] [ACE] Also excluding associated range MTs: {additional}")
        else:
            log.info(f"  [INFO] [ACE] Excluding MTs: {sorted(mt_negative)}")
        mt_request = [mt for mt in mt_request if mt not in expanded_mt_negative]

    original_mt_perturb = set(mt_in_cov) & set(mt_request) & set(mt_in_ace)
    mt_perturb = set(mt_in_cov) & set(mt_request) & set(mt_in_ace)
    group_removed_mts = {}

    for single, rng in MT_GROUPS:
        cov_has_single = single in mt_in_cov
        cov_in_range = [mt for mt in mt_in_cov if mt in rng]
        list_has_single = single in mt_request
        list_in_range = [mt for mt in mt_request if mt in rng]
        ace_in_range = [mt for mt in mt_in_ace if mt in rng]
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

        if cov_in_range:
            if (list_has_single or list_in_range) and ace_in_range and not single_is_excluded:
                triple = set(cov_in_range) & set(mt_request) & set(mt_in_ace)
                removed_range = set(cov_in_range) - triple
                for mt in removed_range:
                    if mt in original_mt_perturb:
                        if mt in expanded_mt_negative:
                            group_removed_mts[mt] = f"MT={mt} excluded because summary MT={single} was negatively specified"
                        else:
                            group_removed_mts[mt] = f"MT not in triple intersection for range associated with MT={single}"
                mt_perturb.update(triple)
                to_remove = [(mt, 0) for mt in cov_in_range if mt not in triple]
                if to_remove:
                    cov = cov.remove_matrix(zaid, to_remove)
            else:
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

    for mt, reason in group_removed_mts.items():
        summary_fragment["removed_mts"][mt] = reason

    mt_perturb = sorted(mt_perturb)
    remaining = cov.reactions_by_isotope(zaid)
    to_remove = [(mt, 0) for mt in remaining if mt not in mt_perturb]
    if to_remove:
        cov = cov.remove_matrix(zaid, to_remove)

    # The cov-mapped `mt_perturb` is what sampling operates on (its length
    # × n_groups is the sampling dimensionality, reported as `mt_count`
    # below). At ACE-write time each summary MT in `mt_perturb` expands
    # onto its ACE partials per `_partials_to_apply`, which can be
    # restricted to a user-requested subset. Print the *actual* perturbed
    # partials so the log reflects what gets touched in the ACE file, not
    # just the covariance index.
    _ace_partials_perturbed = []
    for _mt in mt_perturb:
        if _mt in _SUMMARY_MT_PARTIAL_RANGES:
            _ace_partials_perturbed.extend(
                _partials_to_apply(
                    _mt, _SUMMARY_MT_PARTIAL_RANGES[_mt],
                    set(mt_in_ace), mt_request,
                )
            )
        else:
            _ace_partials_perturbed.append(_mt)
    _ace_partials_perturbed = sorted(set(_ace_partials_perturbed))

    log.info("  [INFO] [ACE] MT selection:")
    log.info(f"    MTs in ACE file:          {mt_in_ace}")
    log.info(f"    MTs in covariance matrix: {mt_in_cov}")
    log.info(f"    MTs to be perturbed:      {_ace_partials_perturbed}")
    if _ace_partials_perturbed != list(mt_perturb):
        log.info(f"      (sampled on cov MTs {mt_perturb}; summary MTs expanded onto ACE partials)")
    log.info(f">> mt_count = {len(mt_perturb)}")

    energy_grid_eV = cov.energy_grid
    if cov.energy_unit == 'eV':
        energy_grid = [e / 1e6 for e in energy_grid_eV]
        log.info("  [INFO] [ACE] Converted energy grid from eV to MeV for ACE compatibility")
    elif cov.energy_unit == 'MeV':
        energy_grid = energy_grid_eV
    else:
        raise ValueError(f"Unknown energy unit '{cov.energy_unit}' in covariance matrix")

    mt_thresholds: Dict[int, float] = {}
    threshold_source: Dict[int, str] = {}
    # Track MTs whose σ(E) was *consulted* in ACE (regardless of whether a
    # threshold was extracted). An MT with σ(E) data but nz[0]==0 (open at
    # the first ACE energy) is a no-threshold reaction (elastic, capture,
    # etc.) — those must NOT receive a cov-diag fallback, otherwise we
    # would invent a fake threshold for a reaction that has none and the
    # subthreshold mask would later drop real physical bins.
    mt_consulted_in_ace: set = set()
    if ace.cross_section is not None:
        for mt in mt_perturb:
            reac = ace.cross_section.reaction.get(mt)
            if reac is None:
                continue
            xs_arr = np.asarray(reac.xs_values, dtype=float)
            e_arr = np.asarray(reac.energies, dtype=float)
            if xs_arr.size == 0 or e_arr.size == 0:
                continue
            mt_consulted_in_ace.add(int(mt))
            nz = np.where(xs_arr > 0.0)[0]
            if nz.size > 0 and nz[0] > 0:
                mt_thresholds[int(mt)] = float(e_arr[nz[0]])
                threshold_source[int(mt)] = "ace"

    # Fallback ONLY for MTs whose σ(E) is genuinely absent from ACE (e.g.,
    # MT=108 in some libraries). Use the cov-diagonal: take the bin with
    # the largest σ² in the MT block and treat its lower edge as the proxy
    # threshold — NJOY threshold artifacts manifest as a single high-σ²
    # spike in the threshold-spanning bin.
    cov_diag_full = np.diag(cov.covariance_matrix)
    bins_arr_mev = np.asarray(energy_grid, dtype=float)
    cov_param_pairs = cov._get_param_pairs()
    cov_num_groups = cov.num_groups
    for pair_idx, (z, mt) in enumerate(cov_param_pairs):
        mt_int = int(mt)
        if (int(z) != int(zaid)
                or mt_int in mt_thresholds
                or mt_int in mt_consulted_in_ace):
            continue
        block = cov_diag_full[pair_idx * cov_num_groups:(pair_idx + 1) * cov_num_groups]
        finite_pos = np.isfinite(block) & (block > 0)
        if not finite_pos.any():
            continue
        g_max = int(np.argmax(np.where(finite_pos, block, -np.inf)))
        if g_max < len(bins_arr_mev) - 1:
            mt_thresholds[mt_int] = float(bins_arr_mev[g_max])
            threshold_source[mt_int] = "cov-diag"

    log.info(
        f"  [INFO] [ACE] Extracted {len(mt_thresholds)} reaction threshold(s) "
        f"from ACE σ(E) for NJOY artifact detection"
    )
    for mt, e in sorted(mt_thresholds.items()):
        src = threshold_source.get(mt, "ace")
        tag = "" if src == "ace" else f"  [{src}]"
        log.info(f"    MT={mt}: E_threshold = {e:.4e} MeV{tag}")

    pre_autofix_mts = list(mt_perturb)

    # The repairs run ahead of the draw, as a plan, instead of inside the
    # sampler -- `draw_relative_factors` reproduces `generate_samples`'
    # sequence in its order and is gated bit-for-bit against it.
    try:
        cov, mt_perturb_final, fix_info = apply_legacy_autofix(
            cov, autofix,
            mt_numbers=mt_perturb,
            high_val_thresh=high_val_thresh,
            accept_tol=accept_tol,
            verbose=verbose,
            logger=log,
        )
        soft_missed = soft_autofix_missed(autofix, fix_info)
        (block_key, joint), = cross_section_carrier_blocks(cov)
        (_key, block_index), = cross_section_carrier_index(cov).items()
        try:
            factors, draw_info = draw_relative_factors(
                joint,
                num_samples,
                key=block_key,
                pairs=block_index["pairs"],
                stride=block_index["stride"],
                bins=energy_grid,
                space=space,
                decomposition_method=decomposition_method,
                sampling_method=sampling_method,
                seed=None if seed is None else seed + zaid,
                psd_method=psd_method,
                max_relative_std=max_relative_std,
                mt_thresholds=mt_thresholds,
                verbose=verbose,
                logger=log,
                label=str(zaid),
            )
        except Exception as exc:
            # What `generate_samples` did with its `soft_autofix_failed` flag:
            # a decomposition that fails after soft autofix missed its
            # threshold is diagnosed as that, so the isotope is skipped with a
            # reason rather than taking the run down.
            if soft_missed:
                min_eigenvalue = fix_info.get("min_eigenvalue", float("nan"))
                raise SoftAutofixWarning(
                    f"Soft autofix failed to meet threshold "
                    f"(λ_min={min_eigenvalue:.4e} < {accept_tol:.4e}) and "
                    f"decomposition failed: {exc}"
                ) from exc
            raise
        if soft_missed:
            mark_soft_autofix_survived(fix_info)
        log.info(
            f"  [INFO] [ACE] zaid={zaid}: conditioning plan = "
            f"{[s.remedy for s in draw_info['plan'].steps] or ['none']}, "
            f"{draw_info['n_inert_dropped']} inert bin(s) dropped"
        )
    except SoftAutofixWarning as e:
        log.error(f"  [ERROR] [ACE] Soft autofix warning for isotope {zaid}")
        log.error(f"    Error details: {str(e)}")
        warning_increments["autofix_warnings"] += 1
        summary_fragment["warnings"].append(
            "Soft autofix failed to meet eigenvalue threshold and decomposition failed"
        )
        summary_fragment["autofix_info"]["converged"] = False
        summary_fragment["autofix_info"]["soft_threshold_warning"] = True
        return {
            "zaid": zaid,
            "summary_fragment": summary_fragment,
            "skip_reason": str(e),
            "warning_increments": warning_increments,
            "metadata_fragment": None,
            "factors_data": None,
        }
    except CovarianceFixError as e:
        log.error(f"  [ERROR] [ACE] Covariance matrix fixing failed for isotope {zaid}")
        log.error(f"    Error details: {str(e)}")
        log.error(
            f"    Suggestion: Try processing isotope {zaid} separately with "
            "autofix='medium' or 'hard'"
        )
        warning_increments["autofix_warnings"] += 1
        summary_fragment["warnings"].append(
            "Covariance matrix could not be fixed to meet eigenvalue threshold"
        )
        summary_fragment["autofix_info"]["converged"] = False
        return {
            "zaid": zaid,
            "summary_fragment": summary_fragment,
            "skip_reason": str(e),
            "warning_increments": warning_increments,
            "metadata_fragment": None,
            "factors_data": None,
        }

    if autofix is not None and fix_info is not None:
        if fix_info.get("soft_autofix_failed", False):
            summary_fragment["autofix_info"]["soft_threshold_warning"] = True
            min_eigenvalue = fix_info.get("min_eigenvalue", float('nan'))
            summary_fragment["warnings"].append(
                f"Soft autofix failed to meet eigenvalue threshold "
                f"(λ_min={min_eigenvalue:.4e} < {accept_tol:.4e}) but decomposition succeeded"
            )
        removed_in_autofix = set(pre_autofix_mts) - set(mt_perturb_final if mt_perturb_final else [])
        for mt in removed_in_autofix:
            summary_fragment["removed_mts"][mt] = (
                f"Removed during covariance autofix (level='{autofix}')"
            )
            summary_fragment["autofix_info"]["removed_mts"].append(mt)
        if fix_info.get("removed_correlations"):
            summary_fragment["autofix_info"]["removed_correlations"] = fix_info["removed_correlations"]
        if fix_info.get("removed_pairs"):
            summary_fragment["autofix_info"]["removed_pairs"] = fix_info["removed_pairs"]

    if energy_ranges is not None and factors is not None and mt_perturb_final:
        factors = _mask_factors_by_energy_range_ace(
            factors, mt_perturb_final, energy_grid, energy_ranges, space
        )
        log.info(f"  [INFO] [ACE] Applied energy range masking for {zaid}")

    summary_fragment["mt_perturbed"] = mt_perturb_final if mt_perturb_final else []
    if factors is not None:
        log.info(f">> factors_shape = {factors.shape}")

    factors_data = None
    metadata_fragment = None
    if mt_perturb_final and factors is not None:
        factors_data = {
            "factors": factors.astype(np.float32),
            "mt_numbers": mt_perturb_final,
            "energy_grid": energy_grid,
        }
        metadata_fragment = _write_isotope_parquet(
            matrix_dir, zaid, factors, mt_perturb_final, energy_grid,
            verbose=verbose, logger=log,
        )

    return {
        "zaid": zaid,
        "summary_fragment": summary_fragment,
        "skip_reason": None,
        "warning_increments": warning_increments,
        "metadata_fragment": metadata_fragment,
        "factors_data": factors_data,
    }


def _process_one_isotope_dryrun(args: dict) -> dict:
    """Top-level worker for parallel dry-run mode. Pickleable.

    Wraps ``_do_isotope_work`` with a per-worker streaming ``QueueLogger``,
    timing, step header/footer, and a try/except so one bad isotope
    doesn't kill the pool. Log records flow to the parent listener thread
    in real time so progress is visible during long Higham iterations.
    """
    import traceback

    # 1) Cap BLAS threads inside this worker. Without a cap, NumPy/SciPy
    # spin up one thread per core per worker; with N workers in the pool
    # that oversubscribes the machine and every Higham iteration thrashes.
    # The parent picks the budget (cores / workers); we just enforce it.
    try:
        from threadpoolctl import threadpool_limits
        _tp = threadpool_limits(_DRYRUN_BLAS_THREADS)
    except ImportError:
        _tp = None

    # 2) Replace the inherited module-level DualLogger with a streaming
    # QueueLogger. generate_samples and the cov-decomposition layer call
    # _get_logger() to find the active logger; without this swap, all
    # workers would race on the parent's DualLogger file handle.
    step_num = args["step_num"]
    log = QueueLogger(_DRYRUN_LOG_QUEUE, tag=f"step{step_num}")
    _set_logger(log)

    ace_file = args["ace_file"]
    step_t0 = time.time()

    log.info(
        f"\n#-- STEP {step_num}: Process isotope from "
        f"{os.path.basename(ace_file)} ------------------------------------------------"
    )

    zaid = None
    summary_fragment = None
    skip_reason = None
    warning_increments = {
        "covariance_load_failed": 0,
        "file_not_found": 0,
        "autofix_warnings": 0,
        "negative_factors": 0,
    }
    metadata_fragment = None
    factors_data = None
    exception_str = None

    try:
        result = _do_isotope_work(args, log)
        zaid = result["zaid"]
        summary_fragment = result["summary_fragment"]
        skip_reason = result["skip_reason"]
        warning_increments = result["warning_increments"]
        metadata_fragment = result["metadata_fragment"]
        factors_data = result["factors_data"]
    except Exception:
        exception_str = traceback.format_exc()
        log.error(f"[ACE] Worker exception in step {step_num}:\n{exception_str}")

    elapsed = time.time() - step_t0
    log.info(
        f"#-- END STEP {step_num} (elapsed: {elapsed:.1f}s) -------------------------------------"
    )

    return {
        "zaid": zaid,
        "ace_basename": os.path.basename(ace_file),
        "step_num": step_num,
        "elapsed": elapsed,
        "summary_fragment": summary_fragment,
        "skip_reason": skip_reason,
        "warning_increments": warning_increments,
        "metadata_fragment": metadata_fragment,
        "factors_data": factors_data,
        "exception": exception_str,
    }


def perturb_ACE_files(
    ace_files: Union[str, List[str]],
    cov_files: Union[str, List[str]],
    mt_list: Union[List[int], List[List[int]]],
    num_samples: int,
    space: str = "log",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    output_dir: str = '.',
    xsdir_file: Optional[str] = None,
    seed: Optional[int] = None,
    nprocs: int = 1,
    dry_run: bool = False,
    only_xsdir: bool = False,
    autofix: Optional[str] = None,
    high_val_thresh: float = 1.0,
    accept_tol: float = -1.0e-4,
    remove_blocks: Optional[Dict[int, Union[Tuple[int, int], List[Tuple[int, int]]]]] = None,
    verbose: bool = True,
    energy_ranges: Optional[List[Tuple[float, float]]] = None,
    psd_method: str = "auto",
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
    mt_list : List[int] or List[List[int]]
        MT reactions to perturb. Either a flat list (applied to every ACE file)
        or a list-of-lists with one entry per ACE file (per-file selection).
        An empty (or inner-empty) list means "perturb all MTs available in the
        covariance" for that file.
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

    # Normalize MT input to a per-file list-of-lists. Accepts either a flat
    # ``List[int]`` (broadcast to every ACE file) or a pre-expanded
    # ``List[List[int]]`` (one list per ACE file).
    mt_lists = normalize_mt_list(mt_list, len(ace_files), unit_label="ACE files")

    # Format MT list(s) for the log header
    def _fmt_mts(mts):
        if not mts:
            return "All available MTs"
        if len(mts) <= 10:
            return ", ".join(str(mt) for mt in mts)
        return f"{len(mts)} MTs: {', '.join(str(mt) for mt in mts[:5])}..."

    unique_mt_lists = {tuple(sub) for sub in mt_lists}
    if len(unique_mt_lists) <= 1:
        formatted_mt = _fmt_mts(mt_lists[0])
    else:
        formatted_mt = "per-file (see below)"
    
    # Print all parameters TO LOG FILE
    _logger.info(f"  ACE_FILES = {formatted_ace}")
    _logger.info(f"  COV_FILES = {formatted_cov}")
    _logger.info(f"  MT_NUMBERS = {formatted_mt}")
    if len(unique_mt_lists) > 1:
        for ace_path, mts in zip(ace_files, mt_lists):
            _logger.info(f"    - {os.path.basename(ace_path)}: {_fmt_mts(mts)}")
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

    if only_xsdir:
        if dry_run:
            raise ValueError("only_xsdir and dry_run cannot both be True")
        _logger.info(
            "[INFO] [ACE] only_xsdir=True — skipping sampling/ACE writing; "
            "rewrapping xsdir entries for the requested isotopes only.",
            console=True,
        )
        zaids = []
        for af in ace_files:
            try:
                zaids.append(_read_ace_zaid(af))
            except Exception as e:
                _logger.error(f"  [ERROR] [ACE] Could not read zaid from {af}: {e}")
        count = _rewrite_xsdir_for_zaids(
            zaids=zaids,
            output_dir=output_dir,
            xsdir_file=xsdir_file,
            logger=_logger,
        )
        _logger.info(f"[INFO] [ACE] only_xsdir complete: {count} file(s) modified", console=True)
        return

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

    # =====================================================================
    #  PARALLEL DRY-RUN (one CPU per isotope, gated on dry_run + nprocs > 1)
    # =====================================================================
    use_parallel_dryrun = dry_run and nprocs > 1 and len(ace_files) > 1
    if use_parallel_dryrun:
        workers = min(nprocs, len(ace_files))
        # Split the user's nprocs budget across workers for BLAS threading.
        # Higham does many eigendecompositions on large dense matrices —
        # giving each worker a couple of BLAS threads makes those eigh
        # calls noticeably faster than the previous hardcoded 1.
        blas_threads = max(1, nprocs // workers)
        _logger.info(
            f"  [INFO] [ACE] Dry-run parallel: {workers} workers across "
            f"{len(ace_files)} isotopes ({blas_threads} BLAS thread(s)/worker)",
            console=True,
        )
        tasks = [
            {
                "ace_file": af,
                "cov_file": cf,
                "mt_list": ml,
                "num_samples": num_samples,
                "space": space,
                "decomposition_method": decomposition_method,
                "sampling_method": sampling_method,
                "psd_method": psd_method,
                "autofix": autofix,
                "high_val_thresh": high_val_thresh,
                "accept_tol": accept_tol,
                "max_relative_std": max_relative_std,
                "energy_ranges": energy_ranges,
                "matrix_dir": matrix_dir,
                "remove_blocks": remove_blocks,
                "seed": seed,
                "step_num": idx + 1,
                "verbose": verbose,
            }
            for idx, (af, cf, ml) in enumerate(zip(ace_files, cov_files, mt_lists))
        ]
        metadata_fragments = []

        # Manager-backed queue is picklable across the Pool's fork boundary
        # and survives worker churn; a plain mp.Queue would also work on
        # Linux fork but Manager keeps the API identical on spawn-based
        # platforms. The listener thread drains records in real time and
        # forwards each one to the same DualLogger the sequential path uses.
        with Manager() as manager:
            log_queue = manager.Queue()
            stop_sentinel = "__ACE_LOG_STOP__"
            listener = threading.Thread(
                target=_drain_log_queue,
                args=(log_queue, _logger, stop_sentinel),
                name="ACEDryRunLogListener",
                daemon=True,
            )
            listener.start()

            try:
                with Pool(
                    processes=workers,
                    initializer=_dryrun_worker_init,
                    initargs=(log_queue, blas_threads),
                ) as pool:
                    for result in pool.imap_unordered(_process_one_isotope_dryrun, tasks):
                        if result.get("exception"):
                            # Worker already streamed the traceback through
                            # the queue; here we just record the failure.
                            _logger.error(
                                f"  [ERROR] [ACE] Worker for {result.get('ace_basename')} "
                                f"raised an unexpected exception (see log above)"
                            )
                            continue

                        zaid = result["zaid"]
                        if result["summary_fragment"] is not None and zaid is not None:
                            summary_data[zaid] = result["summary_fragment"]
                        if result.get("skip_reason") and zaid is not None:
                            skipped_isotopes[zaid] = result["skip_reason"]
                        for k, v in result["warning_increments"].items():
                            warning_counts[k] += v
                        if result.get("metadata_fragment"):
                            metadata_fragments.append(result["metadata_fragment"])
                        if result.get("factors_data") is not None and zaid is not None:
                            all_factors_data[zaid] = result["factors_data"]
            finally:
                # Tell the listener to stop and wait briefly for it to drain.
                try:
                    log_queue.put(stop_sentinel)
                except Exception:
                    pass
                listener.join(timeout=5.0)

        _merge_isotope_metadata(matrix_dir, metadata_fragments)
        _logger.info(
            f"  [INFO] [ACE] Dry-run parallel complete: "
            f"{len(summary_data)} isotopes processed, "
            f"{len(skipped_isotopes)} skipped",
            console=True,
        )

    for i, (ace_file, cov_file, mt_list) in enumerate(zip(ace_files, cov_files, mt_lists)):
        # Skip the sequential body when the parallel path already did the work
        if use_parallel_dryrun:
            continue

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

        # The cov-mapped `mt_perturb` is what sampling operates on (its
        # length × n_groups is the sampling dimensionality, reported as
        # `mt_count`). At ACE-write time each summary MT in `mt_perturb`
        # expands onto its ACE partials via `_partials_to_apply`, which
        # can be restricted to a user-requested subset. Print the *actual*
        # perturbed partials so the log reflects what gets touched in the
        # ACE file, not just the covariance index.
        _ace_partials_perturbed = []
        for _mt in mt_perturb:
            if _mt in _SUMMARY_MT_PARTIAL_RANGES:
                _ace_partials_perturbed.extend(
                    _partials_to_apply(
                        _mt, _SUMMARY_MT_PARTIAL_RANGES[_mt],
                        set(mt_in_ace), mt_request,
                    )
                )
            else:
                _ace_partials_perturbed.append(_mt)
        _ace_partials_perturbed = sorted(set(_ace_partials_perturbed))

        _logger.info(f"  [INFO] [ACE] MT selection:")
        _logger.info(f"    MTs in ACE file:          {mt_in_ace}")
        _logger.info(f"    MTs in covariance matrix: {mt_in_cov}")
        _logger.info(f"    MTs to be perturbed:      {_ace_partials_perturbed}")
        if _ace_partials_perturbed != list(mt_perturb):
            _logger.info(f"      (sampled on cov MTs {mt_perturb}; summary MTs expanded onto ACE partials)")
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

        # Extract per-MT reaction thresholds from ACE σ(E) (in MeV).
        # Used to flag NJOY threshold-spanning bins so their variance can be
        # rescaled to the per-MT median before PSD projection.
        mt_thresholds: Dict[int, float] = {}
        threshold_source: Dict[int, str] = {}
        # See _run_dry_one for why we track ACE-consulted MTs separately:
        # no-threshold reactions (elastic, capture) have ACE σ(E) but with
        # nz[0]==0; the cov-diag fallback must NOT fire for them.
        mt_consulted_in_ace: set = set()
        if ace.cross_section is not None:
            for mt in mt_perturb:
                reac = ace.cross_section.reaction.get(mt)
                if reac is None:
                    continue
                xs_arr = np.asarray(reac.xs_values, dtype=float)
                e_arr = np.asarray(reac.energies, dtype=float)
                if xs_arr.size == 0 or e_arr.size == 0:
                    continue
                mt_consulted_in_ace.add(int(mt))
                nz = np.where(xs_arr > 0.0)[0]
                # nz[0] == 0 means the reaction is open at the first ACE energy
                # (no threshold inside the union grid) — skip.
                if nz.size > 0 and nz[0] > 0:
                    mt_thresholds[int(mt)] = float(e_arr[nz[0]])
                    threshold_source[int(mt)] = "ace"

        # Fallback ONLY for MTs whose σ(E) is genuinely absent from ACE.
        cov_diag_full = np.diag(cov.covariance_matrix)
        bins_arr_mev = np.asarray(energy_grid, dtype=float)
        cov_param_pairs = cov._get_param_pairs()
        cov_num_groups = cov.num_groups
        for pair_idx, (z, mt) in enumerate(cov_param_pairs):
            mt_int = int(mt)
            if (int(z) != int(zaid)
                    or mt_int in mt_thresholds
                    or mt_int in mt_consulted_in_ace):
                continue
            block = cov_diag_full[pair_idx * cov_num_groups:(pair_idx + 1) * cov_num_groups]
            finite_pos = np.isfinite(block) & (block > 0)
            if not finite_pos.any():
                continue
            g_max = int(np.argmax(np.where(finite_pos, block, -np.inf)))
            if g_max < len(bins_arr_mev) - 1:
                mt_thresholds[mt_int] = float(bins_arr_mev[g_max])
                threshold_source[mt_int] = "cov-diag"

        _logger.info(
            f"  [INFO] [ACE] Extracted {len(mt_thresholds)} reaction threshold(s) "
            f"from ACE σ(E) for NJOY artifact detection"
        )
        for mt, e in sorted(mt_thresholds.items()):
            src = threshold_source.get(mt, "ace")
            tag = "" if src == "ace" else f"  [{src}]"
            _logger.info(f"    MT={mt}: E_threshold = {e:.4e} MeV{tag}")

        # Save pre-autofix MT list
        pre_autofix_mts = list(mt_perturb)

        # The repairs run ahead of the draw, as a plan -- see the migrated
        # call site in ``_do_isotope_work`` for why.
        try:
            cov, mt_perturb_final, fix_info = apply_legacy_autofix(
                cov, autofix,
                mt_numbers=mt_perturb,
                high_val_thresh=high_val_thresh,
                accept_tol=accept_tol,
                verbose=verbose,
                logger=_logger,
            )
            soft_missed = soft_autofix_missed(autofix, fix_info)
            (block_key, joint), = cross_section_carrier_blocks(cov)
            (_key, block_index), = cross_section_carrier_index(cov).items()
            try:
                factors, draw_info = draw_relative_factors(
                    joint,
                    num_samples,
                    key                  = block_key,
                    pairs                = block_index["pairs"],
                    stride               = block_index["stride"],
                    bins                 = energy_grid,
                    space                = space,
                    decomposition_method = decomposition_method,
                    sampling_method      = sampling_method,
                    seed                 = None if seed is None else seed + zaid,
                    psd_method           = psd_method,
                    max_relative_std     = max_relative_std,
                    mt_thresholds        = mt_thresholds,
                    verbose              = verbose,
                    logger               = _logger,
                    label                = str(zaid),
                )
            except Exception as exc:
                if soft_missed:
                    min_eigenvalue = fix_info.get("min_eigenvalue", float("nan"))
                    raise SoftAutofixWarning(
                        f"Soft autofix failed to meet threshold "
                        f"(λ_min={min_eigenvalue:.4e} < {accept_tol:.4e}) and "
                        f"decomposition failed: {exc}"
                    ) from exc
                raise
            if soft_missed:
                mark_soft_autofix_survived(fix_info)
            _logger.info(
                f"  [INFO] [ACE] zaid={zaid}: conditioning plan = "
                f"{[s.remedy for s in draw_info['plan'].steps] or ['none']}, "
                f"{draw_info['n_inert_dropped']} inert bin(s) dropped"
            )
        except Exception as e:
            from kika.sampling.multigroup_draw import CovarianceFixError

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

        tasks = [(ace_file, factors[j], j, energy_grid, mt_perturb_final, output_dir, xsdir_file, mt_request) for j in range(num_samples)]

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


# Map from a summary MT (as stored in the covariance) to the range of
# partial MTs it lumps in the ACE file. Used by
# apply_perturbation_factor_to_ace to expand summary-MT factors onto the
# partials. Centralized here so the same table drives both the default
# (expand to all partials in ACE) and the restricted (user-requested
# subset) behavior.
_SUMMARY_MT_PARTIAL_RANGES = {
    4:   range(51,  92),   # total inelastic = MT 50-91
    103: range(600, 650),  # (n,p)
    104: range(650, 700),  # (n,d)
    105: range(700, 750),  # (n,t)
    106: range(750, 800),  # (n,3He)
    107: range(800, 850),  # (n,α)
}


def _partials_to_apply(summary_mt, partial_range, ace_mts, mt_request):
    """Return the ACE partial MTs that a summary-MT factor should be applied to.

    Default (``mt_request`` is None, or user requested the summary MT itself,
    or requested no partial in the range): apply to *every* partial in
    ``partial_range`` that exists in the ACE file. This is the conventional
    interpretation of a lumped-MT covariance: the same relative perturbation
    is shared across all partials in the group (fully correlated levels).

    Restricted (user requested only specific partials in the range, e.g.
    ``mt_request = [51, 52]`` for the MT 4 group, and did *not* request the
    summary MT itself): apply only to the requested partials. This makes
    the MC sample exactly the same reaction scope as a sandwich pipeline
    whose ``PERT`` cards cover only those partials. Note: the resulting
    sample under-propagates the lumped-MT covariance, because the other
    partials in the group are left at their nominal values.
    """
    all_in_ace = [m for m in partial_range if m in ace_mts]
    if mt_request is None:
        return all_in_ace
    if summary_mt in mt_request:
        # User asked for the lumped MT explicitly → standard behavior.
        return all_in_ace
    explicit = [m for m in mt_request if m in partial_range]
    if explicit:
        return [m for m in explicit if m in ace_mts]
    return all_in_ace


def apply_perturbation_factor_to_ace(
    ace, sample, sample_index, energy_grid, mt_numbers, verbose=True,
    mt_request=None,
):
    """Apply per-group perturbation factors to ACE, and list which MTs were actually perturbed.

    Parameters
    ----------
    ace, sample, sample_index, energy_grid, mt_numbers, verbose
        Standard arguments. ``mt_numbers`` is the cov-mapped list (e.g.
        ``[2, 4, 102]`` even when the user originally asked for
        ``[2, 51, 52, 102]``).
    mt_request : list[int] or None, optional
        The user's *original* positive MT request, before the
        cov-mapping step collapsed partials onto summary MTs. When
        provided, a summary MT's factor is applied only to the partials
        the user explicitly named (see :func:`_partials_to_apply` for
        the full rule). ``None`` preserves the historical behavior of
        expanding every summary MT to its full partial range.
    """
    logger = _get_logger()

    # Convert sample to float32
    sample = sample.astype(np.float32)

    n_groups = len(energy_grid) - 1
    if sample.shape[0] != len(mt_numbers) * n_groups:
        raise ValueError(f"sample length {sample.shape[0]} ≠ {len(mt_numbers)}×{n_groups}")

    boundaries = np.asarray(energy_grid)
    perturbed_mts = []  # collect the actual MT numbers
    ace_mts_set = set(ace.mt_numbers)

    for mt_idx, mt in enumerate(mt_numbers):
        start = mt_idx * n_groups
        end   = start + n_groups
        factors = sample[start:end]

        if mt in _SUMMARY_MT_PARTIAL_RANGES:
            partial_range = _SUMMARY_MT_PARTIAL_RANGES[mt]
            targets = _partials_to_apply(mt, partial_range, ace_mts_set, mt_request)
            for partial_mt in targets:
                _apply_factors_to_mt(ace, partial_mt, factors, boundaries, verbose)
                perturbed_mts.append(partial_mt)
        else:
            # direct mt
            if mt in ace_mts_set:
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