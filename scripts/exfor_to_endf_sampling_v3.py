"""EXFOR → ENDF MF34 pipeline (v3) with MF3↔MF34 cross-correlations.

Per-bin angular-distribution Legendre fits with joint Monte Carlo over
EXFOR data and MF33 cross-section perturbations. Outputs an MF34 file with
the standard ``Cov(a_l, a_l')`` for ``l, l' >= 1`` plus l=0 cross-correlation
rows holding ``Cov(δσ, a_l)`` propagated from MF33.

This is a parallel pipeline to ``exfor_to_endf_sampling_v2.py``; v2 stays at
"Level 2" (point-wise σ_total fitting; per-experiment normalization shifts
deliberately distort c_l so MF34 captures their variance). v3 adds a
principled MF3 propagation channel by drawing δσ from MF33 covariance and
threading it through the AD fit as a multiplicative data perturbation.

Design philosophy: all reused-function parameters from v2 stay configurable
at the top of this file (running v3 with new inputs is one config-block edit,
not source surgery). Only v2 research toggles that v3 doesn't exercise are
hard-coded inside.
"""
from __future__ import annotations

import os
import shutil
import sys
import time

# Pin BLAS threads before numpy import — avoids oversubscription with Pool.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool

import numpy as np
import pandas as pd

_kika_path = Path(__file__).parent.parent
if str(_kika_path) not in sys.path:
    sys.path.insert(0, str(_kika_path))

# ENDF I/O
from kika.endf.read_endf import read_endf
from kika.endf.writers import create_mf34_from_covariance, write_mf34_to_file

# Cross sections + covariance utilities
from kika.cov.decomposition import cholesky_decomposition

# EXFOR
from kika.exfor import read_all_exfor

# Cross-section + derived-covariance orchestration. ``_build_xs_map`` is the
# canonical helper used by kika-app to assemble an MT→σ(E) map with
# resonance-region reconstruction (NJOY when an executable is provided,
# kika's in-Python reconstructor otherwise). Reusing it keeps v3 in lockstep
# with the app's MF33 reconstruction logic — including NC LTY=0 sum-rule
# resolution for derived MTs (e.g. JENDL-5 Fe-56 MT2 = MT1 - MT16 - …).
from kika.processing.derived_covariance import _build_xs_map  # noqa: F401

# On-disk pickle cache around _build_xs_map so repeated v3 runs against the
# same input ENDF skip NJOY reconstruction (~10-60s) and load the pickled
# xs_map instead (<1s). See scripts/njoy_pendf_cache.py.
from scripts.njoy_pendf_cache import get_or_reconstruct_xs_map

# Reused v2 helpers
from scripts.exfor_utils import (
    DualLogger,
    EnergyBinInfo,
    build_exfor_cache_from_objects,
    build_union_energy_grid,
    check_angular_quality,
    compute_energy_bins_with_tof_resolution,
    filter_exfor_with_energy_bin,
    precompute_overlap_weights,
    run_mc_with_kernel_weights,
    save_legendre_matrix_to_parquet,
    stack_samples_to_matrix,
    write_nominal_endf,  # Phase F audit follow-up: splice v3 nominal into MF4
)
from scripts.resample_AD import (
    endf_normalize_legendre_coeffs,
    sample_legendre_coefficients,
)
from scripts.multigroup_collapse import (
    perform_adaptive_multigroup_collapse,
    build_l0_row_aggregator,
    cov_to_corr,
)
from scripts.tof_parameters import load_tof_parameters_file



# =========================================================================
# CONFIG
# =========================================================================

# I/O ---------------------------------------------------------------------
EXFOR_DB_PATH                 = '/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db'
ENDF_FILE               = '/share_snc/snc/JuanMonleon/jeff40_with_MF4_from_jeff33/26-Fe-56g.txt'
OUTPUT_DIR                    = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/v3_01/"
ENDF_OUTPUT_NAME              = 'Fe56_v3_with_mf34.endf'
# Cache for NJOY-reconstructed xs_map (Dict[MT -> CrossSection|MF3MT]).
# First v3 run on a given (ENDF, NJOY_TOLERANCE) writes a pickle here;
# subsequent runs reload it in <1 s instead of re-running NJOY (~10-60 s).
# Set to None to disable caching and re-run NJOY every time.
NJOY_PENDF_CACHE_DIR: Optional[str] = '/share_snc/snc/JuanMonleon/cache/njoy_pendf'
SUPPLEMENTARY_JSON_FILES: List[str] = [
    '/share_snc/snc/JuanMonleon/EXFOR/data_v1/27673002.json',
]
EXCLUDE_EXPERIMENTS: List[str] = ["32246002"]

# Reaction ---------------------------------------------------------------
TARGET_ZAID                   = 26056
TARGET_ZAIDS: List[int]       = [26056, 26000]   # for EXFOR loading (natural-Fe data falls back here)
MT_NUMBER                     = 2
ENERGY_RANGE_MEV              = (0.847, 4.0)
M_PROJ_U                      = 1.008665
M_TARG_U                      = 55.93494

# TOF binning ------------------------------------------------------------
DELTA_T_NS                    = 5.0
FLIGHT_PATH_M                 = 27.037
TOF_PARAMETERS_FILE: Optional[str] = "/share_snc/snc/JuanMonleon/EXFOR/exfor_tof_parameters.json"

# Union energy grid (which EXFOR subentries set the AD energy bins).
# Each entry is (subentry_id, lo_mev, hi_mev); None for lo/hi falls back to
# ENERGY_RANGE_MEV. Empty list → use ALL loaded energies as the grid.
UNION_GRID_SUBENTRIES: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("10571002", 0.847, 2.5),
    ("23365005", 2.5, 4.0),
]

# EXFOR filtering / floor (passed to filter_exfor_with_energy_bin) -------
MIN_STAT_RELATIVE_UNCERTAINTY = 0.01
UNCERTAINTY_FLOOR_STRATEGY    = 'bin_median'
NORMALIZE_BY_N_POINTS         = True
NORM_SYSTEMATIC_SIGMA         = 0.05
BAND_AWARE_ESS                = False
MAX_EXP_WEIGHT_FRAC_BIN       = 0.80

# Angular quality gate ---------------------------------------------------
ANGULAR_QUALITY_GATE          = True
MIN_ANGULAR_POINTS            = 4
MIN_BANDS_COVERED             = 3
MAX_BIN_EXPANSION             = 3

# Fit (passed to sample_legendre_coefficients / _weighted_ridge_fit) -----
MAX_LEGENDRE_DEGREE           = 6
# Sample (in MC) only orders 1..MAX_SAMPLE_ORDER. Higher orders are kept frozen
# at their nominal-fit values across all samples, so MF34 only carries variance
# for the orders most physical evaluations use. Matches v2's default.
MAX_SAMPLE_ORDER              = 3
SELECT_DEGREE                 = 'aicc'
RIDGE_LAMBDA                  = 1e-4
RIDGE_POWER                   = 4
DF_METHOD                     = 'hat'
# Per-sample degree drawing. When True, every MC sample draws its Legendre
# degree from the bin's AICc weight distribution (so a bin with AICc weights
# {L3: 0.7, L4: 0.3} sees ~70/30 fits across samples instead of always
# fitting at L=3). Frozen high orders are restored from THAT sample's
# drawn-degree nominal, not from the AICc winner's. False reproduces the
# legacy v2 behaviour (every sample at the AICc winner).
USE_DEGREE_SAMPLING_IN_MC       = True
# Re-run the per-bin AICc scan with τ-refined uncertainties after τ-IRLS
# converges, so per-sample degree draws sample a model-degree distribution
# consistent with the band-discrepancy noise model. When the post-τ winner
# differs from the pre-τ winner, the bin is also refit at the new degree
# under τ-IRLS. See resample_AD.py:sample_legendre_coefficients.
RERUN_AICC_POST_TAU             = True

# Band discrepancy (passed to compute_angular_band_discrepancy) ----------
USE_BAND_DISCREPANCY          = True
MIN_POINTS_PER_BAND           = 5
MAX_BAND_SCALE_FACTOR                = 5.0
BAND_SCALE_METHOD             = 'mad'
TAU_IRLS_MAX_ITERS            = 20
TAU_IRLS_TOL                  = 1e-2
TAU_IRLS_DAMPING              = 0.5

# Sigma-sys-aware fitting (Level 2 — always on in v3) --------------------
SIGMA_SYS_AWARE_FIT           = True

# MC ---------------------------------------------------------------------
N_SAMPLES                  = 10000
BASE_SEED                   = 42
N_PROCS                     = 40
# Overlap-weight floor for kernel-weight Pass 1. Datasets whose TOF-overlap
# weight in a given bin is below this are dropped from that bin's KW fit.
KW_MC_MIN_WEIGHT              = 1e-3

# Positivity projection. 0 disables both check + projection; otherwise the
# integer is the number of μ check points fed to
# check_angular_distribution_positivity / project_to_positive_distribution
# (resample_AD.py). Applied in BOTH Pass 1 (KW worker) and Pass 2
# (per-bin worker). Each pass logs a single summary line with the count.
POSITIVITY_CHECK_POINTS       = 101

# v3-specific ------------------------------------------------------------
WRITE_MF34_L0_ROW             = True
# Splice v3's per-bin nominal Legendre coefficients into the source ENDF's
# MF4 / MT section so the published MF4 matches the denominator MF34 was
# relativized against. Reuses scripts.exfor_utils.write_nominal_endf
# (line ~4880) — same helper v2 uses. The MF4 region outside ENERGY_RANGE_MEV
# stays byte-identical to the source. The MF34 writer in step 9 then uses
# the spliced file as its source_endf so the output ENDF carries both v3's
# nominal MF4 and the matching MF34.
REWRITE_MF4_FROM_NOMINAL      = True
MF33_PSD_JITTER               = 1e-10        # base for cholesky_decomposition
MF3_AD_C0_DISCREPANCY_WARN    = 0.05
# Skip bins where MF3 c_0 falls below this threshold (b). Below this,
# the relative perturbation factor `1 + δσ/c_0` becomes numerically
# unstable and a_l = c_l/c_0 amplifies any noise. Also catches negative
# c_0 from resonance-reconstruction smearing.
MIN_C0_MF3_BARN               = 1e-3

# Multigroup covariance ---------------------------------------------------
# When MF34_COVARIANCE_TYPE is "multigroup" or "both", v3 collapses the
# fine-grid (a_l, a_l') covariance with the v2 adaptive multigroup collapse
# (l=1-driven grouping), and applies the same grouping to the v3 (a_0, a_l)
# cross block via build_l0_row_aggregator. Output is written to a separate
# `<output>_mg.endf` file mirroring v2's convention.
GENERATE_MULTIGROUP_COVARIANCE  = True
MULTIGROUP_RHO_MIN              = 0.85
MULTIGROUP_SIGMA_RATIO_MAX: Optional[float] = 5.0   # None disables
MULTIGROUP_VARIANCE_PCT_MIN     = 67.0
MULTIGROUP_VARIANCE_PCT_MAX     = 85.0
MULTIGROUP_VARIANCE_RATIO_REF   = 5.0
MF34_COVARIANCE_TYPE            = "both"            # "fine" | "multigroup" | "both"
SAVE_MULTIGROUP_DIAGNOSTICS_CSV = True               # multigroup_boundary_decisions.csv
PSD_WARN_TOL                    = -1e-10            # min eig threshold for PSD warning

# Single verbosity flag — gates ALL extra diagnostic logging (per-bin AICc
# winner shifts, lost-covariance entries from relativization, expansion-used
# histogram, post-merge multigroup higher-order checks, per-bin MF4 deltas,
# per-bin positivity counts). Mirrors v2's pattern at v2 line 385.
VERBOSE_DIAGNOSTICS             = True

# Pipeline-A artifacts (mirror v2 names + defaults so analyses can be reused) -
# Redundant with the MF34 file but convenient for downstream notebooks.
SAVE_COVARIANCE_FILES           = False              # legendre_covariance(_multigroup).npy
                                                      # plus multigroup_{boundaries_ev,mean_coeffs}.npy.
                                                      # When WRITE_MF34_L0_ROW also writes the v3-only
                                                      # legendre_cross_c0_al(_multigroup).npy.
SAVE_CORRELATION_MATRICES       = False              # legendre_correlation(_multigroup).npy alongside
                                                      # the cov files (only honored if SAVE_COVARIANCE_FILES=True).
SAVE_TMC_PARQUET                = True               # legendre_samples_tmc.parquet — Pass-1 Pearson
                                                      # correlations × Pass-2 marginals (recommended TMC input).
SAVE_RAW_KW_PARQUET             = False              # legendre_samples_raw_kw.parquet — raw KW Pass-1
                                                      # samples (research/diagnostic only).

# NJOY reconstruction (optional). When NJOY_EXECUTABLE is set, v3 calls
# kika.processing.njoy_reconstruct to obtain resonance-reconstructed σ(E),
# and uses that for both c_0 evaluation and MF33 relative→absolute
# conversion. Recommended for inputs whose resonance region overlaps the
# AD energy range (otherwise raw MF3 may have reconstruction artifacts).
# Set to None to skip and use the raw MF3 from ENDF_FILE.
NJOY_EXECUTABLE: Optional[str] = "/soft_snc/NJOY/2016.78/bin/njoy"
NJOY_TOLERANCE                = 0.001
NJOY_TIMEOUT_SEC              = 600.0


# =========================================================================
# DATA STRUCTURES
# =========================================================================

class _NominalBin:
    """Per-bin nominal-fit result, packed into picklable form for the MC pool.

    ``bin_idx`` is the original index in the full ``energy_bins`` list (used
    for downstream output indexing). ``dsigma_pos`` is the position into the
    compacted ``dsigma_vec`` (i.e. index into the list of *fitted* bins, with
    skipped bins removed) — this is what the MC sample uses to look up the
    per-bin perturbation.
    """
    __slots__ = (
        "bin_idx", "dsigma_pos", "energy_mev", "bin_lower_mev", "bin_upper_mev",
        "exfor_df", "kernel_weights", "experiments_info",
        "frozen_degree", "nominal_coeffs", "tau_info",
        # c0_mf3         : bin-averaged σ over the *original* AD bin window. Used
        #                  as the σ_nom denominator when relativizing the MF34 L=0
        #                  cross-row, so it stays consistent with how MF33 was
        #                  projected onto the AD bin grid.
        # c0_mf3_post_QG : bin-averaged σ over the bin window AFTER the angular
        #                  quality gate expanded it to grab more data. Drives the
        #                  multiplicative perturbation factor (1 + δσ/c0_mf3_post_QG)
        #                  on the EXFOR data, so the factor's denominator matches
        #                  the energy range the data actually covers. Equals c0_mf3
        #                  when no QG expansion happened.
        "c0_mf3", "c0_mf3_post_QG",
        # Per-degree fits + AICc weights from the nominal AICc loop. The
        # frozen-high-orders step uses the sampled-degree's nominal as the
        # source, so a sample drawn at L=4 freezes order 4 at the L=4 nominal,
        # not at the L=3 winner's (which is zero for orders > 3).
        "aicc_weights", "nominal_coeffs_by_degree",
        # The next four are aliases / no-op fields so this class quacks like
        # v2's NominalFitResult dataclass for run_mc_with_kernel_weights.
        # energy_index = bin_idx; has_data = True (skipped bins never construct
        # a _NominalBin); mc_order_cap = None (v3 doesn't compute the adaptive
        # cap); interpolated = False (v3 skips rather than interpolating).
        "energy_index", "has_data", "mc_order_cap", "interpolated",
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))


# =========================================================================
# PIPELINE STEPS
# =========================================================================

def _load_endf_with_reconstruction(
    endf_path: str,
    mt: int,
    njoy_executable: Optional[str],
    njoy_tolerance: float,
    logger,
    pendf_cache_dir: Optional[str] = None,
):
    """Load ENDF, build resonance-reconstructed σ map, return everything v3 needs.

    Returns ``(endf, xs_source, mf33_mt, mf33_sections, xs_map)`` where:

    - ``endf``         — full parsed ENDF (used for ZA/AWR/MAT and as a sanity probe)
    - ``xs_source``    — ``CrossSection`` (NJOY or in-Python reconstructed) or
                         raw ``MF3MT`` if no MF2 / reconstruction failed.
                         Both expose ``.get_cross_section(energies)``.
    - ``mf33_mt``      — the requested MT's MF33MT section
    - ``mf33_sections``— the full ``Dict[int, MF33MT]`` from the file (passed to
                         ``to_xs_covmat`` as ``sibling_sections`` so NC LTY=0
                         records get resolved)
    - ``xs_map``       — the full ``Dict[int, MF3MT|CrossSection]`` (passed to
                         ``to_xs_covmat`` as ``mf3_sections`` for relative→
                         absolute conversion of contributing MTs)

    When ``pendf_cache_dir`` is set, the xs_map is pickled there on first
    call (key: ENDF stem + tolerance) and reloaded on subsequent calls,
    skipping NJOY entirely. Set to ``None`` to bypass the cache.

    Mirrors the kika-app's MF33 reconstruction pipeline (see
    ``kika-api/app/routers/endf.py:_combine_selected_mf33_sections``).
    """
    logger.info(f"Reading ENDF: {endf_path}")

    has_njoy = njoy_executable is not None
    logger.info(
        f"  NJOY: {'enabled (' + str(njoy_executable) + ')' if has_njoy else 'disabled'}"
        f"{', cache=' + pendf_cache_dir if pendf_cache_dir else ', cache=disabled'}"
    )

    # Build MT → σ(E) map (raw MF3 + reconstructed PENDF overlay where applicable).
    # If njoy_executable is set, NJOY RECONR is used; otherwise kika's
    # in-Python reconstructor is attempted (with raw-MF3 fallback on failure).
    if pendf_cache_dir is not None:
        endf, xs_map, src = get_or_reconstruct_xs_map(
            endf_path,
            njoy_executable=njoy_executable,
            tolerance=njoy_tolerance,
            cache_dir=pendf_cache_dir,
            verbose=False,
        )
        logger.info(f"  xs_map source: {src} ({len(xs_map)} MTs)")
    else:
        endf = read_endf(endf_path)
        xs_map = _build_xs_map(
            endf,
            njoy_executable=njoy_executable,
            tolerance=njoy_tolerance,
            endf_path=Path(endf_path),
        )

    mf33 = endf.files.get(33)
    if mf33 is None or mt not in mf33.sections:
        raise RuntimeError(f"MF33 MT={mt} not found in {endf_path}")
    mf33_mt = mf33.sections[mt]

    has_mf2 = 2 in endf.files
    logger.info(f"  MF2 present: {has_mf2}")

    if mt not in xs_map:
        raise RuntimeError(
            f"MT={mt} not present in MF3 (or in reconstructed PENDF) of {endf_path}"
        )
    xs_source = xs_map[mt]
    src_kind = type(xs_source).__name__
    n_pts = (
        xs_source.energies.size if hasattr(xs_source, 'energies')
        else xs_source.num_energy_points
    )
    logger.info(
        f"  MT={mt} σ(E) source: {src_kind} with {n_pts} points"
    )
    logger.info(
        f"  MF33 MT={mt}: {len(mf33_mt.subsections)} subsections "
        f"(sibling_sections to be passed: {len(mf33.sections)} MTs)"
    )

    # Detect NC LTY=0 derived covariance (e.g. JENDL-5 Fe-56 MT2)
    has_nc_lty0 = any(
        nc.lty == 0
        for sub in mf33_mt.subsections
        for nc in sub.nc_records
    )
    if has_nc_lty0:
        logger.info(
            f"  MT={mt} MF33 has NC LTY=0 record(s) — covariance will be "
            f"resolved via the sum-rule using sibling MF33 sections"
        )

    return endf, xs_source, mf33_mt, mf33.sections, xs_map


def _verify_psd(matrix: np.ndarray, name: str, logger, tol: float = -1e-10):
    """Log a warning if min eigenvalue < tol; never auto-fix.

    Returns ``(is_psd, min_eig)``. Symmetrizes before computing eigenvalues so
    floating-point asymmetry doesn't trigger spurious complex eigvals.
    """
    try:
        eigs = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        min_eig = float(eigs.min())
        is_psd = min_eig >= tol
        if not is_psd:
            n_neg = int(np.sum(eigs < tol))
            logger.warning(
                f"  [PSD] {name} is NOT PSD: min eig = {min_eig:.3e}, "
                f"{n_neg}/{eigs.size} eigenvalues below tol={tol:.0e}"
            )
        else:
            logger.info(f"  [PSD] {name} ok (min eig = {min_eig:.3e})")
        return is_psd, min_eig
    except np.linalg.LinAlgError as e:
        logger.warning(f"  [PSD] {name}: eigvalsh failed ({e})")
        return False, float('nan')


def _build_joint_sigma_al_cov(
    cov_sigma: np.ndarray,
    cov_cross: np.ndarray,
    cov_al_al: np.ndarray,
) -> np.ndarray:
    """Assemble the joint (δσ, a_l) absolute covariance block matrix.

    Inputs are absolute covariances on the same bin grid. Returns the
    ``(n_bins + n_bins*L, n_bins + n_bins*L)`` block matrix used to verify
    PSD of the joint distribution — the load-bearing test for the v3
    cross-block extension.
    """
    n_g = cov_sigma.shape[0]
    n_p = cov_al_al.shape[0]
    J = np.zeros((n_g + n_p, n_g + n_p), dtype=float)
    J[:n_g, :n_g] = cov_sigma
    J[:n_g, n_g:] = cov_cross
    J[n_g:, :n_g] = cov_cross.T
    J[n_g:, n_g:] = cov_al_al
    return J


def _slice_orders(
    cov_full: np.ndarray, n_bins: int, full_max_order: int, kept_max_order: int,
) -> np.ndarray:
    """Slice a covariance matrix laid out as [a_1(E_1)..a_L(E_1), a_1(E_2)..]
    down to keep only orders 1..kept_max_order at each energy.

    Works for square matrices (a_l, a_l') or rectangular ones where the second
    axis has the (energy, order) layout — pass ``cov_full`` and we'll slice the
    last axis only when ``cov_full.shape[0] != n_bins * full_max_order``.
    """
    keep = np.array(
        [i * full_max_order + l
         for i in range(n_bins)
         for l in range(kept_max_order)],
        dtype=int,
    )
    if cov_full.shape[0] == cov_full.shape[1]:
        return cov_full[np.ix_(keep, keep)]
    return cov_full[:, keep]


def _merge_source_mf34_outside_range(
    pipeline_mf34,
    source_endf_path: str,
    mt: int,
    max_sample_order: int,
    e_min_ev: float,
    e_max_ev: float,
    logger,
) -> Dict[str, int]:
    """Splice source MF34 data into pipeline MF34 outside our energy range.

    The pipeline MF34 covers our AD energy grid (inside [e_min_ev, e_max_ev])
    for orders 1..max_sample_order plus the v3 L=0 cross block. The source
    MF34 may have:
      (a) higher-order pairs (l > max_sample_order or l1 > max_sample_order)
          that the pipeline doesn't address at all;
      (b) lower-order pairs (both l, l1 ≤ max_sample_order) at energies
          outside our AD grid that we'd otherwise lose by replacing the
          whole MT34 section.

    For (a): if the source pair is non-zero anywhere, append a new sub-subsection
    holding the source data EXCISED of [e_min_ev, e_max_ev]. All-zero source
    pairs are silently dropped (nothing to preserve).

    For (b): find the matching pipeline sub-subsection and APPEND source
    records covering [outside the pipeline range] to its ``records`` list,
    updating NI. The pipeline's existing record (covering our AD grid) stays
    as-is. ENDF processors sum LIST records within a sub-subsection on the
    union grid, so this gives the pipeline our data inside, source data
    outside, with zero cross-source coupling between the two.

    L=0 entries from the source are skipped (kika's MF34 parser doesn't
    surface them — see mf34_writer.py:360-365 — and v3's L=0 cross block
    is already in the pipeline).

    Returns a dict with counts: ``n_high_grafted``, ``n_high_dropped_zero``,
    ``n_low_appended``, ``n_low_dropped`` for diagnostic logging.
    """
    from kika.endf.read_endf import read_endf as _read_endf
    from kika.endf.classes.mf34.mf34 import (
        SubSubsection as _SubSubsection,
    )
    from kika.endf.writers.mf34_writer import (
        _make_lb5_record as _mk_lb5,
        _make_lb6_record as _mk_lb6,
        _split_matrix_excluding_range as _split,
    )

    src_endf = _read_endf(source_endf_path)
    src_mf34 = src_endf.files.get(34)
    if src_mf34 is None or mt not in src_mf34.sections:
        logger.info(
            f"  Source MF34 MT={mt} not present — no records to splice"
        )
        return {"n_high_grafted": 0, "n_high_dropped_zero": 0,
                "n_low_appended": 0, "n_low_dropped": 0}
    src_mt = src_mf34.sections[mt]
    if not pipeline_mf34._subsections:
        logger.warning("  Pipeline MF34 has no subsections; cannot splice")
        return {"n_high_grafted": 0, "n_high_dropped_zero": 0,
                "n_low_appended": 0, "n_low_dropped": 0}
    pipe_subsection = pipeline_mf34._subsections[0]

    # Index pipeline sub-subsections by (l, l1) for low-order append lookup.
    pipe_by_pair: Dict[Tuple[int, int], _SubSubsection] = {
        (s.l, s.l1): s for s in pipe_subsection.sub_subsections
    }

    counts = {"n_high_grafted": 0, "n_high_dropped_zero": 0,
              "n_low_appended": 0, "n_low_dropped": 0}
    atol = 1e-15

    for src_sub in src_mt._subsections:
        for src_subsub in src_sub.sub_subsections:
            l, l1 = src_subsub.l, src_subsub.l1
            if l == 0 or l1 == 0:
                continue  # v3 extension; not in source

            is_low_order = (l <= max_sample_order and l1 <= max_sample_order)

            new_records: List = []
            all_zero_for_pair = True
            for rec in src_subsub.records:
                try:
                    if rec.lb == 5:
                        mat, egrid = src_mt._decode_lb5_matrix(rec)
                    elif rec.lb == 6:
                        mat, row_eg, col_eg = src_mt._decode_lb6_matrix(rec)
                        if row_eg != col_eg:
                            logger.warning(
                                f"  Source MF34 (L={l}, L1={l1}) LB=6 has "
                                f"unequal row/col grids — splice skipped"
                            )
                            continue
                        egrid = row_eg
                    else:
                        logger.warning(
                            f"  Source MF34 (L={l}, L1={l1}) LB={rec.lb} "
                            f"not supported by splice — skipped"
                        )
                        continue
                except Exception as exc:
                    logger.warning(
                        f"  Source MF34 (L={l}, L1={l1}) decode failed: "
                        f"{exc} — skipped"
                    )
                    continue

                if not np.allclose(mat, 0.0, atol=atol):
                    all_zero_for_pair = False

                splits = _split(mat, egrid, e_min_ev, e_max_ev)
                for sub_mat, sub_grid in splits:
                    if np.allclose(sub_mat, 0.0, atol=atol):
                        continue
                    sub_grid_f = [float(e) for e in sub_grid]
                    if l == l1:
                        new_records.append(_mk_lb5(sub_mat, sub_grid_f))
                    else:
                        new_records.append(_mk_lb6(
                            sub_mat, sub_grid_f, sub_grid_f,
                        ))

            if is_low_order:
                # All-zero outside our range: nothing to preserve, skip.
                if all_zero_for_pair or not new_records:
                    counts["n_low_dropped"] += 1
                    continue
                pipe_ss = pipe_by_pair.get((l, l1))
                if pipe_ss is None:
                    # Pipeline didn't write this (l, l1) at all — shouldn't
                    # happen since create_mf34_from_covariance writes the
                    # full upper triangle, but be defensive.
                    new_subsub = _SubSubsection()
                    new_subsub.l = l
                    new_subsub.l1 = l1
                    new_subsub.lct = src_subsub.lct
                    new_subsub.ni = len(new_records)
                    new_subsub.records = new_records
                    pipe_subsection.sub_subsections.append(new_subsub)
                    pipe_by_pair[(l, l1)] = new_subsub
                else:
                    # Append source-outside records to the pipeline's existing
                    # sub-subsection. Multiple LIST records per (l, l1) are
                    # well-supported; processors sum them on the union grid
                    # (cross-source cells = 0 by construction since the energy
                    # ranges don't overlap).
                    pipe_ss.records.extend(new_records)
                    pipe_ss.ni = len(pipe_ss.records)
                counts["n_low_appended"] += 1
            else:
                # Higher-order pair: append a new sub-subsection if non-zero.
                if all_zero_for_pair:
                    counts["n_high_dropped_zero"] += 1
                    continue
                if not new_records:
                    continue
                new_subsub = _SubSubsection()
                new_subsub.l = l
                new_subsub.l1 = l1
                new_subsub.lct = src_subsub.lct
                new_subsub.ni = len(new_records)
                new_subsub.records = new_records
                pipe_subsection.sub_subsections.append(new_subsub)
                pipe_by_pair[(l, l1)] = new_subsub
                counts["n_high_grafted"] += 1

    # Update NL/NL1 to reflect the highest order present after splicing.
    max_l_present = max(
        (max(s.l, s.l1) for s in pipe_subsection.sub_subsections),
        default=max_sample_order,
    )
    pipe_subsection.nl = max_l_present
    pipe_subsection.nl1 = max_l_present
    return counts


def _bin_average_xs(xs_source, e_lo_ev: float, e_hi_ev: float) -> float:
    """Trapezoidal bin-average of σ(E) over [e_lo_ev, e_hi_ev].

    Uses the source's native energy grid inside the bin (no smoothing) plus
    the bin endpoints, so resonance structure is integrated rather than point-
    sampled. Both ``MF3MT`` and a NJOY-reconstructed ``CrossSection`` expose
    ``.energies`` and ``.get_cross_section(...)``.
    """
    if e_hi_ev <= e_lo_ev:
        return float(np.asarray(
            xs_source.get_cross_section(np.array([0.5 * (e_lo_ev + e_hi_ev)])),
            dtype=float,
        )[0])
    native_e = np.asarray(xs_source.energies, dtype=float)
    inside = native_e[(native_e > e_lo_ev) & (native_e < e_hi_ev)]
    e_int = np.unique(np.concatenate([[e_lo_ev], inside, [e_hi_ev]]))
    sigma = np.asarray(xs_source.get_cross_section(e_int), dtype=float)
    return float(np.trapezoid(sigma, e_int) / (e_hi_ev - e_lo_ev))


def _run_filter(exfor_cache, sorted_energies, bin_lower_mev, bin_upper_mev,
                target_energy_mev):
    return filter_exfor_with_energy_bin(
        exfor_cache=exfor_cache,
        sorted_energies=sorted_energies,
        bin_lower_mev=bin_lower_mev,
        bin_upper_mev=bin_upper_mev,
        target_energy_mev=target_energy_mev,
        m_proj_u=M_PROJ_U,
        m_targ_u=M_TARG_U,
        dedupe_per_experiment=True,
        exclude_experiments=EXCLUDE_EXPERIMENTS,
        min_relative_uncertainty=MIN_STAT_RELATIVE_UNCERTAINTY,
        unc_floor_strategy=UNCERTAINTY_FLOOR_STRATEGY,
        normalize_by_n_points=NORMALIZE_BY_N_POINTS,
        sigma_norm=NORM_SYSTEMATIC_SIGMA,
        band_aware_ess=BAND_AWARE_ESS,
        max_experiment_weight_fraction=MAX_EXP_WEIGHT_FRAC_BIN,
    )


def _filter_with_quality_gate(exfor_cache, sorted_energies, energy_bins, bin_idx,
                              logger):
    """Run the standard filter; if quality gate fails, expand the bin window."""
    bin_info = energy_bins[bin_idx]
    res = _run_filter(
        exfor_cache, sorted_energies,
        bin_info.bin_lower_mev, bin_info.bin_upper_mev,
        bin_info.energy_mev,
    )
    df = res[0]
    expansion_used = 0

    if ANGULAR_QUALITY_GATE:
        passes, reason = check_angular_quality(df, MIN_ANGULAR_POINTS, MIN_BANDS_COVERED)
        if not passes:
            for expansion in range(1, MAX_BIN_EXPANSION + 1):
                left_idx = max(0, bin_idx - expansion)
                right_idx = min(len(energy_bins) - 1, bin_idx + expansion)
                lo = energy_bins[left_idx].bin_lower_mev
                hi = energy_bins[right_idx].bin_upper_mev
                res_exp = _run_filter(
                    exfor_cache, sorted_energies, lo, hi, bin_info.energy_mev,
                )
                ok, _ = check_angular_quality(
                    res_exp[0], MIN_ANGULAR_POINTS, MIN_BANDS_COVERED,
                )
                if ok:
                    res = res_exp
                    expansion_used = expansion
                    break
            else:
                logger.warning(
                    f"  E={bin_info.energy_mev:.4f} MeV: quality gate failed "
                    f"after {MAX_BIN_EXPANSION} expansions — skipping bin"
                )
                return None, expansion_used

    return res, expansion_used


def _attach_sigma_sys(exfor_df: pd.DataFrame) -> pd.DataFrame:
    """Add the absolute σ_sys column required by Level-2 sys-aware fits."""
    if not SIGMA_SYS_AWARE_FIT or 'sigma_sys_relative' not in exfor_df.columns:
        return exfor_df
    out = exfor_df.copy()
    out['_sigma_sys_abs'] = (
        out['sigma_sys_relative'].to_numpy(dtype=float)
        * np.abs(out['value'].to_numpy(dtype=float))
    )
    return out


def _nominal_fit_one_bin(exfor_df, kernel_weights):
    """Single nominal Legendre fit (Level 2; c_0 free; band-discrepancy IRLS)."""
    sys_unc_col_arg = (
        '_sigma_sys_abs' if '_sigma_sys_abs' in exfor_df.columns else None
    )
    coef_df, info = sample_legendre_coefficients(
        exfor_df,
        value_col='value',
        unc_col='unc',
        sys_unc_col=sys_unc_col_arg,
        degree=None,
        max_degree=MAX_LEGENDRE_DEGREE,
        select_degree=SELECT_DEGREE,
        ridge_lambda=RIDGE_LAMBDA,
        ridge_power=RIDGE_POWER,
        df_method=DF_METHOD,
        external_weights=kernel_weights if len(kernel_weights) > 0 else None,
        n_samples=1,
        rescale_unc_by_chi2=False,
        allow_shrink_unc=False,
        use_band_discrepancy=USE_BAND_DISCREPANCY,
        min_points_per_band=MIN_POINTS_PER_BAND,
        max_band_scale=MAX_BAND_SCALE_FACTOR,
        tau_irls_max_iters=TAU_IRLS_MAX_ITERS,
        tau_irls_tol=TAU_IRLS_TOL,
        tau_irls_damping=TAU_IRLS_DAMPING,
        band_scale_method=BAND_SCALE_METHOD,
        freeze_c0=False,
        rerun_aicc_post_tau=RERUN_AICC_POST_TAU,
    )
    return coef_df.iloc[0].to_numpy(), info


def _extract_aicc_weights_and_per_degree_coeffs(
    info: Dict, nominal_coeffs: np.ndarray, max_degree: int,
) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
    """Pull per-degree coeffs + Akaike weights from sample_legendre_coefficients info.

    ``info['all_degrees_info']`` is built inside the AICc scan (resample_AD.py
    around line 1432) and contains, for each degree d in 0..max_feasible:
    coeffs (length d+1), chi2, dof, eff_params, aicc. We turn the AICc scores
    into Akaike weights w_d ∝ exp(-½ ΔAICc_d) and pad each per-degree coeff
    array out to ``max_degree+1`` with zeros so downstream freeze-high
    indexing is uniform across degrees.

    The AICc-loop coeffs at the winner are pre-τ-IRLS, but ``nominal_coeffs``
    (returned by ``_nominal_fit_one_bin``) is the τ-refined fit at the winner.
    We overwrite the winner's slot with ``nominal_coeffs`` so a sample drawn
    at L=winner uses the same coeffs the pipeline's headline nominal does.

    Falls back to a single-degree distribution (winner with weight 1.0) if
    all_degrees_info is missing — keeps the pipeline working on legacy paths.
    """
    all_d = info.get('all_degrees_info')
    winner = int(info['degree'])
    nom_padded = np.zeros(max_degree + 1, dtype=float)
    nom_padded[:min(len(nominal_coeffs), max_degree + 1)] = (
        nominal_coeffs[:max_degree + 1]
    )
    if not all_d:
        return {winner: 1.0}, {winner: nom_padded}

    aicc_scores = {int(d): float(rec['aicc']) for d, rec in all_d.items()}
    min_aicc = min(aicc_scores.values())
    raw = {d: float(np.exp(-0.5 * (s - min_aicc))) for d, s in aicc_scores.items()}
    total = sum(raw.values())
    weights = {d: w / total for d, w in raw.items()} if total > 0 else {winner: 1.0}

    coeffs_by_deg: Dict[int, np.ndarray] = {}
    for d, rec in all_d.items():
        c = np.asarray(rec['coeffs'], dtype=float)
        if c.size < max_degree + 1:
            c = np.pad(c, (0, max_degree + 1 - c.size))
        else:
            c = c[:max_degree + 1].copy()
        coeffs_by_deg[int(d)] = c
    # Prefer the τ-refined post-IRLS coeffs at the winner over the AICc-loop
    # ridge fit, so a sample drawn at L=winner is identical to the legacy path.
    coeffs_by_deg[winner] = nom_padded
    return weights, coeffs_by_deg


def _build_sampled_degrees_matrix(
    nominal_bins: List["_NominalBin"],
    *,
    n_samples: int,
    seed: int,
    enabled: bool,
) -> Optional[np.ndarray]:
    """Pre-draw a Legendre degree per (bin, sample) from each bin's AICc weights.

    Pass 1 and Pass 2 share the same matrix so the congruence merge always
    pairs Pass 1 correlation entries with Pass 2 stds taken at the same
    underlying model. Independent draws across passes would destroy that
    pairing.

    Returns None when ``enabled`` is False — both workers fall back to their
    per-bin ``frozen_degree`` (legacy behaviour).
    """
    if not enabled:
        return None

    rng = np.random.default_rng(seed + 9999)  # separate stream from MC noise
    n_bins = len(nominal_bins)
    out = np.zeros((n_bins, n_samples), dtype=int)
    for k, nb in enumerate(nominal_bins):
        weights = nb.aicc_weights or {nb.frozen_degree: 1.0}
        degs = np.array(sorted(weights.keys()), dtype=int)
        probs = np.array([weights[int(d)] for d in degs], dtype=float)
        if probs.sum() <= 0:
            out[k, :] = int(nb.frozen_degree)
            continue
        probs = probs / probs.sum()
        out[k] = rng.choice(degs, size=n_samples, p=probs)
    return out


# =========================================================================
# JOINT MC — TWO-PASS (Pass 1 = KW correlations + MF33; Pass 2 = honest variance)
# =========================================================================
#
# Pass 1 is the kernel-weight machinery from v2 (precompute_overlap_weights +
# run_mc_with_kernel_weights), with v3's MF33 hooks turned on. It captures all
# cross-bin correlation channels — shared per-experiment systematics, TOF
# kernel data overlap, and the new shared MF33 δσ — but its per-bin marginal
# stds are biased low because the kernel pools data across bins.
#
# Pass 2 is independent per bin: each bin uses only its own data, draws its
# own per-experiment lognormal + stat noise + diagonal-only MF33 δσ. No
# cross-bin coupling, but the marginal variance is honest.
#
# Combine: take the correlation pattern from Pass 1, rescale to Pass 2's
# marginals via congruence transform. Cov_aa = D₂ · Corr₁ · D₂. PSD by
# construction. Cross-row Cov(δσ, a_l) comes from Pass 1 (only place δσ is
# shared with a_l) and is rescaled on the a_l axis by std₂/std₁; the σ axis
# is unchanged (Var(δσ) is exact in both passes by construction).


def _build_home_bin_map(
    overlap_weights: Dict[int, List[Tuple[Dict, float]]],
) -> Dict[str, int]:
    """Map each EXFOR dataset's e_key → its home bin index.

    "Home" is the bin in which the dataset has its maximum overlap weight —
    robust at bin edges and for measurements that bleed into neighbours via
    the TOF kernel. Iterates only fitted bins (= keys of ``overlap_weights``);
    skipped bins are never selected as homes.
    """
    best_w: Dict[str, float] = {}
    home: Dict[str, int] = {}
    for bin_idx, datasets in overlap_weights.items():
        for ds, w in datasets:
            e_key = f"{ds['experiment_id']}_{ds['exfor_energy_mev']:.6f}"
            if w > best_w.get(e_key, -np.inf):
                best_w[e_key] = float(w)
                home[e_key] = int(bin_idx)
    return home


def _pass2_one_bin(args):
    """Pass-2 worker: independent per-bin MC for honest marginal variance.

    Takes one bin and an array of independent δσ draws (one per sample),
    returns ``(bin_idx, a_l_samples)`` of shape ``(n_samples, max_degree)``.
    Mirrors the KW worker's frozen-orders semantics (orders > effective_sample
    _order are restored from nominal before normalization) so Pass 1 and Pass 2
    align element-by-element on the std vectors used to rescale.

    When ``sampled_degrees_for_bin`` is provided, each sample uses its drawn
    degree and the freeze-high step pulls from THAT degree's nominal coeffs.
    Otherwise every sample uses ``frozen_degree`` and ``nominal_coeffs``.
    """
    (
        bin_idx, exfor_df, kernel_weights, frozen_degree, nominal_coeffs,
        nominal_coeffs_by_degree, sampled_degrees_for_bin, tau_info,
        mc_order_cap, c0_mf3_post_QG, dsigma_indep_for_bin, n_samples, base_seed,
        max_degree, max_sample_order, ridge_lambda, ridge_power, df_method,
        sigma_norm, norm_dist, positivity_check_points,
    ) = args

    # Effective order cap — same rule as the KW worker (exfor_utils.py:2238-2245)
    if mc_order_cap is not None and max_sample_order is not None:
        effective_sample_order = min(max_sample_order, mc_order_cap)
    elif mc_order_cap is not None:
        effective_sample_order = mc_order_cap
    else:
        effective_sample_order = max_sample_order

    sys_unc_col_arg = (
        '_sigma_sys_abs' if '_sigma_sys_abs' in exfor_df.columns else None
    )
    a_l_samples = np.zeros((n_samples, max_degree), dtype=float)
    n_failed = 0
    # Phase D audit follow-up: count samples where the post-fit Legendre
    # series went negative across the μ-grid and was projected to positive.
    # Returned alongside n_failed so run_pass2_independent can aggregate
    # and log a per-pass summary.
    n_pos_violations = 0

    # Pre-compute the nominal a_l fallback once (used on fit failure). Built
    # from the AICc winner's coeffs — a sample-agnostic safe default.
    nom_padded = np.zeros(max_degree + 1, dtype=float)
    nom_padded[:min(len(nominal_coeffs), max_degree + 1)] = (
        nominal_coeffs[:max_degree + 1]
    )
    a_l_nom = endf_normalize_legendre_coeffs(nom_padded, include_a0=False)
    if a_l_nom.size < max_degree:
        a_l_nom = np.pad(a_l_nom, (0, max_degree - a_l_nom.size))

    # Cache base values + precompute the per-sample MF33 multiplicative
    # factor once. Avoids the per-sample ``df.copy()`` and factor recompute
    # that dominated Python overhead at 10k samples.
    base_vals = exfor_df['value'].to_numpy(dtype=float).copy()
    factor_vec = 1.0 + dsigma_indep_for_bin / max(c0_mf3_post_QG, 1e-30)
    work_df = exfor_df.copy()  # one copy, reused across all samples

    # Pre-inflate σ_stat by τ once — frozen at the nominal value across all
    # samples (matches the KW Pass 1 worker's frozen-τ path). Without this,
    # Pass 2 fit weights and noise are stat-only while the nominal used τ·σ_stat,
    # so Pass 1/Pass 2 marginals fall on different uncertainty models.
    if tau_info:
        from scripts.resample_AD import sigma_eff_from_tau
        mu_arr = exfor_df['mu'].to_numpy(dtype=float)
        unc_raw = exfor_df['unc'].to_numpy(dtype=float)
        unc_inflated = sigma_eff_from_tau(mu_arr, unc_raw, tau_info)
        work_df['unc'] = unc_inflated
    # σ_sys_abs scales with the perturbed value (σ_sys_rel × |value|), so
    # refresh it inside the per-sample loop. Pre-extract the relative column
    # once to avoid re-reading on each iteration.
    has_sys_rel = 'sigma_sys_relative' in exfor_df.columns
    sys_rel_vec = (
        exfor_df['sigma_sys_relative'].to_numpy(dtype=float)
        if has_sys_rel else None
    )
    # Stride must exceed n_samples to keep (bin, sample) seeds disjoint;
    # floor at the original 10_000 so existing n_samples<=9999 runs stay
    # byte-identical.
    seed_stride = max(n_samples + 1, 10_000)

    for s in range(n_samples):
        # Resolve the degree and the matching nominal coeffs for this sample.
        if sampled_degrees_for_bin is not None:
            deg_s = int(sampled_degrees_for_bin[s])
        else:
            deg_s = int(frozen_degree)
        if (nominal_coeffs_by_degree is not None
                and deg_s in nominal_coeffs_by_degree):
            nom_for_sample = nominal_coeffs_by_degree[deg_s]
        else:
            nom_for_sample = nominal_coeffs

        pert_vals = base_vals * factor_vec[s]
        work_df['value'] = pert_vals
        if sys_rel_vec is not None and '_sigma_sys_abs' in work_df.columns:
            work_df['_sigma_sys_abs'] = sys_rel_vec * np.abs(pert_vals)
        try:
            coef_df, _info = sample_legendre_coefficients(
                work_df,
                value_col='value',
                unc_col='unc',
                sys_unc_col=sys_unc_col_arg,
                degree=deg_s,
                max_degree=max_degree,
                select_degree=None,
                ridge_lambda=ridge_lambda,
                ridge_power=ridge_power,
                df_method=df_method,
                external_weights=(kernel_weights
                                  if len(kernel_weights) > 0 else None),
                n_samples=1,
                stochastic=True,
                rescale_unc_by_chi2=False,
                allow_shrink_unc=False,
                random_state=base_seed + bin_idx * seed_stride + s,
                use_band_discrepancy=False,  # τ frozen at nominal
                freeze_c0=True,
                fixed_c0_value=float(nom_for_sample[0]),
                sigma_norm=sigma_norm,
                norm_dist=norm_dist,
                max_sample_order=effective_sample_order,
            )
            c_full = coef_df.iloc[0].to_numpy()
            if c_full.size < max_degree + 1:
                c_full = np.pad(c_full, (0, max_degree + 1 - c_full.size))
            # Restore frozen orders from THIS SAMPLE's drawn-degree nominal.
            # If the drawn degree didn't fit a given order (e.g. drawn deg=2
            # so nom_for_sample has length 3), the order stays zero — that's
            # the right physics: a sample that 'chose' L=2 has no L≥3 content.
            if effective_sample_order is not None:
                for l in range(effective_sample_order + 1, len(c_full)):
                    if l < len(nom_for_sample):
                        c_full[l] = nom_for_sample[l]
            # Positivity check + projection. Mirrors the KW Pass-1 worker
            # (exfor_utils._run_one_kw_sample). Frozen indices match: c_0 is
            # always pinned (Pass 2 fits with freeze_c0=True), and orders >
            # effective_sample_order are pinned at the drawn-degree's nominal.
            if positivity_check_points > 0:
                from scripts.resample_AD import (
                    check_angular_distribution_positivity,
                    project_to_positive_distribution,
                )
                if not check_angular_distribution_positivity(
                    c_full, positivity_check_points,
                ):
                    frozen = {0: c_full[0]}
                    if (effective_sample_order is not None
                            and effective_sample_order + 1 < len(c_full)):
                        frozen.update({
                            i: c_full[i]
                            for i in range(effective_sample_order + 1, len(c_full))
                        })
                    c_full = project_to_positive_distribution(
                        c_full, positivity_check_points,
                        frozen_indices=frozen or None,
                    )
                    n_pos_violations += 1
            a_l = endf_normalize_legendre_coeffs(c_full, include_a0=False)
            if a_l.size < max_degree:
                a_l = np.pad(a_l, (0, max_degree - a_l.size))
            a_l_samples[s, :] = a_l[:max_degree]
        except Exception:
            n_failed += 1
            a_l_samples[s, :] = a_l_nom

    return bin_idx, a_l_samples, n_failed, n_pos_violations


def run_pass1_kw(
    nominal_bins: List[_NominalBin],
    energy_bins: List[EnergyBinInfo],
    overlap_weights: Dict[int, List[Tuple[Dict, float]]],
    cov_c0_used: np.ndarray,
    c0_post_QG_vec: np.ndarray,
    *,
    n_samples: int,
    n_workers: int,
    seed: int,
    max_degree: int,
    max_sample_order: int,
    sampled_degrees: Optional[np.ndarray],
    logger,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Pass 1 — KW MC with shared per-experiment + shared MF33 perturbations.

    Returns
    -------
    al_pass1 : (n_samples, n_bins * max_degree) — ENDF-normalized a_l
    dsigma_pass1 : (n_samples, n_bins) — the MF33 δσ draws used in Pass 1
    home_bin_map : {e_key → bin_idx} — for diagnostic / re-use
    """
    n_bins = len(nominal_bins)
    rng = np.random.default_rng(seed)

    # Cholesky on the compacted MF33 covariance for the active bins.
    L_c0 = cholesky_decomposition(
        cov_obj=None,
        space="linear",
        psd_method="auto",
        jitter_scale=MF33_PSD_JITTER,
        verbose=False,
        matrix=cov_c0_used,
    )
    dsigma_pass1 = (L_c0 @ rng.standard_normal((n_bins, n_samples))).T  # (n_samples, n_bins)

    # Map e_key → home bin (use compacted index space — same indices the KW
    # worker uses, since c0_post_QG_vec is indexed by compacted position too)
    bin_idx_to_compacted = {nb.bin_idx: nb.dsigma_pos for nb in nominal_bins}
    home_bin_map_full: Dict[str, int] = _build_home_bin_map(overlap_weights)
    # Translate full-grid bin_idx → compacted dsigma_pos used by the MF33 vector
    home_bin_map: Dict[str, int] = {}
    for e_key, full_bin_idx in home_bin_map_full.items():
        compacted = bin_idx_to_compacted.get(full_bin_idx)
        if compacted is not None:
            home_bin_map[e_key] = compacted
    if len(home_bin_map) < len(home_bin_map_full):
        logger.warning(
            f"  Pass 1: {len(home_bin_map_full) - len(home_bin_map)} datasets "
            f"have homes outside the fitted-bin set (skipped from MF33 perturbation)"
        )

    logger.info(
        f"  Pass 1 (KW + MF33): {n_samples} samples × {n_bins} bins, "
        f"{len(home_bin_map)} datasets with MF33 home"
    )

    # Build worker-friendly views of the per-(bin, sample) degree matrix and
    # per-degree nominal coeffs, both keyed by the full-grid bin_idx that the
    # KW worker uses internally.
    if sampled_degrees is not None:
        sampled_degrees_per_bin_sample = {
            int(nb.bin_idx): sampled_degrees[k]
            for k, nb in enumerate(nominal_bins)
        }
        nominal_coeffs_by_bin_by_degree = {
            int(nb.bin_idx): nb.nominal_coeffs_by_degree
            for nb in nominal_bins
        }
    else:
        sampled_degrees_per_bin_sample = None
        nominal_coeffs_by_bin_by_degree = None

    kw_samples = run_mc_with_kernel_weights(
        nominal_results=nominal_bins,
        energy_bins=energy_bins,
        overlap_weights=overlap_weights,
        n_samples=n_samples,
        n_workers=n_workers,
        sigma_norm=NORM_SYSTEMATIC_SIGMA,
        sigma_norm_elastic=0.0,
        norm_dist='lognormal',
        max_degree=max_degree,
        ridge_lambda=RIDGE_LAMBDA,
        base_seed=seed,
        # use_band_discrepancy=True activates the worker's frozen-τ pre-inflation
        # path (exfor_utils._run_one_kw_sample): noise drawn from σ_eff = τ·σ_stat
        # and fit_df['unc'] pre-inflated, then `fit_use_band_discrepancy=False` is
        # passed to sample_legendre_coefficients so τ is NOT re-estimated. This
        # makes Pass 1 use the same τ-band model as the nominal and Pass 2.
        use_band_discrepancy=True,
        min_points_per_band=MIN_POINTS_PER_BAND,
        max_band_scale=MAX_BAND_SCALE_FACTOR,
        freeze_c0=True,
        # Pin c0 at the per-degree nominal so MF33 perturbations propagate to a_l.
        # With c0 floating (the previous behavior) a uniform multiplicative MF33
        # factor cancels out of a_l = c_l/c_0 and the cross block measures noise
        # rather than the MF33→a_l channel.
        fix_c0_at_nominal=True,
        # Sys-aware MC fit weights — σ_total² = (τ·σ_stat)² + σ_sys². Mirrors
        # the nominal and Pass 2 so the congruence merge in _combine_passes
        # combines correlations and stds taken under the same noise model.
        sys_aware_mc_fit=True,
        max_sample_order=max_sample_order,
        apply_positivity_projection=POSITIVITY_CHECK_POINTS > 0,
        positivity_check_points=POSITIVITY_CHECK_POINTS,
        max_experiment_weight_fraction=MAX_EXP_WEIGHT_FRAC_BIN,
        min_relative_uncertainty=MIN_STAT_RELATIVE_UNCERTAINTY,
        band_aware_ess=BAND_AWARE_ESS,
        mf33_dsigma_per_sample=dsigma_pass1,
        mf33_c0_per_bin=c0_post_QG_vec,
        mf33_home_bin_by_e_key=home_bin_map,
        sampled_degrees_per_bin_sample=sampled_degrees_per_bin_sample,
        nominal_coeffs_by_bin_by_degree=nominal_coeffs_by_bin_by_degree,
        logger=logger,
    )

    # Stack into a flat (n_samples, n_bins * max_degree) matrix indexed by
    # the compacted bin position (so the layout matches Pass 2 and the MF33
    # vector). For each compacted position k, the matching full-grid bin
    # index is nominal_bins[k].bin_idx.
    al_pass1 = stack_samples_to_matrix(
        kw_samples,
        [nb.bin_idx for nb in nominal_bins],
        n_samples,
        max_degree,
    )
    del kw_samples  # free the dict-of-dicts overhead before Pass 2 allocates

    return al_pass1, dsigma_pass1, home_bin_map


def run_pass2_independent(
    nominal_bins: List[_NominalBin],
    cov_c0_used: np.ndarray,
    *,
    n_samples: int,
    n_workers: int,
    seed: int,
    max_degree: int,
    max_sample_order: int,
    sampled_degrees: Optional[np.ndarray],
    logger,
) -> np.ndarray:
    """Pass 2 — independent per-bin MC for honest marginal variance.

    Each bin sees only its own data; per-experiment + MF33 + stat draws are
    independent across bins. Returns ``al_pass2`` of shape
    ``(n_samples, n_bins * max_degree)`` (same layout as Pass 1).

    ``sampled_degrees`` (optional ``(n_bins, n_samples)`` int array) is the
    pre-drawn per-(bin, sample) Legendre degree. Pass 1 sees the same matrix
    so the congruence merge keeps correlations and stds aligned.
    """
    n_bins = len(nominal_bins)
    rng = np.random.default_rng(seed + 1)  # different stream from Pass 1

    # Diagonal-only MF33 draws — independent across bins, marginal variance
    # matches the per-bin diagonal of cov_c0_used.
    diag_var = np.maximum(np.diag(cov_c0_used), 0.0)
    diag_std = np.sqrt(diag_var)
    dsigma_pass2 = rng.standard_normal((n_samples, n_bins)) * diag_std[None, :]

    args_list = []
    for k, nb in enumerate(nominal_bins):
        args_list.append((
            nb.bin_idx,
            nb.exfor_df,
            nb.kernel_weights,
            nb.frozen_degree,
            nb.nominal_coeffs,
            nb.nominal_coeffs_by_degree,
            sampled_degrees[k] if sampled_degrees is not None else None,
            nb.tau_info or {},   # frozen τ from nominal — applied to unc once
            nb.mc_order_cap,
            nb.c0_mf3_post_QG,
            dsigma_pass2[:, k],   # (n_samples,)
            n_samples,
            seed,
            max_degree,
            max_sample_order,
            RIDGE_LAMBDA,
            RIDGE_POWER,
            DF_METHOD,
            NORM_SYSTEMATIC_SIGMA,
            'lognormal',
            POSITIVITY_CHECK_POINTS,
        ))

    logger.info(
        f"  Pass 2 (independent per-bin): {n_samples} samples × {n_bins} bins"
    )

    al_pass2 = np.zeros((n_samples, n_bins * max_degree), dtype=float)
    total_failed = 0
    total_pos_violations = 0
    bin_idx_to_k = {nb.bin_idx: k for k, nb in enumerate(nominal_bins)}
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for bin_idx, a_l_samples, n_failed, n_pos in pool.imap_unordered(
                _pass2_one_bin, args_list,
            ):
                k = bin_idx_to_k[bin_idx]
                al_pass2[:, k * max_degree:(k + 1) * max_degree] = a_l_samples
                total_failed += n_failed
                total_pos_violations += n_pos
    else:
        for args in args_list:
            bin_idx, a_l_samples, n_failed, n_pos = _pass2_one_bin(args)
            k = bin_idx_to_k[bin_idx]
            al_pass2[:, k * max_degree:(k + 1) * max_degree] = a_l_samples
            total_failed += n_failed
            total_pos_violations += n_pos

    if total_failed > 0:
        frac = total_failed / max(n_samples * n_bins, 1)
        msg = (f"  Pass 2 fit failures: {total_failed} / {n_samples * n_bins} "
               f"({frac * 100:.2f}%) — fell back to nominal a_l on those")
        if frac > 0.05:
            logger.warning(msg)
        else:
            logger.info(msg)

    if POSITIVITY_CHECK_POINTS > 0:
        denom = n_samples * max(n_bins, 1)
        pct = 100.0 * total_pos_violations / max(denom, 1)
        logger.info(
            f"  Pass 2 positivity: {total_pos_violations}/{denom} (bin, sample) "
            f"distributions projected ({pct:.2f}%)"
        )

    return al_pass2


def _combine_passes(
    al_pass1: np.ndarray,
    al_pass2: np.ndarray,
    dsigma_pass1: np.ndarray,
    logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """Congruence-transform combine. Returns (cov_aa_abs, cov_cross_abs).

    - Correlation pattern from Pass 1 (with cross-bin couplings).
    - Marginal stds from Pass 2 (honest, no kernel pooling).
    - Cross-row from Pass 1, rescaled on the a_l axis only (σ axis exact in
      both passes by construction, since dsigma is the MF33 input draw).
    """
    n_samples = al_pass1.shape[0]
    cov_aa_pass1 = np.cov(al_pass1.T)
    std1 = np.sqrt(np.maximum(np.diag(cov_aa_pass1), 0.0))
    std2 = np.sqrt(np.var(al_pass2, axis=0))

    # Relative floor — std varies by orders of magnitude across (bin, l)
    floor = 1e-12 * max(float(std1.max()), 1.0)
    active = std1 > floor

    # Correlation matrix from Pass 1 (zero on rows/cols where Pass 1 is degenerate)
    Corr = np.zeros_like(cov_aa_pass1)
    if active.any():
        denom = np.outer(std1, std1)
        mask = active[:, None] & active[None, :]
        Corr[mask] = cov_aa_pass1[mask] / denom[mask]
    # Pin diagonal to exactly 1 on active params (numerical drift can leave it at 0.999…)
    diag = np.where(active, 1.0, 0.0)
    np.fill_diagonal(Corr, diag)

    # Combined covariance via congruence — PSD iff Corr is PSD (it is, since
    # zeroing rows/cols of a PSD matrix preserves PSD).
    cov_aa_abs = (std2[:, None] * Corr) * std2[None, :]

    # Cross-row from Pass 1, rescaled on the a_l axis
    al_centred = al_pass1 - al_pass1.mean(axis=0, keepdims=True)
    dsigma_centred = dsigma_pass1 - dsigma_pass1.mean(axis=0, keepdims=True)
    cov_cross_pass1 = (dsigma_centred.T @ al_centred) / max(n_samples - 1, 1)
    rescale = np.where(active, std2 / np.where(active, std1, 1.0), 0.0)
    cov_cross_abs = cov_cross_pass1 * rescale[None, :]

    # Diagnostics
    if active.any():
        ratio = std2[active] / std1[active]
        logger.info(
            f"  Combine std₂/std₁: min={ratio.min():.2f} median={np.median(ratio):.2f} "
            f"max={ratio.max():.2f} (>1 expected — Pass 1 pooling tightens marginals)"
        )
    n_inactive = int(np.sum(~active))
    if n_inactive:
        logger.info(
            f"  Combine: {n_inactive}/{active.size} parameters had std₁≈0 "
            f"(frozen orders, skipped bins, or low-variance params) — set to zero in Corr"
        )

    return cov_aa_abs, cov_cross_abs


def run_joint_mc_two_pass(
    nominal_bins: List[_NominalBin],
    energy_bins: List[EnergyBinInfo],
    cov_c0_used: np.ndarray,
    c0_post_QG_vec: np.ndarray,
    *,
    n_samples: int,
    n_workers: int,
    seed: int,
    max_degree: int,
    max_sample_order: int,
    sampled_degrees: Optional[np.ndarray],
    logger,
    step_times: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-pass orchestrator.

    Returns
    -------
    cov_aa_abs    : (n_bins·L, n_bins·L)  absolute Cov(a_l, a_l′)
    cov_cross_abs : (n_bins,    n_bins·L) absolute Cov(δσ, a_l)
    dsigma_pass1  : (n_samples, n_bins)   raw MF33 draws (kept for diagnostics)
    al_pass1      : (n_samples, n_bins·L) raw KW Pass-1 a_l samples
                    (Pass-1 marginals + cross-bin correlations).
    al_pass2      : (n_samples, n_bins·L) raw per-bin Pass-2 a_l samples
                    (honest marginals, independent across bins).

    ``sampled_degrees`` is an optional ``(n_bins, n_samples)`` integer matrix
    giving the Legendre degree to fit per (bin, sample). Same array fed into
    both passes so the congruence merge remains aligned. None ⇒ each pass uses
    its bin's ``frozen_degree`` everywhere.
    """
    logger.info("  Building TOF overlap weights for KW Pass 1")
    tof_params_cache: Dict = {}
    if TOF_PARAMETERS_FILE:
        try:
            tof_params_cache = load_tof_parameters_file(TOF_PARAMETERS_FILE)
            logger.info(
                f"  Loaded TOF parameters for {len(tof_params_cache)} experiments"
            )
        except FileNotFoundError:
            logger.warning(
                f"  TOF parameters file not found: {TOF_PARAMETERS_FILE} — "
                f"falling back to defaults ({FLIGHT_PATH_M} m, {DELTA_T_NS} ns) "
                f"for every dataset"
            )
    overlap_weights = precompute_overlap_weights(
        nominal_results=nominal_bins,
        energy_bins=energy_bins,
        min_weight=KW_MC_MIN_WEIGHT,
        tof_params_cache=tof_params_cache if tof_params_cache else None,
        default_flight_path_m=FLIGHT_PATH_M,
        default_time_resolution_ns=DELTA_T_NS,
    )
    n_pairs = sum(len(v) for v in overlap_weights.values())
    logger.info(f"  Overlap weights: {n_pairs} (dataset, bin) pairs")

    t_p1 = time.perf_counter()
    al_pass1, dsigma_pass1, _home_map = run_pass1_kw(
        nominal_bins, energy_bins, overlap_weights, cov_c0_used, c0_post_QG_vec,
        n_samples=n_samples, n_workers=n_workers, seed=seed,
        max_degree=max_degree, max_sample_order=max_sample_order,
        sampled_degrees=sampled_degrees,
        logger=logger,
    )
    t_p1_elapsed = time.perf_counter() - t_p1
    logger.info(f"  Pass 1 (KW + MF33) elapsed: {t_p1_elapsed:.2f}s")
    if step_times is not None:
        step_times["6a"] = t_p1_elapsed

    t_p2 = time.perf_counter()
    al_pass2 = run_pass2_independent(
        nominal_bins, cov_c0_used,
        n_samples=n_samples, n_workers=n_workers, seed=seed,
        max_degree=max_degree, max_sample_order=max_sample_order,
        sampled_degrees=sampled_degrees,
        logger=logger,
    )
    t_p2_elapsed = time.perf_counter() - t_p2
    logger.info(f"  Pass 2 (independent per-bin) elapsed: {t_p2_elapsed:.2f}s")
    if step_times is not None:
        step_times["6b"] = t_p2_elapsed

    t_cb = time.perf_counter()
    cov_aa_abs, cov_cross_abs = _combine_passes(
        al_pass1, al_pass2, dsigma_pass1, logger,
    )
    t_cb_elapsed = time.perf_counter() - t_cb
    logger.info(f"  Combine (congruence merge) elapsed: {t_cb_elapsed:.2f}s")
    if step_times is not None:
        step_times["6c"] = t_cb_elapsed

    return cov_aa_abs, cov_cross_abs, dsigma_pass1, al_pass1, al_pass2


# =========================================================================
# LOGGING HELPERS — config dump + step banners (mirrors v2's run.log layout)
# =========================================================================


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def _step_start(logger, step_no: str, name: str) -> float:
    logger.info("")
    logger.info(f"#-- STEP {step_no}: {name} " + "-" * max(0, 60 - len(name)))
    return time.perf_counter()


def _step_end(logger, step_no: str, t_start: float, step_times: Dict[str, float]) -> None:
    elapsed = time.perf_counter() - t_start
    step_times[step_no] = elapsed
    logger.info(f"#-- END STEP {step_no} (elapsed: {elapsed:.2f}s) " + "-" * 35)


def _log_v3_config(logger) -> None:
    """Dump every v3 CONFIG constant. Mirrors v2's CONFIG block format so
    side-by-side log diffs are meaningful. Sections cover the same ground as
    v2 plus v3-only blocks (NJOY reconstruction, MF33 hooks, two-pass MC,
    cross-block / multigroup)."""
    logger.info("#== CONFIG ================================================================")
    logger.info("")

    logger.info("  # Paths & I/O")
    logger.info(f"  ENDF_FILE = {ENDF_FILE}")
    logger.info(f"  EXFOR_DB_PATH = {EXFOR_DB_PATH}")
    logger.info(f"  OUTPUT_DIR = {OUTPUT_DIR}")
    logger.info(f"  ENDF_OUTPUT_NAME = {ENDF_OUTPUT_NAME}")
    logger.info(f"  NJOY_PENDF_CACHE_DIR = {NJOY_PENDF_CACHE_DIR}")
    logger.info(f"  TOF_PARAMETERS_FILE = {TOF_PARAMETERS_FILE}")
    logger.info(f"  SUPPLEMENTARY_JSON_FILES = {SUPPLEMENTARY_JSON_FILES}")
    logger.info("")

    logger.info("  # Reaction & Data Source")
    logger.info(f"  TARGET_ZAID = {TARGET_ZAID}")
    logger.info(f"  TARGET_ZAIDS = {TARGET_ZAIDS}")
    logger.info(f"  MT_NUMBER = {MT_NUMBER}")
    logger.info(f"  ENERGY_RANGE_MEV = {ENERGY_RANGE_MEV}")
    logger.info(f"  M_PROJ_U = {M_PROJ_U}")
    logger.info(f"  M_TARG_U = {M_TARG_U}")
    logger.info(f"  EXCLUDE_EXPERIMENTS = {EXCLUDE_EXPERIMENTS if EXCLUDE_EXPERIMENTS else 'None'}")
    logger.info("")

    logger.info("  # TOF binning")
    logger.info(f"  DELTA_T_NS = {DELTA_T_NS}")
    logger.info(f"  FLIGHT_PATH_M = {FLIGHT_PATH_M}")
    logger.info(f"  UNION_GRID_SUBENTRIES = {UNION_GRID_SUBENTRIES}")
    logger.info("")

    logger.info("  # EXFOR filtering / floor")
    logger.info(f"  MIN_STAT_RELATIVE_UNCERTAINTY = {MIN_STAT_RELATIVE_UNCERTAINTY} "
                f"({MIN_STAT_RELATIVE_UNCERTAINTY * 100:.2f}%)")
    logger.info(f"  UNCERTAINTY_FLOOR_STRATEGY = {UNCERTAINTY_FLOOR_STRATEGY}")
    logger.info(f"  NORMALIZE_BY_N_POINTS = {NORMALIZE_BY_N_POINTS}")
    logger.info(f"  NORM_SYSTEMATIC_SIGMA = {NORM_SYSTEMATIC_SIGMA} "
                f"({NORM_SYSTEMATIC_SIGMA * 100:.1f}%, lognormal)")
    logger.info(f"  BAND_AWARE_ESS = {BAND_AWARE_ESS}")
    logger.info(f"  MAX_EXP_WEIGHT_FRAC_BIN = {MAX_EXP_WEIGHT_FRAC_BIN}")
    logger.info("")

    logger.info("  # Angular quality gate")
    logger.info(f"  ANGULAR_QUALITY_GATE = {ANGULAR_QUALITY_GATE}")
    if ANGULAR_QUALITY_GATE:
        logger.info(f"  MIN_ANGULAR_POINTS = {MIN_ANGULAR_POINTS}")
        logger.info(f"  MIN_BANDS_COVERED = {MIN_BANDS_COVERED}")
        logger.info(f"  MAX_BIN_EXPANSION = {MAX_BIN_EXPANSION}")
    logger.info("")

    logger.info("  # Legendre Fitting")
    logger.info(f"  MAX_LEGENDRE_DEGREE = {MAX_LEGENDRE_DEGREE}")
    logger.info(f"  MAX_SAMPLE_ORDER = {MAX_SAMPLE_ORDER}")
    logger.info(f"  SELECT_DEGREE = {SELECT_DEGREE}")
    logger.info(f"  RIDGE_LAMBDA = {RIDGE_LAMBDA}")
    logger.info(f"  RIDGE_POWER = {RIDGE_POWER}")
    logger.info(f"  DF_METHOD = {DF_METHOD}")
    logger.info(f"  USE_DEGREE_SAMPLING_IN_MC = {USE_DEGREE_SAMPLING_IN_MC}")
    logger.info(f"  RERUN_AICC_POST_TAU = {RERUN_AICC_POST_TAU}")
    logger.info(f"  POSITIVITY_CHECK_POINTS = {POSITIVITY_CHECK_POINTS} "
                f"({'enabled' if POSITIVITY_CHECK_POINTS > 0 else 'disabled'})")
    logger.info(f"  VERBOSE_DIAGNOSTICS = {VERBOSE_DIAGNOSTICS}")
    logger.info(f"  SIGMA_SYS_AWARE_FIT = {SIGMA_SYS_AWARE_FIT}")
    logger.info("")

    logger.info("  # Band discrepancy (τ-IRLS)")
    logger.info(f"  USE_BAND_DISCREPANCY = {USE_BAND_DISCREPANCY}")
    logger.info(f"  MIN_POINTS_PER_BAND = {MIN_POINTS_PER_BAND}")
    logger.info(f"  MAX_BAND_SCALE_FACTOR = {MAX_BAND_SCALE_FACTOR}")
    logger.info(f"  BAND_SCALE_METHOD = {BAND_SCALE_METHOD}")
    logger.info(f"  TAU_IRLS_MAX_ITERS = {TAU_IRLS_MAX_ITERS}")
    logger.info(f"  TAU_IRLS_TOL = {TAU_IRLS_TOL}")
    logger.info(f"  TAU_IRLS_DAMPING = {TAU_IRLS_DAMPING}")
    logger.info("")

    logger.info("  # MC sampling")
    logger.info(f"  N_SAMPLES = {N_SAMPLES}")
    logger.info(f"  BASE_SEED = {BASE_SEED}")
    logger.info(f"  N_PROCS = {N_PROCS}")
    logger.info(f"  KW_MC_MIN_WEIGHT = {KW_MC_MIN_WEIGHT}")
    logger.info("")

    logger.info("  # v3-specific (MF3↔MF34 cross-correlation)")
    logger.info(f"  WRITE_MF34_L0_ROW = {WRITE_MF34_L0_ROW}")
    logger.info(f"  REWRITE_MF4_FROM_NOMINAL = {REWRITE_MF4_FROM_NOMINAL}")
    logger.info(f"  MF33_PSD_JITTER = {MF33_PSD_JITTER}")
    logger.info(f"  MF3_AD_C0_DISCREPANCY_WARN = {MF3_AD_C0_DISCREPANCY_WARN}")
    logger.info(f"  MIN_C0_MF3_BARN = {MIN_C0_MF3_BARN}")
    logger.info(f"  PSD_WARN_TOL = {PSD_WARN_TOL}")
    logger.info("")

    logger.info("  # NJOY reconstruction")
    logger.info(f"  NJOY_EXECUTABLE = {NJOY_EXECUTABLE}")
    logger.info(f"  NJOY_TOLERANCE = {NJOY_TOLERANCE}")
    logger.info(f"  NJOY_TIMEOUT_SEC = {NJOY_TIMEOUT_SEC}")
    logger.info("")

    logger.info("  # Multigroup Covariance")
    logger.info(f"  GENERATE_MULTIGROUP_COVARIANCE = {GENERATE_MULTIGROUP_COVARIANCE}")
    logger.info(f"  MULTIGROUP_RHO_MIN = {MULTIGROUP_RHO_MIN}")
    logger.info(f"  MULTIGROUP_SIGMA_RATIO_MAX = {MULTIGROUP_SIGMA_RATIO_MAX}")
    logger.info(f"  MULTIGROUP_VARIANCE_PCT_MIN = {MULTIGROUP_VARIANCE_PCT_MIN}")
    logger.info(f"  MULTIGROUP_VARIANCE_PCT_MAX = {MULTIGROUP_VARIANCE_PCT_MAX}")
    logger.info(f"  MULTIGROUP_VARIANCE_RATIO_REF = {MULTIGROUP_VARIANCE_RATIO_REF}")
    logger.info(f"  MF34_COVARIANCE_TYPE = {MF34_COVARIANCE_TYPE}")
    logger.info(f"  SAVE_MULTIGROUP_DIAGNOSTICS_CSV = {SAVE_MULTIGROUP_DIAGNOSTICS_CSV}")
    logger.info("")

    logger.info("  # Output artifacts")
    logger.info(f"  SAVE_COVARIANCE_FILES = {SAVE_COVARIANCE_FILES}")
    logger.info(f"  SAVE_CORRELATION_MATRICES = {SAVE_CORRELATION_MATRICES}")
    logger.info(f"  SAVE_TMC_PARQUET = {SAVE_TMC_PARQUET}")
    logger.info(f"  SAVE_RAW_KW_PARQUET = {SAVE_RAW_KW_PARQUET}")
    logger.info("")
    logger.info("#== END CONFIG ============================================================")
    logger.info("")


# =========================================================================
# MAIN
# =========================================================================

def run_v3(logger=None):
    # All run artifacts (fine + multigroup MF34 files, log, multigroup CSV)
    # land under OUTPUT_DIR. Created on demand. The ENDF input is read from
    # ENDF_FILE and is not modified.
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    endf_output_path = str(output_path / ENDF_OUTPUT_NAME)

    if logger is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = str(output_path / f'exfor_to_endf_{timestamp}.log')
        # DualLogger writes to file; we wrap it to also echo to stdout for
        # interactive runs (matches v2's typical usage).
        _file_logger = DualLogger(log_file=log_path)
        class _EchoLogger:
            def info(self, msg: str):
                _file_logger.info(msg, console=True)
            def warning(self, msg: str):
                _file_logger.warning(msg, console=True)
            def error(self, msg: str):
                _file_logger.error(msg, console=True)
            def debug(self, msg: str):
                _file_logger.debug(msg)
        logger = _EchoLogger()
        logger.info(f"Log file: {log_path}")

    logger.info("EXFOR-to-ENDF Angular Distribution Sampling (v3)")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"OUTPUT_DIR: {output_path}")
    logger.info(f"Target ZAID={TARGET_ZAID}, MT={MT_NUMBER}, "
                f"E∈[{ENERGY_RANGE_MEV[0]}, {ENERGY_RANGE_MEV[1]}] MeV")
    logger.info("")

    _log_v3_config(logger)

    # Per-step elapsed times collected for the SUMMARY block at the end.
    step_times: Dict[str, float] = {}
    t_pipeline_start = time.perf_counter()

    # ---- STEP 1: Load EXFOR ----
    t_step = _step_start(logger, "1", "Load EXFOR data")
    logger.info(f"  [INFO] [EXFOR] Source: database ({EXFOR_DB_PATH})")
    logger.info(f"  [INFO] [EXFOR] Target ZAIDs: {TARGET_ZAIDS}, MT={MT_NUMBER}")
    if SUPPLEMENTARY_JSON_FILES:
        logger.info(f"  [INFO] [EXFOR] Supplementary JSON files: "
                    f"{len(SUPPLEMENTARY_JSON_FILES)}")
        for f in SUPPLEMENTARY_JSON_FILES:
            logger.info(f"    - {f}")
    if EXCLUDE_EXPERIMENTS:
        logger.info(f"  [INFO] [EXFOR] Excluding: {EXCLUDE_EXPERIMENTS}")
    exfor_dict, _load_status = read_all_exfor(
        source='database',
        db_path=EXFOR_DB_PATH,
        target_zaid=TARGET_ZAIDS,
        projectile='N',
        mt=MT_NUMBER,
        energy_range=ENERGY_RANGE_MEV,
        supplementary_json_files=SUPPLEMENTARY_JSON_FILES,
        exclude_experiments=EXCLUDE_EXPERIMENTS,
        group_by_energy=False,
        return_load_status=True,
    )
    exfor_cache, sorted_energies = build_exfor_cache_from_objects(
        list(exfor_dict.values()),
        exclude_experiments=EXCLUDE_EXPERIMENTS,
    )
    n_exfor_files = sum(len(v) for v in exfor_cache.values())
    logger.info(
        f"  [INFO] [EXFOR] Loaded {n_exfor_files} experiments at "
        f"{len(sorted_energies)} unique energies"
    )
    if sorted_energies:
        logger.info(
            f"  [INFO] [EXFOR] Energy range: "
            f"[{min(sorted_energies):.4f}, {max(sorted_energies):.4f}] MeV"
        )
    logger.info(f">> exfor_experiments = {n_exfor_files}")
    logger.info(f">> exfor_energies = {len(sorted_energies)}")
    _step_end(logger, "1", t_step, step_times)

    # ---- STEP 2: Load ENDF + reconstruct σ(E) + assemble sibling MF33 dict ----
    # Mirrors the kika-app's MF33 reconstruction pipeline: NJOY RECONR (or
    # in-Python fallback) produces an MT→σ(E) map, then to_xs_covmat is
    # called with the full sibling MF33 dict so NC LTY=0 derived MTs (e.g.
    # JENDL-5 Fe-56 MT2 = MT1 - MT16 - …) get resolved via the sum rule.
    t_step = _step_start(logger, "2", "Read ENDF + NJOY reconstruct + MF33 sibling dict")
    endf, xs_source, mf33_mt, mf33_sections, xs_map = _load_endf_with_reconstruction(
        ENDF_FILE, MT_NUMBER, NJOY_EXECUTABLE, NJOY_TOLERANCE, logger,
        pendf_cache_dir=NJOY_PENDF_CACHE_DIR,
    )
    _step_end(logger, "2", t_step, step_times)

    # ---- STEP 3: Build AD energy bins ----
    t_step = _step_start(logger, "3", "Compute AD energy bins (TOF resolution)")
    if UNION_GRID_SUBENTRIES:
        grid_energies_ev = build_union_energy_grid(
            exfor_cache=exfor_cache,
            subentries=UNION_GRID_SUBENTRIES,
            energy_min_mev=ENERGY_RANGE_MEV[0],
            energy_max_mev=ENERGY_RANGE_MEV[1],
        )
    else:
        # Default: every energy in the cache that's inside ENERGY_RANGE_MEV
        all_e_mev = sorted(
            e for e in exfor_cache
            if ENERGY_RANGE_MEV[0] <= e <= ENERGY_RANGE_MEV[1]
        )
        grid_energies_ev = np.array(
            sorted(set(all_e_mev) | set(ENERGY_RANGE_MEV))
        ) * 1e6
    energy_bins = compute_energy_bins_with_tof_resolution(
        energies_ev=grid_energies_ev,
        energy_min_mev=ENERGY_RANGE_MEV[0],
        energy_max_mev=ENERGY_RANGE_MEV[1],
        delta_t_ns=DELTA_T_NS,
        flight_path_m=FLIGHT_PATH_M,
    )
    logger.info(f"  [INFO] Built {len(energy_bins)} AD energy bins")
    logger.info(f">> energy_bins = {len(energy_bins)}")

    # Bin edges for MF33 projection (MeV → eV inside)
    bin_edges_ev = np.array(
        [b.bin_lower_mev for b in energy_bins] + [energy_bins[-1].bin_upper_mev]
    ) * 1e6
    _step_end(logger, "3", t_step, step_times)

    # ---- STEP 4: Project MF3 + MF33 onto AD bins ----
    t_step = _step_start(logger, "4", "Project MF3 + MF33 onto AD bin grid")
    # to_xs_covmat with sibling_sections + mf3_sections handles:
    #   - Direct NI records (LB=0/1/2/5/6) summed on union grid
    #   - NC LTY=0 sum-rule resolution for derived MTs (e.g. JENDL-5 Fe-56 MT2)
    #   - is_relative tracking per matrix
    # project_to_grid then does relative→absolute via xs_source and projects
    # onto the AD bin grid via piecewise-constant overlap.
    #
    # c0_on_ad: bin-averaged σ over each *original* bin window. We're in the
    # resonance region (MF2 reconstructed via NJOY), so a single bin-centre
    # evaluation is not representative — integrate. This stays consistent with
    # cov_c0_on_ad, which is itself a bin-by-bin overlap projection.
    c0_on_ad = np.array([
        _bin_average_xs(xs_source, b.bin_lower_mev * 1e6, b.bin_upper_mev * 1e6)
        for b in energy_bins
    ], dtype=float)
    xs_cov = mf33_mt.to_xs_covmat(
        sibling_sections=mf33_sections,
        mf3_sections=xs_map,
    )
    cov_c0_on_ad = xs_cov.project_to_grid(
        bin_edges_ev, xs_source=xs_source,
        target_mt=MT_NUMBER, target_isotope=int(mf33_mt._za),
    )
    diag = np.sqrt(np.maximum(np.diag(cov_c0_on_ad), 0.0))
    rel_unc = diag / np.maximum(np.abs(c0_on_ad), 1e-30)
    logger.info(
        f"  c_0 on AD grid: mean={c0_on_ad.mean():.3f} b, "
        f"rel σ_c0 = [{rel_unc.min()*100:.2f}, {rel_unc.max()*100:.2f}]%"
    )
    logger.info(f">> mf33_bins = {cov_c0_on_ad.shape[0]}")
    _step_end(logger, "4", t_step, step_times)

    # ---- STEP 5: Per-bin nominal fits ----
    t_step = _step_start(logger, "5", "Per-bin nominal Legendre fits (Level 2 + AICc)")
    nominal_bins: List[_NominalBin] = []
    n_skipped = 0
    n_skipped_c0 = 0
    # Phase B audit follow-up: count bins where the post-τ AICc rescan picked
    # a different winner than the pre-τ scan. Logged as a single summary line
    # at the end of step 5; per-bin shifts are logged under VERBOSE_DIAGNOSTICS.
    n_bins_post_tau_winner_changed = 0
    # Phase E.2 audit follow-up: track per-bin angular-quality-gate expansions.
    # ``expansion_used`` is the number of bin widths the original window was
    # expanded by to satisfy MIN_ANGULAR_POINTS / MIN_BANDS_COVERED. Aggregated
    # into a histogram at the end of step 5 (always summary, per-bin under verbose).
    expansion_per_bin: List[Tuple[int, float, int]] = []  # (bin_idx, energy_mev, expansion)
    for bin_idx, bin_info in enumerate(energy_bins):
        # Sanity check: skip bins where MF3 c_0 is non-positive or below
        # numerical floor — the multiplicative perturbation factor and the
        # downstream a_l = c_l/c_0 division would explode otherwise.
        if c0_on_ad[bin_idx] < MIN_C0_MF3_BARN:
            n_skipped_c0 += 1
            continue

        res, expansion_used = _filter_with_quality_gate(
            exfor_cache, sorted_energies, energy_bins, bin_idx, logger,
        )
        if res is None:
            n_skipped += 1
            continue
        expansion_per_bin.append((bin_idx, bin_info.energy_mev, expansion_used))
        exfor_df, experiments_info, kernel_weights, _diag, _floor_stats = res
        exfor_df = _attach_sigma_sys(exfor_df)
        try:
            nominal_coeffs, info = _nominal_fit_one_bin(exfor_df, kernel_weights)
        except Exception as e:
            logger.warning(
                f"  E={bin_info.energy_mev:.4f} MeV: nominal fit failed ({e}) — skipping"
            )
            n_skipped += 1
            continue
        c0_fit = float(nominal_coeffs[0])
        c0_mf3 = float(c0_on_ad[bin_idx])
        # Bin window after quality-gate expansion (same as the original bin if
        # no expansion happened). Used to compute the σ_nom that the multiplicative
        # MC factor is normalised by, so the factor is consistent with the energy
        # range the EXFOR data covers.
        post_QG_left = max(0, bin_idx - expansion_used)
        post_QG_right = min(len(energy_bins) - 1, bin_idx + expansion_used)
        post_QG_lo_mev = energy_bins[post_QG_left].bin_lower_mev
        post_QG_hi_mev = energy_bins[post_QG_right].bin_upper_mev
        if expansion_used > 0:
            c0_mf3_post_QG = _bin_average_xs(
                xs_source, post_QG_lo_mev * 1e6, post_QG_hi_mev * 1e6,
            )
        else:
            c0_mf3_post_QG = c0_mf3
        rel_disc = abs(c0_fit - c0_mf3) / max(c0_mf3, 1e-30)
        if rel_disc > MF3_AD_C0_DISCREPANCY_WARN:
            logger.info(
                f"  E={bin_info.energy_mev:.4f} MeV: c0_fit={c0_fit:.3f} vs "
                f"c0_MF3={c0_mf3:.3f} ({rel_disc*100:.1f}% disagreement)"
            )
        # bin_info.index is the key used by overlap_weights (and v2's KW worker).
        # In v3 the union grid is always inside ENERGY_RANGE_MEV, so this equals
        # the position in the energy_bins list — but assert it so we don't
        # silently break if someone widens the grid later.
        if bin_info.index != bin_idx:
            raise RuntimeError(
                f"v3 assumes energy_bins[k].index == k (got {bin_info.index} at "
                f"position {bin_idx}). Update the bin-id mapping if widening the grid."
            )
        aicc_weights, nominal_coeffs_by_degree = (
            _extract_aicc_weights_and_per_degree_coeffs(
                info, nominal_coeffs, MAX_LEGENDRE_DEGREE,
            )
        )
        # Phase B: track + log post-τ AICc winner shifts.
        if info.get('post_tau_winner_changed'):
            n_bins_post_tau_winner_changed += 1
            if VERBOSE_DIAGNOSTICS:
                pre = info.get('all_degrees_info_pre_tau') or {}
                pre_winner = (
                    min(pre.items(), key=lambda kv: kv[1]['aicc'])[0]
                    if pre else None
                )
                logger.info(
                    f"  E={bin_info.energy_mev:.4f} MeV: AICc winner shifted "
                    f"L={pre_winner} -> L={info['degree']} after τ rescan"
                )
        nominal_bins.append(_NominalBin(
            bin_idx=bin_info.index,
            dsigma_pos=len(nominal_bins),  # position in compacted dsigma_vec
            energy_mev=bin_info.energy_mev,
            bin_lower_mev=bin_info.bin_lower_mev,
            bin_upper_mev=bin_info.bin_upper_mev,
            exfor_df=exfor_df,
            kernel_weights=kernel_weights,
            experiments_info=experiments_info,
            frozen_degree=info['degree'],
            nominal_coeffs=nominal_coeffs,
            tau_info=info.get('tau_info', {}),
            c0_mf3=c0_mf3,
            c0_mf3_post_QG=c0_mf3_post_QG,
            aicc_weights=aicc_weights,
            nominal_coeffs_by_degree=nominal_coeffs_by_degree,
            # NominalFitResult-shaped fields for run_mc_with_kernel_weights:
            energy_index=bin_info.index,
            has_data=True,
            mc_order_cap=None,
            interpolated=False,
        ))
    logger.info(
        f"  [INFO] [FIT] Nominal fits: {len(nominal_bins)} bins fitted, "
        f"{n_skipped} skipped (quality/fit), {n_skipped_c0} skipped (c0 < {MIN_C0_MF3_BARN} b)"
    )
    logger.info(f">> bins_total = {len(energy_bins)}")
    logger.info(f">> bins_fitted = {len(nominal_bins)}")
    logger.info(f">> bins_skipped_quality = {n_skipped}")
    logger.info(f">> bins_skipped_c0_floor = {n_skipped_c0}")
    if RERUN_AICC_POST_TAU and len(nominal_bins) > 0:
        logger.info(
            f"  AICc winner shifted in {n_bins_post_tau_winner_changed}/"
            f"{len(nominal_bins)} bins after τ rescan"
        )

    # Phase E.2 audit follow-up: angular-quality-gate expansion histogram.
    if expansion_per_bin:
        from collections import Counter
        counts = Counter(exp for _, _, exp in expansion_per_bin)
        n_exp = sum(c for k, c in counts.items() if k > 0)
        logger.info(
            f"  Angular-quality-gate expansions: {n_exp}/{len(expansion_per_bin)} "
            f"bins expanded; histogram (expansion=count): "
            f"{dict(sorted(counts.items()))}"
        )
        if VERBOSE_DIAGNOSTICS and n_exp > 0:
            for bi, e_mev, exp in expansion_per_bin:
                if exp > 0:
                    logger.info(
                        f"    bin {bi} (E={e_mev:.4f} MeV): expansion={exp}"
                    )

    if not nominal_bins:
        _step_end(logger, "5", t_step, step_times)
        raise RuntimeError("No usable bins after nominal fits — abort.")

    # Compact dsigma covariance to the bins we actually fitted
    bin_indices_used = np.array([nb.bin_idx for nb in nominal_bins])
    cov_c0_used = cov_c0_on_ad[np.ix_(bin_indices_used, bin_indices_used)]
    # σ_nom on the compacted bin set, for the MF33 multiplicative factor in MC.
    # Uses the post-quality-gate bin window per bin, so the factor matches the
    # energy range the EXFOR data actually covers.
    c0_post_QG_vec = np.array([nb.c0_mf3_post_QG for nb in nominal_bins], dtype=float)

    # Pre-draw the per-(bin, sample) Legendre degree from each bin's AICc
    # weights. Same matrix is fed into Pass 1 and Pass 2 so the congruence
    # merge keeps correlation/std on aligned models. None ⇒ legacy
    # frozen-degree behaviour everywhere.
    sampled_degrees = _build_sampled_degrees_matrix(
        nominal_bins,
        n_samples=N_SAMPLES,
        seed=BASE_SEED,
        enabled=USE_DEGREE_SAMPLING_IN_MC,
    )
    if sampled_degrees is not None:
        # Per-bin diagnostic: how spread out is the AICc choice across samples?
        deg_var_bins = int(np.sum(
            np.array([len(np.unique(sampled_degrees[k])) for k in range(len(nominal_bins))]) > 1
        ))
        logger.info(
            f"  AICc-weighted degree sampling enabled: "
            f"{deg_var_bins}/{len(nominal_bins)} bins draw >1 distinct degree across samples"
        )
    _step_end(logger, "5", t_step, step_times)

    # ---- STEP 6: Joint MC (two-pass: KW correlations + per-bin variance) ----
    t_step = _step_start(logger, "6",
                         f"Joint MC two-pass — N_SAMPLES={N_SAMPLES}, N_PROCS={N_PROCS}")
    cov_al_al_abs, cov_c0_al_abs, _dsigma_pass1, al_pass1, al_pass2 = run_joint_mc_two_pass(
        nominal_bins,
        energy_bins,
        cov_c0_used,
        c0_post_QG_vec,
        n_samples=N_SAMPLES,
        n_workers=N_PROCS,
        seed=BASE_SEED,
        max_degree=MAX_LEGENDRE_DEGREE,
        max_sample_order=MAX_SAMPLE_ORDER,
        sampled_degrees=sampled_degrees,
        logger=logger,
        step_times=step_times,
    )
    logger.info(
        f"  cov(a_l, a_l') shape {cov_al_al_abs.shape}, "
        f"cov(δσ, a_l) shape {cov_c0_al_abs.shape}"
    )
    logger.info(f">> samples_generated = {N_SAMPLES}")
    logger.info(f">> covariance_shape = {cov_al_al_abs.shape}")
    logger.info(f">> cross_block_shape = {cov_c0_al_abs.shape}")

    # PSD check on the fine joint (δσ, a_l) absolute covariance — catches
    # breakage in the congruence merge of _combine_passes independent of the
    # later relativization or multigroup collapse.
    J_fine_abs = _build_joint_sigma_al_cov(cov_c0_used, cov_c0_al_abs, cov_al_al_abs)
    psd_fine_ok, _ = _verify_psd(
        J_fine_abs, "joint (δσ, a_l) fine absolute", logger, tol=PSD_WARN_TOL,
    )
    _step_end(logger, "6", t_step, step_times)

    L = MAX_LEGENDRE_DEGREE

    # ---- STEP 7: Multigroup collapse (optional) -------------------------------
    multigroup_result = None
    A_0 = None
    cov_c0_al_grouped_abs = None
    cov_aa_grouped_rel = None
    cov_c0_al_grouped_rel = None
    psd_mg_ok: Optional[bool] = None
    if (GENERATE_MULTIGROUP_COVARIANCE
            and MF34_COVARIANCE_TYPE in ("multigroup", "both")):
        t_step = _step_start(logger, "7", "Adaptive multigroup covariance collapse")

        # perform_adaptive_multigroup_collapse expects nominal_results that
        # quack like NominalFitResult — _NominalBin already does (see slots
        # set at the end of run_v3 step 5). It also slices the covariance
        # internally to non-interpolated bins, but v3 never interpolates
        # (skipped bins are dropped, not filled), so the active bins ARE the
        # non-interpolated ones — the cov_al_al_abs we pass in is already
        # compacted to len(nominal_bins).
        corr_aa_abs = cov_to_corr(cov_al_al_abs)

        diag_csv = None
        if SAVE_MULTIGROUP_DIAGNOSTICS_CSV:
            diag_csv = output_path / "multigroup_boundary_decisions.csv"

        multigroup_result = perform_adaptive_multigroup_collapse(
            cov_matrix=cov_al_al_abs,
            corr_matrix=corr_aa_abs,
            nominal_results=nominal_bins,
            energy_bins=[energy_bins[nb.bin_idx] for nb in nominal_bins],
            max_order=L,
            rho_min=MULTIGROUP_RHO_MIN,
            sigma_ratio_max=MULTIGROUP_SIGMA_RATIO_MAX,
            variance_percentile_min=MULTIGROUP_VARIANCE_PCT_MIN,
            variance_percentile_max=MULTIGROUP_VARIANCE_PCT_MAX,
            variance_ratio_ref=MULTIGROUP_VARIANCE_RATIO_REF,
            logger=logger,
            diagnostics_file=diag_csv,
        )
        n_groups = len(multigroup_result.groups)
        logger.info(
            f"  Multigroup collapse: {len(nominal_bins)} fine bins -> "
            f"{n_groups} groups ({len(nominal_bins) / max(n_groups, 1):.1f}x compression)"
        )

        # Phase E.3 audit follow-up: post-merge consistency diagnostics. The
        # grouping decision uses only l=1 adjacent correlation, but the same
        # groups are applied to every order and to the σ–a_l cross block. Log
        # how many merged groups have weak l=2 / l=3 adjacent correlation
        # (candidate vetoes if we ever add higher-order grouping criteria).
        n_weak_l2 = 0
        n_weak_l3 = 0
        for grp in multigroup_result.groups:
            if len(grp) < 2:
                continue
            for l in (2, 3):
                if l > MAX_LEGENDRE_DEGREE:
                    continue
                corrs = []
                for k in range(len(grp) - 1):
                    i, j = grp[k], grp[k + 1]
                    s_i = float(cov_al_al_abs[
                        i * MAX_LEGENDRE_DEGREE + (l - 1),
                        i * MAX_LEGENDRE_DEGREE + (l - 1),
                    ])
                    s_j = float(cov_al_al_abs[
                        j * MAX_LEGENDRE_DEGREE + (l - 1),
                        j * MAX_LEGENDRE_DEGREE + (l - 1),
                    ])
                    if s_i <= 0 or s_j <= 0:
                        continue
                    rho = cov_al_al_abs[
                        i * MAX_LEGENDRE_DEGREE + (l - 1),
                        j * MAX_LEGENDRE_DEGREE + (l - 1),
                    ] / np.sqrt(s_i * s_j)
                    corrs.append(float(rho))
                if not corrs:
                    continue
                rho_min = min(corrs)
                if rho_min < 0.5:
                    if l == 2:
                        n_weak_l2 += 1
                    else:
                        n_weak_l3 += 1
                if VERBOSE_DIAGNOSTICS and rho_min < 0.5:
                    logger.info(
                        f"    Group [{grp[0]}-{grp[-1]}]: l={l} min adj corr "
                        f"= {rho_min:.3f} (l=1 grouping criterion drove the merge)"
                    )
        logger.info(
            f"  Multigroup post-merge: l=2 weak adj-corr in {n_weak_l2} groups, "
            f"l=3 weak adj-corr in {n_weak_l3} groups (threshold ρ<0.5)"
        )

        # Collapse the v3 cross block on the same groups
        if WRITE_MF34_L0_ROW:
            fine_bin_widths_mev = np.array(
                [nb.bin_upper_mev - nb.bin_lower_mev for nb in nominal_bins],
                dtype=float,
            )
            A_0 = build_l0_row_aggregator(
                multigroup_result.groups, fine_bin_widths_mev, len(nominal_bins),
            )
            A = multigroup_result.aggregation_matrix
            # Apply the same diagonal variance compensation S that the a_l block
            # received in apply_percentile_variance_scaling. Without this, the
            # cross block is under-scaled relative to the (now-inflated) grouped
            # a_l variances and the joint correlation magnitudes are wrong.
            # σ-axis is intentionally untouched: cov_c0_used was not scaled
            # (it's the MF33 projection, untouched by the percentile scaling).
            S_diag = multigroup_result.scale_factors  # shape (n_groups * L,)
            cov_c0_al_grouped_abs = (A_0 @ cov_c0_al_abs @ A.T) * S_diag[None, :]
            logger.info(
                f"  Cross block collapsed: {cov_c0_al_abs.shape} -> "
                f"{cov_c0_al_grouped_abs.shape}"
            )

            # Diagnostic: width-aggregated MF33 reference rel-σ on grouped grid
            mf33_grouped_ref = A_0 @ cov_c0_used @ A_0.T
            c0_fine_compact = np.array([nb.c0_mf3 for nb in nominal_bins])
            c0_grouped_ref = A_0 @ c0_fine_compact
            rel_ref = np.sqrt(np.maximum(np.diag(mf33_grouped_ref), 0.0)) / np.maximum(
                np.abs(c0_grouped_ref), 1e-30
            )
            logger.info(
                f"  MG MF33 ref on grouped grid: rel σ_c0 ∈ "
                f"[{rel_ref.min() * 100:.2f}, {rel_ref.max() * 100:.2f}]% "
                f"(should track the fine-grid range logged earlier)"
            )

            # PSD check on the joint (δσ, a_l) absolute covariance on the
            # multigroup grid — load-bearing test for the cross-block math.
            J_mg_abs = _build_joint_sigma_al_cov(
                mf33_grouped_ref,
                cov_c0_al_grouped_abs,
                multigroup_result.cov_grouped,
            )
            psd_mg_ok, _ = _verify_psd(
                J_mg_abs, "joint (δσ, a_l) multigroup absolute", logger,
                tol=PSD_WARN_TOL,
            )

        # Relativize the multigroup (a_l, a_l') and cross matrices
        nom_params_fine = np.zeros(len(nominal_bins) * L, dtype=float)
        for k, nb in enumerate(nominal_bins):
            a_nom = endf_normalize_legendre_coeffs(
                nb.nominal_coeffs[:L + 1], include_a0=False,
            )
            n = min(len(a_nom), L)
            nom_params_fine[k * L: k * L + n] = a_nom[:n]
        nom_grouped = multigroup_result.aggregation_matrix @ nom_params_fine

        denom_aa_g = np.outer(nom_grouped, nom_grouped)
        safe_aa_g = np.abs(denom_aa_g) > 1e-12
        cov_aa_grouped_rel = np.zeros_like(multigroup_result.cov_grouped)
        cov_aa_grouped_rel[safe_aa_g] = (
            multigroup_result.cov_grouped[safe_aa_g] / denom_aa_g[safe_aa_g]
        )
        cov_aa_grouped_rel[np.abs(cov_aa_grouped_rel) < 1e-15] = 0.0

        if WRITE_MF34_L0_ROW and cov_c0_al_grouped_abs is not None:
            c0_fine_compact = np.array([nb.c0_mf3 for nb in nominal_bins])
            c0_grouped = A_0 @ c0_fine_compact
            denom_ca_g = np.outer(c0_grouped, nom_grouped)
            safe_ca_g = np.abs(denom_ca_g) > 1e-12
            cov_c0_al_grouped_rel = np.zeros_like(cov_c0_al_grouped_abs)
            cov_c0_al_grouped_rel[safe_ca_g] = (
                cov_c0_al_grouped_abs[safe_ca_g] / denom_ca_g[safe_ca_g]
            )
            cov_c0_al_grouped_rel[np.abs(cov_c0_al_grouped_rel) < 1e-15] = 0.0

        _verify_psd(
            cov_aa_grouped_rel, "(a_l, a_l') multigroup relative", logger,
            tol=PSD_WARN_TOL,
        )
        logger.info(f">> multigroup_bins = {len(multigroup_result.groups)}")
        _step_end(logger, "7", t_step, step_times)
    else:
        logger.info("")
        logger.info("#-- STEP 7: Adaptive multigroup covariance — SKIPPED "
                    f"(GENERATE_MULTIGROUP_COVARIANCE={GENERATE_MULTIGROUP_COVARIANCE}, "
                    f"MF34_COVARIANCE_TYPE={MF34_COVARIANCE_TYPE})")

    # ---- STEP 8: Pad to the FULL bin grid + relativize fine matrices ----
    t_step = _step_start(logger, "8", "Pad to full grid + relativize covariance")
    # MF34 LB=5 / LB=6 store *relative* covariance:
    #   Cov_rel(a_li, a_l'j) = Cov_abs(a_li, a_l'j) / (a_li_nom * a_l'j_nom)
    #   Cov_rel(δσ_i, a_lj)  = Cov_abs(δσ_i, a_lj)  / (σ_nom_i * a_lj_nom)
    # Skipped bins (quality gate, fit failure, c0 floor) get zero rows/cols
    # so the output grid stays contiguous and aligned with the AD bin set.
    n_bins_full = len(energy_bins)
    n_params_full = n_bins_full * L

    # Indexing maps: compacted (MC) ↔ full (output)
    active_bin_idxs = np.array([nb.bin_idx for nb in nominal_bins], dtype=int)
    active_param_idxs = np.concatenate([
        np.arange(nb.bin_idx * L, nb.bin_idx * L + L) for nb in nominal_bins
    ]).astype(int)

    # Embed the absolute covariances onto the full grid
    cov_al_al_full = np.zeros((n_params_full, n_params_full), dtype=float)
    cov_al_al_full[np.ix_(active_param_idxs, active_param_idxs)] = cov_al_al_abs

    cov_c0_al_full = np.zeros((n_bins_full, n_params_full), dtype=float)
    cov_c0_al_full[np.ix_(active_bin_idxs, active_param_idxs)] = cov_c0_al_abs

    # Build a_l_nom on the full grid (zero where the bin was skipped)
    a_l_nom_full = np.zeros(n_params_full, dtype=float)
    for nb in nominal_bins:
        a_nom = endf_normalize_legendre_coeffs(
            nb.nominal_coeffs[:L + 1], include_a0=False,
        )
        if a_nom.size < L:
            a_nom = np.pad(a_nom, (0, L - a_nom.size))
        a_l_nom_full[nb.bin_idx * L:(nb.bin_idx + 1) * L] = a_nom[:L]

    # σ_nom for the L=0 cross-row: bin-averaged σ on the *original* bin window.
    # This must match the σ that MF33 was projected against — c0_mf3, not
    # c0_mf3_post_QG (which carries the post-expansion window).
    c0_nom_full = np.zeros(n_bins_full, dtype=float)
    for nb in nominal_bins:
        c0_nom_full[nb.bin_idx] = nb.c0_mf3

    # Relativize Cov(a_l, a_l').  Use 6-digit ENDF precision floor for the
    # outer-product denominator; entries with near-zero a_l_nom are set to
    # zero (consistent with v2's compute_covariance_from_samples).
    denom_aa = np.outer(a_l_nom_full, a_l_nom_full)
    safe_aa = np.abs(denom_aa) > 1e-12
    cov_al_al_rel = np.zeros_like(cov_al_al_full)
    cov_al_al_rel[safe_aa] = cov_al_al_full[safe_aa] / denom_aa[safe_aa]
    cov_al_al_rel[np.abs(cov_al_al_rel) < 1e-15] = 0.0

    # Phase E.1 audit follow-up: list non-zero abs-cov entries that the
    # near-zero-denominator floor zeroed during relativization. MF34 LB=5
    # is a relative-covariance format, so any entry where a_l_nom crosses
    # zero is an information loss baked into the format. The summary line
    # is unconditional; the top-N detail list is gated by VERBOSE_DIAGNOSTICS.
    _zeroed_aa = (~safe_aa) & (np.abs(cov_al_al_full) > 0)
    n_lost_aa = int(_zeroed_aa.sum())
    n_lost_aa_off = int((_zeroed_aa & ~np.eye(_zeroed_aa.shape[0], dtype=bool)).sum())
    if n_lost_aa > 0:
        max_lost_std_aa = float(np.sqrt(np.abs(cov_al_al_full[_zeroed_aa]).max()))
        logger.info(
            f"  Relativization (a_l, a_l'): zeroed {n_lost_aa} nonzero entries "
            f"({n_lost_aa_off} off-diagonal); max lost |std| = {max_lost_std_aa:.3e}"
        )
        if VERBOSE_DIAGNOSTICS:
            absvals = np.abs(cov_al_al_full[_zeroed_aa])
            top = np.argsort(absvals)[::-1][:20]
            zeroed_idx = np.argwhere(_zeroed_aa)
            for k in top:
                i, j = int(zeroed_idx[k, 0]), int(zeroed_idx[k, 1])
                bi, li = i // MAX_LEGENDRE_DEGREE, (i % MAX_LEGENDRE_DEGREE) + 1
                bj, lj = j // MAX_LEGENDRE_DEGREE, (j % MAX_LEGENDRE_DEGREE) + 1
                logger.info(
                    f"    lost (bin={bi}, l={li}) ↔ (bin={bj}, l={lj}): "
                    f"|Cov_abs|={absvals[k]:.3e}"
                )

    # Relativize Cov(δσ, a_l).  Asymmetric: row i divided by σ_nom_i, col j
    # by a_l_nom_j.
    denom_ca = np.outer(c0_nom_full, a_l_nom_full)
    safe_ca = np.abs(denom_ca) > 1e-12
    cov_c0_al_rel = np.zeros_like(cov_c0_al_full)
    cov_c0_al_rel[safe_ca] = cov_c0_al_full[safe_ca] / denom_ca[safe_ca]
    cov_c0_al_rel[np.abs(cov_c0_al_rel) < 1e-15] = 0.0

    _zeroed_ca = (~safe_ca) & (np.abs(cov_c0_al_full) > 0)
    n_lost_ca = int(_zeroed_ca.sum())
    if n_lost_ca > 0:
        max_lost_std_ca = float(np.sqrt(np.abs(cov_c0_al_full[_zeroed_ca]).max()))
        logger.info(
            f"  Relativization (δσ, a_l): zeroed {n_lost_ca} nonzero entries; "
            f"max lost |std| = {max_lost_std_ca:.3e}"
        )

    n_skipped_in_grid = n_bins_full - len(nominal_bins)
    if n_skipped_in_grid > 0:
        logger.info(
            f"  MF34 grid: {n_bins_full} bins ({len(nominal_bins)} fitted, "
            f"{n_skipped_in_grid} zero-cov padded for skipped bins)"
        )

    _verify_psd(cov_al_al_rel, "(a_l, a_l') fine relative", logger, tol=PSD_WARN_TOL)

    full_energy_grid_ev = bin_edges_ev  # already (lo_1, …, lo_N, hi_N)
    _step_end(logger, "8", t_step, step_times)

    # ---- STEP 8b: Save covariance .npy artifacts (mirror v2 names) -----------
    t_step = _step_start(logger, "8b", "Save covariance .npy artifacts (optional)")
    if SAVE_COVARIANCE_FILES:
        np.save(output_path / "legendre_covariance.npy", cov_al_al_rel)
        if SAVE_CORRELATION_MATRICES:
            np.save(output_path / "legendre_correlation.npy",
                    cov_to_corr(cov_al_al_rel))
        if WRITE_MF34_L0_ROW:
            np.save(output_path / "legendre_cross_c0_al.npy", cov_c0_al_rel)
        if multigroup_result is not None:
            np.save(output_path / "legendre_covariance_multigroup.npy",
                    cov_aa_grouped_rel)
            if SAVE_CORRELATION_MATRICES:
                np.save(output_path / "legendre_correlation_multigroup.npy",
                        cov_to_corr(cov_aa_grouped_rel))
            np.save(output_path / "multigroup_boundaries_ev.npy",
                    multigroup_result.group_boundaries_ev)
            np.save(output_path / "multigroup_mean_coeffs.npy",
                    multigroup_result.mean_grouped)
            if WRITE_MF34_L0_ROW and cov_c0_al_grouped_rel is not None:
                np.save(output_path / "legendre_cross_c0_al_multigroup.npy",
                        cov_c0_al_grouped_rel)
        logger.info(f"  Saved .npy covariance artifacts to {output_path}")
    else:
        logger.info("  SAVE_COVARIANCE_FILES=False — skipping .npy artifacts")
    _step_end(logger, "8b", t_step, step_times)

    # ---- STEP 8c: Splice v3 nominal a_l into MF4 (audit follow-up #2) -------
    # Phase F audit follow-up: the source ENDF carries an MF4 evaluated by the
    # original group, but v3 relativizes MF34 against its OWN per-bin Legendre
    # nominal. Without this step, downstream consumers multiplying MF34_rel by
    # MF4 get the wrong absolute covariance. write_nominal_endf (v2 helper at
    # exfor_utils.py:4880) splices v3's nominal into the source MF4 only inside
    # ENERGY_RANGE_MEV; outside is byte-preserved. The MF4-spliced file then
    # becomes the source for Step 9's MF34 writer.
    mf4_source_for_mf34 = ENDF_FILE
    if REWRITE_MF4_FROM_NOMINAL:
        t_step = _step_start(logger, "8c", "Rewrite MF4 from v3 nominal coefficients")
        try:
            mf4_spliced_path = write_nominal_endf(
                original_endf_file=ENDF_FILE,
                mt_number=MT_NUMBER,
                nominal_results=nominal_bins,
                output_dir=str(output_path),
                energy_bins=energy_bins,
                energy_range_mev=ENERGY_RANGE_MEV,
            )
            mf4_source_for_mf34 = mf4_spliced_path
            logger.info(f"  MF4 splice: wrote {mf4_spliced_path}")
            logger.info(
                f"  MF4 splice range: [{ENERGY_RANGE_MEV[0]:.4f}, "
                f"{ENERGY_RANGE_MEV[1]:.4f}] MeV ({len(nominal_bins)} bins)"
            )
            if VERBOSE_DIAGNOSTICS:
                # Per-bin |Δa_l| log so we can sanity-check the splice. We
                # don't have the raw source MF4 a_l handy here without re-
                # parsing; the spliced file IS readable, so future diagnostic
                # work can re-load both and diff. For now we note the bin
                # count and let the user inspect with a notebook.
                logger.info(
                    "  (per-bin |Δa_l| MF4 vs v3 diff is best inspected in a "
                    "notebook by parsing both source and spliced MF4 sections)"
                )
        except Exception as exc:
            logger.error(
                f"  MF4 splice FAILED: {exc} — falling back to source ENDF "
                f"as MF34 base (MF34 will not match the source's MF4)"
            )
            mf4_source_for_mf34 = ENDF_FILE
        _step_end(logger, "8c", t_step, step_times)
    else:
        logger.info("")
        logger.info("#-- STEP 8c: MF4 rewrite — SKIPPED "
                    f"(REWRITE_MF4_FROM_NOMINAL={REWRITE_MF4_FROM_NOMINAL})")

    # ---- STEP 9: Write MF34 (fine and/or multigroup) ----
    t_step = _step_start(logger, "9", "Write MF34 covariance file(s)")
    mf34_fine = None
    mf34_mg = None
    mg_out_path = None

    # Phase C audit follow-up: cap MF34 output at MAX_SAMPLE_ORDER. Orders
    # above MAX_SAMPLE_ORDER were not actually sampled (frozen at nominal
    # in both passes) so writing them with zero variance would mislead
    # downstream consumers. We then graft any non-zero higher-order pairs
    # from the source MF34 — but only OUTSIDE our energy range, since
    # inside it we don't trust the source at orders we didn't sample.
    if MF34_COVARIANCE_TYPE in ("fine", "both"):
        cov_al_al_rel_capped = _slice_orders(
            cov_al_al_rel, n_bins_full, MAX_LEGENDRE_DEGREE, MAX_SAMPLE_ORDER,
        )
        cov_c0_al_rel_capped = (
            _slice_orders(cov_c0_al_rel, n_bins_full,
                          MAX_LEGENDRE_DEGREE, MAX_SAMPLE_ORDER)
            if WRITE_MF34_L0_ROW else None
        )
        mf34_fine = create_mf34_from_covariance(
            cov_matrix=cov_al_al_rel_capped,
            energy_grid_ev=full_energy_grid_ev,
            max_order=MAX_SAMPLE_ORDER,
            za=float(mf33_mt._za),
            awr=float(mf33_mt._awr),
            mat=int(mf33_mt._mat),
            mt=MT_NUMBER,
            cov_c0_cl=cov_c0_al_rel_capped,
        )
        # Splice source MF34 outside our energy range so we don't lose its
        # low-order data above E_max or below E_min, and so any non-zero
        # higher-order pairs are preserved (outside our range only — see
        # _merge_source_mf34_outside_range docstring). Use the pipeline MF34's
        # own energy grid as the splice boundary so source-outside records
        # don't overlap pipeline-inside records under any union-grid logic.
        splice_e_min_ev = float(full_energy_grid_ev[0])
        splice_e_max_ev = float(full_energy_grid_ev[-1])
        merge_counts = _merge_source_mf34_outside_range(
            pipeline_mf34=mf34_fine,
            source_endf_path=mf4_source_for_mf34,
            mt=MT_NUMBER,
            max_sample_order=MAX_SAMPLE_ORDER,
            e_min_ev=splice_e_min_ev,
            e_max_ev=splice_e_max_ev,
            logger=logger,
        )
        logger.info(
            f"  MF34 fine: capped at order {MAX_SAMPLE_ORDER}; "
            f"low-order source outside range appended to "
            f"{merge_counts['n_low_appended']} pair(s) "
            f"(dropped {merge_counts['n_low_dropped']} all-zero/empty); "
            f"higher-order source grafted as {merge_counts['n_high_grafted']} "
            f"new pair(s) (dropped {merge_counts['n_high_dropped_zero']} all-zero)"
        )
        write_mf34_to_file(
            source_endf=mf4_source_for_mf34,
            mf34=mf34_fine,
            output_path=endf_output_path,
            replace_existing=True,
        )
        logger.info(f"Wrote fine MF34 to {endf_output_path}")

    if (MF34_COVARIANCE_TYPE in ("multigroup", "both")
            and multigroup_result is not None):
        n_groups = len(multigroup_result.groups)
        cov_aa_grouped_rel_capped = _slice_orders(
            cov_aa_grouped_rel, n_groups, MAX_LEGENDRE_DEGREE, MAX_SAMPLE_ORDER,
        )
        cov_c0_al_grouped_rel_capped = (
            _slice_orders(cov_c0_al_grouped_rel, n_groups,
                          MAX_LEGENDRE_DEGREE, MAX_SAMPLE_ORDER)
            if WRITE_MF34_L0_ROW and cov_c0_al_grouped_rel is not None else None
        )
        mf34_mg = create_mf34_from_covariance(
            cov_matrix=cov_aa_grouped_rel_capped,
            energy_grid_ev=multigroup_result.group_boundaries_ev,
            max_order=MAX_SAMPLE_ORDER,
            za=float(mf33_mt._za),
            awr=float(mf33_mt._awr),
            mat=int(mf33_mt._mat),
            mt=MT_NUMBER,
            cov_c0_cl=cov_c0_al_grouped_rel_capped,
        )
        mg_splice_e_min_ev = float(multigroup_result.group_boundaries_ev[0])
        mg_splice_e_max_ev = float(multigroup_result.group_boundaries_ev[-1])
        mg_merge_counts = _merge_source_mf34_outside_range(
            pipeline_mf34=mf34_mg,
            source_endf_path=mf4_source_for_mf34,
            mt=MT_NUMBER,
            max_sample_order=MAX_SAMPLE_ORDER,
            e_min_ev=mg_splice_e_min_ev,
            e_max_ev=mg_splice_e_max_ev,
            logger=logger,
        )
        logger.info(
            f"  MF34 multigroup: capped at order {MAX_SAMPLE_ORDER}; "
            f"low-order source outside range appended to "
            f"{mg_merge_counts['n_low_appended']} pair(s) "
            f"(dropped {mg_merge_counts['n_low_dropped']} all-zero/empty); "
            f"higher-order source grafted as {mg_merge_counts['n_high_grafted']} "
            f"new pair(s) (dropped {mg_merge_counts['n_high_dropped_zero']} all-zero)"
        )
        if endf_output_path.endswith('.endf'):
            mg_out_path = endf_output_path[:-len('.endf')] + '_mg.endf'
        else:
            mg_out_path = endf_output_path + '_mg.endf'
        write_mf34_to_file(
            source_endf=mf4_source_for_mf34,
            mf34=mf34_mg,
            output_path=mg_out_path,
            replace_existing=True,
        )
        logger.info(f"Wrote multigroup MF34 to {mg_out_path}")
    _step_end(logger, "9", t_step, step_times)

    # ---- STEP 10: TMC + raw KW parquets (mirror v2 names + schema) -----------
    # Both parquets are keyed by the compacted (active) bin set; skipped bins
    # don't appear, matching v2 which also restricts to bins with data.
    t_step = _step_start(logger, "10", "Write TMC / raw-KW sample parquets (optional)")
    active_energy_indices = [int(nb.bin_idx) for nb in nominal_bins]

    if SAVE_RAW_KW_PARQUET:
        try:
            raw_kw_path = save_legendre_matrix_to_parquet(
                nominal_results=nominal_bins,
                sample_matrix=al_pass1,
                energy_indices=active_energy_indices,
                output_dir=str(output_path),
                max_degree=MAX_LEGENDRE_DEGREE,
                filename='legendre_samples_raw_kw.parquet',
            )
            logger.info(f"  Raw KW samples parquet: {raw_kw_path}")
        except Exception as exc:
            logger.error(f"  Failed to save raw KW parquet: {exc}")

    if SAVE_TMC_PARQUET:
        try:
            # Affine rescale Pass-1 samples to Pass-2 marginals — preserves
            # Pearson correlations exactly while matching the published
            # MF34's per-parameter mean/std. Mirrors v2's TMC construction.
            mean_kw = al_pass1.mean(axis=0)
            std_kw = al_pass1.std(axis=0, ddof=0)
            mean_pb = al_pass2.mean(axis=0)
            std_pb = al_pass2.std(axis=0, ddof=0)
            eps = 1e-30
            n_zero_std = int(np.sum(std_kw < eps))
            if n_zero_std > 0:
                logger.info(
                    f"  [TMC] {n_zero_std}/{std_kw.size} parameters had "
                    f"std_kw < {eps:.0e}; TMC values pinned to Pass-2 mean"
                )
            scale = np.where(std_kw >= eps, std_pb / np.maximum(std_kw, eps), 0.0)
            mat_tmc = mean_pb[None, :] + (al_pass1 - mean_kw[None, :]) * scale[None, :]
            tmc_path = save_legendre_matrix_to_parquet(
                nominal_results=nominal_bins,
                sample_matrix=mat_tmc,
                energy_indices=active_energy_indices,
                output_dir=str(output_path),
                max_degree=MAX_LEGENDRE_DEGREE,
                filename='legendre_samples_tmc.parquet',
            )
            logger.info(f"  TMC parquet: {tmc_path}")
            logger.info(
                "  [TMC] Pass-1 Pearson correlations preserved exactly; "
                "marginals match Pass-2 mean/std (recommended TMC input). "
                "Filter df[~df.is_nominal] before computing sample statistics."
            )
        except Exception as exc:
            logger.error(f"  Failed to save TMC parquet: {exc}")
    _step_end(logger, "10", t_step, step_times)

    # ---- SUMMARY ----
    total_time = time.perf_counter() - t_pipeline_start
    logger.info("")
    logger.info("#== SUMMARY ================================================================")
    logger.info(f">> total_elapsed = {total_time:.2f}s ({_fmt_hms(total_time)})")
    logger.info(f">> n_procs = {N_PROCS}")
    logger.info(f">> n_samples = {N_SAMPLES}")
    logger.info(f">> bins_total = {len(energy_bins)}")
    logger.info(f">> bins_fitted = {len(nominal_bins)}")
    logger.info(f">> bins_skipped_quality = {n_skipped}")
    logger.info(f">> bins_skipped_c0_floor = {n_skipped_c0}")
    logger.info(f">> covariance_shape = {cov_al_al_rel.shape}")
    logger.info(f">> cross_block_shape = {cov_c0_al_rel.shape}")
    if multigroup_result is not None:
        logger.info(f">> multigroup_bins = {len(multigroup_result.groups)}")
    logger.info(f">> joint_psd_fine = {'ok' if psd_fine_ok else 'WARNING'}")
    if psd_mg_ok is not None:
        logger.info(f">> joint_psd_mg = {'ok' if psd_mg_ok else 'WARNING'}")
    logger.info("")
    logger.info("  Per-step elapsed (seconds):")
    step_labels = {
        "1":  "Load EXFOR",
        "2":  "Read ENDF + NJOY reconstruct",
        "3":  "Compute AD energy bins",
        "4":  "Project MF3 + MF33 onto AD bins",
        "5":  "Per-bin nominal fits",
        "6":  "Joint MC two-pass",
        "6a": "  Pass 1 (KW + MF33 shared)",
        "6b": "  Pass 2 (independent per-bin)",
        "6c": "  Combine (congruence merge)",
        "7":  "Multigroup collapse",
        "8":  "Pad + relativize fine matrices",
        "8b": "Save .npy artifacts",
        "8c": "Splice v3 nominal into MF4",
        "9":  "Write MF34",
        "10": "Write TMC / raw-KW parquets",
    }
    for key in ["1", "2", "3", "4", "5", "6", "6a", "6b", "6c",
                "7", "8", "8b", "8c", "9", "10"]:
        if key in step_times:
            label = step_labels.get(key, key)
            logger.info(f"    STEP {key:<3} {label:<40} {step_times[key]:7.2f}s")
    accounted = sum(v for k, v in step_times.items() if k in ("1","2","3","4","5","6","7","8","8b","8c","9","10"))
    logger.info(f"    {'(sum of accounted top-level steps)':<48} {accounted:7.2f}s")
    logger.info(f"    {'(total pipeline elapsed)':<48} {total_time:7.2f}s")
    logger.info("")
    logger.info("  Output files:")
    if mf34_fine is not None:
        logger.info(f"    fine MF34         -> {endf_output_path}")
    if mf34_mg is not None and mg_out_path is not None:
        logger.info(f"    multigroup MF34   -> {mg_out_path}")
    if SAVE_TMC_PARQUET:
        logger.info(f"    TMC parquet       -> {output_path / 'legendre_samples_tmc.parquet'}")
    if SAVE_RAW_KW_PARQUET:
        logger.info(f"    raw KW parquet    -> {output_path / 'legendre_samples_raw_kw.parquet'}")
    if SAVE_MULTIGROUP_DIAGNOSTICS_CSV and multigroup_result is not None:
        logger.info(f"    MG diagnostics    -> {output_path / 'multigroup_boundary_decisions.csv'}")
    logger.info("#== END SUMMARY ============================================================")

    # Return the primary artifact: prefer fine when available, else multigroup
    if mf34_fine is not None:
        return mf34_fine, cov_al_al_rel, cov_c0_al_rel
    return mf34_mg, cov_aa_grouped_rel, cov_c0_al_grouped_rel


if __name__ == "__main__":
    run_v3()
