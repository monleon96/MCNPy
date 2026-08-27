"""
Precompute chi^2 data for Fe-56 elastic angular distributions — PREDICTIVE scenario.

Methodology: every evaluation is judged by **what it actually ships**. Its own MF3
and its own MF4 are forward-folded through each experiment's TOF energy
resolution, and the variance budget is that library's own MF34 **and MF33**.

This is the "S1 / final-file predictive" scenario of `docs/chi2-mf4/mf3_mf33_roadmap.md`.
It supersedes three earlier scripts, each of which is a special case of it:

    precompute_chi2_library_c0.py    same covariance (MF34+MF33), but no fold
    precompute_chi2_folded_c0.py     folds sigma only, MF34-only covariance
    precompute_chi2_folded_al_c0.py  folds sigma and a_l separately, MF34-only

What it is NOT: `precompute_chi2_exfor_c0.py` remains a separate, complementary
scenario, not an inferior one. There c_0 is fitted from the EXFOR data, which
neutralizes magnitude and isolates angular *shape*; each library's MF3 and MF33
correctly play no part. Here magnitude is deliberately on trial.

Why MF33 belongs here and not there
-----------------------------------
With c_0 fitted from the data, adding a library's declared magnitude uncertainty
only cushions whoever declares more — it rewards a large MF33. Once c_0 comes
from each library's own MF3, the central value is on the hook too: a library
whose magnitude is wrong pays for it in the residual, and its MF33 has to earn
its place in the denominator. Only then is the magnitude channel actually tested.

The fold: average the PRODUCT, not the factors
----------------------------------------------
A TOF detector at nominal energy E integrates the real differential cross
section over its resolution, so the measured quantity is

    y_eval(mu) = < sigma(E') * F(mu; a_l(E')) >_kernel                    (product)

and NOT

    < sigma(E') > * F(mu; < a_l(E') >)                                    (factors)

The two agree only if sigma and a_l are uncorrelated across the window. In the
resolved resonance region they are not — both are driven by the same resonances.
Probed on the JEFF-4.0 host over 0.9-3.8 MeV with the run-82 kernel (5 ns FWHM,
27.037 m), the factor form differs from the product form by:

    0.90 MeV  5.3% median, 14.4% max        2.38 MeV  2.1% median,  4.6% max
    1.00 MeV  3.8% median, 18.4% max        2.85 MeV  0.01% median, 0.04% max
    1.20 MeV  3.3% median,  6.1% max        3.20 MeV  0.02% median, 0.04% max
    1.40 MeV  2.6% median, 11.0% max        3.80 MeV  0.02% median, 0.14% max

i.e. negligible above ~2.8 MeV where sigma(E) is smooth, and up to 18% below
2.5 MeV, which is where most of the database lives. For scale, folding at all
(either form) moves the prediction ~28% median against the unfolded nominal, so
the fold is first-order and the product-vs-factor choice is a real second-order
correction, not a rounding detail.

Window width
------------
`N_SIGMA = 3` and nothing else. The window is a *truncated* Gaussian renormalized
to unit weight, so a narrow window is not a conservative choice — it is a
different, wrong kernel. Measured convergence of the product fold against the
3-sigma answer:

    +-1 sigma   0.5-12.5% median, up to 21% max   -> NOT converged
    +-4 sigma   <=0.14% median, <=0.51% max       -> converged

So +-1 sigma is not a sensitivity knob worth reporting, and +-4 buys nothing.

Per datapoint, for each library:

    y_eval(mu):
        sum_i w_i * [ sigma_i/(4 pi) * (1 + sum_l (2l+1) a_l(E_i) P_l(mu)) ]
        over N_WINDOW_SAMPLES nodes E_i spanning +-N_SIGMA*sigma_E, with
        Gaussian weights w_i renormalized to 1. sigma_E comes from the
        experiment's own flight path / timing (same TOF metadata and FWHM
        convention as the sampling pipeline).

    c_0:
        < sigma_MF3 >_fold / (4 pi). Reported per row and used as the
        linearization scale of the MF34 sandwich. Note this is the *factor*
        average — it is the scale the covariance is linearized about, not the
        central value, which is the product fold above.

    Sigma_eval:
        MF34 sandwich on a_l propagated to dsigma/dOmega(mu), plus the MF33
        magnitude block, summed (no MF33<->MF34 cross block: Phase 2 sets it to
        zero, a documented factorization). Built by
        scripts.eval_covariance.build_eval_cov_for_groups, which picks the MF33
        fields up automatically when the library dict carries them.

    Sandwich linearization:
        the MF34 shape term uses the *nominal* a_l(E) rather than the folded
        one — a second-order effect on a variance, documented and accepted in
        the roadmap.

Output: one parquet row per (library, experiment, datapoint) plus a dense
per-(library, experiment) Sigma_eval sidecar, consumed by
scripts/chi2_analysis_cluster.py under the `predictive` methodology key.
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legval

# Make BLAS single-threaded BEFORE numpy/scipy load.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_kika_root = Path(__file__).resolve().parent.parent
if str(_kika_root) not in sys.path:
    sys.path.insert(0, str(_kika_root))

from scripts.precompute_chi2_exfor_c0 import load_exfor, build_experiment_dataframe
from scripts.precompute_chi2_library_c0 import (
    interp_a_l_to_energy,
    load_library_lib_c0,
)
from scripts.eval_covariance import build_eval_cov_for_groups, save_eval_cov
from scripts.tof_parameters import (
    load_tof_parameters_file,
    get_tof_parameters,
    compute_sigma_E,
    CORPUS_MEDIAN_REL_SIGMA_E,
)


# ── Configuration ─────────────────────────────────────────────────────────────

MT_NUMBER = 2  # elastic scattering

# ── Resolution window ──
# Single value on purpose; see the module docstring for the convergence numbers
# that retire the 1-sigma/3-sigma sweep. N_WINDOW_SAMPLES sets the sampling
# density across the window — enough to resolve MF3 resonance structure inside it.
N_SIGMA           = 3.0
N_WINDOW_SAMPLES  = 65

# ── Library ENDF files ──
# This_work uses its own MF3, MF4, MF34 and MF33 from the pipeline product.
# Defaults to the MT1-repaired run-82 directory: its MF34 is identical to the
# unrepaired one, and its MF33 is the ungrouped (1738-group) matrix rather than
# the 190-group collapse, so the magnitude term is not pre-averaged.
#
# Overridable so the same script can score a different pipeline run (run 83, the
# EN-RSL re-evaluation) without editing the file. Unset ⇒ byte-identical
# behaviour to run 82, which must stay reproducible.
THIS_WORK_DIR  = os.environ.get(
    "KIKA_THIS_WORK_DIR",
    "/share_snc/snc/JuanMonleon/ENDF_samples/new_test_82_mt1fix",
).rstrip("/")

# Which of the two shipped products to score. The pipeline writes both:
#   26-Fe-56g_nominal_mg.endf   ~205 MB, ~660 multigroups   <- the default
#   26-Fe-56g_nominal.endf      ~843 MB, 1738 fine bins
#
# Every run from 82 to 85 scored the multigroup file, and that is the default so
# they stay reproducible. KIKA_THIS_WORK_ENDF overrides it.
#
# ⚠ WHY THIS EXISTS. The two products are NOT interchangeable for Phase 3. The
# mixture covariance reaches the fine MF34 and not the multigroup one (roadmap
# §8.4 — dead parameters at l=6: 92.3% -> 1.0% fine, 81.6% -> 81.7% multigroup),
# so scoring the multigroup file measures the mixture's ABSENCE. Pointing this at
# run 85's fine ENDF answers the Phase-3 question with no re-evaluation.
#
# Memory: the fine MF34 is ~734 MB on disk and ~0.5 GB parsed. Ask for 300G, the
# same as the representation sweep — the 100G that suffices for _mg.endf will not
# do here.
THIS_WORK_ENDF = os.environ.get("KIKA_THIS_WORK_ENDF", "26-Fe-56g_nominal_mg.endf")
THIS_WORK_FILE = f"{THIS_WORK_DIR}/{THIS_WORK_ENDF}"

JEFF_FILE  = "/share_snc/snc/JuanMonleon/jeff40_with_MF4_from_jeff33/26-Fe-56g.txt"
JENDL_FILE = "/share_snc/snc/JuanMonleon/JENDL-5/260560.jendl5"

# ── EXFOR sources ──
EXFOR_DB_PATH = "/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db"
TARGET_ZAIDS  = [26056, 26000]
SUPPLEMENTARY_JSON_FILES = [
    "/share_snc/snc/JuanMonleon/EXFOR/data_v1/27673002.json",
]
EXCLUDE_EXPERIMENTS = ["32246002", "400750022"]  # Tostkii 1957; Morozov 1972 pointer 2

# ── TOF energy resolution (same source/defaults as the sampling pipeline) ──
TOF_PARAMETERS_FILE        = "/share_snc/snc/JuanMonleon/EXFOR/exfor_tof_parameters.json"
DEFAULT_FLIGHT_PATH_M      = 27.037
DEFAULT_TIME_RESOLUTION_NS = 5.0
# Must match DELTA_T_IS_FWHM in exfor_to_endf_sampling_v2.py, or this chi2 folds
# over a different kernel than the fit used. From run 82 delta_t is a FWHM.
DELTA_T_IS_FWHM            = True
# 2026-08-26: must match DEFAULT_REL_SIGMA_E in exfor_to_endf_research.py for
# the same reason. sigma_E for subentries EXFOR documents no width for, as a
# fraction of E, in place of the ORELA-like (L, delta_t) default.
DEFAULT_REL_SIGMA_E        = CORPUS_MEDIAN_REL_SIGMA_E
QUARANTINED_AS_BOX         = True   # read a review_required EN-RSL* width as a box

E_MIN_MEV = 0.85
E_MAX_MEV = 4.0

# Legendre truncation. In [0.85, 4] MeV the highest order any of the three
# libraries reports is 6, so all evaluations are compared at L=6.
L_MAX = 6

# ── Fold mode ──
# Which part of the forward model gets resolution-averaged. Set from the
# environment so one sbatch array can sweep every mode without editing the file:
#
#   product  <sigma . F(a_l)>            the physical average -- what a TOF
#                                        detector integrating over its own
#                                        resolution actually records. DEFAULT.
#   factors  <sigma> . F(<a_l>)          average both factors separately, then
#                                        rebuild. Drops Cov(sigma, a_l) across
#                                        the window.
#   sigma    <sigma> . F(a_l(E0))        average the magnitude only.
#   al       sigma(E0) . F(<a_l>)        average the Legendre shape only.
#   none     sigma(E0) . F(a_l(E0))      no resolution model at all.
#
# The notebook (kika_dev/exfor/uncertainty/energy_resolution_impact.ipynb) shows
# `product` and `factors` differ by up to ~18% below 2.5 MeV at a handful of
# hand-picked energies. This switch is what turns that into a statement over
# every (experiment, energy) bin in the database: run all five, compare V2
# (central values, no evaluated covariance) across modes.
FOLD_MODE = os.environ.get("FOLD_MODE", "product").strip().lower()

_FOLD_SUFFIX = {
    "product": "",            # keep the original name so run 82 is not invalidated
    "factors": "_factors",
    "sigma":   "_foldsigma",
    "al":      "_foldal",
    "none":    "_nofold",
}
if FOLD_MODE not in _FOLD_SUFFIX:
    raise SystemExit(
        f"FOLD_MODE={FOLD_MODE!r} is not one of {sorted(_FOLD_SUFFIX)}"
    )

# Which pipeline run this scoring belongs to. Tags the parquet so run 82 and
# run 83 products never collide. Default "82" keeps every existing filename.
#
# NOTE the two are independent on purpose: KIKA_THIS_WORK_DIR chooses the
# evaluation being scored, KIKA_RUN_TAG names the output. Scoring run 82 under a
# changed σ_E (tag "82_enrsl") is a real use case — it isolates the effect of the
# resolution fix on the *scoring* from its effect on the *evaluation*.
RUN_TAG = os.environ.get("KIKA_RUN_TAG", "82").strip()

# ── MF33 ↔ MF34 cross block (roadmap §9.4.3 / run 87) ────────────────────────
# Σ_eval = Σ^MF34 + Σ^MF33 + Σ^cross. The third term has always been fed None,
# i.e. exactly zero, because the evaluation splits one Monte Carlo into two
# one-at-a-time channels (magnitude at frozen shape, shape at frozen c₀) and
# that partition discards Cov(σ, a_ℓ) by construction.
#
# It is NOT zero. Run 86 measured it from the paired Pass-2 replicas
# (`compute_mf33_mf34_cross`, diagnostic-only, no MC cost): median |ρ| = 0.594 at
# a₁, 81 % of bins above |ρ| = 0.3 — but the sign alternates with energy, so the
# median ρ is only −0.173 and the mean −0.009. Large per bin, cancelling in the
# aggregate. Feeding it here is what turns that parameter-space measurement into
# a per-datapoint effect, because Σ_cross carries P_ℓ(μ), which alternates in
# ANGLE on top of ρ's alternation in ENERGY.
#
# Point this at the evaluation directory holding the sidecars:
#     mf33_mf34_cross_covariance.npy   (n_bins, L)  absolute Cov(c₀, a_ℓ)
#     mf33_energy_grid_ev.npy          (n_bins+1,)  bin edges, eV
#     mf33_c0_host.npy                 (n_bins,)    host magnitude per bin
# Unset ⇒ zero block ⇒ runs 82–86 stay byte-reproducible.
MF33_MF34_CROSS_DIR = os.environ.get("KIKA_MF33_MF34_CROSS_DIR", "").strip()

# Multiplies the whole cross block. 1.0 = as measured; 0.0 = off (equivalent to
# not setting the directory, but keeps the provenance line in the log). Exists so
# a PSD failure can be bracketed by damping rather than by rebuilding sidecars,
# and so the ±1 Cauchy–Schwarz ends can be probed without a new evaluation.
MF33_MF34_CROSS_SCALE = float(os.environ.get("KIKA_MF33_MF34_CROSS_SCALE", "1.0"))

# ⚑ THE CROSS TERM FROM THE FILE ITSELF, which is the whole of roadmap §10.7-4
# step 5. Set KIKA_MF33_MF34_CROSS_FROM_FILE=1 and the term is read out of
# This_work's MF34 (L=0, L1) a₀ blocks instead of a sidecar.
#
# Why that is not a cosmetic change of source. A sidecar is a second object,
# and the cross term is Cauchy-Schwarz-compatible only with the marginals it
# was built from (§10.1) — run 89 shipped one next to marginals nothing had
# diagnosed and measured λ_min/scale = -0.447. The sidecar also declares
# `is_relative=False` against a relative MF34 family, which the §L13 guard
# refuses outright, so it has never been scorable at all. Read from the file,
# both legs come from one ENDF in one convention and the units question cannot
# arise.
#
# ⚑ MUTUALLY EXCLUSIVE with KIKA_MF33_MF34_CROSS_DIR, and this is §L8's
# no-double-counting check INVERTED. It used to read "eval_covariance skips
# l_r < 1, so the sidecar is the only source"; it now reads "the a₀ blocks are
# the source, so the sidecar must be ABSENT". Both set is the double count, and
# nothing else would warn.
MF33_MF34_CROSS_FROM_FILE = os.environ.get(
    "KIKA_MF33_MF34_CROSS_FROM_FILE", "").strip() not in ("", "0", "false", "False")

# ── Removing MF34 parameters the evaluation never determined (roadmap §10.6-1) ─
# Path to the npz from `build_group_cross.py --write-null-mask`. Unset → nothing
# changes, which is what every run up to 90 did.
#
# WHY. `row_aggregator` returns an all-zero row for any (shape group, order) with
# no valid fine bin, so the MC constrains nothing at 1542 of 4218 parameters --
# and the shipped file still declares relative variance up to 7.6 (sigma_rel
# ~ 276 %) at ~95 % of them. §10.1.8-L11 established the chi2 folds them at FULL
# weight, because `build_mf34_block` scales the relative block by the file's own
# MF4 interpolated onto each EXFOR energy, which is NOT zero there. Measured on
# run 86 by `null_slot_exposure.py`: 2.14 % of the Sigma_eval diagonal, 82.9 % of
# points touched, two small datasets above 44 %, Cierjacks at 3.0 %.
#
# ⚠ THIS_WORK ONLY, and that asymmetry is the point rather than a bug: JEFF and
# JENDL ship no such term, so the inflation is one-sided in OUR favour. Applying
# a mask derived from our MC to their files would be meaningless.
MF34_NULL_MASK = os.environ.get("KIKA_MF34_NULL_MASK", "").strip()

OUTPUT_PARQUET = (
    "/share_snc/snc/JuanMonleon/chi2/"
    f"chi2_data_predictive_{RUN_TAG}{_FOLD_SUFFIX[FOLD_MODE]}.parquet"
)


# ── Resolution window ─────────────────────────────────────────────────────────

def _resolution_window(e_mev: float, sigma_E_mev: float) -> Tuple[np.ndarray, np.ndarray]:
    """Truncated-Gaussian resolution kernel nodes (eV) and unit-sum weights.

    With sigma_E <= 0 the kernel collapses to a single node at e (delta),
    matching fold_xs_over_resolution.
    """
    if sigma_E_mev <= 0.0 or N_WINDOW_SAMPLES < 2:
        return np.array([e_mev * 1e6]), np.array([1.0])
    half = N_SIGMA * sigma_E_mev
    e_grid_mev = np.linspace(e_mev - half, e_mev + half, N_WINDOW_SAMPLES)
    w = np.exp(-0.5 * ((e_grid_mev - e_mev) / sigma_E_mev) ** 2)
    w /= w.sum()
    return e_grid_mev * 1e6, w


def fold_dcs(
    lib: Dict, mu: np.ndarray, sample_e_ev: np.ndarray, weights: np.ndarray,
    mode: str = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Evaluate dsigma/dOmega under one of the five FOLD_MODE conventions.

    Returns ``(y_eval, sigma_used_b, a_l_used)`` where ``sigma_used_b`` and
    ``a_l_used`` are the magnitude and Legendre vector this mode actually
    predicted with — they set ``c0`` and therefore the linearization scale of
    Sigma_eval, so each mode's covariance is built about its own central value.

    Only ``product`` keeps the nodes separate; every other mode collapses to a
    single effective (sigma, a_l) pair and is evaluated once. That makes the
    modes differ in exactly one place — what is averaged — with the window,
    the weights and everything downstream held fixed.
    """
    mode = FOLD_MODE if mode is None else mode
    sigma_samples = np.interp(sample_e_ev, lib["e_mf3_ev"], lib["xs_mf3"])
    n_nodes = sample_e_ev.size

    a_samples = np.empty((n_nodes, L_MAX), dtype=float)
    for i, e_ev in enumerate(sample_e_ev):
        a_samples[i, :] = interp_a_l_to_energy(lib, e_ev / 1e6, L_MAX)

    sigma_avg = float(weights @ sigma_samples)
    a_avg = weights @ a_samples
    # The window is symmetric about the nominal energy, so the centre node is
    # the unfolded evaluation point (and the only node when sigma_E <= 0).
    i0 = n_nodes // 2
    sigma_0 = float(sigma_samples[i0])
    a_0 = a_samples[i0]

    if mode == "product":
        s_nodes, a_nodes, w_nodes = sigma_samples, a_samples, weights
        sigma_used, a_used = sigma_avg, a_avg
    else:
        if mode == "factors":
            sigma_used, a_used = sigma_avg, a_avg
        elif mode == "sigma":
            sigma_used, a_used = sigma_avg, a_0
        elif mode == "al":
            sigma_used, a_used = sigma_0, a_avg
        elif mode == "none":
            sigma_used, a_used = sigma_0, a_0
        else:  # pragma: no cover -- guarded at import
            raise ValueError(f"unknown fold mode {mode!r}")
        s_nodes = np.array([sigma_used], dtype=float)
        a_nodes = a_used[None, :]
        w_nodes = np.array([1.0])

    # Legendre coefficient matrix, one column per node, so legval evaluates
    # every node's distribution in one call.
    scale = s_nodes / (4.0 * np.pi)
    coeffs = np.empty((L_MAX + 1, s_nodes.size), dtype=float)
    coeffs[0, :] = scale
    for l in range(1, L_MAX + 1):
        coeffs[l, :] = scale * (2 * l + 1) * a_nodes[:, l - 1]

    # legval(mu, coeffs) -> shape (n_nodes, mu.size)
    per_node = np.atleast_2d(legval(mu, coeffs))
    y_eval = w_nodes @ per_node

    return y_eval, sigma_used, a_used


# ── Per-energy row builder ────────────────────────────────────────────────────

def build_rows_at_energy(
    e_mev: float,
    experiments_at_energy: List[Tuple[pd.DataFrame, Dict]],
    libraries: Dict[str, Dict],
    tof_cache: Dict,
) -> List[dict]:
    """One row per (library, experiment, datapoint) at this incident energy.

    sigma_E depends on the subentry's TOF parameters, so the fold is computed
    inside the experiment loop rather than cached once per energy.
    """
    rows: List[dict] = []
    for cache_df, meta in experiments_at_energy:
        exp_df = build_experiment_dataframe(cache_df, meta)
        if len(exp_df) < 1:
            continue

        subentry_id = f"{meta['entry']}{meta['subentry']}"
        tof = get_tof_parameters(
            subentry_id, tof_cache,
            default_flight_path_m=DEFAULT_FLIGHT_PATH_M,
            default_time_resolution_ns=DEFAULT_TIME_RESOLUTION_NS,
            default_delta_t_is_fwhm=DELTA_T_IS_FWHM,
            default_rel_sigma_E=DEFAULT_REL_SIGMA_E,
            quarantined_as_box=QUARANTINED_AS_BOX,
        )
        sigma_E_mev = compute_sigma_E(e_mev, tof)
        sample_e_ev, weights = _resolution_window(e_mev, sigma_E_mev)

        mu = exp_df["mu"].to_numpy(dtype=float)
        y_exp = exp_df["value"].to_numpy(dtype=float)
        sigma_stat = exp_df["sigma_stat"].to_numpy(dtype=float)
        sigma_sys = exp_df["sigma_sys"].to_numpy(dtype=float)
        sigma_exp = np.sqrt(sigma_stat ** 2 + sigma_sys ** 2)

        for lib_key, lib in libraries.items():
            y_eval, sigma_avg_b, a_l_folded = fold_dcs(
                lib, mu, sample_e_ev, weights,
            )
            if not np.isfinite(sigma_avg_b) or sigma_avg_b <= 0:
                continue
            c0 = sigma_avg_b / (4.0 * np.pi)

            for j in range(len(exp_df)):
                rows.append({
                    "energy_mev":      float(e_mev),
                    "mu":              float(mu[j]),
                    "angle_deg":       float(exp_df["angle_deg"].iloc[j]),
                    "y_exp":           float(y_exp[j]),
                    "sigma_exp_stat":  float(sigma_stat[j]),
                    "sigma_exp_sys":   float(sigma_sys[j]),
                    "sigma_exp":       float(sigma_exp[j]),
                    "sigma_sys_rel":       float(exp_df["sigma_sys_rel"].iloc[j]),
                    "sigma_sys_indep_rel": float(exp_df["sigma_sys_indep_rel"].iloc[j]),
                    "sigma_sys_dep_rel":   float(exp_df["sigma_sys_dep_rel"].iloc[j]),
                    "experiment_id":   str(exp_df["experiment_id"].iloc[j]),
                    "author":          str(exp_df["author"].iloc[j]),
                    "year":            int(exp_df["year"].iloc[j]),
                    "is_natural":      bool(exp_df["is_natural"].iloc[j]),
                    "ks_subentry":     str(exp_df["ks_subentry"].iloc[j]),
                    "library":         lib_key,
                    "y_eval":          float(y_eval[j]),
                    "c0":              float(c0),
                    "L_max":           int(L_MAX),
                    "sigma_avg_b":     float(sigma_avg_b),
                    "a1_folded":       float(a_l_folded[0]),
                    "sigma_E_mev":     float(sigma_E_mev),
                    "n_sigma":         float(N_SIGMA),
                    "tof_source":      str(tof.source),
                    "fold_mode":       FOLD_MODE,
                })
    return rows


# ── MF33 ↔ MF34 cross block ───────────────────────────────────────────────────

def refuse_double_cross_source(from_file: bool, cross_dir: str) -> None:
    """⚑ §L8's no-double-counting check, INVERTED for item 5.

    It used to be structural and needed no code: `build_mf34_block` skips
    ``l_r < 1``, so MF34's a₀ blocks were invisible to the fold and the sidecar
    was necessarily the only source of the cross term.

    That skip is still there — deliberately — but the a₀ blocks are now read by
    `mf34_cross_reader` and written to the SAME ``mf33_mf34_cross`` key the
    sidecar loader uses. So with both enabled the term is folded twice, or one
    silently overwrites the other depending on assignment order, and nothing
    downstream can tell: Σ_eval simply comes out with the magnitude↔shape
    correlation counted at 2×. The assertion therefore flips from "the sidecar
    is the only source" to "the sidecar is ABSENT", and it has to be checked.
    """
    if from_file and cross_dir:
        raise SystemExit(
            "KIKA_MF33_MF34_CROSS_FROM_FILE and KIKA_MF33_MF34_CROSS_DIR are "
            "both set. That folds the cross term TWICE -- once from "
            "This_work's MF34 a_0 blocks and once from the sidecar -- and "
            "nothing else would warn, because both write the same key. This is "
            "§L8's no-double-counting check inverted: with the term in the "
            "file, the assertion is that the sidecar is absent. Unset one."
        )


def load_mf33_mf34_cross(
    run_dir: str, l_max: int = L_MAX, scale: float = 1.0,
) -> Tuple[List[dict], dict]:
    """Build the `mf33_mf34_cross` blocks from a pipeline run's sidecars.

    Returns ``(blocks, info)`` in exactly the layout
    :func:`scripts.eval_covariance.build_mf33_mf34_cross_block` expects.

    THE UNIT CONVENTION, WHICH IS THE ONLY THING THAT CAN SILENTLY GO WRONG.
    The builder's sensitivities are dy/dσ = ``y_eval`` on the magnitude side and
    dy/da_ℓ = ``c₀(2ℓ+1)P_ℓ`` on the shape side (times a_ℓ when the block
    declares itself relative). Using ``y_eval`` as the magnitude sensitivity
    means the magnitude axis is **relative**, δσ/σ — the same currency
    ``build_mf33_block`` consumes.

    So the block must be Cov(δσ/σ, δa_ℓ), and the denominator has to be the one
    the shipped MF33 already uses. Measured on run 86's sidecars, the shipped
    relative MF33 is ``absolute / (c0_host ⊗ c0_host)`` **exactly** (max
    difference 0.0), not ``c0_nominal`` — our own c₀ sits ~9.5 % above the
    host's and the pipeline recentres onto the host. Dividing by the wrong one
    would rescale the cross term by 1.095 against its own diagonals and break
    Cauchy–Schwarz consistency with them. Hence ``c0_host``.

    The shape axis is left ABSOLUTE (``is_relative=False``). The relative form
    would divide by a_ℓ_nom, which passes through zero — the near-zero blow-up
    ``REGULARIZE_NEAR_ZERO_REL_UNC`` exists to contain — and the builder applies
    the correct absolute sensitivity itself.

    WHAT IS AND IS NOT MEASURED — TWO CASES, AND THE DIFFERENCE IS THE WHOLE
    POINT (roadmap §10.1.5/§10.1.6).

    * ``mf33_mf34_cross_covariance_full.npy`` present (evaluations from the
      2026-08-04 change onwards): the complete block Cov(c₀(E_i), a_ℓ(E_j)) over
      one common replica set. **This is the only form that is a valid
      covariance**, and it is preferred whenever it exists.
    * only ``mf33_mf34_cross_covariance.npy`` (runs 86 and 87): within-bin
      Cov(c₀(E_i), a_ℓ(E_i)) only, so the per-order matrix is diagonal. **This
      form is known to be non-PSD against complete MF33/MF34 diagonals** — it is
      what killed run 87 — and is kept ONLY so those runs stay reproducible.
      Loading it emits a warning; do not ship a result built on it.

    The cross-energy zero in the diagonal form is *absence of measurement, not
    physics*: measured on run 81's paired replicas, the cross-energy entries are
    the same size as the within-bin ones and ~159× more numerous.

    Slots the MC could not define (orders restored from nominal) are NaN in the
    sidecar and are set to zero here, counted and reported rather than filled.
    """
    d = Path(run_dir)
    cov_path = d / "mf33_mf34_cross_covariance.npy"
    grid_path = d / "mf33_energy_grid_ev.npy"
    c0h_path = d / "mf33_c0_host.npy"
    for p in (cov_path, grid_path, c0h_path):
        if not p.exists():
            raise SystemExit(
                f"KIKA_MF33_MF34_CROSS_DIR={run_dir!r} is missing {p.name}. "
                f"The cross sidecars are written only when the evaluation ran "
                f"with COMPUTE_MF33_MF34_CROSS=True (run 86 onwards)."
            )

    cov = np.load(cov_path)                      # (n_bins, L) absolute Cov(c0, a_l)
    grid_ev = np.load(grid_path).astype(float)   # (n_bins + 1,) edges
    c0_host = np.load(c0h_path).astype(float)    # (n_bins,)

    n_bins = cov.shape[0]
    if grid_ev.size != n_bins + 1:
        raise SystemExit(
            f"Cross grid mismatch: {n_bins} bins but {grid_ev.size} edges in "
            f"{grid_path.name}. These sidecars must come from the SAME run."
        )
    if c0_host.size != n_bins:
        raise SystemExit(
            f"Cross c0_host mismatch: {n_bins} bins but {c0_host.size} values."
        )

    # PREFERRED FORM (roadmap §10.1.8-J): the block built in the pipeline's own
    # adaptive group space from PASS-1 replicas, by `scripts/build_group_cross.py`.
    #
    # Why it supersedes the fine-grid form below rather than sitting beside it.
    # The fine block was estimated by pairing replica s across energies in PASS 2,
    # whose RNG is seeded per BIN (`exfor_to_endf_research.py:1535`), while every
    # correlation in the shipped MF33/MF34 comes from PASS 1, seeded per REPLICA
    # (`exfor_utils.py:2002`). Measured: the Pass-2 cross-energy correlation
    # equals its own permutation null exactly, so that block carried no
    # information -- and gluing it to Pass-1 marginals is what made runs 87 and 88
    # non-PSD. The group-space block is built from one common Pass-1 replica set
    # with the two-pass sigma rescaling applied AFTER the collapse, which makes it
    # a single positive diagonal congruence on the whole joint: lam_min/scale =
    # -2.8e-17 against -0.26 for run 88's, with the published sigmas reproduced to
    # 3e-17.
    #
    # UNITS. Already relative on the magnitude axis (it is built against the
    # shipped relative MF33), so unlike the fine sidecars it must NOT be divided
    # by c0_host again. The shape axis is absolute, as before.
    #
    # ⚠⚠ GRIDS — AND THIS ARTEFACT NO LONGER SATISFIES THE CONTRACT.
    # Its two axes are the run's OWN adaptive grids (MF33 magnitude ~188 groups,
    # MF34 shape ~703), and the magnitude one is NOT the grid the shipped MF33
    # is written on (~2317 fine bins). Since §10.7-7 the fold requires them to
    # be the same, because `Sigma_eval = M J M^T` is a congruence only if the
    # magnitude parameter reaches the points through ONE map — and a cross term
    # is Cauchy-Schwarz-compatible only with the marginals it was built from
    # (§10.1). Folding this against the fine MF33 is what produced runs 89 and
    # 90 and no chi2.
    #
    # `build_mf33_mf34_cross_block` now rejects it by row count rather than
    # silently regridding, so the block dict no longer carries a magnitude grid
    # of its own. To use a group cross term, rebuild BOTH the marginal and the
    # cross on one magnitude grid — which, per §10.7-9, means the fine one,
    # since job B disqualified grouping MF33.
    grp_path = d / "mf33_mf34_cross_group_covariance.npy"
    grp_mag_path = d / "mf33_mf34_cross_group_mag_grid_ev.npy"
    grp_shape_path = d / "mf33_mf34_cross_group_shape_grid_ev.npy"
    if grp_path.exists():
        missing = [p.name for p in (grp_mag_path, grp_shape_path) if not p.exists()]
        if missing:
            raise SystemExit(
                f"{run_dir}: {grp_path.name} is present but {missing} are not. "
                f"The group block is meaningless without its own grids — rerun "
                f"scripts/build_group_cross.py --write on this run."
            )
        grp = np.load(grp_path)                       # (n_mag, n_shape, L)
        mag_ev = np.load(grp_mag_path).astype(float)
        shape_ev = np.load(grp_shape_path).astype(float)
        if grp.ndim != 3:
            raise SystemExit(f"{grp_path.name} has shape {grp.shape}, expected 3-D")
        n_mag, n_shape, n_l = grp.shape
        if mag_ev.size != n_mag + 1 or shape_ev.size != n_shape + 1:
            raise SystemExit(
                f"Group cross grid mismatch: block is {n_mag}x{n_shape} but the "
                f"grids have {mag_ev.size} and {shape_ev.size} edges."
            )
        n_undef = int(np.count_nonzero(~np.isfinite(grp[:, :, :l_max])))
        grp = np.nan_to_num(grp, nan=0.0, posinf=0.0, neginf=0.0) * float(scale)

        blocks = [{
            "l": L,
            "shape_grid_ev": shape_ev,
            "matrix": grp[:, :, L - 1],
            "is_relative": False,
        } for L in range(1, min(l_max, n_l) + 1)]
        info = {
            "n_bins": n_mag,
            "n_orders": len(blocks),
            "n_undefined_slots": n_undef,
            "scale": float(scale),
            "form": "group",
            "median_abs_rel_cov": {
                L: float(np.median(np.abs(grp[:, :, L - 1])))
                for L in range(1, len(blocks) + 1)
            },
        }
        return blocks, info

    # "Undefined" lives in the CORRELATION sidecar, not the covariance one.
    # `compute_mf33_mf34_cross` leaves rho as NaN for orders restored from
    # nominal (zero spread ⇒ rho genuinely undefined), while cov for those slots
    # comes out numerically zero (~1e-31 on run 86) because a constant column has
    # no covariance with anything. So zeroing them is exact rather than a choice
    # — but the count has to be reported from rho, or the log claims the MC
    # measured 10428 slots when it measured 10392.
    rho_path = d / "mf33_mf34_cross_correlation.npy"
    if rho_path.exists():
        rho = np.load(rho_path)
        n_undef = int(np.count_nonzero(~np.isfinite(rho[:, :l_max])))
    else:
        n_undef = int(np.count_nonzero(~np.isfinite(cov[:, :l_max])))
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        cov_rel_mag = np.where(c0_host[:, None] > 0.0, cov / c0_host[:, None], 0.0)
    cov_rel_mag = np.nan_to_num(cov_rel_mag, nan=0.0, posinf=0.0, neginf=0.0)
    cov_rel_mag *= float(scale)

    # The COMPLETE block when the run produced one, otherwise the legacy
    # within-bin-only diagonal. `full[i, j, L-1]` = Cov(δσ/σ at E_i, a_L at E_j)
    # and is deliberately NOT symmetric — row is magnitude, column is shape.
    full_path = d / "mf33_mf34_cross_covariance_full.npy"
    full = None
    if full_path.exists():
        full = np.load(full_path)
        if full.shape[:2] != (n_bins, n_bins) or full.shape[2] < min(l_max, cov.shape[1]):
            raise SystemExit(
                f"Full cross block has shape {full.shape}, expected "
                f"({n_bins}, {n_bins}, >={min(l_max, cov.shape[1])}). "
                f"These sidecars must come from the SAME run."
            )
        full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            full = np.where(c0_host[:, None, None] > 0.0, full / c0_host[:, None, None], 0.0)
        full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0) * float(scale)
    else:
        warnings.warn(
            f"{run_dir}: no mf33_mf34_cross_covariance_full.npy — falling back to "
            "the WITHIN-BIN-ONLY cross block. That form is not a valid covariance "
            "against complete MF33/MF34 diagonals (roadmap §10.1.5: it is what made "
            "run 87 non-PSD). Reproducibility only; do not ship a result on it.",
            RuntimeWarning,
            stacklevel=2,
        )

    blocks: List[dict] = []
    for L in range(1, min(l_max, cov.shape[1]) + 1):
        blocks.append({
            "l": L,
            "shape_grid_ev": grid_ev,
            "matrix": full[:, :, L - 1] if full is not None
                      else np.diag(cov_rel_mag[:, L - 1]),
            "is_relative": False,
        })

    info = {
        "n_bins": n_bins,
        "n_orders": len(blocks),
        "n_undefined_slots": n_undef,
        "scale": float(scale),
        # "full" is the only valid form; "within_bin_only" is runs 86/87 and is
        # known non-PSD. The [XCROSS] banner prints this, so a run's own log says
        # which one it used rather than leaving it to be inferred.
        "form": "full" if full is not None else "within_bin_only",
        "median_abs_rel_cov": {
            L: float(np.median(np.abs(cov_rel_mag[:, L - 1]))) for L in
            range(1, len(blocks) + 1)
        },
    }
    return blocks, info


def _min_eig_lanczos(block: np.ndarray, tol: float = 1e-6) -> Optional[float]:
    """Smallest eigenvalue of a large dense symmetric block, or None if it fails.

    WHY NOT `eigvalsh`. Cierjacks is 28631 x 28631; an exact decomposition is
    O(N^3) ~ 2e13 flops and would dominate the whole job. But we do not need the
    spectrum, only its bottom, and Lanczos reaches that in a few dozen matrix
    -vector products (~0.8 GFLOP each here).

    WHY THE SHIFT. ARPACK converges quickly for LARGEST-magnitude eigenvalues and
    slowly for `which='SA'`. So take lam_max first, then the largest eigenvalue of
    (lam_max*I - A), which is lam_max - lam_min. Two fast problems instead of one
    slow one.

    Kept in float32 deliberately: the block is stored float32 and upcasting
    Cierjacks would allocate 6.5 GB for no accuracy that matters at this
    tolerance. Returns None rather than raising — this is a diagnostic, and a
    non-converged estimate must never be the reason a 12-hour job dies.
    """
    try:
        from scipy.sparse.linalg import LinearOperator, eigsh
    except Exception:
        return None

    n = block.shape[0]
    A = np.asarray(block)

    def _mv(x):
        return (A @ np.asarray(x, dtype=A.dtype).ravel()).astype(np.float64)

    op = LinearOperator((n, n), matvec=_mv, dtype=np.float64)
    try:
        lam_max = float(eigsh(op, k=1, which="LA", tol=tol,
                              return_eigenvectors=False)[0])
        shifted = LinearOperator(
            (n, n), matvec=lambda x: lam_max * np.asarray(x, float).ravel() - _mv(x),
            dtype=np.float64,
        )
        gap = float(eigsh(shifted, k=1, which="LA", tol=tol,
                          return_eigenvectors=False)[0])
        return lam_max - gap
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # load_library_lib_c0 reads MF3 + MF4 + MF34 + MF33 and returns exactly the
    # dict build_eval_cov_for_groups expects, so the MF33/MF34 handling here is
    # identical to the library_c0 scenario by construction.
    libraries = {
        "JEFF":      load_library_lib_c0(JEFF_FILE,       "JEFF-4.0"),
        "JENDL":     load_library_lib_c0(JENDL_FILE,      "JENDL-5"),
        "This_work": load_library_lib_c0(THIS_WORK_FILE,  "This work"),
    }

    # Opt-in MF33↔MF34 cross block, This_work only. JEFF and JENDL ship no such
    # measurement, and inventing one for them would make the comparison unfair in
    # the opposite direction to the missing-MF33 warning below. Say so out loud.
    refuse_double_cross_source(MF33_MF34_CROSS_FROM_FILE, MF33_MF34_CROSS_DIR)

    if MF33_MF34_CROSS_FROM_FILE and MF33_MF34_CROSS_SCALE != 0.0:
        cross_blocks = libraries["This_work"].get("mf34_a0_cross") or []
        cross_info = dict(libraries["This_work"].get("mf34_a0_info") or {})
        if not cross_blocks:
            raise SystemExit(
                f"KIKA_MF33_MF34_CROSS_FROM_FILE is set but {THIS_WORK_FILE} "
                f"ships no MF34 a_0 blocks, so the cross term would be a silent "
                f"zero and the run would be misattributed. Write the file with "
                f"`build_group_cross.py --mag-grid fine --write-endf`, or unset "
                f"the variable."
            )
        # ⚑ s in [0, 1] is PSD-safe and s > 1 is not:
        #   [[A, sB], [sB.T, C]] = s*J + (1-s)*diag(A, C)
        # is a convex combination of two PSD matrices for s in [0, 1], and
        # nothing bounds it beyond. The knob exists to bracket a PSD failure by
        # damping rather than by rebuilding the file; it is not a free dial.
        if MF33_MF34_CROSS_SCALE != 1.0:
            if not 0.0 <= MF33_MF34_CROSS_SCALE <= 1.0:
                raise SystemExit(
                    f"KIKA_MF33_MF34_CROSS_SCALE={MF33_MF34_CROSS_SCALE} is "
                    f"outside [0, 1]. Only that range is a convex combination "
                    f"of the joint and its block diagonal, i.e. only there is "
                    f"the damped matrix guaranteed PSD when the joint is."
                )
            cross_blocks = [dict(b, matrix=b["matrix"] * MF33_MF34_CROSS_SCALE)
                            for b in cross_blocks]
        libraries["This_work"]["mf33_mf34_cross"] = cross_blocks
        print(f"\n[XCROSS] MF33↔MF34 cross ENABLED for This_work, FROM THE FILE.")
        print(f"[XCROSS]   source : {THIS_WORK_FILE}")
        print(f"[XCROSS]   form: MF34 (L=0, L1) a_0 blocks — "
              f"{cross_info.get('n_mag_bins')} magnitude bins x "
              f"{cross_info.get('n_shape_bins')} shape groups, "
              f"orders {cross_info.get('orders')}, scale="
              f"{MF33_MF34_CROSS_SCALE:g}")
        print(f"[XCROSS]   both legs relative, one file, one convention — "
              f"§L13 cannot arise and the sidecar is not read (§L8 inverted).")
        print(f"[XCROSS]   JEFF and JENDL keep a zero cross block; This_work "
              f"therefore carries a term the others do not.")
    elif MF33_MF34_CROSS_DIR and MF33_MF34_CROSS_SCALE != 0.0:
        cross_blocks, cross_info = load_mf33_mf34_cross(
            MF33_MF34_CROSS_DIR, l_max=L_MAX, scale=MF33_MF34_CROSS_SCALE,
        )
        libraries["This_work"]["mf33_mf34_cross"] = cross_blocks
        print(f"\n[XCROSS] MF33↔MF34 cross block ENABLED for This_work only.")
        print(f"[XCROSS]   source : {MF33_MF34_CROSS_DIR}")
        print(f"[XCROSS]   {cross_info['n_bins']} bins x {cross_info['n_orders']} "
              f"orders, scale={cross_info['scale']:g}")
        print(f"[XCROSS]   undefined slots set to zero: "
              f"{cross_info['n_undefined_slots']}")
        print(f"[XCROSS]   median |Cov(dsigma/sigma, a_l)| by order: " + ", ".join(
            f"a_{L} {v:.3e}" for L, v in cross_info["median_abs_rel_cov"].items()))
        # This line used to print unconditionally and so contradicted itself on
        # run 88, which DID ship a complete block (roadmap §10.1.7-B). It now
        # reports the form the loader actually chose.
        _form = cross_info.get("form")
        if _form == "group":
            print(f"[XCROSS]   form: GROUP space, Pass-1 replicas, sigma rescaled "
                  f"AFTER the collapse — PSD by congruence (roadmap §10.1.8-J).")
        elif _form == "full":
            print(f"[XCROSS]   form: complete cross-ENERGY block on the FINE grid, "
                  f"Pass-2 replica pairing — known non-PSD (roadmap §10.1.8-A).")
        else:
            print(f"[XCROSS]   form: WITHIN-BIN only; cross-ENERGY covariance is "
                  f"zero — never sampled, not asserted to vanish.")
        print(f"[XCROSS]   JEFF and JENDL keep a zero cross block; This_work "
              f"therefore carries a term the others do not.")
    else:
        print(f"\n[XCROSS] MF33↔MF34 cross block OFF (zero) — "
              f"Sigma_eval = Sigma^MF34 + Sigma^MF33, as in runs 82-86.")

    # Opt-in removal of unsupported MF34 parameters, This_work only (§10.6-1).
    if MF34_NULL_MASK:
        _z = np.load(MF34_NULL_MASK)
        _m, _g = np.asarray(_z["null_mask"], bool), np.asarray(_z["shape_grid_ev"], float)
        if _g.size != _m.shape[0] + 1:
            raise SystemExit(
                f"{MF34_NULL_MASK}: mask is {_m.shape} but the grid has {_g.size} "
                f"edges. These must come from the SAME build_group_cross run."
            )
        libraries["This_work"]["mf34_null_mask"] = _m
        libraries["This_work"]["mf34_null_grid_ev"] = _g
        print(f"\n[NULLMASK] Unsupported MF34 parameters REMOVED for This_work only.")
        print(f"[NULLMASK]   source : {MF34_NULL_MASK}")
        print(f"[NULLMASK]   {int(_m.sum())} of {_m.size} (group, order) slots, "
              f"on {_m.shape[0]} shape groups")
        print(f"[NULLMASK]   by order: " + "  ".join(
            f"a_{l+1} {int(_m[:, l].sum())}" for l in range(_m.shape[1])))
        print(f"[NULLMASK]   The mask says WHICH (group, order) parameters to "
              f"drop; it does NOT say why. Two masks are in use and they are "
              f"not the same thing:")
        print(f"[NULLMASK]     null slots      — `row_aggregator` never "
              f"populated them, the file declares up to 7.6 relative variance "
              f"anyway, and the fold applied it at full weight (§10.1.8-L14.2).")
        print(f"[NULLMASK]     degenerate slots— LIVE, but the group mean of "
              f"a_l cancels, so the file declares up to 4016 relative variance "
              f"and the fold rescales it by MF4's POINTWISE a_l "
              f"(deliverable_tape_audit.md §6.0-bis).")
        print(f"[NULLMASK]   Read the mask's own file name to know which.")
        print(f"[NULLMASK]   JEFF and JENDL are untouched — they ship no such "
              f"term, so this asymmetry REMOVES a bias in our favour.")
    else:
        print(f"\n[NULLMASK] OFF — unsupported MF34 parameters are folded at "
              f"full weight, as in runs 82-90.")

    missing_mf33 = [k for k, v in libraries.items() if v.get("mf33_rel_cov") is None]
    if missing_mf33:
        print(f"\n[WARN] No MF33 for {missing_mf33} — their magnitude term will be "
              f"zero while the others carry one. That is an unfair comparison; "
              f"check the ENDF files before using the result.")

    _fold_label = {
        "product": "<sigma * F(a_l)>          (product fold)",
        "factors": "<sigma> * F(<a_l>)        (both factors averaged)",
        "sigma":   "<sigma> * F(a_l(E0))      (magnitude averaged only)",
        "al":      "sigma(E0) * F(<a_l>)      (Legendre shape averaged only)",
        "none":    "sigma(E0) * F(a_l(E0))    (no resolution model)",
    }[FOLD_MODE]
    print(f"\nPredictive scenario, FOLD_MODE={FOLD_MODE}: {_fold_label}")
    print(f"  truncated Gaussian ±{N_SIGMA:g}σ_E, {N_WINDOW_SAMPLES} nodes; "
          f"covariance MF34 + MF33.")
    # Provenance: which evaluation was scored, and under which tag. Both are
    # environment-driven, so recording them is the only way a reader of the
    # output can tell run 82 from run 83.
    print(f"  run tag  : {RUN_TAG}")
    print(f"  This_work: {THIS_WORK_FILE}")
    print(f"  output: {OUTPUT_PARQUET}")

    try:
        tof_cache = load_tof_parameters_file(TOF_PARAMETERS_FILE)
        n_declared = sum(
            1 for v in tof_cache.values() if isinstance(v, dict)
            and (v.get("energy_resolution") or {}).get("fwhm_mev") is not None
            and not (v.get("energy_resolution") or {}).get("review_required")
        )
        n_with_data = sum(
            1 for v in tof_cache.values() if isinstance(v, dict)
            and (v.get("tof") or {}).get("flight_path_m") is not None
            and (v.get("tof") or {}).get("time_resolution_ns") is not None
        )
        print(f"Loaded TOF parameters: {len(tof_cache)} entries "
              f"({n_declared} declared EN-RSL* widths, {n_with_data} curated L/δt; "
              f"the rest fall back to "
              f"L={DEFAULT_FLIGHT_PATH_M} m, δt={DEFAULT_TIME_RESOLUTION_NS} ns)")
    except FileNotFoundError:
        tof_cache = {}
        print(f"[WARN] TOF parameters file not found: {TOF_PARAMETERS_FILE} — "
              f"all experiments use defaults")

    exfor_cache, _sorted_e = load_exfor(
        db_path=EXFOR_DB_PATH,
        target_zaids=TARGET_ZAIDS,
        mt=MT_NUMBER,
        supplementary_json_files=SUPPLEMENTARY_JSON_FILES,
        exclude_experiments=EXCLUDE_EXPERIMENTS,
    )

    energies_in_range = sorted(
        e for e in exfor_cache if E_MIN_MEV <= e <= E_MAX_MEV
    )
    print(f"Iterating {len(energies_in_range)} EXFOR energies in "
          f"[{E_MIN_MEV}, {E_MAX_MEV}] MeV")

    all_rows: List[dict] = []
    for e_mev in energies_in_range:
        all_rows.extend(build_rows_at_energy(
            e_mev, exfor_cache[e_mev], libraries, tof_cache,
        ))

    df = pd.DataFrame(all_rows)
    out_dir = Path(OUTPUT_PARQUET).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dense per-(library, experiment) Sigma_eval = MF34 sandwich + MF33 block.
    # The sandwich reads c0 per row (the folded magnitude); its a_l shape term
    # uses the nominal a_l from this lookup (second order — see docstring).
    def _a_l_lookup(lib_key: str, lib: dict, e_mev: float) -> np.ndarray:
        return interp_a_l_to_energy(lib, e_mev, L_MAX)

    eval_cov = build_eval_cov_for_groups(
        df, libraries, _a_l_lookup, l_max=L_MAX,
    )

    # Diagonal of Sigma_eval per row, for diagnostic plots only — chi^2 itself
    # uses the full block. _eval_pos lets a consumer slice the block correctly
    # when df is later filtered to a subset.
    # PSD accounting. With the cross block off this is a formality — a sum of two
    # PSD sandwiches is PSD. With it on it is NOT: Σ^MF33 and Σ^MF34 come from the
    # shipped file (collapsed, near-zero-guarded), while Σ^cross comes from the
    # raw MC, so the three are no longer guaranteed mutually consistent and a
    # negative eigenvalue is a real possibility rather than a rounding artifact.
    # A non-PSD Σ_eval can hand a datapoint negative variance, so this is checked
    # rather than assumed. Full eigendecomposition only for the small groups —
    # Cierjacks alone is 28631² — plus a diagonal check everywhere, which is
    # cheap and catches the failure mode that actually bites the χ².
    _EIG_MAX_N = 2500
    _cross_on = ((bool(MF33_MF34_CROSS_DIR) or MF33_MF34_CROSS_FROM_FILE)
                 and MF33_MF34_CROSS_SCALE != 0.0)
    n_neg_diag_total = 0
    worst_min_eig = (None, np.inf)
    n_eig_checked = 0
    n_lanczos = 0

    sigma_eval_diag = np.zeros(len(df), dtype=float)
    # The SIGNED, UNCLIPPED diagonal. sigma_eval_diag below is sqrt(max(d, 0)),
    # so a negative variance leaves no trace in the parquet at all — it becomes
    # an exact zero, indistinguishable from a genuinely zero uncertainty. That
    # cost a live run to notice.
    #
    # Keeping the raw value makes the whole damping question answerable offline:
    # Sigma_eval(s) = Sigma^MF33 + Sigma^MF34 + s*Sigma^cross is EXACTLY LINEAR in
    # the scale, so one run at any s0 fixes Sigma^cross's diagonal everywhere
    # (c = (v(s0) - v(0))/s0) and hence the largest PSD-safe scale, with no
    # further cluster time. Additive column: every existing consumer is
    # unaffected, and no value already written changes.
    eval_var_diag = np.zeros(len(df), dtype=float)
    eval_pos = np.full(len(df), -1, dtype=np.int32)
    for (lib_key, exp_id), block in eval_cov.items():
        d = np.diag(block)
        n_neg = int(np.count_nonzero(d < 0.0))
        if n_neg:
            n_neg_diag_total += n_neg
            print(f"[PSD] negative Sigma_eval diagonal: ({lib_key}, {exp_id}) "
                  f"{n_neg}/{d.size} points, min {d.min():.3e}")
        if block.shape[0] <= _EIG_MAX_N and block.shape[0] > 0:
            n_eig_checked += 1
            lo = float(np.linalg.eigvalsh(block.astype(np.float64)).min())
            scale_ = float(np.max(np.abs(d))) or 1.0
            if lo / scale_ < worst_min_eig[1]:
                worst_min_eig = ((lib_key, exp_id), lo / scale_)
        elif _cross_on and block.shape[0] > _EIG_MAX_N:
            # A positive diagonal does NOT make a matrix PSD, and the two groups
            # this skips (Cierjacks 28631, K&S 13698) are 90 % of the points. An
            # exact eigendecomposition is O(N^3) and out of the question, but the
            # SMALLEST eigenvalue is reachable by Lanczos in a few dozen matvecs.
            # Only worth it when the cross block is on — with it off the sum of
            # two PSD sandwiches is PSD by construction.
            lo = _min_eig_lanczos(block)
            if lo is not None:
                n_lanczos += 1
                scale_ = float(np.max(np.abs(d))) or 1.0
                if lo / scale_ < worst_min_eig[1]:
                    worst_min_eig = ((lib_key, exp_id), lo / scale_)
                print(f"[PSD] Lanczos min eigenvalue ({lib_key}, {exp_id}) "
                      f"N={block.shape[0]}: {lo:.3e} (normalised {lo / scale_:.3e})")
            else:
                print(f"[PSD] Lanczos did NOT converge for ({lib_key}, {exp_id}) "
                      f"N={block.shape[0]} — smallest eigenvalue unknown, not zero")
        mask = (df["library"].astype(str) == lib_key) & \
               (df["experiment_id"].astype(str) == exp_id)
        idx = np.flatnonzero(mask.to_numpy())
        if idx.size != block.shape[0]:
            raise RuntimeError(
                f"Row count mismatch for ({lib_key}, {exp_id}): "
                f"{idx.size} parquet rows vs {block.shape[0]} cov rows"
            )
        sigma_eval_diag[idx] = np.sqrt(np.maximum(d, 0.0))
        eval_var_diag[idx] = d
        eval_pos[idx] = np.arange(idx.size, dtype=np.int32)
    df["sigma_eval_diag"] = sigma_eval_diag
    df["sigma_eval_var_diag"] = eval_var_diag
    df["_eval_pos"] = eval_pos

    print(f"\n[PSD] negative Sigma_eval diagonal entries: {n_neg_diag_total} "
          f"(clipped to 0 in sigma_eval_diag; kept signed in sigma_eval_var_diag, "
          f"which is what makes the damping scale solvable offline)")
    if worst_min_eig[0] is not None:
        print(f"[PSD] worst normalised min eigenvalue: {worst_min_eig[1]:.3e} "
              f"at {worst_min_eig[0]}")
        print(f"[PSD]   exact eigendecomposition on {n_eig_checked} groups of "
              f"N <= {_EIG_MAX_N}; Lanczos on {n_lanczos} larger groups")
        if not _cross_on:
            print(f"[PSD]   large groups were NOT checked (cross block off ⇒ the sum "
                  f"of two PSD sandwiches is PSD by construction)")

    df.to_parquet(OUTPUT_PARQUET, index=False)
    sidecar = OUTPUT_PARQUET + ".eval_cov.npz"
    save_eval_cov(sidecar, eval_cov)

    print(f"\nSaved: {OUTPUT_PARQUET}")
    print(f"  shape: {df.shape}")
    print(f"  sidecar: {sidecar} ({len(eval_cov)} per-experiment blocks)")
    if not df.empty:
        print(f"  libraries: {sorted(df['library'].unique())}")


if __name__ == "__main__":
    main()
