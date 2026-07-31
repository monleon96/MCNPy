"""
Precompute chi^2 data for Fe-56 elastic angular distributions — PREDICTIVE scenario.

Methodology: every evaluation is judged by **what it actually ships**. Its own MF3
and its own MF4 are forward-folded through each experiment's TOF energy
resolution, and the variance budget is that library's own MF34 **and MF33**.

This is the "S1 / final-file predictive" scenario of `docs/mf3_mf33_roadmap.md`.
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
from pathlib import Path
from typing import Dict, List, Tuple

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
THIS_WORK_FILE = f"{THIS_WORK_DIR}/26-Fe-56g_nominal_mg.endf"

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
    sigma_eval_diag = np.zeros(len(df), dtype=float)
    eval_pos = np.full(len(df), -1, dtype=np.int32)
    for (lib_key, exp_id), block in eval_cov.items():
        mask = (df["library"].astype(str) == lib_key) & \
               (df["experiment_id"].astype(str) == exp_id)
        idx = np.flatnonzero(mask.to_numpy())
        if idx.size != block.shape[0]:
            raise RuntimeError(
                f"Row count mismatch for ({lib_key}, {exp_id}): "
                f"{idx.size} parquet rows vs {block.shape[0]} cov rows"
            )
        sigma_eval_diag[idx] = np.sqrt(np.maximum(np.diag(block), 0.0))
        eval_pos[idx] = np.arange(idx.size, dtype=np.int32)
    df["sigma_eval_diag"] = sigma_eval_diag
    df["_eval_pos"] = eval_pos

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
