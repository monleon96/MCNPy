"""
Precompute chi^2 data for Fe-56 elastic angular distributions.

Methodology: like precompute_chi2_folded_c0.py, but the angular *shape* is folded
too. Both the cross section sigma (-> c_0) AND the Legendre coefficients a_l are
averaged over each experiment's TOF energy-resolution window; sigma_eval = MF34.

Motivation. A real TOF experiment measures the differential cross section averaged
over its finite incident-energy resolution, so at a nominal energy E both the
normalization sigma(E) and the angular shape a_l(E) are broadened across the
resolution window. The folded_c0 methodology already broadens sigma; here we also
broaden a_l, exactly like the "both" case in kika/exfor/tests/energy_resolution_impact.ipynb.

The averaging kernel is a *truncated Gaussian* over [E - N_SIGMA*sigma_E,
E + N_SIGMA*sigma_E], renormalized to unit weight. N_SIGMA is a config knob (run
1 and 3 to see how much the window width matters). At large N_SIGMA the sigma fold
recovers folded_c0's full-range Gauss-Hermite c_0 as a nested limit.

sigma and a_l are folded *separately* (<sigma> and <a_l>, then reconstructed),
not as a product <sigma*a_l>: MF3 sigma and MF4 a_l come from different underlying
data, so a product fold would impose a point-by-point energy correlation the
evaluation does not claim.

For each EXFOR datapoint at energy E in [E_MIN_MEV, E_MAX_MEV] (across all
experiments, not just K&S), each library's evaluation is constructed as:

    c_0(E):
        <sigma_MF3>_fold / (4 pi), the truncated-Gaussian window average of the
        library's own pointwise MF3 sigma(E'). sigma(E') is interpolated on MF3's
        native pointwise grid.

    a_l(E):
        <a_l_MF4>_fold, the truncated-Gaussian window average of the library's own
        MF4 a_l(E') (interpolated per order), truncated to L=L_MAX. Folded with the
        same kernel and window as c_0.

    y_eval(E, mu):
        c_0/(4 pi) * [1 + sum_l (2l+1) <a_l> P_l(mu)], reconstructed from the two
        separately-folded factors.

    sigma_eval(E, mu):
        MF34 sandwich propagated to dsigma/dOmega(mu). Value-only: no MF33 magnitude
        term and no c_0 fit-error contribution. The sandwich reads the folded c_0
        from each row; its a_l shape term uses the nominal (unfolded) a_l via the
        lookup (same treatment as folded_c0) — a second-order effect.

Output: one row per (library, experiment, datapoint). The notebook / cluster driver
subsets into all / no_KS / only_KS etc., exactly like the other methodologies.

Run: python scripts/precompute_chi2_folded_al_c0.py
     (edit N_SIGMA and rerun to sweep the window width; the output filename encodes it)
"""
from __future__ import annotations

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

from kika.endf import read_endf
from kika.cov.mf34_covmat import MF34CovMat

from scripts.precompute_chi2_exfor_c0 import load_exfor, build_experiment_dataframe
from scripts.precompute_chi2_library_c0 import interp_a_l_to_energy
from scripts.eval_covariance import build_eval_cov_for_groups, save_eval_cov
from scripts.tof_parameters import (
    load_tof_parameters_file,
    get_tof_parameters,
    compute_sigma_E,
)


# ── Configuration ─────────────────────────────────────────────────────────────

MT_NUMBER = 2  # elastic scattering

# ── Resolution window (the fold applied to both sigma and a_l) ──
# N_SIGMA sets the truncated-Gaussian window half-width in units of sigma_E. The
# user sweeps this (1 and 3); the output filename encodes it. N_WINDOW_SAMPLES is
# the uniform sampling density across the window (enough to resolve MF3 resonance
# structure inside the window).
N_SIGMA           = 1.0
N_WINDOW_SAMPLES  = 65

# ── Library ENDF files (one per evaluation compared in the chi^2 analysis) ──
# This_work uses its own MF3 *and* MF4 from the pipeline's nominal ENDF, for a
# fully symmetric treatment (no nominal_fits.parquet dependency).
THIS_WORK_DIR  = "/share_snc/snc/JuanMonleon/ENDF_samples/new_test_82_mt1fix"
THIS_WORK_FILE = f"{THIS_WORK_DIR}/26-Fe-56g_nominal_mg.endf"

JEFF_FILE  = "/share_snc/snc/JuanMonleon/jeff40_with_MF4_from_jeff33/26-Fe-56g.txt"
JENDL_FILE = "/share_snc/snc/JuanMonleon/JENDL-5/260560.jendl5"

# ── EXFOR sources ──
EXFOR_DB_PATH = "/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db"
TARGET_ZAIDS  = [26056, 26000]
SUPPLEMENTARY_JSON_FILES = [
    "/share_snc/snc/JuanMonleon/EXFOR/data_v1/27673002.json",
]
EXCLUDE_EXPERIMENTS = ["32246002", "400750022"]  # Tostkii 1957; Morozov 1972 pointer 2 (SPA-fitted duplicate of 400750021)

# Note: K&S subentry labels (used to tag each EXFOR row with 'Kinney_1976',
# 'Smith_1980' or 'OTHER') live in precompute_chi2_exfor_c0.py — they are read
# inside the imported build_experiment_dataframe(). Edit them there.

# ── TOF energy resolution (same source/defaults as the sampling pipeline) ──
TOF_PARAMETERS_FILE        = "/share_snc/snc/JuanMonleon/EXFOR/exfor_tof_parameters.json"
DEFAULT_FLIGHT_PATH_M      = 27.037  # ORELA-like; used when the JSON has no entry
DEFAULT_TIME_RESOLUTION_NS = 5.0
# Must match DELTA_T_IS_FWHM in exfor_to_endf_sampling_v2.py, or this chi2 folds
# the library MF3 over a different resolution kernel than the fit used. Runs <=81
# read delta_t as a sigma; from run 82 it is a FWHM (sigma_E = FWHM/2.3548).
# Passed explicitly because get_tof_parameters() now defaults it to True.
DELTA_T_IS_FWHM            = True

# Energy range for the analysis. EXFOR datapoints outside this window are
# dropped at the precompute stage.
E_MIN_MEV = 0.85
E_MAX_MEV = 4.0

# Legendre truncation. In [0.85, 4] MeV the highest order any of the three
# libraries reports is 6, so all evaluations are compared at L=6.
L_MAX = 6

# ── Output (filename encodes the window width so the two runs don't collide) ──
OUTPUT_PARQUET = f"/share_snc/snc/JuanMonleon/chi2/chi2_data_folded_al_c0_82_ns{N_SIGMA:g}.parquet"


# ── Resolution window ─────────────────────────────────────────────────────────

def _resolution_window(e_mev: float, sigma_E_mev: float) -> Tuple[np.ndarray, np.ndarray]:
    """Truncated-Gaussian resolution kernel nodes and weights.

    Returns (sample_e_ev, weights): N_WINDOW_SAMPLES energies uniformly spaced on
    [e - N_SIGMA*sigma_E, e + N_SIGMA*sigma_E] and Gaussian weights
    w_i ∝ exp(-½((E_i - e)/sigma_E)²) renormalized to sum 1. With sigma_E <= 0 the
    kernel collapses to a single node at e (delta), matching fold_xs_over_resolution.
    """
    if sigma_E_mev <= 0.0 or N_WINDOW_SAMPLES < 2:
        return np.array([e_mev * 1e6]), np.array([1.0])
    half = N_SIGMA * sigma_E_mev
    e_grid_mev = np.linspace(e_mev - half, e_mev + half, N_WINDOW_SAMPLES)
    w = np.exp(-0.5 * ((e_grid_mev - e_mev) / sigma_E_mev) ** 2)
    w /= w.sum()
    return e_grid_mev * 1e6, w


# ── ENDF loading (MF3 + MF4 + MF34; no MF33, so no NJOY/PENDF) ────────────────

def load_library_folded(filepath: str, label: str) -> Dict:
    """Read MF3 (pointwise sigma), MF4 (a_l grid), MF34 (a_l covariance)."""
    print(f"Loading {label} from {filepath}")
    endf = read_endf(filepath, mf_numbers=[3, 4])

    mt4 = endf.get_file(4).sections[MT_NUMBER]
    energies_mf4_mev = np.asarray(mt4.legendre_energies, dtype=float) / 1e6
    coefficients = [np.asarray(c, dtype=float) for c in mt4.legendre_coefficients]
    print(f"  MF4: {len(energies_mf4_mev)} grid points, "
          f"E=[{energies_mf4_mev[0]:.4g}, {energies_mf4_mev[-1]:.4g}] MeV, "
          f"max L={max(len(c) for c in coefficients)}")

    mf3_file = endf.get_file(3)
    if mf3_file is None or MT_NUMBER not in mf3_file.sections:
        raise RuntimeError(
            f"{label}: MF3/MT={MT_NUMBER} not present in {filepath}; the folded "
            f"methodology needs each library's own elastic cross section."
        )
    mt3 = mf3_file.sections[MT_NUMBER]
    e_mf3_ev = np.asarray(mt3.energies, dtype=float)
    xs_mf3 = np.asarray(mt3.cross_sections, dtype=float)
    print(f"  MF3: {len(e_mf3_ev)} pointwise σ values, "
          f"E=[{e_mf3_ev[0] / 1e6:.4g}, {e_mf3_ev[-1] / 1e6:.4g}] MeV")

    try:
        mf34 = MF34CovMat.from_endf(filepath, energy_unit="MeV")
        mf34 = mf34.filter_by_isotope_reaction(26056, MT_NUMBER)
        print(f"  MF34: {mf34.num_matrices} covariance blocks")
    except Exception as e:
        mf34 = None
        print(f"  MF34: not available ({e})")

    return dict(
        label=label,
        energies_mf4_mev=energies_mf4_mev, coefficients=coefficients,
        e_mf3_ev=e_mf3_ev, xs_mf3=xs_mf3,
        mf34=mf34,
    )


def _fold_a_l(lib: Dict, sample_e_ev: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Window-average a_l over the resolution kernel: <a_l> = sum_i w_i a_l(E_i).

    Returns a length-L_MAX vector [<a_1>..<a_L_MAX>]. Each order is interpolated
    on MF4's native grid at every window node, then Gaussian-weighted. With a
    single node this reduces to the nominal a_l(E).
    """
    if sample_e_ev.size == 1:
        return interp_a_l_to_energy(lib, sample_e_ev[0] / 1e6, L_MAX)
    a_samples = np.empty((sample_e_ev.size, L_MAX), dtype=float)
    for i, e_ev in enumerate(sample_e_ev):
        a_samples[i, :] = interp_a_l_to_energy(lib, e_ev / 1e6, L_MAX)
    return weights @ a_samples


# ── Per-energy row builder ────────────────────────────────────────────────────

def build_rows_at_energy(
    e_mev: float,
    experiments_at_energy: List[Tuple[pd.DataFrame, Dict]],
    libraries: Dict[str, Dict],
    tof_cache: Dict,
) -> List[dict]:
    """Compute one row per (library, experiment, datapoint) at this energy.

    Both c_0 and a_l are folded per experiment (sigma_E depends on the subentry's
    TOF params), so the fold is computed inside the experiment loop rather than
    cached once per energy.
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
            # <sigma>_fold over the truncated-Gaussian window.
            sigma_samples = np.interp(sample_e_ev, lib["e_mf3_ev"], lib["xs_mf3"])
            sigma_avg_b = float(weights @ sigma_samples)
            if not np.isfinite(sigma_avg_b) or sigma_avg_b <= 0:
                continue
            c0 = sigma_avg_b / (4.0 * np.pi)

            # <a_l>_fold over the same window (separate fold, then reconstruct).
            a_l = _fold_a_l(lib, sample_e_ev, weights)
            full = np.empty(L_MAX + 1, dtype=float)
            full[0] = c0
            for l in range(1, L_MAX + 1):
                full[l] = c0 * (2 * l + 1) * a_l[l - 1]

            y_eval = legval(mu, full)
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
                    "a1_folded":       float(a_l[0]),
                    "sigma_E_mev":     float(sigma_E_mev),
                    "n_sigma":         float(N_SIGMA),
                    "tof_source":      str(tof.source),
                })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    libraries = {
        "JEFF":      load_library_folded(JEFF_FILE,       "JEFF-4.0"),
        "JENDL":     load_library_folded(JENDL_FILE,      "JENDL-5"),
        "This_work": load_library_folded(THIS_WORK_FILE,  "This work"),
    }

    print(f"\nResolution fold: truncated Gaussian, ±{N_SIGMA:g}σ_E window, "
          f"{N_WINDOW_SAMPLES} samples; both c_0 and a_l folded (separately).")

    try:
        tof_cache = load_tof_parameters_file(TOF_PARAMETERS_FILE)
        n_with_data = sum(
            1 for v in tof_cache.values()
            if (v.get("energy_resolution_input") or {}).get("distance", {}).get("value") is not None
        )
        print(f"Loaded TOF parameters: {len(tof_cache)} subentries "
              f"({n_with_data} with measured L/δt; the rest fall back to "
              f"L={DEFAULT_FLIGHT_PATH_M} m, δt={DEFAULT_TIME_RESOLUTION_NS} ns)")
    except FileNotFoundError:
        tof_cache = {}
        print(f"[WARN] TOF parameters file not found: {TOF_PARAMETERS_FILE} — "
              f"all experiments use defaults "
              f"(L={DEFAULT_FLIGHT_PATH_M} m, δt={DEFAULT_TIME_RESOLUTION_NS} ns)")

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

    # Build per-(library, experiment) dense Sigma_eval blocks. Each library dict
    # carries no MF33 fields, so build_mf33_block returns zero — MF34 (shape)
    # only. The MF34 sandwich reads c0 from each row (the folded c0); its a_l
    # shape term uses the nominal a_l from this lookup (see module docstring).
    def _a_l_lookup(lib_key: str, lib: dict, e_mev: float) -> np.ndarray:
        return interp_a_l_to_energy(lib, e_mev, L_MAX)

    eval_cov = build_eval_cov_for_groups(
        df, libraries, _a_l_lookup, l_max=L_MAX,
    )

    # Diagonal of Sigma_eval, exposed per row for diagnostic plots only — chi^2
    # itself uses the full block via scripts.chi2_metrics. _eval_pos lets the
    # consumer slice the block correctly when df is later filtered to a subset.
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
        print("  rows per (library, ks_subentry):")
        print(df.groupby(["library", "ks_subentry"]).size().to_string())
        print(f"  energies: {df['energy_mev'].nunique()} unique, "
              f"range [{df['energy_mev'].min():.4g}, {df['energy_mev'].max():.4g}] MeV")
        print(f"  experiments: {df['experiment_id'].nunique()}")
        print("  σ_E (MeV) by tof_source:")
        print(df.groupby("tof_source")["sigma_E_mev"]
              .agg(["count", "min", "median", "max"]).to_string())


if __name__ == "__main__":
    main()
