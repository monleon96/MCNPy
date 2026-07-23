"""
Precompute chi^2 data for Fe-56 elastic angular distributions.

Methodology: c_0 from each library's MF3 cross section, sigma_eval = MF34 + MF33.

For each EXFOR datapoint at energy E in [E_MIN_MEV, E_MAX_MEV] (across all
experiments, not just K&S), each library's evaluation is constructed as:

    a_l(E):
        MF4 a_l interpolated linearly to E and truncated to L=L_MAX (for
        all libraries — JEFF, JENDL, and This_work).

    c_0(E):
        ⟨σ_MF3⟩ / (4π) averaged over the MF4 energy bin containing E.
        Each library uses its own MF4 grid. Integration is trapezoidal on
        MF3's native pointwise grid (already linearised through the
        resonance region in the ENDF file). The bin average is robust to
        resonance peaks/dips that would make a pointwise σ(E_EXFOR) noisy.

    sigma_eval(E, mu):
        sqrt(
            (MF34 sandwich on a_l propagated to dsigma/dOmega(mu))^2
          + (delta_sigma_rel * |y_eval(mu)|)^2
        )
        with delta_sigma_rel = sqrt of the MF33 relative variance of
        sigma averaged over the same MF4 bin used for c_0 (length-weighted
        average across the MF33 bins overlapping that window). The MF34
        and MF33 contributions are summed in quadrature — libraries do
        not ship MF33<->MF34 cross-covariance.

Output: one row per (library, experiment, datapoint). The notebook subsets
into all / no_KS / only_KS for the 6 scenarios.

Interpretation note: c_0 from each library's MF3 partly retraces the EXFOR
data the library was fit to, and sigma_eval here propagates each library's
own claimed magnitude uncertainty (MF33). The resulting chi^2 is therefore a
*library-trust self-consistency check* — "do the libraries agree with their
own MF33-claimed uncertainties on the EXFOR points they were fit to?" — not
an objective shape-agreement metric. The principled shape comparison lives
in precompute_chi2_exfor_c0.py / chi2_analysis_exfor_c0.ipynb, where c_0 is
anchored to Kinney/Smith directly and only MF34 (shape) covariance is
propagated.

Run: python scripts/precompute_chi2_library_c0.py
"""
from __future__ import annotations

import os
import sys
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

from kika.endf import read_endf
from kika.cov.mf34_covmat import MF34CovMat

from scripts.precompute_chi2_exfor_c0 import (
    load_exfor, build_experiment_dataframe,
)
from kika.processing.njoy_pendf_cache import (
    mf33_needs_pendf, get_or_create_pendf, read_pendf_mf3_sections,
)
from scripts.eval_covariance import build_eval_cov_for_groups, save_eval_cov


# ── Configuration ─────────────────────────────────────────────────────────────

MT_NUMBER = 2  # elastic scattering

# ── Library ENDF files (one per evaluation compared in the chi^2 analysis) ──
# This_work uses the pipeline's nominal ENDF written by
# exfor_to_endf_sampling_v2.py into OUTPUT_DIR.
THIS_WORK_DIR  = "/share_snc/snc/JuanMonleon/ENDF_samples/new_test_80"
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
# 'Smith_1980' or 'OTHER') live in precompute_chi2_exfor_c0.py — they are
# read inside the imported build_experiment_dataframe(). Edit them there.

# Energy range for the analysis. EXFOR datapoints outside this window are
# dropped at the precompute stage.
E_MIN_MEV = 0.85
E_MAX_MEV = 4.0

# Legendre truncation. In [0.85, 4] MeV the highest order any of the three
# libraries reports is 6, so all evaluations are compared at L=6.
L_MAX = 6

# ── Output ──
OUTPUT_PARQUET = "/share_snc/snc/JuanMonleon/chi2/chi2_data_library_c0_80.parquet"

# ── NJOY-PENDF cache (only used to resolve MF33 NC LTY=0 sub-subsections, e.g.
# JENDL). JEFF/This_work have no NC and skip NJOY entirely. ──
NJOY_EXECUTABLE  = "/soft_snc/NJOY/2016.78/bin/njoy"
PENDF_TOLERANCE  = 0.005  # matches kika-app default
PENDF_CACHE_DIR  = "/share_snc/snc/JuanMonleon/kika_pendf_cache"


# ── ENDF loading ──────────────────────────────────────────────────────────────

def load_library_lib_c0(filepath: str, label: str) -> Dict:
    """Read MF3, MF4, MF33 (sigma covariance), MF34 (a_l covariance) from one ENDF."""
    print(f"Loading {label} from {filepath}")
    endf = read_endf(filepath, mf_numbers=[3, 4, 33])

    # MF4: a_l grid + coefficients
    mt4 = endf.get_file(4).sections[MT_NUMBER]
    energies_mf4_mev = np.asarray(mt4.legendre_energies, dtype=float) / 1e6
    coefficients = [np.asarray(c, dtype=float) for c in mt4.legendre_coefficients]
    print(f"  MF4: {len(energies_mf4_mev)} grid points, "
          f"E=[{energies_mf4_mev[0]:.4g}, {energies_mf4_mev[-1]:.4g}] MeV, "
          f"max L={max(len(c) for c in coefficients)}")

    # MF3: pointwise sigma(E) for MT=2 (in eV / barns).
    mt3 = endf.get_file(3).sections[MT_NUMBER]
    e_mf3_ev = np.asarray(mt3.energies, dtype=float)
    xs_mf3 = np.asarray(mt3.cross_sections, dtype=float)
    print(f"  MF3: {len(e_mf3_ev)} pointwise σ values")

    # MF34: a_l covariance (a-l covariance reader handles relative/absolute).
    try:
        mf34 = MF34CovMat.from_endf(filepath, energy_unit="MeV")
        mf34 = mf34.filter_by_isotope_reaction(26056, MT_NUMBER)
        print(f"  MF34: {mf34.num_matrices} covariance blocks")
    except Exception as e:
        mf34 = None
        print(f"  MF34: not available ({e})")

    # MF33: sigma(E) covariance for MT=2. Libraries with NC LTY=0 sub-subsections
    # (e.g. JENDL) need reconstructed σ(E) for the sum-rule contraction — we
    # generate a PENDF via NJOY (cached) and pass its MF3 to to_xs_covmat.
    # Libraries with only relative NI records (JEFF/This_work) skip NJOY.
    mf33_grid_ev: Optional[np.ndarray] = None
    mf33_rel_cov: Optional[np.ndarray] = None
    try:
        mf33_file = endf.get_file(33)
        if mf33_file is None or MT_NUMBER not in mf33_file.sections:
            raise RuntimeError("MF33/MT=2 not present")
        mt33 = mf33_file.sections[MT_NUMBER]

        kwargs = {}
        if mf33_needs_pendf(mf33_file, MT_NUMBER):
            kwargs["sibling_sections"] = mf33_file.sections
            pendf_path = get_or_create_pendf(
                filepath, tolerance=PENDF_TOLERANCE,
                njoy_exe=NJOY_EXECUTABLE, cache_dir=PENDF_CACHE_DIR,
            )
            merged_mf3 = {
                int(k): v
                for k, v in endf.get_file(3).sections.items()
            }
            merged_mf3.update(read_pendf_mf3_sections(pendf_path))
            kwargs["mf3_sections"] = merged_mf3
        cov33 = mt33.to_xs_covmat(energy_unit="MeV", **kwargs)
        if cov33.num_matrices == 0:
            raise RuntimeError("MF33 to_xs_covmat returned no matrices")

        # Find the (MT=2, MT=2) block.
        idx = None
        for i in range(cov33.num_matrices):
            if cov33.reaction_rows[i] == MT_NUMBER and cov33.reaction_cols[i] == MT_NUMBER:
                idx = i
                break
        if idx is None:
            raise RuntimeError("No MT=2 self-block in MF33 covariance")

        # NB: energy_grids are stored in eV regardless of the energy_unit flag.
        mf33_grid_ev = np.asarray(cov33.energy_grids[idx], dtype=float)
        mat = np.asarray(cov33.matrices[idx], dtype=float)
        if not bool(cov33.is_relative[idx]):
            print(f"  MF33: matrix is absolute — converting to relative against MF3")
            sigma_at_bin_mid = _bin_average_sigma_array(
                e_mf3_ev, xs_mf3, mf33_grid_ev,
            )
            denom = np.outer(sigma_at_bin_mid, sigma_at_bin_mid)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_cov = np.where(denom > 0, mat / denom, 0.0)
            mf33_rel_cov = np.where(np.isfinite(rel_cov), rel_cov, 0.0)
        else:
            mf33_rel_cov = mat
        print(f"  MF33: {mf33_rel_cov.shape[0]} relative-variance bins, "
              f"sqrt-diag in [0.85,4] MeV "
              f"≈ [{_sqrt_diag_summary(mf33_grid_ev, np.diag(mf33_rel_cov))}]")
    except Exception as e:
        print(f"  MF33: not available ({e}) — magnitude term will be 0")

    return dict(
        label=label,
        energies_mf4_mev=energies_mf4_mev, coefficients=coefficients,
        e_mf3_ev=e_mf3_ev, xs_mf3=xs_mf3,
        mf34=mf34,
        mf33_grid_ev=mf33_grid_ev, mf33_rel_cov=mf33_rel_cov,
    )


def _sqrt_diag_summary(grid_ev: np.ndarray, rel_var: np.ndarray) -> str:
    """Quick-look sqrt(diag) range over [E_MIN, E_MAX]."""
    e_lo = E_MIN_MEV * 1e6
    e_hi = E_MAX_MEV * 1e6
    lo = max(0, int(np.searchsorted(grid_ev, e_lo, side="right") - 1))
    hi = min(rel_var.size - 1, int(np.searchsorted(grid_ev, e_hi, side="left")))
    if hi <= lo:
        return "n/a"
    sub = np.sqrt(np.maximum(rel_var[lo:hi + 1], 0.0))
    return f"min={sub.min():.3g}, median={np.median(sub):.3g}, max={sub.max():.3g}"


# ── MF4 bin lookup and MF3 / MF33 averaging ───────────────────────────────────

def find_mf4_bin(energies_mf4_mev: np.ndarray, e_mev: float) -> Tuple[float, float]:
    """Return the (E_lo, E_hi) MF4 bin containing e_mev. Clamps at endpoints."""
    if e_mev <= energies_mf4_mev[0]:
        return float(energies_mf4_mev[0]), float(energies_mf4_mev[1])
    if e_mev >= energies_mf4_mev[-1]:
        return float(energies_mf4_mev[-2]), float(energies_mf4_mev[-1])
    idx_hi = int(np.searchsorted(energies_mf4_mev, e_mev, side="right"))
    return float(energies_mf4_mev[idx_hi - 1]), float(energies_mf4_mev[idx_hi])


def _trapezoid_mean(x: np.ndarray, y: np.ndarray, x_lo: float, x_hi: float) -> float:
    """⟨y⟩ = (1/(x_hi-x_lo)) ∫_{x_lo}^{x_hi} y(x) dx, trapezoidal on x-native grid.

    Edge crossings are linearly interpolated to honor the exact window. Returns
    NaN if no coverage.
    """
    if x_hi <= x_lo:
        return float("nan")
    # Indices of native points strictly inside (x_lo, x_hi)
    i_lo = int(np.searchsorted(x, x_lo, side="right"))
    i_hi = int(np.searchsorted(x, x_hi, side="left"))
    pieces_x: List[float] = [x_lo]
    pieces_y: List[float] = [float(np.interp(x_lo, x, y))]
    if i_lo < i_hi:
        pieces_x.extend(x[i_lo:i_hi].tolist())
        pieces_y.extend(y[i_lo:i_hi].tolist())
    pieces_x.append(x_hi)
    pieces_y.append(float(np.interp(x_hi, x, y)))
    xs = np.asarray(pieces_x, dtype=float)
    ys = np.asarray(pieces_y, dtype=float)
    integral = float(np.trapezoid(ys, xs))
    return integral / (x_hi - x_lo)


def avg_sigma_over_bin(library: Dict, e_mev: float) -> Tuple[float, float, float]:
    """Average MF3 σ over the MF4 bin containing e_mev.

    Returns (sigma_avg_b, e_lo_mev, e_hi_mev). σ_avg is in barns.
    """
    e_lo, e_hi = find_mf4_bin(library["energies_mf4_mev"], e_mev)
    sigma_avg = _trapezoid_mean(
        library["e_mf3_ev"], library["xs_mf3"], e_lo * 1e6, e_hi * 1e6,
    )
    return sigma_avg, e_lo, e_hi


def _bin_average_sigma_array(
    e_mf3_ev: np.ndarray, xs_mf3: np.ndarray, mf33_bins_ev: np.ndarray,
) -> np.ndarray:
    """Per-MF33-bin average of σ for absolute → relative MF33 conversion."""
    n = mf33_bins_ev.size - 1
    out = np.zeros(n, dtype=float)
    for k in range(n):
        out[k] = _trapezoid_mean(
            e_mf3_ev, xs_mf3, mf33_bins_ev[k], mf33_bins_ev[k + 1],
        )
    return out


def avg_mf33_rel_var_over_bin(
    library: Dict, e_lo_mev: float, e_hi_mev: float,
) -> float:
    """Length-weighted average of MF33 diagonal relative variance over (e_lo, e_hi).

    MF33 is piecewise-constant per ENDF spec, so this is equivalent to
    summing (Δ_overlap × rel_var_k) / (e_hi - e_lo) over the MF33 bins
    overlapping the requested window. Returns 0.0 if MF33 is unavailable
    or fully outside the MF33 grid. Diagnostic only — the chi^2 path uses
    the full off-diagonal MF33 covariance via build_mf33_block.
    """
    grid = library.get("mf33_grid_ev")
    rel_cov = library.get("mf33_rel_cov")
    if grid is None or rel_cov is None or rel_cov.shape[0] == 0:
        return 0.0
    rel_var = np.diag(rel_cov)
    e_lo = e_lo_mev * 1e6
    e_hi = e_hi_mev * 1e6
    if e_hi <= e_lo:
        return 0.0
    if e_hi <= grid[0] or e_lo >= grid[-1]:
        return 0.0

    n = rel_var.size
    # MF33 bin k spans (grid[k], grid[k+1]).
    k_first = max(0, int(np.searchsorted(grid, e_lo, side="right") - 1))
    k_last = min(n - 1, int(np.searchsorted(grid, e_hi, side="left")))
    weighted_sum = 0.0
    width_sum = 0.0
    for k in range(k_first, k_last + 1):
        bin_lo = max(grid[k], e_lo)
        bin_hi = min(grid[k + 1], e_hi)
        if bin_hi <= bin_lo:
            continue
        w = bin_hi - bin_lo
        v = float(rel_var[k])
        if not np.isfinite(v) or v < 0:
            continue
        weighted_sum += w * v
        width_sum += w
    return weighted_sum / width_sum if width_sum > 0 else 0.0


# ── a_l interpolation (shared signature with the EXFOR-c0 script) ─────────────

def interp_a_l_to_energy(
    library: Dict, e_mev: float, l_max: int,
) -> np.ndarray:
    """Interpolate a_l (l=1..l_max) from a library's MF4 grid to e_mev (linear)."""
    grid = library["energies_mf4_mev"]
    coeffs = library["coefficients"]
    if e_mev <= grid[0]:
        a_full = coeffs[0]
    elif e_mev >= grid[-1]:
        a_full = coeffs[-1]
    else:
        idx_hi = int(np.searchsorted(grid, e_mev, side="right"))
        idx_lo = idx_hi - 1
        e_lo, e_hi = grid[idx_lo], grid[idx_hi]
        w = (e_mev - e_lo) / (e_hi - e_lo)
        c_lo, c_hi = coeffs[idx_lo], coeffs[idx_hi]
        n = max(len(c_lo), len(c_hi))
        c_lo_p = np.zeros(n); c_lo_p[:len(c_lo)] = c_lo
        c_hi_p = np.zeros(n); c_hi_p[:len(c_hi)] = c_hi
        a_full = (1.0 - w) * c_lo_p + w * c_hi_p

    out = np.zeros(l_max, dtype=float)
    n = min(l_max, len(a_full))
    out[:n] = a_full[:n]
    return out


# ── Per-energy row builder ────────────────────────────────────────────────────

def build_rows_at_energy(
    e_mev: float,
    experiments_at_energy: List[Tuple[pd.DataFrame, Dict]],
    libraries: Dict[str, Dict],
) -> List[dict]:
    """Compute one row per (library, experiment, datapoint) at this energy."""
    per_lib: Dict[str, dict] = {}
    for lib_key, lib in libraries.items():
        a_l = interp_a_l_to_energy(lib, e_mev, L_MAX)
        sigma_avg_b, e_lo_mev, e_hi_mev = avg_sigma_over_bin(lib, e_mev)
        if not np.isfinite(sigma_avg_b) or sigma_avg_b <= 0:
            return []
        c0 = sigma_avg_b / (4.0 * np.pi)

        delta_sigma_rel = float(np.sqrt(
            max(avg_mf33_rel_var_over_bin(lib, e_lo_mev, e_hi_mev), 0.0)
        ))

        # Full Legendre coefficient vector for legval: [c_0, c_1, ..., c_L]
        # with c_l = c_0 * (2l+1) * a_l.
        full = np.empty(L_MAX + 1, dtype=float)
        full[0] = c0
        for l in range(1, L_MAX + 1):
            full[l] = c0 * (2 * l + 1) * a_l[l - 1]

        per_lib[lib_key] = dict(
            c0=c0, a_l=a_l, full=full, l_max_used=L_MAX,
            mf34=lib.get("mf34"),
            sigma_avg_b=sigma_avg_b,
            e_bin_lo=e_lo_mev, e_bin_hi=e_hi_mev,
            delta_sigma_rel=delta_sigma_rel,
        )

    rows: List[dict] = []
    for cache_df, meta in experiments_at_energy:
        exp_df = build_experiment_dataframe(cache_df, meta)
        if len(exp_df) < 1:
            continue
        mu = exp_df["mu"].to_numpy(dtype=float)
        y_exp = exp_df["value"].to_numpy(dtype=float)
        sigma_stat = exp_df["sigma_stat"].to_numpy(dtype=float)
        sigma_sys = exp_df["sigma_sys"].to_numpy(dtype=float)
        sigma_exp = np.sqrt(sigma_stat ** 2 + sigma_sys ** 2)

        for lib_key, d in per_lib.items():
            y_eval = legval(mu, d["full"])
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
                    "c0":              float(d["c0"]),
                    "L_max":           int(d["l_max_used"]),
                    "sigma_avg_b":     float(d["sigma_avg_b"]),
                    "e_bin_lo":        float(d["e_bin_lo"]),
                    "e_bin_hi":        float(d["e_bin_hi"]),
                    "delta_sigma_rel": float(d["delta_sigma_rel"]),
                })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    libraries = {
        "JEFF":      load_library_lib_c0(JEFF_FILE,       "JEFF-4.0"),
        "JENDL":     load_library_lib_c0(JENDL_FILE,      "JENDL-5"),
        "This_work": load_library_lib_c0(THIS_WORK_FILE,  "This work"),
    }

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
            e_mev, exfor_cache[e_mev], libraries,
        ))

    df = pd.DataFrame(all_rows)
    out_dir = Path(OUTPUT_PARQUET).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-(library, experiment) dense Sigma_eval blocks, mixing MF34
    # (Legendre shape) and MF33 (cross-section magnitude) cross-energy
    # correlations into a single dense N×N block per experiment.
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
        n_exp = df["experiment_id"].nunique()
        print(f"  experiments: {n_exp}")
        # Diagnostics on the magnitude channel (diagonal only).
        for lib in sorted(df["library"].unique()):
            sub = df[df["library"] == lib]
            print(f"  {lib}: median delta_sigma_rel = "
                  f"{sub['delta_sigma_rel'].median():.4f}, "
                  f"median sqrt(Sigma_eval_diag) = "
                  f"{sub['sigma_eval_diag'].median():.3g}")


if __name__ == "__main__":
    main()
