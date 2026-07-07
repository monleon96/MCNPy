"""
Precompute chi^2 data for Fe-56 elastic angular distributions.

Methodology: c_0 anchored to EXFOR Kinney/Smith, sigma_eval = MF34 only.

For each EXFOR datapoint at energy E in [E_MIN_MEV, E_MAX_MEV] (across all
experiments, not just K&S), each library's evaluation is constructed as:

    a_l(E):
        JEFF / JENDL : MF4 a_l interpolated to E and truncated to L=L_MAX.
        This_work    : Legendre coefficients c_l(E) read directly from the
                       pipeline's `nominal_fits.parquet`, interpolated linearly
                       to E and truncated to L=L_MAX. The MF4 a_l would give
                       the same shape; using c_l keeps This_work decoupled
                       from MF4 round-tripping.

    c_0(E):
        JEFF / JENDL : 1-parameter GLS fit at each K&S energy (Kinney for
                       E<2.5 MeV, Smith for 2.5<=E<4 MeV) with library a_l fixed,
                       then linearly interpolated in log E vs log c_0 between
                       K&S energies. Outside K&S coverage the nearest endpoint
                       is used.
        This_work    : c_0 from `nominal_fits.parquet` (the pipeline's own fit
                       to the full EXFOR data, K&S included).

    sigma_eval(E, mu):
        MF34 sandwich on a_l propagated to dsigma/dOmega(mu). No c_0 fit-error
        contribution: the chi^2 in the analysis notebook tests against the
        same data the c_0 fit anchored to, so adding the c_0 fit error to the
        denominator would partially double-count.

Output: one row per (library, experiment, datapoint). The notebook subsets
into all / no_KS / only_KS for the 6 scenarios.

Run: python scripts/precompute_chi2_exfor_c0.py
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
from kika.exfor import read_all_exfor
import kika.exfor as exfor
from kika.exfor.transforms import transform_lab_to_cm
from kika.cov.mf34_covmat import MF34CovMat

from scripts.exfor_utils import build_exfor_cache_from_objects
from scripts.eval_covariance import build_eval_cov_for_groups, save_eval_cov


# ── Configuration ─────────────────────────────────────────────────────────────

MT_NUMBER = 2  # elastic scattering

# ── Library ENDF files (one per evaluation compared in the chi^2 analysis) ──
# This_work uses the pipeline's nominal ENDF and the matching nominal_fits
# parquet that lives next to it (both written by exfor_to_endf_sampling_v2.py
# into OUTPUT_DIR).
THIS_WORK_DIR        = "/share_snc/snc/JuanMonleon/ENDF_samples/new_test_80"
THIS_WORK_FILE       = f"{THIS_WORK_DIR}/26-Fe-56g_nominal_mg.endf"
NOMINAL_FITS_PARQUET = f"{THIS_WORK_DIR}/nominal_fits.parquet"

JEFF_FILE  = "/share_snc/snc/JuanMonleon/jeff40_with_MF4_from_jeff33/26-Fe-56g.txt"
JENDL_FILE = "/share_snc/snc/JuanMonleon/JENDL-5/260560.jendl5"

# ── EXFOR sources ──
EXFOR_DB_PATH = "/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db"
TARGET_ZAIDS  = [26056, 26000]
SUPPLEMENTARY_JSON_FILES = [
    "/share_snc/snc/JuanMonleon/EXFOR/data_v1/27673002.json",
]
EXCLUDE_EXPERIMENTS = ["32246002", "400750022"]  # Tostkii 1957; Morozov 1972 pointer 2 (SPA-fitted duplicate of 400750021)

# K&S subentries used as the c_0 anchor for JEFF/JENDL. Smith capped at 4 MeV
# to match the analysis range (Kinney covers below 2.5).
KS_SUBENTRIES = {
    "Kinney_1976": dict(subentry="10571002", e_lo_mev=0.0, e_hi_mev=2.5),
    "Smith_1980":  dict(subentry="10886002", e_lo_mev=2.5, e_hi_mev=4.0),
}

# Energy range for the analysis. EXFOR datapoints outside this window are
# dropped at the precompute stage.
E_MIN_MEV = 0.85
E_MAX_MEV = 4.0

# Physics
M_PROJ_U = 1.008665
M_TARG_U = 55.93494

# Legendre truncation. In [0.85, 4] MeV the highest order any of the three
# libraries reports is 6, so all evaluations are compared at L=6.
L_MAX = 6

# ── Output ──
OUTPUT_PARQUET = "/share_snc/snc/JuanMonleon/chi2/chi2_data_exfor_c0_80.parquet"


# ── ENDF loading ──────────────────────────────────────────────────────────────

def load_library(filepath: str, label: str) -> Dict:
    """Read MF4 (Legendre grid + coefficients) and MF34 (covariance) from an ENDF file."""
    print(f"Loading {label} from {filepath}")
    endf = read_endf(filepath)
    mt = endf.get_file(4).sections[MT_NUMBER]
    energies_mev = np.asarray(mt.legendre_energies, dtype=float) / 1e6
    coefficients = [np.asarray(c, dtype=float) for c in mt.legendre_coefficients]
    print(f"  MF4: {len(energies_mev)} grid points, "
          f"E=[{energies_mev[0]:.4g}, {energies_mev[-1]:.4g}] MeV, "
          f"max L={max(len(c) for c in coefficients)}")

    try:
        mf34 = MF34CovMat.from_endf(filepath, energy_unit="MeV")
        mf34 = mf34.filter_by_isotope_reaction(26056, MT_NUMBER)
        print(f"  MF34: {mf34.num_matrices} covariance blocks")
    except Exception as e:
        mf34 = None
        print(f"  MF34: not available ({e})")

    return dict(label=label, energies_mev=energies_mev,
                coefficients=coefficients, mf34=mf34)


# ── EXFOR loading ─────────────────────────────────────────────────────────────

def load_exfor(
    db_path: str = EXFOR_DB_PATH,
    target_zaids: list = TARGET_ZAIDS,
    mt: int = MT_NUMBER,
    supplementary_json_files: list = SUPPLEMENTARY_JSON_FILES,
    exclude_experiments: list = EXCLUDE_EXPERIMENTS,
) -> Tuple[Dict, List[float]]:
    """Load all EXFOR Fe elastic data (manifest applied at the cache boundary).

    Defaults read from this module's constants so it can be called with no args
    from this script; callers in other scripts (e.g. precompute_chi2_library_c0.py)
    can pass their own values.
    """
    print("Loading EXFOR data...")
    exfor.configure(db_path=db_path)
    exfor_dict = read_all_exfor(
        target=target_zaids, mt=mt, source="database",
        group_by_energy=False,
        supplementary_json_files=supplementary_json_files,
        exclude_experiments=exclude_experiments,
    )
    objects = list(exfor_dict.values())
    cache, sorted_e = build_exfor_cache_from_objects(
        objects, exclude_experiments=exclude_experiments,
    )
    print(f"  {len(cache)} unique EXFOR energies in [{sorted_e[0]:.4g}, {sorted_e[-1]:.4g}] MeV")
    return cache, sorted_e


def _identify_ks_label(entry: str, subentry: str) -> str:
    """Return 'Kinney_1976' / 'Smith_1980' / 'OTHER' for an experiment."""
    full = f"{entry}{subentry}"
    for label, cfg in KS_SUBENTRIES.items():
        if cfg["subentry"] == full:
            return label
    return "OTHER"


def build_experiment_dataframe(cache_df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """Convert a (cache_df, meta) tuple into the per-row schema used downstream.

    Columns: angle_deg, mu, value, sigma_stat, sigma_sys, sigma_sys_rel,
    sigma_sys_indep_rel, sigma_sys_dep_rel, experiment_id, author, year,
    is_natural, ks_subentry.
    """
    df = cache_df.copy()
    angle_deg = df["angle"].to_numpy(dtype=float)
    value = df["dsig"].to_numpy(dtype=float)
    sigma_stat = df["error_stat"].to_numpy(dtype=float)
    sigma_sys = df["error_sys"].to_numpy(dtype=float)

    # Frame conversion. The library MF4 Legendre coefficients are CM-frame, so
    # experiments reported in the LAB frame (e.g. Gkatis 27673002) must be
    # transformed before mu / dsig are compared. dsig and its absolute
    # uncertainties all share the same LAB->CM Jacobian; relative uncertainties
    # are frame-invariant. Mirrors the fit pipeline (exfor_utils build path).
    frame = str(meta.get("angle_frame", "CM")).upper()
    if frame == "LAB":
        mu_lab = np.cos(np.deg2rad(angle_deg))
        mu, value_cm, sigma_stat = transform_lab_to_cm(
            mu_lab, value, sigma_stat, M_PROJ_U, M_TARG_U
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            jac_scale = np.where(value != 0.0, value_cm / value, 1.0)
        sigma_sys = sigma_sys * jac_scale
        value = value_cm
        angle_deg = np.rad2deg(np.arccos(np.clip(mu, -1.0, 1.0)))
    else:
        mu = np.cos(np.deg2rad(angle_deg))

    n = len(df)
    indep = float(meta.get("sigma_sys_indep_relative", 0.0) or 0.0)
    sigma_sys_rel = np.where(np.abs(value) > 0,
                              sigma_sys / np.maximum(np.abs(value), 1e-300),
                              0.0)
    dep_rel = np.sqrt(np.maximum(sigma_sys_rel ** 2 - indep ** 2, 0.0))
    citation = meta.get("citation") or {}
    authors = citation.get("authors") or [""]
    author = authors[0] if isinstance(authors, list) and authors else ""
    year = citation.get("year", 0) or 0
    reaction = (meta.get("reaction") or "").upper()
    is_natural = "FE-0" in reaction
    entry = meta["entry"]
    sub_no = meta["subentry"]
    return pd.DataFrame({
        "angle_deg": angle_deg,
        "mu": mu,
        "value": value,
        "sigma_stat": sigma_stat,
        "sigma_sys":  sigma_sys,
        "sigma_sys_rel":       sigma_sys_rel,
        "sigma_sys_indep_rel": np.full(n, indep, dtype=float),
        "sigma_sys_dep_rel":   dep_rel,
        "experiment_id": np.full(n, entry + sub_no, dtype=object),
        "author":        np.full(n, author, dtype=object),
        "year":          np.full(n, int(year), dtype=int),
        "is_natural":    np.full(n, bool(is_natural), dtype=bool),
        "ks_subentry":   np.full(n, _identify_ks_label(entry, sub_no), dtype=object),
    })


def collect_ks_fit_energies(exfor_cache: Dict[float, list]) -> List[Tuple[float, str, str]]:
    """List (energy_mev, ks_label, subentry_id) for K&S subentries in [E_MIN, E_MAX].

    These are the energies where the JEFF/JENDL c_0 fits anchor.
    """
    out: List[Tuple[float, str, str]] = []
    for ks_label, cfg in KS_SUBENTRIES.items():
        sub = cfg["subentry"]
        entry, sub_no = sub[:5], sub[5:]
        e_lo, e_hi = cfg["e_lo_mev"], cfg["e_hi_mev"]
        for e_mev, entries in exfor_cache.items():
            if not (E_MIN_MEV <= e_mev <= E_MAX_MEV):
                continue
            if not (e_lo <= e_mev < e_hi):
                continue
            if any(meta["entry"] == entry and meta["subentry"] == sub_no
                   for _df, meta in entries):
                out.append((e_mev, ks_label, sub))
    out.sort(key=lambda t: t[0])
    print(f"K&S c_0 fit grid: {len(out)} energies "
          f"({sum(1 for _, lbl, _ in out if lbl == 'Kinney_1976')} Kinney, "
          f"{sum(1 for _, lbl, _ in out if lbl == 'Smith_1980')} Smith)")
    return out


def get_subentry_dataframe(
    exfor_cache: Dict[float, list],
    energy_mev: float,
    subentry: str,
) -> Optional[pd.DataFrame]:
    """Return the per-row DataFrame for one EXFOR subentry at one energy."""
    entry, sub_no = subentry[:5], subentry[5:]
    for cache_df, meta in exfor_cache.get(energy_mev, []):
        if meta["entry"] == entry and meta["subentry"] == sub_no:
            return build_experiment_dataframe(cache_df, meta)
    return None


# ── This_work nominal-fits loader ─────────────────────────────────────────────

def load_this_work_nominal_fits(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load `nominal_fits.parquet` from the pipeline.

    Returns (energies_mev, c_l_matrix) sorted by energy. NaN coefficients are
    treated as zero (degree N's c_l is NaN whenever the bin's frozen_degree<N).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"This_work nominal-fits parquet not found at {p}. "
            f"Copy it from the pipeline's OUTPUT_DIR/nominal_fits.parquet."
        )
    df = pd.read_parquet(p)
    if "has_data" in df.columns:
        df = df[df["has_data"]].copy()
    c_cols = sorted(
        [c for c in df.columns if c.startswith("c_") and c[2:].isdigit()],
        key=lambda s: int(s.split("_")[1]),
    )
    if not c_cols:
        raise RuntimeError(f"No c_l columns found in {p}.")
    df = df.sort_values("energy_mev").reset_index(drop=True)
    energies = df["energy_mev"].to_numpy(dtype=float)
    cmat = df[c_cols].to_numpy(dtype=float)
    cmat = np.nan_to_num(cmat, nan=0.0)
    print(f"This_work nominal_fits: {len(energies)} bins, max L={cmat.shape[1] - 1}, "
          f"E=[{energies[0]:.4g}, {energies[-1]:.4g}] MeV")
    return energies, cmat


def interp_this_work_coeffs(
    energies: np.ndarray, cmat: np.ndarray, e_target: float,
) -> np.ndarray:
    """Linear interpolation in E of c_l for each l. Clamps at the edges."""
    if e_target <= energies[0]:
        return cmat[0].copy()
    if e_target >= energies[-1]:
        return cmat[-1].copy()
    out = np.empty(cmat.shape[1], dtype=float)
    for l in range(cmat.shape[1]):
        out[l] = float(np.interp(e_target, energies, cmat[:, l]))
    return out


# ── JEFF / JENDL: a_l interpolation and c_0 fit ───────────────────────────────

def interp_a_l_to_energy(library: Dict, e_mev: float, l_max: int) -> np.ndarray:
    """Interpolate a_l (l=1..l_max) from a library's MF4 grid to e_mev (linear)."""
    grid = library["energies_mev"]
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


def fit_c0_from_ks(
    ks_df: pd.DataFrame, a_l: np.ndarray, use_gls: bool = True,
) -> Tuple[float, float]:
    """1-parameter c_0 fit at fixed a_l from one K&S subentry's data at one energy.

    Model: y_j = c_0 * b_j with b_j = 1 + sum_{l>=1} (2l+1) a_l P_l(mu_j).
    GLS uses the rank-2 Woodbury form Σ = D + u uᵀ + v vᵀ with:
        D_jj = σ_stat,j² (σ_dep is NOT on the diagonal)
        u_j  = σ_indep_rel · y_j  (per-experiment normalization)
        v_j  = σ_dep_rel,j · y_j  (per-experiment shape, per-row amplitude)
    Both u and v are correlated within an experiment per the manifest. WLS
    fallback uses diagonal (stat² + sys²) and is only invoked if GLS fails.
    """
    mu = ks_df["mu"].to_numpy(dtype=float)
    y = ks_df["value"].to_numpy(dtype=float)
    sigma_stat = ks_df["sigma_stat"].to_numpy(dtype=float)
    sigma_sys = ks_df["sigma_sys"].to_numpy(dtype=float)

    b = np.ones_like(mu)
    for l in range(1, len(a_l) + 1):
        coef = np.zeros(l + 1); coef[l] = 1.0
        b += (2 * l + 1) * a_l[l - 1] * legval(mu, coef)

    # Floor sigma_stat at 1% of |y| to mirror the pipeline's
    # MIN_STAT_RELATIVE_UNCERTAINTY and prevent Woodbury overflow.
    sigma_floor = 0.01 * np.maximum(np.abs(y), 1e-30)
    sigma_stat = np.maximum(sigma_stat, sigma_floor)

    if use_gls:
        d = np.maximum(sigma_stat ** 2, 1e-300)
        d_inv = 1.0 / d
        u = ks_df["sigma_sys_indep_rel"].to_numpy(dtype=float) * y
        v = ks_df["sigma_sys_dep_rel"].to_numpy(dtype=float) * y
        # 2×2 Woodbury middle matrix M = I₂ + Uᵀ D⁻¹ U
        s_uu = float(np.sum(u * u * d_inv))
        s_vv = float(np.sum(v * v * d_inv))
        s_uv = float(np.sum(u * v * d_inv))
        M_uu = 1.0 + s_uu
        M_vv = 1.0 + s_vv
        det = M_uu * M_vv - s_uv ** 2
        if det <= 0.0:
            return float("nan"), float("nan")
        # Inner products with b and y projected on [u, v]
        bd_u = float(np.sum(b * u * d_inv))
        bd_v = float(np.sum(b * v * d_inv))
        yd_u = float(np.sum(y * u * d_inv))
        yd_v = float(np.sum(y * v * d_inv))
        bdy = float(np.sum(b * y * d_inv))
        bdb = float(np.sum(b * b * d_inv))
        # bᵀ Σ⁻¹ y = bᵀD⁻¹y - z_bᵀ M⁻¹ z_y  with z = [bd_u, bd_v]ᵀ, etc.
        # M⁻¹ = (1/det) [[M_vv, -s_uv], [-s_uv, M_uu]]
        inv_det = 1.0 / det
        bSi_y = bdy - inv_det * (
            M_vv * bd_u * yd_u
            - s_uv * (bd_u * yd_v + bd_v * yd_u)
            + M_uu * bd_v * yd_v
        )
        bSi_b = bdb - inv_det * (
            M_vv * bd_u ** 2
            - 2.0 * s_uv * bd_u * bd_v
            + M_uu * bd_v ** 2
        )
    else:
        sigma2 = sigma_stat ** 2 + sigma_sys ** 2
        sigma2 = np.maximum(sigma2, 1e-300)
        bSi_y = float(np.sum(b * y / sigma2))
        bSi_b = float(np.sum(b * b / sigma2))

    if bSi_b <= 0 or not np.isfinite(bSi_b):
        return float("nan"), float("nan")
    return bSi_y / bSi_b, 1.0 / np.sqrt(bSi_b)


def fit_c0_at_ks_energies(
    library: Dict,
    exfor_cache: Dict[float, list],
    ks_fit_energies: List[Tuple[float, str, str]],
    l_max: int,
) -> List[Tuple[float, float, str]]:
    """Fit c_0 for each K&S energy with library's a_l fixed.

    Returns a sorted list of (energy_mev, c_0, ks_label). Energies where the fit
    fails (c_0 <= 0 or non-finite) are dropped.
    """
    out: List[Tuple[float, float, str]] = []
    for e_mev, ks_label, subentry in ks_fit_energies:
        df = get_subentry_dataframe(exfor_cache, e_mev, subentry)
        if df is None or len(df) < 2:
            continue
        a_l = interp_a_l_to_energy(library, e_mev, l_max)
        c0, _ = fit_c0_from_ks(df, a_l, use_gls=True)
        if not np.isfinite(c0) or c0 <= 0:
            c0, _ = fit_c0_from_ks(df, a_l, use_gls=False)
        if not np.isfinite(c0) or c0 <= 0:
            continue
        out.append((e_mev, c0, ks_label))
    out.sort(key=lambda t: t[0])
    return out


def build_c0_interpolator(
    c0_fits: List[Tuple[float, float, str]],
):
    """Build c_0(E) interpolator from K&S fits.

    Linear in log E vs log c_0 within each leg (Kinney for E<2.5, Smith for
    E>=2.5). Outside coverage on either leg, falls back to the nearest endpoint.
    """
    kinney = [(e, c0) for (e, c0, lbl) in c0_fits if lbl == "Kinney_1976"]
    smith  = [(e, c0) for (e, c0, lbl) in c0_fits if lbl == "Smith_1980"]
    kinney.sort(key=lambda t: t[0])
    smith.sort(key=lambda t: t[0])

    def _interp_leg(seq, e):
        if not seq:
            return float("nan")
        es = np.array([s[0] for s in seq], dtype=float)
        cs = np.array([s[1] for s in seq], dtype=float)
        if e <= es[0]:
            return float(cs[0])
        if e >= es[-1]:
            return float(cs[-1])
        return float(np.exp(np.interp(np.log(e), np.log(es), np.log(cs))))

    def c0_func(e_mev: float) -> float:
        leg = kinney if e_mev < 2.5 else smith
        # If the chosen leg is empty, fall back to the other.
        if not leg:
            leg = smith if e_mev < 2.5 else kinney
        return _interp_leg(leg, e_mev)

    return c0_func


# ── Per-energy row builder ────────────────────────────────────────────────────

def build_rows_at_energy(
    e_mev: float,
    experiments_at_energy: List[Tuple[pd.DataFrame, Dict]],
    libraries: Dict[str, Dict],
    c0_funcs: Dict[str, callable],
    this_work_energies: np.ndarray,
    this_work_cmat: np.ndarray,
) -> List[dict]:
    """Compute one row per (library, experiment, datapoint) at this energy."""
    # Per-library evaluation: a_l, c_0, full Legendre vector. Independent of μ.
    per_lib: Dict[str, dict] = {}
    for lib_key, lib in libraries.items():
        if lib_key == "This_work":
            c_l_full = interp_this_work_coeffs(this_work_energies, this_work_cmat, e_mev)
            c0 = float(c_l_full[0])
            if not np.isfinite(c0) or c0 <= 0:
                return []
            # Truncate the per-bin frozen-degree fit at L_MAX so all libraries
            # are compared at the same Legendre order.
            full = np.zeros(L_MAX + 1, dtype=float)
            n_avail = min(L_MAX + 1, len(c_l_full))
            full[:n_avail] = c_l_full[:n_avail]
            a_l = np.array(
                [full[l] / (c0 * (2 * l + 1)) for l in range(1, L_MAX + 1)],
                dtype=float,
            )
        else:
            a_l = interp_a_l_to_energy(lib, e_mev, L_MAX)
            c0 = c0_funcs[lib_key](e_mev)
            if not np.isfinite(c0) or c0 <= 0:
                return []
            full = np.empty(L_MAX + 1, dtype=float)
            full[0] = c0
            for l in range(1, L_MAX + 1):
                full[l] = c0 * (2 * l + 1) * a_l[l - 1]
        l_max_used = L_MAX

        per_lib[lib_key] = dict(c0=c0, a_l=a_l, full=full,
                                l_max_used=l_max_used, mf34=lib.get("mf34"))

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
                })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    libraries = {
        "JEFF":      load_library(JEFF_FILE,       "JEFF-4.0"),
        "JENDL":     load_library(JENDL_FILE,      "JENDL-5"),
        "This_work": load_library(THIS_WORK_FILE,  "This work"),
    }

    exfor_cache, _sorted_e = load_exfor()

    # Step 1 — fit JEFF/JENDL c_0 at K&S energies, build interpolators.
    ks_fit_energies = collect_ks_fit_energies(exfor_cache)
    if not ks_fit_energies:
        raise RuntimeError("No K&S energies available in [E_MIN, E_MAX] for c_0 fitting.")

    c0_funcs: Dict[str, callable] = {}
    for lib_key in ("JEFF", "JENDL"):
        fits = fit_c0_at_ks_energies(
            libraries[lib_key], exfor_cache, ks_fit_energies, L_MAX,
        )
        n_kin = sum(1 for _, _, lbl in fits if lbl == "Kinney_1976")
        n_smi = sum(1 for _, _, lbl in fits if lbl == "Smith_1980")
        print(f"  {lib_key}: c_0 fits at {len(fits)} K&S energies "
              f"({n_kin} Kinney + {n_smi} Smith)")
        c0_funcs[lib_key] = build_c0_interpolator(fits)

    # Step 2 — load This_work nominal fits.
    tw_energies, tw_cmat = load_this_work_nominal_fits(NOMINAL_FITS_PARQUET)

    # Step 3 — iterate every EXFOR energy in range, write rows for every
    # experiment at that energy.
    energies_in_range = sorted(
        e for e in exfor_cache if E_MIN_MEV <= e <= E_MAX_MEV
    )
    print(f"Iterating {len(energies_in_range)} EXFOR energies in "
          f"[{E_MIN_MEV}, {E_MAX_MEV}] MeV")

    all_rows: List[dict] = []
    for e_mev in energies_in_range:
        all_rows.extend(build_rows_at_energy(
            e_mev, exfor_cache[e_mev], libraries, c0_funcs,
            tw_energies, tw_cmat,
        ))

    df = pd.DataFrame(all_rows)
    out_dir = Path(OUTPUT_PARQUET).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-(library, experiment) dense Sigma_eval blocks. The library-c0
    # MF33 fields are not present here (no `mf33_grid_ev`/`mf33_rel_cov`),
    # so the MF33 contribution is identically zero — only MF34 enters.
    def _a_l_lookup(lib_key: str, lib: dict, e_mev: float) -> np.ndarray:
        if lib_key == "This_work":
            c_l_full = interp_this_work_coeffs(tw_energies, tw_cmat, e_mev)
            c0 = float(c_l_full[0])
            if not np.isfinite(c0) or c0 <= 0:
                return np.zeros(L_MAX, dtype=float)
            full = np.zeros(L_MAX + 1, dtype=float)
            n_avail = min(L_MAX + 1, len(c_l_full))
            full[:n_avail] = c_l_full[:n_avail]
            return np.array(
                [full[l] / (c0 * (2 * l + 1)) for l in range(1, L_MAX + 1)],
                dtype=float,
            )
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


if __name__ == "__main__":
    main()
