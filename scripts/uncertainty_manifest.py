"""
EXFOR uncertainty manifest resolver.

Reads ``uncertainty_manifest.yaml`` and computes per-point sigma_stat and
sigma_sys for each EXFOR dataset, augmenting the X4Pro database loader with
prose-documented ERR-ANALYS information that isn't surfaced as data-table
columns.

The manifest schema is documented in the YAML header.

Pipeline contract
-----------------
For every dataset, the resolver returns:

    sigma_stat_b_sr            : np.ndarray, per-point uncorrelated noise (abs b/sr)
    sigma_sys_b_sr             : np.ndarray, per-point correlated systematic (abs b/sr)
    sigma_sys_scalar_relative  : float, representative scalar sigma_sys (fraction)
    flag                       : 'curated' | 'uncurated' | 'excluded'
    prose                      : str | None — verbatim ERR-ANALYS quote

The pipeline uses ``sigma_stat`` for GLS Legendre fit weights / chi^2 (treated
as uncorrelated noise) and the per-experiment ``sigma_sys_scalar_relative``
for MC perturbation and Kish rho in GLS-ESS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for the uncertainty manifest. Install with `pip install pyyaml`."
    ) from e


import os

# Default manifest location (alongside the EXFOR data on the shared volume).
# Override with the ``KIKA_UNCERTAINTY_MANIFEST_PATH`` environment variable.
_DEFAULT_MANIFEST_PATH = "/share_snc/snc/JuanMonleon/EXFOR/uncertainty_manifest.yaml"
_MANIFEST_PATH = Path(os.environ.get(
    "KIKA_UNCERTAINTY_MANIFEST_PATH", _DEFAULT_MANIFEST_PATH
))
_manifest_cache: Optional[dict] = None


# =============================================================================
# Manifest IO
# =============================================================================


def load_manifest(path: Optional[str] = None, force_reload: bool = False) -> dict:
    """Load the YAML manifest. Cached after first call."""
    global _manifest_cache
    if _manifest_cache is None or force_reload or path is not None:
        p = Path(path) if path else _MANIFEST_PATH
        with open(p) as f:
            data = yaml.safe_load(f)
        if path is None:
            _manifest_cache = data
        return data
    return _manifest_cache


# =============================================================================
# Spec helpers
# =============================================================================

_PERCENT_UNITS = {"PER-CENT", "PC", "%", "PERCENT"}


def _is_percent(unit: str) -> bool:
    return unit.upper() in _PERCENT_UNITS


def _component_to_b_sr(component: dict, values_b_sr: np.ndarray) -> np.ndarray:
    """Convert a uncertainty_components entry to per-point absolute b/sr."""
    unit = component.get("unit", "").upper()
    if component.get("kind") == "scalar":
        v = float(component["value"])
        if _is_percent(unit):
            return np.abs(values_b_sr) * (v / 100.0)
        if unit == "MB/SR":
            return np.full_like(values_b_sr, v / 1000.0)
        if unit == "B/SR":
            return np.full_like(values_b_sr, v)
        return np.full_like(values_b_sr, v)  # unknown → assume absolute b/sr
    # per-point
    arr = np.asarray(component["values"], dtype=float)
    if _is_percent(unit):
        return np.abs(values_b_sr) * (arr / 100.0)
    if unit == "MB/SR":
        return arr / 1000.0
    if unit == "B/SR":
        return arr
    return arr


def _find_component(components: List[dict], header: str) -> Optional[dict]:
    for c in components:
        if c.get("header") == header:
            return c
    return None


def _eval_piecewise_E(
    spec: List[dict], energies_mev: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate piecewise-E spec.

    Returns
    -------
    pcts : np.ndarray
        Per-point relative percent within covered segments (0 elsewhere).
    covered : np.ndarray (bool)
        True where some segment covered the point; False where the point
        fell outside the spec (caller should apply a fallback).
    """
    pcts = np.zeros_like(energies_mev, dtype=float)
    covered = np.zeros_like(energies_mev, dtype=bool)
    for seg in spec:
        e_min = float(seg.get("E_MeV_min", -np.inf))
        e_max = float(seg.get("E_MeV_max", np.inf))
        mask = (energies_mev >= e_min) & (energies_mev <= e_max)
        if not np.any(mask):
            continue
        if "value_pct" in seg:
            pcts[mask] = float(seg["value_pct"])
        elif "value_pct_lo" in seg and "value_pct_hi" in seg:
            lo = float(seg["value_pct_lo"])
            hi = float(seg["value_pct_hi"])
            t = (energies_mev[mask] - e_min) / max(e_max - e_min, 1e-30)
            pcts[mask] = lo + t * (hi - lo)
        covered |= mask
    return pcts, covered


# =============================================================================
# Spec resolvers
# =============================================================================


def _best_available_column(
    components: List[dict], values_b_sr: np.ndarray
) -> np.ndarray:
    """Default fallback: try DATA-ERR > ERR-T > ERR-S."""
    for header in ("DATA-ERR", "ERR-T", "ERR-S"):
        c = _find_component(components, header)
        if c is not None and (c.get("kind") == "scalar" or c.get("values")):
            return _component_to_b_sr(c, values_b_sr)
    return np.zeros_like(values_b_sr)


def _resolve_stat_spec(
    spec: dict,
    components: List[dict],
    values_b_sr: np.ndarray,
    energies_mev: np.ndarray,
    defaults: dict,
) -> np.ndarray:
    """Compute per-point sigma_stat in absolute b/sr."""
    source = spec.get("source", "default") if spec else "default"

    if source == "default":
        return _resolve_stat_spec(
            defaults.get("stat", {"source": "column", "column": "best_available"}),
            components, values_b_sr, energies_mev, defaults={}
        )

    if source == "column":
        col = spec.get("column", "best_available")
        if col == "best_available":
            return _best_available_column(components, values_b_sr)
        c = _find_component(components, col)
        if c is None:
            return np.zeros_like(values_b_sr)
        return _component_to_b_sr(c, values_b_sr)

    if source == "scalar":
        if "value_pct" in spec:
            return np.abs(values_b_sr) * float(spec["value_pct"]) / 100.0
        if "value_abs_b_per_sr" in spec:
            return np.full_like(values_b_sr, float(spec["value_abs_b_per_sr"]))
        return np.zeros_like(values_b_sr)

    if source == "combine":
        total_sq = np.zeros_like(values_b_sr)
        for part in spec.get("parts", []):
            arr = _resolve_stat_spec(part, components, values_b_sr, energies_mev, defaults)
            total_sq = total_sq + arr ** 2
        return np.sqrt(total_sq)

    if source == "rule":
        # Currently only the Cox 10%/30mb correction rule is implemented.
        total_expr = spec.get("total_expr", "")
        if "max(0.10 * value, 0.030" in total_expr:
            return np.maximum(0.10 * np.abs(values_b_sr), 0.030)
        return np.zeros_like(values_b_sr)

    return np.zeros_like(values_b_sr)


def _resolve_sys_spec(
    spec: dict,
    values_b_sr: np.ndarray,
    energies_mev: np.ndarray,
    defaults: dict,
) -> np.ndarray:
    """Compute per-point sigma_sys in absolute b/sr (combined indep + dep)."""
    indep_rel, dep_rel = _split_sys_spec(spec, values_b_sr, energies_mev, defaults)
    total_rel_sq = indep_rel ** 2 + dep_rel ** 2
    return np.abs(values_b_sr) * np.sqrt(total_rel_sq)


def _split_sys_spec(
    spec: dict,
    values_b_sr: np.ndarray,
    energies_mev: np.ndarray,
    defaults: dict,
) -> Tuple[float, np.ndarray]:
    """
    Split a sys-spec into energy-independent (scalar) and energy-dependent
    (piecewise_E) contributions, both expressed as relative fractions.

    Returns
    -------
    indep_relative : float
        Scalar relative-fraction (constant across the experiment), from all
        ``source: scalar`` components combined in quadrature.
    dep_relative_per_point : np.ndarray
        Per-point relative fraction from all ``source: piecewise_E``
        components (zero where no E-dependent component exists).

    The total per-point sigma_sys (relative) is sqrt(indep² + dep[i]²).
    """
    source = spec.get("source", "default") if spec else "default"
    zeros = np.zeros_like(values_b_sr)

    if source == "default":
        return _split_sys_spec(
            defaults.get("sys", {"source": "scalar", "value_pct": 5.0}),
            values_b_sr, energies_mev, defaults={}
        )

    if source == "scalar":
        v = float(spec.get("value_pct", 0.0)) / 100.0
        return v, zeros

    if source == "piecewise_E":
        pcts, covered = _eval_piecewise_E(spec.get("spec", []), energies_mev)
        dep = pcts / 100.0
        # For points outside the piecewise spec, fall back to the defaults sys
        # (treat the fallback as energy-independent for those points).
        if not np.all(covered) and defaults:
            fb_indep, _ = _split_sys_spec(
                defaults.get("sys", {"source": "scalar", "value_pct": 5.0}),
                values_b_sr, energies_mev, defaults={}
            )
            dep = np.where(covered, dep, fb_indep)
        return 0.0, dep

    if source == "components":
        indep_sq = 0.0
        dep_sq = zeros.copy()
        for part in spec.get("parts", []):
            i_part, d_part = _split_sys_spec(part, values_b_sr, energies_mev, defaults)
            indep_sq += i_part ** 2
            dep_sq = dep_sq + d_part ** 2
        return float(np.sqrt(indep_sq)), np.sqrt(dep_sq)

    return 0.0, zeros


# =============================================================================
# Public API
# =============================================================================


def resolve_for_dataset(
    dataset_id: str,
    uncertainty_components: List[dict],
    energies_mev: np.ndarray,
    values_b_sr: np.ndarray,
    manifest: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Resolve sigma_stat and sigma_sys for a single EXFOR dataset.

    Parameters
    ----------
    dataset_id : str
        EXFOR DatasetID (X4Pro encoding, e.g. "20743002").
    uncertainty_components : list of dict
        ``uncertainty_components`` from ``_parse_x4data_json`` — one entry per
        ``cvar=='dy'`` column. Each entry has keys ``header``, ``kind``,
        ``unit``, plus either ``values`` (per_point) or ``value`` (scalar).
    energies_mev : np.ndarray
        Per-point energies (MeV).
    values_b_sr : np.ndarray
        Per-point cross sections (b/sr).
    manifest : dict, optional
        Pre-loaded manifest. If None, loads the default manifest (cached).

    Returns
    -------
    dict with keys:
        sigma_stat_b_sr            : np.ndarray (N,) per-point uncorrelated noise
        sigma_sys_b_sr             : np.ndarray (N,) per-point correlated systematic
        sigma_sys_scalar_relative  : float — representative scalar (fraction, not %)
        flag                       : str — 'curated' | 'uncurated' | 'excluded' | 'default'
        prose                      : str | None
    """
    m = manifest if manifest is not None else load_manifest()
    defaults = m.get("defaults", {})
    entry = m.get("datasets", {}).get(dataset_id, {})

    energies_mev = np.asarray(energies_mev, dtype=float)
    values_b_sr = np.asarray(values_b_sr, dtype=float)

    stat_spec = entry.get("stat", {"source": "default"})
    sys_spec = entry.get("sys", {"source": "default"})
    flag = entry.get("flag", "default" if not entry else "uncurated")

    # Split sys into energy-independent (scalar) and energy-dependent (piecewise_E)
    # parts. Pipeline can then draw two factors: one per-experiment for indep,
    # one per-(experiment, energy) for dep.
    indep_rel, dep_rel = _split_sys_spec(sys_spec, values_b_sr, energies_mev, defaults)
    sigma_sys_indep_b_sr = np.abs(values_b_sr) * indep_rel
    sigma_sys_dep_b_sr = np.abs(values_b_sr) * dep_rel
    sigma_sys = np.sqrt(sigma_sys_indep_b_sr ** 2 + sigma_sys_dep_b_sr ** 2)

    sigma_stat = _resolve_stat_spec(
        stat_spec, uncertainty_components, values_b_sr, energies_mev, defaults
    )

    # When prose says the column is a TOTAL with sys "included", subtract sys
    # from stat in quadrature to recover the pure-stat (uncorrelated) component.
    # This avoids double-counting σ_sys when the per-experiment MC factor is
    # layered on top — per-point MC variance then reproduces σ_total² exactly.
    # Floored at 0 when σ_sys > σ_total (signals a manifest/prose mismatch
    # that should be fixed by adjusting the sys spec, not papered over).
    if isinstance(stat_spec, dict) and stat_spec.get("derive_stat_only"):
        sigma_stat = np.sqrt(np.maximum(sigma_stat ** 2 - sigma_sys ** 2, 0.0))

    # Per-experiment representative scalar sigma_sys (relative, for diagnostics)
    nz = np.abs(values_b_sr) > 0
    if np.any(nz):
        rel = np.zeros_like(values_b_sr)
        rel[nz] = sigma_sys[nz] / np.abs(values_b_sr[nz])
        sigma_sys_scalar = float(np.median(rel[rel > 0])) if np.any(rel > 0) else 0.0
    else:
        sigma_sys_scalar = 0.0

    return {
        "sigma_stat_b_sr": sigma_stat,
        "sigma_sys_b_sr": sigma_sys,                    # combined per-point (abs)
        "sigma_sys_indep_relative": float(indep_rel),   # scalar — energy-independent
        "sigma_sys_dep_relative_per_point": dep_rel,    # per-point energy-dependent
        "sigma_sys_scalar_relative": sigma_sys_scalar,  # representative total
        "flag": flag,
        "prose": entry.get("prose") if entry else None,
    }


def get_dataset_flag(dataset_id: str, manifest: Optional[dict] = None) -> str:
    """Quick lookup: 'curated' | 'uncurated' | 'excluded' | 'default'."""
    m = manifest if manifest is not None else load_manifest()
    entry = m.get("datasets", {}).get(dataset_id)
    if entry is None:
        return "default"
    return entry.get("flag", "uncurated")


def apply_manifest_to_exfor(
    exfor,
    uncertainty_components: Optional[List[dict]] = None,
    manifest: Optional[dict] = None,
) -> None:
    """
    Apply the uncertainty manifest to an ExforAngularDistribution **in place**.

    Resolves sigma_stat and sigma_sys using the manifest, then:
      - Overwrites each ``data`` point's ``uncertainty_stat`` (per-point uncorrelated)
        and ``uncertainty_sys`` (per-point correlated)
      - Sets ``exfor.sigma_sys_scalar_relative`` (per-experiment representative)
      - Sets ``exfor.uncertainty_manifest_flag``

    Parameters
    ----------
    exfor : ExforAngularDistribution
        Must already have ``_data_blocks`` populated.
    uncertainty_components : list of dict, optional
        Raw EXFOR ``cvar=='dy'`` columns (from ``_parse_x4data_json``). When
        provided (database loader path), the manifest can resolve ``column: ERR-S``
        / ``column: ERR-T`` references. When None (JSON loader path), a synthetic
        ``DATA-ERR`` component is built from the per-point ``uncertainty_stat``
        already in ``_data_blocks`` so ``column: DATA-ERR`` and
        ``column: best_available`` still work.
    manifest : dict, optional
        Pre-loaded manifest. If None, the cached default is used.
    """
    dataset_id = f"{exfor.entry}{exfor.subentry}"

    # Gather per-point energies, values, and existing per-point uncertainties.
    energies: List[float] = []
    values: List[float] = []
    existing_stat: List[float] = []
    point_refs: List[dict] = []
    for blk in exfor._data_blocks:
        e_val = blk.get("value")
        if e_val is None:
            e_val = blk.get("E", 0.0)
        for pt in blk.get("data", []):
            energies.append(float(e_val))
            values.append(float(pt.get("cross_section", pt.get("result", 0.0)) or 0.0))
            existing_stat.append(float(pt.get("uncertainty_stat", pt.get("error_stat", 0.0)) or 0.0))
            point_refs.append(pt)

    if not energies:
        exfor.sigma_sys_scalar_relative = 0.0
        exfor.uncertainty_manifest_flag = "default"
        return

    energies_arr = np.asarray(energies, dtype=float)
    values_arr = np.asarray(values, dtype=float)

    # Build synthetic components for the JSON path (no raw column structure).
    if uncertainty_components is None:
        uncertainty_components = [{
            "header": "DATA-ERR",
            "kind": "per_point",
            "values": existing_stat,
            "unit": "B/SR",
        }]

    res = resolve_for_dataset(
        dataset_id=dataset_id,
        uncertainty_components=uncertainty_components,
        energies_mev=energies_arr,
        values_b_sr=values_arr,
        manifest=manifest,
    )

    # Write back per-point sigma_stat / sigma_sys into the data points.
    # sigma_sys is the combined indep⊕dep total; sigma_sys_dep_rel is the
    # energy-dependent portion alone (relative fraction) — pipeline uses it
    # to draw the per-(experiment, energy) MC factor without recomputing.
    sigma_stat = res["sigma_stat_b_sr"]
    sigma_sys = res["sigma_sys_b_sr"]
    sigma_sys_dep_rel = res["sigma_sys_dep_relative_per_point"]
    for i, pt in enumerate(point_refs):
        pt["uncertainty_stat"] = float(sigma_stat[i])
        pt["uncertainty_sys"] = float(sigma_sys[i])
        pt["uncertainty_sys_dep_rel"] = float(sigma_sys_dep_rel[i])

    exfor.sigma_sys_scalar_relative = float(res["sigma_sys_scalar_relative"])
    exfor.sigma_sys_indep_relative = float(res["sigma_sys_indep_relative"])
    exfor.uncertainty_manifest_flag = res["flag"]
