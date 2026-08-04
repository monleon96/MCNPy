"""
EXFOR uncertainty manifest resolver.

Reads the EXFOR uncertainty manifest YAML (see ``_DEFAULT_MANIFEST_PATH``
below for which one, and why there is only one) and computes per-point sigma_stat and
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
#
# This is the PRODUCTION manifest: Cierjacks ERR-T routed to a diagonal residual
# through a ``sys_dep`` block. It is what run 77 (the production MF4/MF34) and
# run 81 were built with, and it is now the default so a run is reproducible
# from the repo without remembering to export anything.
#
# It is now the ONLY manifest in the EXFOR directory. The default used to be
# ``uncertainty_manifest.yaml``, which applies the legacy ``kind:
# total_minus_stat`` rank-1 treatment to Cierjacks instead. Selecting it by
# accident silently changes the fits — median n_eff 3.84 vs 2.65 and a different
# frozen_degree in 945 of 1738 bins — so on 2026-07-27 it was moved out of the
# way, to
#     EXFOR/_manifest_archive/uncertainty_manifest_legacy_rank1_cierjacks.yaml
# (sha256 9d32823b...). It is kept only so runs 68-75, whose ``run_metadata.json``
# records that hash, stay reproducible; nothing should select it for new work.
# The env var still overrides, if you ever need to point at it deliberately.
_DEFAULT_MANIFEST_PATH = (
    "/share_snc/snc/JuanMonleon/EXFOR/uncertainty_manifest_diagonal_cierjacks.yaml"
)

def manifest_path() -> Path:
    """The manifest this process should read, resolved *now*.

    Deliberately a function. This used to be a module-level constant assigned
    at import, which made ``KIKA_UNCERTAINTY_MANIFEST_PATH`` a cluster-only
    toggle: the sbatch script exports it before python starts, so it worked
    there, while setting it in-process after any module had imported this one
    silently did nothing.
    """
    return Path(os.environ.get(
        "KIKA_UNCERTAINTY_MANIFEST_PATH", _DEFAULT_MANIFEST_PATH
    ))


# Parsed manifests, keyed on the path they came from. Keying on the path is
# what lets the cache survive a change of manifest mid-process instead of
# serving the first one read to every later caller.
_manifest_cache: Dict[Path, dict] = {}

# Minimum statistical (uncorrelated) uncertainty as fraction of |value|.
# Applied at the resolver when ``derive_stat_only`` decomposes a loaded total
# into σ_stat ⊕ σ_sys: replaces the previous floor at 0, so a row never
# leaves the resolver with σ_stat = 0. Matches the per-point floor already
# applied later during the GLS fit (``MIN_STAT_RELATIVE_UNCERTAINTY`` in
# ``exfor_to_endf_sampling_v2.py``); keeping the same value at both stages
# means the published parquet carries a σ_stat consistent with what the
# pipeline actually fitted with.
SIGMA_STAT_MIN_REL = 0.01


# =============================================================================
# Manifest IO
# =============================================================================


def load_manifest(path: Optional[str] = None, force_reload: bool = False) -> dict:
    """Load the YAML manifest. Cached per resolved path.

    Parameters
    ----------
    path : str, optional
        Read this file rather than the one ``manifest_path()`` resolves.
    force_reload : bool
        Re-read from disk even if this path is already cached.
    """
    p = Path(path) if path else manifest_path()
    if force_reload or p not in _manifest_cache:
        with open(p) as f:
            _manifest_cache[p] = yaml.safe_load(f)
    return _manifest_cache[p]


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


def _piecewise_E_coverage(
    sys_spec: dict, energies_mev: np.ndarray,
) -> np.ndarray:
    """Per-row coverage mask for a top-level piecewise_E sys-spec.

    Returns a boolean array of the same length as ``energies_mev``: True at
    rows whose energy falls inside at least one segment of the piecewise_E
    spec, False otherwise. For nested specs (``source: components``) the
    coverage is True wherever any piecewise_E part covers the row. Returns
    all-False for non-piecewise specs.

    Used by ``resolve_for_dataset`` to apply ``kind: total_minus_stat`` only
    on rows where the piecewise spec actually carries the σ_total amplitude
    (outside-coverage rows fall back to the defaults σ_sys, which is already
    σ_sys not σ_total, and must not be reduced).
    """
    energies_mev = np.asarray(energies_mev, dtype=float)
    if not isinstance(sys_spec, dict):
        return np.zeros_like(energies_mev, dtype=bool)
    src = sys_spec.get("source")
    if src == "piecewise_E":
        _pcts, covered = _eval_piecewise_E(sys_spec.get("spec", []), energies_mev)
        return covered
    if src == "components":
        out = np.zeros_like(energies_mev, dtype=bool)
        for part in sys_spec.get("parts", []):
            out |= _piecewise_E_coverage(part, energies_mev)
        return out
    return np.zeros_like(energies_mev, dtype=bool)


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
) -> Tuple[np.ndarray, Optional[str]]:
    """Default fallback. Returns (sigma_per_point_b_sr, picked_header).

    Search order:
      1. ``DATA-ERR1`` (+ ``DATA-ERR2`` in quadrature if both present) — split
         components reported by some labs (e.g. EXFOR entries 10384, 10633).
      2. ``DATA-ERR``  — most common single column.
      3. ``ERR-T``     — explicitly the total uncertainty.
      4. ``ERR-S``     — explicitly the statistical uncertainty.

    ``picked_header`` is the literal name returned (or
    ``"DATA-ERR1+DATA-ERR2"`` when both halves are combined). ``None`` means
    no usable column was found.
    """
    c1 = _find_component(components, "DATA-ERR1")
    if c1 is not None and (c1.get("kind") == "scalar" or c1.get("values")):
        v1 = _component_to_b_sr(c1, values_b_sr)
        c2 = _find_component(components, "DATA-ERR2")
        if c2 is not None and (c2.get("kind") == "scalar" or c2.get("values")):
            v2 = _component_to_b_sr(c2, values_b_sr)
            return np.sqrt(v1 ** 2 + v2 ** 2), "DATA-ERR1+DATA-ERR2"
        return v1, "DATA-ERR1"
    for header in ("DATA-ERR", "ERR-T", "ERR-S"):
        c = _find_component(components, header)
        if c is not None and (c.get("kind") == "scalar" or c.get("values")):
            return _component_to_b_sr(c, values_b_sr), header
    return np.zeros_like(values_b_sr), None


# Headers whose column value is interpreted as a TOTAL uncertainty (stat ⊕
# sys). When ``best_available`` returns one of these and the stat-spec is the
# default fallback (i.e. the dataset is uncurated or has no manifest entry),
# the resolver applies the same ``derive_stat_only`` decomposition that
# explicit curated entries with ``semantics: total_per_point`` use:
# σ_stat = √(σ_total² − σ_sys²), σ_sys capped at σ_total. Documented in
# ``scripts/uncertainty_manifest_methodology.md`` §3.2.
_DEFAULT_TOTAL_HEADERS = {
    "DATA-ERR", "ERR-T", "DATA-ERR1", "DATA-ERR1+DATA-ERR2",
}

# Stat-spec labels that document the resolved σ_stat as a TOTAL uncertainty
# rather than a pure statistical one. When present, the resolver treats the
# spec as if ``derive_stat_only: true`` had been set explicitly: σ_stat is
# decomposed into σ_stat ⊕ σ_sys (with σ_sys capped at σ_total). This lets
# uncurated entries that document the column semantics in prose (e.g.
# Jacquot 20379003, ``kind: total_per_point``) get the correct treatment
# without every YAML author having to remember to set the boolean flag.
_TOTAL_LABEL_VALUES = {"total_per_point"}


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
            sigma, _picked = _best_available_column(components, values_b_sr)
            return sigma
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
        # The YAML's stat_expr is sqrt(max(total_expr^2 - (0.10*value)^2, 0));
        # i.e. the residual after removing the correlated 10% sys (which is
        # added back through _resolve_sys_spec). Returning the total here
        # would double-count the 10% sys.
        total_expr = spec.get("total_expr", "")
        if "max(0.10 * value, 0.030" in total_expr:
            total = np.maximum(0.10 * np.abs(values_b_sr), 0.030)
            sys = 0.10 * np.abs(values_b_sr)
            return np.sqrt(np.maximum(total ** 2 - sys ** 2, 0.0))
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
        per_point = pcts / 100.0
        # For points outside the piecewise spec, fall back to the defaults sys
        # (treat the fallback as energy-independent for those points).
        if not np.all(covered) and defaults:
            fb_indep, _ = _split_sys_spec(
                defaults.get("sys", {"source": "scalar", "value_pct": 5.0}),
                values_b_sr, energies_mev, defaults={}
            )
            per_point = np.where(covered, per_point, fb_indep)
        # Routing depends on whether this systematic is correlated *within*
        # an experiment. correlated=true (the natural default for a
        # piecewise calibration error like Cierjacks's ERR-T) means the
        # whole experiment shifts together — one shared draw per experiment
        # per MC sample, with per-point amplitude varying along the
        # piecewise spec. We surface that via dep (per-point), but the
        # downstream worker uses the per-row total (sigma_sys_relative)
        # plus a single shared z per experiment to apply it correctly.
        # correlated=false (rare; not used by any current manifest entry)
        # would mean each (experiment, energy) is genuinely independent —
        # the legacy "dep" semantics — and in that case the worker would
        # need a per-(experiment, energy) draw instead. The per-row total
        # column (sigma_sys_relative) is identical in both cases; only the
        # routing semantics for downstream consumers change.
        return 0.0, per_point

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

    # ── Default-spec column sniffing ─────────────────────────────────────────
    # When the dataset is uncurated (or absent from the manifest entirely) and
    # the stat-spec falls through to ``best_available``, peek at *which* column
    # the fallback would pick. Headers in ``_DEFAULT_TOTAL_HEADERS``
    # (DATA-ERR, ERR-T, DATA-ERR1, DATA-ERR1+DATA-ERR2) are conventionally
    # totals — the resolver promotes them to ``derive_stat_only`` so that the
    # default 5% σ_sys is *carved out* of the column rather than added on top
    # (which would silently inflate the lab's reported total by ~12% in
    # quadrature). Headers explicitly named ``ERR-S`` are statistical-only and
    # keep the pre-existing additive treatment (column = σ_stat, +5% σ_sys).
    # Documented in scripts/uncertainty_manifest_methodology.md §3.2.
    default_promoted_total = False
    label_promoted_total = False
    if (
        isinstance(stat_spec, dict)
        and stat_spec.get("source", "default") == "default"
    ):
        _picked = _best_available_column(uncertainty_components, values_b_sr)[1]
        if _picked in _DEFAULT_TOTAL_HEADERS:
            stat_spec = {
                "source": "column",
                "column": "best_available",
                "semantics": "total_per_point",
                "derive_stat_only": True,
            }
            default_promoted_total = True

    # ── Label-based total promotion ──────────────────────────────────────────
    # YAML entries can document the column semantics with either
    # ``semantics: total_per_point`` or ``kind: total_per_point``. Without
    # this promotion, only ``derive_stat_only: true`` actually triggered the
    # decomposition — labels alone were silently inert and the default 5% sys
    # was added on top of a lab-reported total. Promotion preserves the
    # documented intent for entries that named themselves as totals without
    # remembering the boolean flag (e.g. Jacquot 20379003).
    if isinstance(stat_spec, dict) and not stat_spec.get("derive_stat_only"):
        sem = stat_spec.get("semantics")
        knd = stat_spec.get("kind")
        if sem in _TOTAL_LABEL_VALUES or knd in _TOTAL_LABEL_VALUES:
            stat_spec = dict(stat_spec)        # copy: don't mutate cached manifest
            stat_spec["derive_stat_only"] = True
            label_promoted_total = True

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

    # ── kind: total_minus_stat ───────────────────────────────────────────────
    # When a sys-spec is labeled ``kind: total_minus_stat`` the resolved
    # σ_sys here actually represents σ_total (not σ_sys), and σ_sys must be
    # recovered by subtracting σ_stat in quadrature per row:
    #     σ_sys_per_row = √(max(σ_total² − σ_stat², 0))
    # The label is currently used by Cierjacks 20743002, where the piecewise
    # ERR-T spec gives 7-12% σ_total and ERR-S column gives σ_stat. Without
    # the subtraction the per-point σ_sys is overestimated, which inflates
    # GLS weights (under-fits the data) and pulls the rank-1 Mahalanobis
    # variance away from the documented model. Points outside the piecewise
    # coverage fall back to the defaults σ_sys (see ``_split_sys_spec``),
    # which is interpreted directly as σ_sys (not σ_total), so the
    # subtraction is applied row-wise only where the dep amplitude came from
    # the labeled piecewise_E spec — tracked via the ``covered`` mask.
    sys_kind = sys_spec.get("kind") if isinstance(sys_spec, dict) else None
    if sys_kind == "total_minus_stat":
        covered = _piecewise_E_coverage(sys_spec, energies_mev)
        # On covered rows σ_total = √(indep² + dep²) · |y| (indep is 0 in
        # practice for total_minus_stat entries, but the formula stays
        # general). Subtract σ_stat² in quadrature.
        sigma_total_sq = sigma_sys_indep_b_sr ** 2 + sigma_sys_dep_b_sr ** 2
        sigma_sys_new_sq = np.where(
            covered,
            np.maximum(sigma_total_sq - sigma_stat ** 2, 0.0),
            sigma_total_sq,  # uncovered: defaults σ_sys is already σ_sys, leave it
        )
        # Re-split: indep stays as-is (any scalar component in `components`
        # is unaffected by the subtraction since it's not the total
        # carrier); dep absorbs all of the change in covered rows.
        sigma_sys = np.sqrt(sigma_sys_new_sq)
        sigma_sys_dep_b_sr = np.sqrt(np.maximum(
            sigma_sys ** 2 - sigma_sys_indep_b_sr ** 2, 0.0
        ))
        nz = np.abs(values_b_sr) > 0
        dep_rel = np.zeros_like(values_b_sr)
        with np.errstate(divide="ignore", invalid="ignore"):
            dep_rel[nz] = sigma_sys_dep_b_sr[nz] / np.abs(values_b_sr[nz])

    # ── derive_stat_only: σ_total preserved, σ_sys capped at σ_total ─────────
    # When prose (or the auto-promotion above) says the loaded column is a
    # TOTAL containing the named sys components, we need to recover σ_stat
    # without inflating the lab's reported total. The interpretation adopted
    # here is "trust the reported total":
    #
    #   σ_total_per_point = the loaded column (= ``sigma_stat`` at this point
    #                       in the function, before the next two lines run)
    #   σ_sys_per_point  := min(σ_sys_per_point, σ_total_per_point)   (cap)
    #   σ_stat_per_point := √(σ_total² − σ_sys²)  floored at 1% · |y|
    #
    # When the manifest's named σ_sys exceeds the lab's reported total at
    # some points (e.g. a constant 8% sys on a Kinney point with DATA-ERR=6%),
    # the cap reduces σ_sys at those points so σ_stat ⊕ σ_sys ≤ σ_total + ε
    # rather than blowing past σ_total. Documented in
    # scripts/uncertainty_manifest_methodology.md §3.3.
    #
    # Coherence with rank-1 GLS: the per-experiment scalar ``indep_rel`` is
    # also capped to the *minimum* per-point relative σ_sys after cap so the
    # rank-1 column never exceeds per-point σ_sys at any row (otherwise the
    # downstream Woodbury update would re-inflate the variance there).
    derive_stat_only = isinstance(stat_spec, dict) and stat_spec.get("derive_stat_only")
    if derive_stat_only:
        sigma_total = sigma_stat  # rename for clarity: column-as-total
        sigma_stat_floor = SIGMA_STAT_MIN_REL * np.abs(values_b_sr)

        # Cap σ_sys at √(σ_total² − floor²) so that, when the cap binds,
        # σ_stat = floor produces a total of exactly σ_total (preserving
        # the lab's reported total to machine precision instead of
        # over-estimating it by floor² in quadrature).
        sigma_sys_max = np.sqrt(np.maximum(sigma_total ** 2 - sigma_stat_floor ** 2, 0.0))
        n_violations = int(np.sum(sigma_sys > sigma_sys_max))
        sigma_sys = np.minimum(sigma_sys, sigma_sys_max)

        # Recompute σ_stat from the (possibly capped) σ_sys — exact
        # decomposition; the floor only binds where the cap binds.
        sigma_stat = np.sqrt(np.maximum(sigma_total ** 2 - sigma_sys ** 2,
                                        sigma_stat_floor ** 2))

        # Cap indep_rel scalar so the rank-1 GLS column ≤ σ_sys at every row.
        nz = np.abs(values_b_sr) > 0
        if np.any(nz):
            per_point_rel = sigma_sys[nz] / np.abs(values_b_sr[nz])
            indep_cap = float(np.min(per_point_rel))
            if indep_rel > indep_cap:
                indep_rel = indep_cap
        # Recompute the per-point indep/dep split from the capped scalars.
        sigma_sys_indep_b_sr = np.abs(values_b_sr) * indep_rel
        sigma_sys_dep_b_sr = np.sqrt(np.maximum(
            sigma_sys ** 2 - sigma_sys_indep_b_sr ** 2, 0.0
        ))
        # dep_rel becomes per-point after the cap; expose the per-row array.
        with np.errstate(divide="ignore", invalid="ignore"):
            dep_rel = np.where(
                nz, sigma_sys_dep_b_sr / np.maximum(np.abs(values_b_sr), 1e-300), 0.0
            )

        n_floored = int(np.sum(sigma_stat <= sigma_stat_floor + 1e-30))
        if n_violations or n_floored:
            tag = "default→total" if default_promoted_total else "manifest"
            print(f"  [uncertainty_manifest] {dataset_id} ({tag}): "
                  f"σ_sys capped at σ_total on {n_violations}/{len(sigma_total)} rows; "
                  f"σ_stat floored at {SIGMA_STAT_MIN_REL*100:g}% of |y| on "
                  f"{n_floored}/{len(sigma_stat)} rows.")

    # ── Optional sys_dep block: diagonal residual ────────────────────────────
    # Backward-compatible extension. When an entry carries a top-level
    # ``sys_dep`` block with ``covariance_role: diagonal_residual``, the
    # piecewise_E ``total_error_spec`` defines σ_total per row (% of |y|), and
    # the residual after removing σ_stat and σ_sys_indep in quadrature is
    # folded into σ_stat. Effect: the documented energy-dependent envelope
    # becomes uncorrelated point-to-point noise instead of a rank-1 shape
    # mode (which is the legacy ``kind: total_minus_stat`` treatment). This is
    # how Cierjacks is treated in the production manifest
    # ``uncertainty_manifest_diagonal_cierjacks.yaml``.
    # Entries without a ``sys_dep`` block are unaffected.
    sys_dep_spec = entry.get("sys_dep") if entry else None
    if (
        isinstance(sys_dep_spec, dict)
        and sys_dep_spec.get("covariance_role") == "diagonal_residual"
        and sys_dep_spec.get("correlated") is False
        and sys_dep_spec.get("source") == "piecewise_E"
    ):
        pw = sys_dep_spec.get("total_error_spec") or sys_dep_spec.get("spec", [])
        pcts, covered = _eval_piecewise_E(pw, energies_mev)
        sigma_total_dep = np.abs(values_b_sr) * (pcts / 100.0)
        residual_sq = np.maximum(
            sigma_total_dep ** 2 - sigma_stat ** 2 - sigma_sys_indep_b_sr ** 2,
            0.0,
        )
        residual = np.where(covered, np.sqrt(residual_sq), 0.0)
        n_applied = int(np.sum(residual > 0))
        if n_applied:
            sigma_stat = np.sqrt(sigma_stat ** 2 + residual ** 2)
            print(f"  [uncertainty_manifest] {dataset_id}: sys_dep diagonal "
                  f"residual folded into σ_stat on {n_applied}/{len(residual)} rows "
                  f"(max {100*np.max(residual[covered] / np.maximum(np.abs(values_b_sr[covered]), 1e-30)):.2f}% of |y|).")

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
