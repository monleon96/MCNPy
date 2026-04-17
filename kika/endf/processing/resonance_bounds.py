"""Detect the overall resonance energy range covered by an MF2/MT151 section.

Used to auto-pick sensible bounds when a caller (e.g., the group-averaging
pipeline) wants to operate on the resonance region without being told
where it is.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

__all__ = ["detect_resonance_bounds"]


def _has_real_parameters(params: Any) -> bool:
    """True when the energy range carries actual resolved/unresolved data."""
    if params is None:
        return False
    for lb in getattr(params, "l_values", []) or []:
        if getattr(lb, "resonances", None):
            return True
        if getattr(lb, "j_states", None):
            return True
        for js in getattr(lb, "j_states", []) or []:
            if getattr(js, "energy_points", None):
                return True
    return False


def detect_resonance_bounds(mf2: Any) -> Optional[Tuple[float, float]]:
    """Return ``(E_low, E_high)`` spanning all real resonance ranges in MF2.

    Accepts either a parsed MF2 object (with ``.mt[151].isotopes``) or an
    MF2MT151 instance directly (with ``.isotopes``). Energy ranges with
    ``lru in {1, 2}`` and non-empty parameters contribute to the span;
    scattering-radius-only ranges (``lru=0``) are ignored.

    Returns ``None`` when there is no real resolved or unresolved
    resonance data — e.g., stub MF2 sections that only carry the LRP
    header (H-1, H-2).
    """
    if mf2 is None:
        return None

    mt151 = getattr(mf2, "mt", None)
    if isinstance(mt151, dict):
        mt151 = mt151.get(151)
    if mt151 is None:
        # Maybe we were passed an MF2MT151 directly
        if getattr(mf2, "isotopes", None) is not None:
            mt151 = mf2
        else:
            return None

    isotopes = getattr(mt151, "isotopes", None) or []
    e_low: Optional[float] = None
    e_high: Optional[float] = None

    for iso in isotopes:
        for rng in getattr(iso, "energy_ranges", None) or []:
            lru = getattr(rng, "lru", 0)
            if lru not in (1, 2):
                continue
            if not _has_real_parameters(getattr(rng, "parameters", None)):
                continue
            el = getattr(rng, "el", None)
            eh = getattr(rng, "eh", None)
            if el is None or eh is None:
                continue
            try:
                el = float(el)
                eh = float(eh)
            except (TypeError, ValueError):
                continue
            if eh <= el:
                continue
            e_low = el if e_low is None else min(e_low, el)
            e_high = eh if e_high is None else max(e_high, eh)

    if e_low is None or e_high is None:
        return None
    return e_low, e_high
