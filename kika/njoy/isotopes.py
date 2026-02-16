"""Isotope and module-definition loaders for NJOY."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, ENDF_MAT_TO_ZAID

_DATA_DIR = Path(__file__).parent / "data"
_MODULES_DIR = _DATA_DIR / "modules"

_isotopes_cache: dict[str, int] | None = None
_modules_cache: dict[str, Any] | None = None


def _build_isotope_map() -> dict[str, int]:
    """Build isotope name -> MAT mapping from ``ENDF_MAT_TO_ZAID``.

    Ground-state isotopes get plain names (e.g. ``"U235"``).
    When multiple MATs share the same ZAID the lowest MAT is treated as
    ground state; subsequent ones get an ``"m"`` suffix (e.g. ``"Am242m"``).
    """
    # Group MATs by ZAID
    zaid_to_mats: dict[int, list[int]] = defaultdict(list)
    for mat, zaid in ENDF_MAT_TO_ZAID.items():
        zaid_to_mats[zaid].append(mat)

    mapping: dict[str, int] = {}
    for zaid, mats in zaid_to_mats.items():
        z = zaid // 1000
        a = zaid % 1000
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z, f"Z{z}")
        base_name = f"{symbol}{a}" if a else f"{symbol}-nat"

        mats_sorted = sorted(mats)
        # First (lowest) MAT = ground state
        mapping[base_name] = mats_sorted[0]
        # Additional MATs = metastable states
        for i, mat in enumerate(mats_sorted[1:], start=1):
            suffix = "m" if i == 1 else f"m{i}"
            mapping[f"{base_name}{suffix}"] = mat

    return mapping


def load_isotopes() -> dict[str, int]:
    """Return isotope name -> MAT number mapping.

    Derived from :data:`kika._constants.ENDF_MAT_TO_ZAID`; cached after
    first call.
    """
    global _isotopes_cache
    if _isotopes_cache is None:
        _isotopes_cache = _build_isotope_map()
    return _isotopes_cache


def load_module_definitions() -> dict[str, Any]:
    """Load all available module JSON definitions from ``data/modules/``."""
    global _modules_cache
    if _modules_cache is None:
        _modules_cache = {}
        for filename in os.listdir(_MODULES_DIR):
            if filename.endswith(".json"):
                with open(_MODULES_DIR / filename, "r") as f:
                    mod = json.load(f)
                    _modules_cache[mod["name"].upper()] = mod
    return _modules_cache


def get_mat_number(isotope: str) -> int:
    """Get MAT number for an isotope name (e.g. ``'U235'`` -> ``9228``)."""
    return load_isotopes().get(isotope, 9228)
