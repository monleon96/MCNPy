"""Parser for Serpent depletion output files (_dep.m).

Serpent writes burnup-dependent isotope inventories in MATLAB format.
This module extracts ZAI lists, material property arrays, totals,
and burnup/time vectors.
"""

from __future__ import annotations

import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

from kika.serpent.depletion import DepletionFile, DepletionMaterial


# ── Regex patterns ───────────────────────────────────────────────────────────

# ZAI = [ 922350  922380  ... 0 ];
_ZAI_RE = re.compile(r"^ZAI\s*=\s*\[([^\]]*)\]\s*;", re.MULTILINE)

# NAMES = [ 'U-235  ' 'U-238  ' ... 'total' ];
_NAMES_RE = re.compile(r"^NAMES\s*=\s*\[([^\]]*)\]\s*;", re.MULTILINE)

# Material arrays: MAT_<name>_<PROP> = [ ... ];
# Use a broad capture for the material name (greedy up to last underscore+PROP)
_MAT_PROP_RE = re.compile(
    r"^MAT_(\w+?)_(VOLUME|BURNUP|ADENS|MDENS|A|H|SF|GSRC|ING_TOX|INH_TOX)\s*=\s*\[([^\]]*)\]\s*;",
    re.MULTILINE,
)

# Total arrays: TOT_<PROP> = [ ... ];
_TOT_PROP_RE = re.compile(
    r"^TOT_(VOLUME|BURNUP|ADENS|MDENS|A|H|SF|GSRC|ING_TOX|INH_TOX)\s*=\s*\[([^\]]*)\]\s*;",
    re.MULTILINE,
)

# BU = [ ... ];
_BU_RE = re.compile(r"^BU\s*=\s*\[([^\]]*)\]\s*;", re.MULTILINE)

# DAYS = [ ... ];
_DAYS_RE = re.compile(r"^DAYS\s*=\s*\[([^\]]*)\]\s*;", re.MULTILINE)

_NUM_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?")

# Quoted name strings: 'U-235  '
_QUOTED_NAME_RE = re.compile(r"'([^']*)'")


def _parse_floats(body: str) -> List[float]:
    return [float(x) for x in _NUM_RE.findall(body)]


def _parse_ints(body: str) -> List[int]:
    return [int(float(x)) for x in _NUM_RE.findall(body)]


# ── Public API ───────────────────────────────────────────────────────────────

def parse_depletion_text(text: str) -> DepletionFile:
    """Parse raw text content of a Serpent ``_dep.m`` file.

    Parameters
    ----------
    text : str
        Full text content of the depletion output file.

    Returns
    -------
    DepletionFile
    """
    # --- ZAI vector ---
    zai: List[int] = []
    m = _ZAI_RE.search(text)
    if m:
        raw = _parse_ints(m.group(1))
        # ZAI list is terminated with special markers (666, 0)
        # Filter out non-physical entries
        zai = [z for z in raw if z > 0 and z != 666]

    # --- NAMES ---
    names: List[str] = []
    m = _NAMES_RE.search(text)
    if m:
        names = [n.strip() for n in _QUOTED_NAME_RE.findall(m.group(1))]

    # --- BU and DAYS vectors ---
    burnup = np.array([])
    m = _BU_RE.search(text)
    if m:
        burnup = np.array(_parse_floats(m.group(1)))

    days = np.array([])
    m = _DAYS_RE.search(text)
    if m:
        days = np.array(_parse_floats(m.group(1)))

    n_steps = len(burnup) if len(burnup) > 0 else len(days)
    # Number of rows per property matrix: n_isotopes + 2 (lost + total)
    # The ZAI list has the real isotopes; rows = len(zai) + 2
    n_rows = len(zai) + 2 if zai else 0

    # --- Material property arrays ---
    mat_data: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for m_match in _MAT_PROP_RE.finditer(text):
        mat_name = m_match.group(1)
        prop_name = m_match.group(2)
        body = m_match.group(3)
        values = _parse_floats(body)

        if n_rows > 0 and n_steps > 0 and len(values) >= n_rows * n_steps:
            arr = np.array(values[:n_rows * n_steps]).reshape(n_rows, n_steps)
        elif n_steps > 0:
            # Scalar property per step (e.g., VOLUME, BURNUP)
            arr = np.array(values[:n_steps]).reshape(1, n_steps)
        else:
            arr = np.array(values)

        mat_data[mat_name][prop_name] = arr

    materials: Dict[str, DepletionMaterial] = {}
    for mat_name, props in mat_data.items():
        materials[mat_name] = DepletionMaterial(name=mat_name, properties=props)

    # --- Total arrays ---
    totals: Dict[str, np.ndarray] = {}
    for m_match in _TOT_PROP_RE.finditer(text):
        prop_name = m_match.group(1)
        body = m_match.group(2)
        values = _parse_floats(body)

        if n_rows > 0 and n_steps > 0 and len(values) >= n_rows * n_steps:
            arr = np.array(values[:n_rows * n_steps]).reshape(n_rows, n_steps)
        elif n_steps > 0:
            arr = np.array(values[:n_steps]).reshape(1, n_steps)
        else:
            arr = np.array(values)

        totals[prop_name] = arr

    return DepletionFile(
        zai=zai,
        names=names,
        materials=materials,
        totals=totals,
        burnup=burnup,
        days=days,
    )


def read_depletion_file(path_or_text: Union[str, Path]) -> DepletionFile:
    """Parse a Serpent ``_dep.m`` file.

    Parameters
    ----------
    path_or_text : str or Path
        File system path or raw text content.

    Returns
    -------
    DepletionFile
    """
    p = Path(path_or_text) if not isinstance(path_or_text, Path) else path_or_text
    if p.exists():
        text = p.read_text(encoding="utf-8")
    else:
        text = str(path_or_text)
    return parse_depletion_text(text)
