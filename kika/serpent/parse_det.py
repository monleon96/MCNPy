"""Parser for Serpent detector output files (_det.m).

Serpent writes detector results in MATLAB format.  This module extracts
``DET<name>``, ``DET<name>E``, and ``DET<name>T`` arrays and wraps them
in :class:`DetectorResult` / :class:`DetectorFile` objects.
"""

from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from typing import Dict, Union

from kika.serpent.detector import DetectorFile, DetectorResult


# ── Regex patterns ───────────────────────────────────────────────────────────

# Main detector data: DET<name> = [...]
_DET_DATA_RE = re.compile(
    r"^DET(\w+)\s*=\s*\[([^\]]*)\]\s*;",
    re.MULTILINE,
)

# Energy grid: DET<name>E = [...]
_DET_ENERGY_RE = re.compile(
    r"^DET(\w+)E\s*=\s*\[([^\]]*)\]\s*;",
    re.MULTILINE,
)

# Time grid: DET<name>T = [...]
_DET_TIME_RE = re.compile(
    r"^DET(\w+)T\s*=\s*\[([^\]]*)\]\s*;",
    re.MULTILINE,
)

_NUM_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?")


def _parse_matrix(body: str, ncols: int = 0) -> np.ndarray:
    """Parse a MATLAB-style numeric matrix from the body of ``[...]``."""
    values = [float(x) for x in _NUM_RE.findall(body)]
    if not values:
        return np.empty((0, 0))
    arr = np.array(values)
    if ncols > 0:
        nrows = len(arr) // ncols
        return arr[:nrows * ncols].reshape(nrows, ncols)
    return arr


# ── Public API ───────────────────────────────────────────────────────────────

def parse_detector_text(text: str) -> DetectorFile:
    """Parse raw text content of a Serpent ``_det.m`` file.

    Parameters
    ----------
    text : str
        Full text content of the detector output file.

    Returns
    -------
    DetectorFile
    """
    # First pass: collect energy and time grids (3-column matrices)
    energy_grids: Dict[str, np.ndarray] = {}
    for m in _DET_ENERGY_RE.finditer(text):
        name = m.group(1)
        energy_grids[name] = _parse_matrix(m.group(2), ncols=3)

    time_grids: Dict[str, np.ndarray] = {}
    for m in _DET_TIME_RE.finditer(text):
        name = m.group(1)
        time_grids[name] = _parse_matrix(m.group(2), ncols=3)

    # Second pass: collect detector data (exclude energy/time grid entries)
    detectors: Dict[str, DetectorResult] = {}
    for m in _DET_DATA_RE.finditer(text):
        name = m.group(1)
        # Skip if this is actually an energy or time grid (DET<name>E or DET<name>T)
        # The regex should already exclude these since E/T suffix is part of the name
        # but double check
        if name.endswith("E") and name[:-1] in energy_grids:
            continue
        if name.endswith("T") and name[:-1] in time_grids:
            continue

        body = m.group(2)
        values = [float(x) for x in _NUM_RE.findall(body)]
        if not values:
            continue

        # Determine column count: 12 (no time) or 13 (with time)
        # If time grid exists for this detector, assume 13 columns
        has_time = name in time_grids
        ncols = 13 if has_time else 12

        # Validate: total values should be divisible by ncols
        if len(values) % ncols != 0:
            # Try the other column count
            alt = 13 if ncols == 12 else 12
            if len(values) % alt == 0:
                ncols = alt
            else:
                # Fall back to 12
                ncols = 12

        nrows = len(values) // ncols
        data = np.array(values[:nrows * ncols]).reshape(nrows, ncols)

        detectors[name] = DetectorResult(
            name=name,
            data=data,
            energy_grid=energy_grids.get(name),
            time_grid=time_grids.get(name),
        )

    return DetectorFile(detectors=detectors)


def read_detector_file(path_or_text: Union[str, Path]) -> DetectorFile:
    """Parse a Serpent ``_det.m`` file.

    Parameters
    ----------
    path_or_text : str or Path
        File system path or raw text content.

    Returns
    -------
    DetectorFile
    """
    p = Path(path_or_text) if not isinstance(path_or_text, Path) else path_or_text
    if p.exists():
        text = p.read_text(encoding="utf-8")
    else:
        text = str(path_or_text)
    return parse_detector_text(text)
