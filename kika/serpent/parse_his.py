"""Parser for Serpent history output files (_his.m).

Serpent writes per-cycle statistics in MATLAB format.  This module
extracts ``HIS_TIME`` and all ``HIS_<VARNAME>`` arrays, splitting
inactive and active cycles.
"""

from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Union

from kika.serpent.history import HistoryFile, HistoryVariable


# ── Regex patterns ───────────────────────────────────────────────────────────

# Generic block: HIS_<NAME> = [ ... ];
_HIS_BLOCK_RE = re.compile(
    r"^HIS_(\w+)\s*=\s*\[([^\]]*)\]\s*;",
    re.MULTILINE,
)

# Active-cycle separator comment
_ACTIVE_SEP_RE = re.compile(
    r"^%.*Begin active cycles",
    re.MULTILINE | re.IGNORECASE,
)

_NUM_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?")


def _parse_floats(body: str) -> List[float]:
    return [float(x) for x in _NUM_RE.findall(body)]


def _count_inactive_cycles(text: str) -> int:
    """Count inactive cycles by finding the separator in the full file text.

    We look at the HIS_TIME block: rows before the separator are inactive.
    """
    # Find HIS_TIME block
    m = re.search(r"^HIS_TIME\s*=\s*\[([^\]]*)\]\s*;", text, re.MULTILINE)
    if not m:
        return 0
    block = m.group(1)
    # Count data rows before the active-cycle separator
    n_inactive = 0
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^%.*Begin active cycles", stripped, re.IGNORECASE):
            break
        # Skip pure comment lines
        if stripped.startswith("%"):
            continue
        n_inactive += 1
    return n_inactive


# ── Public API ───────────────────────────────────────────────────────────────

def parse_history_text(text: str) -> HistoryFile:
    """Parse raw text content of a Serpent ``_his.m`` file.

    Parameters
    ----------
    text : str
        Full text content of the history output file.

    Returns
    -------
    HistoryFile
    """
    n_inactive = _count_inactive_cycles(text)

    time_arr = np.empty((0, 2))
    variables: Dict[str, HistoryVariable] = {}

    for m in _HIS_BLOCK_RE.finditer(text):
        name = m.group(1)  # e.g., "TIME", "IMP_KEFF", "ANA_KEFF", "ENTROPY"
        body = m.group(2)

        # Strip comment lines (% ...) from body before parsing numbers
        clean_lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("%"):
                clean_lines.append(stripped)

        values = _parse_floats("\n".join(clean_lines))
        if not values:
            continue

        if name == "TIME":
            # HIS_TIME: 2 columns [cycle, time]
            n_rows = len(values) // 2
            time_arr = np.array(values[:n_rows * 2]).reshape(n_rows, 2)
        else:
            # Variable blocks: first column is cycle number,
            # then groups of 3 (value, mean, error) per score.
            # Detect column count from the first data line
            first_line_vals = _parse_floats(clean_lines[0]) if clean_lines else []
            ncols = len(first_line_vals)
            if ncols < 4:
                continue  # need at least cycle + 1 score (val, mean, err)

            n_rows = len(values) // ncols
            if n_rows == 0:
                continue
            data = np.array(values[:n_rows * ncols]).reshape(n_rows, ncols)

            cycles = data[:, 0]
            n_scores = (ncols - 1) // 3

            # Extract value/mean/error for each score
            vals = np.zeros((n_rows, n_scores))
            means = np.zeros((n_rows, n_scores))
            errors = np.zeros((n_rows, n_scores))
            for s in range(n_scores):
                base = 1 + s * 3
                vals[:, s] = data[:, base]
                means[:, s] = data[:, base + 1]
                errors[:, s] = data[:, base + 2]

            variables[name] = HistoryVariable(
                name=name,
                cycles=cycles,
                values=vals,
                means=means,
                errors=errors,
                n_scores=n_scores,
            )

    n_total = time_arr.shape[0] if time_arr.size > 0 else 0

    return HistoryFile(
        time=time_arr,
        variables=variables,
        n_inactive=n_inactive,
        n_total=n_total,
    )


def read_history_file(path_or_text: Union[str, Path]) -> HistoryFile:
    """Parse a Serpent ``_his.m`` file.

    Parameters
    ----------
    path_or_text : str or Path
        File system path or raw text content.

    Returns
    -------
    HistoryFile
    """
    p = Path(path_or_text) if not isinstance(path_or_text, Path) else path_or_text
    if p.exists():
        text = p.read_text(encoding="utf-8")
    else:
        text = str(path_or_text)
    return parse_history_text(text)
