"""Parser for Serpent results output files (_res.m).

Serpent writes run statistics in MATLAB format.  This module extracts all
indexed variables and wraps them in a :class:`ResultsFile` object.

Two variable patterns are handled::

    VARNAME (idx, [1: N]) = [ val1 val2 ... ];   % array variable
    VARNAME (idx, 1)      = [ val ];              % scalar variable
"""

from __future__ import annotations

import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

from kika.serpent.results import ResultsFile


# ── Regex patterns ───────────────────────────────────────────────────────────

# Array variable: VARNAME (idx, [1: N]) = [ val1 val2 ... ];
_ARRAY_VAR_RE = re.compile(
    r"^(\w+)\s*\(\s*idx\s*,\s*\[\s*1\s*:\s*(\d+)\s*\]\s*\)\s*=\s*\[([^\]]*)\]\s*;",
    re.MULTILINE,
)

# Scalar variable: VARNAME (idx, 1) = [ val ];  or  VARNAME (idx, 1) = val ;
_SCALAR_VAR_RE = re.compile(
    r"^(\w+)\s*\(\s*idx\s*,\s*1\s*\)\s*=\s*\[?\s*([^\];]+?)\s*\]?\s*;",
    re.MULTILINE,
)

_NUM_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?")


def _parse_floats(body: str) -> List[float]:
    """Extract all numeric values from a MATLAB array body string."""
    return [float(x) for x in _NUM_RE.findall(body)]


# ── Public API ───────────────────────────────────────────────────────────────

def parse_results_text(text: str) -> ResultsFile:
    """Parse raw text content of a Serpent ``_res.m`` file.

    Parameters
    ----------
    text : str
        Full text content of the results output file.

    Returns
    -------
    ResultsFile
    """
    # Collect all occurrences of each variable (one per idx / burnup step)
    var_rows: Dict[str, List[np.ndarray]] = defaultdict(list)

    # Pass 1: array variables  VARNAME (idx, [1: N]) = [ ... ];
    for m in _ARRAY_VAR_RE.finditer(text):
        name = m.group(1)
        n_values = int(m.group(2))
        body = m.group(3)
        values = _parse_floats(body)
        if len(values) >= n_values:
            var_rows[name].append(np.array(values[:n_values]))

    # Pass 2: scalar variables  VARNAME (idx, 1) = value ;
    for m in _SCALAR_VAR_RE.finditer(text):
        name = m.group(1)
        # Skip if already captured as array variable
        if name in var_rows:
            continue
        body = m.group(2)
        values = _parse_floats(body)
        if values:
            var_rows[name].append(np.array(values[:1]))

    # Stack rows into (n_steps, n_values) arrays
    variables: Dict[str, np.ndarray] = {}
    n_steps = 0
    for name, rows in var_rows.items():
        arr = np.array(rows)  # shape (n_occurrences, n_values)
        variables[name] = arr
        n_steps = max(n_steps, arr.shape[0])

    return ResultsFile(
        variables=variables,
        n_burnup_steps=max(n_steps, 1),
    )


def read_results_file(path_or_text: Union[str, Path]) -> ResultsFile:
    """Parse a Serpent ``_res.m`` file.

    Parameters
    ----------
    path_or_text : str or Path
        File system path or raw text content.

    Returns
    -------
    ResultsFile
    """
    p = Path(path_or_text) if not isinstance(path_or_text, Path) else path_or_text
    if p.exists():
        text = p.read_text(encoding="utf-8")
    else:
        text = str(path_or_text)
    return parse_results_text(text)
