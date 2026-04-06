"""
Auto-detection of MCNP file types (input deck vs MCTAL output).
"""

import re
from pathlib import Path
from typing import Optional

# Known MCNP code names that appear in MCTAL header line 1 (positions 0-7)
_MCTAL_CODE_NAMES = {'mcnp', 'mcnpx', 'mcnp6', 'mcnp5'}

# Date pattern found in MCTAL header (MM/DD/YY)
_DATE_PATTERN = re.compile(r'\d{2}/\d{2}/\d{2}')


def detect_mcnp_file_type(file_path: Optional[str] = None, content: Optional[str] = None) -> str:
    """Detect whether a file is an MCNP input deck or MCTAL output.

    Uses content-based heuristics: MCTAL files have a distinctive fixed-width
    header with a code name, version, date, and a third line starting with 'ntal'.

    :param file_path: Path to the file on disk (preferred — avoids loading full content).
    :param content: File content as a string (used if file_path is not given).
    :returns: ``'mcnp-input'`` or ``'mcnp-mctal'``
    :raises ValueError: If neither file_path nor content is provided.
    """
    if file_path is not None:
        lines = _read_first_lines(file_path, n=4)
    elif content is not None:
        lines = _lines_from_content(content, n=4)
    else:
        raise ValueError("Either file_path or content must be provided")

    if _is_mctal(lines):
        return 'mcnp-mctal'
    return 'mcnp-input'


def _read_first_lines(file_path: str, n: int) -> list:
    """Read the first *n* non-blank lines from a file on disk."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    result = []
    with open(p, 'r', errors='replace') as f:
        for raw in f:
            stripped = raw.rstrip('\n\r')
            if stripped.strip():
                result.append(stripped)
                if len(result) >= n:
                    break
    return result


def _lines_from_content(content: str, n: int) -> list:
    """Extract the first *n* non-blank lines from a content string."""
    result = []
    for raw in content.splitlines():
        stripped = raw.rstrip()
        if stripped.strip():
            result.append(stripped)
            if len(result) >= n:
                break
    return result


def _is_mctal(lines: list) -> bool:
    """Check whether the first few lines match the MCTAL header format.

    MCTAL header (line 1): code_name (8 chars) + spaces + version + date (MM/DD/YY) + ...
    MCTAL line 3: starts with 'ntal' followed by tally count.
    """
    if len(lines) < 3:
        return False

    header = lines[0]

    # Check code name in first 8 characters
    code_name = header[:8].strip().lower()
    if code_name not in _MCTAL_CODE_NAMES:
        return False

    # Check for date pattern in the header line
    if not _DATE_PATTERN.search(header):
        return False

    # Check line 3 starts with 'ntal'
    line3 = lines[2].strip()
    if not line3.lower().startswith('ntal'):
        return False

    return True
