"""Shared helpers for NJOY module generators and parsers."""

from __future__ import annotations

from typing import List

Lines = List[str]


def _get(p: dict, key: str, default=None):
    """Like dict.get but treats None values as missing (returns default)."""
    val = p.get(key)
    return val if val is not None else default

_IPRINT_MAP = {"min": 0, "max": 1, "check": 2}


def parse_iprint(value) -> int | None:
    """Convert iprint string/value to integer.

    "min" → 0, "max" → 1, "check" → 2. Returns *None* for unrecognised
    values so callers can decide on their own default.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return _IPRINT_MAP.get(str(value).lower())


# ------------------------------------------------------------------
# Parsing utilities (used by per-module parse() functions)
# ------------------------------------------------------------------


def parse_card_values(line: str) -> list[str]:
    """Extract numeric/token data from an NJOY card line.

    In NJOY input, ``/`` terminates the data portion of a card — everything
    after the first ``/`` (outside single quotes) is treated as a comment.
    Returns the list of whitespace-separated tokens from the data portion.
    """
    line = line.strip()
    # Find the first '/' that is not inside single quotes
    in_quote = False
    for i, ch in enumerate(line):
        if ch == "'":
            in_quote = not in_quote
        elif ch == "/" and not in_quote:
            line = line[:i].strip()
            break
    return line.split()


def resolve_mat(p: dict, key: str = "mat") -> str | int:
    """Return MAT number or ``{MAT}`` placeholder when isotope is unset."""
    isotope = p.get(key, "")
    if not isotope or str(isotope).strip() == "":
        return "{MAT}"
    from ..isotopes import get_mat_number
    return get_mat_number(isotope)


def resolve_temp(p: dict, key: str = "temp") -> str:
    """Return temperature string or ``{TEMP}`` placeholder when unset."""
    val = p.get(key, "")
    if val is None or str(val).strip() == "":
        return "{TEMP}"
    return str(val)


def resolve_temps(p: dict, key: str = "temperatures") -> tuple[list[str], int]:
    """Return (list of temp strings, count) with ``{TEMP}`` fallback.

    When the temperature value is empty or missing, returns ``(["{TEMP}"], 1)``
    so the card still renders with a placeholder.
    """
    raw = p.get(key, "")
    if raw is None or str(raw).strip() == "":
        return (["{TEMP}"], 1)
    temps = str(raw).split()
    return (temps, len(temps))


def fmt_float(v) -> str:
    """Format a numeric value for Fortran list-directed I/O.

    Uses scientific notation for non-integer, non-zero values to avoid
    locale-dependent decimal point parsing.  On systems where the locale
    uses ',' as the decimal separator, bare decimals like ``0.001`` are
    silently mis-read as ``0`` by gfortran's list-directed read.
    Scientific notation (``1.0000e-03``) is parsed correctly regardless
    of locale because the ``e`` marker makes the intent unambiguous.
    """
    if isinstance(v, str) and v.startswith("{"):
        return v  # placeholder — pass through
    f = float(v)
    if f == 0.0:
        return "0"
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.4e}"


def _fmt_value(v: str) -> str:
    """Normalize a numeric string to 4-decimal scientific notation."""
    try:
        return f"{float(v):.4e}"
    except (ValueError, TypeError):
        return v


def format_multiline_card(values: list[str], cols: int = 5) -> list[str]:
    """Format a list of values into multiple NJOY card lines.

    Each value is normalized to 4-decimal scientific notation.
    Writes *cols* values per line. The final line is terminated with ``/``.

    Returns a list of strings (one per output line).
    """
    formatted = [_fmt_value(v) for v in values]
    result: list[str] = []
    for i in range(0, len(formatted), cols):
        chunk = formatted[i : i + cols]
        line = " ".join(chunk)
        if i + cols >= len(formatted):
            line += " /"
        result.append(line)
    return result


def parse_quoted_string(line: str) -> str:
    """Extract text between single quotes from a ``'label'/`` card."""
    line = line.strip()
    start = line.index("'") + 1
    end = line.index("'", start)
    return line[start:end]
