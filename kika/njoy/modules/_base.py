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


def parse_quoted_string(line: str) -> str:
    """Extract text between single quotes from a ``'label'/`` card."""
    line = line.strip()
    start = line.index("'") + 1
    end = line.index("'", start)
    return line[start:end]
