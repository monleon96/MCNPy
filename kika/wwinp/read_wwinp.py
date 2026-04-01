"""High-level entry point for reading WWINP files."""

from __future__ import annotations

from pathlib import Path

from kika.wwinp.classes.wwinp import WWINP
from kika.wwinp.parsers.parse_wwinp import parse_wwinp_file


def read_wwinp(filepath: str | Path) -> WWINP:
    """Read a WWINP file and return a :class:`WWINP` object.

    Parameters
    ----------
    filepath : str or Path
        Path to the WWINP file.

    Returns
    -------
    WWINP
        Parsed weight window data.

    Examples
    --------
    >>> from kika import read_wwinp
    >>> ww = read_wwinp("path/to/wwinp")
    >>> print(ww)
    """
    return parse_wwinp_file(filepath)
