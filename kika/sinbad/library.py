"""
Discovery of SINBAD packages in a configured directory.

kika ships no benchmark data. Point it at your own folder of packages once,
then refer to benchmarks by identifier instead of by path.

Example
-------
    >>> import kika.sinbad as sinbad
    >>> sinbad.configure(path="~/sinbad-packages")
    >>> sinbad.catalogue()
    >>> b = sinbad.open("SINBAD-ASPIS-IRON88")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from kika.sinbad._constants import (
    GENERATED_DESCRIPTION,
    MANIFEST,
    PACKAGE_SUFFIX,
)
from kika.sinbad.config import get_library_path
from kika.sinbad.exceptions import LibraryNotConfiguredError, PackageNotFoundError
from kika.sinbad.package import SinbadPackage

__all__ = ["scan", "list_benchmarks", "catalogue", "find_package"]


def _candidates(root: Path):
    """Yield paths that look like SINBAD packages, both forms."""
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix == PACKAGE_SUFFIX:
            yield p
        elif p.is_dir() and (p / MANIFEST).is_file():
            yield p


def scan(path: Optional[str] = None) -> dict:
    """
    Map benchmark identifier to package path for a library directory.

    When both package forms of the same benchmark are present, the single-file
    ``.sinbad`` archive wins -- it is the distribution unit, and it carries the
    checksums that make it citable.

    Parameters
    ----------
    path : str, optional
        Library directory. Defaults to the configured one.

    Returns
    -------
    dict
        ``{identifier: pathlib.Path}``.

    Raises
    ------
    LibraryNotConfiguredError
        If the resolved directory does not exist.
    """
    root = Path(get_library_path(path))
    if not root.is_dir():
        raise LibraryNotConfiguredError(
            f"SINBAD package library not found at '{root}'. Set it with "
            "kika.sinbad.configure(path=...), the KIKA_SINBAD_PATH environment "
            "variable, or pass an explicit path to a package."
        )

    found = {}
    for candidate in _candidates(root):
        try:
            pkg = SinbadPackage(candidate)
            model = json.loads(pkg.read_text(GENERATED_DESCRIPTION))
            pkg.close()
        except Exception:  # noqa: BLE001 - a malformed folder is not an error here
            continue
        identifier = model["identification"]["id"]
        if identifier not in found or candidate.suffix == PACKAGE_SUFFIX:
            found[identifier] = candidate
    return found


def list_benchmarks(path: Optional[str] = None) -> list:
    """
    Return the identifiers of every benchmark in the library.

    Parameters
    ----------
    path : str, optional
        Library directory. Defaults to the configured one.

    Returns
    -------
    list of str
    """
    return sorted(scan(path))


def catalogue(path: Optional[str] = None) -> pd.DataFrame:
    """
    Return a table of every benchmark in the library.

    Parameters
    ----------
    path : str, optional
        Library directory. Defaults to the configured one.

    Returns
    -------
    pandas.DataFrame
        Columns ``id``, ``title``, ``facility``, ``year``, ``measurements``,
        ``libraries``, ``form``, ``path``.
    """
    rows = []
    for identifier, pkg_path in sorted(scan(path).items()):
        pkg = SinbadPackage(pkg_path)
        model = json.loads(pkg.read_text(GENERATED_DESCRIPTION))
        pkg.close()
        ident = model["identification"]
        rows.append(
            {
                "id": identifier,
                "title": ident["title"],
                "facility": ident["facility"],
                "year": ident["year"],
                "measurements": len(model["experiment"]["measurements"]),
                "libraries": ", ".join(
                    sorted({c["nuclearDataLibrary"] for c in model["calculations"]})
                ),
                "form": "archive" if pkg_path.suffix == PACKAGE_SUFFIX else "directory",
                "path": str(pkg_path),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "title",
            "facility",
            "year",
            "measurements",
            "libraries",
            "form",
            "path",
        ],
    )


def find_package(identifier: str, path: Optional[str] = None) -> Path:
    """
    Resolve a benchmark identifier to a package path.

    Parameters
    ----------
    identifier : str
        Benchmark identifier, e.g. ``"SINBAD-ASPIS-IRON88"``.
    path : str, optional
        Library directory. Defaults to the configured one.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    PackageNotFoundError
        If no package in the library declares that identifier.
    """
    found = scan(path)
    if identifier in found:
        return found[identifier]

    # Be forgiving about the common shorthand -- people say "ASPIS-IRON88".
    matches = [k for k in found if identifier.upper() in k.upper()]
    if len(matches) == 1:
        return found[matches[0]]

    raise PackageNotFoundError(
        f"No SINBAD package with identifier '{identifier}' in "
        f"'{get_library_path(path)}'. Available: {sorted(found) or 'none'}"
    )
