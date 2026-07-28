"""
Byte access to a SINBAD package, whichever form it arrived in.

A SINBAD package exists in two forms holding identical bytes: a working
directory that a curator edits and a reviewer diffs, and a single ``.sinbad``
archive that a user downloads and cites. :class:`SinbadPackage` is the only
class in this subpackage that knows which one it was handed.

Nothing here interprets the content -- that is :mod:`kika.sinbad.benchmark`.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Union

from kika.sinbad.exceptions import PackageNotFoundError


class SinbadPackage:
    """
    Component-level access to a SINBAD package.

    Parameters
    ----------
    path : str or pathlib.Path
        A package directory or a ``.sinbad`` archive.

    Attributes
    ----------
    path : pathlib.Path
        The path this package was opened from.
    kind : str
        ``"directory"`` or ``"archive"``.

    Examples
    --------
    >>> pkg = SinbadPackage("aspis-iron88.sinbad")
    >>> pkg.names()
    ['arrays.h5', 'benchmark.json', 'benchmark.xml', 'manifest.xml']
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).expanduser()
        if self.path.is_dir():
            self.kind = "directory"
            self._zip = None
            self._prefix = ""
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.kind = "archive"
            self._zip = zipfile.ZipFile(self.path)
            self._prefix = self._detect_prefix(self._zip.namelist())
        else:
            raise PackageNotFoundError(
                f"{self.path} is neither a package directory nor a .sinbad archive"
            )

    @staticmethod
    def _detect_prefix(names: list) -> str:
        """Members may sit at the archive root or under one top-level folder."""
        real = [n for n in names if not n.endswith("/")]
        tops = {n.split("/")[0] for n in real}
        return f"{tops.pop()}/" if len(tops) == 1 and all("/" in n for n in real) else ""

    def names(self) -> list:
        """Return the component filenames, sorted."""
        if self._zip is None:
            return sorted(p.name for p in self.path.iterdir() if p.is_file())
        return sorted(
            n[len(self._prefix):] for n in self._zip.namelist() if not n.endswith("/")
        )

    def read_bytes(self, name: str) -> bytes:
        """Return the raw bytes of one component."""
        if self._zip is None:
            return (self.path / name).read_bytes()
        return self._zip.read(f"{self._prefix}{name}")

    def read_text(self, name: str) -> str:
        """Return one component decoded as UTF-8 text."""
        return self.read_bytes(name).decode("utf-8")

    def open(self, name: str):
        """
        Return a seekable binary handle to one component.

        This is what ``h5py`` and ``numpy.load`` need. For the archive form the
        member is read into memory first; at benchmark-package sizes that is
        cheaper than extracting to a temporary directory.
        """
        if self._zip is None:
            return (self.path / name).open("rb")
        return io.BytesIO(self.read_bytes(name))

    def close(self) -> None:
        """Release the archive handle, if any."""
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __repr__(self) -> str:
        return (
            f"<SinbadPackage {self.path.name} ({self.kind}), "
            f"{len(self.names())} components>"
        )
