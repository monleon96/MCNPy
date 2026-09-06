"""Nuclide-first index of everything the IAEA ``download-endf`` service holds.

``endf_catalog.json.gz`` is a snapshot of the IAEA directory listings (see
:mod:`kika.endf.remote.catalog_build`): every library, every sub-library, one
row per file with its exact name, size and date. This module parses those rows
into :class:`CatalogEntry` objects and indexes them by nuclide, so the questions
the app asks are cheap:

- which libraries carry Fe-56 neutron data, and how big is each file?
- which nuclides does JEFF-4.0 evaluate for protons?
- what is the real URL of that file — no MAT table, no filename-style guess.

**The snapshot is not shipped.** It is 2 MB of scraped directory listings, so
it is neither carried in the repository nor put in the wheel; the first call
builds it into the user's cache directory. Evaluated libraries are frozen
releases, so once built it stays correct until the IAEA adds a library.
:func:`refresh_catalog` rebuilds it, and :func:`get_catalog` reads the cached
copy — falling back to :data:`PACKAGE_CATALOG`, the slot beside this module, for
a deployment that chooses to place one there.

Example::

    from kika.endf.remote.catalog import get_catalog

    cat = get_catalog()
    for e in cat.entries("Fe56"):            # neutron sub-library by default
        print(e.library, e.size_bytes, e.url)
    cat.nuclides("p", library="jendl5")      # every proton target in JENDL-5
"""

from __future__ import annotations

import gzip
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from kika._constants import ATOMIC_NUMBER_TO_SYMBOL

from .constants import (
    IAEA_BASE_URL,
    LIBRARY_ALIASES,
    LIBRARY_PATHS,
    SUBLIB_NAMES,
    get_cache_dir,
    library_display_name,
    library_family,
)

#: The slot beside this module where a snapshot *may* be placed. Empty in the
#: repository and in the wheel (see the module docstring); read as a fallback
#: so that a site which does want to ship one only has to drop it in.
PACKAGE_CATALOG = Path(__file__).with_name("endf_catalog.json.gz")
CACHE_CATALOG_NAME = "endf_catalog.json.gz"

# The two filename conventions the IAEA uses, see LIBRARY_FILENAME_STYLES:
#   n_001-H-1_0125.zip     "za_mat"  (Z padded to 3, MAT to 4)
#   n_0125_1-H-1.zip       "mat_za"  (MAT padded to 4, Z bare)
# plus the odd variants met in the tree: an unpadded Z (``p_12-Mg-22_0160``),
# an upper-case symbol (``he4_002-HE-4_0228``), the neutron as ``n``/``nn``
# with Z=0 in the decay sub-libraries, and ``.dat`` instead of ``.zip``.
_NUCLIDE_FILE = re.compile(
    r"^[A-Za-z0-9]+_(?:"
    r"(?P<mat1>\d{4})_(?P<z1>\d{1,3})-(?P<el1>[A-Za-z]{1,2})-(?P<a1>\d{1,3})(?P<m1>[mM]\d?)?"
    r"|"
    r"(?P<z2>\d{1,3})-(?P<el2>[A-Za-z]{1,2})-(?P<a2>\d{1,3})(?P<m2>[mM]\d?)?_(?P<mat2>\d{4})"
    r")\.(?:zip|dat)$",
    re.IGNORECASE,
)
# Thermal-scattering files name a material, not a nuclide:
#   tsl_H(H2O)_0001.dat  tsl_10Graphite_0031.zip  tsl_026-Fe-56_0056.zip
_TSL_FILE = re.compile(r"^tsl_(?P<label>.+?)_(?P<mat>\d{4})\.(?:zip|dat)$", re.IGNORECASE)

_NEUTRON_SYMBOLS = {"n", "nn"}


def _library_id(directory: str) -> str:
    """Canonical id for a library directory: the historical short id when one
    exists (``endfb8.1``), the lower-cased directory name otherwise."""
    for lib_id, path in LIBRARY_PATHS.items():
        if path == directory:
            return lib_id
    return directory.lower()


def nuclide_label(z: int, a: int, isomer: int = 0) -> str:
    symbol = "n" if z == 0 else ATOMIC_NUMBER_TO_SYMBOL.get(z, f"Z{z}")
    if a == 0:
        return symbol  # elemental (photo-atomic, electro-atomic, relaxation)
    suffix = "" if not isomer else ("m" if isomer == 1 else f"m{isomer}")
    return f"{symbol}-{a}{suffix}"


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One file on the IAEA server."""

    library: str
    """Canonical library id, e.g. ``endfb8.1`` or ``fendl-3.2``."""
    library_dir: str
    """Directory name on the server, e.g. ``ENDF-B-VIII.1``."""
    sublib: str
    """Sub-library directory: ``n``, ``p``, ``decay``, ``tsl``..."""
    filename: str
    size_bytes: int
    modified: str
    """``YYYY-MM-DD`` as reported by the server."""
    mat: int | None
    z: int | None
    a: int | None
    isomer: int
    label: str
    """``Fe-56``, ``Am-242m``, ``H`` (elemental), or the TSL material name."""

    @property
    def url(self) -> str:
        return f"{IAEA_BASE_URL}/{self.library_dir}/{self.sublib}/{self.filename}"

    @property
    def symbol(self) -> str | None:
        if self.z is None:
            return None
        return "n" if self.z == 0 else ATOMIC_NUMBER_TO_SYMBOL.get(self.z)

    @property
    def zaid(self) -> int | None:
        if self.z is None or self.a is None:
            return None
        return self.z * 1000 + self.a

    @property
    def nuclide_key(self) -> str | None:
        """``Z-A-m`` string shared by every entry of the same nuclide."""
        if self.z is None or self.a is None:
            return None
        return f"{self.z}-{self.a}-{self.isomer}"

    @property
    def cache_key(self) -> str:
        """Filename stem used by :class:`~kika.endf.remote.cache.ENDFCache`:
        the MCNP-style ZAID (A offset by 400 for an isomer), or the server's
        own stem for files that name no nuclide."""
        if self.zaid is not None:
            return str(self.zaid + (400 if self.isomer else 0))
        return Path(self.filename).stem

    @property
    def library_name(self) -> str:
        return library_display_name(self.library_dir)

    @property
    def is_archive(self) -> bool:
        return self.filename.lower().endswith(".zip")

    def to_dict(self) -> dict:
        return {
            "library": self.library,
            "library_dir": self.library_dir,
            "library_name": self.library_name,
            "family": library_family(self.library_dir),
            "sublib": self.sublib,
            "filename": self.filename,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "modified": self.modified,
            "mat": self.mat,
            "z": self.z,
            "a": self.a,
            "isomer": self.isomer,
            "symbol": self.symbol,
            "zaid": self.zaid,
            "label": self.label,
            "nuclide_key": self.nuclide_key,
        }


@dataclass(frozen=True, slots=True)
class NuclideInfo:
    """A nuclide as seen across the catalogue."""

    z: int
    a: int
    isomer: int
    symbol: str
    label: str
    libraries: int
    """How many libraries carry it (in the sub-library that was asked for)."""

    @property
    def key(self) -> str:
        return f"{self.z}-{self.a}-{self.isomer}"

    @property
    def zaid(self) -> int:
        return self.z * 1000 + self.a

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "z": self.z,
            "a": self.a,
            "isomer": self.isomer,
            "symbol": self.symbol,
            "zaid": self.zaid,
            "label": self.label,
            "libraries": self.libraries,
        }


@dataclass(frozen=True, slots=True)
class LibraryInfo:
    id: str
    directory: str
    name: str
    family: str
    sublibs: dict[str, int]
    """Files per sub-library."""
    latest: str
    """Most recent file date in the library, ``YYYY-MM-DD``."""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "directory": self.directory,
            "name": self.name,
            "family": self.family,
            "sublibs": self.sublibs,
            "latest": self.latest,
        }


def parse_catalog_filename(filename: str, sublib: str) -> dict | None:
    """Read Z/A/isomer/MAT and a display label off a listing filename.

    Returns ``None`` for files that name neither a nuclide nor a TSL material,
    which is how whole-library tapes and stray listings are dropped.
    """
    m = _NUCLIDE_FILE.match(filename)
    if m:
        side = "1" if m.group("mat1") else "2"
        z = int(m.group(f"z{side}"))
        a = int(m.group(f"a{side}"))
        el = m.group(f"el{side}")
        iso = m.group(f"m{side}")
        mat = int(m.group(f"mat{side}"))
        isomer = int(iso[1:] or 1) if iso else 0
        if el.lower() in _NEUTRON_SYMBOLS:
            z = 0
        return {"z": z, "a": a, "isomer": isomer, "mat": mat, "label": nuclide_label(z, a, isomer)}

    if sublib.lower() == "tsl":
        t = _TSL_FILE.match(filename)
        if t:
            return {"z": None, "a": None, "isomer": 0, "mat": int(t.group("mat")), "label": t.group("label")}
    return None


class Catalog:
    """Parsed, indexed view of one catalogue document."""

    def __init__(self, document: dict, source: Path | None = None):
        if document.get("schema") != 1:
            raise ValueError(f"Unsupported catalogue schema: {document.get('schema')!r}")
        self.source = source
        self.generated_at: str = document.get("generated_at", "")
        self._libraries: dict[str, LibraryInfo] = {}
        self._entries: list[CatalogEntry] = []
        self._by_library: dict[tuple[str, str], list[CatalogEntry]] = {}
        self._by_nuclide: dict[str, list[CatalogEntry]] = {}
        self._load(document)

    def _load(self, document: dict) -> None:
        for lib in document["libraries"]:
            directory = lib["dir"]
            lib_id = _library_id(directory)
            counts: dict[str, int] = {}
            latest = ""
            for sublib, rows in lib["sublibs"].items():
                bucket: list[CatalogEntry] = []
                for filename, size, modified in rows:
                    parsed = parse_catalog_filename(filename, sublib)
                    if parsed is None:
                        continue
                    entry = CatalogEntry(
                        library=lib_id,
                        library_dir=directory,
                        sublib=sublib,
                        filename=filename,
                        size_bytes=int(size),
                        modified=modified,
                        mat=parsed["mat"],
                        z=parsed["z"],
                        a=parsed["a"],
                        isomer=parsed["isomer"],
                        label=parsed["label"],
                    )
                    bucket.append(entry)
                    key = entry.nuclide_key
                    if key is not None:
                        self._by_nuclide.setdefault(key, []).append(entry)
                    if modified > latest:
                        latest = modified
                if bucket:
                    bucket.sort(key=lambda e: (e.z or 0, e.a or 0, e.isomer, e.label))
                    self._by_library[(lib_id, sublib)] = bucket
                    self._entries.extend(bucket)
                    counts[sublib] = len(bucket)
            if counts:
                self._libraries[lib_id] = LibraryInfo(
                    id=lib_id,
                    directory=directory,
                    name=library_display_name(directory),
                    family=library_family(directory),
                    sublibs=counts,
                    latest=latest,
                )

    # -- lookups ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def libraries(self, sublib: str | None = None) -> list[LibraryInfo]:
        """Every library, or only those with the given sub-library."""
        libs: Iterable[LibraryInfo] = self._libraries.values()
        if sublib:
            libs = (lib for lib in libs if sublib in lib.sublibs)
        return sorted(libs, key=lambda lib: (lib.family, lib.directory.lower()))

    def library(self, library: str) -> LibraryInfo | None:
        return self._libraries.get(self.resolve_library(library) or "")

    def resolve_library(self, name: str) -> str | None:
        """Map a library id, alias or directory name to the canonical id."""
        key = name.strip().lower()
        if key in self._libraries:
            return key
        if key in LIBRARY_ALIASES and LIBRARY_ALIASES[key] in self._libraries:
            return LIBRARY_ALIASES[key]
        for lib in self._libraries.values():
            if lib.directory.lower() == key:
                return lib.id
        return None

    def sublibs(self) -> list[dict]:
        """Sub-libraries present anywhere, with display names and file counts."""
        totals: dict[str, int] = {}
        for lib in self._libraries.values():
            for sublib, n in lib.sublibs.items():
                totals[sublib] = totals.get(sublib, 0) + n
        order = list(SUBLIB_NAMES)
        return [
            {"id": s, "name": SUBLIB_NAMES.get(s, s), "files": totals[s]}
            for s in sorted(totals, key=lambda s: (order.index(s) if s in order else len(order), s))
        ]

    def entries_for_library(self, library: str, sublib: str = "n") -> list[CatalogEntry]:
        lib_id = self.resolve_library(library)
        if lib_id is None:
            return []
        return list(self._by_library.get((lib_id, sublib), []))

    def nuclides(self, sublib: str = "n", library: str | None = None) -> list[NuclideInfo]:
        """Every nuclide with at least one file in ``sublib``, sorted by Z, A."""
        lib_id = self.resolve_library(library) if library else None
        if library and lib_id is None:
            return []
        out: list[NuclideInfo] = []
        for entries in self._by_nuclide.values():
            hits = {e.library for e in entries if e.sublib == sublib and (lib_id is None or e.library == lib_id)}
            if not hits:
                continue
            first = entries[0]
            out.append(
                NuclideInfo(
                    z=first.z,  # type: ignore[arg-type]
                    a=first.a,  # type: ignore[arg-type]
                    isomer=first.isomer,
                    symbol=first.symbol or "?",
                    label=first.label,
                    libraries=len(hits),
                )
            )
        out.sort(key=lambda n: (n.z, n.a, n.isomer))
        return out

    def entries(
        self,
        nuclide: str | int,
        sublib: str | None = "n",
        library: str | None = None,
    ) -> list[CatalogEntry]:
        """Every file evaluating ``nuclide`` (``"Fe56"``, ``"Fe-56"``, ``26056``,
        ``"Am242m"``), grouped by family and ordered by library name."""
        from .iaea_client import parse_isotope_state

        z, a, _symbol, isomer = parse_isotope_state(nuclide)
        hits = self._by_nuclide.get(f"{z}-{a}-{isomer}", [])
        if sublib:
            hits = [e for e in hits if e.sublib == sublib]
        if library:
            lib_id = self.resolve_library(library)
            hits = [e for e in hits if e.library == lib_id]
        return sorted(hits, key=lambda e: (library_family(e.library_dir), e.library_dir.lower(), e.sublib))

    def find(self, library: str, nuclide: str | int, sublib: str = "n") -> CatalogEntry | None:
        hits = self.entries(nuclide, sublib=sublib, library=library)
        return hits[0] if hits else None

    def find_file(self, library: str, sublib: str, filename: str) -> CatalogEntry | None:
        for entry in self.entries_for_library(library, sublib):
            if entry.filename == filename:
                return entry
        return None

    def search(self, query: str, sublib: str = "n", limit: int = 50) -> list[NuclideInfo]:
        """Prefix match on nuclide labels, ZAIDs and element symbols."""
        q = query.strip().lower().replace(" ", "")
        if not q:
            return []
        out = []
        for n in self.nuclides(sublib):
            label = n.label.lower()
            if (
                label.startswith(q)
                or label.replace("-", "").startswith(q)
                or str(n.zaid).startswith(q)
                or n.symbol.lower() == q
            ):
                out.append(n)
                if len(out) >= limit:
                    break
        return out


# -- loading ---------------------------------------------------------------

_catalog: Catalog | None = None
_lock = threading.Lock()


def read_catalog_file(path: Path) -> dict:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def catalog_paths() -> list[Path]:
    """Candidate catalogue files, most preferred first: a refreshed copy in
    the cache directory, then the package's own slot."""
    return [get_cache_dir() / CACHE_CATALOG_NAME, PACKAGE_CATALOG]


def load_catalog(path: Path | None = None) -> Catalog:
    """Parse a catalogue file (default: the best available one)."""
    candidates = [Path(path)] if path else catalog_paths()
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return Catalog(read_catalog_file(candidate), source=candidate)
        except (OSError, ValueError, KeyError) as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise FileNotFoundError(
        "no ENDF catalogue on this machine. It is not shipped with kika -- 2 MB "
        "of scraped IAEA directory listings -- so it is built once, into "
        f"{get_cache_dir() / CACHE_CATALOG_NAME}, by calling "
        "kika.endf.remote.catalog.refresh_catalog() or running "
        "`python -m kika.scripts.build_endf_catalog`. It needs the network and "
        "takes about half a minute"
    )


def get_catalog() -> Catalog:
    """The process-wide catalogue, parsed on first use."""
    global _catalog
    if _catalog is None:
        with _lock:
            if _catalog is None:
                _catalog = load_catalog()
    return _catalog


def reset_catalog() -> None:
    """Forget the parsed catalogue so the next call re-reads it from disk."""
    global _catalog
    with _lock:
        _catalog = None


def refresh_catalog(
    libraries: Iterable[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Rebuild the catalogue from the live IAEA listings into the cache
    directory and make it the active one."""
    from .catalog_build import build_catalog

    dest = get_cache_dir() / CACHE_CATALOG_NAME
    build_catalog(dest, list(libraries) if libraries else None, progress=progress)
    reset_catalog()
    return dest
