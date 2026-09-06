"""Build the IAEA ``download-endf`` catalogue by walking the server's directory
listings.

The IAEA serves every evaluated library as a plain Apache index at
``https://nds.iaea.org/public/download-endf/<LIBRARY>/<SUBLIB>/``, one file per
material. Walking those listings once gives, for every library and every
sub-library (incident particle, decay, fission yields, thermal scattering...),
the exact filename the server holds together with its size and date. That is
what :mod:`kika.endf.remote.catalog` reads: it turns the raw listing into a
nuclide-first index and hands :class:`~kika.endf.remote.iaea_client.IAEAClient`
real URLs instead of guessed ones.

The scraper is deliberately dumb: it records every ``.zip``/``.dat`` in every
directory that is not a known backup or archive folder, and leaves deciding
what a filename *means* to the catalogue parser. A parser fix therefore never
needs a fresh scrape.

Run it through ``python -m kika.scripts.build_endf_catalog`` to regenerate the
cached snapshot, or through :func:`kika.endf.remote.catalog.refresh_catalog`
to drop an up-to-date copy into the user's cache directory.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from .constants import IAEA_BASE_URL

logger = logging.getLogger(__name__)

CATALOG_SCHEMA = 1

# Apache index rows: href, "last modified" and size cells.
_ROW = re.compile(
    r'<a href="(?P<href>[^"?]+)">[^<]*</a></td>'
    r'<td align="right">\s*(?P<date>\d{4}-\d{2}-\d{2})[^<]*</td>'
    r'<td align="right">\s*(?P<size>[^<]*?)\s*</td>'
)
_DIR_HREF = re.compile(r'<a href="(?P<href>[^"?/][^"]*/)">')

# Directories that hold whole-library tapes, previous revisions or tooling
# rather than one file per material. Everything else is recorded and the
# catalogue parser decides file by file.
_SKIP_DIRS = {
    "backup",
    "_backup-by-nsub",
    "orig",
    "original",
    "endf-all-zip",
    "sublib",
    "nmt",  # TENDL-2010 MT-split variant of the neutron sub-library
}
_SKIP_LIBS = {
    "tools",
    "endf-manual",
    "xtool",
    "zv_vms_archive",
    "w3000",
    "dxs",
    "llcrp",
    "medical",
    "sigmacalcdata-2013",
}

_SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}


def parse_apache_size(text: str) -> int:
    """Turn an Apache index size cell (``226K``, ``1.5M``, ``807``) into bytes."""
    text = text.strip()
    if not text or text == "-":
        return 0
    match = re.fullmatch(r"([\d.]+)\s*([KMG]?)", text)
    if not match:
        return 0
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2)])


def parse_listing(html: str) -> tuple[list[str], list[tuple[str, int, str]]]:
    """Split an Apache index page into ``(subdirectories, files)``.

    Files come back as ``(name, size_bytes, modified)`` with the date as
    ``YYYY-MM-DD``.
    """
    dirs = [m.group("href").rstrip("/") for m in _DIR_HREF.finditer(html)]
    files: list[tuple[str, int, str]] = []
    for m in _ROW.finditer(html):
        href = m.group("href")
        if href.endswith("/") or href.startswith("/"):
            continue
        files.append((href, parse_apache_size(m.group("size")), m.group("date")))
    return dirs, files


class CatalogBuilder:
    """Walk the IAEA tree and assemble the raw catalogue document."""

    def __init__(
        self,
        base_url: str = IAEA_BASE_URL,
        timeout: float = 90.0,
        workers: int = 8,
        progress: Callable[[str], None] | None = None,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.workers = workers
        self.progress = progress or (lambda _msg: None)
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "kika-nd catalogue builder (+https://kika-app.com)"},
        )

    def _get(self, url: str, attempts: int = 4) -> str:
        # The TENDL listings run to 700 kB and the server is not always quick
        # about them; a timeout on one directory should not sink the whole scan.
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.get(url)
                response.raise_for_status()
                return response.text
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == attempts:
                    raise
                logger.warning("%s: %s (attempt %d/%d)", url, exc, attempt, attempts)
                time.sleep(2.0 * attempt)
        raise AssertionError("unreachable")

    def list_libraries(self) -> list[str]:
        dirs, _ = parse_listing(self._get(self.base_url))
        return [d for d in dirs if d.lower() not in _SKIP_LIBS]

    def scan_library(self, library_dir: str) -> dict:
        """Return ``{"dir": ..., "sublibs": {name: [[file, size, date], ...]}}``."""
        dirs, _top_files = parse_listing(self._get(f"{self.base_url}{library_dir}/"))
        sublibs: dict[str, list[list]] = {}
        for sub in dirs:
            if sub.lower() in _SKIP_DIRS:
                continue
            try:
                _, files = parse_listing(self._get(f"{self.base_url}{library_dir}/{sub}/"))
            except httpx.HTTPError as exc:
                logger.warning("Skipping %s/%s: %s", library_dir, sub, exc)
                continue
            rows = [
                [name, size, date]
                for name, size, date in files
                if name.lower().endswith((".zip", ".dat"))
            ]
            if rows:
                sublibs[sub] = rows
        self.progress(f"{library_dir}: {sum(len(v) for v in sublibs.values())} files")
        return {"dir": library_dir, "sublibs": sublibs}

    def build(self, libraries: list[str] | None = None) -> dict:
        """Assemble the catalogue document for ``libraries`` (default: all)."""
        libs = libraries or self.list_libraries()
        self.progress(f"{len(libs)} libraries to scan")
        with ThreadPoolExecutor(self.workers) as pool:
            scanned = list(pool.map(self.scan_library, libs))
        scanned = [lib for lib in scanned if lib["sublibs"]]
        scanned.sort(key=lambda lib: lib["dir"].lower())
        return {
            "schema": CATALOG_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_url": self.base_url.rstrip("/"),
            "libraries": scanned,
        }

    def close(self) -> None:
        self._client.close()


def write_catalog(document: dict, path: Path) -> Path:
    """Write the catalogue document, gzip-compressed, to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(payload)
    return path


def build_catalog(
    path: Path,
    libraries: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
    workers: int = 8,
) -> Path:
    """Scrape the IAEA tree and write the gzipped catalogue to ``path``."""
    builder = CatalogBuilder(workers=workers, progress=progress)
    try:
        document = builder.build(libraries)
    finally:
        builder.close()
    return write_catalog(document, path)
