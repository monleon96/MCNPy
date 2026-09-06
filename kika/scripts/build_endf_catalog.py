"""Build the IAEA catalogue snapshot into the cache directory.

    python -m kika.scripts.build_endf_catalog            # every library
    python -m kika.scripts.build_endf_catalog JEFF-4.0   # only these directories

Walks the ``download-endf`` directory listings (about 100 libraries, ~300 000
files, half a minute on a good connection) and writes the compressed snapshot
that :func:`kika.endf.remote.catalog.get_catalog` reads. Run it once per
machine, and again when the IAEA publishes a new library; the existing ones are
frozen releases and never change.

The snapshot is **not** shipped with kika -- it is 2 MB of scraped listings --
so the destination is the user's cache directory, which is also where
:func:`~kika.endf.remote.catalog.refresh_catalog` puts it. This module is the
same thing with a command line and a summary, for a machine being set up.
"""

from __future__ import annotations

import sys
import time

from kika.endf.remote.catalog import (
    CACHE_CATALOG_NAME,
    get_cache_dir,
    load_catalog,
)
from kika.endf.remote.catalog_build import build_catalog


def main(argv: list[str] | None = None) -> int:
    libraries = list(argv if argv is not None else sys.argv[1:]) or None
    destination = get_cache_dir() / CACHE_CATALOG_NAME
    started = time.time()
    path = build_catalog(destination, libraries, progress=print)
    catalog = load_catalog(path)
    print(
        f"{path}: {len(catalog)} files in {len(catalog.libraries())} libraries, "
        f"{path.stat().st_size / 1024:.0f} kB, {time.time() - started:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
