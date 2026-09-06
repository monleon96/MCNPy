"""
Remote ENDF file download functionality.

This module provides functions to download ENDF neutron data files directly
from the IAEA Nuclear Data Service, with local caching support.

Example usage:

    from kika.endf import fetch_endf, download_endf

    # Download Fe-56 from ENDF/B-VIII.1 (default)
    endf = fetch_endf("Fe56")

    # Download U-235 from JEFF-4.0
    endf = fetch_endf("U235", library="jeff4.0")

    # Using ZAID format
    endf = fetch_endf(26056)

    # Force re-download (ignore cache)
    endf = fetch_endf("Fe56", force_download=True)

    # Save to a specific location (also parses and returns ENDF object)
    endf = fetch_endf("Fe56", save_to="./data/Fe56_endfb81.endf")

    # Just download without parsing (returns the file path)
    path = download_endf("Fe56", save_to="./data/Fe56.endf")

Cache structure:
    ~/.kika/endf_cache/           # Default (override with KIKA_ENDF_CACHE_DIR env var)
        {library}/
            {particle}/
                {zaid}.endf       # e.g., endfb8.1/n/26056.endf
        cache_metadata.json
"""

import shutil
import tempfile
from pathlib import Path

from ..read_endf import read_endf
from .cache import clear_cache, get_cache, get_cache_info
from .catalog import (
    Catalog,
    CatalogEntry,
    LibraryInfo,
    NuclideInfo,
    get_catalog,
    refresh_catalog,
)
from .constants import list_available_libraries
from .exceptions import (
    CacheError,
    ENDFRemoteError,
    IsotopeNotFoundError,
    LibraryNotFoundError,
    NetworkError,
)
from .iaea_client import (
    IAEAClient,
    get_client,
    isotope_key,
    parse_isotope,
    parse_isotope_state,
)


def _generate_filename(isotope: str | int, library: str, particle: str = "n") -> str:
    """Generate a descriptive filename for an ENDF file."""
    from .constants import normalize_library_name

    _z, a, symbol, isomer = parse_isotope_state(isotope)
    canonical_lib = normalize_library_name(library)
    state = "m" if isomer else ""
    # e.g., "Fe56_endfb8.1_n.endf"
    return f"{symbol}{a}{state}_{canonical_lib}_{particle}.endf"


def download_endf(
    isotope: str | int,
    save_to: str | Path,
    library: str = "endfb8.1",
    particle: str = "n",
    force_download: bool = False,
    timeout: float = 30.0,
) -> Path:
    """
    Download an ENDF file and save it to a specific location.

    This function downloads the file without parsing it, useful when you
    want to save the raw ENDF file for external use or editing.

    Parameters
    ----------
    isotope : str or int
        Isotope specification (e.g., "Fe56", 26056, "U-235")
    save_to : str or Path
        Path where to save the file. Can be:
        - A full file path: "./data/Fe56.endf"
        - A directory: "./data/" (filename will be auto-generated)
    library : str
        Nuclear data library (default: "endfb8.1")
    particle : str
        Incident particle type (default: "n")
    force_download : bool
        Force re-download even if cached (default: False)
    timeout : float
        Request timeout in seconds (default: 30.0)

    Returns
    -------
    Path
        Path to the saved file

    Examples
    --------
    >>> from kika.endf import download_endf
    >>> path = download_endf("Fe56", "./my_data/Fe56.endf")
    >>> path = download_endf("U235", "./data/", library="jeff4.0")  # Auto-generates filename
    """
    from .constants import normalize_library_name

    save_path = Path(save_to)

    # If save_to is a directory, generate filename
    if save_path.is_dir() or str(save_to).endswith(("/", "\\")):
        save_path.mkdir(parents=True, exist_ok=True)
        filename = _generate_filename(isotope, library, particle)
        save_path = save_path / filename
    else:
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse isotope to get the cache key
    try:
        zaid = isotope_key(isotope)
    except ValueError as e:
        raise IsotopeNotFoundError(str(isotope), library) from e

    # Normalize library name
    try:
        canonical_lib = normalize_library_name(library)
    except KeyError:
        raise LibraryNotFoundError(library, list_available_libraries())

    # Check if we have it cached already
    cache_instance = get_cache()
    if not force_download:
        cached_path = cache_instance.get(zaid, canonical_lib, particle)
        if cached_path:
            # Copy from cache to destination
            shutil.copy2(cached_path, save_path)
            return save_path

    # Download the file
    client = IAEAClient(timeout=timeout)
    content = client.download(
        isotope=isotope,
        library=library,
        particle=particle,
        use_cache=True,  # Still cache it for future use
        force_download=force_download,
    )

    # Write to destination
    save_path.write_bytes(content)
    return save_path


def cached_entry_path(entry: CatalogEntry) -> Path | None:
    """Where the cache keeps *entry*, or None if it has not been fetched."""
    return get_cache().get(entry.cache_key, entry.library, entry.sublib)


def download_entry(
    entry: CatalogEntry,
    save_to: str | Path | None = None,
    force_download: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Fetch one catalogue entry into the cache and return the cached path.

    This is the catalogue-first counterpart of :func:`download_endf`: it takes
    the exact file the IAEA lists (any library, any sub-library, thermal
    scattering included) and does not need a MAT table to find it. The cached
    file is the deliverable — an app can point at it directly and never copy
    the bytes — and ``save_to`` additionally writes a copy where asked.

    Examples
    --------
    >>> from kika.endf.remote import get_catalog, download_entry
    >>> entry = get_catalog().find("jeff4.0", "Fe56")
    >>> path = download_entry(entry)
    """
    client = IAEAClient(timeout=timeout)
    client.download_entry(entry, use_cache=True, force_download=force_download)
    cached = client.cached_path(entry)
    if cached is None:  # pragma: no cover - the put above makes this unreachable
        raise CacheError(f"{entry.filename} did not land in the cache")
    if save_to is not None:
        save_path = Path(save_to)
        if save_path.is_dir() or str(save_to).endswith(("/", "\\")):
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / entry_filename(entry)
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, save_path)
    return cached


def entry_filename(entry: CatalogEntry) -> str:
    """A descriptive filename for a catalogue entry, e.g. ``Fe56_jeff4.0_n.endf``."""
    if entry.z is not None and entry.a is not None:
        state = "m" if entry.isomer else ""
        stem = f"{entry.symbol}{entry.a}{state}"
    else:
        stem = Path(entry.filename).stem
    return f"{stem}_{entry.library}_{entry.sublib}.endf"


def fetch_endf(
    isotope: str | int,
    library: str = "endfb8.1",
    particle: str = "n",
    cache: bool = True,
    force_download: bool = False,
    timeout: float = 30.0,
    save_to: str | Path | None = None,
):
    """
    Download and parse an ENDF file from IAEA Nuclear Data Service.

    Parameters
    ----------
    isotope : str or int
        Isotope specification. Supported formats:
        - Element + mass: "Fe56", "Fe-56", "U235", "U-235"
        - ZAID: 26056, "26056"
    library : str
        Nuclear data library. Flexible naming is supported:
        - "endfb8.1", "endfb81", "ENDF/B-VIII.1" (default: "endfb8.1")
        - "jeff4.0", "jeff40", "JEFF-4.0"
        - "jendl5", "jendl50", "JENDL-5"
        - "tendl2023", "tendl", "TENDL-2023"
        See list_available_libraries() for all options.
    particle : str
        Incident particle type (default: "n" for neutron)
    cache : bool
        Whether to use local caching (default: True)
    force_download : bool
        Force re-download even if cached (default: False)
    timeout : float
        Request timeout in seconds (default: 30.0)
    save_to : str or Path, optional
        Additionally save the file to this location. Can be:
        - A full file path: "./data/Fe56.endf"
        - A directory: "./data/" (filename will be auto-generated)
        The file is saved in addition to normal caching behavior.

    Returns
    -------
    ENDF
        Parsed ENDF object

    Raises
    ------
    LibraryNotFoundError
        If the library name is not recognized
    IsotopeNotFoundError
        If the isotope is not found in the specified library
    NetworkError
        If the download fails due to network issues

    Examples
    --------
    >>> from kika.endf import fetch_endf
    >>> endf = fetch_endf("Fe56")
    >>> endf = fetch_endf(26056, library="jeff4.0")
    >>> endf = fetch_endf("U-235", library="endfb8.1", force_download=True)
    >>> # Download, parse, AND save to a specific location
    >>> endf = fetch_endf("Fe56", save_to="./data/Fe56.endf")
    """
    from .constants import normalize_library_name

    # Create client with specified timeout
    client = IAEAClient(timeout=timeout)

    # Parse isotope to get the cache key
    try:
        zaid = isotope_key(isotope)
    except ValueError as e:
        raise IsotopeNotFoundError(str(isotope), library) from e

    # Normalize library name
    try:
        canonical_lib = normalize_library_name(library)
    except KeyError:
        raise LibraryNotFoundError(library, list_available_libraries())

    # Check if we can use the cached file directly (avoids reading bytes into memory)
    cache_instance = get_cache()
    if cache and not force_download:
        cached_path = cache_instance.get(zaid, canonical_lib, particle)
        if cached_path:
            # If save_to requested, copy the cached file there
            if save_to:
                save_path = Path(save_to)
                if save_path.is_dir() or str(save_to).endswith(("/", "\\")):
                    save_path.mkdir(parents=True, exist_ok=True)
                    filename = _generate_filename(isotope, library, particle)
                    save_path = save_path / filename
                else:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_path, save_path)
            return read_endf(str(cached_path))

    # Download the file (this will also cache it)
    content = client.download(
        isotope=isotope,
        library=library,
        particle=particle,
        use_cache=cache,
        force_download=force_download,
    )

    # If save_to requested, save the content there
    if save_to:
        save_path = Path(save_to)
        if save_path.is_dir() or str(save_to).endswith(("/", "\\")):
            save_path.mkdir(parents=True, exist_ok=True)
            filename = _generate_filename(isotope, library, particle)
            save_path = save_path / filename
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(content)

    # If caching is enabled, the file is now in the cache, use it
    if cache:
        cached_path = cache_instance.get(zaid, canonical_lib, particle)
        if cached_path:
            return read_endf(str(cached_path))

    # Fallback: write to temp file and parse
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".endf", delete=False
    ) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        return read_endf(tmp_path)
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


__all__ = [
    # Main functions
    "fetch_endf",
    "download_endf",
    "download_entry",
    "cached_entry_path",
    "entry_filename",
    # Catalogue
    "get_catalog",
    "refresh_catalog",
    "Catalog",
    "CatalogEntry",
    "LibraryInfo",
    "NuclideInfo",
    # Utility functions
    "list_available_libraries",
    "get_cache_info",
    "clear_cache",
    # Exceptions
    "ENDFRemoteError",
    "IsotopeNotFoundError",
    "LibraryNotFoundError",
    "NetworkError",
    "CacheError",
    # For advanced usage
    "IAEAClient",
    "parse_isotope",
    "parse_isotope_state",
]
