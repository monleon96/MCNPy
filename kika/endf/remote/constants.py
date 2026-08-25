"""Constants for IAEA ENDF download functionality."""

import os
from pathlib import Path

# IAEA Nuclear Data Service base URL
IAEA_BASE_URL = "https://nds.iaea.org/public/download-endf"

# Default cache directory (can be overridden via environment variable)
DEFAULT_CACHE_DIR = Path.home() / ".kika" / "endf_cache"
CACHE_DIR_ENV_VAR = "KIKA_ENDF_CACHE_DIR"


def get_cache_dir() -> Path:
    """Get the cache directory, respecting environment variable override."""
    env_path = os.environ.get(CACHE_DIR_ENV_VAR)
    if env_path:
        return Path(env_path)
    return DEFAULT_CACHE_DIR


# Library mappings: canonical name -> IAEA path
LIBRARY_PATHS = {
    "endfb8.1": "ENDF-B-VIII.1",
    "endfb8.0": "ENDF-B-VIII.0",
    "endfb7.1": "ENDF-B-VII.1",
    "endfb7.0": "ENDF-B-VII.0",
    "jeff4.0": "JEFF-4.0",
    "jeff3.3": "JEFF-3.3",
    "jeff3.2": "JEFF-3.2",
    "jeff3.1.1": "JEFF-3.1.1",
    "jendl5": "JENDL-5",
    "jendl4.0": "JENDL-4.0",
    "tendl2023": "TENDL-2023",
    "tendl2021": "TENDL-2021",
    "cendl3.2": "CENDL-3.2",
}

# IAEA serves the files under two different naming conventions, and which one a
# library uses is not derivable from anything else -- it tracks when the release
# was put online:
#   "za_mat"  ->  n_092-U-235_9228.zip   (Z zero-padded to 3, MAT to 4)
#   "mat_za"  ->  n_9228_92-U-235.zip    (MAT zero-padded to 4, Z bare)
# Anything not listed here defaults to "za_mat".
LIBRARY_FILENAME_STYLES = {
    "endfb8.1": "za_mat",
    "endfb8.0": "mat_za",
    "endfb7.1": "mat_za",
    "endfb7.0": "mat_za",
    "jeff4.0": "za_mat",
    "jeff3.3": "mat_za",
    "jeff3.2": "mat_za",
    "jeff3.1.1": "mat_za",
    "jendl5": "za_mat",
    "jendl4.0": "mat_za",
    "tendl2023": "za_mat",
    "tendl2021": "za_mat",
    "cendl3.2": "za_mat",
}

# Library aliases for flexible naming
# Maps various user inputs to canonical names
LIBRARY_ALIASES = {
    # ENDF/B-VIII.1
    "endfb81": "endfb8.1",
    "endf/b-viii.1": "endfb8.1",
    "endfb-8.1": "endfb8.1",
    "endf-b-viii.1": "endfb8.1",
    # ENDF/B-VIII.0
    "endfb80": "endfb8.0",
    "endf/b-viii.0": "endfb8.0",
    "endfb-8.0": "endfb8.0",
    "endf-b-viii.0": "endfb8.0",
    # ENDF/B-VII.1
    "endfb71": "endfb7.1",
    "endf/b-vii.1": "endfb7.1",
    "endfb-7.1": "endfb7.1",
    "endf-b-vii.1": "endfb7.1",
    # ENDF/B-VII.0
    "endfb70": "endfb7.0",
    "endf/b-vii.0": "endfb7.0",
    "endfb-7.0": "endfb7.0",
    "endf-b-vii.0": "endfb7.0",
    # JEFF-4.0
    "jeff40": "jeff4.0",
    "jeff-4.0": "jeff4.0",
    # JEFF-3.3
    "jeff33": "jeff3.3",
    "jeff-3.3": "jeff3.3",
    # JEFF-3.2
    "jeff32": "jeff3.2",
    "jeff-3.2": "jeff3.2",
    # JEFF-3.1.1
    "jeff311": "jeff3.1.1",
    "jeff-3.1.1": "jeff3.1.1",
    # JENDL-5
    "jendl50": "jendl5",
    "jendl-5": "jendl5",
    # JENDL-4.0
    "jendl4": "jendl4.0",
    "jendl40": "jendl4.0",
    "jendl-4.0": "jendl4.0",
    # TENDL-2023
    "tendl": "tendl2023",
    "tendl-2023": "tendl2023",
    # TENDL-2021
    "tendl21": "tendl2021",
    "tendl-2021": "tendl2021",
    # CENDL-3.2
    "cendl": "cendl3.2",
    "cendl32": "cendl3.2",
    "cendl-3.2": "cendl3.2",
}


def normalize_library_name(library: str) -> str:
    """
    Normalize a library name to its canonical form.

    Parameters
    ----------
    library : str
        Library name in any supported format

    Returns
    -------
    str
        Canonical library name (e.g., "endfb8.1")

    Raises
    ------
    KeyError
        If the library name is not recognized
    """
    # Convert to lowercase for case-insensitive matching
    lib_lower = library.lower().strip()

    # Check if it's already canonical
    if lib_lower in LIBRARY_PATHS:
        return lib_lower

    # Check aliases
    if lib_lower in LIBRARY_ALIASES:
        return LIBRARY_ALIASES[lib_lower]

    # Not found
    raise KeyError(lib_lower)


def get_library_path(library: str) -> str:
    """
    Get the IAEA path for a library.

    Parameters
    ----------
    library : str
        Library name in any supported format

    Returns
    -------
    str
        IAEA path (e.g., "ENDF-B-VIII.1")
    """
    canonical = normalize_library_name(library)
    return LIBRARY_PATHS[canonical]


def get_library_filename_style(library: str) -> str:
    """
    Get the IAEA filename convention used by a library.

    Parameters
    ----------
    library : str
        Library name in any supported format

    Returns
    -------
    str
        Either ``"za_mat"`` (``n_092-U-235_9228.zip``) or ``"mat_za"``
        (``n_9228_92-U-235.zip``)
    """
    canonical = normalize_library_name(library)
    return LIBRARY_FILENAME_STYLES.get(canonical, "za_mat")


def list_available_libraries() -> list[str]:
    """
    List all available library canonical names.

    Returns
    -------
    list[str]
        List of canonical library names
    """
    return list(LIBRARY_PATHS.keys())
