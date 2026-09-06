"""
ENDF module for reading and working with Evaluated Nuclear Data Files.

This module provides functionality for:
- Reading local ENDF files
- Downloading ENDF files from IAEA Nuclear Data Service
- Caching downloaded files locally
"""
from .read_endf import (
    read_endf,
    read_mt451,
    read_mf2,
    read_mf3_mt,
    read_mf4_mt,
    read_mf7_mt,
)
from .classes.mf7.scatterer import ThermalScatterer, thermal_scatterer
from . import dcs
from .remote import (
    fetch_endf,
    download_endf,
    list_available_libraries,
    get_cache_info,
    clear_cache,
    ENDFRemoteError,
    IsotopeNotFoundError,
    LibraryNotFoundError,
    NetworkError,
    CacheError,
)

__all__ = [
    # Differential cross sections from MF4 + MF3 (angular reconstruction, the
    # elastic frame transform, and the three readings of sigma(E))
    "dcs",
    # Local file reading
    "read_endf",
    "read_mt451",
    "read_mf2",
    "read_mf3_mt",
    "read_mf4_mt",
    "read_mf7_mt",
    # Thermal scattering identity
    "ThermalScatterer",
    "thermal_scatterer",
    # Remote download
    "fetch_endf",
    "download_endf",
    "list_available_libraries",
    "get_cache_info",
    "clear_cache",
    # Exceptions
    "ENDFRemoteError",
    "IsotopeNotFoundError",
    "LibraryNotFoundError",
    "NetworkError",
    "CacheError",
]
