"""IAEA Nuclear Data Service client for downloading ENDF files."""

import io
import re
import zipfile

import httpx

from kika._constants import (
    ATOMIC_NUMBER_TO_SYMBOL,
    SYMBOL_TO_ATOMIC_NUMBER,
    ZAID_TO_ENDF_MAT,
    ZAID_TO_ENDF_MAT_ISOMER,
)

from .cache import ENDFCache, get_cache
from .constants import (
    IAEA_BASE_URL,
    get_library_filename_style,
    get_library_path,
    list_available_libraries,
    normalize_library_name,
)
from .exceptions import (
    IsotopeNotFoundError,
    LibraryNotFoundError,
    NetworkError,
)


def parse_isotope_state(isotope: str | int) -> tuple[int, int, str, int]:
    """
    Parse an isotope specification into Z, A, element symbol and isomeric state.

    Supports formats:
    - "Fe56", "Fe-56", "fe56" (element symbol + mass)
    - "U235", "U-235", "u235"
    - "Am242m", "Am-242m", "Ag110M" (isomer, trailing m/M)
    - 26056 (ZAID integer)
    - "26056" (ZAID string)
    - 95642 (ZAID with the +400 isomer offset)

    Parameters
    ----------
    isotope : str or int
        Isotope specification

    Returns
    -------
    tuple[int, int, str, int]
        (Z, A, symbol, isomer) e.g., (26, 56, "Fe", 0) or (95, 242, "Am", 1)
    """
    if isinstance(isotope, int) or (isinstance(isotope, str) and isotope.isdigit()):
        # ZAID format: ZZZAAA, with A offset by 400 for isomers (MCNP convention)
        zaid = int(isotope)
        z = zaid // 1000
        a = zaid % 1000
        isomer = 0
        if a > 400:
            a -= 400
            isomer = 1
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z)
        if symbol is None:
            raise ValueError(f"Unknown atomic number: {z}")
        return z, a, symbol, isomer

    # String format: Element + mass (e.g., "Fe56", "Fe-56", "U235", "Am242m")
    isotope = isotope.strip()

    # Try to match element-mass pattern, with an optional isomer suffix
    match = re.fullmatch(r"([A-Za-z]+)-?(\d+)-?([mM]\d?)?", isotope)
    if match:
        symbol = match.group(1).capitalize()
        a = int(match.group(2))
        z = SYMBOL_TO_ATOMIC_NUMBER.get(symbol)
        if z is None:
            raise ValueError(f"Unknown element symbol: {symbol}")
        isomer = 0
        if match.group(3):
            isomer = int(match.group(3)[1:] or 1)
        return z, a, symbol, isomer

    raise ValueError(f"Cannot parse isotope: {isotope}")


def parse_isotope(isotope: str | int) -> tuple[int, int, str]:
    """
    Parse an isotope specification into Z, A, and element symbol.

    Isomers parse to the same triplet as their ground state; use
    :func:`parse_isotope_state` when the isomeric state matters.

    Parameters
    ----------
    isotope : str or int
        Isotope specification

    Returns
    -------
    tuple[int, int, str]
        (Z, A, symbol) e.g., (26, 56, "Fe")
    """
    z, a, symbol, _ = parse_isotope_state(isotope)
    return z, a, symbol


def isotope_key(isotope: str | int) -> int:
    """
    Build the cache key for an isotope: its ZAID, with A offset by 400 for isomers.

    Parameters
    ----------
    isotope : str or int
        Isotope specification

    Returns
    -------
    int
        e.g., 26056 for Fe-56, 95642 for Am-242m
    """
    z, a, _, isomer = parse_isotope_state(isotope)
    return z * 1000 + a + (400 if isomer else 0)


def get_endf_mat(z: int, a: int, isomer: int = 0) -> int:
    """
    Get the ENDF MAT number of a nuclide.

    Parameters
    ----------
    z : int
        Atomic number
    a : int
        Mass number
    isomer : int
        Isomeric state (0 = ground state)

    Returns
    -------
    int
        ENDF MAT number

    Raises
    ------
    ValueError
        If no MAT number is known for the nuclide
    """
    zaid = z * 1000 + a
    mat = ZAID_TO_ENDF_MAT_ISOMER.get(zaid) if isomer else ZAID_TO_ENDF_MAT.get(zaid)
    if mat is None:
        state = f" (isomeric state {isomer})" if isomer else ""
        raise ValueError(f"No MAT number found for ZAID {zaid}{state}")
    return mat


def build_iaea_url(
    z: int,
    a: int,
    symbol: str,
    library: str,
    particle: str = "n",
    isomer: int = 0,
) -> str:
    """
    Build the IAEA download URL for an ENDF file.

    URL pattern, for the two naming conventions IAEA uses (see
    :data:`~kika.endf.remote.constants.LIBRARY_FILENAME_STYLES`):

    https://nds.iaea.org/public/download-endf/{LIBRARY}/{particle}/n_{ZZZ}-{El}-{A}_{MAT}.zip
    https://nds.iaea.org/public/download-endf/{LIBRARY}/{particle}/n_{MAT}_{Z}-{El}-{A}.zip

    Parameters
    ----------
    z : int
        Atomic number
    a : int
        Mass number
    symbol : str
        Element symbol
    library : str
        Library name
    particle : str
        Particle type (default: "n")
    isomer : int
        Isomeric state (0 = ground state)

    Returns
    -------
    str
        Full download URL
    """
    library_path = get_library_path(library)
    mat = get_endf_mat(z, a, isomer)
    state = "M" if isomer else ""

    if get_library_filename_style(library) == "mat_za":
        # e.g. n_9228_92-U-235.zip
        filename = f"{particle}_{mat:04d}_{z}-{symbol}-{a}{state}.zip"
    else:
        # e.g. n_092-U-235_9228.zip
        filename = f"{particle}_{z:03d}-{symbol}-{a}{state}_{mat:04d}.zip"
    return f"{IAEA_BASE_URL}/{library_path}/{particle}/{filename}"


class IAEAClient:
    """Client for downloading ENDF files from IAEA Nuclear Data Service."""

    def __init__(
        self,
        cache: ENDFCache | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the IAEA client.

        Parameters
        ----------
        cache : ENDFCache, optional
            Cache instance. If None, uses the global cache.
        timeout : float
            Request timeout in seconds (default: 30.0)
        """
        self.cache = cache or get_cache()
        self.timeout = timeout

    def download(
        self,
        isotope: str | int,
        library: str = "endfb8.1",
        particle: str = "n",
        use_cache: bool = True,
        force_download: bool = False,
    ) -> bytes:
        """
        Download an ENDF file from IAEA.

        Parameters
        ----------
        isotope : str or int
            Isotope specification (e.g., "Fe56", 26056, "U-235")
        library : str
            Library name (default: "endfb8.1")
        particle : str
            Particle type (default: "n")
        use_cache : bool
            Whether to use local cache (default: True)
        force_download : bool
            Force re-download even if cached (default: False)

        Returns
        -------
        bytes
            ENDF file content

        Raises
        ------
        LibraryNotFoundError
            If the library is not recognized
        IsotopeNotFoundError
            If the isotope is not found in the library
        NetworkError
            If the download fails
        """
        # Validate and normalize library
        try:
            canonical_lib = normalize_library_name(library)
        except KeyError:
            raise LibraryNotFoundError(library, list_available_libraries())

        # Parse isotope
        try:
            z, a, symbol, isomer = parse_isotope_state(isotope)
        except ValueError as e:
            raise IsotopeNotFoundError(str(isotope), library) from e

        # Isomers share the ZAID of their ground state, so the cache key offsets
        # A by 400 to keep the two files apart
        zaid = z * 1000 + a + (400 if isomer else 0)

        # Check cache first
        if use_cache and not force_download:
            cached_path = self.cache.get(zaid, canonical_lib, particle)
            if cached_path:
                return cached_path.read_bytes()

        # Build URL and download
        try:
            url = build_iaea_url(z, a, symbol, canonical_lib, particle, isomer)
        except ValueError as e:
            raise IsotopeNotFoundError(str(isotope), library) from e

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
                "Accept": "application/zip, application/octet-stream, */*",
            }
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException:
            raise NetworkError(f"Request timed out after {self.timeout}s", url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise IsotopeNotFoundError(str(isotope), library) from e
            raise NetworkError(f"HTTP error {e.response.status_code}: {e}", url) from e
        except httpx.RequestError as e:
            raise NetworkError(f"Request failed: {e}", url) from e

        # Extract ENDF file from ZIP
        try:
            content = self._extract_endf_from_zip(response.content)
        except Exception as e:
            raise NetworkError(f"Failed to extract ENDF from ZIP: {e}", url) from e

        # Cache the result
        if use_cache:
            self.cache.put(zaid, canonical_lib, content, particle)

        return content

    def _extract_endf_from_zip(self, zip_content: bytes) -> bytes:
        """Extract the ENDF file from a ZIP archive."""
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            # Find the ENDF file in the archive
            for name in zf.namelist():
                # Skip directories
                if name.endswith("/"):
                    continue
                # The ENDF file is typically the only file or has no extension
                if not name.endswith((".zip", ".gz", ".tar")):
                    return zf.read(name)
            raise ValueError("No ENDF file found in ZIP archive")


# Module-level client instance
_client = None


def get_client() -> IAEAClient:
    """Get the global IAEA client instance."""
    global _client
    if _client is None:
        _client = IAEAClient()
    return _client
