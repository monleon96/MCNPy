"""
EXFOR module configuration.

This module provides session-level configuration for the EXFOR module,
allowing users to set default paths once instead of passing them to every function.

Example:
    >>> import kika.exfor as exfor
    >>> exfor.configure(db_path="C:/path/to/x4sqlite1.db")
    >>>
    >>> # Now all calls use that path automatically
    >>> data = exfor.read_all_exfor(target="Fe56", mt=2)
"""

import os
from pathlib import Path
from typing import Optional

from kika.exfor._constants import DB_PATH_ENV_VAR

# Default path for TOF metadata file (next to this module). Curated per-EXFOR
# flight path / time resolution, keyed by dataset ID, nested-schema format.
_DEFAULT_TOF_METADATA_PATH = Path(__file__).parent / "exfor_tof_parameters.json"

# Module-level configuration storage
_config = {
    "db_path": None,
    "tof_metadata_path": None,
}


def configure(
    db_path: Optional[str] = None,
    tof_metadata_path: Optional[str] = None,
) -> None:
    """
    Configure default settings for the EXFOR module.

    Settings persist for the duration of the Python session. Call this once
    at the start of your script or notebook.

    Parameters
    ----------
    db_path : str, optional
        Path to the X4Pro SQLite database. Once set, all functions that
        accept db_path will use this as the default.
    tof_metadata_path : str, optional
        Path to the TOF metadata JSON file. This file contains flight path
        and time resolution parameters for experiments not in the database.
        If not set, uses the default file in kika/exfor/tof_metadata.json.

    Examples
    --------
    >>> import kika.exfor as exfor
    >>> exfor.configure(db_path="C:/Data/x4sqlite1.db")
    >>>
    >>> # Now these work without specifying db_path
    >>> data = exfor.read_all_exfor(target="Fe56", mt=2)
    >>> db = exfor.X4ProDatabase()
    """
    if db_path is not None:
        _config["db_path"] = db_path
    if tof_metadata_path is not None:
        _config["tof_metadata_path"] = tof_metadata_path


def get_config() -> dict:
    """
    Get current EXFOR module configuration.

    Returns
    -------
    dict
        Current configuration settings
    """
    return _config.copy()


def get_db_path(explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Get the database path to use, with fallback chain.

    Priority:
    1. Explicitly passed path (if not None)
    2. Module configuration (set via configure())
    3. Environment variable KIKA_X4PRO_DB_PATH
    4. None (will cause error in calling code)

    Parameters
    ----------
    explicit_path : str, optional
        Path explicitly passed to a function

    Returns
    -------
    str or None
        The database path to use
    """
    if explicit_path is not None:
        return explicit_path

    if _config["db_path"] is not None:
        return _config["db_path"]

    env_path = os.environ.get(DB_PATH_ENV_VAR)
    if env_path is not None:
        return env_path

    return None


def get_tof_metadata_path(explicit_path: Optional[str] = None) -> str:
    """
    Get the TOF metadata file path to use, with fallback chain.

    Priority:
    1. Explicitly passed path (if not None)
    2. Module configuration (set via configure())
    3. Default file in kika/exfor/tof_metadata.json

    Parameters
    ----------
    explicit_path : str, optional
        Path explicitly passed to a function

    Returns
    -------
    str
        The TOF metadata file path to use
    """
    if explicit_path is not None:
        return explicit_path

    if _config["tof_metadata_path"] is not None:
        return _config["tof_metadata_path"]

    return str(_DEFAULT_TOF_METADATA_PATH)


def reset_config() -> None:
    """Reset configuration to defaults."""
    _config["db_path"] = None
    _config["tof_metadata_path"] = None
