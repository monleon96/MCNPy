"""
Benchmarks module configuration.

Session-level configuration for the ICSBEP/DICE benchmarks subpackage, mirroring
the pattern used by ``kika.exfor.config``. Set the database path once instead of
passing it to every function.

Example:
    >>> import kika.benchmarks as benchmarks
    >>> benchmarks.configure(db_path="/path/to/kika_benchmarks.db")
    >>> hits = benchmarks.find_sensitive_benchmarks("Fe-56", reaction="elastic",
    ...                                              energy_region="fast")
"""

import os
from typing import Optional

from kika.benchmarks._constants import (
    DB_PATH_ENV_VAR,
    DEFAULT_DB_PATH,
    SOURCE_DIR_ENV_VAR,
)

# Module-level configuration storage.
_config = {
    "db_path": None,
    "source_dir": None,
}


def configure(db_path: Optional[str] = None, source_dir: Optional[str] = None) -> None:
    """
    Configure default settings for the benchmarks module.

    Settings persist for the duration of the Python session.

    Parameters
    ----------
    db_path : str, optional
        Path to the built ``kika_benchmarks.db`` SQLite database. Once set, all
        functions that accept ``db_path`` use this as the default.
    source_dir : str, optional
        Path to the raw DICE ``sensitivity/`` folder, used only when building the
        database via :func:`kika.benchmarks.build_benchmarks_db`.
    """
    if db_path is not None:
        _config["db_path"] = db_path
    if source_dir is not None:
        _config["source_dir"] = source_dir


def get_config() -> dict:
    """Return a copy of the current benchmarks module configuration."""
    return _config.copy()


def get_db_path(explicit_path: Optional[str] = None) -> str:
    """
    Resolve the database path to use.

    Priority:
    1. Explicitly passed path (if not None)
    2. Module configuration (set via :func:`configure`)
    3. Environment variable ``KIKA_BENCHMARKS_DB_PATH``
    4. Default ``~/.kika/benchmarks/kika_benchmarks.db``

    Parameters
    ----------
    explicit_path : str, optional
        Path explicitly passed to a function.

    Returns
    -------
    str
        The database path to use.
    """
    if explicit_path is not None:
        return explicit_path
    if _config["db_path"] is not None:
        return _config["db_path"]
    env_path = os.environ.get(DB_PATH_ENV_VAR)
    if env_path is not None:
        return env_path
    return str(DEFAULT_DB_PATH)


def get_source_dir(explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Resolve the DICE source directory to use for ingest.

    Priority: explicit arg > configured > env ``KIKA_DICE_SOURCE_DIR`` > None.
    """
    if explicit_path is not None:
        return explicit_path
    if _config["source_dir"] is not None:
        return _config["source_dir"]
    return os.environ.get(SOURCE_DIR_ENV_VAR)


def reset_config() -> None:
    """Reset configuration to defaults."""
    _config["db_path"] = None
    _config["source_dir"] = None
