"""
SINBAD module configuration.

Session-level configuration for the SINBAD shielding-benchmark subpackage,
mirroring the pattern used by :mod:`kika.benchmarks.config` and
:mod:`kika.exfor.config`. Point kika at a folder of packages once instead of
passing a path to every call.

Example
-------
    >>> import kika.sinbad as sinbad
    >>> sinbad.configure(path="/path/to/sinbad-packages")
    >>> sinbad.list_benchmarks()
    ['SINBAD-ASPIS-IRON88']
"""

import os
from typing import Optional

from kika.sinbad._constants import DEFAULT_LIBRARY_PATH, LIBRARY_PATH_ENV_VAR

# Module-level configuration storage.
_config = {
    "path": None,
}


def configure(path: Optional[str] = None) -> None:
    """
    Configure default settings for the SINBAD module.

    Settings persist for the duration of the Python session.

    Parameters
    ----------
    path : str, optional
        Directory holding SINBAD packages, in either form -- ``*.sinbad``
        archives, package directories, or a mixture. Once set, functions that
        accept ``path`` use this as the default.
    """
    if path is not None:
        _config["path"] = os.path.expanduser(path)


def get_config() -> dict:
    """Return a copy of the current SINBAD module configuration."""
    return _config.copy()


def get_library_path(explicit_path: Optional[str] = None) -> str:
    """
    Resolve the package library directory to use.

    Priority:

    1. Explicitly passed path (if not None)
    2. Module configuration (set via :func:`configure`)
    3. Environment variable ``KIKA_SINBAD_PATH``
    4. Default ``~/.kika/sinbad``

    Parameters
    ----------
    explicit_path : str, optional
        Path explicitly passed to a function.

    Returns
    -------
    str
        The library directory to use.
    """
    if explicit_path is not None:
        return os.path.expanduser(explicit_path)
    if _config["path"] is not None:
        return _config["path"]
    env_path = os.environ.get(LIBRARY_PATH_ENV_VAR)
    if env_path is not None:
        return os.path.expanduser(env_path)
    return str(DEFAULT_LIBRARY_PATH)


def reset_config() -> None:
    """Reset configuration to defaults."""
    _config["path"] = None
