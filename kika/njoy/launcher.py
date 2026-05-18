"""Build the OS-level command line that launches NJOY.

On Linux/macOS, and on Windows with a native ``njoy.exe``, this is just
``[executable]``.  On Windows where the user has configured KIKA with a
path *into* WSL — e.g. ``\\\\wsl.localhost\\IRSN-Debian\\home\\you\\NJOY2016\\build\\njoy``
or the legacy ``\\\\wsl$\\<distro>\\...`` form — the file at that path is
a Linux ELF binary that Windows cannot execute directly (``CreateProcess``
fails with ``WinError 193 / "%1 is not a valid Win32 application"``).
In that case we re-route the spawn through ``wsl.exe -d <distro> -- <linux-path>``
so the binary runs under WSL while still being driven by the Windows
parent process (stdin/stdout/stderr and cwd are forwarded by WSL).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional


_WSL_UNC_RE = re.compile(
    r"^[\\/]{2}wsl(?:\.localhost|\$)[\\/]+([^\\/]+)[\\/]+(.+)$",
    re.IGNORECASE,
)


def _parse_wsl_unc(executable: str) -> Optional[tuple]:
    """If ``executable`` is a WSL UNC path, return ``(distro, linux_path)``.

    Accepts both ``\\\\wsl.localhost\\<distro>\\...`` and ``\\\\wsl$\\<distro>\\...``
    (case-insensitive).  Returns ``None`` for any other input.
    """
    m = _WSL_UNC_RE.match(executable)
    if not m:
        return None
    distro = m.group(1)
    linux_path = "/" + m.group(2).replace("\\", "/").lstrip("/")
    return distro, linux_path


def is_wsl_unc_path(executable: str | os.PathLike) -> bool:
    """Return ``True`` if ``executable`` is a ``\\\\wsl.localhost\\...`` path."""
    return _parse_wsl_unc(str(executable)) is not None


def build_njoy_command(
    executable: str | os.PathLike,
    *,
    forward_env: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return the argv list to hand to ``subprocess.Popen`` / ``subprocess.run``.

    - Native path (Linux/macOS, or Windows ``njoy.exe``): ``[executable]``.
    - WSL UNC on Windows: ``["wsl.exe", "-d", <distro>, "--", *env_prefix, <linux-path>]``
      where ``env_prefix`` is ``["env", "K=V", ...]`` built from ``forward_env``.
      ``forward_env`` is used because Windows environment variables set on
      the wsl.exe process are *not* automatically visible to the Linux child
      (that would require ``WSLENV``).  Injecting ``env VAR=value`` inside
      the WSL command guarantees they reach NJOY.
    """
    exe_str = str(executable)
    if sys.platform == "win32":
        parsed = _parse_wsl_unc(exe_str)
        if parsed is not None:
            distro, linux_path = parsed
            cmd: List[str] = ["wsl.exe", "-d", distro, "--"]
            if forward_env:
                cmd.append("env")
                for key, value in forward_env.items():
                    cmd.append(f"{key}={value}")
            cmd.append(linux_path)
            return cmd
    return [exe_str]


__all__ = ["build_njoy_command", "is_wsl_unc_path"]
