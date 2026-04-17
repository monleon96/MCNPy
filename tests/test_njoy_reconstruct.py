"""Integration tests for :func:`kika.processing.njoy_reconstruct`.

These tests spawn the NJOY executable so they require a local NJOY install.
They skip gracefully when NJOY (or the reference ENDF files) are not
available, which keeps CI green on machines without NJOY while still
guarding the wrapper on development workstations.

Environment variables honoured:

- ``NJOY_EXECUTABLE`` — absolute path to a working NJOY binary.
- ``KIKA_ENDF_FILES`` — directory containing the reference ENDF files
  (``Fe56_jeff4.0_n.endf``, ``U238_jeff4.0_n.endf``).  Defaults to
  ``<repo>/files/endf``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kika.processing.njoy_reconstruct import (
    NjoyReconstructError,
    njoy_reconstruct,
    njoy_reconstruct_stream,
)

_DEFAULT_ENDF_DIR = Path(__file__).resolve().parent.parent / "files" / "endf"
_DEFAULT_NJOY_PATHS = [
    Path(r"C:\Users\Usuario\BaradDur\Codes\NJOY2016\build\njoy.exe"),
    Path("/usr/local/bin/njoy"),
    Path("/opt/njoy2016/njoy"),
]


def _resolve_njoy() -> Path | None:
    env = os.environ.get("NJOY_EXECUTABLE")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    for candidate in _DEFAULT_NJOY_PATHS:
        if candidate.is_file():
            return candidate
    return None


def _resolve_endf_dir() -> Path:
    env = os.environ.get("KIKA_ENDF_FILES")
    return Path(env) if env else _DEFAULT_ENDF_DIR


@pytest.fixture(scope="module")
def njoy_exe() -> Path:
    exe = _resolve_njoy()
    if exe is None:
        pytest.skip("NJOY executable not found; set NJOY_EXECUTABLE to run")
    return exe


@pytest.fixture(scope="module")
def endf_dir() -> Path:
    d = _resolve_endf_dir()
    if not d.is_dir():
        pytest.skip(f"ENDF reference directory not found: {d}")
    return d


def _assert_reasonable_xs(result, expected_mts):
    assert isinstance(result, dict) and result, "Empty reconstruction result"
    for mt in expected_mts:
        assert mt in result, f"Expected MT={mt} in result, got {sorted(result.keys())}"
        xs = result[mt]
        assert xs.energies.size > 100, f"MT={mt}: grid too small ({xs.energies.size})"
        assert xs.energies.size == xs.values.size
        # Strictly monotone energy grid (reconr output is linearised).
        assert np.all(np.diff(xs.energies) > 0), f"MT={mt}: grid not strictly increasing"
        # Non-negative, finite cross sections.
        assert np.all(np.isfinite(xs.values)), f"MT={mt}: non-finite values"
        assert np.all(xs.values >= 0), f"MT={mt}: negative cross sections"


def _interp(xs, E: float) -> float:
    """Log-linear interpolation at energy *E*."""
    return float(np.interp(E, xs.energies, xs.values))


def test_fe56_reconstruction(njoy_exe: Path, endf_dir: Path) -> None:
    """Fe-56 JEFF-4.0: reconr auto-relaxes tolerance on ill-behaved threshold."""
    endf = endf_dir / "Fe56_jeff4.0_n.endf"
    if not endf.is_file():
        pytest.skip(f"{endf.name} not available")

    result = njoy_reconstruct(endf, njoy_exe, tolerance=1e-3)

    _assert_reasonable_xs(result, expected_mts=(1, 2, 102))

    # Thermal capture should be non-trivial (Fe-56 thermal σ_γ ≈ 2.6 b).
    thermal_cap = _interp(result[102], 0.0253)
    assert 1.0 < thermal_cap < 5.0, f"Fe-56 thermal σ_γ unreasonable: {thermal_cap} b"


def test_u238_reconstruction(njoy_exe: Path, endf_dir: Path) -> None:
    """U-238 JEFF-4.0: the motivating regression test."""
    endf = endf_dir / "U238_jeff4.0_n.endf"
    if not endf.is_file():
        pytest.skip(f"{endf.name} not available")

    result = njoy_reconstruct(endf, njoy_exe, tolerance=1e-3)

    _assert_reasonable_xs(result, expected_mts=(1, 2, 18, 102))

    # Thermal capture ≈ 2.68 b (well known).
    thermal_cap = _interp(result[102], 0.0253)
    assert 2.3 < thermal_cap < 3.1, f"U-238 thermal σ_γ off: {thermal_cap} b"

    # Peak of the 6.674 eV capture resonance: look in a narrow window.
    mask = (result[102].energies >= 6.5) & (result[102].energies <= 6.9)
    peak = float(result[102].values[mask].max()) if mask.any() else 0.0
    assert peak > 1.0e4, f"U-238 6.674 eV capture peak missing/too small: {peak} b"


def test_missing_njoy_executable(endf_dir: Path) -> None:
    """Graceful error when the NJOY path is wrong."""
    endf = endf_dir / "Fe56_jeff4.0_n.endf"
    if not endf.is_file():
        pytest.skip(f"{endf.name} not available")

    with pytest.raises(NjoyReconstructError) as excinfo:
        njoy_reconstruct(endf, "/nonexistent/path/njoy", tolerance=1e-3)
    assert "not found" in str(excinfo.value).lower()


def test_missing_endf(njoy_exe: Path) -> None:
    """Graceful error when the ENDF input does not exist."""
    with pytest.raises(NjoyReconstructError):
        njoy_reconstruct("/no/such/file.endf", njoy_exe, tolerance=1e-3)


def test_fe56_streaming(njoy_exe: Path, endf_dir: Path) -> None:
    """Streaming variant emits log lines before the final result event."""
    endf = endf_dir / "Fe56_jeff4.0_n.endf"
    if not endf.is_file():
        pytest.skip(f"{endf.name} not available")

    log_lines: list[str] = []
    result_payload = None
    pendf_path_seen = None
    saw_result_last = False

    for kind, payload in njoy_reconstruct_stream(endf, njoy_exe, tolerance=1e-3):
        if kind == "log":
            log_lines.append(payload)
            saw_result_last = False
        elif kind == "warning":
            # tolerance relaxation is acceptable; just record it
            saw_result_last = False
        elif kind == "pendf_path":
            # The raw tape22 path is surfaced just before the final
            # result event so consumers can copy the file out of the
            # generator's TemporaryDirectory before it's cleaned up.
            pendf_path_seen = payload
            saw_result_last = False
        elif kind == "result":
            result_payload = payload
            saw_result_last = True
        else:
            pytest.fail(f"Unexpected event kind: {kind}")

    assert log_lines, "Expected at least one log line from NJOY"
    assert any("reconr" in ln.lower() for ln in log_lines), (
        "Expected 'reconr' banner in the NJOY log"
    )
    assert saw_result_last, "'result' must be the terminal event"
    assert pendf_path_seen is not None, "Expected a pendf_path event before result"
    _assert_reasonable_xs(result_payload, expected_mts=(1, 2, 102))
