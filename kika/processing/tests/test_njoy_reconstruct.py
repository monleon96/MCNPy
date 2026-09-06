"""Integration tests for :func:`kika.processing.njoy_reconstruct`.

These tests spawn the NJOY executable so they require a local NJOY install.
They skip gracefully when NJOY (or the reference ENDF files) are not
available, which keeps CI green on machines without NJOY while still
guarding the wrapper on development workstations.

The ``njoy_exe``, ``fe56_host_tape`` and ``u238_tape`` fixtures come from the
root ``conftest.py``; see there for ``$KIKA_TAPES``, ``$NJOY_EXECUTABLE`` and
the ``--deep`` flag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.processing.njoy_reconstruct import (
    NjoyReconstructError,
    njoy_reconstruct,
    njoy_reconstruct_stream,
)


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


def test_fe56_reconstruction(njoy_exe: Path, fe56_host_tape: Path) -> None:
    """Fe-56 JEFF-4.0: reconr auto-relaxes tolerance on ill-behaved threshold."""
    result = njoy_reconstruct(fe56_host_tape, njoy_exe, tolerance=1e-3)

    _assert_reasonable_xs(result, expected_mts=(1, 2, 102))

    # Thermal capture should be non-trivial (Fe-56 thermal σ_γ ≈ 2.6 b).
    thermal_cap = _interp(result[102], 0.0253)
    assert 1.0 < thermal_cap < 5.0, f"Fe-56 thermal σ_γ unreasonable: {thermal_cap} b"


def test_u238_reconstruction(njoy_exe: Path, u238_tape: Path) -> None:
    """U-238 JEFF-4.0: the motivating regression test."""
    result = njoy_reconstruct(u238_tape, njoy_exe, tolerance=1e-3)

    _assert_reasonable_xs(result, expected_mts=(1, 2, 18, 102))

    # Thermal capture ≈ 2.68 b (well known).
    thermal_cap = _interp(result[102], 0.0253)
    assert 2.3 < thermal_cap < 3.1, f"U-238 thermal σ_γ off: {thermal_cap} b"

    # Peak of the 6.674 eV capture resonance: look in a narrow window.
    mask = (result[102].energies >= 6.5) & (result[102].energies <= 6.9)
    peak = float(result[102].values[mask].max()) if mask.any() else 0.0
    assert peak > 1.0e4, f"U-238 6.674 eV capture peak missing/too small: {peak} b"


def test_missing_njoy_executable(fe56_host_tape: Path) -> None:
    """Graceful error when the NJOY path is wrong."""
    with pytest.raises(NjoyReconstructError) as excinfo:
        njoy_reconstruct(fe56_host_tape, "/nonexistent/path/njoy", tolerance=1e-3)
    assert "not found" in str(excinfo.value).lower()


def test_missing_endf(njoy_exe: Path) -> None:
    """Graceful error when the ENDF input does not exist."""
    with pytest.raises(NjoyReconstructError):
        njoy_reconstruct("/no/such/file.endf", njoy_exe, tolerance=1e-3)


def test_fe56_streaming(njoy_exe: Path, fe56_host_tape: Path) -> None:
    """Streaming variant emits log lines before the final result event."""
    endf = fe56_host_tape
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


# ---------------------------------------------------------------------------
# Header scan (no NJOY needed)
# ---------------------------------------------------------------------------

def _record(mat: int, mf: int, mt: int, text: str = "") -> str:
    """One 80-column ENDF record: 66 columns of payload, then MAT/MF/MT/NS."""
    return f"{text:<66}{mat:4d}{mf:2d}{mt:3d}{1:5d}\n"


def test_mat_from_tape_header_reads_first_material(tmp_path: Path) -> None:
    from kika.processing.njoy_reconstruct import _mat_from_tape_header

    tape = tmp_path / "n-026_Fe_056.endf"
    tape.write_text(
        _record(7777, 0, 0, "tape identification line")
        + _record(2631, 1, 451, " 2.605600+4 5.545443+1          1          0          0          1")
        + _record(2631, 1, 451, " 0.000000+0 1.000000+0          0          0          0          6"),
        encoding="utf-8",
    )
    assert _mat_from_tape_header(tape) == 2631


def test_mat_from_tape_header_ignores_tape_id_without_mf1(tmp_path: Path) -> None:
    from kika.processing.njoy_reconstruct import _mat_from_tape_header

    tape = tmp_path / "odd.endf"
    # A TPID line with a positive MAT must not be mistaken for the material.
    tape.write_text(
        _record(9999, 0, 0, "tape identification line")
        + "not an ENDF record at all\n",
        encoding="utf-8",
    )
    assert _mat_from_tape_header(tape) is None


def test_mat_from_tape_header_missing_file(tmp_path: Path) -> None:
    from kika.processing.njoy_reconstruct import _mat_from_tape_header

    assert _mat_from_tape_header(tmp_path / "nope.endf") is None
