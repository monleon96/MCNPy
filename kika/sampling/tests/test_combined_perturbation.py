"""Validation + integration tests for ``perturb_ENDF_PENDF_combined``.

Validation tests run without NJOY. The integration smoke test pairs MF34 ENDF
with MF33 PENDF perturbation and runs NJOY ACER per pair. It is skipped when
NJOY or the reference ENDF file are absent.

Honours the same env vars as ``test_pendf_perturbation.py``:
  - ``NJOY_EXECUTABLE`` — absolute path to a working NJOY binary.
  - ``KIKA_ENDF_FILES`` — directory holding ``Fe56_jeff4.0_n.endf``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

from kika.endf import read_endf
from kika.sampling.combined_perturbation import perturb_ENDF_PENDF_combined


_DEFAULT_ENDF_DIR = Path(__file__).resolve().parents[3] / "files" / "endf"
_DEFAULT_NJOY_PATHS = [
    Path("/home/MONLEON-JUAN/NJOY2016/build/njoy"),
    Path("/usr/local/bin/njoy"),
    Path("/opt/njoy2016/njoy"),
]


def _resolve_njoy() -> Optional[Path]:
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
def fe56_endf() -> Path:
    d = _resolve_endf_dir()
    candidate = d / "Fe56_jeff4.0_n.endf"
    if not candidate.is_file():
        pytest.skip(f"Reference ENDF not found: {candidate}")
    return candidate


def test_combined_requires_njoy_when_generate_ace(tmp_path):
    with pytest.raises(ValueError, match="njoy_exe"):
        perturb_ENDF_PENDF_combined(
            endf_files="/nonexistent.endf",
            mt_list_mf33=[2],
            mt_list_mf34=[2],
            legendre_coeffs=[1],
            num_samples=1,
            generate_ace=True,
            njoy_exe=None,
            output_dir=str(tmp_path),
        )


def test_combined_smoke(tmp_path, njoy_exe, fe56_endf):
    """End-to-end: produce 2 paired (ENDF, PENDF, ACE) samples."""
    output_dir = tmp_path / "combined_smoke"
    summary = perturb_ENDF_PENDF_combined(
        endf_files=str(fe56_endf),
        mt_list_mf33=[2, 102],
        mt_list_mf34=[2],
        legendre_coeffs=[1, 2],
        num_samples=2,
        generate_ace=True,
        njoy_exe=str(njoy_exe),
        ace_temperatures=[293.6],
        ace_library_name="jeff40",
        output_dir=str(output_dir),
        seed=42,
        nprocs=1,
        verbose_diagnostics=0,
    )
    assert "isotopes" in summary
    assert "mf33_error" not in summary, summary.get("mf33_error")
    assert "mf34_error" not in summary, summary.get("mf34_error")
    iso_summary = next(iter(summary["isotopes"].values()))
    assert iso_summary["n_attempted"] == 2
    assert iso_summary["n_ace_ok"] >= 1, summary

    # Verify intermediate artifacts exist
    zaid = int(read_endf(str(fe56_endf)).zaid)
    base, ext = os.path.splitext(os.path.basename(str(fe56_endf)))
    for s in (1, 2):
        sample_str = f"{s:04d}"
        endf_path = output_dir / "mf4" / "endf" / str(zaid) / sample_str / f"{base}_{sample_str}{ext}"
        pendf_path = output_dir / "mf3" / "pendf" / str(zaid) / sample_str / f"{base}_{sample_str}.pendf"
        assert endf_path.exists(), f"missing perturbed ENDF: {endf_path}"
        assert pendf_path.exists(), f"missing perturbed PENDF: {pendf_path}"

    # ACE landed under output_dir/ace/<library>/<TK>/
    ace_glob = list((output_dir / "ace").rglob("*.*c"))
    assert ace_glob, "no ACE files produced by Stage B"
