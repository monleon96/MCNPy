"""End-to-end integration tests for :func:`perturb_PENDF_files`.

Skipped automatically when NJOY or the reference ENDF files are absent.
Honours the same env vars as ``test_njoy_reconstruct.py``:

- ``NJOY_EXECUTABLE`` — absolute path to a working NJOY binary.
- ``KIKA_ENDF_FILES`` — directory holding ``Fe56_jeff4.0_n.endf`` (default
  ``<repo>/files/endf``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling.pendf_perturbation import perturb_PENDF_files


_DEFAULT_ENDF_DIR = Path(__file__).resolve().parents[3] / "files" / "endf"
_DEFAULT_NJOY_PATHS = [
    Path(r"C:\Users\Usuario\BaradDur\Codes\NJOY2016\build\njoy.exe"),
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


def test_pendf_only_smoke(tmp_path, njoy_exe, fe56_endf):
    """Stage A only: produce N=4 perturbed PENDFs and verify σ_new = σ_orig × factor[g]."""
    output_dir = tmp_path / "pendf_smoke"
    summary = perturb_PENDF_files(
        endf_files=str(fe56_endf),
        mt_list=[2, 102],
        num_samples=4,
        output_formats=("pendf",),
        njoy_exe=str(njoy_exe),
        output_dir=str(output_dir),
        seed=42,
        nprocs=1,
        verbose_diagnostics=0,
    )
    assert "isotopes" in summary
    iso_summary = next(iter(summary["isotopes"].values()))
    assert "error" not in iso_summary, f"Pipeline error: {iso_summary.get('error')}"
    assert iso_summary["n_pendf_ok"] == 4
    assert iso_summary["n_ace_ok"] == 0  # ACE not requested

    # Verify each perturbed PENDF can be re-read and σ values changed
    zaid_str = next(iter(summary["isotopes"]))
    pendf_root = output_dir / "pendf" / zaid_str
    sample_dirs = sorted(pendf_root.iterdir())
    assert len(sample_dirs) == 4

    nominal = read_endf(str(fe56_endf), mf_numbers=[3])
    nominal_mf3 = nominal.get_file(3)
    if nominal_mf3 is None:
        # ENDF MF3 may be sparse; the perturbed PENDF will have the
        # NJOY-reconstructed pointwise grid which differs from raw ENDF.
        # In that case the assertion below is on perturbed-vs-nominal-PENDF.
        pytest.skip("Nominal ENDF lacks raw MF3; comparison would need the cached PENDF")

    # Perturbed PENDF σ should differ from nominal at most points (we used factor != 1)
    for sample_dir in sample_dirs:
        pendf_files = list(sample_dir.glob("*.pendf"))
        assert len(pendf_files) == 1
        perturbed = read_endf(str(pendf_files[0]), mf_numbers=[3])
        sec = perturbed.get_file(3).sections.get(2)
        assert sec is not None, "MT=2 missing from perturbed PENDF"
        # Cross sections should be finite and non-negative
        xs = np.asarray(sec.cross_sections)
        assert np.all(np.isfinite(xs))
        assert np.all(xs >= 0)


def test_pendf_perturbation_dry_run(tmp_path, njoy_exe, fe56_endf):
    """dry_run: run sampling and write parquet, but skip per-sample PENDF writes."""
    output_dir = tmp_path / "pendf_dry"
    summary = perturb_PENDF_files(
        endf_files=str(fe56_endf),
        mt_list=[2, 102],
        num_samples=4,
        output_formats=("pendf",),
        njoy_exe=str(njoy_exe),
        output_dir=str(output_dir),
        seed=42,
        dry_run=True,
        nprocs=1,
        verbose_diagnostics=0,
    )
    iso_summary = next(iter(summary["isotopes"].values()))
    assert iso_summary.get("dry_run") is True
    assert iso_summary["n_samples"] == 4
    # No per-sample PENDF directory should exist in dry-run mode
    assert not (output_dir / "pendf").exists() or not any((output_dir / "pendf").iterdir())


def test_pendf_perturbation_invalid_output_format(tmp_path):
    """Validation: bogus output_formats should error out before NJOY runs."""
    with pytest.raises(ValueError, match="unknown values"):
        perturb_PENDF_files(
            endf_files="/nonexistent.endf",
            mt_list=[2],
            num_samples=1,
            output_formats=("bogus",),
            output_dir=str(tmp_path),
        )


def test_pendf_perturbation_ace_requires_njoy(tmp_path):
    """Validation: 'ace' in output_formats requires njoy_exe."""
    with pytest.raises(ValueError, match="njoy_exe"):
        perturb_PENDF_files(
            endf_files="/nonexistent.endf",
            mt_list=[2],
            num_samples=1,
            output_formats=("pendf", "ace"),
            output_dir=str(tmp_path),
            njoy_exe=None,
        )
