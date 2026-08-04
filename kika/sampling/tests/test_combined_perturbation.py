"""Validation + integration tests for ``perturb_ENDF_PENDF_combined``.

Validation tests run without NJOY. The integration smoke test pairs MF34 ENDF
with MF33 PENDF perturbation and runs NJOY ACER per pair. It is skipped when
NJOY or the reference ENDF file are absent.

The ``njoy_exe`` and ``fe56_host_tape`` fixtures come from the root
``conftest.py``; see there for ``$KIKA_TAPES``, ``$NJOY_EXECUTABLE`` and
``--deep``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from kika.endf import read_endf
from kika.sampling.combined_perturbation import perturb_ENDF_PENDF_combined


@pytest.fixture(scope="module")
def fe56_endf(fe56_host_tape: Path) -> Path:
    """Alias kept so the test bodies below read unchanged."""
    return fe56_host_tape


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

    # Verify intermediate artifacts exist, at the flat layout the module
    # docstring documents: output_dir/{endf,pendf}/<zaid>/<NNNN>/. The paths
    # asserted here used to be output_dir/mf4/endf/... and output_dir/mf3/
    # pendf/..., which match neither stage — they never ran, because the tape
    # was unreachable until the root conftest resolved it.
    zaid = int(read_endf(str(fe56_endf)).zaid)
    base, ext = os.path.splitext(os.path.basename(str(fe56_endf)))
    for s in (1, 2):
        sample_str = f"{s:04d}"
        endf_path = output_dir / "endf" / str(zaid) / sample_str / f"{base}_{sample_str}{ext}"
        pendf_path = output_dir / "pendf" / str(zaid) / sample_str / f"{base}_{sample_str}.pendf"
        assert endf_path.exists(), f"missing perturbed ENDF: {endf_path}"
        assert pendf_path.exists(), f"missing perturbed PENDF: {pendf_path}"

    # The defect this test was pinned for: stage A must not write to
    # endf/unknown/, which is what a zaid of None produced.
    assert not (output_dir / "endf" / "unknown").exists(), (
        "stage A fell back to endf/unknown/ — the targeted parse lost its MAT again"
    )

    # ACE landed under output_dir/ace/<library>/<TK>/
    ace_glob = list((output_dir / "ace").rglob("*.*c"))
    assert ace_glob, "no ACE files produced by Stage B"
