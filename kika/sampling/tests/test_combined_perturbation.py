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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing defect, pinned by GNDS phase 0 and to be fixed in phase 1. "
        "endf_perturbation.py:133 builds the output directory from "
        "read_endf(f, mf_numbers=[1]).zaid, but a targeted MF1-only parse leaves "
        "zaid=None (a full parse gives 26056), so Stage A.MF34 writes to "
        "endf/unknown/<sample>/ while Stage B looks under endf/<zaid>/<sample>/ "
        "(combined_perturbation.py:384) and pairs nothing — n_attempted is 0. "
        "The comment at endf_perturbation.py:128 claims MF1 is enough to recover "
        "the ZAID; it is not. This test never ran before the root conftest "
        "resolved its tape, which is why the defect went unnoticed. "
        "Separately, the artifact paths asserted below ('mf4/endf/...') do not "
        "match what either stage uses."
    ),
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
