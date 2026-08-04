"""Byte-identical writer regression gate.

Phase-1 acceptance criterion: the covariance writers must produce byte-identical
output before and after dormant-capability changes.  These tests build small,
fully deterministic reference sections and compare ``str(section)`` against a
committed baseline fixture.  Any change to the default (no-cross-block) writer
output — intended or not — trips this gate.

To (re)generate the baselines after an *intentional* format change, run::

    REGEN_WRITER_BASELINE=1 pytest kika/endf/tests/test_writer_regression.py

and commit the updated files under ``data/``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from kika.endf.writers.mf34_writer import create_mf34_from_covariance
from kika.endf.writers.mf33_writer import create_mf33_from_covariance


DATA = Path(__file__).resolve().parent / "data"
ZA, AWR, MAT, MT = 26056.0, 55.454, 2631, 2


def _reference_mf34_str() -> str:
    """Deterministic MF34: 3 energy intervals, orders a_1..a_2 (LTT=1)."""
    grid = np.array([0.85e6, 1.5e6, 2.5e6, 4.0e6], dtype=float)
    max_order = 2
    n = (len(grid) - 1) * max_order  # 6
    base = np.arange(1, n + 1, dtype=float)
    cov = 0.0005 * np.outer(base, base) + np.diag(0.01 * base)
    cov = 0.5 * (cov + cov.T)
    mf34 = create_mf34_from_covariance(
        cov, grid, max_order=max_order, za=ZA, awr=AWR, mat=MAT, mt=MT,
    )
    return str(mf34)


def _reference_mf33_str() -> str:
    """Deterministic MF33 self-covariance: 3 energy intervals, LB=5."""
    grid = np.array([0.85e6, 1.5e6, 2.5e6, 4.0e6], dtype=float)
    n = len(grid) - 1
    base = np.arange(1, n + 1, dtype=float)
    cov = 0.001 * np.outer(base, base) + np.diag(0.02 * base)
    cov = 0.5 * (cov + cov.T)
    mf33 = create_mf33_from_covariance(cov, grid, ZA, AWR, MAT, MT)
    return str(mf33)


_CASES = {
    "mf34_writer_baseline.endf": _reference_mf34_str,
    "mf33_writer_baseline.endf": _reference_mf33_str,
}


def _check(filename: str, generator) -> None:
    produced = generator()
    path = DATA / filename
    if os.environ.get("REGEN_WRITER_BASELINE"):
        DATA.mkdir(exist_ok=True)
        path.write_text(produced)
        return
    assert path.is_file(), (
        f"missing baseline {path}; regenerate with "
        f"REGEN_WRITER_BASELINE=1 pytest {__file__}"
    )
    assert produced == path.read_text(), (
        f"{filename} changed — writer output is no longer byte-identical to the "
        f"committed baseline. If this change is intentional, regenerate with "
        f"REGEN_WRITER_BASELINE=1 and commit the new baseline."
    )


def test_mf34_writer_byte_identical():
    _check("mf34_writer_baseline.endf", _reference_mf34_str)


def test_mf33_writer_byte_identical():
    _check("mf33_writer_baseline.endf", _reference_mf33_str)
