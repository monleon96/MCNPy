"""Tests for the MF33 covariance writer.

Mirrors ``test_mf34_split.py``: builds a small relative covariance, serializes
via ``create_mf33_from_covariance`` + ``str()``, parses it back with
``parse_mf33_mt``, and checks the round-trip recovers the input matrix.  Also
covers the shared LB=5/LB=6 record builders and the insert-before-MEND file
writer.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.writers.mf33_writer import (
    create_mf33_from_covariance,
    write_mf33_to_file,
)
from kika.endf.writers._records import populate_lb5_record, populate_lb6_record
from kika.endf.classes.mf33.mf33 import NISubSubsectionRecord
from kika.endf.parsers.parse_mf33 import parse_mf33_mt
from kika.endf.utils import format_endf_data_line, ENDF_FORMAT_INT


ZA, AWR, MAT, MT = 26056.0, 55.454, 2631, 2


def _reference_cov(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic n×n relative covariance and matching (n+1) grid."""
    grid = np.array([1.0e6 * (i + 1) for i in range(n + 1)], dtype=float)
    base = np.arange(1, n + 1, dtype=float)
    cov = 0.001 * np.outer(base, base) + np.diag(0.01 * (base + 1.0))
    cov = 0.5 * (cov + cov.T)
    return cov, grid


# ---------- shared record builders ----------


def test_populate_lb5_record_fields():
    """Known 3x3 matrix → LS/LB/NE/NT and upper-triangle packing."""
    mat = np.array([
        [1.0, 0.2, 0.3],
        [0.2, 2.0, 0.4],
        [0.3, 0.4, 3.0],
    ])
    grid = [1.0, 2.0, 3.0, 4.0]
    rec = populate_lb5_record(NISubSubsectionRecord(), mat, grid)

    assert rec.ls == 1 and rec.lb == 5 and rec.ne == 4
    assert rec.energies == grid
    assert rec.matrix == [1.0, 0.2, 0.3, 2.0, 0.4, 3.0]
    assert rec.nt == 4 + 6


def test_populate_lb6_record_fields():
    """Rectangular 2x3 block → NER/NEC/rect packing and NT."""
    mat = np.arange(6, dtype=float).reshape(2, 3)
    row_grid = [1.0, 2.0, 3.0]     # 2 intervals
    col_grid = [1.0, 2.0, 3.0, 4.0]  # 3 intervals
    rec = populate_lb6_record(NISubSubsectionRecord(), mat, row_grid, col_grid)

    assert rec.ls == 0 and rec.lb == 6
    assert rec.row_energies == row_grid and rec.col_energies == col_grid
    assert rec.rect_matrix == list(range(6))
    assert rec.nt == len(row_grid) + len(col_grid) + 2 * 3


def test_populate_lb5_shape_mismatch():
    mat = np.eye(3)
    grid = [1.0, 2.0]  # 1 interval vs 3x3
    with pytest.raises(ValueError, match="doesn't match"):
        populate_lb5_record(NISubSubsectionRecord(), mat, grid)


# ---------- create_mf33_from_covariance round-trip ----------


def test_mf33_roundtrip_recovers_matrix():
    """str() → parse_mf33_mt → to_xs_covmat recovers the input covariance."""
    cov, grid = _reference_cov()
    mf33 = create_mf33_from_covariance(cov, grid, ZA, AWR, MAT, MT)

    parsed = parse_mf33_mt(str(mf33).split("\n"), MT)
    M = np.asarray(parsed.to_xs_covmat().matrices[0])

    assert M.shape == cov.shape
    np.testing.assert_allclose(M, cov, atol=1e-9)


def test_mf33_header_fields():
    """One subsection, self-pair, LB=5 record present."""
    cov, grid = _reference_cov()
    mf33 = create_mf33_from_covariance(cov, grid, ZA, AWR, MAT, MT)

    assert mf33._mf == 33 and mf33.number == MT and mf33._nl == 1
    assert len(mf33._subsections) == 1
    sub = mf33._subsections[0]
    assert sub.mt1 == MT and sub.nc == 0 and sub.ni == 1
    assert sub.ni_records[0].lb == 5 and sub.ni_records[0].ls == 1


def test_mf33_lb6_exposed():
    """LB=6 rectangular path builds a valid record (magnitude ≠ shape grid)."""
    row_grid = np.array([1.0e6, 2.0e6, 3.0e6])       # 2 intervals
    col_grid = np.array([1.0e6, 2.0e6, 3.0e6, 4.0e6])  # 3 intervals
    cov = np.arange(6, dtype=float).reshape(2, 3) * 0.001
    mf33 = create_mf33_from_covariance(
        cov, row_grid, ZA, AWR, MAT, MT, mt1=MT, lb=6, col_energy_grid_ev=col_grid,
    )
    rec = mf33._subsections[0].ni_records[0]
    assert rec.lb == 6
    assert rec.row_energies == list(row_grid)
    assert rec.col_energies == list(col_grid)


def test_mf33_rejects_bad_shape():
    cov = np.eye(4)
    grid = np.array([1.0e6, 2.0e6, 3.0e6, 4.0e6])  # 3 intervals vs 4x4
    with pytest.raises(ValueError, match="doesn't match"):
        create_mf33_from_covariance(cov, grid, ZA, AWR, MAT, MT)


def test_mf33_rejects_nonfinite():
    cov, grid = _reference_cov()
    cov[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        create_mf33_from_covariance(cov, grid, ZA, AWR, MAT, MT)


# ---------- write_mf33_to_file (insert before MEND) ----------


def _minimal_template(path):
    """A tiny ENDF template: one MF3 MT2 line, its FEND, and MEND."""
    def line(values, mat, mf, mt):
        return format_endf_data_line(
            values, mat, mf, mt, 0, formats=[ENDF_FORMAT_INT] * 6
        )
    lines = [
        line([26056, 0, 0, 0, 0, 0], MAT, 3, MT),   # MF3/MT2 stub
        line([0, 0, 0, 0, 0, 0], MAT, 3, 0),          # SEND
        line([0, 0, 0, 0, 0, 0], MAT, 0, 0),          # FEND
        line([0, 0, 0, 0, 0, 0], 0, 0, 0),            # MEND
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_write_mf33_inserts_before_mend(tmp_path):
    """Insert an MF33 into a file lacking one; re-parse recovers the matrix."""
    cov, grid = _reference_cov()
    mf33 = create_mf33_from_covariance(cov, grid, ZA, AWR, MAT, MT)

    src = _minimal_template(tmp_path / "template.endf")
    out = tmp_path / "with_mf33.endf"
    write_mf33_to_file(str(src), mf33, str(out), update_directory=False)

    text = out.read_text()
    # MF33 lines carry "33" in the MF column (positions 71-72, 1-indexed).
    mf33_lines = [ln for ln in text.splitlines() if len(ln) >= 72 and ln[70:72] == "33"]
    assert mf33_lines, "no MF33 lines written"

    parsed = parse_mf33_mt(mf33_lines, MT)
    M = np.asarray(parsed.to_xs_covmat().matrices[0])
    np.testing.assert_allclose(M, cov, atol=1e-9)
