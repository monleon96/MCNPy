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
from kika.endf.utils import (
    ENDF_FORMAT_INT,
    format_endf_data_line,
    format_endf_fend_record,
    format_endf_mend_record,
    format_endf_send_record,
)


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
        format_endf_send_record(MAT, 3),
        format_endf_fend_record(MAT),
        format_endf_mend_record(),
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


# ---------- per-MT splice: sibling MT sections survive ----------


def _extract_mt_lines(text: str, mf: int, mt: int) -> list[str]:
    """All lines of one (MF, MT) section (data lines only, SEND excluded)."""
    return [
        ln for ln in text.splitlines()
        if len(ln) >= 75 and ln[70:72].strip() == str(mf) and ln[72:75].strip() == str(mt)
    ]


def _multi_mt_file(tmp_path, mts=(2, 4, 102)):
    """ENDF file with one small MF33 section per MT (distinct matrices)."""
    src = _minimal_template(tmp_path / "template.endf")
    path = tmp_path / "multi_mt.endf"
    current = str(src)
    for k, mt in enumerate(mts):
        cov, grid = _reference_cov()
        cov = cov * (k + 1)  # distinct per MT
        sec = create_mf33_from_covariance(cov, grid, ZA, AWR, MAT, mt)
        write_mf33_to_file(current, sec, str(path), update_directory=False)
        current = str(path)
    return path


def test_write_mf33_replaces_only_target_mt(tmp_path):
    """Replacing MT2 leaves sibling MF33 MT sections byte-identical."""
    path = _multi_mt_file(tmp_path)
    before = path.read_text()

    new_cov, grid = _reference_cov()
    new_cov = new_cov * 7.0
    sec = create_mf33_from_covariance(new_cov, grid, ZA, AWR, MAT, 2)
    out = tmp_path / "replaced.endf"
    write_mf33_to_file(str(path), sec, str(out), update_directory=False)
    after = out.read_text()

    # Siblings byte-identical.
    for mt in (4, 102):
        assert _extract_mt_lines(after, 33, mt) == _extract_mt_lines(before, 33, mt)
    # MT2 replaced: re-parse recovers the new matrix.
    parsed = parse_mf33_mt(_extract_mt_lines(after, 33, 2), 2)
    M = np.asarray(parsed.to_xs_covmat().matrices[0])
    np.testing.assert_allclose(M, new_cov, atol=1e-9)
    # Exactly one FEND for the whole MF33 block (MAT, MF=0 line after it).
    lines = after.splitlines()
    mf33_idx = [i for i, ln in enumerate(lines)
                if len(ln) >= 75 and ln[70:72].strip() == "33"]
    fend = lines[mf33_idx[-1] + 1]
    assert fend[66:70].strip() == str(MAT) and fend[70:72].strip() in ("0", "")


def test_write_mf33_inserts_in_mt_order(tmp_path):
    """A new MT lands between existing MTs in ascending order."""
    path = _multi_mt_file(tmp_path, mts=(2, 102))
    cov, grid = _reference_cov()
    sec = create_mf33_from_covariance(cov, grid, ZA, AWR, MAT, 4)
    out = tmp_path / "inserted.endf"
    write_mf33_to_file(str(path), sec, str(out), update_directory=False)

    seen: list[int] = []
    for ln in out.read_text().splitlines():
        if len(ln) >= 75 and ln[70:72].strip() == "33":
            mt = int(ln[72:75].strip() or "0")
            if mt and (not seen or seen[-1] != mt):
                seen.append(mt)
    assert seen == [2, 4, 102]


# ---------- range merge into the host MT2 ----------


from kika.endf.writers.mf33_writer import merge_mf33_covariance_into_host  # noqa: E402


def _host_file(tmp_path, cov=None):
    """Single full-range MF33 MT2 host (5 bins, correlated by default)."""
    grid = np.array([1e-5, 1e5, 1e6, 2e6, 1e7, 1.5e8])
    if cov is None:
        d = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        cov = np.outer(np.sqrt(d), np.sqrt(d)) * 0.5 + np.diag(d) * 0.5
        cov = 0.5 * (cov + cov.T)
    sec = create_mf33_from_covariance(cov, grid, ZA, AWR, MAT, 2)
    src = _minimal_template(tmp_path / "template.endf")
    path = tmp_path / "host.endf"
    write_mf33_to_file(str(src), sec, str(path), update_directory=False)
    return path, cov, grid


def test_merge_preserves_host_outside_range(tmp_path):
    """Host survives outside [E_lo, E_hi]; new inside; in-out cross zeroed."""
    path, host_cov, host_grid = _host_file(tmp_path)
    new_grid = np.array([1e6, 1.5e6, 2e6])
    new_cov = np.array([[0.10, 0.05], [0.05, 0.20]])

    sec = merge_mf33_covariance_into_host(str(path), new_cov, new_grid, mt=2)
    m, g, is_rel = sec._self_covariance_matrix()
    m = np.asarray(m)

    assert is_rel
    np.testing.assert_allclose(
        g, [1e-5, 1e5, 1e6, 1.5e6, 2e6, 1e7, 1.5e8], rtol=1e-12)
    # bins: 0,1 below | 2,3 new | 4,5 above.  Host-derived values round-trip
    # through the 11-char ENDF fields (~6 sig figs) → rtol 1e-5.
    np.testing.assert_allclose(np.diag(m)[:2], np.diag(host_cov)[:2], rtol=1e-5)
    np.testing.assert_allclose(m[2:4, 2:4], new_cov)
    np.testing.assert_allclose(np.diag(m)[4:], np.diag(host_cov)[3:], rtol=1e-5)
    # host below-above cross preserved, in-out cross zeroed
    np.testing.assert_allclose(m[0, 4], host_cov[0, 3], rtol=1e-5)
    np.testing.assert_allclose(m[1, 5], host_cov[1, 4], rtol=1e-5)
    assert np.all(m[2:4, :2] == 0.0) and np.all(m[2:4, 4:] == 0.0)


def test_merge_splits_straddling_host_bins(tmp_path):
    """Range boundaries inside host bins keep host values on the outside parts."""
    path, host_cov, host_grid = _host_file(tmp_path)
    new_grid = np.array([1.5e6, 3e6, 5e6])       # inside host bins 2 and 3
    new_cov = np.array([[0.11, 0.01], [0.01, 0.22]])

    sec = merge_mf33_covariance_into_host(str(path), new_cov, new_grid, mt=2)
    m, g, _ = sec._self_covariance_matrix()
    m = np.asarray(m)

    np.testing.assert_allclose(
        g, [1e-5, 1e5, 1e6, 1.5e6, 3e6, 5e6, 1e7, 1.5e8], rtol=1e-12)
    # [1e6, 1.5e6) keeps host bin-2 value; [5e6, 1e7) keeps host bin-3 value.
    np.testing.assert_allclose(m[2, 2], host_cov[2, 2], rtol=1e-5)
    np.testing.assert_allclose(m[5, 5], host_cov[3, 3], rtol=1e-5)
    np.testing.assert_allclose(m[3:5, 3:5], new_cov)


def test_merge_drops_cross_mt1_subsection_with_warning(tmp_path, caplog):
    """A host cross-reaction subsection (mt1 != mt) is dropped, loudly."""
    import logging
    from kika.endf.classes.mf33.mf33 import Subsection, NISubSubsectionRecord
    from kika.endf.writers._records import populate_lb5_record

    path, host_cov, host_grid = _host_file(tmp_path)
    # Rebuild the host with an extra mt1=4 subsection and rewrite it.
    from kika.endf import read_endf
    endf = read_endf(str(path), mf_numbers=[33])
    host_sec = endf.get_file(33).sections[2]
    rec = populate_lb5_record(
        NISubSubsectionRecord(), np.diag([0.01, 0.02]), [1e5, 1e6, 1e7])
    host_sec._subsections.append(Subsection(
        xmf1=0.0, xlfs1=0.0, mat1=0, mt1=4, nc=0, ni=1, ni_records=[rec]))
    host_sec._nl = 2
    crossed = tmp_path / "host_cross.endf"
    write_mf33_to_file(str(path), host_sec, str(crossed), update_directory=False)

    new_grid = np.array([1e6, 2e6])
    new_cov = np.array([[0.10]])
    with caplog.at_level(logging.WARNING, logger="kika.endf.writers.mf33_writer"):
        sec = merge_mf33_covariance_into_host(str(crossed), new_cov, new_grid, mt=2)
    assert any("DROPPED" in r.message for r in caplog.records)
    assert len(sec._subsections) == 1 and int(sec._subsections[0].mt1) == 2


def test_merge_raises_on_nc_records(tmp_path, monkeypatch):
    """Derived (NC) records in the host self subsection cannot be merged.

    The dummy NC record does not serialize, so the parsed section is patched
    in via the reader (the helper imports ``read_endf`` at call time from
    ``kika.endf``).
    """
    from types import SimpleNamespace
    import kika.endf as ke

    path, _, _ = _host_file(tmp_path)
    endf = ke.read_endf(str(path), mf_numbers=[33])
    host_sec = endf.get_file(33).sections[2]
    host_sec._subsections[0].nc_records = [SimpleNamespace(lty=0)]
    host_sec._subsections[0].nc = 1

    class _FakeFile:
        sections = {2: host_sec}

    class _FakeEndf:
        def get_file(self, n):
            return _FakeFile()

    monkeypatch.setattr(ke, "read_endf", lambda *a, **k: _FakeEndf())
    with pytest.raises(ValueError, match="NC-type"):
        merge_mf33_covariance_into_host(
            str(path), np.array([[0.1]]), np.array([1e6, 2e6]), mt=2)


def test_write_mf33_replaces_a_larger_section_with_a_smaller_one(tmp_path):
    """Shrinking a section must not leave orphan lines or a stale directory.

    The pipeline copies the multigroup product from the nominal file *after* the
    fine MF33 is written, so the copy carries a large fine-grid section that the
    coarse one then replaces.  Grow-then-shrink is the real write pattern.
    """
    src = _minimal_template(tmp_path / "template.endf")
    path = tmp_path / "product.endf"

    n_fine = 40
    fine_grid = np.linspace(1e6, 4e6, n_fine + 1)
    fine_cov = np.eye(n_fine) * 0.05 + 0.01
    write_mf33_to_file(
        str(src),
        create_mf33_from_covariance(fine_cov, fine_grid, ZA, AWR, MAT, MT),
        str(path),
    )
    big = len([ln for ln in path.read_text().splitlines()
               if len(ln) >= 72 and ln[70:72] == "33"])

    coarse_grid = np.array([1e6, 2e6, 4e6])
    coarse_cov = np.array([[0.09, 0.02], [0.02, 0.16]])
    write_mf33_to_file(
        str(path),
        create_mf33_from_covariance(coarse_cov, coarse_grid, ZA, AWR, MAT, MT),
        str(path),
    )

    text = path.read_text()
    mf33_lines = [ln for ln in text.splitlines()
                  if len(ln) >= 72 and ln[70:72] == "33"]
    assert len(mf33_lines) < big, "the section should have shrunk"

    parsed = parse_mf33_mt(mf33_lines, MT)
    M = np.asarray(parsed.to_xs_covmat().matrices[0])
    assert M.shape == (2, 2), "no remnant of the fine grid may survive"
    np.testing.assert_allclose(M, coarse_cov, atol=1e-9)


def test_merge_raises_named_error_on_non_finite(tmp_path):
    """A NaN matrix must raise ValueError naming the rows, not a LAPACK error.

    Before this guard, non-finite entries reached ``np.linalg.eigvalsh`` and
    surfaced as "Eigenvalues did not converge" — which says nothing about where
    the NaNs came from.  For a relative covariance a non-finite row almost
    always means the caller divided by a zero central value upstream.
    """
    path, _host_cov, _host_grid = _host_file(tmp_path)
    new_grid = np.array([1e6, 1.5e6, 2e6])
    new_cov = np.array([[0.10, 0.05], [0.05, 0.20]])
    new_cov[1, :] = np.nan
    new_cov[:, 1] = np.nan

    with pytest.raises(ValueError) as exc:
        merge_mf33_covariance_into_host(str(path), new_cov, new_grid, mt=2)

    msg = str(exc.value)
    assert "non-finite" in msg
    assert "eV" in msg, "the error should name the offending rows by energy"
    assert "refusing to write" in msg


def test_merge_accepts_zeroed_rows_where_central_was_absent(tmp_path):
    """Zeroed (not NaN) rows are legitimate and must still merge.

    Groups with no host central are zeroed upstream by the MF33 collapse; that
    is a valid, writable matrix and must not trip the finiteness guard.
    """
    path, _host_cov, _host_grid = _host_file(tmp_path)
    new_grid = np.array([1e6, 1.5e6, 2e6])
    new_cov = np.array([[0.0, 0.0], [0.0, 0.20]])

    sec = merge_mf33_covariance_into_host(str(path), new_cov, new_grid, mt=2)
    m, _g, _ = sec._self_covariance_matrix()
    np.testing.assert_allclose(np.asarray(m)[2:4, 2:4], new_cov)


def test_merge_without_host_mf33_falls_back(tmp_path):
    """No host MF33 section → the new covariance is written as-is."""
    src = _minimal_template(tmp_path / "template.endf")
    new_grid = np.array([1e6, 2e6, 3e6])
    new_cov = np.array([[0.1, 0.02], [0.02, 0.2]])
    sec = merge_mf33_covariance_into_host(
        str(src), new_cov, new_grid, mt=2, za=ZA, awr=AWR, mat=MAT)
    m, g, _ = sec._self_covariance_matrix()
    np.testing.assert_allclose(np.asarray(m), new_cov)
    np.testing.assert_allclose(g, new_grid, rtol=1e-12)


def test_merge_against_real_host_structure(tmp_path, fe56_host_tape):
    """Read-only structural check on the real JEFF-4.0 Fe-56 host MT2."""
    new_grid = np.array([0.9e6, 2.0e6, 4.0e6])
    new_cov = np.array([[0.0025, 0.001], [0.001, 0.0030]])
    sec = merge_mf33_covariance_into_host(str(fe56_host_tape), new_cov, new_grid, mt=2)
    m, g, is_rel = sec._self_covariance_matrix()
    m = np.asarray(m); g = np.asarray(g)

    assert is_rel
    # Full host range preserved at the ends; strictly increasing grid.
    assert g[0] == pytest.approx(1e-5) and g[-1] == pytest.approx(1.5e8)
    assert np.all(np.diff(g) > 0)
    # The in-range block is exactly ours.
    i_lo = int(np.searchsorted(g, new_grid[0]))
    np.testing.assert_allclose(m[i_lo:i_lo + 2, i_lo:i_lo + 2], new_cov)
    # In-out cross terms zeroed; host variance survives outside.
    assert np.all(m[i_lo:i_lo + 2, :i_lo] == 0.0)
    assert np.all(m[i_lo:i_lo + 2, i_lo + 2:] == 0.0)
    out_diag = np.concatenate([np.diag(m)[:i_lo], np.diag(m)[i_lo + 2:]])
    assert np.all(out_diag >= 0.0) and np.any(out_diag > 0.0)
