"""Tests for _make_lb5_record, _split_matrix_excluding_range, and multi-NI merge."""

import numpy as np
import pytest

from kika.endf.writers.mf34_writer import (
    _make_lb5_record,
    _split_matrix_excluding_range,
)


# ---------- _make_lb5_record tests ----------


def test_make_lb5_record_basic():
    """Known 3x3 matrix → verify NT, NE, upper-triangle values."""
    mat = np.array([
        [1.0, 0.2, 0.3],
        [0.2, 2.0, 0.4],
        [0.3, 0.4, 3.0],
    ])
    grid = [1.0, 2.0, 3.0, 4.0]  # 4 boundaries → 3 intervals

    rec = _make_lb5_record(mat, grid)

    assert rec.ls == 1
    assert rec.lb == 5
    assert rec.ne == 4
    assert rec.energies == grid
    # Upper triangle (row-wise): (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
    expected = [1.0, 0.2, 0.3, 2.0, 0.4, 3.0]
    np.testing.assert_allclose(rec.matrix, expected)
    assert rec.nt == 4 + 6  # NE + upper-triangle size


def test_make_lb5_record_1x1():
    """Single-interval matrix (NE=2)."""
    mat = np.array([[0.5]])
    grid = [10.0, 20.0]
    rec = _make_lb5_record(mat, grid)

    assert rec.ne == 2
    assert rec.matrix == [0.5]
    assert rec.nt == 3


def test_make_lb5_record_shape_mismatch():
    """Shape mismatch raises ValueError."""
    mat = np.eye(3)
    grid = [1.0, 2.0]  # 1 interval, but 3x3 matrix
    with pytest.raises(ValueError, match="doesn't match"):
        _make_lb5_record(mat, grid)


# ---------- _split_matrix_excluding_range tests ----------


def _make_test_data(n=6):
    """Create a test matrix and grid with n intervals."""
    grid = list(np.linspace(1e6, 20e6, n + 1))
    mat = np.arange(n * n, dtype=float).reshape(n, n)
    # Make symmetric for realism
    mat = mat + mat.T
    return mat, grid


def test_split_no_overlap():
    """Exclude range outside original → returns full data."""
    mat, grid = _make_test_data(4)
    # grid midpoints are well within [1e6, 20e6]; exclude range far above
    result = _split_matrix_excluding_range(mat, grid, 50e6, 60e6)

    assert len(result) == 1
    np.testing.assert_array_equal(result[0][0], mat)
    assert result[0][1] == grid


def test_split_full_overlap():
    """Exclude range covers everything → returns empty."""
    mat, grid = _make_test_data(4)
    result = _split_matrix_excluding_range(mat, grid, 0.0, 100e6)

    assert len(result) == 0


def test_split_below_only():
    """Only low-energy intervals survive."""
    # 6 intervals: midpoints at ~2.6, 5.8, 8.9, 12.1, 15.3, 18.4 MeV
    mat, grid = _make_test_data(6)
    midpoints = [0.5 * (grid[i] + grid[i + 1]) for i in range(6)]

    # Exclude from 10 MeV upward → keep intervals 0,1,2 (midpoints < 10e6)
    result = _split_matrix_excluding_range(mat, grid, 10e6, 100e6)

    assert len(result) == 1
    sub_mat, sub_grid = result[0]
    n_kept = sub_mat.shape[0]
    assert n_kept == sum(1 for m in midpoints if m < 10e6)
    assert len(sub_grid) == n_kept + 1


def test_split_above_only():
    """Only high-energy intervals survive."""
    mat, grid = _make_test_data(6)
    midpoints = [0.5 * (grid[i] + grid[i + 1]) for i in range(6)]

    # Exclude below 10 MeV
    result = _split_matrix_excluding_range(mat, grid, 0.0, 10e6)

    assert len(result) == 1
    sub_mat, sub_grid = result[0]
    n_kept = sub_mat.shape[0]
    assert n_kept == sum(1 for m in midpoints if m > 10e6)


def test_split_both_sides():
    """Exclude middle → returns two sub-ranges."""
    mat, grid = _make_test_data(6)
    midpoints = [0.5 * (grid[i] + grid[i + 1]) for i in range(6)]

    # Exclude 7–14 MeV → keeps low and high intervals
    result = _split_matrix_excluding_range(mat, grid, 7e6, 14e6)

    assert len(result) == 2
    n_low = result[0][0].shape[0]
    n_high = result[1][0].shape[0]
    expected_low = sum(1 for m in midpoints if m < 7e6)
    expected_high = sum(1 for m in midpoints if m > 14e6)
    assert n_low == expected_low
    assert n_high == expected_high

    # Verify sub-grids have correct number of boundaries
    assert len(result[0][1]) == n_low + 1
    assert len(result[1][1]) == n_high + 1


def test_split_preserves_matrix_values():
    """Verify the sub-matrix contains the correct values from the original."""
    mat, grid = _make_test_data(6)
    midpoints = [0.5 * (grid[i] + grid[i + 1]) for i in range(6)]

    result = _split_matrix_excluding_range(mat, grid, 7e6, 14e6)

    # Check first (low-energy) sub-matrix
    low_idx = [i for i, m in enumerate(midpoints) if m < 7e6]
    expected_low_mat = mat[np.ix_(low_idx, low_idx)]
    np.testing.assert_array_equal(result[0][0], expected_low_mat)

    # Check second (high-energy) sub-matrix
    high_idx = [i for i, m in enumerate(midpoints) if m > 14e6]
    expected_high_mat = mat[np.ix_(high_idx, high_idx)]
    np.testing.assert_array_equal(result[1][0], expected_high_mat)


# ---------- merge_mf34 multi-NI integration test ----------


def test_merge_original_only_pair_excludes_pipeline_range():
    """Original-only (L,L1) pair gets pipeline range excised in merge."""
    from kika.endf.writers.mf34_writer import (
        create_mf34_from_covariance,
        merge_mf34,
    )

    # Original: orders 1-4, 5 energy intervals, grid 1-20 MeV
    orig_max = 4
    n_orig = 5
    orig_grid = np.linspace(1e6, 20e6, n_orig + 1)
    orig_cov = np.eye(n_orig * orig_max) * 0.01

    orig_mf34 = create_mf34_from_covariance(
        cov_matrix=orig_cov,
        energy_grid_ev=orig_grid,
        max_order=orig_max,
        za=26056.0, awr=55.47, mat=2631, mt=2,
    )

    # Pipeline: orders 1-2, 3 energy intervals, grid 5-15 MeV
    pipe_max = 2
    n_pipe = 3
    pipe_grid = np.linspace(5e6, 15e6, n_pipe + 1)
    pipe_cov = np.eye(n_pipe * pipe_max) * 0.05

    pipe_mf34 = create_mf34_from_covariance(
        cov_matrix=pipe_cov,
        energy_grid_ev=pipe_grid,
        max_order=pipe_max,
        za=26056.0, awr=55.47, mat=2631, mt=2,
    )

    merged = merge_mf34(orig_mf34, pipe_mf34, 5e6, 15e6)
    covmat = merged.to_ang_covmat()

    # Check that original-only pairs (e.g., (1,3), (3,4), (4,4))
    # have zeros in the pipeline energy range
    for i in range(covmat.num_matrices):
        l, l1 = covmat.l_rows[i], covmat.l_cols[i]
        egrid = covmat.energy_grids[i]
        matrix = covmat.matrices[i]

        if l > pipe_max or l1 > pipe_max:
            # Original-only pair: pipeline-range intervals should be zero
            # (the aggregator may create gap intervals from union grid,
            # but they must contain zero covariance)
            egrid_arr = np.asarray(egrid, dtype=float)
            mids = 0.5 * (egrid_arr[:-1] + egrid_arr[1:])
            in_pipe = (mids >= 5e6) & (mids <= 15e6)
            pipe_idx = np.where(in_pipe)[0]
            if pipe_idx.size > 0:
                sub = matrix[np.ix_(pipe_idx, pipe_idx)]
                assert np.allclose(sub, 0.0), (
                    f"Pair ({l},{l1}) has non-zero data in pipeline range"
                )


def test_merge_original_only_multi_ni_records():
    """Original-only pair spanning pipeline range gets multiple NI records."""
    from kika.endf.writers.mf34_writer import (
        create_mf34_from_covariance,
        merge_mf34,
    )

    # Original: orders 1-3, 8 intervals over 1-20 MeV
    orig_max = 3
    n_orig = 8
    orig_grid = np.linspace(1e6, 20e6, n_orig + 1)
    orig_cov = np.eye(n_orig * orig_max) * 0.02

    orig_mf34 = create_mf34_from_covariance(
        cov_matrix=orig_cov,
        energy_grid_ev=orig_grid,
        max_order=orig_max,
        za=26056.0, awr=55.47, mat=2631, mt=2,
    )

    # Pipeline: orders 1-2 only, 4 intervals over 6-14 MeV
    pipe_max = 2
    n_pipe = 4
    pipe_grid = np.linspace(6e6, 14e6, n_pipe + 1)
    pipe_cov = np.eye(n_pipe * pipe_max) * 0.05

    pipe_mf34 = create_mf34_from_covariance(
        cov_matrix=pipe_cov,
        energy_grid_ev=pipe_grid,
        max_order=pipe_max,
        za=26056.0, awr=55.47, mat=2631, mt=2,
    )

    merged = merge_mf34(orig_mf34, pipe_mf34, 6e6, 14e6)

    # Find a sub-subsection for an original-only pair
    subsec = merged._subsections[0]
    for ss in subsec.sub_subsections:
        if ss.l > pipe_max or ss.l1 > pipe_max:
            # Should have NI > 1 if pipeline range splits the original
            assert ss.ni == len(ss.records)
            # With 8 intervals over 1-20 MeV and exclusion 6-14 MeV,
            # expect 2 sub-ranges (below and above)
            assert len(ss.records) == 2, (
                f"Pair ({ss.l},{ss.l1}): expected 2 NI records, got {len(ss.records)}"
            )


def test_merge_roundtrip_serialization():
    """Merged MF34 with multi-NI records survives str() → parse round-trip."""
    from kika.endf.writers.mf34_writer import (
        create_mf34_from_covariance,
        merge_mf34,
    )
    from kika.endf.parsers.parse_mf34 import parse_mf34_mt

    # Same setup as above
    orig_max = 3
    n_orig = 6
    orig_grid = np.linspace(1e6, 20e6, n_orig + 1)
    orig_cov = np.eye(n_orig * orig_max) * 0.01
    orig_mf34 = create_mf34_from_covariance(
        cov_matrix=orig_cov, energy_grid_ev=orig_grid, max_order=orig_max,
        za=26056.0, awr=55.47, mat=2631, mt=2,
    )

    pipe_max = 2
    n_pipe = 3
    pipe_grid = np.linspace(5e6, 15e6, n_pipe + 1)
    pipe_cov = np.eye(n_pipe * pipe_max) * 0.05
    pipe_mf34 = create_mf34_from_covariance(
        cov_matrix=pipe_cov, energy_grid_ev=pipe_grid, max_order=pipe_max,
        za=26056.0, awr=55.47, mat=2631, mt=2,
    )

    merged = merge_mf34(orig_mf34, pipe_mf34, 5e6, 15e6)

    # Round-trip: serialize and parse back
    endf_str = str(merged)
    lines = endf_str.split('\n')
    parsed = parse_mf34_mt(lines, mt=2)

    # Verify the parsed object can produce a valid covmat
    covmat = parsed.to_ang_covmat()
    assert covmat.num_matrices > 0
