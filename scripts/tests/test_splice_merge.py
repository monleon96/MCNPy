"""The splice merge, and what it does with a repeated abscissa.

``splice_legendre_grid`` replaces the MF4 energy points inside the fitted
window with the pipeline's own grid and keeps everything outside it verbatim.
That last part is the interesting one: an evaluated file may legitimately
repeat an energy, and JEFF-4.0 Fe-56 does — MF4/MT2 carries two consecutive
LIST subsections at 3.905 MeV with byte-identical coefficients. Kept verbatim,
both copies reached a strict ``>`` sortedness check, which rejected a grid that
had come straight out of the input.

The roadmap recorded that as "the splice emits it twice". It does not: it
preserves what it was given, faithfully, and then refuses it. Which is why the
fix is not to deduplicate blindly — that would delete evaluation data.

These run without the pipeline, so the policy is pinned in CI rather than only
through a several-second end-to-end run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exfor_utils import (  # noqa: E402
    SPLICE_KEEP_TOLERANCE_EV,
    merge_spliced_grid,
    split_original_grid,
)


def _merge(left_e, left_c, pipe_e, pipe_c, right_e, right_c, logger=None):
    return merge_spliced_grid(left_e, left_c, pipe_e, pipe_c, right_e, right_c,
                              logger=logger)


# ---------------------------------------------------------------------------
# The ordinary case
# ---------------------------------------------------------------------------

def test_a_clean_merge_is_left_then_pipeline_then_right():
    e, c = _merge([1.0, 2.0], [[0.1], [0.2]],
                  [3.0, 4.0], [[0.3], [0.4]],
                  [5.0], [[0.5]])
    assert e == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert c == [[0.1], [0.2], [0.3], [0.4], [0.5]]


def test_an_empty_pipeline_still_merges():
    e, _ = _merge([1.0], [[0.1]], [], [], [2.0], [[0.2]])
    assert e == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Repeated abscissae
# ---------------------------------------------------------------------------

def test_an_exact_duplicate_is_dropped():
    """Same energy and same coefficients: a redundant record, no information."""
    e, c = _merge([1.0, 1.0], [[0.5, 0.25], [0.5, 0.25]],
                  [2.0], [[0.9]], [], [])
    assert e == [1.0, 2.0]
    assert c == [[0.5, 0.25], [0.9]]


def test_dropping_a_duplicate_is_logged():
    """Project rule: a pipeline operation that modifies a sample says so."""
    class Recorder:
        def __init__(self):
            self.info_lines, self.warning_lines = [], []

        def info(self, message):
            self.info_lines.append(message)

        def warning(self, message):
            self.warning_lines.append(message)

    log = Recorder()
    _merge([3905000.0, 3905000.0], [[0.5], [0.5]], [], [], [], [], logger=log)

    assert any("3905000" in line for line in log.info_lines), log.info_lines
    assert not log.warning_lines


def test_a_genuine_discontinuity_is_kept():
    """Same energy, different coefficients: that is data, not redundancy."""
    e, c = _merge([1.0, 1.0], [[0.5], [0.7]], [2.0], [[0.9]], [], [])
    assert e == [1.0, 1.0, 2.0]
    assert c == [[0.5], [0.7], [0.9]]


def test_a_genuine_discontinuity_is_warned_about():
    class Recorder:
        def __init__(self):
            self.info_lines, self.warning_lines = [], []

        def info(self, message):
            self.info_lines.append(message)

        def warning(self, message):
            self.warning_lines.append(message)

    log = Recorder()
    _merge([1.0, 1.0], [[0.5], [0.7]], [], [], [], [], logger=log)

    assert any("1.0" in line for line in log.warning_lines), log.warning_lines


def test_a_duplicate_across_the_seam_is_handled_too():
    """The repeat need not be within one portion."""
    e, _ = _merge([1.0], [[0.5]], [1.0, 2.0], [[0.5], [0.6]], [], [])
    assert e == [1.0, 2.0]


# ---------------------------------------------------------------------------
# The check that has to survive
# ---------------------------------------------------------------------------

def test_an_out_of_order_grid_is_still_an_error():
    """Relaxing strict > to non-decreasing must not relax it to anything."""
    with pytest.raises(AssertionError, match="out of order"):
        _merge([5.0], [[0.5]], [2.0], [[0.2]], [], [])


def test_the_error_names_the_offending_energies():
    with pytest.raises(AssertionError) as excinfo:
        _merge([], [], [9.0, 3.0], [[0.9], [0.3]], [], [])
    assert "9.0" in str(excinfo.value) and "3.0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The window split
# ---------------------------------------------------------------------------

def test_points_inside_the_window_are_dropped():
    left_e, _, right_e, _ = split_original_grid(
        [1.0, 5.0, 10.0, 15.0, 20.0], [[i] for i in range(5)], 4.0, 16.0
    )
    assert left_e == [1.0]
    assert right_e == [20.0]


def test_a_point_on_the_boundary_is_dropped_within_the_tolerance():
    """The pipeline grid owns the window, including its edges."""
    boundary = 4.0 + SPLICE_KEEP_TOLERANCE_EV / 2
    left_e, _, right_e, _ = split_original_grid(
        [1.0, boundary, 20.0], [[0], [1], [2]], 4.0, 16.0
    )
    assert left_e == [1.0]
    assert right_e == [20.0]


def test_the_fe56_duplicate_survives_the_real_split():
    """The exact configuration that used to fail: 3.905 MeV, twice, outside."""
    energies = [3905000.0, 3905000.0, 4.5e6, 5.0e6]
    coeffs = [[0.5788, 0.4253], [0.5788, 0.4253], [0.4], [0.3]]

    left_e, left_c, right_e, right_c = split_original_grid(
        energies, coeffs, 4.4e6, 5.6e6
    )
    assert left_e == [3905000.0, 3905000.0], "both copies are kept, as originals"

    merged_e, _ = _merge(left_e, left_c, [4.6e6], [[0.45]], right_e, right_c)
    assert merged_e == [3905000.0, 4.6e6], "the redundant copy is gone, in order"
