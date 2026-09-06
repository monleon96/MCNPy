"""What crosses the boundary into kika-app, pinned by shape.

Three ``kika.processing`` / ``kika.endf.processing`` entry points are called by
the desktop backend and their return values are serialised straight into
pydantic response models:

===========================================  ====================================
``njoy_pendf_cache.read_pendf_mf3_sections``  ``endf_service`` reconstruction
``detect_resonance_bounds``                   ``routers/plot.py:1110`` auto-bounds
``resonance_group_average``                   ``routers/plot.py:1155`` group plot
===========================================  ====================================

The physics of all three is already covered elsewhere. What is *not* covered is
their **shape** — arity, container type, the NaN convention, which errors are
``ValueError``. That is what phase 3/4 will move, and what the app depends on.

**Deviation from the phase 3 plan, deliberate.** The plan called for a fourth
committed fixture, a real ``reconr`` slice, so that ``read_pendf_mf3_sections``
could be pinned without NJOY. It is not needed:
``read_pendf_mf3_sections`` is ``read_endf(path, mf_numbers=[3])`` plus a dict
comprehension, so any MF3-bearing tape exercises the whole function, and the
committed structural micro-tape already is one. A real PENDF would add
provenance, not coverage, at the cost of another megabyte in the wheel
exclusion list. The NJOY subprocess plumbing stays proven by the existing
``njoy``-marked tests.

**One asymmetry worth having in writing**: this function returns ``MF3MT``
objects, while ``kika.processing.njoy_reconstruct`` — the other producer of
"reconstructed sigma per MT" — returns ``CrossSection``. Two shapes, one
concept, and ``ENDF.pendf`` accepts both without saying which its consumers
need. See ``kika/tests/test_duck_typed_consumers.py`` for what that costs.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.processing import detect_resonance_bounds
from kika.endf.read_endf import read_endf
from kika.processing import resonance_group_average
from kika.processing.njoy_pendf_cache import read_pendf_mf3_sections

#: The resolved-range span of the committed Fe-56 slice, in eV.
MICRO_TAPE_BOUNDS = (1e-05, 850000.0)


# ---------------------------------------------------------------------------
# read_pendf_mf3_sections
# ---------------------------------------------------------------------------

def test_read_pendf_mf3_sections_returns_mt_to_mf3mt(micro_tape):
    sections = read_pendf_mf3_sections(micro_tape)

    assert isinstance(sections, dict)
    assert sections, "a tape with MF3 must yield at least one section"
    assert all(isinstance(mt, int) for mt in sections)

    for mt, section in sections.items():
        assert type(section).__name__ == "MF3MT", (
            f"MT{mt} came back as {type(section).__name__}. The app reads "
            f"`.energies` and `.cross_sections` off these; CrossSection has "
            f"`.values` instead and would break it silently."
        )
        assert hasattr(section, "energies")
        assert hasattr(section, "cross_sections")


def test_read_pendf_mf3_sections_refuses_a_tape_without_mf3(micro_cov_tape):
    """A tape whose MF3 does not parse must raise, not return ``{}``.

    The covariance micro-tape carries a stub MF3 that fails to parse. An empty
    dict here would reach ``ENDF.pendf`` and turn into "no sigma for this MT",
    which downstream reads as flux-only weighting — a silently wrong number
    rather than an error.
    """
    with pytest.raises(RuntimeError, match="no MF3 sections"):
        read_pendf_mf3_sections(micro_cov_tape)


def test_read_pendf_mf3_sections_reports_a_missing_file_as_such(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_pendf_mf3_sections(tmp_path / "does-not-exist.endf")


# ---------------------------------------------------------------------------
# detect_resonance_bounds
# ---------------------------------------------------------------------------

def test_detect_resonance_bounds_returns_none_for_no_input():
    assert detect_resonance_bounds(None) is None


def test_detect_resonance_bounds_accepts_both_mf2_shapes(micro_tape):
    """``routers/plot.py`` passes the MF2 *file*; other callers pass MT151."""
    endf = read_endf(str(micro_tape))

    from_file = detect_resonance_bounds(endf.mf[2])
    from_section = detect_resonance_bounds(endf.mf[2].mt[151])

    assert from_file == from_section == MICRO_TAPE_BOUNDS
    assert isinstance(from_file, tuple) and len(from_file) == 2
    assert all(isinstance(v, float) for v in from_file)


# ---------------------------------------------------------------------------
# resonance_group_average
# ---------------------------------------------------------------------------

@pytest.fixture
def pointwise():
    energies = np.array([1.0, 10.0, 100.0, 1000.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    return energies, values


def test_resonance_group_average_returns_edges_and_averages(pointwise):
    energies, values = pointwise
    boundaries = np.array([1.0, 100.0, 1000.0])

    result = resonance_group_average(energies, values, boundaries)

    assert isinstance(result, tuple) and len(result) == 2
    edges, averages = result
    np.testing.assert_array_equal(edges, boundaries)
    assert averages.shape == (len(boundaries) - 1,)
    assert np.all(np.isfinite(averages))


def test_groups_outside_the_pointwise_range_are_nan_not_zero(pointwise):
    """The convention the app's plotting code relies on to drop empty groups.

    Zero would be a legitimate cross section; NaN is the only value that says
    "no data here". Changing this silently turns gaps into hard zeros on every
    group plot in the desktop app.
    """
    energies, values = pointwise
    edges, averages = resonance_group_average(
        energies, values, np.array([1e4, 1e5])
    )

    assert averages.shape == (1,)
    assert np.isnan(averages[0])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(energies=np.array([[1.0, 2.0]]), cross_sections=np.array([[1.0, 2.0]])),
         "1-D arrays"),
        (dict(cross_sections=np.array([1.0, 2.0])), "same shape"),
        (dict(energies=np.array([1.0]), cross_sections=np.array([1.0])),
         "at least 2"),
        (dict(energies=np.array([10.0, 1.0, 100.0, 1000.0])), "strictly increasing"),
        (dict(group_boundaries=np.array([1.0])), "group_boundaries"),
        (dict(group_boundaries=np.array([100.0, 1.0])), "strictly increasing"),
        (dict(weighting="uniform"), "unknown weighting"),
        (dict(energies=np.array([-10.0, 0.0, 100.0, 1000.0])), "strictly positive"),
    ],
    ids=[
        "not-1d", "shape-mismatch", "too-few-points", "energies-unsorted",
        "one-boundary", "boundaries-unsorted", "bad-weighting", "non-positive-energy",
    ],
)
def test_every_rejection_is_a_valueerror(pointwise, kwargs, match):
    """All eight guards, so a phase 4 rewrite cannot quietly downgrade one.

    The plan for this file said "the three ``ValueError`` paths"; there are
    eight. Counting them is the point — an exception type that changes to
    ``AssertionError`` or a guard that disappears turns a 422 response into a
    500 in the app.
    """
    energies, values = pointwise
    call = dict(
        energies=energies,
        cross_sections=values,
        group_boundaries=np.array([1.0, 100.0, 1000.0]),
    )
    call.update(kwargs)

    with pytest.raises(ValueError, match=match):
        resonance_group_average(**call)
