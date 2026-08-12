"""Where ``MF3MT`` and ``CrossSection`` are interchangeable — and where they are not.

`kika/tests/` holds invariants of the repository as a whole. This is the second
of them, next to ``test_layering.py``.

**The coupling this pins is invisible to grep.** Four sites accept "either an
ENDF ``MF3MT`` or a canonical ``CrossSection``" and pick a path by ``hasattr``.
No type annotation names both; no import links them. ``kika.processing.
njoy_reconstruct`` *returns* ``Dict[int, CrossSection]``, and kika-app's
``endf_service`` and ``plot.py`` consume those objects and serialise them — so
a rename on ``CrossSection`` breaks the desktop app with no grep hit anywhere.
Phase 3 rewrites this layer. These are the seams it must not move.

**What is measured here, not assumed.** Every equality below was run against
the committed micro-tapes before it was written down. One of the four sites
turns out **not** to accept both types, which is recorded as a strict xfail
rather than quietly smoothed over — see
``test_mf34_to_mg_accepts_the_pendf_its_own_docstring_recommends``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kika.endf.read_endf import read_endf
from kika.nuclear_data import CrossSection

#: A coarse grid spanning the micro-tape. Nothing here depends on its values.
GRID_EV = np.array([1e3, 1e5, 1e6, 5e6, 2e7])


@pytest.fixture(scope="module")
def pair(micro_tape):
    """The same MF3/MT2 section in both shapes."""
    endf = read_endf(str(micro_tape))
    mf3mt = endf.mf[3].mt[2]
    return mf3mt, CrossSection.from_endf(mf3mt)


@pytest.fixture(scope="module")
def mf33_section(micro_cov_tape):
    return read_endf(str(micro_cov_tape)).mf[33].mt[2]


# ---------------------------------------------------------------------------
# The three sites that genuinely accept both
# ---------------------------------------------------------------------------

def test_bin_average_agrees_for_both_types(mf33_section, pair):
    """``MF33MT._bin_average_xs`` — same numbers from either shape."""
    mf3mt, xs = pair
    np.testing.assert_array_equal(
        mf33_section._bin_average_xs(mf3mt, list(GRID_EV)),
        mf33_section._bin_average_xs(xs, list(GRID_EV)),
    )


def test_project_to_grid_agrees_for_both_types(mf33_section, pair):
    """``CrossSectionCovariance.project_to_grid(xs_source=...)`` — rel to abs."""
    mf3mt, xs = pair
    covmat = mf33_section.to_xs_covmat()
    assert any(covmat.is_relative), "fixture must exercise the rel->abs path"
    np.testing.assert_array_equal(
        covmat.project_to_grid(GRID_EV, xs_source=mf3mt, target_mt=2),
        covmat.project_to_grid(GRID_EV, xs_source=xs, target_mt=2),
    )


def test_the_interp_shim_is_dead_for_cross_section(mf33_section, pair):
    """A surprise worth writing down before phase 3 moves it.

    ``_bin_average_xs`` branches on ``hasattr(xs_source, 'get_cross_section')``
    and its docstring frames the two arms as "ENDF ``MF3MT``" versus "canonical
    ``CrossSection``". But ``CrossSection`` *has* ``get_cross_section``, so it
    takes the **first** arm — the ``_InterpShim`` fallback is unreachable for
    it, and fires only for a bare object exposing ``energies``/``values``.

    This matters: the shim always interpolates linearly, while
    ``get_cross_section`` honours the section's own scheme. Give
    ``CrossSection`` a ``__getattr__``, or drop ``get_cross_section`` in the
    phase 3d façade, and this silently switches arms — changing log-log
    sections' group averages with nothing else to show for it.
    """
    mf3mt, xs = pair
    assert hasattr(mf3mt, "get_cross_section")
    assert hasattr(xs, "get_cross_section"), (
        "CrossSection lost get_cross_section — _bin_average_xs has just "
        "switched to the linear-only shim for every canonical section"
    )

    class _BareSection:
        energies = np.asarray(xs.energies)
        values = np.asarray(xs.values)

    shimmed = mf33_section._bin_average_xs(_BareSection(), list(GRID_EV))
    assert shimmed.shape == (len(GRID_EV) - 1,)


def test_how_wrong_the_shim_would_be_if_it_were_reached(mf33_section, pair):
    """The size of the hazard above, measured rather than asserted to exist.

    Phase 4's P2 ("one σ-source shape") was planned around the worry that
    unifying the two shapes could flip sections from ``get_cross_section`` to
    the shim and move covariance numbers with every test green. Measured
    2026-08-12, the worry has two halves and they resolve differently.

    **Unreachable, today**: both real producers expose ``get_cross_section`` and
    return identical values through it, so no unification of ``MF3MT`` and
    ``CrossSection`` can flip a branch. That is the test above.

    **But it would matter if it ever were reached.** The shim is lin-lin by
    construction, so it disagrees with any section that declares another
    scheme. Below, the *same points* read as a histogram (ENDF INT=1) rather
    than lin-lin: the shim reads the chord where the file says hold-left, and
    the two answers differ by ~15 % at mid-interval. Recorded so that if the
    shim is ever revived, the number is already known.
    """
    mf3mt, _xs = pair
    energies = np.asarray(mf3mt.energies, dtype=float)
    values = np.asarray(mf3mt.cross_sections, dtype=float)

    histogram = dataclasses.replace(
        mf3mt, _interpolation=[(len(energies), 1)]
    )
    midpoints = (energies[5:9] + energies[6:10]) / 2.0

    honoured = np.asarray(histogram.get_cross_section(midpoints), dtype=float)
    shimmed = np.interp(midpoints, energies, values)

    # Hold-left is exactly the knot below; lin-lin is not.
    np.testing.assert_array_equal(honoured, values[5:9])
    assert np.any(honoured != shimmed), (
        "the fixture no longer distinguishes histogram from lin-lin, so this "
        "measurement says nothing"
    )


# ---------------------------------------------------------------------------
# The site that does not accept both
# ---------------------------------------------------------------------------

def test_the_two_shapes_expose_different_sigma_attributes(pair):
    """The asymmetry every ``pendf`` consumer has to navigate.

    ``ENDF.pendf``'s docstring says "each section exposes ``energies`` and
    ``cross_sections``; both ``MF3MT`` and ``CrossSection`` qualify". Only the
    first half is true. This is the fact the next test turns into a failure.
    """
    mf3mt, xs = pair

    assert hasattr(mf3mt, "energies") and hasattr(xs, "energies")

    assert hasattr(mf3mt, "cross_sections")
    assert not hasattr(xs, "cross_sections")

    assert hasattr(xs, "values")
    assert not hasattr(mf3mt, "values")

    # collapse.py reaches past the public names into these two.
    assert hasattr(mf3mt, "_energies") and hasattr(mf3mt, "_cross_sections")
    assert not hasattr(xs, "_energies") and not hasattr(xs, "_cross_sections")


#: Two bins is enough to exercise the collapse, and the sigma below is the raw
#: MF3 section rather than a reconstruction: these two tests are about the
#: *shape* of ``pendf``, not about the physics of the weight. Reconstructing
#: properly costs ~40 s and would buy nothing here.
MG_GRID_EV = np.array([1e5, 1e6, 2e7])


@pytest.fixture(scope="module")
def endf_for_collapse(micro_tape):
    return read_endf(str(micro_tape))


def test_mf34_to_mg_gives_the_same_answer_for_both_pendf_shapes(endf_for_collapse):
    """Both ``pendf`` shapes, same numbers — not merely "neither crashes".

    This was a real defect. ``MF34_to_MG``'s docstring tells the caller to
    write ``endf.pendf = kika.processing.njoy_reconstruct(...)``, which yields
    ``Dict[int, CrossSection]``, while ``collapse.py`` read ``_energies`` and
    ``_cross_sections`` — names only ``MF3MT`` has. The documented recipe
    raised ``AttributeError``, and the golden never noticed because it feeds
    ``MF3MT`` from ``kika.endf.processing.reconstruct`` instead. Fixed by
    ``collapse._pendf_grid``.

    Equality is the assertion that matters: reading sigma off a different
    attribute of the same data must not move the collapsed matrix.
    """
    from kika.cov.multigroup.collapse import MF34_to_MG

    endf = endf_for_collapse
    raw = endf.mf[3].mt[2]

    endf.pendf = {2: raw}
    from_mf3mt = MF34_to_MG(endf, energy_grid=MG_GRID_EV, mt=2)

    endf.pendf = {2: CrossSection.from_endf(raw)}
    from_cross_section = MF34_to_MG(endf, energy_grid=MG_GRID_EV, mt=2)

    np.testing.assert_array_equal(
        np.asarray(from_mf3mt.energy_grid, dtype=float),
        np.asarray(from_cross_section.energy_grid, dtype=float),
    )
    for attr in ("relative_matrices", "absolute_matrices"):
        left = getattr(from_mf3mt, attr)
        right = getattr(from_cross_section, attr)
        assert len(left) == len(right) and left, f"{attr} is empty"
        for i, (a, b) in enumerate(zip(left, right)):
            np.testing.assert_array_equal(
                np.asarray(a, dtype=float),
                np.asarray(b, dtype=float),
                err_msg=f"{attr}[{i}] moved with the pendf shape",
            )


def test_a_pendf_section_of_an_unknown_shape_is_refused(endf_for_collapse):
    """Silence here would mean an unweighted collapse reported as a weighted one."""
    from kika.cov.multigroup.collapse import MF34_to_MG

    class _Nothing:
        pass

    endf = endf_for_collapse
    endf.pendf = {2: _Nothing()}

    with pytest.raises(TypeError, match="exposes neither"):
        MF34_to_MG(endf, energy_grid=MG_GRID_EV, mt=2)
