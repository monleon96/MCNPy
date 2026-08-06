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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MF34_to_MG reads _pendf_mt._energies / ._cross_sections "
        "(kika/cov/multigroup/collapse.py:958 and :941), which only MF3MT has. "
        "But its own docstring at :734 tells the caller to write "
        "`endf.pendf = kika.processing.njoy_reconstruct(...)`, and that returns "
        "Dict[int, CrossSection]. So the documented recipe raises AttributeError. "
        "The existing golden passes because it feeds MF3MT objects from "
        "kika.endf.processing.reconstruct instead. Not fixed here: this "
        "increment adds tests only. Fix belongs with phase 4's move of the "
        "calculations onto the model, or sooner as its own commit."
    ),
)
def test_mf34_to_mg_accepts_the_pendf_its_own_docstring_recommends(endf_for_collapse):
    from kika.cov.multigroup.collapse import MF34_to_MG

    endf = endf_for_collapse
    # The shape njoy_reconstruct hands back, which is what the docstring names.
    endf.pendf = {2: CrossSection.from_endf(endf.mf[3].mt[2])}

    MF34_to_MG(endf, energy_grid=MG_GRID_EV, mt=2)


def test_mf34_to_mg_works_on_the_shape_the_golden_actually_feeds(endf_for_collapse):
    """The other half of the pair above: MF3MT sections do work.

    Keeping both means the xfail above is attributable to the *shape* of
    ``pendf`` and not to anything else about the collapse.
    """
    from kika.cov.multigroup.collapse import MF34_to_MG

    endf = endf_for_collapse
    endf.pendf = {2: endf.mf[3].mt[2]}

    MF34_to_MG(endf, energy_grid=MG_GRID_EV, mt=2)
