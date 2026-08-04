"""The single interpolation scheme a CrossSection falls back on.

``CrossSection.interpolation`` is a one-word summary of a grid that may have
several ENDF interpolation regions. It is only consulted when the per-region
detail is missing from ``metadata['interpolation_regions']`` — an ACE-sourced
section, or one built by hand — but when it is consulted it decides how σ(E)
is evaluated between points, so picking the wrong one silently changes numbers.

It used to be chosen with ``max(regions, key=lambda x: x[0])[1]``. ENDF's NBT
is the cumulative 1-based index of the last point in each region, so it
increases by construction and ``max`` always returned the *last* region — the
opposite of the "covers the most points" the comment claimed.
"""
from __future__ import annotations

import pytest

from kika.nuclear_data.cross_section import _dominant_interpolation


def test_no_regions_gives_linlin():
    assert _dominant_interpolation([]) == "linlin"


def test_single_region_is_that_region():
    assert _dominant_interpolation([(500, 3)]) == "linlog"


def test_the_widest_region_wins_when_it_is_first():
    """The case the old code got wrong.

    990 points lin-log, then 10 points histogram. NBT is cumulative, so the
    old max() picked (1000, 1) — histogram — for a grid that is 99% lin-log.
    """
    assert _dominant_interpolation([(990, 3), (1000, 1)]) == "linlog"


def test_the_widest_region_wins_when_it_is_last():
    """The case the old code got right by accident, which must keep working."""
    assert _dominant_interpolation([(10, 1), (1000, 2)]) == "linlin"


def test_the_widest_region_wins_from_the_middle():
    """Neither first nor last, so neither a max nor a [-1] finds it."""
    assert _dominant_interpolation([(5, 1), (900, 5), (910, 2)]) == "loglog"


def test_ties_go_to_the_earlier_region():
    assert _dominant_interpolation([(50, 4), (100, 2)]) == "loglin"


def test_an_unknown_interpolation_code_falls_back_to_linlin():
    """ENDF defines 1-5; anything else is not something to guess about."""
    assert _dominant_interpolation([(100, 22)]) == "linlin"


@pytest.mark.parametrize(
    "code,name",
    [(1, "histogram"), (2, "linlin"), (3, "linlog"), (4, "loglin"), (5, "loglog")],
)
def test_every_endf_code_maps_to_its_name(code, name):
    assert _dominant_interpolation([(100, code)]) == name


def test_it_agrees_with_from_endf_on_a_real_section(micro_tape):
    """End to end: the attribute a CrossSection ends up carrying.

    MF3/MT2 of the committed slice is a single lin-lin region, so this pins the
    ordinary case through the real parser rather than through a hand-built
    region list.
    """
    from kika.endf import read_endf
    from kika.nuclear_data.cross_section import CrossSection

    endf = read_endf(str(micro_tape), mf_numbers=[3])
    mf3mt = endf.mf[3].mt[2]
    xs = CrossSection.from_endf(mf3mt)

    regions = list(mf3mt.energy_interpolation)
    assert xs.interpolation == _dominant_interpolation(regions)
    assert xs.metadata["interpolation_regions"] == regions, (
        "the per-region detail must survive; the summary is only a fallback"
    )
