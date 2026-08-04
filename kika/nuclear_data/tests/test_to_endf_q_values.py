"""``to_endf`` must not invent the Q values it cannot know.

An MF3 section header carries QM (mass-difference Q) and QI (reaction Q). ACE
records neither — it is a processed representation, and the Q values did not
survive processing. ``to_endf`` used to paper over that with

    qm = self.metadata.get("qm", 0.0)
    qi = self.metadata.get("qi", 0.0)

so an ACE-sourced section round-tripped to ENDF with QM = QI = 0. For elastic
scattering that is right by accident. For every threshold reaction it is a
physically wrong header, written silently.

The distinction that matters is *absent* versus *zero*. A section that came
from ENDF has real Q values, zero or not, and must keep writing them; a section
that never had them must say so rather than guess.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.nuclear_data.cross_section import CrossSection


def _ace_shaped() -> CrossSection:
    """What from_ace builds: awr, but no mat/qm/qi/lr."""
    return CrossSection(
        energies=np.array([1.0e3, 1.0e6]),
        values=np.array([2.0, 1.0]),
        reaction=102,
        nuclide_id=26056,
        interpolation="linlin",
        metadata={
            "source_format": "ace",
            "ace_zaid": "26056.02c",
            "awr": 55.454,
        },
    )


def _endf_shaped(qm: float = -2.5e6, qi: float = -2.5e6) -> CrossSection:
    """What from_endf builds: the full MF3 header."""
    return CrossSection(
        energies=np.array([1.0e3, 1.0e6]),
        values=np.array([2.0, 1.0]),
        reaction=102,
        nuclide_id=26056,
        interpolation="linlin",
        metadata={
            "source_format": "endf",
            "mat": 2631,
            "awr": 55.454,
            "qm": qm,
            "qi": qi,
            "lr": 0,
            "interpolation_regions": [(2, 2)],
        },
    )


def test_an_ace_sourced_section_refuses_to_write_a_header():
    with pytest.raises(ValueError) as excinfo:
        _ace_shaped().to_endf()

    message = str(excinfo.value)
    assert "qm" in message and "qi" in message, message
    assert "ace" in message, "the message should name where the section came from"


def test_explicit_q_values_make_it_writable():
    """The raise has to be escapable by a caller who knows the values."""
    mf3mt = _ace_shaped().to_endf(mat=2631, qm=-7.6e6, qi=-7.6e6, lr=0)

    assert mf3mt._qm == -7.6e6
    assert mf3mt._qi == -7.6e6
    assert mf3mt._lr == 0
    assert mf3mt._mat == 2631


def test_an_endf_sourced_section_still_writes_its_own_values():
    mf3mt = _endf_shaped().to_endf()

    assert mf3mt._qm == -2.5e6
    assert mf3mt._qi == -2.5e6
    assert mf3mt._lr == 0
    assert mf3mt._mat == 2631


def test_a_genuine_zero_q_is_not_treated_as_missing():
    """Elastic scattering has QM = QI = 0. That is data, not a gap."""
    mf3mt = _endf_shaped(qm=0.0, qi=0.0).to_endf()

    assert mf3mt._qm == 0.0
    assert mf3mt._qi == 0.0


def test_an_explicit_argument_beats_metadata():
    mf3mt = _endf_shaped(qm=-2.5e6, qi=-2.5e6).to_endf(qm=-1.0e6)

    assert mf3mt._qm == -1.0e6, "explicit qm should win"
    assert mf3mt._qi == -2.5e6, "qi was not overridden, so metadata stands"


@pytest.mark.parametrize("dropped", ["qm", "qi", "lr"])
def test_each_missing_field_is_named(dropped):
    xs = _endf_shaped()
    del xs.metadata[dropped]

    with pytest.raises(ValueError, match=dropped):
        xs.to_endf()


def test_the_reconstruction_path_still_works(micro_tape):
    """to_endf's one caller in the tree is the ENDF reconstruction adapter.

    It builds CrossSections that do carry qm/qi/lr, so the raise must not
    reach it. Round-tripping a parsed MF3 section is the same contract.
    """
    from kika.endf import read_endf

    endf = read_endf(str(micro_tape), mf_numbers=[3])
    original = endf.mf[3].mt[2]

    rebuilt = CrossSection.from_endf(original).to_endf()

    assert rebuilt.q_mass_difference == original.q_mass_difference
    assert rebuilt.q_reaction == original.q_reaction
    np.testing.assert_array_equal(rebuilt._energies, original.energies)
