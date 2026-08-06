"""MF2/151 → ``resonances``, element for element, split by formalism.

**Decode only, and stated rather than implied.** There is no MF2 encoder here.
``ResonanceParameters`` has never had a ``to_endf``, so unlike MF3 and MF4 there
is no existing path to gate an encoder against, and writing one means emitting
five formalisms' record layouts from scratch. So the gate is: every number the
flat ``ResonanceParameters.from_endf`` extracts is reproduced, **plus** the two
things it either loses or cannot name —

* the l-dependent scattering radius, which phase 1 found MF2/151 was dropping
  and which moves elastic by up to 42% locally on the committed Fe-56 slice;
* the meaning of ``c3..c6``, which depends on LRF and which the model replaces
  with named widths (Breit-Wigner) or channel-indexed widths (R-matrix).

The available tapes are all Reich-Moore, so the Breit-Wigner and R-matrix-limited
paths are driven from hand-built sections. That is a real limitation and it is
better written down than papered over: LRF=1/2 and LRF=7 are tested for
*structure*, not against a real evaluation.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.model_adapter import decodeMF2MT151
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import BreitWigner, BreitWignerApproximation, RMatrix
from kika.nuclear_data.resonance_parameters import (ResonanceParameters,
                                                    UnresolvedResonanceParameters)

REAL_TAPES = ["fe56_host_tape", "fe57_host_tape", "fe56_jendl_tape",
              "u235_tape", "th232_tape", "pu241_tape"]


@pytest.fixture(scope="module")
def mf2(micro_tape):
    return read_endf(str(micro_tape)).mf[2].mt[151]


@pytest.fixture(scope="module")
def decoded(mf2):
    return decodeMF2MT151(mf2)


# ---------------------------------------------------------------------------
# The l-dependent radius — the phase 1 finding, kept
# ---------------------------------------------------------------------------

def test_the_fixture_actually_has_an_l_dependent_radius(mf2):
    """Without this the assertion below would pass on a file that has no APL."""
    energyRange = mf2.isotopes[0].energy_ranges[0]
    assert energyRange.lrf == 3
    radii = {
        block.l: energyRange.scattering_radius_for_l(block.l)
        for block in energyRange.parameters.l_values
    }
    assert len(set(radii.values())) > 1, f"no per-l radius on this tape: {radii}"


def test_the_l_dependent_radius_reaches_the_channels(decoded, mf2):
    """ENDF's APL is per l-block; §19.3.4 puts a radius on a channel."""
    resonances, _ = decoded
    formalism = resonances.resolved[0].formalism
    energyRange = mf2.isotopes[0].energy_ranges[0]
    ap = energyRange.parameters.ap

    for group, block in zip(formalism.spinGroups, energyRange.parameters.l_values):
        expected = energyRange.scattering_radius_for_l(block.l)
        expected = expected if expected != ap else None
        for channel in group.channels:
            assert channel.scatteringRadius == expected, (
                f"L={block.l}: the per-l scattering radius did not survive"
            )

    perL = [g.channels[0].scatteringRadius for g in formalism.spinGroups]
    assert any(r is not None for r in perL), "every l fell back to AP"


def test_a_radius_equal_to_ap_is_none_and_not_a_copy(decoded, mf2):
    """"this l uses the range's AP" and "this l has a radius that equals AP" are
    different statements, and only one of them is in the file."""
    resonances, _ = decoded
    formalism = resonances.resolved[0].formalism
    ap = mf2.isotopes[0].energy_ranges[0].parameters.ap
    assert resonances.scatteringRadius.constant == ap
    assert not any(
        g.channels[0].scatteringRadius == ap for g in formalism.spinGroups
    )


# ---------------------------------------------------------------------------
# Element for element against the flat path
# ---------------------------------------------------------------------------

def _flatResolved(mf2):
    return [f for f in ResonanceParameters.from_endf(mf2)
            if isinstance(f, ResonanceParameters)]


def test_every_resonance_energy_and_width_matches_the_flat_path(decoded, mf2):
    resonances, _ = decoded
    flat = _flatResolved(mf2)[0]
    formalism = resonances.resolved[0].formalism

    assert len(formalism.spinGroups) == len(flat.l_groups)
    for group, lgroup in zip(formalism.spinGroups, flat.l_groups):
        assert len(group) == len(lgroup.resonances)
        np.testing.assert_array_equal(
            np.asarray(group.energies, dtype=float),
            np.asarray([r.energy for r in lgroup.resonances], dtype=float),
        )
        np.testing.assert_array_equal(
            np.asarray(group.widths, dtype=float),
            np.asarray([[r.c3, r.c4, r.c5, r.c6] for r in lgroup.resonances], dtype=float),
        )


def test_the_energy_range_is_the_files_own(decoded, mf2):
    resonances, _ = decoded
    energyRange = mf2.isotopes[0].energy_ranges[0]
    region = resonances.resolved[0]
    assert (region.domainMin, region.domainMax) == (energyRange.el, energyRange.eh)
    assert resonances.domain == (energyRange.el, energyRange.eh)


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_real_tapes_decode_every_resolved_range_the_file_carries(request, tape):
    """Under ``--deep``. Counted against the **file**, not against the flat path.

    The flat path is not a complete oracle here: it drops LRF=7 entirely (see
    the test below), so "same number of regions as ``from_endf`` returned" would
    pass by agreeing that Fe-57 has no resonances.
    """
    mf2 = read_endf(str(request.getfixturevalue(tape))).mf[2].mt[151]
    resonances, report = decodeMF2MT151(mf2)

    inFile = [er for iso in mf2.isotopes for er in iso.energy_ranges if er.lru == 1]
    assert len(resonances.resolved) == len(inFile)
    for region, energyRange in zip(resonances.resolved, inFile):
        assert (region.domainMin, region.domainMax) == (energyRange.el, energyRange.eh)

    unresolvedRanges = [er for iso in mf2.isotopes for er in iso.energy_ranges
                        if er.lru == 2]
    assert (resonances.unresolved is not None) == bool(unresolvedRanges)


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_real_tapes_reproduce_the_flat_path_where_it_produces_anything(request, tape):
    """Every number ``ResonanceParameters.from_endf`` extracts is reproduced."""
    mf2 = read_endf(str(request.getfixturevalue(tape))).mf[2].mt[151]
    resonances, _ = decodeMF2MT151(mf2)
    flat = ResonanceParameters.from_endf(mf2)

    resolved = [f for f in flat if isinstance(f, ResonanceParameters)]
    for region, entry in zip(resonances.resolved, resolved):
        assert (region.domainMin, region.domainMax) == entry.energy_range
        assert sum(len(g) for g in region.formalism.spinGroups) == entry.num_resonances

    unresolved = [f for f in flat if isinstance(f, UnresolvedResonanceParameters)]
    if unresolved:
        widths = resonances.unresolved.tabulatedWidths
        assert len(widths.spinGroups) == sum(
            len(lg.j_groups) for lg in unresolved[0].l_groups
        )
        assert widths.selfShieldingOnly == bool(unresolved[0].lssf)


def test_an_lrf7_evaluation_keeps_its_resonances(fe57_host_tape):
    """Fe-57 JEFF-4.0 is R-Matrix-Limited, and the flat path loses all of it.

    ``ResonanceParameters.from_endf`` guards on
    ``isinstance(er.parameters, ResolvedResonanceRange)``; an ``RMatrixLimited``
    is not one, so the range is skipped and the function returns an **empty
    list** — no exception, no warning. Everything downstream then behaves as
    though Fe-57 had no resolved resonance region at all. The model keeps it,
    which is the point of splitting by formalism.
    """
    mf2 = read_endf(str(fe57_host_tape)).mf[2].mt[151]
    assert [er.lrf for iso in mf2.isotopes for er in iso.energy_ranges] == [7]
    assert ResonanceParameters.from_endf(mf2) == [], (
        "the flat path has started keeping LRF=7; update this test and the "
        "roadmap finding it records"
    )

    resonances, report = decodeMF2MT151(mf2)
    assert len(resonances.resolved) == 1
    formalism = resonances.resolved[0].formalism
    assert isinstance(formalism, RMatrix)
    assert formalism.numberOfResonances > 0
    # Channel-shaped in the file, so the widths are per channel here too.
    group = formalism.spinGroups[0]
    assert all(len(row) == len(group.channels) for row in group.widths)


# ---------------------------------------------------------------------------
# The formalism split — what ``c3..c6`` could not say
# ---------------------------------------------------------------------------

def test_reich_moore_becomes_an_r_matrix_with_two_fission_channels(decoded):
    """LRF=3's columns are ``GN, GG, GFA, GFB`` — two fission widths, which no
    four-name Breit-Wigner vocabulary can express."""
    resonances, _ = decoded
    formalism = resonances.resolved[0].formalism
    assert isinstance(formalism, RMatrix)
    assert formalism.approximation == "ReichMoore"
    labels = [c.label for c in formalism.spinGroups[0].channels]
    assert labels == ["neutron", "capture", "fissionA", "fissionB"]
    assert [r.label for r in formalism.resonanceReactions][:2] == ["elastic", "capture"]


def test_a_width_is_identified_by_its_channel_not_by_a_column_number(decoded):
    resonances, _ = decoded
    group = resonances.resolved[0].formalism.spinGroups[0]
    for row in group.widths:
        assert len(row) == len(group.channels)
    assert [c.columnIndex for c in group.channels] == list(range(len(group.channels)))


def _breitWignerSection(lrf: int, qx: float = 0.0, lrx: int = 0):
    from kika.endf.classes.mf2.mf2mt151 import (EnergyRange, Isotope, LValueBlock,
                                                MF2MT151, Resonance,
                                                ResolvedResonanceRange)

    block = LValueBlock(
        awri=55.45, l=0, num_resonances=2, apl_or_qx=qx, lrx=lrx,
        resonances=[
            Resonance(energy=1000.0, spin=0.5, c3=1.5, c4=1.0, c5=0.5, c6=0.0),
            Resonance(energy=2000.0, spin=0.5, c3=3.0, c4=2.0, c5=1.0, c6=0.0),
        ],
    )
    parameters = ResolvedResonanceRange(spi=0.0, ap=0.5444, nls=1, nlsc=3,
                                        l_values=[block])
    energyRange = EnergyRange(el=1e-5, eh=8.5e5, lru=1, lrf=lrf, nro=0, naps=0,
                              parameters=parameters)
    section = MF2MT151(number=151)
    section._za, section._awr, section._mat, section._nis = 26056.0, 55.45, 2631, 1
    section._isotopes = [Isotope(za=26056.0, abn=1.0, lfw=0, num_energy_ranges=1,
                                 energy_ranges=[energyRange])]
    return section


@pytest.mark.parametrize("lrf,approximation", [
    (1, BreitWignerApproximation.singleLevel),
    (2, BreitWignerApproximation.multiLevel),
])
def test_slbw_and_mlbw_get_named_widths(lrf, approximation):
    """``c3..c6`` under LRF=1/2 are ``GT, GN, GG, GF``, and here they have names."""
    resonances, report = decodeMF2MT151(_breitWignerSection(lrf))
    formalism = resonances.resolved[0].formalism

    assert isinstance(formalism, BreitWigner)
    assert formalism.approximation is approximation

    first = formalism.resonanceParameters.spinGroups[0].resonances[0]
    assert (first.totalWidth, first.neutronWidth, first.captureWidth,
            first.fissionWidth) == (1.5, 1.0, 0.5, 0.0)
    assert formalism.numberOfResonances == 2


def test_qx_is_not_read_as_a_scattering_radius():
    """The same C2 field is APL under LRF=3 and QX under LRF=1/2.

    Reading it as a radius would put a Q value in fm into the hard-sphere phase
    shift, which is the kind of error that produces plausible cross sections.
    """
    resonances, report = decodeMF2MT151(_breitWignerSection(2, qx=-1.5e6, lrx=1))
    group = resonances.resolved[0].formalism.resonanceParameters.spinGroups[0]
    assert group.scatteringRadius is None
    assert any("QX" in entry for entry in report.losses), (
        "the competitive width was dropped without the report saying so"
    )


def test_an_unsupported_lrf_is_declared_and_the_range_is_not_invented():
    section = _breitWignerSection(4)
    resonances, report = decodeMF2MT151(section)
    assert resonances.resolved == []
    assert any("LRF=4" in entry for entry in report.unsupported)


def test_a_section_with_no_ranges_says_so():
    from kika.endf.classes.mf2.mf2mt151 import MF2MT151

    section = MF2MT151(number=151)
    section._isotopes = []
    resonances, report = decodeMF2MT151(section)
    assert resonances.domain is None
    assert any("no resonance region" in entry for entry in report.losses)


def test_an_energy_dependent_radius_wins_over_the_constant(mf2):
    """NRO=1. ENDF gives no per-l version of a tabulated radius, so the table is
    the whole story and the constant is kept only as a fallback record."""
    from kika.endf.classes.mf2.mf2mt151 import EnergyDependentScatteringRadius

    section = _breitWignerSection(2)
    energyRange = section._isotopes[0].energy_ranges[0]
    energyRange.nro = 1
    energyRange.ap_e = EnergyDependentScatteringRadius(
        interpolation=[(2, 2)], energies=[1e-5, 1e6], ap_values=[0.54, 0.55]
    )

    resonances, _ = decodeMF2MT151(section)
    radius = resonances.scatteringRadius
    assert radius.isEnergyDependent
    np.testing.assert_array_equal(radius.values, np.array([0.54, 0.55]))
    assert radius.constant == 0.5444
