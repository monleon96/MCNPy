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
    resonances, _, report = decodeMF2MT151(mf2)
    return resonances, report


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

    for group, block in zip(formalism.spinGroups, energyRange.parameters.l_values):
        # The file's own APL field, not the *resolved* radius: what the model
        # must reproduce is what was written, and the fallback to AP is a
        # reading of a zero rather than a value in the file.
        expected = block.apl_or_qx if block.apl_or_qx else None
        for channel in group.channels:
            assert channel.scatteringRadius == expected, (
                f"L={block.l}: the per-l scattering radius did not survive"
            )

    perL = [g.channels[0].scatteringRadius for g in formalism.spinGroups]
    assert any(r is not None for r in perL), "every l fell back to AP"


def test_none_means_the_file_wrote_zero_not_a_radius_that_equals_ap(decoded, mf2):
    """``None`` is a statement about the *file*, not about a comparison.

    ENDF's convention is that APL = 0 means "this l uses the range's AP", so
    ``None`` here means the field was zero. It must **not** mean "the radius
    came out equal to AP" — those are different statements and only the first
    is in the file.

    This test used to assert the second, and it was wrong in a way its own
    docstring already argued against. The consequence was measured in B1a and is
    the reason the assertion was inverted rather than adjusted: **U-235, Th-232
    and Pu-241 all write APL explicitly equal to AP**, so the old collapse
    stored ``None`` for them and an encoder would have written ``0.0`` where the
    file has ``9.686000-1``. Half the corpus, and byte identity lost on all of
    it. ``test_an_apl_equal_to_ap_is_kept`` below is the real-tape half.
    """
    resonances, _ = decoded
    formalism = resonances.resolved[0].formalism
    blocks = mf2.isotopes[0].energy_ranges[0].parameters.l_values
    ap = mf2.isotopes[0].energy_ranges[0].parameters.ap
    assert resonances.scatteringRadius.constant == ap

    wroteZero = [not block.apl_or_qx for block in blocks]
    assert any(wroteZero) and not all(wroteZero), (
        f"this fixture no longer has both spellings of APL: {[b.apl_or_qx for b in blocks]}"
    )
    for group, block, zero in zip(formalism.spinGroups, blocks, wroteZero):
        radius = group.channels[0].scatteringRadius
        assert (radius is None) == zero, (
            f"L={block.l}: the file wrote APL={block.apl_or_qx!r} and the model "
            f"holds {radius!r}; None must mean exactly 'the file wrote 0'"
        )


@pytest.mark.parametrize("tape", ["u235_tape", "th232_tape", "pu241_tape"])
def test_an_apl_equal_to_ap_is_kept(request, tape):
    """The three tapes that write APL == AP, which is where the collapse bit.

    Parametrized over the tapes rather than asserted on one, because the point
    is that this is *half the corpus* and not a curiosity of one evaluation. If
    a future tape stops writing APL the guard below fails rather than the test
    passing vacuously.
    """
    path = request.getfixturevalue(tape)
    section = read_endf(str(path), mf_numbers=[2]).mf[2].mt[151]
    resonances, _, _ = decodeMF2MT151(section)

    energyRange = section.isotopes[0].energy_ranges[0]
    assert energyRange.lrf == 3, f"{tape} is no longer Reich-Moore"
    blocks = energyRange.parameters.l_values
    ap = energyRange.parameters.ap
    equal = [b for b in blocks if b.apl_or_qx and b.apl_or_qx == ap]
    assert equal, (
        f"{tape} no longer writes an APL equal to its AP={ap}; this test exists "
        f"for that case and would now pass without checking it"
    )

    formalism = resonances.resolved[0].formalism
    for group, block in zip(formalism.spinGroups, blocks):
        if block.apl_or_qx and block.apl_or_qx == ap:
            assert group.channels[0].scatteringRadius == ap, (
                f"{tape} L={block.l}: APL was written as {ap} and equals the "
                f"range's AP; collapsing it to None loses what the file said"
            )


@pytest.mark.parametrize("tape", ["u235_tape", "th232_tape", "pu241_tape"])
def test_the_flat_lgroup_ap_changed_with_it_and_that_is_deliberate(request, tape):
    """The public-API consequence of the APL fix, pinned rather than discovered.

    ``ResonanceParameters.from_endf`` reads ``channels[0].scatteringRadius``
    through ``interop.flatResonanceParameters``, so un-collapsing the decoder
    changes what a *public* class returns: on these three tapes ``LGroup.ap``
    went from ``None`` to the range's AP.

    **It is numerically equivalent** — ``None`` has always meant "use the
    range's AP", and that is the same number — so nothing downstream computes
    anything different. It is still a visible change, and decision 3(a) of
    ``docs/mf2-encoder-notes.md`` took it knowingly: the alternative was to
    re-collapse in ``interop`` and keep two sources of truth for one field.
    """
    path = request.getfixturevalue(tape)
    section = read_endf(str(path), mf_numbers=[2]).mf[2].mt[151]
    flat = [f for f in ResonanceParameters.from_endf(section)
            if isinstance(f, ResonanceParameters)]
    assert flat, f"{tape} produced no resolved flat parameters"

    ap = flat[0].scattering_radius
    written = [group.ap for group in flat[0].l_groups]
    assert all(value == ap for value in written), (
        f"{tape}: LGroup.ap is {written}, expected every l to carry AP={ap}. "
        f"If these are None again the decoder has re-collapsed APL == AP."
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
    resonances, _, report = decodeMF2MT151(mf2)

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
    resonances, _, _ = decodeMF2MT151(mf2)
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

    # It still returns nothing — `ResonanceRecord` has four width columns and an
    # RML spin group has one per channel, five of them here, so there is nothing
    # to return into. But as of the `library-gaps.md` D3 fix it is no longer
    # silent about it, which is the difference between an unsupported format and
    # a wrong answer.
    with pytest.warns(UserWarning, match="LRF=7"):
        assert ResonanceParameters.from_endf(mf2) == [], (
            "the flat path has started keeping LRF=7; update this test and the "
            "D3 entry in docs/library-gaps.md"
        )

    resonances, _, report = decodeMF2MT151(mf2)
    assert len(resonances.resolved) == 1
    formalism = resonances.resolved[0].formalism
    assert isinstance(formalism, RMatrix)
    assert formalism.numberOfResonances > 0
    # Channel-shaped in the file, so the widths are per channel here too.
    group = formalism.spinGroups[0]
    assert all(len(row) == len(group.channels) for row in group.widths)


# ---------------------------------------------------------------------------
# What a *calculation* needs the range itself to carry
# ---------------------------------------------------------------------------

def test_the_target_spin_reaches_the_formalism_as_pops(decoded, mf2):
    """ENDF's SPI, on the formalism's own ``PoPs`` (§19.3.1) rather than only in
    provenance.

    The statistical factor is ``g_J = (2J+1) / (2(2I+1))``, so the target spin
    is an *input to the arithmetic*, not a label: a wrong I is a wrong cross
    section. A reconstructor that has to open an ``EndfProvenance`` to find it
    is not format-agnostic, which is the whole point of computing on the model.

    The location is the shipped ENDF/B-VIII.1 GNDS distribution's, not an
    inference — ``n-026_Fe_056.endf.gnds.xml`` writes it inside ``<RMatrix>``
    as ``<PoPs name="resolved resonances">…<spin><fraction value="0"
    unit="hbar"/>``. Per range rather than per suite, so a file whose ranges
    disagree stays representable.
    """
    resonances, _ = decoded
    formalism = resonances.resolved[0].formalism
    spi = mf2.isotopes[0].energy_ranges[0].parameters.spi

    assert formalism.PoPs is not None, "the range carries no target"
    assert len(formalism.PoPs) == 1
    target = formalism.PoPs["Fe56"]
    assert (target.Z, target.A) == (26, 56)
    assert target.spin.value == spi
    assert target.spin.unit == "hbar"


def test_a_spin_of_zero_is_a_spin_and_not_a_missing_one(decoded):
    """Fe-56 has I = 0, and ``0.0`` is falsy.

    This is the whole trap in the helper: ``if spi:`` would give even-even
    targets — the structural-material case this thesis is about — no ``PoPs``
    at all, and a reconstructor reading ``I = None`` would either guess or
    raise on exactly the nuclides that matter most here. Only ``None`` means
    the file said nothing.
    """
    resonances, _ = decoded
    target = resonances.resolved[0].formalism.PoPs["Fe56"]
    assert target.spin is not None
    assert target.spin.value == 0.0


def test_a_reich_moore_range_carries_the_radius_it_declared(decoded, mf2):
    """AP on the *range*, not only on the file.

    ``BreitWigner`` and ``TabulatedWidths`` have carried the range's own AP
    since 3b and ``RMatrix`` did not, which left a Reich-Moore range's radius
    reachable only through ``Resonances.scatteringRadius`` — a **file-level**
    slot filled by whichever range was decoded first. On every tape to hand the
    two are the same number, which is exactly why the gap was invisible; on a
    file whose ranges declare different radii a reconstruction would have used
    another range's radius without saying so.

    Per-*channel* radii are a different quantity and stay on ``Channel``: the
    assertion below is against the range record's AP, and
    ``test_the_l_dependent_radius_reaches_the_channels`` owns APL.
    """
    resonances, _ = decoded
    formalism = resonances.resolved[0].formalism
    parameters = mf2.isotopes[0].energy_ranges[0].parameters

    assert formalism.scatteringRadius == parameters.ap
    assert resonances.scatteringRadius.constant == parameters.ap


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_an_unresolved_region_carries_the_same_spin(request, tape):
    """Under ``--deep``. The URR half of the same requirement.

    The unresolved cross sections carry the same ``g_J`` factor the resolved
    ones do, so ``urr_formulas`` needs I exactly as much as
    ``resonance_formulas`` does. Before this the two halves of one calculation
    asked for the same number in two different ways — the resolved half from
    the formalism, the unresolved half from provenance.

    Parametrized over every tape rather than one, with a skip for the files
    that have no unresolved range — three of the six have one. The skip is a
    real weakness and is worth stating: if the corpus ever lost its URR tapes
    this would pass by checking nothing, and no other test in the suite asserts
    that the corpus still covers LRU=2.
    """
    mf2 = read_endf(str(request.getfixturevalue(tape)), mf_numbers=[2]).mf[2].mt[151]
    unresolvedRanges = [er for iso in mf2.isotopes for er in iso.energy_ranges
                        if er.lru == 2]
    if not unresolvedRanges:
        pytest.skip(f"{tape} has no unresolved range")

    resonances, _, _ = decodeMF2MT151(mf2)
    widths = resonances.unresolved.tabulatedWidths
    assert widths.PoPs is not None, f"{tape}: the unresolved region carries no target"
    assert len(widths.PoPs) == 1
    target = next(iter(widths.PoPs.particles.values()))
    assert target.spin.value == unresolvedRanges[0].parameters.spi
    assert target.spin.unit == "hbar"


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
    resonances, _, report = decodeMF2MT151(_breitWignerSection(lrf))
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
    resonances, _, report = decodeMF2MT151(_breitWignerSection(2, qx=-1.5e6, lrx=1))
    group = resonances.resolved[0].formalism.resonanceParameters.spinGroups[0]
    assert group.scatteringRadius is None
    assert any("QX" in entry for entry in report.losses), (
        "the competitive width was dropped without the report saying so"
    )


def test_an_unsupported_lrf_is_declared_and_the_range_is_not_invented():
    section = _breitWignerSection(4)
    resonances, _, report = decodeMF2MT151(section)
    assert resonances.resolved == []
    assert any("LRF=4" in entry for entry in report.unsupported)


def test_a_section_with_no_ranges_says_so():
    from kika.endf.classes.mf2.mf2mt151 import MF2MT151

    section = MF2MT151(number=151)
    section._isotopes = []
    resonances, _, report = decodeMF2MT151(section)
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

    resonances, _, _ = decodeMF2MT151(section)
    radius = resonances.scatteringRadius
    assert radius.isEnergyDependent
    np.testing.assert_array_equal(radius.values, np.array([0.54, 0.55]))
    assert radius.constant == 0.5444
