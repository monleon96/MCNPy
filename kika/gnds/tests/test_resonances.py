"""§19 resonances, against the two trimmed evaluations that cover it.

``micro_fe56`` carries an ``RMatrix``; ``micro_ta182`` carries a
``BreitWigner`` and a ``tabulatedWidths``. Between them that is every §19
*formalism* in ENDF/B-VIII.1-GNDS. ``micro_sr88`` joined them on 2026-08-24 for
a node rather than a formalism: §19.3.4's ``externalRMatrix`` exists 7 times in
the distribution and all 7 are in that one evaluation. In all three the
``resonances`` subtree is **verbatim** — the build script never touches it,
which is the reason those files exist.

Numbers here may therefore be asserted, unlike the cross sections in the same
fixtures. Whether they are the *right* numbers is
:mod:`kika.gnds.tests.test_resonance_oracle`'s question.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kika.gnds.decode import readReactionSuite
from kika.gnds.resonances import readResonances
from kika.gnds.xpath import Document
from kika.nuclear_data.model import (BreitWigner, BreitWignerApproximation,
                                     ConversionReport, RMatrix, ScatteringRadius)


@pytest.fixture(scope="module")
def fe56(micro_fe56_gnds):
    return readReactionSuite(Document.parse(micro_fe56_gnds))


@pytest.fixture(scope="module")
def ta182(micro_ta182_gnds):
    return readReactionSuite(Document.parse(micro_ta182_gnds))


def _readBlock(path, root=None):
    """One ``resonances`` block on its own, for the planted-defect tests."""
    document = Document.parse(path) if root is None else Document(root=root)
    report = ConversionReport()
    resonances = readResonances(
        document.root.find("resonances"), "/reactionSuite", None, report,
        lambda element: None,
    )
    return resonances, report


# ---------------------------------------------------------------------------
# the block
# ---------------------------------------------------------------------------

def test_the_scattering_radius_arrives_with_the_unit_it_is_stated_in(fe56):
    """5.444 **fm**. The ENDF path gives 0.5444 for the same evaluation.

    ENDF writes AP in units of 10^-12 cm — ten femtometres — so both are the
    same radius stated differently. Before this field existed both numbers
    landed in ``constant`` and nothing said which was which.
    """
    suite, _ = fe56
    radius = suite.resonances.scatteringRadius
    assert isinstance(radius, ScatteringRadius)
    assert radius.constant == 5.444
    assert radius.unit == "fm"
    assert not radius.isEnergyDependent


def test_the_region_domains_are_read_with_their_unit(fe56, ta182):
    suite, _ = fe56
    region = suite.resonances.resolved[0]
    assert (region.domainMin, region.domainMax) == (1e-5, 8.5e5)
    assert region.domainUnit == "eV"

    suite, _ = ta182
    assert suite.resonances.domain == (1e-5, 1e4)
    assert suite.resonances.unresolved.domainMin == 35.0


# ---------------------------------------------------------------------------
# RMatrix
# ---------------------------------------------------------------------------

def test_the_rmatrix_reads_its_formalism_attributes(fe56):
    suite, _ = fe56
    formalism = suite.resonances.resolved[0].formalism
    assert isinstance(formalism, RMatrix)
    assert formalism.approximation == "ReichMoore"
    assert formalism.boundaryCondition == "EliminateShiftFunction"
    assert formalism.label == "eval"
    assert formalism.numberOfResonances == 312


def test_a_resonance_reaction_keeps_the_link_that_ties_it_to_a_reaction(fe56):
    """§19.3.3's ``<link href=...>``. A matching ``label`` is a convention the
    files keep; the href is the statement the format actually makes."""
    suite, _ = fe56
    reactions = suite.resonances.resolved[0].formalism.resonanceReactions
    assert [r.label for r in reactions] == [
        "Fe57 + photon [inclusive]", "n + Fe56"
    ]
    capture, elastic = reactions
    assert capture.eliminated and capture.ejectile == "photon"
    assert not elastic.eliminated and elastic.ejectile == "n"
    assert elastic.href == "/reactionSuite/reactions/reaction[@label='n + Fe56']"


def test_the_spin_groups_carry_their_J_and_parity_as_numbers(fe56):
    """§3.4's spins are fractions in the file (``spin="1/2"``) and floats here."""
    suite, _ = fe56
    groups = suite.resonances.resolved[0].formalism.spinGroups
    assert [(g.spin, g.parity) for g in groups] == [
        (0.5, 1), (0.5, -1), (1.5, -1), (1.5, 1), (2.5, 1)
    ]
    assert [len(g) for g in groups] == [40, 63, 78, 75, 56]


def test_a_width_belongs_to_a_channel_and_not_to_a_column(fe56):
    """The whole reason §19.3 is shaped the way it is.

    ``channel/@columnIndex`` says which table column holds this channel's width,
    and the reader uses it rather than assuming Reich-Moore's two-column layout
    — spin groups in this library carry two to six width columns.
    """
    suite, _ = fe56
    group = suite.resonances.resolved[0].formalism.spinGroups[0]
    byReaction = {c.resonanceReaction: i for i, c in enumerate(group.channels)}
    assert [c.columnIndex for c in group.channels] == [1, 2]
    assert [c.L for c in group.channels] == [0, 0]
    assert [c.channelSpin for c in group.channels] == [1.0, 0.5]

    assert group.energies[0] == -473000.0
    row = group.widths[0]
    assert row[byReaction["Fe57 + photon [inclusive]"]] == 1.0
    assert row[byReaction["n + Fe56"]] == 308000.0


def test_every_width_row_is_as_wide_as_the_channel_list(fe56):
    """The model asserts this in ``__post_init__``; here it is on a real file."""
    suite, _ = fe56
    for group in suite.resonances.resolved[0].formalism.spinGroups:
        assert len(group.widths) == len(group.energies)
        assert all(len(row) == len(group.channels) for row in group.widths)


# ---------------------------------------------------------------------------
# BreitWigner
# ---------------------------------------------------------------------------

def test_the_breit_wigner_table_is_regrouped_into_l_blocks(ta182):
    """GNDS states one flat table with ``L`` as a column; the model groups by L,
    which is ENDF's l-block and what a reconstructor iterates."""
    suite, _ = ta182
    formalism = suite.resonances.resolved[0].formalism
    assert isinstance(formalism, BreitWigner)
    assert formalism.approximation is BreitWignerApproximation.multiLevel
    assert formalism.calculateChannelRadius is True
    assert [g.L for g in formalism.resonanceParameters.spinGroups] == [0]
    assert formalism.numberOfResonances == 10


def test_the_breit_wigner_widths_are_named_rather_than_numbered(ta182):
    """``c3..c6`` is what this package exists to replace."""
    suite, _ = ta182
    resonance = suite.resonances.resolved[0].formalism.resonanceParameters \
        .spinGroups[0].resonances[0]
    assert resonance.energy == -20.0
    assert resonance.spin == 2.5
    assert resonance.totalWidth == 0.697
    assert resonance.neutronWidth == 0.63
    assert resonance.captureWidth == 0.067
    # Ta-182's table has no fissionWidth column; the field is 0, not invented.
    assert resonance.fissionWidth == 0.0


def test_a_breit_wigner_table_missing_an_index_column_is_refused(micro_ta182_gnds):
    """Without ``L`` there is nothing to group by, and grouping everything into
    L=0 would be a plausible-looking wrong answer."""
    tree = ET.parse(micro_ta182_gnds)
    headers = tree.getroot().find(".//BreitWigner/resonanceParameters/table/columnHeaders")
    headers.remove(headers.find("column[@name='L']"))

    resonances, report = _readBlock(None, tree.getroot())
    assert resonances.resolved[0].formalism.numberOfResonances == 0
    assert any("has no 'L' column" in entry for entry in report.unsupported)


# ---------------------------------------------------------------------------
# tabulatedWidths
# ---------------------------------------------------------------------------

def test_the_unresolved_region_reads_its_l_and_j_nesting(ta182):
    suite, _ = ta182
    widths = suite.resonances.unresolved.tabulatedWidths
    assert widths.label == "eval"
    assert widths.selfShieldingOnly is False
    assert [(g.L, g.J) for g in widths.spinGroups] == [
        (0, 2.5), (0, 3.5), (1, 1.5), (1, 2.5), (1, 3.5), (1, 4.5)
    ]


def test_an_average_width_knows_which_channel_it_is_and_how_many_degrees(ta182):
    suite, _ = ta182
    group = suite.resonances.unresolved.tabulatedWidths.spinGroups[0]
    assert [c.label for c in group.channels] == [
        "n + Ta182", "Ta183 + photon [inclusive]"
    ]
    assert [c.degreesOfFreedom for c in group.channels] == [1.0, 0.0]
    np.testing.assert_allclose(group.channels[0].widths[:3], 0.001379)
    np.testing.assert_allclose(group.levelSpacing[:3], 7.697)


def test_one_shared_grid_is_lifted_to_the_block_and_the_curves_stop_repeating_it(
        ta182):
    """285 of the library's 351 unresolved blocks tabulate everything on one
    grid, and for those the model comes out the shape ENDF produces."""
    suite, _ = ta182
    widths = suite.resonances.unresolved.tabulatedWidths
    assert widths.energyGrid is not None
    np.testing.assert_allclose(widths.energyGrid[:3], [35.0, 50.0, 70.0])
    for group in widths.spinGroups:
        assert group.levelSpacingEnergies is None
        assert all(channel.energies is None for channel in group.channels)


def test_a_second_grid_keeps_every_curve_on_its_own_and_says_so(micro_ta182_gnds):
    """66 of the 351 blocks do this — up to seven grids in one block.

    Picking one grid and attaching it to all of them would give average widths
    that are wrong at every energy with nothing in the result saying so, which
    is why the block-level grid is left unset instead.
    """
    tree = ET.parse(micro_ta182_gnds)
    values = tree.getroot().find(
        ".//unresolved//Ls/L/Js/J/widths/width/XYs1d/values"
    )
    numbers = values.text.split()
    numbers[0] = "36"          # move one abscissa off the shared grid
    values.text = " ".join(numbers)

    resonances, report = _readBlock(None, tree.getroot())
    widths = resonances.unresolved.tabulatedWidths
    assert widths.energyGrid is None
    assert any("different energy grids" in w for w in report.warnings), report.warnings
    grids = [c.energies for g in widths.spinGroups for c in g.channels]
    assert all(grid is not None for grid in grids)


def test_a_constant_average_width_is_a_constant_and_not_a_one_point_table(
        micro_ta182_gnds):
    """253 widths in the library are a ``constant1d``, and the model has a field
    for exactly that — a fabricated two-point table would look like data."""
    tree = ET.parse(micro_ta182_gnds)
    width = tree.getroot().find(".//unresolved//widths/width")
    for child in list(width):
        width.remove(child)
    ET.SubElement(width, "constant1d").attrib.update(
        {"label": "eval", "value": "0.5", "domainMin": "35", "domainMax": "1e4"}
    )

    resonances, _ = _readBlock(None, tree.getroot())
    channel = resonances.unresolved.tabulatedWidths.spinGroups[0].channels[0]
    assert channel.constantWidth == 0.5
    assert channel.widths is None


# ---------------------------------------------------------------------------
# the failures that must not become plausible numbers
# ---------------------------------------------------------------------------

def test_a_table_whose_data_is_short_is_refused_rather_than_reshaped(
        micro_fe56_gnds):
    """A reshape that "works" puts every energy after the first row into the
    wrong column, and the only place it shows is the cross section."""
    tree = ET.parse(micro_fe56_gnds)
    data = tree.getroot().find(".//spinGroup/resonanceParameters/table/data")
    data.text = " ".join(data.text.split()[:-3])

    resonances, report = _readBlock(None, tree.getroot())
    groups = resonances.resolved[0].formalism.spinGroups
    assert len(groups[0]) == 0                    # the short one, not guessed at
    assert len(groups[1]) == 63                   # the rest still read
    assert any("is not read rather than reshaped" in entry
               for entry in report.unsupported)


def test_an_unmodelled_resolved_formalism_is_reported(micro_fe56_gnds):
    tree = ET.parse(micro_fe56_gnds)
    resolved = tree.getroot().find("resonances/resolved")
    resolved.remove(resolved.find("RMatrix"))
    ET.SubElement(resolved, "energyIntervals")

    resonances, report = _readBlock(None, tree.getroot())
    assert resonances.resolved[0].formalism is None
    assert any("/resolved/energyIntervals:" in entry
               for entry in report.unsupported)


# ---------------------------------------------------------------------------
# what the node registry turned up: two flags and a nested PoPs
# ---------------------------------------------------------------------------

def test_the_two_r_matrix_flags_are_read_and_not_only_written(micro_fe56_gnds):
    """§19.3.1's ``reducedWidthAmplitudes`` and ``relativisticKinematics``.

    Both were **written** from the model (``encode_resonances.py:161-162``) and
    read by nobody, so a file declaring either came back out of kika with it
    ``False``. ``reducedWidthAmplitudes`` is ENDF's **IFG**: it says whether the
    ``widths`` column holds widths in eV or reduced-width amplitudes in eV^½,
    and the two are not interchangeable — losing it does not drop a flag, it
    silently reinterprets every width in the table.

    Planted rather than fixture-borne: no committed evaluation declares either,
    which is exactly why nothing noticed.
    """
    tree = ET.parse(micro_fe56_gnds)
    rMatrix = tree.getroot().find("resonances/resolved/RMatrix")
    assert "reducedWidthAmplitudes" not in rMatrix.attrib, "fixture changed"
    rMatrix.attrib["reducedWidthAmplitudes"] = "true"
    rMatrix.attrib["relativisticKinematics"] = "true"

    resonances, _ = _readBlock(None, tree.getroot())
    formalism = resonances.resolved[0].formalism
    assert formalism.reducedWidthAmplitudes is True
    assert formalism.relativisticKinematics is True


def test_a_file_that_declares_neither_flag_still_reads_them_false(fe56):
    """The other half. GNDS omits a false flag rather than writing it, so
    ``absent`` must not become ``True`` by some helper's default."""
    suite, _ = fe56
    formalism = suite.resonances.resolved[0].formalism
    assert formalism.reducedWidthAmplitudes is False
    assert formalism.relativisticKinematics is False


def test_the_nested_pops_is_announced_instead_of_disappearing(micro_fe56_gnds,
                                                              tmp_path):
    """§19 admits a ``PoPs`` inside the formalism and every RMatrix in
    ENDF/B-VIII.1-GNDS carries one. kika reads it and does not write it —
    writing it is §12 work, blocked on ``gnds_endf_conflicts.md`` §3.3 — and it
    was the **only** node kika read and dropped without a report entry.

    The writer is unchanged. What is asserted is that the silence is gone.
    """
    import kika

    suite = kika.read(micro_fe56_gnds, covariances=False)
    assert suite.resonances.resolved[0].formalism.PoPs is not None

    report = kika.write(suite, tmp_path / "out.gnds.xml")
    assert any("nested <PoPs>" in loss for loss in report.losses), report.losses


# ---------------------------------------------------------------------------
# §19.3.3's second radius — read since 2026-08-24
# ---------------------------------------------------------------------------

def _graftReactionHardSphereRadius(source, tmp_path, value="8.0", unit="fm"):
    """Put a ``<hardSphereRadius>`` on the first ``<resonanceReaction>``.

    Grafted, not found: four nodes in the whole distributed library carry one
    (V-51, Ca-40, Cl-35 — measured 2026-08-24 over the 558 neutron
    evaluations) and the smallest of those files is 5,9 MB, far past what
    belongs in ``tests/data``. Same argument, and the same technique, as the
    forms ``test_distributions.py`` grafts in.
    """
    tree = ET.parse(source)
    reaction = tree.getroot().find(".//resonanceReaction")
    assert reaction is not None, source
    reaction.append(ET.fromstring(
        f'<hardSphereRadius><constant1d label="hardSphereRadius" '
        f'value="{value}" domainMin="1e-5" domainMax="2e7">'
        f'<axes><axis index="1" label="energy_in" unit="eV"/>'
        f'<axis index="0" label="radius" unit="{unit}"/></axes>'
        f'</constant1d></hardSphereRadius>'))
    path = tmp_path / "grafted-radius.gnds.xml"
    tree.write(path)
    return path


def test_a_resonance_reactions_hard_sphere_radius_is_read_not_reported(
        micro_fe56_gnds, tmp_path):
    """It used to be dropped with a report entry, and the reason was wrong.

    The old message said the *channel's* radius is the one the phase shift
    uses. That is true, and it is a statement about which radius the physics
    reads — not about which one the file states. A reader that drops the second
    cannot write the file back, which is the test §6.4 was failing.
    """
    grafted = _graftReactionHardSphereRadius(micro_fe56_gnds, tmp_path)
    suite, report = readReactionSuite(Document.parse(grafted))

    reaction = suite.resonances.resolved[0].formalism.resonanceReactions[0]
    assert reaction.hardSphereRadius == 8.0
    assert reaction.radiusUnit == "fm"
    assert not [line for line in report.losses if "hardSphereRadius" in line]


def test_the_reactions_hard_sphere_radius_survives_a_write(micro_fe56_gnds,
                                                           tmp_path):
    """Read is half of it: the point of modelling it was writing it back."""
    import kika

    grafted = _graftReactionHardSphereRadius(micro_fe56_gnds, tmp_path)
    suite, _ = readReactionSuite(Document.parse(grafted))
    written = tmp_path / "out.gnds.xml"
    kika.write(suite, written)

    reaction = ET.parse(written).getroot().find(".//resonanceReaction")
    radius = reaction.find("hardSphereRadius/constant1d")
    assert radius is not None, "the radius did not survive to the output"
    assert float(radius.attrib["value"]) == 8.0
    # The schema's order: scatteringRadius, then hardSphereRadius. Fe-56's
    # resonanceReaction states no scatteringRadius, so this one is first.
    assert [child.tag for child in reaction] == ["link", "hardSphereRadius"]


def test_supports_angular_reconstruction_stays_a_report_entry(ta182):
    """The one row of §6.4 that closes as won't-fix rather than as work.

    65 files set it and it still gets no node: there is no number under it. It
    qualifies parameters that are all read, and modelling it would state inside
    a format-neutral model what one particular reader can do with them.
    """
    _suite, report = ta182
    said = "\n".join(report.losses)
    if "supportsAngularReconstruction" not in said:
        pytest.skip("this fixture does not set the attribute")
    assert "FUDGE's capability and not the evaluation's" in said


# ---------------------------------------------------------------------------
# §19.3.4's externalRMatrix — the last open row of §6.4, closed 2026-08-24
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sr88(micro_sr88_gnds):
    return readReactionSuite(Document.parse(micro_sr88_gnds))


def _externalRMatrices(suite):
    """Every ``externalRMatrix`` on the suite's R-matrix, channel order kept."""
    formalism = suite.resonances.resolved[0].formalism
    return [channel.externalRMatrix
            for group in formalism.spinGroups
            for channel in group.channels
            if channel.externalRMatrix is not None]


def test_the_external_r_matrix_is_read_from_the_only_file_that_has_one(sr88):
    """Seven nodes, all SAMMY, all seven terms — the whole population.

    This is not a sample of §19.3.4's node, it **is** §19.3.4's node: 7
    occurrences across the 558 distributed neutron evaluations, every one of
    them in ``n-038_Sr_088``, measured 2026-08-24. Until this fixture existed
    the reader named the node and dropped it, and there was nothing that could
    have gated doing better.

    The terms are asserted with their units because the units are the half a
    reader is most likely to get wrong: they are **not uniform** — ``1/eV``,
    ``1/eV**2``, ``eV`` and dimensionless in one node — and a term read without
    one is a number the reconstruction cannot use.
    """
    suite, report = sr88
    externals = _externalRMatrices(suite)
    assert len(externals) == 7
    assert {external.type for external in externals} == {"SAMMY"}

    stated = [(term.label, term.value, term.unit) for term in externals[0].terms]
    assert stated == [
        ("constantExternalR", -0.043, ""),
        ("linearExternalR", 2.8e-8, "1/eV"),
        ("quadraticExternalR", 0.0, "1/eV**2"),
        ("constantLogarithmicCoefficient", 0.01, ""),
        ("linearLogarithmicCoefficient", 0.0, "1/eV"),
        ("singularityEnergyBelow", 0.0, "eV"),
        ("singularityEnergyAbove", 9.55e5, "eV"),
    ]
    assert not [entry for entry in report.unsupported
                if "externalRMatrix" in entry]


def test_a_term_the_file_omits_is_none_and_not_zero(sr88):
    """``None`` and zero are different statements and only one is the file's.

    ``quadraticExternalR`` **is** stated here, and stated as zero, so the two
    readings are distinguishable on this very node: asking for a term the file
    does not carry has to come back ``None``. Folding the two together is what
    FUDGE's ``getTerm`` does at the point of *use*, which is the right place —
    doing it at the point of reading would make a file that says nothing
    indistinguishable from one that says zero.
    """
    external = _externalRMatrices(sr88[0])[0]
    assert external.term("quadraticExternalR").value == 0.0
    assert external.term("averageRadiationWidth") is None


def test_the_external_r_matrix_survives_a_round_trip(micro_sr88_gnds, tmp_path):
    """Read is half of it; §6.4's argument for modelling it was writing it back.

    The channel's children are asserted **in order**, and that assert is not
    decoration: ``RML_ChannelType`` (``gnds.xsd:915-919``) is an ``xs:sequence``
    with ``externalRMatrix`` ahead of both radii, so a writer that appended it
    where it was convenient would emit a file no validator accepts — measured,
    by moving the node to the end of a written file and watching the schema
    reject it. It is the same lesson §25.3's ``parameterCovariances`` container
    taught on 2026-08-19, one node along.
    """
    import kika

    suite, _ = readReactionSuite(Document.parse(micro_sr88_gnds))
    written = tmp_path / "out.gnds.xml"
    kika.write(suite, written)

    root = ET.parse(written).getroot()
    channels = [channel for channel in root.iter("channel")
                if channel.find("externalRMatrix") is not None]
    assert len(channels) == 7
    assert [child.tag for child in channels[0]] == ["externalRMatrix",
                                                    "hardSphereRadius"]

    again, _ = readReactionSuite(Document.parse(written))
    before = _externalRMatrices(suite)
    after = _externalRMatrices(again)
    assert [(e.type, [(t.label, t.value, t.unit) for t in e.terms])
            for e in after] == \
           [(e.type, [(t.label, t.value, t.unit) for t in e.terms])
            for e in before]


@pytest.mark.parametrize("break_it, expected", [
    (lambda node: node.attrib.__setitem__("type", "Bayes"), "is not one of"),
    (lambda node: node.remove(node.findall("double")[-1]), "is missing"),
    (lambda node: node.append(copy.deepcopy(node.findall("double")[0])),
     "more than one term labelled"),
])
def test_an_external_r_matrix_the_model_refuses_is_reported_not_repaired(
        micro_sr88_gnds, break_it, expected):
    """A file kika cannot represent is named, not guessed at.

    The three defects are the three invariants the model states, and each is a
    place where repairing would mean **choosing a formula**: SAMMY's
    parametrisation is purely real and Froehner's has an imaginary part, so an
    unknown ``type`` is not a label to default and a missing singularity energy
    is not a zero to supply. The channel around it still reads — losing a
    background R-matrix must not lose the resonances.
    """
    tree = ET.parse(micro_sr88_gnds)
    node = tree.getroot().find(".//channel/externalRMatrix")
    assert node is not None, "fixture changed"
    break_it(node)

    resonances, report = _readBlock(None, tree.getroot())
    externals = [channel.externalRMatrix
                 for group in resonances.resolved[0].formalism.spinGroups
                 for channel in group.channels]
    assert externals.count(None) == len(externals) - 6   # the broken one dropped
    assert any(expected in entry and "externalRMatrix" in entry
               for entry in report.unsupported)


def test_a_channel_boundary_condition_is_read_and_written(micro_sr88_gnds,
                                                          tmp_path):
    """ENDF's **BND**, found on the way to the node above and fixed with it.

    ``Channel.boundaryConditionValue`` has been on the model since B1a and the
    ENDF adapter fills it from every LRF=7 tape, but the GNDS side neither read
    nor wrote it — so an ENDF -> GNDS conversion dropped the evaluator's
    boundary condition in silence, which is the worst shape a loss can take.

    **Zero of the 558 distributed GNDS evaluations state the attribute**
    (measured 2026-08-24), so this is planted rather than fixture-borne, and
    that measurement is also the reason nothing noticed: no oracle and no round
    trip over the corpus could have.
    """
    import kika

    tree = ET.parse(micro_sr88_gnds)
    channel = tree.getroot().find(".//spinGroup/channels/channel")
    assert "boundaryConditionValue" not in channel.attrib, "fixture changed"
    channel.attrib["boundaryConditionValue"] = "-1.5"
    source = tmp_path / "planted.gnds.xml"
    tree.write(source, encoding="UTF-8", xml_declaration=True)

    suite, _ = readReactionSuite(Document.parse(source))
    first = suite.resonances.resolved[0].formalism.spinGroups[0].channels[0]
    assert first.boundaryConditionValue == -1.5

    written = tmp_path / "out.gnds.xml"
    kika.write(suite, written)
    back = ET.parse(written).getroot().find(".//spinGroup/channels/channel")
    assert back.attrib["boundaryConditionValue"] == "-1.5"
