"""§19 resonances, against the two trimmed evaluations that cover it.

``micro_fe56`` carries an ``RMatrix``; ``micro_ta182`` carries a
``BreitWigner`` and a ``tabulatedWidths``. Between them that is every §19
formalism in ENDF/B-VIII.1-GNDS, and in both files the ``resonances`` subtree is
**verbatim** — the build script never touches it, which is the reason those two
files exist.

Numbers here may therefore be asserted, unlike the cross sections in the same
fixtures. Whether they are the *right* numbers is
:mod:`kika.gnds.tests.test_resonance_oracle`'s question.
"""
from __future__ import annotations

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


def test_an_external_r_matrix_is_named_rather_than_ignored(micro_fe56_gnds):
    """7 channels in the whole library carry one and kika has no node for it."""
    tree = ET.parse(micro_fe56_gnds)
    channel = tree.getroot().find(".//spinGroup/channels/channel")
    ET.SubElement(channel, "externalRMatrix")

    resonances, report = _readBlock(None, tree.getroot())
    assert len(resonances.resolved[0].formalism.spinGroups) == 5
    assert any("externalRMatrix" in entry for entry in report.unsupported)


def test_an_unmodelled_resolved_formalism_is_reported(micro_fe56_gnds):
    tree = ET.parse(micro_fe56_gnds)
    resolved = tree.getroot().find("resonances/resolved")
    resolved.remove(resolved.find("RMatrix"))
    ET.SubElement(resolved, "energyIntervals")

    resonances, report = _readBlock(None, tree.getroot())
    assert resonances.resolved[0].formalism is None
    assert any("/resolved/energyIntervals:" in entry
               for entry in report.unsupported)
