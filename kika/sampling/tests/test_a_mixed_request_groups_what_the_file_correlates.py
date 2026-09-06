"""Asking for two quantities at once must not change either of them.

The reason a combined pipeline was never written is that combining looked like
a decision: draw the cross sections and the angular distributions together, or
apart? :mod:`kika.sampling.joint_blocks` refuses the decision and reads the
answer off the file, so what has to be held here is that the reading is right
in both directions.

**Direction one, and it is the one that makes the feature safe to use.** A
request for MF33 alone has to assemble the matrix ``mf33_sampling`` ships, bit
for bit, and a request for MF34 alone the matrix ``endf_perturbation`` ships;
and a request for both has to assemble those same two, unchanged, as two blocks
that are drawn independently. If that holds, adding a quantity to a request
costs nothing and no existing result moves.

**Direction two.** When a section *does* state a block across the two files,
the two must land in one matrix and be drawn once -- with the off-diagonal in
it, at the right rows, and symmetric. No committed tape states such a block, so
that half is measured against a fabricated entry built on the real grids of a
real tape. Fabricating it is the honest option and the alternative was worse:
asserting the merge on a tape that has nothing to merge would have passed
against code that cannot merge at all.

``micro_fe56_cov.endf`` is the fixture that makes this file possible -- it is
the one committed tape carrying MF33 and MF34 for the same reaction (MT2), so
"the same evaluation states both quantities" is a real situation here and not
two tapes pretending to be one. Its MF3 is truncated and does not parse, which
does not matter: nothing here applies a perturbation, it only assembles
covariances.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling.joint_blocks import (ComponentKey, Selection, assembleRequest,
                                        collectEntries, describeRequest,
                                        requestIndex, samplingGroups)
from kika.sampling.model_blocks import (cross_section_covariance_blocks,
                                        legendre_covariance_blocks)

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
MIXED = str(DATA / "micro_fe56_cov.endf")            # MF33/MT2 and MF34/MT2, L=1,2
MF33_ONLY = str(DATA / "micro_fe56_mf33.endf")       # MF33/MT4 and MT16, uncorrelated
MF34_ONLY = str(DATA / "micro_fe56_structural.endf")  # MF34/MT2, L=1..6, three grids


def _suite(path):
    from kika.endf.model_adapter import decodeCovarianceSuite

    suite, _report = decodeCovarianceSuite(read_endf(path))
    return suite


@pytest.fixture(scope="module")
def mixed():
    return _suite(MIXED)


@pytest.fixture(scope="module")
def mf33Only():
    return _suite(MF33_ONLY)


@pytest.fixture(scope="module")
def mf34Only():
    return _suite(MF34_ONLY)


# ----------------------------------------------------------------------
# Direction one: a request assembles what the single-MF entry points assemble
# ----------------------------------------------------------------------

@pytest.mark.parametrize("path", [MF33_ONLY, MIXED])
def test_an_mf33_request_assembles_the_matrix_the_mf33_pipeline_ships(path):
    """Bit for bit, including the key order the carrier's layout depends on."""
    suite = _suite(path)
    entries = collectEntries(suite, {33: None})
    blocks, index = assembleRequest(entries)

    (_shippedKey, shipped), = cross_section_covariance_blocks(
        suite, mt=None, union="global", mf=33, relative=True)

    assert len(blocks) == 1
    (_key, mine), = blocks
    assert np.array_equal(mine, shipped)

    (meta,) = index.values()
    assert [k.mf for k in meta["components"]] == [33] * len(meta["components"])
    assert meta["union"] == "global"
    assert meta["dimension"] == mine.shape[0]


@pytest.mark.parametrize("path", [MF34_ONLY, MIXED])
def test_an_mf34_request_assembles_the_matrix_the_mf34_pipeline_ships(path):
    """Same claim for the angular side, on a tape whose orders differ in grid.

    ``micro_fe56_structural.endf`` states L=1..6 on 42, 23 and 12 bins, so the
    per-component union and its zero padding are exercised rather than assumed.
    """
    suite = _suite(path)
    entries = collectEntries(suite, {34: None})
    blocks, index = assembleRequest(entries)

    (_shippedKey, shipped), = legendre_covariance_blocks(suite, relative=True)

    assert len(blocks) == 1
    (_key, mine), = blocks
    assert np.array_equal(mine, shipped)

    (meta,) = index.values()
    assert meta["union"] == "per-component"
    assert meta["dimension"] == mine.shape[0]


def test_asking_for_both_at_once_changes_neither(mixed):
    """The whole reason a combined request is safe: it is the two, side by side."""
    both, index = assembleRequest(collectEntries(mixed, {33: None, 34: None}))
    ((_k33, alone33),), _ = assembleRequest(collectEntries(mixed, {33: None}))
    ((_k34, alone34),), _ = assembleRequest(collectEntries(mixed, {34: None}))

    byLabel = {key[1]: matrix for key, matrix in both}
    assert sorted(byLabel) == ["MF33", "MF34"]
    assert np.array_equal(byLabel["MF33"], alone33)
    assert np.array_equal(byLabel["MF34"], alone34)
    assert [meta["quantities"] for meta in index.values()] == [
        ["crossSection"], ["angularDistribution"]]


def test_two_quantities_the_file_does_not_correlate_are_two_draws(mixed):
    """And the description says so, because nothing else in a run will."""
    entries = collectEntries(mixed, {33: None, 34: None})
    groups = samplingGroups(entries)

    assert len(groups) == 2
    assert {k.mf for k in groups[0]} == {33}
    assert {k.mf for k in groups[1]} == {34}
    assert "no section states a block across two files" in describeRequest(entries)


# ----------------------------------------------------------------------
# Direction two: a stated cross-file block makes one draw of the two
# ----------------------------------------------------------------------

def _withFabricatedCrossBlock(entries, magnitude=0.25):
    """The same entries plus one block stating MF33/MT2 against MF34/MT2, L=1.

    Built on the two real grids of the tape, so the lift and the placement are
    exercised as they would be on a file that stated it. The values are a
    constant times an outer product of ones -- what matters here is where the
    block lands and that it is transposed into the mirror corner, not what is
    in it.
    """
    row = next(k for k in {e[0] for e in entries} if k.mf == 33)
    col = next(k for k in sorted({e[0] for e in entries}) if k.mf == 34)
    rowGrid = next(e[3] for e in entries if e[0] == row)
    colGrid = next(e[3] for e in entries if e[0] == col)
    cross = np.full((len(rowGrid) - 1, len(colGrid) - 1), magnitude, dtype=float)
    return entries + [(row, col, cross, rowGrid, colGrid)], row, col


def test_a_stated_cross_file_block_puts_both_quantities_in_one_draw(mixed):
    entries = collectEntries(mixed, {33: None, 34: None})
    withCross, row, col = _withFabricatedCrossBlock(entries)

    groups = samplingGroups(withCross)
    assert len(groups) == 1, "a stated block between two files is one covariance"
    assert {k.mf for k in groups[0]} == {33, 34}

    blocks, index = assembleRequest(withCross)
    (key, joint), = blocks
    assert key[1] == "MF33+MF34"

    meta = index[key]
    assert meta["union"] == "per-component", (
        "a mixed group cannot be pooled onto one grid: MF33's bins and MF34's "
        "are different quantities' bins")
    assert meta["quantities"] == ["angularDistribution", "crossSection"]
    assert joint.shape == (meta["dimension"],) * 2

    stride = meta["stride"]
    i = meta["components"].index(row)
    j = meta["components"].index(col)
    corner = joint[i * stride:i * stride + meta["widths"][row],
                   j * stride:j * stride + meta["widths"][col]]
    assert np.allclose(corner, 0.25)
    mirror = joint[j * stride:j * stride + meta["widths"][col],
                   i * stride:i * stride + meta["widths"][row]]
    assert np.array_equal(mirror, corner.T)


def test_the_merge_takes_whole_files_and_not_the_component_that_was_named(mixed):
    """L=2 joins the group too, though nothing states a block from it to MF33.

    Half a merge would put MF33 in a matrix with L=1 and leave L=1's correlation
    with L=2 outside it -- a covariance stating a correlation with a component
    that is not in the matrix, which is the same defect one level out that the
    entry builders' "both sides must pass" rule prevents.
    """
    entries = collectEntries(mixed, {33: None, 34: None})
    withCross, _row, col = _withFabricatedCrossBlock(entries)

    (group,) = samplingGroups(withCross, grouping="mf")
    orders = sorted(k.index for k in group if k.mf == 34)
    assert orders == [1, 2]
    assert col.index == 1

    stated = samplingGroups(withCross, grouping="stated")
    assert len(stated) == 1 and sorted(k.index for k in stated[0] if k.mf == 34) == [1, 2], (
        "here the two orders are correlated anyway, so both groupings agree")


def test_the_description_names_the_files_a_block_crosses(mixed):
    entries, _row, _col = _withFabricatedCrossBlock(
        collectEntries(mixed, {33: None, 34: None}))
    text = describeRequest(entries)
    assert "1 independent draw(s)" in text
    assert "stated cross-file blocks: [(33, 34)]" in text


# ----------------------------------------------------------------------
# The finer grouping, and what it costs
# ----------------------------------------------------------------------

def test_the_stated_grouping_splits_what_the_file_leaves_unconnected(mf33Only):
    """MT4 and MT16 are uncorrelated on this tape, so they can be two draws.

    Both partitions describe the same distribution; ``"mf"`` reproduces the
    shipped realisations and ``"stated"`` does not, which is why the default is
    the coarse one. What is asserted here is that the coarse joint really is
    block-diagonal over the two -- if it were not, splitting would be wrong
    rather than merely different.
    """
    entries = collectEntries(mf33Only, {33: None})
    assert len(samplingGroups(entries, grouping="mf")) == 1
    assert len(samplingGroups(entries, grouping="stated")) == 2

    ((_key, joint),), index = assembleRequest(entries, grouping="mf")
    (meta,) = index.values()
    stride = meta["stride"]
    offDiagonal = joint[:stride, stride:]
    assert np.array_equal(offDiagonal, np.zeros_like(offDiagonal))


def test_every_group_is_the_size_its_index_says(mixed, mf34Only):
    for suite, request in ((mixed, {33: None, 34: None}), (mf34Only, {34: None})):
        blocks, index = assembleRequest(collectEntries(suite, request))
        for key, matrix in blocks:
            meta = index[key]
            assert matrix.shape == (meta["dimension"],) * 2
            assert meta["dimension"] == len(meta["components"]) * meta["stride"]
            for component in meta["components"]:
                assert meta["widths"][component] <= meta["stride"]
                assert len(meta["grids"][component]) - 1 == meta["widths"][component]


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------

def test_a_selection_that_matches_nothing_is_refused(mixed):
    with pytest.raises(ValueError, match="mistake in the request"):
        collectEntries(mixed, {33: [16]})


def test_a_quantity_with_no_applier_is_refused_rather_than_assembled():
    with pytest.raises(ValueError, match="no model applier"):
        Selection(mf=35)


def test_an_index_on_a_quantity_that_has_none_is_refused():
    with pytest.raises(ValueError, match="no third coordinate"):
        Selection(mf=33, index=2)


def test_the_key_says_which_quantity_a_row_belongs_to():
    """The one thing a 2-tuple key could not say, and the reason for the 4-tuple."""
    assert ComponentKey(26056, 33, 2).quantity == "crossSection"
    assert ComponentKey(26056, 34, 2, 1).quantity == "angularDistribution"
    assert ComponentKey(26056, 31, 452).quantity == "multiplicity"
    assert "L=1" in ComponentKey(26056, 34, 2, 1).describe()
    assert ComponentKey(26056, 33, 2) < ComponentKey(26056, 34, 2, 1), (
        "cross sections sort ahead of distributions, so a mixed matrix has a "
        "stable row order")
