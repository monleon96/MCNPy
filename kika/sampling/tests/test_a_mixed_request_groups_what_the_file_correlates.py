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
from kika.sampling.joint_blocks import (SUPPORTED_MF, ComponentKey, Selection,
                                        assembleRequest, collectEntries,
                                        describeRequest, requestIndex,
                                        samplingGroups)
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
    """The rule, not the list: MF32 has no applier, so it cannot be assembled.

    MF35 was this test's subject until its model applier was written
    (``applySpectrumFactors``). It is now supported and MF32 stands in its
    place, which is the honest version of the same rule: resonance-parameter
    covariances have no applier and projecting one onto a cross-section grid is
    a physics calculation, not a formatting one.
    """
    with pytest.raises(ValueError, match="no model applier"):
        Selection(mf=32)


def test_the_fission_spectrum_is_assembled_now_that_it_can_be_applied():
    assert 35 in SUPPORTED_MF
    assert Selection(mf=35, index=[0, 2]).index == [0, 2], (
        "MF35's third coordinate is the incident-energy band")


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


# ----------------------------------------------------------------------
# The multiplicity: assembled and drawn, and stopping exactly where it should
# ----------------------------------------------------------------------

NUBAR = str(DATA / "micro_u235_nubar.endf")


def test_a_multiplicity_request_assembles_and_draws_like_any_other():
    """MF31's components are reactions on an energy grid, same as MF33's.

    Worth a test of its own because the *applier* for a multiplicity does not
    exist yet, and that could easily be mistaken for the whole path being
    absent. It is not: the covariance assembles, the draw is a draw, and the
    envelope cuts it. Only the last step -- putting the realisation on a model
    node -- is missing, and it is missing on a decision rather than on work.
    """
    from kika.sampling.core import draw_samples
    from kika.sampling.perturbation_set import PerturbationSet

    suite = _suite(NUBAR)
    entries = collectEntries(suite, {31: None})
    blocks, index = assembleRequest(entries)

    (_key, matrix), = blocks
    (meta,) = index.values()
    assert meta["quantities"] == ["multiplicity"]
    assert sorted(component.mt for component in meta["components"]) == [452, 455, 456]
    assert meta["union"] == "global", (
        "MF31 is laid out like MF33: §31.1 makes its formats directly analogous")
    assert matrix.shape == (meta["dimension"],) * 2

    samples, _diagnostics = draw_samples(blocks, 2, returns="factors",
                                         space="log", seed=1, psd_method="none",
                                         null_tol=None, verbose=False)
    pset = PerturbationSet.fromDraw({key: samples[key][0] for key in samples},
                                    index, label="realization-0000")
    assert pset.reactions() == (452, 455, 456)
    assert all(np.all(pset.factors[component] > 0.0)
               for component in pset.components())


def test_a_multiplicity_needs_a_resolver_and_says_why():
    """Which model node an ENDF MT names is the adapter's question, not the model's.

    MT456 is the fission product's own multiplicity and MT452/455 are
    ``multiplicitySum`` nodes; nothing in the sampling layer knows that, and
    importing the adapter to find out is what the layering ratchet exists to
    stop. So the caller hands in ``nubarNode`` -- the same lookup the MF1
    encoders use, which is what makes "what was perturbed" and "what gets
    written" the same node.
    """
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.sampling.perturbation_set import PerturbationSet

    reactions, _report = decodeReactionSuite(read_endf(NUBAR))
    component = ComponentKey(92235, 31, 452)
    pset = PerturbationSet(label="realization-0000",
                           factors={component: np.array([1.1])},
                           binEdges={component: np.array([1.0e-5, 2.0e7])})
    with pytest.raises(ValueError, match="needs a resolver"):
        pset.applyToSuite(reactions)


def test_a_full_nubar_family_comes_out_satisfying_the_sum_rule():
    """ENDF-6 requires nu_452 = nu_455 + nu_456, and this is what enforces it.

    Perturbing the components member by member does **not** satisfy it -- the
    total would keep its own baseline while its parts moved, which is what this
    path did before the family rule was ported and is why the numbers below are
    worth pinning: MT452 stayed at 2.5178 at 1 MeV against a prompt that had gone
    to 2.4967.

    The rule is ``perturb_nubar_family``'s: perturb the components, derive the
    redundant member from them, discard the derived member's own factor block.
    Because no member carries a duplicate energy, the sum on the union of their
    grids is a lin-lin identity, so the rule holds to machine precision rather
    than approximately.
    """
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.model_adapter.multiplicity import nubarNode
    from kika.sampling.perturbation_set import PerturbationSet

    reactions, _report = decodeReactionSuite(read_endf(NUBAR))
    edges = np.geomspace(1.0e-5, 2.0e7, 8)
    components = [ComponentKey(92235, 31, mt) for mt in (455, 456)]
    pset = PerturbationSet(
        label="realization-0000",
        factors={component: 1.0 + 0.05 * np.cos(np.arange(edges.size - 1,
                                                          dtype=float) + n)
                 for n, component in enumerate(components)},
        binEdges={component: edges for component in components},
    )
    diagnostics = pset.applyToSuite(reactions, multiplicityResolver=nubarNode)

    total = ComponentKey(92235, 31, 452)
    assert set(diagnostics) == set(components) | {total}, (
        "the derived member is part of this realisation even though the request "
        "did not name it -- it was rewritten, so it has to be reported and "
        "emitted")
    assert diagnostics[total]["derived_from_sum_rule"] is True
    assert diagnostics[total]["contributors"] == [455, 456]
    # Rebuilding repairs the input file's own residual, which moves the central
    # value by something nobody asked for. The size of it is on record.
    assert 0.0 < diagnostics[total]["baseline_residual"]["max_bin_rel"] < 1e-4

    # Beside the evaluation, not instead of it -- the whole reason
    # `Multiplicity` is a `Component`.
    from kika.nuclear_data.model import EVAL_LABEL

    for mt in (452, 455, 456):
        node = nubarNode(reactions, mt)
        assert sorted(node.keys()) == [EVAL_LABEL, "realization-0000"]
        assert node.form is node[EVAL_LABEL], (
            "`.form` still means the evaluated form, so every reader that asks "
            "for 'the' nu-bar keeps getting the one it always got")

    def table(mt):
        xs, ys, _pairs = nubarNode(reactions, mt)["realization-0000"].toEndfRegions()
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    e452, v452 = table(452)
    e455, v455 = table(455)
    e456, v456 = table(456)
    probe = np.geomspace(1.0e-3, 1.0e7, 25)
    total = np.interp(probe, e452, v452)
    parts = np.interp(probe, e455, v455) + np.interp(probe, e456, v456)
    assert np.max(np.abs(total - parts)) / np.max(np.abs(total)) < 1e-14


def test_the_family_rule_on_the_model_reproduces_the_one_on_the_sections():
    """Two implementations of one convention, and they have to stay one.

    ``perturb_nubar_family`` works on MF1 sections and
    ``perturbNubarFamilyOnModel`` on model nodes; they live in the same file so
    that a reader changing one sees the other, and this is what says they still
    agree. Point for point, not approximately: both start from the same table --
    the adapter's decode of MT452/455/456 is bit-identical to the section's
    arrays -- and apply the same factors, so any difference is a difference of
    rule.
    """
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.model_adapter.multiplicity import nubarNode
    from kika.sampling.mf31_sampling import (_modelNubarTable,
                                             _nubar_as_tabulated,
                                             perturbNubarFamilyOnModel,
                                             perturb_nubar_family)

    endf = read_endf(NUBAR)
    reactions, _report = decodeReactionSuite(endf)
    bins = np.geomspace(1.0e-5, 2.0e7, 12)
    blocks = {455: 1.0 + 0.05 * np.cos(np.arange(bins.size - 1, dtype=float)),
              456: 1.0 - 0.03 * np.sin(np.arange(bins.size - 1, dtype=float))}

    sections = {mt: endf.get_file(1).sections[mt] for mt in (452, 455, 456)}
    fromSections, _diagS = perturb_nubar_family(sections, blocks, bins)
    forms = {mt: nubarNode(reactions, mt).form for mt in (452, 455, 456)}
    fromModel, _diagM = perturbNubarFamilyOnModel(forms, blocks, bins)

    for mt in (452, 455, 456):
        e1, v1 = _nubar_as_tabulated(fromSections[mt])
        e2, v2 = _modelNubarTable(fromModel[mt])
        assert e1.size == e2.size, f"MT{mt}: {e1.size} points vs {e2.size}"
        assert np.array_equal(e1, e2), f"MT{mt}: the energies differ"
        assert np.array_equal(v1, v2), f"MT{mt}: the values differ"


def test_a_component_with_no_block_of_its_own_rides_the_total():
    """One of the four coverage patterns, and the one that is easy to get wrong.

    When only MT452 carries covariance, its factor rides onto both components so
    that all three scale together and the rule still holds -- "perturb everything
    with the total". Perturbing the total alone and leaving the components would
    break it in the other direction.
    """
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.model_adapter.multiplicity import nubarNode
    from kika.sampling.mf31_sampling import (_modelNubarTable,
                                             perturbNubarFamilyOnModel)

    reactions, _report = decodeReactionSuite(read_endf(NUBAR))
    bins = np.array([1.0e-5, 1.0e6, 2.0e7])
    forms = {mt: nubarNode(reactions, mt).form for mt in (452, 455, 456)}

    out, diagnostics = perturbNubarFamilyOnModel(forms, {452: np.array([1.5, 0.5])},
                                                 bins)

    assert sorted(diagnostics) == [452, 455, 456], (
        "the components are perturbed and the total derived, and all three are "
        "reported -- the total because it was rewritten")
    assert diagnostics[452]["derived_from_sum_rule"] is True
    assert 455 in diagnostics[455] or "max_factor" in diagnostics[455], (
        "the components should carry an applier's diagnostics, not a derivation")
    assert "derived_from_sum_rule" not in diagnostics[456]
    e452, v452 = _modelNubarTable(out[452])
    e455, v455 = _modelNubarTable(out[455])
    e456, v456 = _modelNubarTable(out[456])
    probe = np.array([1.0e3, 5.0e6])
    total = np.interp(probe, e452, v452)
    parts = np.interp(probe, e455, v455) + np.interp(probe, e456, v456)
    assert np.allclose(total, parts, rtol=1e-14)

    # Against the sum of the *baseline components*, not against the baseline
    # total: the file's own MT452 is not exactly nu_455 + nu_456 -- that is the
    # residual `sum_rule_residual` measures and the rebuild repairs -- so
    # comparing with it would be asserting that the repair did not happen.
    baseParts = (np.interp(probe, *_modelNubarTable(forms[455]))
                 + np.interp(probe, *_modelNubarTable(forms[456])))
    assert total[0] == pytest.approx(1.5 * baseParts[0], rel=1e-6)
    assert total[1] == pytest.approx(0.5 * baseParts[1], rel=1e-6)

    # And the repair is small but not zero, which is why it is on record.
    base452 = np.interp(probe, *_modelNubarTable(forms[452]))
    assert 0.0 < abs(baseParts[0] - base452[0]) / base452[0] < 1e-4


def test_a_family_that_cannot_be_derived_is_perturbed_member_by_member():
    """The case the shipped applier also perturbs directly, and the same way.

    ``perturb_nubar_family``'s ``can_derive`` is false when the tape does not
    carry every contributor, and there it perturbs each present member on its
    own. So does this. The resolver is stubbed to report that shape rather than
    a second fixture being cut, because what is under test is the rule and the
    rule reads the family off the resolver.
    """
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.model_adapter.multiplicity import nubarNode
    from kika.sampling.perturbation_set import PerturbationSet

    reactions, _report = decodeReactionSuite(read_endf(NUBAR))

    def onlyTheTotal(suite, mt):
        return nubarNode(suite, mt) if mt == 452 else None

    component = ComponentKey(92235, 31, 452)
    edges = np.array([1.0e-5, 1.0e6, 2.0e7])
    pset = PerturbationSet(label="realization-0000",
                           factors={component: np.array([1.5, 0.5])},
                           binEdges={component: edges})

    node = nubarNode(reactions, 452)
    before = node.form
    diagnostics = pset.applyToSuite(reactions, multiplicityResolver=onlyTheTotal)

    assert set(diagnostics) == {component}
    assert diagnostics[component]["max_factor"] == pytest.approx(1.5)
    assert node.form is before, "the evaluated form must not have been displaced"
    assert node["realization-0000"] is not before
    assert node["realization-0000"].label == "realization-0000"
