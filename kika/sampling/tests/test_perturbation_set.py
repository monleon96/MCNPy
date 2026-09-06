"""A drawn perturbation says what it is, and says the same thing after a round trip.

The envelope exists because the index and the semantics used to be carried by
convention: the shape of the code that drew the factors and the shape of the
code that applied them. These tests pin what a convention cannot promise -- that
the slicing reads ``widths`` rather than assuming ``stride``, that a set written
to a run directory comes back meaning the same, and that a realisation covering
several quantities lands each of them on the right node.

**The second half is what changed when the envelope stopped being MF33's.** A
realisation is now one object across every block of a request, so the tests that
matter most are the ones about *dispatch*: an MF34 order goes to the angular
distribution, an MF34 order 0 goes to the cross section because that is where the
magnitude lives, both at once for the same reaction is refused rather than
applied twice, and a multiplicity says why it cannot be applied yet instead of
being skipped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.nuclear_data.model.enums import Interpolation
from kika.nuclear_data.model.functions.xys1d import XYs1d
from kika.sampling.joint_blocks import ComponentKey
from kika.sampling.perturbation_set import SEMANTICS, PerturbationSet

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"

XS2 = ComponentKey(26056, 33, 2)
XS102 = ComponentKey(26056, 33, 102)


def _index(components, stride, widths, grids, label="MF33"):
    """A ``requestIndex``-shaped entry, by hand, for one block."""
    return {(2631, label, tuple(components)): {
        "components": list(components),
        "stride": stride,
        "widths": dict(zip(components, widths)),
        "grids": dict(zip(components, grids)),
        "dimension": len(components) * stride,
        "union": "global",
        "quantities": sorted({c.quantity for c in components}),
    }}


def test_a_uniform_stride_cuts_the_vector_by_component():
    grid = [1.0, 5.0, 9.0]
    index = _index([XS2, XS102], stride=2, widths=[2, 2], grids=[grid, grid])

    pset = PerturbationSet.fromDraw([1.1, 1.2, 0.9, 0.8], index,
                                    label="realization-0001")

    assert pset.reactions() == (2, 102)
    assert pset.components() == (XS2, XS102)
    np.testing.assert_allclose(pset.factors[XS2], [1.1, 1.2])
    np.testing.assert_allclose(pset.factors[XS102], [0.9, 0.8])
    assert pset.quantities() == ("crossSection",)


def test_widths_are_read_and_not_assumed_from_the_stride():
    """A per-component union pads each block, and the padding is not factors.

    This is the whole reason ``widths`` is in the index. Slicing by ``stride``
    would give MT102 two real factors and two zeros, and a zero factor is not a
    small perturbation -- it is a cross section deleted.
    """
    index = _index([XS2, XS102], stride=3, widths=[3, 2],
                   grids=[[1.0, 4.0, 7.0, 9.0], [1.0, 5.0, 9.0]])
    drawn = [1.1, 1.2, 1.3, 0.9, 0.8, 0.0]

    pset = PerturbationSet.fromDraw(drawn, index, label="realization-0001")

    np.testing.assert_allclose(pset.factors[XS2], [1.1, 1.2, 1.3])
    np.testing.assert_allclose(pset.factors[XS102], [0.9, 0.8])
    assert 0.0 not in pset.factors[XS102]


def test_a_vector_that_does_not_fit_the_index_is_refused():
    index = _index([XS2], stride=2, widths=[2], grids=[[1.0, 5.0, 9.0]])
    with pytest.raises(ValueError, match="factor"):
        PerturbationSet.fromDraw([1.0, 2.0, 3.0], index, label="r")


def test_a_bare_vector_needs_an_index_of_one_block():
    """With two blocks the rows have to say which block they came from."""
    index = {**_index([XS2], 1, [1], [[1.0, 9.0]]),
             **_index([ComponentKey(26056, 34, 2, 1)], 1, [1], [[1.0, 9.0]],
                      label="MF34")}
    with pytest.raises(ValueError, match="one block"):
        PerturbationSet.fromDraw([1.0], index, label="r")


def test_a_block_that_was_not_drawn_is_refused_rather_than_dropped():
    """A realisation missing a block is perturbed in fewer places than it claims."""
    index = {**_index([XS2], 1, [1], [[1.0, 9.0]]),
             **_index([ComponentKey(26056, 34, 2, 1)], 1, [1], [[1.0, 9.0]],
                      label="MF34")}
    drawn = {next(iter(index)): np.array([1.1])}
    with pytest.raises(ValueError, match="not drawn"):
        PerturbationSet.fromDraw(drawn, index, label="r")


def test_a_draw_over_two_blocks_becomes_one_realisation():
    """Two independent draws, one realisation -- and it records which were which."""
    angular = ComponentKey(26056, 34, 2, 1)
    index = {**_index([XS2], 1, [1], [[1.0, 9.0]]),
             **_index([angular], 2, [2], [[1.0, 5.0, 9.0]], label="MF34")}
    drawn = {key: values for key, values in
             zip(index, ([1.25], [0.9, 1.1]))}

    pset = PerturbationSet.fromDraw(drawn, index, label="realization-0003")

    assert pset.components() == (XS2, angular)
    assert pset.quantities() == ("angularDistribution", "crossSection")
    assert len(pset.groups) == 2, "two blocks were drawn, so two groups"
    assert set(pset.groups) == {(XS2,), (angular,)}
    assert "2 draw(s)" in pset.describe()


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------

def test_a_semantics_it_does_not_know_is_refused():
    """"Relative" and "absolute" differ by multiply versus add, and a file that
    does not say which cannot be read back safely."""
    with pytest.raises(ValueError, match="not one of"):
        PerturbationSet(label="r", factors={XS2: np.array([1.0])},
                        binEdges={XS2: np.array([1.0, 2.0])},
                        semantics="whatever-we-meant-in-august")


def test_factors_and_their_grid_are_one_object():
    with pytest.raises(ValueError, match="one object"):
        PerturbationSet(label="r", factors={XS2: np.array([1.0])},
                        binEdges={XS102: np.array([1.0, 2.0])})


def test_a_block_has_one_factor_per_bin():
    with pytest.raises(ValueError, match="factor"):
        PerturbationSet(label="r", factors={XS2: np.array([1.0, 1.0])},
                        binEdges={XS2: np.array([1.0, 2.0])})


def test_the_groups_have_to_cover_what_the_factors_do():
    with pytest.raises(ValueError, match="two different things"):
        PerturbationSet(label="r",
                        factors={XS2: np.array([1.0]), XS102: np.array([1.0])},
                        binEdges={XS2: np.array([1.0, 2.0]),
                                  XS102: np.array([1.0, 2.0])},
                        groups=((XS2,),))


# ---------------------------------------------------------------------------
# Applying, one node at a time
# ---------------------------------------------------------------------------

def _set(label="realization-0007"):
    return PerturbationSet(label=label, factors={XS2: np.array([2.0, 3.0])},
                           binEdges={XS2: np.array([1.0, 5.0, 9.0])})


def test_applying_gives_the_same_answer_as_the_applier_itself():
    from kika.nuclear_data.model.perturbation import applyFactors

    fn = XYs1d(xs=np.array([1.0, 5.0, 9.0]), ys=np.array([1.0, 1.0, 1.0]),
               interpolation=Interpolation.linlin)
    direct, _ = applyFactors(fn, [2.0, 3.0], [1.0, 5.0, 9.0])
    through, _ = _set().apply(fn, XS2)

    np.testing.assert_array_equal(through.xs, direct.xs)
    np.testing.assert_array_equal(through.ys, direct.ys)


def test_a_component_it_does_not_carry_raises_rather_than_passing_it_through():
    """"No perturbation for this one" and "a perturbation of one" are different
    answers, and conflating them narrows an ensemble silently."""
    fn = XYs1d(xs=np.array([1.0, 9.0]), ys=np.array([1.0, 1.0]),
               interpolation=Interpolation.linlin)
    with pytest.raises(KeyError, match="MT102"):
        _set().apply(fn, XS102)


def test_a_block_can_be_addressed_by_reaction_without_knowing_the_za():
    factors, edges = _set().block(2)
    np.testing.assert_allclose(factors, [2.0, 3.0])
    np.testing.assert_allclose(edges, [1.0, 5.0, 9.0])
    with pytest.raises(KeyError, match="MF33/MT102"):
        _set().block(102)


# ---------------------------------------------------------------------------
# On disk
# ---------------------------------------------------------------------------

def test_it_comes_back_from_disk_meaning_the_same(tmp_path):
    angular = ComponentKey(26056, 34, 2, 3)
    original = PerturbationSet(
        label="realization-0042",
        factors={XS2: np.array([1.1, 0.9]), angular: np.array([1.05])},
        binEdges={XS2: np.array([1.0, 5.0, 9.0]), angular: np.array([1.0, 9.0])},
        groups=((XS2,), (angular,)),
        provenance={"seed": 20260906, "sample": 42, "union": "global"},
    )
    restored = PerturbationSet.read(original.write(tmp_path / "perturbation.json"))

    assert restored.label == original.label
    assert restored.semantics == original.semantics
    assert restored.edgeRule == original.edgeRule
    assert restored.provenance == original.provenance
    assert restored.components() == original.components()
    assert restored.groups == original.groups
    for component in original.components():
        np.testing.assert_array_equal(restored.factors[component],
                                      original.factors[component])
        np.testing.assert_array_equal(restored.binEdges[component],
                                      original.binEdges[component])


def test_the_file_says_which_quantity_each_block_perturbs(tmp_path):
    """A reader a year later has the MF in the key and the name beside it."""
    angular = ComponentKey(26056, 34, 2, 3)
    data = PerturbationSet(label="r", factors={angular: np.array([1.05])},
                           binEdges={angular: np.array([1.0, 9.0])}).to_dict()
    (block,) = data["blocks"]
    assert block["component"] == [26056, 34, 2, 3]
    assert block["quantity"] == "angularDistribution"


def test_a_file_from_another_format_version_is_refused(tmp_path):
    """Rather than read fields that may have meant something else."""
    data = _set().to_dict()
    data["format"] = 99
    with pytest.raises(ValueError, match="format 99"):
        PerturbationSet.from_dict(data)


def test_the_written_file_names_its_semantics_and_its_edge_rule():
    """The two things the factors themselves cannot say."""
    data = _set().to_dict()
    assert data["semantics"] in SEMANTICS
    assert data["edgeRule"] == "endf-step-duplicate"


# ---------------------------------------------------------------------------
# Against the real index, not a hand-written one
# ---------------------------------------------------------------------------

def test_a_real_draw_cuts_into_a_set_that_covers_its_reactions():
    """Source, draw and envelope end to end, on the committed Fe-56 micro-tapes.

    The synthetic indices above pin the slicing rule; this pins that the rule is
    reading the shape the source actually returns. An envelope that agreed with
    a hand-written index and not with ``loadCrossSectionBlocks`` would pass
    every test above and be useless.

    It also pins that the *older* index shape still cuts correctly:
    ``loadCrossSectionBlocks`` returns ``pairs`` and no MF in the component, so
    the MF has to come off the block key. That path stays until the MF33 call
    sites move.
    """
    from kika.endf import read_endf
    from kika.sampling.mf33_sampling import loadCrossSectionBlocks
    from kika.sampling.multigroup_draw import draw_relative_factors

    cov = read_endf(str(DATA / "micro_fe56_mf33.endf"))
    central = dict(read_endf(str(DATA / "micro_fe56_structural.endf")).files[3].sections)

    blocks, index, _grid, mts = loadCrossSectionBlocks(cov, None, central, mf=33)
    (key, matrix), = blocks
    (_indexKey, meta), = index.items()
    factors, _info = draw_relative_factors(
        matrix, 4, key=key, pairs=meta["pairs"], stride=meta["stride"],
        bins=meta["grids"][meta["pairs"][0]], seed=20260906, verbose=False)

    pset = PerturbationSet.fromDraw(
        factors[0], index, label="realization-0000",
        provenance={"seed": 20260906, "sample": 0})

    assert pset.quantities() == ("crossSection",)
    assert pset.reactions() == tuple(sorted(mts))
    for component in pset.components():
        assert component.mf == 33
        assert pset.factors[component].size == meta["widths"][(26056, component.mt)]
        assert pset.binEdges[component].size == pset.factors[component].size + 1
        assert np.all(pset.factors[component] > 0.0)


def test_a_set_puts_its_realisation_beside_the_evaluated_form():
    """§9.1's multi-form container is what a realisation goes into.

    The evaluated form has to stay: it is what the covariance is stated about,
    and it is what ``encodeMF3MT`` falls back to for a reaction this draw did
    not cover.
    """
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.nuclear_data.model import EVAL_LABEL

    endf = read_endf(str(DATA / "micro_fe56_structural.endf"))
    suite, _report = decodeReactionSuite(endf)

    edges = np.geomspace(1.7e-3, 9.0e7, 13)
    pset = PerturbationSet(label="realization-0007",
                           factors={XS2: np.full(edges.size - 1, 1.25)},
                           binEdges={XS2: edges})
    diagnostics = pset.applyToSuite(suite)

    assert set(diagnostics) == {XS2}
    reaction = suite.reactionByENDF_MT(2)
    assert EVAL_LABEL in reaction.crossSection
    assert "realization-0007" in reaction.crossSection

    evaluated = reaction.crossSection[EVAL_LABEL].toEndfRegions()
    realised = reaction.crossSection["realization-0007"].toEndfRegions()
    assert realised[0].size > evaluated[0].size          # the step duplicates
    assert diagnostics[XS2]["n_inserted"] == realised[0].size - evaluated[0].size
    assert diagnostics[XS2]["max_factor"] == pytest.approx(1.25)

    # And the untouched reactions are untouched, not quietly perturbed by one.
    for mt in (1, 102):
        assert "realization-0007" not in suite.reactionByENDF_MT(mt).crossSection


# ---------------------------------------------------------------------------
# Dispatch: which node each quantity lands on
# ---------------------------------------------------------------------------

def _structuralSuite():
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite

    suite, _report = decodeReactionSuite(read_endf(str(DATA
                                                      / "micro_fe56_structural.endf")))
    return suite


def test_an_angular_realisation_goes_on_the_products_distribution():
    """MF34 perturbs a distribution, and §17.2.1 hangs one on a product."""
    from kika.nuclear_data.model import EVAL_LABEL

    suite = _structuralSuite()
    edges = np.array([1.0e-5, 1.0e6, 2.0e7])
    orders = {order: ComponentKey(26056, 34, 2, order) for order in (1, 2)}
    pset = PerturbationSet(
        label="realization-0011",
        factors={component: np.array([1.2, 0.8]) for component in orders.values()},
        binEdges={component: edges for component in orders.values()},
    )
    diagnostics = pset.applyToSuite(suite)

    assert set(diagnostics) == set(orders.values())
    reaction = suite.reactionByENDF_MT(2)
    (product,) = [p for p in reaction.outputChannel.products
                  if p.distribution is not None]
    assert "realization-0011" in product.distribution
    assert EVAL_LABEL in product.distribution

    before = product.distribution[EVAL_LABEL].angular
    after = product.distribution["realization-0011"].angular
    assert after is not before
    assert "realization-0007" not in reaction.crossSection
    assert diagnostics[orders[1]]["n_scaled"] > 0

    # The evaluated distribution is what the covariance is about; it stays put.
    from kika.nuclear_data.model.perturbation import _legendreRegions

    (_c, _p, region), *_rest = _legendreRegions(before)
    assert float(region.function1ds[0].coefficients[1]) == pytest.approx(
        5.877681e-05, rel=1e-9)


def test_a_realisation_carries_its_own_label_and_not_the_evaluations():
    """The container keys by label and the form carries one; they have to agree.

    Found by the GNDS writer and not by a review: ``gnds/encode.py``'s
    ``crossSection`` writes each form with the label the *form* states, so a
    realisation whose node still said ``eval`` came out as a second
    ``label="eval"`` -- two forms with one label in one container, which no
    reader can tell apart. The ENDF side could not see it, because
    ``encodeMF3MT(..., label=)`` looks the form up by key.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    suite = _structuralSuite()
    edges = np.geomspace(1.7e-3, 9.0e7, 5)
    orders = {order: ComponentKey(26056, 34, 2, order) for order in (1,)}
    pset = PerturbationSet(
        label="realization-0021",
        factors={XS2: np.full(4, 1.25),
                 **{c: np.full(4, 0.9) for c in orders.values()}},
        binEdges={XS2: edges, **{c: edges for c in orders.values()}},
    )
    pset.applyToSuite(suite)

    reaction = suite.reactionByENDF_MT(2)
    assert reaction.crossSection["realization-0021"].label == "realization-0021"
    assert reaction.crossSection[EVAL_LABEL].label == EVAL_LABEL

    (product,) = [p for p in reaction.outputChannel.products
                  if p.distribution is not None]
    assert product.distribution["realization-0021"].label == "realization-0021"
    # The evaluated *distribution* carries no label of its own -- the ENDF
    # decoder does not set one and the GNDS writer supplies the container's key
    # for it. So the two halves of the model disagree about whether a form names
    # itself, and this records which is which rather than asserting the tidier
    # of the two. What matters for a realisation is that it does not inherit the
    # evaluation's name, and it does not.
    assert product.distribution[EVAL_LABEL].label is None


def test_the_magnitude_order_lands_on_the_cross_section():
    """L=0 is sigma(E)'s uncertainty on MF34's grid, not a shape coefficient.

    This is the routing that makes a joint XS+DA draw mean one thing: the
    covariance correlates the magnitude with the shape, and the magnitude's half
    of the realisation has to reach MF3 or the correlation is drawn and thrown
    away.
    """
    suite = _structuralSuite()
    edges = np.geomspace(1.7e-3, 9.0e7, 5)
    magnitude = ComponentKey(26056, 34, 2, 0)
    shape = ComponentKey(26056, 34, 2, 1)
    pset = PerturbationSet(
        label="realization-0012",
        factors={magnitude: np.full(4, 1.5), shape: np.full(4, 0.5)},
        binEdges={magnitude: edges, shape: edges},
    )
    diagnostics = pset.applyToSuite(suite)

    reaction = suite.reactionByENDF_MT(2)
    assert "realization-0012" in reaction.crossSection, (
        "the magnitude did not reach the cross section")
    assert diagnostics[magnitude]["max_factor"] == pytest.approx(1.5)

    (product,) = [p for p in reaction.outputChannel.products
                  if p.distribution is not None]
    assert "realization-0012" in product.distribution
    assert diagnostics[shape]["n_scaled"] > 0


def test_two_claims_on_one_cross_section_are_refused():
    """MF33 and MF34's L=0 both perturb sigma; applying both multiplies twice."""
    suite = _structuralSuite()
    edges = np.geomspace(1.7e-3, 9.0e7, 5)
    pset = PerturbationSet(
        label="realization-0013",
        factors={XS2: np.full(4, 1.1),
                 ComponentKey(26056, 34, 2, 0): np.full(4, 1.2)},
        binEdges={XS2: edges, ComponentKey(26056, 34, 2, 0): edges},
    )
    with pytest.raises(ValueError, match="multiplies"):
        pset.applyToSuite(suite)


def test_a_multiplicity_without_a_resolver_is_refused_not_skipped():
    """Skipping it silently is the failure this class exists to prevent.

    It refuses for a narrower reason than it used to: not "a nu-bar has nowhere
    to go" -- ``Multiplicity`` is a ``Component`` now and it has -- but "which
    node an ENDF MT names is not something the sampling layer may look up".
    """
    suite = _structuralSuite()
    nubar = ComponentKey(92235, 31, 452)
    pset = PerturbationSet(label="realization-0014",
                           factors={nubar: np.array([1.1])},
                           binEdges={nubar: np.array([1.0, 9.0])})
    with pytest.raises(ValueError, match="needs a resolver"):
        pset.applyToSuite(suite)
