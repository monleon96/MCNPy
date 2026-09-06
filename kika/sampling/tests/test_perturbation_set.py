"""A drawn perturbation says what it is, and says the same thing after a round trip.

The envelope exists because the index and the semantics used to be carried by
convention: the shape of the code that drew the factors and the shape of the
code that applied them. These tests pin the two things a convention cannot
promise -- that the slicing reads ``widths`` rather than assuming ``stride``,
and that a set written to a run directory comes back meaning the same.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.nuclear_data.model.enums import Interpolation
from kika.nuclear_data.model.functions.xys1d import XYs1d
from kika.sampling.perturbation_set import SEMANTICS, PerturbationSet


def _index(pairs, stride, widths, grids):
    return {(2631, "MF33", tuple(pairs)): {
        "pairs": list(pairs),
        "stride": stride,
        "widths": dict(zip(pairs, widths)),
        "grids": dict(zip(pairs, grids)),
        "dimension": len(pairs) * stride,
    }}


def test_a_uniform_stride_cuts_the_vector_by_component():
    pairs = [(26056, 2), (26056, 102)]
    grid = [1.0, 5.0, 9.0]
    index = _index(pairs, stride=2, widths=[2, 2], grids=[grid, grid])

    pset = PerturbationSet.fromDraw([1.1, 1.2, 0.9, 0.8], index, label="realization-0001")

    assert pset.reactions() == (2, 102)
    np.testing.assert_allclose(pset.factors[2], [1.1, 1.2])
    np.testing.assert_allclose(pset.factors[102], [0.9, 0.8])
    assert pset.key == "MF33"


def test_widths_are_read_and_not_assumed_from_the_stride():
    """A per-component union pads each block, and the padding is not factors.

    This is the whole reason ``widths`` is in the index. Slicing by ``stride``
    would give MT102 two real factors and two zeros, and a zero factor is not a
    small perturbation -- it is a cross section deleted.
    """
    pairs = [(26056, 2), (26056, 102)]
    index = _index(
        pairs, stride=3,
        widths=[3, 2],
        grids=[[1.0, 4.0, 7.0, 9.0], [1.0, 5.0, 9.0]],
    )
    drawn = [1.1, 1.2, 1.3, 0.9, 0.8, 0.0]

    pset = PerturbationSet.fromDraw(drawn, index, label="realization-0001")

    np.testing.assert_allclose(pset.factors[2], [1.1, 1.2, 1.3])
    np.testing.assert_allclose(pset.factors[102], [0.9, 0.8])
    assert 0.0 not in pset.factors[102]


def test_a_vector_that_does_not_fit_the_index_is_refused():
    pairs = [(26056, 2)]
    index = _index(pairs, stride=2, widths=[2], grids=[[1.0, 5.0, 9.0]])
    with pytest.raises(ValueError, match="factor"):
        PerturbationSet.fromDraw([1.0, 2.0, 3.0], index, label="r")


def test_two_covariance_keys_are_two_sets():
    index = {"MF33": {"pairs": [], "stride": 0, "widths": {}, "grids": {},
                      "dimension": 0},
             "MF31": {"pairs": [], "stride": 0, "widths": {}, "grids": {},
                      "dimension": 0}}
    with pytest.raises(ValueError, match="one covariance key"):
        PerturbationSet.fromDraw([], index, label="r")


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------

def test_a_semantics_it_does_not_know_is_refused():
    """"Relative" and "absolute" differ by multiply versus add, and a file that
    does not say which cannot be read back safely."""
    with pytest.raises(ValueError, match="not one of"):
        PerturbationSet(label="r", key="MF33",
                        factors={2: np.array([1.0])},
                        binEdges={2: np.array([1.0, 2.0])},
                        semantics="whatever-we-meant-in-august")


def test_factors_and_their_grid_are_one_object():
    with pytest.raises(ValueError, match="one object"):
        PerturbationSet(label="r", key="MF33",
                        factors={2: np.array([1.0])},
                        binEdges={102: np.array([1.0, 2.0])})


def test_a_block_has_one_factor_per_bin():
    with pytest.raises(ValueError, match="factor"):
        PerturbationSet(label="r", key="MF33",
                        factors={2: np.array([1.0, 1.0])},
                        binEdges={2: np.array([1.0, 2.0])})


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def _set(label="realization-0007"):
    return PerturbationSet(
        label=label, key="MF33",
        factors={2: np.array([2.0, 3.0])},
        binEdges={2: np.array([1.0, 5.0, 9.0])},
    )


def test_applying_gives_the_same_answer_as_the_applier_itself():
    from kika.nuclear_data.model.perturbation import applyFactors

    fn = XYs1d(xs=np.array([1.0, 5.0, 9.0]), ys=np.array([1.0, 1.0, 1.0]),
               interpolation=Interpolation.linlin)
    direct, _ = applyFactors(fn, [2.0, 3.0], [1.0, 5.0, 9.0])
    through, _ = _set().apply(fn, 2)

    np.testing.assert_array_equal(through.xs, direct.xs)
    np.testing.assert_array_equal(through.ys, direct.ys)


def test_an_mt_it_does_not_carry_raises_rather_than_passing_it_through():
    """"No perturbation for this reaction" and "a perturbation of one" are
    different answers, and conflating them narrows an ensemble silently."""
    fn = XYs1d(xs=np.array([1.0, 9.0]), ys=np.array([1.0, 1.0]),
               interpolation=Interpolation.linlin)
    with pytest.raises(KeyError, match="MT102"):
        _set().apply(fn, 102)


# ---------------------------------------------------------------------------
# On disk
# ---------------------------------------------------------------------------

def test_it_comes_back_from_disk_meaning_the_same(tmp_path):
    original = PerturbationSet(
        label="realization-0042", key="MF33",
        factors={2: np.array([1.1, 0.9]), 102: np.array([1.05])},
        binEdges={2: np.array([1.0, 5.0, 9.0]), 102: np.array([1.0, 9.0])},
        provenance={"seed": 20260906, "sample": 42, "union": "global"},
    )
    restored = PerturbationSet.read(original.write(tmp_path / "perturbation.json"))

    assert restored.label == original.label
    assert restored.key == original.key
    assert restored.semantics == original.semantics
    assert restored.edgeRule == original.edgeRule
    assert restored.provenance == original.provenance
    assert restored.reactions() == original.reactions()
    for mt in original.reactions():
        np.testing.assert_array_equal(restored.factors[mt], original.factors[mt])
        np.testing.assert_array_equal(restored.binEdges[mt], original.binEdges[mt])


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

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"


def test_a_real_draw_cuts_into_a_set_that_covers_its_reactions():
    """Source, draw and envelope end to end, on the committed Fe-56 micro-tapes.

    The synthetic indices above pin the slicing rule; this pins that the rule is
    reading the shape the source actually returns. An envelope that agreed with
    a hand-written index and not with ``loadCrossSectionBlocks`` would pass
    every test above and be useless.
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

    assert pset.key == "MF33"
    assert pset.reactions() == tuple(sorted(mts))
    for mt in pset.reactions():
        assert pset.factors[mt].size == meta["widths"][(26056, mt)]
        assert pset.binEdges[mt].size == pset.factors[mt].size + 1
        assert np.all(pset.factors[mt] > 0.0)


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
    pset = PerturbationSet(
        label="realization-0007", key="MF33",
        factors={2: np.full(edges.size - 1, 1.25)},
        binEdges={2: edges},
    )
    diagnostics = pset.applyToSuite(suite)

    assert set(diagnostics) == {2}
    reaction = suite.reactionByENDF_MT(2)
    assert EVAL_LABEL in reaction.crossSection
    assert "realization-0007" in reaction.crossSection

    evaluated = reaction.crossSection[EVAL_LABEL].toEndfRegions()
    realised = reaction.crossSection["realization-0007"].toEndfRegions()
    assert realised[0].size > evaluated[0].size          # the step duplicates
    assert diagnostics[2]["n_inserted"] == realised[0].size - evaluated[0].size
    assert diagnostics[2]["max_factor"] == pytest.approx(1.25)

    # And the untouched reactions are untouched, not quietly perturbed by one.
    for mt in (1, 102):
        assert "realization-0007" not in suite.reactionByENDF_MT(mt).crossSection
