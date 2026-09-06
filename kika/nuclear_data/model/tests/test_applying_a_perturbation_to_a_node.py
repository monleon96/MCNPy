"""The gate that decides whether perturbing on the model is worth doing.

``kika.nuclear_data.model.perturbation`` has to reproduce
``kika.sampling.mf33_sampling.apply_factors_to_pendf_mf3``, which is what the
thesis pipeline runs, on the input that pipeline is given. Not approximately:
the same abscissae, the same values, and the same bytes out of the encoder. A
difference here is the class of discrepancy that cannot be attributed later --
a perturbed ensemble that moved for a reason nobody can name.

The comparison runs on ``micro_fe56_structural.endf``, cut from the tape the
Fe-56 track uses and committed, so this gate runs on either machine. What it
does **not** cover is stated where it matters: that tape is lin-lin in one
region, like every PENDF, so the multi-region and non-lin-lin behaviour is
tested against the definition rather than against the format applier -- which
has no defensible answer there anyway.

**The gate found a defect in the applier it was written to agree with**, which
is what a gate is for. ``_augment_with_step_duplicates`` sorted its insertions
by position alone, so two bin edges landing in one original interval came out
in the wrong order and the table was no longer ascending -- an invalid TAB1,
not merely a different one. Measured on this very tape with the Fe-56 MF33 grid
of ``micro_fe56_mf33.endf``: 2 disordered insertions in MT1, 2 in MT102, 28 in
MT2. Fixed there rather than reproduced here; ``library-gaps.md`` D31 carries
the measurement still owed on a real PENDF, which is what the live pipeline
actually feeds it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter import decodeMF3MT, encodeMF3MT
from kika.nuclear_data.model import EVAL_LABEL
from kika.nuclear_data.model.enums import Interpolation
from kika.nuclear_data.model.functions.regions1d import Regions1d
from kika.nuclear_data.model.functions.xys1d import XYs1d
from kika.nuclear_data.model.perturbation import (
    applyFactors,
    refineAtBinEdges,
)
from kika.sampling.mf33_sampling import (
    _augment_with_step_duplicates,
    perturb_pointwise_xs,
)

TAPE = (Path(__file__).resolve().parents[3]
        / "endf" / "tests" / "data" / "micro_fe56_structural.endf")


@pytest.fixture(scope="module")
def mf3():
    return read_endf(str(TAPE)).files[3].sections


def _bins(section, n=12):
    """A covariance grid that lands inside the section, edges off its points.

    Logarithmic, because an MF33 grid is, and offset by a non-round factor so
    the edges do not coincide with tabulated energies -- which is the case that
    forces an insertion rather than exercising the "already a repeat" branch.
    """
    lo, hi = float(section.energies[0]), float(section.energies[-1])
    return np.geomspace(max(lo, 1e-3) * 1.7, hi * 0.6, n)


def _factors(rng, n):
    return 1.0 + 0.25 * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mt", [1, 2, 102])
def test_the_model_applier_reproduces_the_format_applier(mf3, mt):
    """Same abscissae and same values, exactly, on the real tape."""
    section = mf3[mt]
    edges = _bins(section)
    factors = _factors(np.random.default_rng(20260906), edges.size - 1)

    formatXs, formatYs, _ = _augment_with_step_duplicates(
        np.asarray(section.energies, dtype=float),
        np.asarray(section.cross_sections, dtype=float),
        list(section.energy_interpolation or []),
        edges,
    )
    formatValues, _, formatOut = perturb_pointwise_xs(
        formatXs, formatYs, factors, edges)

    reaction, _ = decodeMF3MT(section)
    perturbed, diagnostics = applyFactors(
        reaction.crossSection[EVAL_LABEL], factors, edges)
    modelXs, modelYs, _ = perturbed.toEndfRegions()

    np.testing.assert_array_equal(modelXs, formatXs)
    np.testing.assert_array_equal(modelYs, formatValues)
    assert diagnostics["n_inserted"] == formatXs.size - len(section.energies)
    assert diagnostics["frac_out_of_coverage"] == pytest.approx(formatOut)


@pytest.mark.parametrize("mt", [1, 2, 102])
def test_the_encoded_section_is_byte_identical(mf3, mt):
    """And through the encoder, which is where the (NBT, INT) pairs are read."""
    section = mf3[mt]
    edges = _bins(section)
    factors = _factors(np.random.default_rng(20260906), edges.size - 1)

    import dataclasses

    formatXs, formatYs, formatInterp = _augment_with_step_duplicates(
        np.asarray(section.energies, dtype=float),
        np.asarray(section.cross_sections, dtype=float),
        list(section.energy_interpolation or []),
        edges,
    )
    formatValues, _, _ = perturb_pointwise_xs(formatXs, formatYs, factors, edges)
    fromFormat = dataclasses.replace(
        section,
        _energies=list(formatXs), _cross_sections=list(formatValues),
        _interpolation=formatInterp, _np=len(formatXs), _nr=len(formatInterp),
    )

    reaction, _ = decodeMF3MT(section)
    perturbed, _ = applyFactors(reaction.crossSection[EVAL_LABEL], factors, edges)
    reaction.crossSection[EVAL_LABEL] = perturbed
    fromModel, report = encodeMF3MT(reaction, section._mat)

    assert str(fromModel) == str(fromFormat)
    assert fromModel._interpolation == formatInterp


def test_the_pairs_come_from_the_regions_and_not_from_arithmetic(mf3):
    """NBT is a consequence of the region lengths, so it cannot drift from NP.

    The defect D28 fixed was exactly this pair disagreeing. The format applier
    keeps them in step by recomputing NBT by hand; here there is nothing to
    recompute, and this asserts the property that makes that true.
    """
    section = mf3[2]
    edges = _bins(section)
    perturbed, _ = applyFactors(
        decodeMF3MT(section)[0].crossSection[EVAL_LABEL],
        np.full(edges.size - 1, 1.5), edges)
    xs, _, pairs = perturbed.toEndfRegions()

    assert pairs[-1][0] == xs.size
    assert [nbt for nbt, _ in pairs] == sorted(nbt for nbt, _ in pairs)


# ---------------------------------------------------------------------------
# The rules, on shapes the real tape does not have
# ---------------------------------------------------------------------------

def _regions(*specs):
    """A Regions1d from ``(xs, ys, interpolation)`` triples."""
    return Regions1d(function1ds=[
        XYs1d(xs=np.asarray(xs, dtype=float), ys=np.asarray(ys, dtype=float),
              interpolation=rule, index=i)
        for i, (xs, ys, rule) in enumerate(specs)
    ])


def test_an_edge_the_table_already_repeats_is_left_alone():
    """Two points there already means the evaluation writes its own step."""
    fn = XYs1d(xs=np.array([1.0, 5.0, 5.0, 9.0]),
               ys=np.array([1.0, 2.0, 8.0, 9.0]),
               interpolation=Interpolation.linlin)
    refined = refineAtBinEdges(fn, [0.0, 5.0, 10.0])
    np.testing.assert_array_equal(refined.xs, fn.xs)


def test_an_edge_the_table_has_once_gains_its_twin():
    fn = XYs1d(xs=np.array([1.0, 5.0, 9.0]), ys=np.array([1.0, 2.0, 3.0]),
               interpolation=Interpolation.linlin)
    refined = refineAtBinEdges(fn, [0.0, 5.0, 10.0])
    np.testing.assert_array_equal(refined.xs, [1.0, 5.0, 5.0, 9.0])
    np.testing.assert_array_equal(refined.ys, [1.0, 2.0, 2.0, 3.0])


def test_an_edge_the_table_does_not_have_is_interpolated_under_its_own_law():
    """The one place this deliberately differs from the format applier.

    ``y = x**2`` written as a log-log pair from (1, 1) to (16, 256). At x = 4 the
    law the evaluator chose says 16; the straight line between the two points
    says 52. The format applier asks ``np.interp`` and gets 52, because its
    input is always a lin-lin PENDF and there it is right. Here the region is
    asked, so the inserted point sits on the curve the file describes rather
    than 3.25x above it.
    """
    fn = XYs1d(xs=np.array([1.0, 16.0]), ys=np.array([1.0, 256.0]),
               interpolation=Interpolation.loglog)
    refined = refineAtBinEdges(fn, [0.5, 4.0, 32.0])

    np.testing.assert_array_equal(refined.xs, [1.0, 4.0, 4.0, 16.0])
    np.testing.assert_allclose(refined.ys, [1.0, 16.0, 16.0, 256.0])
    assert np.interp(4.0, fn.xs, fn.ys) == pytest.approx(52.0)


def test_a_region_boundary_edge_keeps_the_pair_adjacent_in_the_flat_table():
    """An edge on a shared abscissa is owned by the region that ends there.

    ``toEndfRegions`` drops the next region's copy of a shared boundary, so the
    inserted twin has to go on the near side of it or the pair is not adjacent
    once the table is flattened -- and a non-adjacent pair is not a step.
    """
    fn = _regions(
        ([1.0, 4.0], [1.0, 4.0], Interpolation.linlin),
        ([4.0, 9.0], [4.0, 9.0], Interpolation.loglog),
    )
    refined = refineAtBinEdges(fn, [0.0, 4.0, 10.0])
    xs, ys, pairs = refined.toEndfRegions()

    np.testing.assert_array_equal(xs, [1.0, 4.0, 4.0, 9.0])
    np.testing.assert_array_equal(ys, [1.0, 4.0, 4.0, 9.0])
    assert pairs[-1][0] == xs.size


def test_the_step_takes_the_lower_bin_on_its_first_half():
    """The rule the whole refinement exists to make expressible.

    The last point is deliberately left alone. Under the ``side='right'`` rule a
    point sitting exactly on the top edge falls in the bin that would *start*
    there, and there is no such bin, so it is outside coverage at factor 1. That
    is what the format applier does too -- it is the reason
    ``perturb_pointwise_xs`` grew an opt-in ``clamp_top_edge``, for the MF1
    nu-bar table whose last point really is the top of its grid.
    """
    fn = XYs1d(xs=np.array([1.0, 5.0, 9.0]), ys=np.array([1.0, 1.0, 1.0]),
               interpolation=Interpolation.linlin)
    perturbed, _ = applyFactors(fn, [2.0, 3.0], [1.0, 5.0, 9.0])

    np.testing.assert_array_equal(perturbed.xs, [1.0, 5.0, 5.0, 9.0])
    # First of the pair: the bin that *ends* at 5. Second: the one that starts.
    np.testing.assert_allclose(perturbed.ys, [2.0, 2.0, 3.0, 1.0])


def test_points_outside_the_grid_keep_their_value():
    fn = XYs1d(xs=np.array([1.0, 5.0, 100.0]), ys=np.array([1.0, 1.0, 1.0]),
               interpolation=Interpolation.linlin)
    perturbed, diagnostics = applyFactors(fn, [7.0], [2.0, 9.0])

    assert perturbed.ys[0] == 1.0            # below the grid
    assert perturbed.ys[-1] == 1.0           # above it
    assert diagnostics["frac_out_of_coverage"] > 0.0


def test_a_factor_block_that_does_not_match_its_grid_is_refused():
    fn = XYs1d(xs=np.array([1.0, 9.0]), ys=np.array([1.0, 1.0]),
               interpolation=Interpolation.linlin)
    with pytest.raises(ValueError, match="same binning"):
        applyFactors(fn, [1.0, 2.0, 3.0], [1.0, 5.0, 9.0])


def test_the_original_is_not_touched():
    """A drawn ensemble applies thousands of these to the same evaluated form."""
    fn = XYs1d(xs=np.array([1.0, 5.0, 9.0]), ys=np.array([1.0, 1.0, 1.0]),
               interpolation=Interpolation.linlin)
    applyFactors(fn, [2.0, 3.0], [1.0, 5.0, 9.0])

    np.testing.assert_array_equal(fn.xs, [1.0, 5.0, 9.0])
    np.testing.assert_array_equal(fn.ys, [1.0, 1.0, 1.0])
