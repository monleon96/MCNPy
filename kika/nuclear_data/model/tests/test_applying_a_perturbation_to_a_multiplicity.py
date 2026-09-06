"""A nu-bar is scaled the same way a cross section is, and written differently.

``applyNubarFactors`` has to reproduce
``kika.sampling.mf31_sampling.apply_factors_to_mf1_nubar``, which is what
``perturb_nubar_files`` runs. The comparison is on the table -- energies and
values -- and it is exact, because both start from the same numbers: the ENDF
adapter's decode of MT452/455/456 is bit-identical to the section's own arrays,
which is checked here too so that "the two appliers agree" cannot be an artefact
of them being fed different baselines.

**Why there are two refinements at all**, since that is the part that looks like
duplication. A cross section states a step with a repeated abscissa, and
``applyFactors`` writes one. A nu-bar may not: the family has to satisfy
nu_452 = nu_455 + nu_456, and that sum is exact only if no member carries a
duplicate energy -- a lin-lin interpolant summed on the union of two duplicate-
free grids is a lin-lin identity, and one with duplicates is not. So the step is
approximated by a node at the edge and a shoulder just below it. The two rules
are not a choice; they belong to different quantities.

The fixture is ``micro_u235_nubar.endf``: real JEFF-4.0 U-235 MF1 with all three
family members (96, 5 and 95 points -- the delayed table is far coarser than the
prompt one, which is the case the shoulder exists for) and its MF31.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.model_adapter.multiplicity import nubarNode
from kika.nuclear_data.model.functions.regions1d import Regions1d
from kika.nuclear_data.model.perturbation import (NUBAR_STEP_SHOULDER,
                                                  applyNubarFactors,
                                                  refineForNubar)
from kika.sampling.mf31_sampling import (NUBAR_STEP_SHOULDER as SHIPPED_SHOULDER,
                                         apply_factors_to_mf1_nubar)

TAPE = str(Path(__file__).resolve().parents[3] / "endf" / "tests" / "data"
           / "micro_u235_nubar.endf")
FAMILY = (452, 455, 456)


@pytest.fixture(scope="module")
def tape():
    endf = read_endf(TAPE)
    suite, _report = decodeReactionSuite(endf)
    return endf, suite


def _block(size: int) -> np.ndarray:
    return 1.0 + 0.05 * np.cos(np.arange(size, dtype=float))


def _table(form):
    xs, ys, pairs = form.toEndfRegions()
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), pairs


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------

def test_the_model_carries_the_same_table_the_section_does(tape):
    """The premise of every comparison below, asserted rather than assumed."""
    endf, suite = tape
    for mt in FAMILY:
        section = endf.get_file(1).sections[mt]
        assert section.lnu == 2, (
            f"MT{mt} is LNU={section.lnu}; this file compares tabulated nu-bars "
            f"and the polynomial case is a different one")
        xs, ys, _pairs = _table(nubarNode(suite, mt).form)
        assert np.array_equal(xs, np.asarray(section.energies, dtype=float))
        assert np.array_equal(ys, np.asarray(section.nubar_values, dtype=float))


@pytest.mark.parametrize("mt", FAMILY)
def test_the_two_appliers_produce_the_same_table(tape, mt):
    """Point for point. Any difference here is a difference of rule."""
    endf, suite = tape
    bins = np.geomspace(1.0e-5, 2.0e7, 12)
    block = _block(bins.size - 1)

    shipped, shippedDiagnostics = apply_factors_to_mf1_nubar(
        endf.get_file(1).sections[mt], block, bins)
    mine, diagnostics = applyNubarFactors(nubarNode(suite, mt).form, block, bins)

    xs, ys, _pairs = _table(mine)
    assert np.array_equal(xs, np.asarray(shipped.energies, dtype=float))
    assert np.array_equal(ys, np.asarray(shipped.nubar_values, dtype=float))
    for key in ("n_inserted", "min_factor", "max_factor"):
        assert diagnostics[key] == shippedDiagnostics[key]


def test_the_shoulder_is_the_same_number_on_both_sides():
    """Two constants for one quantity, and the gate above depends on them agreeing.

    Written down rather than imported across the layer boundary: the calculation
    layer may not import the sampling one, so the value is duplicated on purpose
    and this is what stops the duplication drifting.
    """
    assert NUBAR_STEP_SHOULDER == SHIPPED_SHOULDER


# ----------------------------------------------------------------------
# What the refinement does, and what it must not do
# ----------------------------------------------------------------------

def test_the_refinement_inserts_no_duplicate_energies(tape):
    """The property the whole sum rule rests on."""
    _endf, suite = tape
    bins = np.geomspace(1.0e-5, 2.0e7, 12)
    for mt in FAMILY:
        perturbed, _diagnostics = applyNubarFactors(
            nubarNode(suite, mt).form, _block(bins.size - 1), bins)
        xs, _ys, _pairs = _table(perturbed)
        assert np.all(np.diff(xs) > 0.0), (
            f"MT{mt}: the perturbed table has a repeated or descending energy, "
            f"so a sum over the union of grids is no longer an identity")


def test_a_node_and_a_shoulder_go_in_at_every_interior_edge():
    """Two per edge, and the shoulder just below it -- what confines the ramp."""
    xs = np.array([1.0, 10.0, 100.0])
    ys = np.array([1.0, 2.0, 3.0])
    edges = np.array([1.0, 5.0, 50.0, 100.0])

    newXs, newYs = refineForNubar(xs, ys, edges)

    assert 5.0 in newXs and 50.0 in newXs
    assert pytest.approx(5.0 * (1 - NUBAR_STEP_SHOULDER)) == newXs[
        np.argmin(np.abs(newXs - 5.0 * (1 - NUBAR_STEP_SHOULDER)))]
    assert newXs.size == xs.size + 4
    # The baseline curve is untouched: every inserted value is on the original
    # lin-lin interpolant, so the perturbation starts from the same nu-bar.
    assert np.allclose(newYs, np.interp(newXs, xs, ys))


def test_the_last_point_of_a_table_takes_the_last_bins_factor(tape):
    """Without the clamp it falls outside coverage and comes back unperturbed.

    Which reads, in the written file, as the top of the last bin ramping back to
    baseline -- a perturbation that quietly stops short of where the covariance
    says it goes.
    """
    _endf, suite = tape
    form = nubarNode(suite, 456).form
    xs, ys, _pairs = _table(form)
    edges = np.array([1.0e-5, 1.0e6, float(xs[-1])])

    perturbed, _diagnostics = applyNubarFactors(form, np.array([1.0, 2.0]), edges)
    newXs, newYs, _newPairs = _table(perturbed)

    assert newXs[-1] == xs[-1]
    assert newYs[-1] == pytest.approx(2.0 * ys[-1])


def test_a_perturbed_multiplicity_is_still_writable(tape):
    """A ``Regions1d`` goes in and a ``Regions1d`` comes out, for one reason.

    ``_tab1FromMultiplicity`` writes MF1 from ``regions1d`` (LNU=2) or
    ``polynomial1d`` (LNU=1) and refuses a bare ``XYs1d``, so a realisation
    returned as a plain curve is one that cannot be written back. The encoder
    found that, which is where it should be found; this keeps it found.
    """
    _endf, suite = tape
    form = nubarNode(suite, 452).form
    assert isinstance(form, Regions1d)

    perturbed, _diagnostics = applyNubarFactors(
        form, _block(11), np.geomspace(1.0e-5, 2.0e7, 12))
    assert isinstance(perturbed, Regions1d)
    _xs, _ys, pairs = _table(perturbed)
    assert len(pairs) == 1 and pairs[0][1] == 2, (
        "the perturbed table is one lin-lin region, which is what the format "
        "applier writes and what the sum rule is stated on")
