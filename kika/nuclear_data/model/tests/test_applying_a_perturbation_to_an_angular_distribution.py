"""The MF4 half of the gate: the model applier against the one the thesis runs.

``kika.nuclear_data.model.perturbation.applyLegendreFactors`` has to reproduce
``kika.sampling.endf_perturbation._apply_factors_to_mf4_legendre``, which is
what ``perturb_ENDF_files`` runs and therefore what every MF34 ensemble of this
project was drawn through. The comparison is on the numbers -- the incident
energies and the coefficient vectors after the perturbation -- rather than on
the encoded bytes, and that is the right level: the two paths reach ENDF through
different writers, so a byte comparison would be measuring the encoders, which
have their own gates, instead of the arithmetic, which does not.

The fixture is ``micro_fe56_structural.endf``: real JEFF-4.0 Fe-56 MF4/MT2 in
LTT=3 form (a Legendre region of 3960 incident energies followed by a tabulated
one) with its own MF34 stating L=1..6 on three different grids -- 42 bins for
the diagonal blocks, 15 and 12 for the cross-order ones. Three properties of
that tape make it the one worth gating on: the orders genuinely differ in grid,
so the per-order factor blocks are exercised rather than assumed; the
distribution is *mixed*, so the applier has to leave the tabulated region alone;
and the number of coefficients varies with energy (5, 7 and 9 across the file),
so the "this order does not exist here" branch is real.

**The factors are constructed, not drawn.** What is being measured is the
applier, and a draw would put a sampler between the question and the answer. A
deterministic vector that differs per order and per bin separates every case a
shared factor would confuse.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter import decodeCovarianceSuite, decodeMF4MT
from kika.nuclear_data.model.perturbation import (MAGNITUDE_ORDER,
                                                  applyLegendreFactors)
from kika.sampling.endf_perturbation import (_apply_factors_to_mf4_legendre,
                                             _parameter_mapping_from_index)
from kika.sampling.joint_blocks import collectEntries, requestIndex
from kika.sampling.model_blocks import legendre_covariance_index

TAPE = str(Path(__file__).resolve().parents[3] / "endf" / "tests" / "data"
           / "micro_fe56_structural.endf")
MT = 2


def _shippedIndexEntry():
    suite, _report = decodeCovarianceSuite(read_endf(TAPE))
    index = legendre_covariance_index(suite, relative=True)
    (entry,) = index.values()
    return entry


def _factorVector(size: int) -> np.ndarray:
    """Distinct, reproducible, and far enough from 1 that a miss is visible."""
    return 1.0 + 0.05 * np.cos(np.arange(size, dtype=float))


def _formatPath(factors, param_mapping, energy_grids):
    endf = read_endf(TAPE)
    mtData = endf.get_file(4).sections[MT]
    _apply_factors_to_mf4_legendre(mtData, factors, param_mapping, energy_grids,
                                   verbose=False)
    return (np.asarray(mtData._energies, dtype=float),
            [np.asarray(c, dtype=float) for c in mtData._legendre_coeffs])


def _cutPerOrder(factors, entry):
    """The flat draw, cut into one block per Legendre order the way the index says."""
    stride = entry["stride"]
    byOrder, edgesByOrder = {}, {}
    for position, triplet in enumerate(entry["triplets"]):
        order = int(triplet[2])
        grid = np.asarray(entry["grids"][triplet], dtype=float)
        width = len(grid) - 1
        start = position * stride
        byOrder[order] = np.asarray(factors[start:start + width], dtype=float)
        edgesByOrder[order] = grid
    return byOrder, edgesByOrder


def _modelPath(factors, entry, coverageEdges="ramp"):
    """The same draw through the model.

    ``coverageEdges="ramp"`` for the equivalence half of this file: it is the
    convention the shipped MF4 applier uses, and comparing under the default
    would be comparing two different conventions and calling the difference a
    regression. The default is measured on its own further down.
    """
    byOrder, edgesByOrder = _cutPerOrder(factors, entry)
    distribution, _provenance, _report = decodeMF4MT(read_endf(TAPE).get_file(4)
                                                     .sections[MT])
    perturbed, diagnostics = applyLegendreFactors(distribution.angular, byOrder,
                                                  edgesByOrder,
                                                  coverageEdges=coverageEdges)
    return perturbed, diagnostics, byOrder, edgesByOrder, distribution


def _legendreTable(node):
    """``(energies, coefficient vectors)`` of every Legendre region, in order."""
    from kika.nuclear_data.model.perturbation import _legendreRegions

    energies, vectors = [], []
    for _container, _position, region in _legendreRegions(node):
        for inner in region.function1ds:
            energies.append(float(inner.outerDomainValue))
            vectors.append(np.asarray(inner.coefficients, dtype=float))
    return np.asarray(energies, dtype=float), vectors


@pytest.fixture(scope="module")
def bothPaths():
    entry = _shippedIndexEntry()
    param_mapping, energy_grids = _parameter_mapping_from_index(entry)
    factors = _factorVector(len(param_mapping))
    formatEnergies, formatCoeffs = _formatPath(factors, param_mapping, energy_grids)
    perturbed, diagnostics, byOrder, edges, baseline = _modelPath(factors, entry)
    modelEnergies, modelCoeffs = _legendreTable(perturbed)
    return {
        "entry": entry, "factors": factors, "byOrder": byOrder, "edges": edges,
        "formatEnergies": formatEnergies, "formatCoeffs": formatCoeffs,
        "modelEnergies": modelEnergies, "modelCoeffs": modelCoeffs,
        "diagnostics": diagnostics, "perturbed": perturbed, "baseline": baseline,
    }


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------

def test_the_two_appliers_refine_the_same_energies(bothPaths):
    """Same count, same values, bit for bit -- the insertion rule is the same rule."""
    assert bothPaths["modelEnergies"].size == bothPaths["formatEnergies"].size
    assert np.array_equal(bothPaths["modelEnergies"], bothPaths["formatEnergies"])


def _factorAt(order, energy, byOrder, edges):
    """The factor *order*'s block states at *energy*; 1.0 outside its coverage."""
    grid = np.asarray(edges[order], dtype=float)
    at = int(np.searchsorted(grid, energy, side="right")) - 1
    if at < 0 or at >= len(byOrder[order]):
        return 1.0
    return float(byOrder[order][at])


#: How close two values have to be at an **inserted** energy to count as equal.
#:
#: Exact everywhere else, and the reason for the exception is worth stating
#: because it looks like a tolerance being granted to hide a difference. Both
#: paths interpolate the baseline vector at a bin edge with the same formula,
#: ``v1 + t*(v2 - v1)``. They differ in what they interpolate *between*: the
#: format applier keeps a copy of the original table and always brackets in it,
#: while this one refines the live table, so when two bin edges fall inside one
#: original interval the second interpolation runs between an original point and
#: a point the first insertion made. That is the same line, evaluated in two
#: steps instead of one -- identical in real arithmetic and not in floating
#: point. Measured on this fixture the disagreement is at the last bit; a
#: tolerance of 1e-12 is four orders of magnitude looser than the largest seen
#: and still tight enough that any difference of *rule* fails.
INSERTED_POINT_RTOL = 1e-12


def _atABinEdge(energy, edges):
    """Is *energy* a bin edge of any order? Those are the points that get inserted."""
    return any(np.any(np.isclose(np.asarray(grid, dtype=float), energy,
                                 rtol=0.0, atol=1e-10))
               for grid in edges.values())


def test_the_two_appliers_produce_the_same_coefficients(bothPaths):
    """Every ``a_l`` at every energy, exactly -- bar one difference, pinned below.

    ``==`` and not ``allclose``: both paths multiply the same baseline by the
    same double, so any difference is a difference of rule and not of rounding.

    **The two count orders differently, and that is what the offset is.** The
    format object stores what MF4 writes, ``a_1..a_NL``; the model stores
    ``a_0..a_NL`` with ``a_0 = 1``, because a Legendre form is a function and its
    zeroth coefficient is part of it. Model ``l`` is format ``l-1``, which is the
    same ``coeff_index = l_coeff - 1`` the format applier performs by hand at
    every access -- one of the two has to know it, and the model is where it is a
    property of the object rather than an arithmetic step in a loop.

    **The one difference, and it is the shipped applier's defect.** See
    ``library-gaps.md`` D32: at every inserted boundary that belongs to some
    *other* order's grid, ``_apply_factors_to_mf4_legendre`` writes the
    **baseline** value for every order whose own grid has no edge there, because
    its boundary pass starts from the baseline vector and only scales the orders
    it finds in ``lower_factors``/``upper_factors``. The coefficient is
    unperturbed at that energy and perturbed on both sides of it -- a notch, in
    a coefficient the covariance says is constant across that bin.

    This test does not paper over it. It asserts the difference is exactly that
    and nowhere else: where the two disagree, the shipped value is the baseline
    and this one is the baseline times the factor that order's own block states
    there. Any other disagreement fails.
    """
    modelCoeffs, formatCoeffs = bothPaths["modelCoeffs"], bothPaths["formatCoeffs"]
    energies = bothPaths["modelEnergies"]
    byOrder, edges = bothPaths["byOrder"], bothPaths["edges"]
    ownEdges = {order: np.asarray(edges[order], dtype=float) for order in byOrder}

    differing = []
    assert len(modelCoeffs) == len(formatCoeffs)
    for at, (mine, theirs) in enumerate(zip(modelCoeffs, formatCoeffs)):
        assert mine[0] == 1.0, (
            f"a_0 was scaled at energy index {at}: the magnitude does not live "
            f"in MF4 and nothing here may touch it")
        assert mine.size == theirs.size + 1, (
            f"different number of orders at energy index {at}: "
            f"{mine.size} model (a_0 included) vs {theirs.size} format")
        energy = float(energies[at])
        inserted = _atABinEdge(energy, edges)
        for order in range(1, mine.size):
            if mine[order] == theirs[order - 1]:
                continue
            if inserted and np.isclose(mine[order], theirs[order - 1],
                                       rtol=INSERTED_POINT_RTOL, atol=0.0):
                continue
            assert inserted, (
                f"L={order} at {energy:.6e} eV differs, and that energy is not a "
                f"bin edge: the two appliers disagree where nothing was inserted")
            factor = _factorAt(order, energy, byOrder, edges)
            assert np.isclose(theirs[order - 1] * factor, mine[order],
                              rtol=INSERTED_POINT_RTOL, atol=0.0), (
                f"L={order} at {energy:.6e} eV differs by something other than "
                f"this order's own factor: {theirs[order - 1]} * {factor} != "
                f"{mine[order]}")
            assert not np.any(np.isclose(ownEdges[order], energy, rtol=0.0,
                                         atol=1e-10)), (
                f"L={order} disagrees at {energy:.6e} eV, which IS an edge of "
                f"its own grid -- that is not D32, it is a new difference")
            differing.append((at, order))

    assert differing, (
        "the two appliers agreed everywhere: either D32 was fixed in "
        "endf_perturbation without this test being retired, or the fixture no "
        "longer has orders on different grids")
    assert len({at for at, _order in differing}) == 44, (
        f"D32 was measured at 44 energy points on this fixture -- 22 bin edges, "
        f"each written twice -- and this run found "
        f"{len({at for at, _order in differing})}")


def test_the_perturbation_actually_moved_something(bothPaths):
    """A gate that both paths pass by doing nothing is not a gate."""
    baseline, _provenance, _report = decodeMF4MT(read_endf(TAPE).get_file(4)
                                                 .sections[MT])
    baseEnergies, baseCoeffs = _legendreTable(baseline.angular)
    assert bothPaths["modelEnergies"].size > baseEnergies.size, (
        "the refinement inserted no energy at all")
    assert bothPaths["diagnostics"]["n_inserted"] > 0
    moved = sum(1 for base, after in zip(baseCoeffs, bothPaths["modelCoeffs"])
                if not np.array_equal(base[:min(base.size, after.size)],
                                      after[:min(base.size, after.size)]))
    assert moved > 100, f"only {moved} energies changed; the factors did not land"


# ----------------------------------------------------------------------
# What the model applier knows that the format one does not
# ----------------------------------------------------------------------

def test_the_tabulated_half_of_a_mixed_distribution_is_untouched(bothPaths):
    """MF34 says nothing about ``f(mu)`` tables, so neither does the applier."""
    from kika.nuclear_data.model.functions.higher import Regions2d

    before, after = bothPaths["baseline"].angular, bothPaths["perturbed"]
    assert isinstance(before, Regions2d) and isinstance(after, Regions2d)
    assert len(before.function2ds) == len(after.function2ds)

    from kika.nuclear_data.model.perturbation import _isLegendreRegion

    tabulated = [position for position, child in enumerate(before.function2ds)
                 if not _isLegendreRegion(child)]
    assert tabulated, "this fixture is meant to be LTT=3; it no longer is"
    for position in tabulated:
        original = before.function2ds[position]
        assert after.function2ds[position] is original, (
            "a region MF34 does not describe was rebuilt, so it is no longer "
            "guaranteed to encode to the same bytes")


def test_the_input_node_is_not_mutated(bothPaths):
    """The realisation goes *beside* the evaluation, so the evaluation must survive."""
    baseline, _provenance, _report = decodeMF4MT(read_endf(TAPE).get_file(4)
                                                 .sections[MT])
    fresh, _diagnostics = applyLegendreFactors(baseline.angular,
                                               bothPaths["byOrder"],
                                               bothPaths["edges"])
    stillOriginal, _vectors = _legendreTable(baseline.angular)
    reference, _reference = _legendreTable(
        decodeMF4MT(read_endf(TAPE).get_file(4).sections[MT])[0].angular)
    assert np.array_equal(stillOriginal, reference)
    assert fresh is not baseline.angular


def test_an_order_the_evaluation_does_not_carry_is_reported(bothPaths):
    """Not silently dropped: a covariance can outrun the evaluation it describes."""
    diagnostics = bothPaths["diagnostics"]
    assert set(diagnostics["per_order"]) == set(bothPaths["byOrder"])
    assert diagnostics["orders_absent"] == [], (
        "this tape carries every order its MF34 states, so nothing should be "
        "reported absent -- if that changed, the fixture changed")
    for order, stats in diagnostics["per_order"].items():
        assert stats["n_scaled"] > 0
        assert stats["min_factor"] <= 1.0 <= stats["max_factor"] or stats["n_scaled"]


def test_the_default_states_the_step_the_block_implies(bothPaths):
    """``coverageEdges="step"`` against ``"ramp"``: one energy, and which one.

    The two conventions are described in ``COVERAGE_EDGES``. What is measured
    here is the size of the choice, so that "the model applier and the shipped
    one differ" can never be a vague claim: on this fixture the whole difference
    is a repeated point at 20 MeV, the upper end of every order's covariance,
    where the shipped applier lets lin-lin ramp from the last perturbed energy
    to the first unperturbed one and this one writes the step the factor block
    actually states.
    """
    entry, factors = bothPaths["entry"], bothPaths["factors"]
    stepped, diagnostics, _byOrder, edges, _baseline = _modelPath(
        factors, entry, coverageEdges="step")
    steppedEnergies, _steppedCoeffs = _legendreTable(stepped)
    rampEnergies = bothPaths["modelEnergies"]

    extra = steppedEnergies.size - rampEnergies.size
    assert extra == 1, f"expected exactly one extra energy, got {extra}"

    added = sorted(set(np.round(steppedEnergies, 6)) - set(np.round(rampEnergies, 6)))
    assert not added, "the extra point repeats an energy the ramp already had"
    counts = {float(e): int(np.sum(steppedEnergies == e)) for e in steppedEnergies}
    doubledOnly = [e for e, n in counts.items()
                   if n > int(np.sum(rampEnergies == e))]
    assert doubledOnly == [2.0e7], (
        f"the step convention duplicated {doubledOnly}, not the covariance's "
        f"upper edge")
    assert all(float(np.asarray(grid)[-1]) == 2.0e7 for grid in edges.values())
    assert diagnostics["n_inserted"] == bothPaths["diagnostics"]["n_inserted"] + 1


def test_the_magnitude_order_is_refused_rather_than_applied():
    """a_0 is the cross section's size, and it is not in MF4 to be scaled."""
    baseline, _provenance, _report = decodeMF4MT(read_endf(TAPE).get_file(4)
                                                 .sections[MT])
    with pytest.raises(ValueError, match="magnitude"):
        applyLegendreFactors(baseline.angular,
                             {MAGNITUDE_ORDER: np.array([1.1])},
                             {MAGNITUDE_ORDER: np.array([1.0e3, 2.0e6])})


def test_factors_and_bins_have_to_describe_the_same_binning():
    baseline, _provenance, _report = decodeMF4MT(read_endf(TAPE).get_file(4)
                                                 .sections[MT])
    with pytest.raises(ValueError, match="factor"):
        applyLegendreFactors(baseline.angular, {1: np.array([1.1, 1.2])},
                             {1: np.array([1.0e3, 2.0e6])})
    with pytest.raises(ValueError, match="without a grid"):
        applyLegendreFactors(baseline.angular, {1: np.array([1.1])},
                             {2: np.array([1.0e3, 2.0e6])})


def test_the_two_indices_describe_the_same_grids():
    """The shipped MF34 index and the mixed-request one, on the same tape.

    They are different functions with different key shapes, and the appliers on
    both sides read their grids from them. If they ever disagreed, the model
    path and the format path would be perturbing different bins while both
    reporting success.
    """
    suite, _report = decodeCovarianceSuite(read_endf(TAPE))
    shipped = _shippedIndexEntry()
    (mine,) = requestIndex(collectEntries(suite, {34: None})).values()

    assert mine["stride"] == shipped["stride"]
    assert [(k.mt, k.index) for k in mine["components"]] == [
        (t[1], t[2]) for t in shipped["triplets"]]
    for component, triplet in zip(mine["components"], shipped["triplets"]):
        assert np.array_equal(mine["grids"][component], shipped["grids"][triplet])
