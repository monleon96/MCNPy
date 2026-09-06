"""Apply a drawn perturbation to a model node, and get a model node back.

**The point of this module is what it does not do.** Perturbing a cross section
has always meant editing a *file*: read the MF3 records, scale the numbers,
write the records back, and recompute the ``(NBT, INT)`` bookkeeping by hand.
That makes the perturbation a property of ENDF, so ACE and GNDS each need their
own version and the three can disagree. Here the operation is on the node --
:class:`~kika.nuclear_data.model.functions.xys1d.XYs1d` or
:class:`~kika.nuclear_data.model.functions.regions1d.Regions1d` -- and the
format is a way of writing the answer down afterwards.

The arithmetic is not new. It reproduces
:func:`kika.sampling.mf33_sampling.apply_factors_to_pendf_mf3`, which is what
the thesis pipeline runs, and the gate is byte-identity of the section that
comes out of ``encodeMF3MT`` against the section that applier writes. Three
pieces:

1. **Step duplicates at the bin edges.** A piecewise-constant factor block has
   a discontinuity at every interior bin edge, and ENDF-6 spells a discontinuity
   as a repeated abscissa carrying the left and right limits. So the table is
   refined first: at each interior edge that is not already a repeat, enough
   points are inserted to make one.
2. **The duplicate-aware bin rule.** The *first* of a repeated pair takes the
   factor of the bin that **ends** at that energy, everything else the factor of
   the bin that **starts** there. Outside the block's coverage the factor is 1.
3. **The ``(NBT, INT)`` bookkeeping** -- which is not here, and that is the whole
   difference. Insertion happens inside the region that owns the edge, so the
   pairs fall out of :meth:`Regions1d.toEndfRegions` counting region lengths.
   The format applier computes ``new_NBT_i = NBT_i + sum(n for pos < NBT_i)``
   by hand; that arithmetic and the point count were two sources of truth for
   the same fact, which is exactly the defect D28 was.

**Where this deliberately does not reproduce the format applier.** The value
inserted at a bin edge that the table does not already carry has to be
interpolated, and the format applier uses ``np.interp`` -- lin-lin, always,
because its input is a PENDF from RECONR and a PENDF is lin-lin. This module
asks the region that contains the edge, under that region's own law. The two
agree exactly wherever the format applier is used today, so the byte gate holds;
they differ on a log-log region, where ``np.interp`` is simply the wrong number
and there is no equivalence worth preserving.

**Not the perturbation envelope.** What a perturbation *is* -- which key, which
covariance, relative or absolute -- belongs to the sampling layer, which knows
about covariances. This module takes arrays: a factor per bin, and the bin
edges. The calculation layer may not import the format or sampling layers
(``kika/tests/test_layering.py``).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .functions.regions1d import Regions1d
from .functions.xys1d import XYs1d

__all__ = ["applyFactors", "refineAtBinEdges", "applyLegendreFactors",
           "MAGNITUDE_ORDER", "applyNubarFactors", "refineForNubar",
           "applySpectrumFactors", "summariseSpectrumNodes",
           "SPECTRUM_STEP_SHOULDER", "SPECTRUM_OUTER_SHOULDER",
           "SPECTRUM_ENERGY_RTOL"]

#: How close two abscissae must be to count as the repeated pair that ENDF-6
#: uses for a step. The same value the format applier uses, and it is an
#: absolute tolerance on purpose: it is asking "did the evaluator write this
#: energy twice", not "are these two numbers close in a relative sense".
ABSCISSA_ATOL = 1e-10

Function1dT = Union[XYs1d, Regions1d]


def _regionsOf(function1d: Function1dT) -> List[XYs1d]:
    """The regions of *function1d*, one for an :class:`XYs1d`."""
    if isinstance(function1d, Regions1d):
        return list(function1d.function1ds)
    return [function1d]


def _rebuild(function1d: Function1dT, regions: Sequence[XYs1d]) -> Function1dT:
    """A node of the same kind as *function1d*, carrying *regions*.

    The type is preserved rather than always returning a ``Regions1d``: a
    refinement inserts points, it does not split a region, so a one-region
    function stays one region and a caller who put an ``XYs1d`` into a
    ``crossSection`` gets an ``XYs1d`` back to put beside it.
    """
    if isinstance(function1d, Regions1d):
        return Regions1d(function1ds=list(regions), axes=function1d.axes,
                         label=function1d.label,
                         outerDomainValue=function1d.outerDomainValue,
                         index=function1d.index)
    region = regions[0]
    return XYs1d(xs=region.xs, ys=region.ys,
                 interpolation=function1d.interpolation, axes=function1d.axes,
                 label=function1d.label,
                 outerDomainValue=function1d.outerDomainValue,
                 index=function1d.index)


def _interiorEdges(edges: ArrayLike, lo: float, hi: float) -> List[float]:
    """The bin edges strictly inside ``(lo, hi)``, sorted and deduplicated.

    An edge on the domain boundary is not interior: there is nothing on the far
    side of it to step to, and inserting a repeat at ``xs[0]`` or ``xs[-1]``
    would add a point that carries no discontinuity.
    """
    edges = np.asarray(edges, dtype=float)
    inside = edges[(edges > lo + ABSCISSA_ATOL) & (edges < hi - ABSCISSA_ATOL)]
    return sorted(set(float(e) for e in inside))


def _regionOwning(regions: Sequence[XYs1d], edge: float) -> int:
    """Index of the region that owns *edge*.

    Regions share their boundary abscissa, so an edge sitting exactly on one
    belongs to the region that **ends** there. That is the choice that keeps the
    inserted repeat adjacent to the shared point in the flat table: the next
    region's copy of that abscissa is the one
    :meth:`Regions1d.toEndfRegions` drops.
    """
    for index, region in enumerate(regions):
        if edge <= float(region.xs[-1]) + ABSCISSA_ATOL:
            return index
    return len(regions) - 1


def refineAtBinEdges(function1d: Function1dT, binEdges: ArrayLike) -> Function1dT:
    """Insert the repeated abscissae a piecewise-constant factor block needs.

    Per interior bin edge, counting the abscissae the owning region already has
    there:

    * **two or more** -- left alone. The evaluation already writes a step at
      that energy (a threshold, most often) and a third point would not make it
      a better one.
    * **exactly one** -- a second copy is inserted beside it, carrying the same
      value. The pair becomes a real step once the two halves take different
      factors, and stays a non-step if they take the same one.
    * **none** -- two copies are inserted, both carrying the value the region's
      own interpolation gives at that energy.

    Returns a node of the same kind, sharing nothing with the original.
    """
    regions = [XYs1d(xs=np.array(r.xs, dtype=float, copy=True),
                     ys=np.array(r.ys, dtype=float, copy=True),
                     interpolation=r.interpolation, axes=r.axes, index=r.index)
               for r in _regionsOf(function1d)]
    if not regions or regions[0].xs.size < 2:
        return _rebuild(function1d, regions)

    edges = _interiorEdges(binEdges, float(regions[0].xs[0]),
                           float(regions[-1].xs[-1]))
    if not edges:
        return _rebuild(function1d, regions)

    # Descending, so an insertion never invalidates the position of one not yet
    # made. Within a region the positions are recomputed anyway, but the region
    # index would still shift if an edge landed on a boundary.
    for edge in reversed(edges):
        k = _regionOwning(regions, edge)
        region = regions[k]
        xs, ys = region.xs, region.ys

        matches = np.flatnonzero(np.abs(xs - edge) <= ABSCISSA_ATOL)
        if matches.size >= 2:
            continue
        if matches.size == 1:
            at = int(matches[0])
            newXs = np.insert(xs, at + 1, xs[at])
            newYs = np.insert(ys, at + 1, ys[at])
        else:
            at = int(np.searchsorted(xs, edge, side="right"))
            value = float(region.evaluate(edge, outOfRange="hold"))
            newXs = np.insert(xs, at, [edge, edge])
            newYs = np.insert(ys, at, [value, value])

        regions[k] = XYs1d(xs=newXs, ys=newYs, interpolation=region.interpolation,
                           axes=region.axes, index=region.index)

    return _rebuild(function1d, regions)


def _flatFactors(xs: np.ndarray, factors: np.ndarray,
                 binEdges: np.ndarray) -> np.ndarray:
    """The factor each abscissa of the flat table takes.

    The duplicate-aware rule, and it is stated on the **flat** table rather than
    per region on purpose. A repeated abscissa is a property of the ENDF table,
    not of an interpolation region -- a pair can sit either side of a region
    boundary -- so asking each region separately would give the wrong answer at
    exactly the energies this exists for.
    """
    nGroups = binEdges.size - 1
    right = np.searchsorted(binEdges, xs, side="right") - 1
    left = np.searchsorted(binEdges, xs, side="left") - 1

    firstOfPair = np.zeros(xs.size, dtype=bool)
    if xs.size >= 2:
        firstOfPair[:-1] = np.isclose(xs[:-1], xs[1:], rtol=0.0,
                                      atol=ABSCISSA_ATOL)

    index = np.where(firstOfPair, left, right)
    covered = (index >= 0) & (index < nGroups)
    out = np.ones(xs.size, dtype=float)
    out[covered] = factors[index[covered]]
    return out


def applyFactors(function1d: Function1dT, factors: ArrayLike,
                 binEdges: ArrayLike) -> Tuple[Function1dT, dict]:
    """Scale *function1d* by a piecewise-constant factor per bin.

    Parameters
    ----------
    function1d
        The form to perturb -- ``crossSection['eval']``, typically. Not
        mutated.
    factors
        One factor per bin, so ``len(binEdges) - 1`` of them.
    binEdges
        The covariance's own energy grid, ascending, in the units the function's
        domain is stated in.

    Returns
    -------
    (perturbed, diagnostics)
        A node of the same kind as *function1d*, and what the application did:
        ``min_factor``, ``max_factor``, ``frac_out_of_coverage`` and
        ``n_inserted``. The diagnostics are returned rather than logged because
        the caller is drawing thousands of these and decides what is worth
        saying.

    Raises
    ------
    ValueError
        If *factors* and *binEdges* do not describe the same binning. Guessing
        which of the two is right would put a silently misaligned perturbation
        on a cross section, which is the failure this whole track exists to
        avoid.
    """
    factors = np.asarray(factors, dtype=float)
    binEdges = np.asarray(binEdges, dtype=float)
    if binEdges.ndim != 1 or binEdges.size < 2:
        raise ValueError(
            f"binEdges must be at least two ascending energies, got shape "
            f"{binEdges.shape}"
        )
    if factors.size != binEdges.size - 1:
        raise ValueError(
            f"{factors.size} factor(s) for {binEdges.size - 1} bin(s): a factor "
            f"block and its grid have to describe the same binning"
        )

    before = sum(r.xs.size for r in _regionsOf(function1d))
    refined = refineAtBinEdges(function1d, binEdges)
    regions = _regionsOf(refined)

    xs, _, _ = refined.toEndfRegions()
    perPoint = _flatFactors(np.asarray(xs, dtype=float), factors, binEdges)

    # Scatter back onto the regions. Region i>0 repeats the boundary abscissa
    # that the flat table states once, so it re-reads the same entry rather
    # than consuming the next one -- the two copies are one point and must take
    # one factor.
    scaled: List[XYs1d] = []
    cursor = 0
    for index, region in enumerate(regions):
        n = region.xs.size
        start = cursor - 1 if index else cursor
        block = perPoint[start:start + n]
        scaled.append(XYs1d(xs=np.array(region.xs, dtype=float, copy=True),
                            ys=np.asarray(region.ys, dtype=float) * block,
                            interpolation=region.interpolation,
                            axes=region.axes, index=region.index))
        cursor = start + n

    covered = float(np.mean(
        (np.searchsorted(binEdges, xs, side="right") - 1 >= 0)
        & (np.searchsorted(binEdges, xs, side="right") - 1 < factors.size)
    )) if len(xs) else 0.0

    diagnostics = {
        "min_factor": float(perPoint.min()) if perPoint.size else 1.0,
        "max_factor": float(perPoint.max()) if perPoint.size else 1.0,
        "frac_out_of_coverage": 1.0 - covered,
        "n_inserted": int(sum(r.xs.size for r in regions)) - int(before),
    }
    return _rebuild(refined, scaled), diagnostics


# ======================================================================
# Angular distributions: the same arithmetic, one axis further out
# ======================================================================

#: Legendre order 0 is the *magnitude*, and it is not applied here. In ENDF MF4
#: ``a_0`` is identically 1 -- the size of the cross section lives in MF3 -- so
#: an MF34 covariance that states an L=0 component is stating the uncertainty of
#: sigma(E), on MF34's own grid. Scaling the model's ``coefficients[0]`` by it
#: would put the magnitude into the *shape*, which is a normalisation error and
#: not a perturbation of anything the file describes.
#:
#: ``endf_perturbation._apply_factors_to_mf4_legendre`` reaches the same
#: conclusion and routes those factors to an optional ``mf3_magnitude_sink``,
#: dormant since it was written. Here the routing is the caller's, because the
#: caller is the only one holding both nodes -- see
#: ``kika.sampling.perturbation_set.PerturbationSet.applyToSuite``.
MAGNITUDE_ORDER = 0


def _isLegendreRegion(node) -> bool:
    from .functions.simple import Legendre

    inner = getattr(node, "function1ds", None)
    return bool(inner) and all(isinstance(f, Legendre) for f in inner)


def _legendreRegions(angular):
    """The ``XYs2d`` children of *angular* whose inner functions are Legendre.

    A mixed (LTT=3) distribution is a ``Regions2d`` of a Legendre region and a
    tabulated one, and MF34 says nothing about the tabulated half -- there are
    no ``a_l`` there to perturb. Returns ``(container, position, xys2d)``
    triples so a caller can put a rebuilt region back where it came from;
    ``container`` is ``None`` when *angular* is itself the region.
    """
    from .functions.higher import Regions2d, XYs2d

    found = []
    if isinstance(angular, XYs2d):
        if _isLegendreRegion(angular):
            found.append((None, 0, angular))
        return found
    if isinstance(angular, Regions2d):
        for position, child in enumerate(angular.function2ds):
            if isinstance(child, XYs2d) and _isLegendreRegion(child):
                found.append((angular, position, child))
            elif isinstance(child, Regions2d):
                found.extend(_legendreRegions(child))
    return found


def _interpolateCoefficients(energy: float, xs: np.ndarray, vectors,
                             interpolation) -> np.ndarray:
    """The Legendre vector at *energy*, under the outer axis's own rule.

    Vectors of different lengths are padded with zeros to the longer, which is
    what an absent ``a_l`` means: the evaluation writes the orders it needs at
    each energy and the rest are zero. ``_interpolate_legendre_coefficients``
    does the same, and this is the one place the two have to agree numerically.
    """
    from .enums import Interpolation

    if energy <= xs[0]:
        return np.array(vectors[0], dtype=float, copy=True)
    if energy >= xs[-1]:
        return np.array(vectors[-1], dtype=float, copy=True)

    upper = int(np.searchsorted(xs, energy, side="left"))
    lower = max(upper - 1, 0)
    e1, e2 = float(xs[lower]), float(xs[upper])
    c1 = np.asarray(vectors[lower], dtype=float)
    c2 = np.asarray(vectors[upper], dtype=float)
    width = max(c1.size, c2.size)
    c1 = np.pad(c1, (0, width - c1.size))
    c2 = np.pad(c2, (0, width - c2.size))

    if abs(e2 - e1) < 1e-15:
        return c1.copy()

    interpolation = Interpolation(interpolation)
    if interpolation is Interpolation.flat:
        return c1.copy()
    if interpolation is not Interpolation.linlin:
        raise NotImplementedError(
            f"the incident-energy axis of this distribution interpolates "
            f"{interpolation.value!r}; inserting a point at a covariance bin "
            f"edge needs the coefficient vector there, and only lin-lin and "
            f"flat have an answer this module will give. MF4 is lin-lin on "
            f"every tape to hand, so this is a case to look at rather than one "
            f"to default"
        )
    t = (energy - e1) / (e2 - e1)
    return c1 + t * (c2 - c1)


def _refineLegendreRegion(region, edges: ArrayLike):
    """Insert the repeated incident energies the factor blocks need.

    The two-dimensional counterpart of :func:`refineAtBinEdges`, with the same
    three cases per edge -- two or more copies already there, exactly one, or
    none -- except that what gets duplicated is a whole Legendre vector rather
    than a single ordinate.
    """
    from .functions.higher import XYs2d
    from .functions.simple import Legendre

    inner = list(region.function1ds)
    xs = np.array([float(f.outerDomainValue) for f in inner], dtype=float)
    interior = _interiorEdges(edges, float(xs[0]), float(xs[-1]))
    if not interior:
        return region, 0

    inserted = 0
    for edge in reversed(interior):
        xs = np.array([float(f.outerDomainValue) for f in inner], dtype=float)
        matches = np.flatnonzero(np.abs(xs - edge) <= ABSCISSA_ATOL)
        if matches.size >= 2:
            continue
        if matches.size == 1:
            at = int(matches[0])
            twin = inner[at]
            inner.insert(at + 1, Legendre(
                coefficients=np.array(twin.coefficients, dtype=float, copy=True),
                axes=twin.axes, outerDomainValue=twin.outerDomainValue,
                index=twin.index))
            inserted += 1
        else:
            at = int(np.searchsorted(xs, edge, side="right"))
            values = _interpolateCoefficients(
                edge, xs, [f.coefficients for f in inner], region.interpolation)
            template = inner[min(at, len(inner) - 1)]
            inner[at:at] = [
                Legendre(coefficients=np.array(values, dtype=float, copy=True),
                         axes=template.axes, outerDomainValue=float(edge),
                         index=template.index)
                for _ in range(2)]
            inserted += 2

    return XYs2d(function1ds=inner, interpolation=region.interpolation,
                 interpolationQualifier=region.interpolationQualifier,
                 axes=region.axes, label=region.label,
                 outerDomainValue=region.outerDomainValue,
                 index=region.index), inserted


#: What happens at the outer boundary of a factor block's coverage, where the
#: perturbation stops and the evaluation continues unperturbed.
#:
#: ``"step"``
#:     A repeated incident energy there, so the last bin's factor holds to the
#:     edge and the untouched evaluation resumes on the far side. This is what
#:     the block says -- the factor is that value up to the edge and 1 beyond --
#:     and it is the convention MF3's applier already uses: ``mf33_sampling``
#:     inserts at *every* bin edge inside the table's span, its own outermost
#:     included, and the byte gate of ``applyFactors`` holds it there.
#:
#: ``"ramp"``
#:     No point inserted, so lin-lin runs from the last perturbed energy to the
#:     first unperturbed one and the discontinuity is smeared across that one
#:     interval. **This is what ``_apply_factors_to_mf4_legendre`` does** --
#:     it collects boundaries per order as ``grid[1:-1]``, dropping each grid's
#:     own ends -- and so it is what every MF34 ensemble of this project was
#:     drawn through.
#:
#: The default is the faithful one and the other is the escape hatch that lets
#: a gate prove equivalence with the shipped path, the same arrangement
#: ``draw_samples``' ``null_tol=None`` has. On the Fe-56 fixture the whole
#: difference is one inserted energy at 20 MeV out of 49: the two agree
#: everywhere the perturbation is actually stated, and disagree only about how
#: it ends.
COVERAGE_EDGES = ("step", "ramp")


def applyLegendreFactors(angular, factors, binEdges, *, coverageEdges="step"):
    """Scale the Legendre coefficients of an angular distribution, order by order.

    Parameters
    ----------
    angular
        The ``XYs2d`` or ``Regions2d`` of :class:`Legendre` forms that an
        :class:`~kika.nuclear_data.model.distributions.AngularTwoBody` carries.
        Not mutated.
    factors
        ``Legendre order -> one factor per bin``. Order 0 is refused; see
        :data:`MAGNITUDE_ORDER`.
    binEdges
        ``Legendre order -> the incident-energy boundaries those factors are
        stated on``. MF34's orders routinely differ in grid, which is why this
        is per order rather than one grid for all of them.
    coverageEdges
        What to do where a factor block's coverage ends; see
        :data:`COVERAGE_EDGES`. The default states the step the block implies;
        ``"ramp"`` reproduces the shipped MF4 applier and is there for gates.

    Returns
    -------
    (perturbed, diagnostics)
        A node of the same kind, and ``per_order`` (min and max factor and how
        many coefficients were scaled, per order), ``n_inserted``, and
        ``orders_absent`` -- the requested orders the distribution carries at no
        energy. That last is a real case, not a defensive check: a file may
        state a covariance for an order the evaluation itself stops short of,
        and a caller should be told rather than have it silently dropped.

    **What this reproduces, and where it deliberately does not.** The arithmetic
    is ``_apply_factors_to_mf4_legendre``'s: the union of every order's interior
    bin edges becomes a repeated incident energy carrying the baseline vector,
    the copy below the edge takes the factor of the bin that ends there and the
    copy above the factor of the bin that starts there, and every other energy
    takes the factor of the bin containing it. What it does not reproduce is
    that function's split into an interior pass and a boundary pass: here it is
    one rule, stated on the refined energy list, which is the same rule
    :func:`applyFactors` states for a cross section. The two-pass version has an
    exposure this one does not -- an energy the evaluation already wrote twice
    gets a third and a fourth copy there, where here it is left alone, which is
    the case :func:`refineAtBinEdges` documents for MF3.

    A perturbed distribution is **not renormalised and its positivity is not
    enforced**. The sum ``sum_l (2l+1)/2 a_l P_l(mu)`` can go negative for a
    large enough factor, and the projection that repairs it lives in
    :mod:`kika.sampling.mf4_positivity` -- a property of the *sample*, decided
    by the sampler, and not something an applier may do unasked.
    """
    from .functions.higher import Regions2d, XYs2d
    from .functions.simple import Legendre

    orders = sorted(int(order) for order in factors)
    if MAGNITUDE_ORDER in orders:
        raise ValueError(
            f"Legendre order {MAGNITUDE_ORDER} is the cross-section magnitude, "
            f"not a shape coefficient: in MF4 a_0 is identically 1. Its factors "
            f"belong on the crossSection node of the same reaction and routing "
            f"them there is the caller's job -- applying them here would put the "
            f"magnitude into the shape"
        )
    if set(orders) != {int(order) for order in binEdges}:
        raise ValueError(
            f"order(s) {sorted(set(orders) ^ {int(o) for o in binEdges})} have "
            f"factors without a grid or a grid without factors; a block and its "
            f"bins are one object"
        )
    for order in orders:
        nFactors = len(np.asarray(factors[order]))
        nBins = len(np.asarray(binEdges[order])) - 1
        if nFactors != nBins:
            raise ValueError(f"L={order}: {nFactors} factor(s) on {nBins} bin(s)")

    regions = _legendreRegions(angular)
    if not regions:
        raise ValueError(
            "this distribution carries no Legendre coefficients, so an MF34 "
            "perturbation has nothing to act on. A tabulated (LTT=2) angular "
            "distribution is perturbed as a table, which is a different applier"
        )

    if coverageEdges not in COVERAGE_EDGES:
        raise ValueError(
            f"coverageEdges must be one of {COVERAGE_EDGES}, got "
            f"{coverageEdges!r}")
    grids = [np.asarray(binEdges[order], dtype=float) for order in orders]
    if coverageEdges == "ramp":
        grids = [grid[1:-1] for grid in grids if grid.size > 2]
    allEdges = (np.unique(np.concatenate(grids)) if grids
                else np.zeros(0, dtype=float))

    perOrder = {order: {"min_factor": 1.0, "max_factor": 1.0, "n_scaled": 0}
                for order in orders}
    inserted = 0
    rebuilt = {}
    for container, position, region in regions:
        refined, added = _refineLegendreRegion(region, allEdges)
        inserted += added

        xs = np.array([float(f.outerDomainValue) for f in refined.function1ds],
                      dtype=float)
        coefficients = [np.array(f.coefficients, dtype=float, copy=True)
                        for f in refined.function1ds]
        for order in orders:
            perPoint = _flatFactors(xs, np.asarray(factors[order], dtype=float),
                                    np.asarray(binEdges[order], dtype=float))
            touched = 0
            for at, vector in enumerate(coefficients):
                if order >= vector.size:
                    continue
                vector[order] *= perPoint[at]
                touched += 1
            if touched:
                stats = perOrder[order]
                stats["min_factor"] = min(stats["min_factor"], float(perPoint.min()))
                stats["max_factor"] = max(stats["max_factor"], float(perPoint.max()))
                stats["n_scaled"] += touched

        scaled = [Legendre(coefficients=vector, axes=f.axes,
                           outerDomainValue=f.outerDomainValue, index=f.index)
                  for f, vector in zip(refined.function1ds, coefficients)]
        rebuilt[(id(container), position)] = XYs2d(
            function1ds=scaled, interpolation=refined.interpolation,
            interpolationQualifier=refined.interpolationQualifier,
            axes=refined.axes, label=refined.label,
            outerDomainValue=refined.outerDomainValue, index=refined.index)

    def rebuild(node):
        if isinstance(node, Regions2d):
            children = []
            for position, child in enumerate(node.function2ds):
                replacement = rebuilt.get((id(node), position))
                children.append(replacement if replacement is not None
                                else rebuild(child))
            return Regions2d(function2ds=children, axes=node.axes,
                             label=node.label,
                             outerDomainValue=node.outerDomainValue,
                             index=node.index)
        return rebuilt.get((id(None), 0), node)

    diagnostics = {
        "per_order": perOrder,
        "n_inserted": inserted,
        "orders_absent": [order for order in orders
                          if perOrder[order]["n_scaled"] == 0],
    }
    return rebuild(angular), diagnostics


# ======================================================================
# Multiplicities: the same arithmetic, and a different way of writing a step
# ======================================================================

#: How far below a bin edge the second inserted node goes, as a fraction of the
#: edge. ``kika.sampling.mf31_sampling.NUBAR_STEP_SHOULDER``'s value, and it has
#: to stay the same number: the two appliers are gated against each other.
NUBAR_STEP_SHOULDER = 1.0e-3

#: How close an inserted node may be to one already there, relatively, before it
#: is dropped. Closer than this and ENDF's six digits would write the two as the
#: same energy -- a duplicate abscissa, which is the one thing this construction
#: exists to avoid.
NUBAR_ENERGY_RTOL = 1.0e-6


def refineForNubar(xs: ArrayLike, ys: ArrayLike, binEdges: ArrayLike, *,
                   shoulder: float = NUBAR_STEP_SHOULDER):
    """Resolve a nu-bar table against a factor block's bins, without duplicates.

    **This is the other way of writing a step, and the reason there are two.**
    A cross section says "the factor changes here" with a repeated abscissa --
    :func:`refineAtBinEdges` -- because ENDF reads a repeat as a discontinuity.
    A nu-bar may not: the family has to satisfy nu_452 = nu_455 + nu_456, and
    that sum is only exact as a piecewise-linear identity if no member carries a
    duplicate energy. NJOY's ACER has the second reason -- it reads MF1 into
    fixed-size buffers and a bloated nu-bar table is not free.

    So the step is approximated instead of stated: one node at each interior bin
    edge, and one *shoulder* just below it, at ``edge * (1 - shoulder)``. The
    ramp between the two factors is then confined to that sliver rather than
    spanning the bin. ``kika.sampling.mf31_sampling._augment_nubar_grid``, which
    this reproduces, measured what that is worth on the three JEFF-4.0 actinide
    tapes with a worst-case alternating +-5 % block: the maximum per-bin factor
    error falls from 0.050, with edges alone, to 5e-4 -- while the table grows
    *less* than a uniform subdivision of every bin would make it.

    Both kinds of node are valued by lin-lin interpolation on the **original**
    table, so the baseline curve is preserved exactly.

    Returns ``(xs, ys)``, one lin-lin region's worth.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return xs, ys

    edges = np.asarray(binEdges, dtype=float)
    lo, hi = float(xs[0]), float(xs[-1])

    candidates: List[float] = []
    for edge in edges[1:-1]:
        edge = float(edge)
        # ``edge == hi`` still wants a shoulder: the table's last point sits on
        # a bin boundary and takes the *upper* bin's factor, so without one the
        # whole top of that bin ramps towards its neighbour. The block's own
        # last edge is excluded because the top-edge clamp below already pulls a
        # point there back into the last bin -- there is no step to resolve.
        if not (lo < edge <= hi):
            continue
        if edge < hi:
            candidates.append(edge)
        shoulderEnergy = edge * (1.0 - shoulder)
        if shoulderEnergy > lo:
            candidates.append(shoulderEnergy)

    if not candidates:
        return xs, ys

    cand = np.unique(np.asarray(candidates, dtype=float))
    cand = cand[(cand > lo) & (cand < hi)]
    if cand.size:
        nearest = np.searchsorted(xs, cand).clip(1, xs.size - 1)
        keep = ~(np.isclose(cand, xs[nearest], rtol=NUBAR_ENERGY_RTOL, atol=0.0)
                 | np.isclose(cand, xs[nearest - 1], rtol=NUBAR_ENERGY_RTOL,
                              atol=0.0))
        cand = cand[keep]
    if not cand.size:
        return xs, ys

    newXs = np.concatenate([xs, cand])
    newYs = np.concatenate([ys, np.interp(cand, xs, ys)])
    order = np.argsort(newXs, kind="mergesort")
    return newXs[order], newYs[order]


def _oneLinLinRegion(template, xs, ys):
    """A node of *template*'s kind holding one lin-lin region.

    **A ``Regions1d`` stays a ``Regions1d`` even with one region**, and that is
    not tidiness: ``_tab1FromMultiplicity`` writes MF1 from ``regions1d``
    (LNU=2) or ``polynomial1d`` (LNU=1) and **refuses an ``XYs1d``**, so a
    perturbed nu-bar returned as a bare curve is a realisation that cannot be
    written back. Found by the encoder, which is the right place to find it.
    """
    from .enums import Interpolation
    from .functions.regions1d import Regions1d
    from .functions.xys1d import XYs1d

    region = XYs1d(xs=np.asarray(xs, dtype=float),
                   ys=np.asarray(ys, dtype=float),
                   interpolation=Interpolation.linlin,
                   axes=getattr(template, "axes", None))
    if isinstance(template, Regions1d):
        return Regions1d(function1ds=[region], axes=template.axes,
                         label=template.label,
                         outerDomainValue=template.outerDomainValue,
                         index=template.index)
    return XYs1d(xs=region.xs, ys=region.ys, interpolation=Interpolation.linlin,
                 axes=getattr(template, "axes", None),
                 label=getattr(template, "label", None),
                 outerDomainValue=getattr(template, "outerDomainValue", None),
                 index=getattr(template, "index", None))


def applyNubarFactors(function1d, factors: ArrayLike, binEdges: ArrayLike, *,
                      shoulder: float = NUBAR_STEP_SHOULDER
                      ) -> Tuple[Any, dict]:
    """Scale a multiplicity by a piecewise-constant factor per bin.

    The nu-bar counterpart of :func:`applyFactors`, and it differs in exactly
    two places, both of them forced:

    * the refinement inserts **single** nodes and a shoulder rather than a
      repeated abscissa -- see :func:`refineForNubar`;
    * a point sitting exactly on the block's **last** edge takes the last bin's
      factor rather than falling outside coverage. Without that, the top point
      of a table whose domain ends where the covariance does comes back
      unperturbed, which reads as a ramp to baseline over the final bin.

    Returns ``(perturbed, diagnostics)``. The perturbed node is an
    :class:`~kika.nuclear_data.model.functions.xys1d.XYs1d` with one lin-lin
    region **whatever went in**, because that is what the format applier writes
    (``_set_nubar`` passes ``[(N, 2)]``) and because the sum rule is stated on
    lin-lin interpolants. A multi-region or non-lin-lin input is therefore
    flattened, and ``regions_collapsed`` in the diagnostics says so rather than
    it happening quietly.
    """
    from .enums import Interpolation

    factors = np.asarray(factors, dtype=float)
    edges = np.asarray(binEdges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            f"binEdges must be at least two ascending energies, got shape "
            f"{edges.shape}")
    if factors.size != edges.size - 1:
        raise ValueError(
            f"{factors.size} factor(s) for {edges.size - 1} bin(s): a factor "
            f"block and its grid have to describe the same binning")

    regions = _regionsOf(function1d)
    xs, ys, pairs = function1d.toEndfRegions()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    collapsed = len(pairs) > 1 or any(
        region.interpolation is not Interpolation.linlin for region in regions)

    before = xs.size
    xs, ys = refineForNubar(xs, ys, edges, shoulder=shoulder)

    perPoint = _flatFactors(xs, factors, edges)
    atTop = np.isclose(xs, edges[-1], rtol=1e-12, atol=0.0)
    if atTop.any():
        perPoint = np.where(atTop, factors[-1], perPoint)

    covered = (np.searchsorted(edges, xs, side="right") - 1 >= 0) & (
        np.searchsorted(edges, xs, side="right") - 1 < factors.size)
    covered = covered | atTop

    perturbed = _oneLinLinRegion(function1d, xs, ys * perPoint)
    diagnostics = {
        "min_factor": float(perPoint.min()) if perPoint.size else 1.0,
        "max_factor": float(perPoint.max()) if perPoint.size else 1.0,
        "frac_out_of_coverage": float(1.0 - covered.mean()) if xs.size else 0.0,
        "n_inserted": int(xs.size - before),
        "regions_collapsed": bool(collapsed),
    }
    return perturbed, diagnostics


# ======================================================================
# Energy distributions: two axes, a norm, and a self-check
# ======================================================================
#
# **The third node, and the one the other two do not prepare you for.** A cross
# section and a nu-bar are scaled: refine at the bin edges, multiply. A fission
# spectrum is not, because MF35 is not the covariance of chi(E') -- it is the
# covariance of the *group-integrated probabilities*
# ``P_j = int_{g_j}^{g_j+1} chi``, measured three independent ways
# (``docs/pfns/pfns_mf5_mf35_roadmap.md`` fact 4). So the perturbation is stated
# on quantities the node does not store, and applying it means integrating the
# node, turning a delta on the integral into a ratio, scaling the table so the
# integral comes out right, and then checking that it did.
#
# The arithmetic reproduces ``kika.sampling.mf35_sampling.perturb_pfns_partial``
# and the gate is that the two produce the same tables point for point. What is
# **not** here is what a *sample* is allowed to be -- positivity, the simplex
# projection, what to do with a group whose P0 is zero. That is a property of
# the draw and it belongs to the sampler; it arrives through *ratiosFor*.

#: How far below an outgoing group boundary the shoulder node goes, as a
#: fraction of the **lower group's width** -- not of the boundary energy.
#:
#: A duplicate-free lin-lin table cannot express a step, so the perturbed curve
#: ramps between two adjacent groups' factors and the shoulder confines that
#: ramp to a sliver. :data:`NUBAR_STEP_SHOULDER` measures the sliver as a
#: fraction of the *energy* and is right to, because MF31's bins span decades so
#: ``s*E`` is always a small part of a bin.
#:
#: **That does not carry over, and copying it was measurably wrong.** An MF5
#: outgoing grid is linear at the top: Cf-252's highest groups are 2e5 wide at
#: 2e7 eV, where ``s*E`` = 2e4 is a *tenth of the group*. The ramp then spans a
#: tenth of the group and the realised group integral misses ``r_j*P0_j`` by
#: percent, which is the same size as the perturbation. Measured on Cf-252: max
#: group error 1.5 with the energy-relative shoulder, 8e-4 with this one.
#:
#: ``kika.sampling.mf35_sampling.OUTGOING_STEP_SHOULDER`` holds the same number
#: and has to: the two appliers are gated against each other.
SPECTRUM_STEP_SHOULDER = 1.0e-3

#: The same, for the step in the *outer* coordinate at a band edge. The factor
#: is piecewise constant in incident energy, so it steps at every band edge, and
#: without a node just below the edge the lower band's factor ramps across the
#: whole preceding incident interval.
SPECTRUM_OUTER_SHOULDER = 1.0e-3

#: How close an inserted node may be to one already there, relatively, before it
#: is dropped -- ENDF's eleven-character field would write the two as the same
#: energy, and a duplicate abscissa is what the shoulder construction exists to
#: avoid.
SPECTRUM_ENERGY_RTOL = 1.0e-6


def _asBandMap(bands) -> "dict":
    """*bands* as ``{key: (lo, hi)}``, whichever way it was given.

    A mapping keeps the caller's own names for its blocks -- for MF35 the band
    index the file states, which is **not** the position in a list the moment a
    request asks for some bands and not others. A bare sequence still works and
    is keyed by position, which is what a caller perturbing every band means.
    """
    if hasattr(bands, "items"):
        return {key: (float(lo), float(hi)) for key, (lo, hi) in bands.items()}
    return {index: (float(lo), float(hi))
            for index, (lo, hi) in enumerate(bands)}


def _bandOf(value: float, bands: "dict"):
    """The key of the band containing *value*: ``[E1, E2)``, the last closed."""
    ordered = sorted(bands.items(), key=lambda item: item[1][0])
    for position, (key, (lo, hi)) in enumerate(ordered):
        if lo <= value < hi:
            return key
        if position == len(ordered) - 1 and value == hi:
            return key
    return None


def _insertOuterShoulders(form, bands, *, shoulder=SPECTRUM_OUTER_SHOULDER,
                          rtol=SPECTRUM_ENERGY_RTOL) -> int:
    """Give every interior band edge an outer node just below it.

    Band edges are already nodes on every tape measured -- 6/6 on ENDF/B-VIII.1
    and 9/9 on JEFF-4.0 -- but that only puts a node *at* the step. The new
    node's table comes from
    :meth:`~kika.nuclear_data.model.functions.higher.XYs2d.evaluateAtOuter`,
    which is an exact refinement; copying the edge's own table instead is ~0.1 %
    wrong and would surface as a central-value shift in a zero-perturbation run,
    which is a defect that looks like physics.
    """
    # Every band's lower edge, and not "all but the lowest". The factor steps
    # from *whatever is below* to this band's, and below the lowest requested
    # band there may be a band nobody asked to perturb -- a partial request is
    # exactly the case where the lowest edge is an interior step. Where it is
    # not interior, the domain guard below drops it, which is what happens on
    # both committed tapes: band 0 starts at the first incident energy. The
    # ENDF twin skips ``bands[1:]`` instead, which is right only because it is
    # always handed the whole band list.
    edges = sorted({float(lo) for lo, _ in bands.values()})
    inserted = 0
    for edge in edges:
        below = edge * (1.0 - shoulder)
        existing = np.asarray(form.outerDomainValues, dtype=float)
        if existing.size == 0 or not (existing[0] < below < existing[-1]):
            continue
        nearest = existing[np.argmin(np.abs(existing - below))]
        if np.isclose(below, nearest, rtol=rtol, atol=0.0):
            continue
        form.insertOuterNode(below)
        inserted += 1
    return inserted


def _refineAndScale(xs, ys, boundaries, ratios, *, maxPoints=None,
                    shoulder=SPECTRUM_STEP_SHOULDER, rtol=SPECTRUM_ENERGY_RTOL):
    """Refine a table where the factor steps, then scale every node.

    Returns ``(xs, ys, nInserted, nStepsDropped)``.

    **This is the load-bearing step and the single most likely silent bug.**
    Scaling the nodes without first inserting the group boundary applies the
    factor over the wrong interval: the panel straddling a boundary gets one
    group's factor across both groups. The realised group integral then stops
    being ``r_j * P0_j``, which is the property that makes the whole
    construction preserve normalisation -- and the renormalisation that follows
    absorbs the difference into a scalar, so the run completes, the file is
    valid, and the perturbation is not the one that was computed.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    lo, hi = float(xs[0]), float(xs[-1])

    # Boundary ``interior[i]`` separates groups i and i+1, so the factor steps
    # there iff ``ratios[i+1] != ratios[i]``. Only steps strictly inside the
    # table's own support can be represented at all.
    interior = boundaries[1:-1]
    widths = np.diff(boundaries)[:-1]            # width of the group *below*
    jumps = np.abs(ratios[1:] - ratios[:-1])
    stepping = (jumps != 0.0) & (interior > lo) & (interior < hi)
    candidates = interior[stepping]
    magnitudes = jumps[stepping]
    shoulderWidths = widths[stepping]

    nDropped = 0
    if maxPoints is not None and candidates.size:
        # Two nodes per surviving step. Drop the smallest steps first, and say
        # how many -- a silent cap reads as "the factor was applied
        # everywhere", which is exactly the claim that would then be false.
        budget = max(int((maxPoints - xs.size) // 2), 0)
        if budget < candidates.size:
            keep = np.argsort(magnitudes)[::-1][:budget]
            nDropped = int(candidates.size - keep.size)
            order = np.sort(keep)
            candidates = candidates[order]
            shoulderWidths = shoulderWidths[order]

    wanted: List[float] = []
    for edge, width in zip(candidates, shoulderWidths):
        wanted.append(float(edge))
        below = float(edge) - shoulder * float(width)
        if below > lo:
            wanted.append(below)

    nInserted = 0
    if wanted:
        new = np.unique(np.asarray(wanted, dtype=float))
        new = new[(new > lo) & (new < hi)]
        if new.size:
            nearest = xs[np.clip(np.searchsorted(xs, new), 1, xs.size - 1)]
            under = xs[np.clip(np.searchsorted(xs, new) - 1, 0, xs.size - 1)]
            duplicate = (np.isclose(new, nearest, rtol=rtol, atol=0.0)
                         | np.isclose(new, under, rtol=rtol, atol=0.0))
            new = new[~duplicate]
        if new.size:
            values = np.interp(new, xs, ys)
            xs = np.concatenate([xs, new])
            ys = np.concatenate([ys, values])
            order = np.argsort(xs, kind="stable")
            xs, ys = xs[order], ys[order]
            nInserted = int(new.size)

    # Group of each node: side='right' puts a node sitting exactly on a
    # boundary into the group *above* it, and the shoulder just below into the
    # group beneath -- which is the assignment the step construction needs.
    group = np.clip(np.searchsorted(boundaries, xs, side="right") - 1,
                    0, ratios.size - 1)
    inside = (xs >= boundaries[0]) & (xs <= boundaries[-1])
    scale = np.where(inside, ratios[group], 1.0)
    return xs, ys * scale, nInserted, nDropped


def _copyOuterContainer(form):
    """*form* with a fresh child list, so replacing a child leaves the original.

    Not a ``deepcopy``: the children themselves are shared until one is
    replaced, and ``replaceTable`` replaces a *slot* rather than mutating the
    function in it. So the evaluated node keeps every table it had, and the
    realisation carries new objects only where the perturbation actually
    changed something.
    """
    import copy as _copy

    from .functions.higher import Regions2d, XYs2d

    if isinstance(form, XYs2d):
        clone = _copy.copy(form)
        clone.function1ds = list(form.function1ds)
        return clone
    if isinstance(form, Regions2d):
        clone = _copy.copy(form)
        clone.function2ds = [_copyOuterContainer(region)
                             for region in form.function2ds]
        return clone
    raise TypeError(
        f"an energy distribution is an XYs2d or a regions2d, not a "
        f"{type(form).__name__}"
    )


def applySpectrumFactors(energyForm, bands, boundariesByBand, ratiosFor, *,
                         maxOutgoingPoints=None,
                         shoulder: float = SPECTRUM_STEP_SHOULDER,
                         outerShoulder: float = SPECTRUM_OUTER_SHOULDER,
                         rtol: float = SPECTRUM_ENERGY_RTOL):
    """Perturb an energy distribution from group-probability factors.

    Parameters
    ----------
    energyForm
        The ``XYs2d`` (or ``regions2d``) of tabulated ``chi(E'|E)`` an
        :class:`~kika.nuclear_data.model.distributions.Uncorrelated` carries.
        **Not mutated**: the perturbed node comes back and the evaluated one is
        left where it was.
    bands
        ``{key: (E1, E2)}`` -- the outer-coordinate range each factor block is
        stated over. For MF35 these are the incident-energy bands, one per
        covariance section, and the key is the band the file states rather than
        a position, so a request for some bands and not others still lines up.
        A bare sequence is accepted and keyed by position.
    boundariesByBand
        ``key -> the outgoing-energy group boundaries`` that band's factors are
        stated on, keyed as *bands* is. One longer than the ratios.
    ratiosFor
        ``callable(band, p0, covered) -> (ratios, info)``. Given a node's own
        group probabilities ``p0`` and the mass ``covered`` by the band's grid,
        it returns one ratio per group and whatever it wants recorded. **This is
        where a sample's physics lives** -- forming ratios from absolute deltas,
        freezing groups with no probability to scale, and projecting onto
        ``{r >= 0, sum P0 r = sum P0}``. It is a callback and not an argument
        because ``p0`` is *per outer node*: every node inside a band shares the
        deltas and has its own probabilities, so the ratios differ node by node
        and only this function knows them.
    maxOutgoingPoints
        Cap on the size of one table. Steps are dropped smallest-first and the
        count is reported; ``None`` means no cap.

    Returns
    -------
    (perturbed, diagnostics)
        A node of the same kind, and per-node diagnostics plus the summary
        :func:`summariseSpectrumNodes` builds.

    **What it does, per outer node, and why in that order.**

    1. integrate the table over the band's groups -> ``P0``;
    2. measure ``int chi`` and ``sum P0`` and record both, before anything
       changes -- the mass outside the band's grid is real (an MF5 table may
       start at ``E'=0`` where MF35 starts at 1e-5, and a top group may run past
       the end of the table), so **the normalisation target is ``sum P0`` as
       measured, not 1**;
    3. ask *ratiosFor* what the factors are;
    4. insert the boundaries the factor steps at, plus a shoulder just below
       each, and scale -- :func:`_refineAndScale`;
    5. rescale the whole table so ``int chi`` is exactly what it was;
    6. **check that step 4 delivered ``P_j = rescale * r_j * P0_j``.** Nothing
       outside this function can see that it did, and step 5 would hide a
       failure of it in a scalar, so it is measured here and reported per node.

    The residual that legitimately remains is the shoulder ramp -- the sliver
    between ``g_j(1-s)`` and ``g_j`` carries a blend of two factors -- and it is
    bounded by about ``s/2 * max|dr|``.
    """
    bands = _asBandMap(bands)
    form = _copyOuterContainer(energyForm)
    insertedOuter = _insertOuterShoulders(form, bands, shoulder=outerShoulder,
                                          rtol=rtol)

    perNode: List[dict] = []
    for k, value in enumerate(list(form.outerDomainValues)):
        band = _bandOf(float(value), bands)
        if band is None or band not in boundariesByBand:
            continue

        boundaries = np.asarray(boundariesByBand[band], dtype=float)
        p0 = form.groupIntegrals(k, boundaries)
        total = form.normalisation(k)
        covered = float(p0.sum())

        ratios, info = ratiosFor(band, p0, covered)
        ratios = np.asarray(ratios, dtype=float)
        if ratios.size != p0.size:
            raise ValueError(
                f"band {band}: {ratios.size} ratio(s) for {p0.size} group(s) of "
                f"the boundaries this band is stated on"
            )

        xs, ys = form.table(k)
        xNew, yNew, nInserted, nDropped = _refineAndScale(
            xs, ys, boundaries, ratios, maxPoints=maxOutgoingPoints,
            shoulder=shoulder, rtol=rtol)
        form.replaceTable(k, xNew, yNew)

        realised = form.normalisation(k)
        rescale = (total / realised) if realised != 0.0 else 1.0
        if rescale != 1.0:
            form.replaceTable(k, xNew, yNew * rescale)

        wanted = rescale * ratios * p0
        realisedGroups = form.groupIntegrals(k, boundaries)
        scored = wanted > 0.0
        groupError = (
            float(np.max(np.abs(realisedGroups[scored] / wanted[scored] - 1.0)))
            if np.any(scored) else 0.0
        )
        # The same discrepancy as a fraction of the whole spectrum. **This is
        # the one to gate on.** The relative form above is unbounded wherever a
        # group's ratio was driven towards zero -- its denominator goes to
        # nothing while the neighbouring shoulder ramp does not -- so it reports
        # percent-level numbers on groups holding 1e-5 of the spectrum and says
        # nothing about whether the perturbation is right.
        groupMassError = (
            float(np.max(np.abs(realisedGroups - wanted)) / covered)
            if covered else 0.0
        )

        perNode.append({
            "node": k,
            "outer_value": float(value),
            "band": band,
            "integral_before": float(total),
            "covered_before": covered,
            "mass_outside_bands": float(total - covered),
            "frac_mass_outside_bands": (float((total - covered) / total)
                                        if total else 0.0),
            "renormalisation_scalar": float(rescale),
            "renormalisation_error": float(abs(rescale - 1.0)),
            "n_outgoing_inserted": nInserted,
            "n_steps_dropped": nDropped,
            "n_outgoing_points": int(xNew.size),
            "max_group_ratio_error": groupError,
            "max_group_mass_error": groupMassError,
            **dict(info or {}),
        })

    diagnostics = summariseSpectrumNodes(perNode)
    diagnostics["n_outer_inserted"] = insertedOuter
    diagnostics["per_node"] = perNode
    return form, diagnostics


def summariseSpectrumNodes(perNode) -> dict:
    """The worst of each per-node quantity, over every node that was perturbed.

    A summary rather than a mean on purpose: a normalisation that is right on
    thirty nodes and wrong on one is a broken file, and an average says it is
    fine.
    """
    def worst(field: str) -> float:
        values = [abs(entry[field]) for entry in perNode if field in entry]
        return float(max(values)) if values else 0.0

    return {
        "n_nodes": len(perNode),
        "max_renormalisation_error": worst("renormalisation_error"),
        "max_group_ratio_error": worst("max_group_ratio_error"),
        "max_group_mass_error": worst("max_group_mass_error"),
        "max_frac_mass_outside_bands": worst("frac_mass_outside_bands"),
        "total_steps_dropped": int(sum(entry.get("n_steps_dropped", 0)
                                       for entry in perNode)),
        "total_outgoing_inserted": int(sum(entry.get("n_outgoing_inserted", 0)
                                           for entry in perNode)),
    }
