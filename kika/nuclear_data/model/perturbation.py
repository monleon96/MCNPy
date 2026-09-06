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

from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .functions.regions1d import Regions1d
from .functions.xys1d import XYs1d

__all__ = ["applyFactors", "refineAtBinEdges"]

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
