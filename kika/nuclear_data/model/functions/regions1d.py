"""GNDS-2.1 §6.4.1 ``regions1d``: one function per interpolation region.

**This is the fix for the "dominant scheme" problem.** The flat
``CrossSection`` keeps a single ``interpolation`` string chosen by
``_dominant_interpolation`` and hides the real ``(NBT, INT)`` regions in
``metadata["interpolation_regions"]``. Anything that reads the attribute rather
than the metadata silently interpolates part of the table under the wrong law.
``regions1d`` has no dominant scheme: every region is its own function with its
own rule, which is what the data actually is.

**Bit-exactness.** :meth:`evaluate` rebuilds the ENDF ``(NBT, INT)`` pairs and
makes a single call to ``kika.processing.interpolation.interpolate_1d`` over the
concatenated grid — the same call, on the same arrays, that the flat path makes.
It is not a reimplementation, so there is nothing for the two to disagree about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from ..axes import Axes
from ..enums import ENDF_INT_TO_INTERPOLATION, INTERPOLATION_TO_ENDF_INT, Interpolation
from .base import Function1d, OutOfRange, _interpolate_1d
from .xys1d import XYs1d

__all__ = ["Regions1d"]


@dataclass
class Regions1d(Function1d):
    """A piecewise function: a list of :class:`XYs1d`, one per region."""

    function1ds: List[XYs1d] = field(default_factory=list)
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None

    def __len__(self) -> int:
        return len(self.function1ds)

    @property
    def domainMin(self) -> float:
        if not self.function1ds:
            return float("nan")
        return self.function1ds[0].domainMin

    @property
    def domainMax(self) -> float:
        if not self.function1ds:
            return float("nan")
        return self.function1ds[-1].domainMax

    # ------------------------------------------------------------------
    # The ENDF shape, in and out
    # ------------------------------------------------------------------

    @classmethod
    def fromEndfRegions(
        cls,
        xs: ArrayLike,
        ys: ArrayLike,
        nbtIntPairs: Sequence[Tuple[int, int]],
        axes: Optional[Axes] = None,
        label: Optional[str] = None,
    ) -> "Regions1d":
        """Build from ENDF's one-grid-plus-``(NBT, INT)``-pairs layout.

        ``NBT`` is **cumulative** — region *i* ends at point ``NBT[i]``, counting
        from the start of the whole table, one-based. Reading it as a per-region
        count is the bug that made ``_dominant_interpolation`` always pick the
        last region instead of the widest; it is written out here so the next
        reader does not have to rediscover it.

        Regions share their boundary point: ENDF region *i* spans points
        ``[NBT[i-1] - 1, NBT[i] - 1]`` inclusive, so consecutive regions overlap
        in one abscissa. That overlap is preserved rather than trimmed, because
        it is what makes each region's own interpolation well defined at its
        lower edge.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        regions: List[XYs1d] = []
        previous = 0
        for order, (nbt, intCode) in enumerate(nbtIntPairs):
            stop = int(nbt)
            if stop <= previous:
                continue
            start = max(previous - 1, 0) if previous else 0
            regions.append(
                XYs1d(
                    xs=xs[start:stop],
                    ys=ys[start:stop],
                    interpolation=ENDF_INT_TO_INTERPOLATION[int(intCode)],
                    axes=axes,
                    index=order,
                )
            )
            previous = stop
        return cls(function1ds=regions, axes=axes, label=label)

    def toEndfRegions(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """The inverse of :meth:`fromEndfRegions`: one grid and ``(NBT, INT)`` pairs."""
        if not self.function1ds:
            return np.asarray([]), np.asarray([]), []
        xs = [self.function1ds[0].xs]
        ys = [self.function1ds[0].ys]
        pairs = [(len(self.function1ds[0]), self.function1ds[0].endfInterpolationCode)]
        for region in self.function1ds[1:]:
            # Drop the shared boundary point that fromEndfRegions duplicated.
            xs.append(region.xs[1:])
            ys.append(region.ys[1:])
            pairs.append((pairs[-1][0] + len(region) - 1, region.endfInterpolationCode))
        return np.concatenate(xs), np.concatenate(ys), pairs

    # ------------------------------------------------------------------

    def evaluate(
        self, x: Union[float, ArrayLike], outOfRange: OutOfRange = "zero"
    ) -> Union[float, np.ndarray]:
        """One call to the shared interpolator over the concatenated grid.

        Deliberately *not* "find the region, then evaluate it": that would be a
        second implementation of region look-up, and the phase 3 gate is that
        this reproduces the flat path exactly.
        """
        xs, ys, pairs = self.toEndfRegions()
        return _interpolate_1d()(xs, ys, pairs, x, outOfRange)

    def __repr__(self) -> str:
        rules = ", ".join(f.interpolation.value for f in self.function1ds)
        return f"Regions1d(n_regions={len(self)}, [{rules}])"
