"""GNDS-2.1 §6.1.1 ``XYs1d``: a tabulated y(x) with one interpolation rule."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from ..axes import Axes
from ..enums import INTERPOLATION_TO_ENDF_INT, Interpolation
from .base import Function1d, OutOfRange, _interpolate_1d

__all__ = ["XYs1d"]


@dataclass
class XYs1d(Function1d):
    """A single-valued function stored as tabulated ``(xi, yi)`` pairs.

    §6.1.1 stores the pairs interleaved in one ``values`` node
    (``x0 y0 x1 y1 ...``). They are kept as two arrays here because every
    consumer wants them that way and the interleaving is a serialisation
    detail; :meth:`interleaved` produces the spec's layout when a writer needs
    it.

    One interpolation rule applies to the whole table. A table that needs more
    than one is a :class:`~kika.nuclear_data.model.functions.regions1d.Regions1d`
    — which is the honest replacement for the flat classes' "dominant scheme,
    with the real regions hidden in ``metadata``".
    """

    xs: np.ndarray
    ys: np.ndarray
    interpolation: Interpolation = Interpolation.linlin
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None

    def __post_init__(self) -> None:
        self.xs = np.asarray(self.xs, dtype=float)
        self.ys = np.asarray(self.ys, dtype=float)
        if self.xs.shape != self.ys.shape:
            raise ValueError(
                f"xs and ys must have the same shape, got {self.xs.shape} and {self.ys.shape}"
            )
        if self.xs.ndim != 1:
            raise ValueError(f"XYs1d is one-dimensional, got shape {self.xs.shape}")
        self.interpolation = Interpolation(self.interpolation)

    def __len__(self) -> int:
        return int(self.xs.size)

    @property
    def domainMin(self) -> float:
        return float(self.xs[0]) if self.xs.size else float("nan")

    @property
    def domainMax(self) -> float:
        return float(self.xs[-1]) if self.xs.size else float("nan")

    @property
    def endfInterpolationCode(self) -> int:
        """The ENDF-6 ``INT`` this rule corresponds to (§3.4.4 adopted it wholesale)."""
        return INTERPOLATION_TO_ENDF_INT[self.interpolation]

    def interleaved(self) -> np.ndarray:
        """``[x0, y0, x1, y1, ...]`` — the layout §6.1.1 serialises."""
        out = np.empty(self.xs.size * 2, dtype=float)
        out[0::2] = self.xs
        out[1::2] = self.ys
        return out

    def evaluate(
        self, x: Union[float, ArrayLike], outOfRange: OutOfRange = "zero"
    ) -> Union[float, np.ndarray]:
        return _interpolate_1d()(
            self.xs, self.ys, [(len(self), self.endfInterpolationCode)], x, outOfRange
        )

    def __repr__(self) -> str:
        return (
            f"XYs1d(n={len(self)}, interpolation={self.interpolation.value!r}, "
            f"domain=[{self.domainMin:.4g}, {self.domainMax:.4g}])"
            if len(self)
            else "XYs1d(empty)"
        )
