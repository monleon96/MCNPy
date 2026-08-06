"""The remaining §6 one-dimensional forms: ``constant1d``, ``polynomial1d``,
``Ys1d``, ``Legendre`` and ``gridded1d``.

Small enough to share a module. Each is a real implementation rather than a
stub — they are cheap, and phase 3c's ENDF decoders need ``Legendre`` (MF4) and
``gridded1d`` (multigroup) as soon as they are written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from ..axes import Axes, Grid
from ..enums import Interpolation
from .base import Function1d, OutOfRange, _interpolate_1d

__all__ = ["Constant1d", "Polynomial1d", "Ys1d", "Legendre", "Gridded1d"]


@dataclass
class Constant1d(Function1d):
    """§6. One value across a stated domain."""

    constant: float
    domainMin_: float
    domainMax_: float
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None

    @property
    def domainMin(self) -> float:
        return self.domainMin_

    @property
    def domainMax(self) -> float:
        return self.domainMax_

    def evaluate(self, x, outOfRange: OutOfRange = "zero"):
        xq = np.asarray(x, dtype=float)
        inside = (xq >= self.domainMin_) & (xq <= self.domainMax_)
        if outOfRange == "hold":
            out = np.full(xq.shape, self.constant, dtype=float)
        else:
            out = np.where(inside, self.constant, 0.0)
        return float(out) if np.ndim(x) == 0 else out


@dataclass
class Polynomial1d(Function1d):
    """§6. Coefficients in ascending order: ``c[0] + c[1]*x + c[2]*x**2 + ...``."""

    coefficients: np.ndarray
    domainMin_: float
    domainMax_: float
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=float)

    @property
    def domainMin(self) -> float:
        return self.domainMin_

    @property
    def domainMax(self) -> float:
        return self.domainMax_

    def evaluate(self, x, outOfRange: OutOfRange = "zero"):
        # polyval wants descending order; the spec stores ascending.
        return np.polyval(self.coefficients[::-1], np.asarray(x, dtype=float))


@dataclass
class Ys1d(Function1d):
    """§6. Ordinates on a grid that lives elsewhere, referenced by the axes.

    The whole point of the form is that the abscissae are *not* repeated per
    function — a multigroup library stores one grid and many ``Ys1d``. So the
    grid is looked up on the axes, and evaluation without one is an error rather
    than a guess.
    """

    ys: np.ndarray
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None
    interpolation: Interpolation = Interpolation.linlin

    def __post_init__(self) -> None:
        self.ys = np.asarray(self.ys, dtype=float)
        self.interpolation = Interpolation(self.interpolation)

    def _grid(self) -> np.ndarray:
        if self.axes is None:
            raise ValueError("Ys1d has no axes, so its grid cannot be resolved")
        axis = self.axes.byIndex(1)
        if not isinstance(axis, Grid) or axis.values is None:
            raise ValueError(
                "Ys1d's independent axis is a plain axis, not a grid with values; "
                "there is nothing to evaluate against"
            )
        return np.asarray(axis.values, dtype=float)

    @property
    def domainMin(self) -> float:
        return float(self._grid()[0])

    @property
    def domainMax(self) -> float:
        return float(self._grid()[-1])

    def evaluate(self, x, outOfRange: OutOfRange = "zero"):
        from ..enums import INTERPOLATION_TO_ENDF_INT

        grid = self._grid()
        return _interpolate_1d()(
            grid, self.ys, [(grid.size, INTERPOLATION_TO_ENDF_INT[self.interpolation])],
            x, outOfRange,
        )


@dataclass
class Legendre(Function1d):
    """§6. A Legendre expansion in ``mu`` over ``[-1, 1]``.

    ``coefficients[l]`` is ``a_l``, with ``a_0`` conventionally 1 for a
    normalised angular distribution. The sum evaluated is
    ``sum_l (2l+1)/2 * a_l * P_l(mu)`` — the normalisation ENDF MF4 uses, which
    is what kika's flat ``AngularDistribution`` also assumes.
    """

    coefficients: np.ndarray
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=float)

    @property
    def maxOrder(self) -> int:
        return int(self.coefficients.size) - 1

    @property
    def domainMin(self) -> float:
        return -1.0

    @property
    def domainMax(self) -> float:
        return 1.0

    def evaluate(self, x, outOfRange: OutOfRange = "zero"):
        from numpy.polynomial import legendre as _legendre

        mu = np.asarray(x, dtype=float)
        scaled = self.coefficients * (2.0 * np.arange(self.coefficients.size) + 1.0) / 2.0
        return _legendre.legval(mu, scaled)


@dataclass
class Gridded1d(Function1d):
    """§6. Values on an explicit grid of boundaries — the multigroup form."""

    values: np.ndarray
    grid: Grid
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        boundaries = np.asarray(self.grid.values, dtype=float)
        if boundaries.size != self.values.size + 1:
            raise ValueError(
                f"a gridded1d needs one more boundary than values; got "
                f"{boundaries.size} boundaries for {self.values.size} values"
            )

    @property
    def domainMin(self) -> float:
        return float(np.asarray(self.grid.values, dtype=float)[0])

    @property
    def domainMax(self) -> float:
        return float(np.asarray(self.grid.values, dtype=float)[-1])

    def evaluate(self, x, outOfRange: OutOfRange = "zero"):
        """Piecewise-constant lookup: the value of the group containing ``x``."""
        boundaries = np.asarray(self.grid.values, dtype=float)
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros(xq.shape, dtype=float)
        inside = (xq >= boundaries[0]) & (xq <= boundaries[-1])
        idx = np.clip(np.searchsorted(boundaries, xq, side="right") - 1, 0, self.values.size - 1)
        out[inside] = self.values[idx[inside]]
        if outOfRange == "hold":
            out[xq < boundaries[0]] = self.values[0]
            out[xq > boundaries[-1]] = self.values[-1]
        return float(out[0]) if np.ndim(x) == 0 else out
