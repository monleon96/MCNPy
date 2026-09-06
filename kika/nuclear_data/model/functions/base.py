"""The abstract ``functional`` node that every §6 container specialises.

GNDS calls the abstract node ``functional`` and gives every concrete form the
same three optional attributes: ``label`` (which style this form belongs to,
when the form sits inside a multi-form container), ``outerDomainValue`` (the
next higher dimension's coordinate, when the function sits inside a
higher-dimensional one) and ``index`` (position, when it sits inside a
``regions1d``).

**Why evaluation delegates.** Every concrete subclass resolves ``evaluate`` to
``kika.processing.interpolation.interpolate_1d`` rather than reimplementing the
interpolation laws. That is not laziness: phase 3's acceptance is that the model
reproduces the flat path *bit for bit*, and the only way to guarantee that of
floating-point code is to run the same code. A second implementation would
agree to about 1e-16 and disagree somewhere, and finding where is not work worth
doing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from ..axes import Axes

__all__ = ["Function1d", "OutOfRange"]

#: What to do outside the domain. The spellings are kika's existing ones, kept
#: so the model and the flat path can be handed the same argument.
OutOfRange = str


def _interpolate_1d():
    """Import the interpolator only when an evaluation actually happens.

    Deferred deliberately. ``kika.processing``'s package ``__init__`` imports
    ``reconstruct``, which imports ``kika.nuclear_data.cross_section`` — so a
    module-level import here would close a cycle the moment anything makes
    ``kika/nuclear_data/__init__.py`` import this package, which is exactly what
    phase 3d does. Phase 2 was blocked for the same reason in the other
    direction; once is enough.
    """
    from kika.processing.interpolation import interpolate_1d

    return interpolate_1d


class Function1d(ABC):
    """Abstract base for the §6 one-dimensional functional containers."""

    axes: Optional[Axes]
    label: Optional[str]
    outerDomainValue: Optional[float]
    index: Optional[int]

    @property
    @abstractmethod
    def domainMin(self) -> float:
        """Lowest abscissa the function is defined at."""

    @property
    @abstractmethod
    def domainMax(self) -> float:
        """Highest abscissa the function is defined at."""

    @abstractmethod
    def evaluate(
        self, x: Union[float, ArrayLike], outOfRange: OutOfRange = "zero"
    ) -> Union[float, np.ndarray]:
        """The function's value at ``x``."""

    # ------------------------------------------------------------------
    # Integrals
    # ------------------------------------------------------------------
    #
    # Here rather than on ``XYs1d`` and ``Regions1d`` separately because the
    # implementation is the same call for both -- ``toEndfRegions`` flattens
    # one region and many into the same triple -- and because a node that is
    # *not* a table should answer the question with a refusal that names it,
    # not with an ``AttributeError`` from somewhere further in. See
    # :mod:`~kika.nuclear_data.model.functions.integration`.

    def integrate(self, domainMin: Optional[float] = None,
                  domainMax: Optional[float] = None) -> float:
        """``int f(x) dx`` over the given limits, exactly on the stated law.

        Both limits default to the function's own domain, so ``f.integrate()``
        is the whole integral -- which for a normalised spectrum is the
        quantity a perturbation has to give back unchanged.
        """
        from .integration import integrateFunction1d

        return integrateFunction1d(self, domainMin, domainMax,
                                   what=type(self).__name__)

    def groupIntegrals(self, boundaries: ArrayLike) -> np.ndarray:
        """``P_j = int_{g_j}^{g_j+1} f`` for every group of *boundaries*.

        The quantity an MF35 covariance is stated over. Exact on the function's
        own panels rather than on a refinement of them; see
        :func:`~kika.nuclear_data.model.functions.integration.groupIntegralsOf`.
        """
        from .integration import groupIntegralsOf

        return groupIntegralsOf(self, boundaries, what=type(self).__name__)

    @property
    def domainUnit(self) -> str:
        """Unit of the independent axis, or ``''`` when no axes are attached."""
        if self.axes is None:
            return ""
        return self.axes.byIndex(1).unit

    @property
    def rangeUnit(self) -> str:
        """Unit of the dependent axis (index 0), or ``''``."""
        if self.axes is None:
            return ""
        return self.axes.dependent.unit
