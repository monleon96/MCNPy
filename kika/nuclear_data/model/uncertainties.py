"""GNDS-2.1 §7: uncertainty attached to a functional.

§7 lets a function carry an ``uncertainty`` node holding either a standard
uncertainty (itself a function) or a covariance, or a ``listOfCovariances``
pointing at entries in a separate ``covarianceSuite`` (§25).

Only the shape is built here. The real covariance work is phase 3b's
``CovarianceSuite`` and phase 3c's MF33/MF34 adapters; what P4 needs is a slot
for ``XYs1d.uncertainty`` to point at, so the functional containers are
structurally complete rather than quietly missing a child the spec allows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["Uncertainty", "Covariance", "ListOfCovariances"]


@dataclass
class Covariance:
    """A pointer to a covariance living in a ``covarianceSuite`` (§25).

    ``href`` is an xPath into another document, which is how §25 links a
    reaction's data to its covariance. Resolving it is phase 5's job.
    """

    href: str
    label: Optional[str] = None


@dataclass
class ListOfCovariances:
    """§7. Several covariances for one functional."""

    covariances: List[Covariance] = field(default_factory=list)


@dataclass
class Uncertainty:
    """§7. The ``uncertainty`` child a functional may carry.

    Exactly one of the three forms is expected to be set. Nothing enforces that
    yet because nothing constructs one yet; the constraint belongs with the
    decoder that first fills it, in phase 3c.
    """

    standard: Optional[object] = None
    covariance: Optional[Covariance] = None
    listOfCovariances: Optional[ListOfCovariances] = None
