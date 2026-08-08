"""GNDS-2.1 §25: the ``covarianceSuite``, a separate root from ``reactionSuite``.

The structure §25.2.2 defines is ``covarianceSection`` carrying ``rowData`` and
optional ``columnData``, each with an optional ``ENDF_MFMT``, an ``href`` xPath
into a ``reactionSuite``, an optional ``dimension``, and ``slices``/``slice``
(§25.2.5-6). A slice takes a ``dimension`` plus either a ``domainValue`` or a
``domainMin``/``domainMax`` pair, plus a ``domainUnit``.

**An MF34 Legendre-order covariance is a slice whose ``domainValue`` is the
Legendre order.** That is the single mapping most likely to be modelled wrongly
in phase 3c/P7, so it is stated here and has a constructor of its own:
:meth:`DataLink.forLegendreOrder`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

__all__ = ["Slice", "Slices", "DataLink", "CovarianceMatrix", "Mixed", "Sum",
           "CovarianceSection", "CovarianceSuite"]


@dataclass
class Slice:
    """§25.2.6. One restriction of a linked dataset along one dimension."""

    dimension: int
    domainValue: Optional[float] = None
    domainMin: Optional[float] = None
    domainMax: Optional[float] = None
    domainUnit: str = ""

    def __post_init__(self) -> None:
        hasPoint = self.domainValue is not None
        hasRange = self.domainMin is not None or self.domainMax is not None
        if hasPoint and hasRange:
            raise ValueError(
                "a slice takes either domainValue or domainMin/domainMax, not both"
            )
        if not hasPoint and not hasRange:
            raise ValueError("a slice needs domainValue or domainMin/domainMax")


@dataclass
class Slices:
    """§25.2.5. The slices applied to one ``rowData``/``columnData``."""

    slices: List[Slice] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.slices)

    def __iter__(self) -> Iterator[Slice]:
        return iter(self.slices)


@dataclass
class DataLink:
    """§25.2.3-4. ``rowData`` or ``columnData``: what this matrix is *about*."""

    href: str
    ENDF_MFMT: Optional[str] = None
    dimension: Optional[int] = None
    slices: Slices = field(default_factory=Slices)

    @classmethod
    def forLegendreOrder(cls, href: str, order: int, ENDF_MFMT: Optional[str] = None,
                         dimension: int = 1) -> "DataLink":
        """An MF34-equivalent link: the Legendre order as a slice ``domainValue``.

        MF34 gives covariances between Legendre coefficients ``a_l`` of an
        angular distribution. In GNDS the distribution is a function of order,
        and a covariance about one order is that function *sliced* at that
        order — not a separate quantity. Modelling it as a separate quantity is
        the mistake this constructor exists to prevent.
        """
        return cls(
            href=href,
            ENDF_MFMT=ENDF_MFMT,
            slices=Slices([Slice(dimension=dimension, domainValue=float(order))]),
        )

    @property
    def legendreOrder(self) -> Optional[int]:
        """The Legendre order this link is sliced at, when it is."""
        for entry in self.slices:
            if entry.domainValue is not None and not entry.domainUnit:
                return int(entry.domainValue)
        return None


@dataclass
class CovarianceMatrix:
    """§25. An explicit matrix, absolute or relative, on a stated grid."""

    matrix: np.ndarray
    rowGrid: Optional[np.ndarray] = None
    columnGrid: Optional[np.ndarray] = None
    isRelative: bool = False
    label: Optional[str] = None
    productFrame: Optional[str] = None

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim != 2:
            raise ValueError(f"a covariance matrix is 2-D, got shape {self.matrix.shape}")
        if self.rowGrid is not None:
            self.rowGrid = np.asarray(self.rowGrid, dtype=float)
            if self.rowGrid.size != self.matrix.shape[0] + 1:
                raise ValueError(
                    f"rowGrid has {self.rowGrid.size} boundaries for "
                    f"{self.matrix.shape[0]} rows; expected one more than the rows"
                )

    @property
    def isSquare(self) -> bool:
        return self.matrix.shape[0] == self.matrix.shape[1]

    @property
    def isSymmetric(self) -> bool:
        return self.isSquare and bool(np.array_equal(self.matrix, self.matrix.T))


@dataclass
class Mixed:
    """§25. Several covariance forms that add up to one covariance."""

    components: List[object] = field(default_factory=list)


@dataclass
class Sum:
    """§25. A covariance defined as a weighted sum of others, by reference."""

    references: List[str] = field(default_factory=list)
    coefficients: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.coefficients and len(self.coefficients) != len(self.references):
            raise ValueError(
                f"{len(self.coefficients)} coefficients for {len(self.references)} references"
            )


@dataclass
class CovarianceSection:
    """§25.2.2. One covariance: what it is about, and the matrix itself."""

    label: str
    rowData: Optional[DataLink] = None
    columnData: Optional[DataLink] = None
    form: Optional[object] = None   # CovarianceMatrix | Mixed | Sum
    #: The ENDF section header the decoder read, kept so the section can be
    #: written back rather than approximated. ZA, AWR, MAT and — for MF34 —
    #: LTT have no GNDS counterpart: §25.2.2 identifies a covariance by its
    #: ``href`` into a ``reactionSuite``, not by a material header. Without
    #: this the encoder would have to default AWR and MAT, which is the third
    #: time in this package that what the *read* path discarded turned out to
    #: be the whole cost of the write path (see ``docs/library-gaps.md`` D2
    #: and M2).
    provenance: Optional[object] = None

    @property
    def isCrossTerm(self) -> bool:
        """A block between two *different* quantities."""
        return self.columnData is not None


@dataclass
class CovarianceSuite:
    """§25.1.1. A root node in its own right, linked to a ``reactionSuite``."""

    evaluation: Optional[str] = None
    projectile: Optional[str] = None
    target: Optional[str] = None
    interaction: str = "nuclear"
    version: Optional[str] = None
    externalFiles: List[object] = field(default_factory=list)
    styles: Optional[object] = None
    covarianceSections: List[CovarianceSection] = field(default_factory=list)
    parameterCovariances: List[object] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.covarianceSections)

    def __iter__(self) -> Iterator[CovarianceSection]:
        return iter(self.covarianceSections)

    def byLabel(self, label: str) -> CovarianceSection:
        for section in self.covarianceSections:
            if section.label == label:
                return section
        raise KeyError(f"no covariance section labelled {label!r}")
