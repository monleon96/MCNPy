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
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .conversion import ConversionReport

__all__ = ["Slice", "Slices", "DataLink", "CovarianceMatrix", "Mixed",
           "Summand", "Sum", "ShortRangeSelfScalingVariance",
           "CovarianceSection", "CovarianceSuite",
           "ParameterLink", "ParameterCovarianceMatrix", "ParameterCovariance",
           "AverageParameterCovariance"]


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

    @classmethod
    def forIncidentEnergyBand(cls, href: str, domainMin: float, domainMax: float,
                              ENDF_MFMT: Optional[str] = None,
                              dimension: int = 2,
                              domainUnit: str = "eV") -> "DataLink":
        """An MF35-equivalent link: the incident-energy band as a slice *range*.

        MF35 gives the covariance of an outgoing-energy spectrum over a range
        of incident energies. The same reasoning as
        :meth:`forLegendreOrder` applies, one step further: an energy
        distribution is one function of incident and outgoing energy, so a
        covariance about the band 5-7 MeV is that function **sliced** to that
        band — not a covariance of eight separate quantities that happen to
        share a name.

        The difference from the Legendre case is that a band is a range rather
        than a point, so this fills ``domainMin``/``domainMax`` where that one
        fills ``domainValue`` (§25.2.6 admits either, and exactly one).
        ``dimension=2`` is the incident-energy axis of ``chi(E'|E)``, matching
        the convention :data:`LEGENDRE_DIMENSION` follows for ``P(mu|E)``.
        """
        return cls(
            href=href,
            ENDF_MFMT=ENDF_MFMT,
            slices=Slices([Slice(dimension=dimension,
                                 domainMin=float(domainMin),
                                 domainMax=float(domainMax),
                                 domainUnit=domainUnit)]),
        )

    @property
    def incidentEnergyBand(self) -> Optional[tuple]:
        """The ``(domainMin, domainMax)`` this link is sliced to, when it is."""
        for entry in self.slices:
            if entry.domainMin is not None or entry.domainMax is not None:
                return (entry.domainMin, entry.domainMax)
        return None

    @property
    def legendreOrder(self) -> Optional[int]:
        """The Legendre order this link is sliced at, when it is."""
        for entry in self.slices:
            if entry.domainValue is not None and not entry.domainUnit:
                return int(entry.domainValue)
        return None

    # -- ENDF_MFMT, which is written two ways in this codebase ------------

    @property
    def ENDF_MF(self) -> Optional[int]:
        """The MF number, whichever separator the link was built with."""
        parts = self._mfmtParts()
        return None if parts is None else parts[0]

    @property
    def ENDF_MT(self) -> Optional[int]:
        """The MT number, whichever separator the link was built with.

        **Two spellings are in circulation and this is the accessor that does
        not care which.** §25.2.3 (p. 363) defines ``ENDF_MFMT`` as *"the ENDF
        MF and MT numbers, stored as a comma-separated"* pair, and every file in
        ENDF/B-VIII.1-GNDS writes ``"33,2"``. kika's **ENDF** adapter writes
        ``"33/2"`` instead, and nine sites across ``kika/cov``,
        ``kika/sampling`` and ``kika/endf`` parse it with ``split("/")[1]``.

        That divergence is *not* fixed here, deliberately. Two of those modules
        are the deployed thesis pipeline, mid-migration, and changing the
        spelling under them is a separate, gated increment — one that belongs
        with the GNDS **writer**, which is the first thing that would emit the
        wrong spelling into a published file. Until then this property is the
        safe way to ask, and a GNDS-decoded suite can be handed to code that was
        written for the ENDF one.
        """
        parts = self._mfmtParts()
        return None if parts is None else parts[1]

    def _mfmtParts(self) -> Optional[Tuple[int, int]]:
        if not self.ENDF_MFMT:
            return None
        text = str(self.ENDF_MFMT).replace("/", ",")
        try:
            mf, mt = (int(part) for part in text.split(",", 1))
        except ValueError:
            return None
        return mf, mt


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
    """§25. Several covariance forms that add up to one covariance.

    **The label is not decoration.** ``covariances.xsd:135-142`` makes it
    ``use="required"`` on ``mixed``, exactly as on ``covarianceMatrix`` and
    ``sum``, and every ``<mixed>`` in ENDF/B-VIII.1-GNDS carries one. It was
    missing here while the reader dropped it and the writer asked for it
    anyway, so writing any suite holding a ``mixed`` raised ``AttributeError``
    (``kika/gnds/encode.py``) and no test saw it, because the one round-trip
    fixture has none.
    """

    components: List[object] = field(default_factory=list)
    label: Optional[str] = None


@dataclass
class Summand:
    """§25. One term of a :class:`Sum`: what to add, and how much of it.

    ``ENDF_MFMT`` travels with the link rather than being looked up through it,
    because that is where the file puts it and because a summand may point at a
    section this suite does not contain.
    """

    href: str
    coefficient: float = 1.0
    ENDF_MFMT: Optional[str] = None


@dataclass
class Sum:
    """§25. A covariance defined as a weighted sum of others, by reference.

    ENDF's NC-type sub-subsections: "the covariance of MT4 is that of MT1 minus
    those of MT2, MT16, …", stated rather than stored. The domain bounds are
    part of the statement and not decoration — the same quantity may be a sum
    over one band and an explicit matrix over another.

    **Reshaped 2026-08-12** from two parallel lists ``references`` and
    ``coefficients``, which could fall out of step and had nowhere to keep each
    term's ``ENDF_MFMT``. Nothing constructed it — it was phase 3b scaffolding
    — so the change cost no caller. The two lists survive as read-only
    properties.
    """

    summands: List[Summand] = field(default_factory=list)
    label: Optional[str] = None
    domainMin: Optional[float] = None
    domainMax: Optional[float] = None
    domainUnit: str = ""

    def __len__(self) -> int:
        return len(self.summands)

    def __iter__(self) -> Iterator[Summand]:
        return iter(self.summands)

    @property
    def references(self) -> List[str]:
        return [summand.href for summand in self.summands]

    @property
    def coefficients(self) -> List[float]:
        return [summand.coefficient for summand in self.summands]


@dataclass
class ShortRangeSelfScalingVariance:
    """§25. The variance ENDF's LB=8/LB=9 states, which does not survive grouping.

    A short-range self-scaling term describes variance fully correlated *within*
    an energy group and uncorrelated between groups, so its magnitude depends on
    how wide the processing groups turn out to be —
    ``dependenceOnProcessedGroupWidth`` says how it scales with that width
    (``'inverse'`` in every file examined).

    It is a class of its own rather than another :class:`CovarianceMatrix`
    precisely so that nothing can add it to one by accident: it is not a
    component that can be summed with its siblings on a fixed grid, and a
    ``mixed`` that contains one does **not** mean "add these matrices together".
    """

    matrix: Optional[CovarianceMatrix] = None
    dependenceOnProcessedGroupWidth: Optional[str] = None
    label: Optional[str] = None


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
    #: be the whole cost of the write path (see ``docs/library/library-gaps.md`` D2
    #: and M2).
    provenance: Optional[object] = None

    #: §25.2.2's own attribute, kept because the file states it rather than
    #: leaving it to be inferred. It is normally redundant with "has a
    #: ``columnData``" and it is **not** always: a section may carry the
    #: attribute for a reader's benefit before its column link is read, and a
    #: writer that dropped it would emit a file the publisher's own tools see
    #: as a different kind of section. :attr:`isCrossTerm` is the question to
    #: ask; this is the answer the file gave.
    crossTerm: bool = False

    @property
    def isCrossTerm(self) -> bool:
        """A block between two *different* quantities.

        True when the file said so **or** when a ``columnData`` is present. The
        two agree on every section in ENDF/B-VIII.1-GNDS; taking either alone
        would make the answer depend on which the file happened to write.
        """
        return self.crossTerm or self.columnData is not None


@dataclass
class ParameterLink:
    """§25.3.2. One contiguous run of matrix rows, and what parameters they are.

    A parameter covariance is not gridded on energy, so nothing about a row is
    recoverable from a grid the way it is for :class:`CovarianceMatrix`: row 47
    means *the neutron width of the twelfth resonance* or it means nothing at
    all. This is what carries that, and it is the reason a parameter covariance
    needs its own container rather than reusing the cross-section one with the
    grids left empty.

    The run is deliberately a *resonance*, not a single parameter: ENDF orders
    an MF32 matrix resonance-major (every parameter of resonance 1, then every
    parameter of resonance 2), so a run per resonance describes the file's own
    blocking. ``parameterNames`` names the run's entries in that order, which is
    what makes :meth:`ParameterCovarianceMatrix.rowLabels` able to say what a
    single row is.
    """

    label: str
    href: str
    nParameters: int = 1
    matrixStartIndex: int = 0
    parameterNames: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.parameterNames and len(self.parameterNames) != self.nParameters:
            raise ValueError(
                f"{len(self.parameterNames)} names for {self.nParameters} "
                f"parameters in {self.label!r}"
            )


@dataclass
class ParameterCovarianceMatrix:
    """§25.3.2. The matrix, plus the parameter list that indexes its rows.

    ``parameterValues`` is the one field with no GNDS counterpart, and it is
    here on purpose. In GNDS the central values live in the ``reactionSuite``
    and the covariance reaches them through ``href``; in ENDF, **File 32
    re-declares them itself** — an LCOMP=2 body writes each parameter and its
    uncertainty in the same record, which is the whole reason that format
    exists. Dropping the copy would mean a caller that wants to sample has to
    go back to File 2 and re-derive the correspondence row by row, which is
    exactly the "parsed and then dropped" mistake ``docs/library/library-gaps.md`` M4
    records for nu-bar.

    The two copies can disagree, and that is a property of the evaluation
    rather than of this class: ``href`` says where the authoritative one is,
    ``parameterValues`` says what File 32 believed when it wrote the matrix.
    """

    matrix: np.ndarray
    parameters: List[ParameterLink] = field(default_factory=list)
    isRelative: bool = False
    label: Optional[str] = None
    parameterValues: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim != 2:
            raise ValueError(f"a covariance matrix is 2-D, got shape {self.matrix.shape}")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"a parameter covariance is square, got shape {self.matrix.shape}"
            )
        if self.parameters:
            declared = sum(link.nParameters for link in self.parameters)
            if declared != self.matrix.shape[0]:
                raise ValueError(
                    f"the parameter links account for {declared} rows, "
                    f"the matrix has {self.matrix.shape[0]}"
                )
        if self.parameterValues is not None:
            self.parameterValues = np.asarray(self.parameterValues, dtype=float)
            if self.parameterValues.size != self.matrix.shape[0]:
                raise ValueError(
                    f"{self.parameterValues.size} central values for "
                    f"{self.matrix.shape[0]} rows"
                )

    @property
    def order(self) -> int:
        return int(self.matrix.shape[0])

    def rowLabels(self) -> List[str]:
        """One human-readable label per row, expanded from the links."""
        labels: List[str] = []
        for link in self.parameters:
            names = link.parameterNames or [
                str(i) for i in range(link.nParameters)
            ]
            labels.extend(f"{link.label}/{name}" for name in names)
        return labels

    def uncertainties(self) -> np.ndarray:
        """The stated standard deviations — the square root of the diagonal.

        ``abs`` before the root rather than after: MF32 matrices are not always
        numerically PSD (Mn-55's short-range block has a smallest eigenvalue of
        -2.3e-9 against a largest of 1.8e7), and a negative diagonal entry is a
        defect worth surfacing as a value rather than as a ``nan`` three call
        frames away.
        """
        return np.sqrt(np.abs(np.diag(self.matrix)))


@dataclass
class ParameterCovariance:
    """§25.3.1. One covariance about a set of model parameters.

    The sibling of :class:`CovarianceSection`, and separate from it for the
    reason §25.3 is a separate subsection of the standard: a covariance whose
    rows are *parameters* cannot be interchanged with one whose rows are bins
    of a grid, and code that treats the two alike will eventually collapse a
    resonance index into an energy. They are kept in different lists on the
    suite so that no consumer has to check which kind it was handed.
    """

    label: str
    rowData: Optional[DataLink] = None
    columnData: Optional[DataLink] = None
    form: Optional[object] = None   # ParameterCovarianceMatrix
    provenance: Optional[object] = None

    @property
    def isCrossTerm(self) -> bool:
        return self.columnData is not None


@dataclass
class AverageParameterCovariance:
    """§25.3. A covariance about an *unresolved-region average* parameter.

    The third kind, and it is neither of the other two. Its rows are not bins of
    a cross-section grid (:class:`CovarianceSection`) and they are not individual
    resonance parameters (:class:`ParameterCovariance`): they are the energy bins
    of one URR average — a level spacing, an average width — whose ``rowData``
    points into ``resonances/unresolved/tabulatedWidths``. So its ``form`` is an
    ordinary gridded :class:`CovarianceMatrix` while the quantity it is about is
    a model parameter.

    It lives in ``CovarianceSuite.parameterCovariances`` beside
    :class:`ParameterCovariance` because that is the container GNDS puts it in.
    """

    label: str
    rowData: Optional[DataLink] = None
    columnData: Optional[DataLink] = None
    form: Optional[object] = None   # CovarianceMatrix
    crossTerm: bool = False

    @property
    def isCrossTerm(self) -> bool:
        return self.crossTerm or self.columnData is not None


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

    #: Not a GNDS node. What the decode did beyond producing this object, on
    #: the same footing as :attr:`ReactionSuite.report` and for the same
    #: reason: the decoders keep returning it as the second element of a tuple,
    #: and a tuple element is what gets dropped by a caller in a hurry. A §25.3
    #: parameter covariance that is a cross term loses that fact on the way out
    #: (``covariances.xsd:160-168`` has no ``columnData`` there), ``flattened``
    #: arrays are read and not written back, and a covariance file read without
    #: its ``reactionSuite`` sibling cannot resolve a single ``rowData`` href —
    #: all three are report entries and none of them is visible in the object
    #: otherwise.
    report: Optional[ConversionReport] = None

    def __len__(self) -> int:
        return len(self.covarianceSections)

    def __iter__(self) -> Iterator[CovarianceSection]:
        return iter(self.covarianceSections)

    def byLabel(self, label: str) -> CovarianceSection:
        for section in self.covarianceSections:
            if section.label == label:
                return section
        raise KeyError(f"no covariance section labelled {label!r}")

    def __repr__(self) -> str:
        return (
            f"CovarianceSuite({self.target or '?'}, "
            f"n_sections={len(self.covarianceSections)}, "
            f"n_parameterCovariances={len(self.parameterCovariances)})"
        )
