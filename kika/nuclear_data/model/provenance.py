"""Typed, per-source-format provenance — the replacement for the ``metadata`` dict.

**What is wrong with the dict.** The flat classes carry ``metadata: Dict``, and
four packages reach into it by string key, so it is public API without being
declared as any. Worse, it mixes namespaces: ENDF puts ``mat, awr, qm, qi, lr,
interpolation_regions`` in it and ACE puts ``source_format, ace_zaid,
ace_extension, ...`` in the same dict, so "does this section have a Q value" is
answered by a ``.get`` that cannot distinguish *absent* from *zero*. That is
half of the Q = 0 defect.

**The int/float trap, recorded before it bites.** ENDF's fixed-format floats are
parsed to ``int`` whenever the value is integral, so on the committed Fe-56
slice ``emax`` is ``150000000`` and ``abundance`` is ``1``, both ``int``, while
on another tape the same fields are ``float``. Every numeric field below is
therefore annotated ``float`` and **coerced at construction**, which is the only
honest option: annotating ``int`` would be wrong on the next tape, and
annotating ``float`` without coercing would be a lie about what is stored.
``kika/nuclear_data/tests/test_metadata_contract.py`` pins the underlying
behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["Provenance", "EndfProvenance", "AceProvenance"]


@dataclass
class Provenance:
    """Base: where a piece of data came from, in that format's own terms."""

    sourceFormat: str = "unknown"

    def asDict(self) -> Dict[str, object]:
        """Flat view, for the interop bridge and for diagnostics."""
        from dataclasses import asdict

        return asdict(self)


def _asFloat(value: object) -> Optional[float]:
    """Coerce an ENDF numeric field, keeping ``None`` distinct from ``0.0``."""
    return None if value is None else float(value)


def _asEndfInt(value: object) -> Optional[int]:
    """Round, do not truncate, an ENDF field that is conceptually an integer.

    ENDF writes ZA as a fixed-format float and kika's reader rebuilds it as
    ``mantissa * 10**exponent``, which is not exact: Th-232's ``9.023200+4``
    comes back as ``90231.99999999999``. ``int()`` on that is 90231, so a
    section decoded and re-encoded names a different nuclide — and because the
    flat classes truncate the same way, a gate that compares the model against
    them agrees on the wrong answer. Caught by comparing against the *file*.
    """
    if value is None:
        return None
    return int(round(float(value)))


@dataclass
class EndfProvenance(Provenance):
    """ENDF-6 bookkeeping that the physics model has no home for.

    Fields that *are* physics live in the model proper: the reaction Q value is
    ``outputChannel.Q``, the interpolation regions are the ``regions1d``, and
    the ZA is the nuclide in ``PoPs``. What remains here is genuinely
    format-specific.

    ``qm`` is the one arguable case. ENDF carries two Q values: ``QM``, the
    mass-difference Q for the ground state, and ``QI``, the Q of the particular
    level. GNDS §17.1.1 has a single ``Q``, which corresponds to ``QI``; ``QM``
    is derivable from the masses in ``PoPs``. So ``QI`` becomes the model's Q
    and ``QM`` stays here, to be written back byte-identically rather than
    recomputed.
    """

    sourceFormat: str = "endf"
    mat: Optional[int] = None
    awr: Optional[float] = None
    #: ENDF ZA = 1000*Z + A, as the section header wrote it.
    za: Optional[int] = None
    #: ENDF QM — the mass-difference Q. See the class docstring.
    qm: Optional[float] = None
    #: ENDF LR — the complex-breakup flag.
    lr: Optional[int] = None
    #: The (NBT, INT) pairs exactly as the file wrote them. The model's
    #: ``regions1d`` is built from these; keeping the originals means a
    #: round-trip does not depend on the reconstruction being lossless.
    interpolationRegions: List[Tuple[int, int]] = field(default_factory=list)
    #: MF1/451 header fields, for a section decoded from one.
    headerFields: Dict[str, object] = field(default_factory=dict)
    evaluationInfo: Dict[str, str] = field(default_factory=dict)
    #: MF1/451's NWD descriptive records, data columns only (the ID columns are
    #: regenerated on the way out). ``evaluationInfo`` holds the seven fields
    #: the first two of these records are *parsed into*; this holds all NWD of
    #: them verbatim, because the rest — the free-text comment block an
    #: evaluator wrote — is parsed into nothing at all and is most of the
    #: section. Without it MF1/451 cannot be written back, only approximated.
    descriptiveText: List[str] = field(default_factory=list)
    #: MF1/451's NXC directory entries, ``(MF, MT, NC, MOD)``. NC is a **line
    #: count**, so this is only valid for the tape it was read from; after a
    #: section is replaced, ``kika.endf.writers.update_directory`` rebuilds it
    #: from the written file. Kept here so a round trip that changes nothing
    #: writes back what it read rather than a recomputation of it.
    directory: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.awr = _asFloat(self.awr)
        self.qm = _asFloat(self.qm)
        if self.lr is not None:
            self.lr = int(self.lr)
        self.mat = _asEndfInt(self.mat)
        self.za = _asEndfInt(self.za)


@dataclass
class AceProvenance(Provenance):
    """ACE bookkeeping.

    ACE is a *processed* representation, not a peer evaluation: it has been
    reconstructed, Doppler-broadened and put on a grid, and it records no
    reaction Q values at all. The absence is declared here rather than defaulted
    to zero, which is the structural version of the phase 1 fix.
    """

    sourceFormat: str = "ace"
    zaid: Optional[int] = None
    extension: Optional[str] = None
    awr: Optional[float] = None
    comment: Optional[str] = None
    date: Optional[str] = None
    matid: Optional[int] = None
    formatVersion: Optional[str] = None
    temperatureMeV: Optional[float] = None

    def __post_init__(self) -> None:
        self.awr = _asFloat(self.awr)
        self.temperatureMeV = _asFloat(self.temperatureMeV)

    @property
    def carriesQValues(self) -> bool:
        """Always ``False``. Stated as a property so a caller can ask."""
        return False
