"""GNDS-2.1 §14.1.1 ``reactionSuite`` — the evaluation root.

Attributes (§14.1.1, all required): ``evaluation``, ``format``, ``projectile``,
``projectileFrame``, ``target``, ``interaction``. Children: ``externalFiles``,
``styles``, ``PoPs``, ``resonances``, ``reactions``, ``orphanProducts``,
``sums``, ``fissionComponents``, ``productions``, ``incompleteReactions``,
``applicationData``.

**Every child is present as an empty container, never as ``None``.** §14.1.2
notes that most children are optional so libraries can set their own
requirements, which is exactly what lets kika ship a partially-filled hierarchy
honestly — a slot that exists and is empty says "kika models this and this file
has none", while a missing attribute says nothing at all and fails three frames
away. The list containers even define ``__bool__`` as ``True`` so that
``if suite.reactions:`` cannot silently read *empty* as *absent*.

MF5, MF6 and MF12-15 have slots here that phase 7 fills. Filling them
restructures nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .covariances import CovarianceSuite
from .enums import Frame
from .pops import PoPs
from .reactions import (FissionComponents, IncompleteReactions, OrphanProducts,
                        Productions, Reaction, Reactions, Sums)
from .resonances import Resonances
from .styles import Styles

__all__ = ["ExternalFile", "ExternalFiles", "ApplicationData", "ReactionSuite"]

#: §14.1.1 ``interaction``. The spec's three values.
INTERACTIONS = ("nuclear", "atomic", "thermalNeutronScatteringLaw")

#: This library targets GNDS 2.1. FUDGE 6.10.0 emits 2.0, so a file read from
#: FUDGE will not match and the reader is expected to warn rather than pretend.
GNDS_FORMAT = "2.1"


@dataclass
class ExternalFile:
    """§14.1.1. A file this evaluation refers to — usually its covarianceSuite."""

    label: str
    path: str
    checksum: Optional[str] = None
    algorithm: Optional[str] = None   # §3.4.3: 'md5' or 'sha1'


@dataclass
class ExternalFiles:
    """The container, present even when empty."""

    files: List[ExternalFile] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self):
        return iter(self.files)

    def __bool__(self) -> bool:
        return True  # present-but-empty is not absent

    def byLabel(self, label: str) -> ExternalFile:
        for entry in self.files:
            if entry.label == label:
                return entry
        raise KeyError(f"no external file labelled {label!r}")


@dataclass
class ApplicationData:
    """§14.1.1. Application-specific data GNDS does not interpret."""

    entries: List[object] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return True


@dataclass
class ReactionSuite:
    """§14.1.1. One evaluation, of one projectile on one target."""

    evaluation: str
    projectile: str
    target: str
    projectileFrame: Frame = Frame.lab
    interaction: str = "nuclear"
    format: str = GNDS_FORMAT

    externalFiles: ExternalFiles = field(default_factory=ExternalFiles)
    styles: Styles = field(default_factory=Styles)
    PoPs: PoPs = field(default_factory=PoPs)
    resonances: Optional[Resonances] = None
    reactions: Reactions = field(default_factory=Reactions)
    orphanProducts: OrphanProducts = field(default_factory=OrphanProducts)
    sums: Sums = field(default_factory=Sums)
    fissionComponents: FissionComponents = field(default_factory=FissionComponents)
    productions: Productions = field(default_factory=Productions)
    incompleteReactions: IncompleteReactions = field(default_factory=IncompleteReactions)
    applicationData: ApplicationData = field(default_factory=ApplicationData)

    #: Not a GNDS child. kika keeps the covariances it has read beside the
    #: evaluation for convenience; in a GNDS file they are a separate root that
    #: `externalFiles` points at, and the writer must emit them that way.
    covarianceSuite: Optional[CovarianceSuite] = None

    #: Not a GNDS node either -- the MF1/451 header, kept so the encoder can
    #: write it back byte for byte. See Reaction.provenance.
    provenance: Optional[object] = None

    def __post_init__(self) -> None:
        self.projectileFrame = Frame(self.projectileFrame)
        if self.interaction not in INTERACTIONS:
            raise ValueError(
                f"interaction must be one of {INTERACTIONS}, got {self.interaction!r}"
            )

    # ------------------------------------------------------------------

    def reactionByLabel(self, label: str) -> Reaction:
        return self.reactions.byLabel(label)

    def reactionByENDF_MT(self, mt: int) -> Reaction:
        """Look-up by MT, for the ENDF adapters.

        Searches ``reactions`` first and then ``sums``, because MT1 and MT4 are
        summed quantities rather than exclusive reactions and a caller asking
        for "MT1" means the sum.
        """
        for container in (self.reactions, self.sums, self.fissionComponents,
                          self.productions, self.incompleteReactions):
            try:
                return container.byENDF_MT(mt)
            except KeyError:
                continue
        raise KeyError(f"no reaction with ENDF_MT {mt} anywhere in this suite")

    @property
    def hasResonances(self) -> bool:
        return self.resonances is not None

    def styleLabels(self) -> List[str]:
        return self.styles.labels

    def __repr__(self) -> str:
        return (
            f"ReactionSuite({self.projectile} + {self.target}, "
            f"evaluation={self.evaluation!r}, format={self.format}, "
            f"n_reactions={len(self.reactions)}, styles={self.styles.labels})"
        )
