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
from typing import List, Optional, Tuple

import numpy as np

from .conversion import ConversionReport
from .covariances import CovarianceSuite
from .cross_section_forms import EVAL_LABEL
from .enums import Frame
from .functions import Regions1d, XYs1d
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


def _eV(value: float) -> str:
    """An energy in eV, rendered at a scale a human reads without counting zeros."""
    for scale, unit in ((1e6, "MeV"), (1e3, "keV")):
        if abs(value) >= scale:
            return f"{value / scale:g} {unit}"
    return f"{value:g} eV"


def _pointwise(form: object) -> Tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` arrays out of whichever 1d container a form turned out to be.

    Copies, so a caller who edits what they were handed does not edit the
    evaluation underneath it.
    """
    if isinstance(form, XYs1d):
        return form.xs.copy(), form.ys.copy()
    if isinstance(form, Regions1d):
        xs, ys, _ = form.toEndfRegions()
        return xs, ys
    raise TypeError(
        f"a {type(form).__name__} is not a tabulated function, so it has no "
        f"(E, sigma) pair to hand back. Forms such as ResonancesWithBackground "
        f"and Reference have to be reconstructed or dereferenced first; reach "
        f"for the form itself rather than this shortcut."
    )


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

    #: Not a GNDS node either. What the decode did beyond producing this object:
    #: what it lost, approximated or refused. The decoders keep returning it as
    #: the second element of a tuple -- that surface is unchanged -- but a tuple
    #: element is discarded by every caller who is in a hurry, and "which MFs did
    #: this evaluation carry that kika cannot read?" is a question the object has
    #: to be able to answer on its own. `kika.read()` fills this in; `repr` and
    #: `summary()` show its counts, so a partial decode announces itself.
    report: Optional[ConversionReport] = None

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

    def findReactionByENDF_MT(self, mt: int) -> Optional[Reaction]:
        """The same look-up, returning ``None`` instead of raising.

        Two callers, two meanings. An encoder asking for the MT it is about
        wants :meth:`reactionByENDF_MT`, because its absence is a bug. A decoder
        asking "is there a reaction to hang this MF4 section on?" is asking a
        question with a legitimate *no* — an ENDF file may carry an MF4/MT with
        no MF3/MT — and should not have to catch ``KeyError`` to find out.
        """
        try:
            return self.reactionByENDF_MT(mt)
        except KeyError:
            return None

    @property
    def hasResonances(self) -> bool:
        return self.resonances is not None

    def styleLabels(self) -> List[str]:
        return self.styles.labels

    def cross_section(
        self, mt: int, form: str = EVAL_LABEL
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(E, sigma)`` for one MT, as two arrays. The 90 % journey, in one call.

        The long road — ``suite.reactions[102].crossSection.evaluated.xs`` — stays
        exactly as it is and remains the honest one, because it says which *form*
        it is reading. This is the shortcut, and it goes through
        :meth:`reactionByENDF_MT`, so summed quantities such as MT1 and MT4 work
        here even though they are not in ``reactions``.

        **The name.** kika's convention is GNDS nouns and Python verbs, so a
        method may be ``snake_case`` where an attribute may not. It is registered
        in ``tests/test_gnds_naming.py::DIVERGENCES`` so that the next reader does
        not mistake it for a snake_case twin of ``Reaction.crossSection`` and
        "fix" it.

        A ``Regions1d`` form is flattened onto one grid by
        :meth:`~kika.nuclear_data.model.functions.regions1d.Regions1d.toEndfRegions`,
        which already drops the boundary point the regions share. **The regions'
        differing interpolation rules do not survive that flattening** — the
        returned pair is faithful at its own nodes and nowhere else, so anything
        that needs a value *between* two nodes must call ``form.evaluate(E)``
        instead, which interpolates each region by its own rule.
        """
        reaction = self.reactionByENDF_MT(mt)
        # Raises with the available labels listed -- CrossSection.__getitem__
        # already writes that message, so it is not rewritten here.
        return _pointwise(reaction.crossSection[form])

    def summary(self) -> str:
        """A few lines saying what is actually in here, including what is missing."""
        lines = [
            f"{self.projectile} + {self.target}   "
            f"[{self.evaluation or 'no evaluation id'}]  GNDS {self.format}",
            f"  styles       {self.styles.labels or 'none'}",
            f"  reactions    {len(self.reactions)}  MT{self.reactions.ENDF_MTs}",
        ]
        for name in ("sums", "fissionComponents", "productions",
                     "orphanProducts", "incompleteReactions"):
            container = getattr(self, name)
            if len(container):
                lines.append(f"  {name:<12} {len(container)}  MT{container.ENDF_MTs}")
        lines.append(f"  PoPs         {len(self.PoPs)} particles")
        if self.resonances is None:
            lines.append("  resonances   none")
        else:
            domain = self.resonances.domain
            span = "no region" if domain is None else f"{_eV(domain[0])} - {_eV(domain[1])}"
            lines.append(
                f"  resonances   {len(self.resonances.resolved)} resolved, "
                f"{'1' if self.resonances.unresolved is not None else 'no'} unresolved"
                f"   [{span}]"
            )
        lines.append(
            "  covariances  none"
            if self.covarianceSuite is None
            else f"  covariances  {len(self.covarianceSuite)} sections"
        )
        lines.append(
            f"  decode       {self.report.summary()}" if self.report is not None
            else "  decode       not recorded"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = [
            f"{self.projectile} + {self.target}",
            f"{self.evaluation!r}" if self.evaluation else "no evaluation id",
            f"{len(self.reactions)} reactions",
        ]
        if self.resonances is not None and self.resonances.domain is not None:
            parts.append(f"resonances to {_eV(self.resonances.domain[1])}")
        if self.covarianceSuite is not None:
            parts.append(f"{len(self.covarianceSuite)} covariance sections")
        if self.report is not None and not self.report.isClean:
            parts.append(self.report.summary())
        return f"<ReactionSuite {' | '.join(parts)}>"
