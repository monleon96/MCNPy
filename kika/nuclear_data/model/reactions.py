"""GNDS-2.1 §15.1.1 ``reaction``, and the containers that hold reactions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from .cross_section_forms import CrossSection
from .output_channel import OutputChannel
from .reaction_id import ReactionId

__all__ = ["Reaction", "Reactions", "Sums", "OrphanProducts",
           "FissionComponents", "Productions", "IncompleteReactions"]


@dataclass
class Reaction:
    """§15.1.1. One reaction: its identity, its cross section, its products."""

    id: ReactionId
    crossSection: CrossSection = field(default_factory=CrossSection)
    outputChannel: OutputChannel = field(default_factory=OutputChannel)
    doubleDifferentialCrossSection: Optional[object] = None
    availableEnergy: Optional[object] = None
    availableMomentum: Optional[object] = None
    #: Not a GNDS node. Where the section came from, in its own format's terms,
    #: so an encoder can write it back without recomputing anything. GNDS keeps
    #: this kind of thing in `documentation`; kika keeps it typed and separate
    #: because the encoders need it and `documentation` is free text.
    provenance: Optional[object] = None

    @property
    def label(self) -> str:
        return self.id.label

    @property
    def ENDF_MT(self) -> Optional[int]:
        """Present because §15.1.1 requires it, and *derived* because §15.1.1
        deprecates it in the same breath. Nothing in the model keys off it."""
        return self.id.ENDF_MT

    def __repr__(self) -> str:
        return f"Reaction({self.id}, forms={sorted(self.crossSection.forms)})"


class _ReactionList:
    """Shared behaviour for the several §14.1.1 children that are lists of reactions."""

    def __init__(self, reactions: Optional[List[Reaction]] = None) -> None:
        self.reactions: List[Reaction] = list(reactions or [])

    def __len__(self) -> int:
        return len(self.reactions)

    def __iter__(self) -> Iterator[Reaction]:
        return iter(self.reactions)

    def __bool__(self) -> bool:
        # An empty container is still *present*, which is the roadmap's rule.
        # Without this, `if suite.reactions:` would read an empty-but-declared
        # container as absent, which is exactly the distinction being made.
        return True

    def append(self, reaction: Reaction) -> None:
        self.reactions.append(reaction)

    def byLabel(self, label: str) -> Reaction:
        for reaction in self.reactions:
            if reaction.label == label:
                return reaction
        raise KeyError(f"no reaction labelled {label!r}")

    def byENDF_MT(self, mt: int) -> Reaction:
        """Look-up by MT, for the ENDF adapters. Deliberately not the primary key."""
        for reaction in self.reactions:
            if reaction.ENDF_MT == mt:
                return reaction
        raise KeyError(f"no reaction with ENDF_MT {mt}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self.reactions)})"


class Reactions(_ReactionList):
    """§14.1.1. The exclusive reactions. Their cross sections sum to the total."""


class Sums(_ReactionList):
    """§14.1.1. Summed cross sections and multiplicities (MT1, MT4, ...)."""


class OrphanProducts(_ReactionList):
    """§14.1.1. Products with no identified parent reaction."""


class FissionComponents(_ReactionList):
    """§14.1.1. First-chance, second-chance, ... fission."""


class Productions(_ReactionList):
    """§14.1.1. Production reactions."""


class IncompleteReactions(_ReactionList):
    """§14.1.1. Reactions with only some data available."""
