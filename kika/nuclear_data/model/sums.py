"""GNDS-2.1 §21: ``sums`` — quantities defined as the sum of other quantities.

§21.1 gives ``sums`` two children: ``crossSectionSums`` and
``multiplicitySums``. kika's :class:`~kika.nuclear_data.model.reactions.Sums`
has always been the first of the two — a list of reactions, holding MT1 and MT4
— and this module adds the second.

**Why nu-bar needs it.** ENDF writes three fission multiplicities: MT452 total,
MT455 delayed, MT456 prompt, with nu_total = nu_prompt + nu_delayed. GNDS does
not store all three in the same place, because they are not the same kind of
thing: the prompt one is *the* multiplicity of the neutron coming out of the
fission channel and lives on the product (§17.3), while the total and the
delayed are **derived** and live here, each as a ``multiplicitySum`` that
carries both its own evaluated values and the links to what it sums.

That split is what MF31's covariances have to point at, and pointing them
anywhere else would be the kind of mistake that stays invisible: three matrices
all hanging off one node, distinguishable only by a label nobody validates.
FUDGE's ENDF reader makes the same three placements
(``ENDF_ITYPE_0.py``, ``MultiplicitySum(label="total fission neutron
multiplicity", ENDF_MT=452)``), which is the check that this is the convention
and not an invention of kika's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .output_channel import Multiplicity

__all__ = ["Add", "Summands", "MultiplicitySum", "MultiplicitySums"]


@dataclass
class Add:
    """§21.3. One summand: an xPath to the quantity being added in.

    The node is called ``add`` because §21.3 admits ``subtract`` as well. Only
    the additive one is used by anything kika reads.
    """

    href: str


@dataclass
class Summands:
    """§21.3. What a sum is a sum of, in the order the evaluation gives them.

    **May legitimately be empty while the sum itself has values.** MF1/455 is
    the case: the aggregate delayed nu-bar is in the file, the per-family
    multiplicities it is the sum of are not (they need MF5's weights). An empty
    ``summands`` says "kika has the total and not the parts", which is true; a
    fabricated list of links to empty nodes would not be.
    """

    summands: List[Add] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.summands)

    def __iter__(self):
        return iter(self.summands)

    def __getitem__(self, index):
        return self.summands[index]

    def append(self, summand: Add) -> None:
        self.summands.append(summand)


@dataclass
class MultiplicitySum:
    """§21.3. A multiplicity defined as a sum of others, with its own values.

    ``multiplicity`` is not redundant with ``summands``: the evaluation states
    the total explicitly, and it need not equal the sum of the parts to the last
    digit. Recomputing it from the summands would silently replace what the
    evaluator wrote.
    """

    label: str
    multiplicity: Optional[Multiplicity] = None
    summands: Summands = field(default_factory=Summands)
    #: §21.3 keeps the originating ENDF MT on the node itself; unlike most ENDF
    #: bookkeeping this one is GNDS's own attribute, so it is not provenance.
    ENDF_MT: Optional[int] = None


@dataclass
class MultiplicitySums:
    """§21.1. The ``multiplicitySums`` of one ``reactionSuite``."""

    multiplicitySums: List[MultiplicitySum] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.multiplicitySums)

    def __iter__(self):
        return iter(self.multiplicitySums)

    def __getitem__(self, index):
        return self.multiplicitySums[index]

    def __bool__(self) -> bool:
        # Present-and-empty is not absent; the reactionSuite's rule.
        return True

    def append(self, multiplicitySum: MultiplicitySum) -> None:
        self.multiplicitySums.append(multiplicitySum)

    def byLabel(self, label: str) -> Optional[MultiplicitySum]:
        for entry in self.multiplicitySums:
            if entry.label == label:
                return entry
        return None

    def byENDF_MT(self, mt: int) -> Optional[MultiplicitySum]:
        for entry in self.multiplicitySums:
            if entry.ENDF_MT == mt:
                return entry
        return None

    def __repr__(self) -> str:
        return f"MultiplicitySums({[s.label for s in self.multiplicitySums]})"
