"""``ConversionReport`` — nothing converts silently.

Every decoder and encoder returns one. The roadmap's hard rule for phase 7 is
that *"while MF coverage is partial, the writer emits only what kika owns and
declares the gaps. A structurally valid, physically incomplete reactionSuite is
worse than none, because it carries authority it has not earned."* This is the
object that does the declaring, and it exists from the first decoder rather than
being retrofitted when the first gap is noticed.

Four kinds, and the distinction between them is the point:

``warnings``       something looked odd but nothing was lost
``losses``         data in the source that is not in the result
``approximations`` data that is in the result but changed on the way
``unsupported``    constructs the reader recognised and cannot handle at all

The one that matters most is ``approximations``. A loss is visible — the field
is missing. An approximation looks like data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = ["ConversionReport"]


@dataclass
class ConversionReport:
    """What a conversion did, beyond producing its result."""

    warnings: List[str] = field(default_factory=list)
    losses: List[str] = field(default_factory=list)
    approximations: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def lost(self, message: str) -> None:
        self.losses.append(message)

    def approximated(self, message: str) -> None:
        self.approximations.append(message)

    def unsupportedNode(self, message: str) -> None:
        self.unsupported.append(message)

    def extend(self, other: "ConversionReport") -> None:
        self.warnings.extend(other.warnings)
        self.losses.extend(other.losses)
        self.approximations.extend(other.approximations)
        self.unsupported.extend(other.unsupported)

    @property
    def isClean(self) -> bool:
        """Nothing was lost, approximated or refused. Warnings do not count."""
        return not (self.losses or self.approximations or self.unsupported)

    @property
    def isEmpty(self) -> bool:
        return self.isClean and not self.warnings

    def __len__(self) -> int:
        return len(self.warnings) + len(self.losses) + len(self.approximations) + len(self.unsupported)

    def __bool__(self) -> bool:
        # A report is an object that always exists; `if report:` must not read
        # "nothing to report" as "no report".
        return True

    def summary(self) -> str:
        parts = []
        for name in ("losses", "approximations", "unsupported", "warnings"):
            entries = getattr(self, name)
            if entries:
                parts.append(f"{len(entries)} {name}")
        return ", ".join(parts) if parts else "clean"

    def __repr__(self) -> str:
        return f"ConversionReport({self.summary()})"
