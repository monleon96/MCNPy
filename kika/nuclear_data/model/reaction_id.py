"""Semantic identity of a reaction, with ``ENDF_MT`` present but derived.

**The spec deprecates the attribute it also requires.** §15.1.1, verbatim:
*"This attribute is currently required, but should be considered deprecated.
Future evaluations may include reactions with no MT equivalent, so codes should
not rely on the ENDF_MT."*

So ``ENDF_MT`` is carried — a GNDS file is invalid without it — but nothing in
this model keys off it. Identity is the products and the residual. A reaction
read from a future evaluation with no MT equivalent gets ``ENDF_MT = None`` and
everything except the ENDF writer keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = ["ReactionId"]


@dataclass(frozen=True)
class ReactionId:
    """What a reaction *is*, independently of how ENDF numbers it."""

    label: str
    products: Tuple[str, ...] = ()
    residual: Optional[str] = None
    ENDF_MT: Optional[int] = None
    fissionGenre: Optional[str] = None

    def __post_init__(self) -> None:
        if self.ENDF_MT is not None and not (0 <= self.ENDF_MT <= 999):
            raise ValueError(f"ENDF_MT out of range: {self.ENDF_MT}")

    @property
    def isFission(self) -> bool:
        return self.fissionGenre is not None

    def __str__(self) -> str:
        mt = f" (MT{self.ENDF_MT})" if self.ENDF_MT is not None else ""
        return f"{self.label}{mt}"
