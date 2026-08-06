"""GNDS-2.1 §5.2: the ``values`` container.

A flat list of numbers plus the one piece of metadata they share, their type.
The interesting part is the leading/trailing zero compression (§5.2.1): ``start``
is the index of the first stored value and ``length`` the total count including
the zeros that were not written. Covariance rows are mostly zero, so this is not
an optimisation the format added for fun — it is what makes a ``covarianceSuite``
a reasonable size.

Round-tripping that compression exactly is a phase 6/7 concern, but the shape
has to be right now, because the decoders written in phase 3c fill it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .enums import ValueType

__all__ = ["Values"]


@dataclass
class Values:
    """§5.2.1. A typed list of numbers, optionally zero-compressed."""

    values: np.ndarray
    valueType: ValueType = ValueType.Float64
    start: int = 0
    length: Optional[int] = None

    def __post_init__(self) -> None:
        dtype = float if self.valueType is ValueType.Float64 else np.int32
        self.values = np.asarray(self.values, dtype=dtype)
        if self.values.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {self.values.shape}")
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.length is not None and self.length < self.start + self.values.size:
            raise ValueError(
                f"length {self.length} is shorter than start {self.start} plus "
                f"{self.values.size} stored values"
            )

    def __len__(self) -> int:
        """The *logical* length, zeros included — not the number stored."""
        return self.length if self.length is not None else self.start + self.values.size

    @property
    def isCompressed(self) -> bool:
        return self.start > 0 or (
            self.length is not None and self.length > self.start + self.values.size
        )

    def expanded(self) -> np.ndarray:
        """The full array with the omitted leading and trailing zeros restored."""
        if not self.isCompressed:
            return self.values
        dtype = self.values.dtype
        out = np.zeros(len(self), dtype=dtype)
        out[self.start : self.start + self.values.size] = self.values
        return out

    @classmethod
    def compressed(cls, array: Sequence[float], valueType: ValueType = ValueType.Float64) -> "Values":
        """Build a :class:`Values`, dropping leading and trailing zeros."""
        dtype = float if valueType is ValueType.Float64 else np.int32
        full = np.asarray(array, dtype=dtype)
        nonzero = np.flatnonzero(full)
        if nonzero.size == 0:
            return cls(np.asarray([], dtype=dtype), valueType, 0, int(full.size))
        first, last = int(nonzero[0]), int(nonzero[-1])
        return cls(full[first : last + 1], valueType, first, int(full.size))
