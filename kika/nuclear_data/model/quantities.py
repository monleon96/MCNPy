"""GNDS-2.1 §2.3.3 ``physicalQuantity``: a scalar that carries its own unit.

One of the two places a unit is allowed to live — the other is
:class:`~kika.nuclear_data.model.axes.Axis`, for the axes of a function. Never
per array element, which is what makes "everything carries its units" cost
nothing in the numpy hot paths.

Attribute spellings are GNDS's, so ``value``, ``unit``, ``label`` and
``uncertainty``. ``label`` exists because §2.3.3 allows several assignments of
the same property to coexist and be told apart by label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .units import Unit, conversion_factor, parse_unit

__all__ = ["PhysicalQuantity", "RangeQuantity", "Uncertainty"]


@dataclass(frozen=True)
class Uncertainty:
    """A standard uncertainty on a scalar, in the same unit as its value.

    Deliberately minimal. GNDS §7 has a much richer uncertainty tree for
    *functions*; this is only the scalar case §2.3.3 refers to.
    """

    value: float

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"uncertainty must not be negative, got {self.value}")


@dataclass(frozen=True)
class PhysicalQuantity:
    """A scalar with a unit (§2.3.3).

    The unit may be omitted, which §2.3.3 defines as dimensionless rather than
    as unknown — an important distinction, because it means a missing unit is
    never a licence to guess.
    """

    value: float
    unit: str = ""
    label: Optional[str] = None
    uncertainty: Optional[Uncertainty] = None

    def __post_init__(self) -> None:
        parse_unit(self.unit)  # reject an unspellable unit at construction

    @property
    def parsedUnit(self) -> Unit:
        return parse_unit(self.unit)

    def convertedTo(self, unit: str) -> "PhysicalQuantity":
        """The same quantity expressed in ``unit``.

        Conversion is explicit and confined to boundaries on purpose: the bug
        class this model replaces is the bare ``* 1e6  # MeV -> eV`` written
        inline, where nothing records what the number meant before or after.
        """
        factor = conversion_factor(self.unit, unit)
        return PhysicalQuantity(
            value=self.value * factor,
            unit=unit,
            label=self.label,
            uncertainty=(
                Uncertainty(self.uncertainty.value * factor)
                if self.uncertainty is not None
                else None
            ),
        )

    def __str__(self) -> str:
        return f"{self.value}" if not self.unit else f"{self.value} {self.unit}"


@dataclass(frozen=True)
class RangeQuantity:
    """An interval carrying its unit — the schema's ``RangeQuantityType``.

    ``min``/``max``/``unit`` are GNDS's spellings, shadowing two builtins as
    *attribute* names, which is the naming rule this model has followed since
    §5.1's ``Axis.index``.

    One node uses it today: every ``evaluated`` style's
    ``projectileEnergyDomain``, in all 558 neutron evaluations. It is not the
    same statement as the union of the reactions' domains — it is the evaluator
    saying what the evaluation is *for*, and a reaction may legitimately stop
    short of it.
    """

    min: float
    max: float
    unit: str = ""

    def __post_init__(self) -> None:
        parse_unit(self.unit)
        if self.max < self.min:
            raise ValueError(
                f"a range runs min to max; got min={self.min}, max={self.max}"
            )

    def __contains__(self, value: float) -> bool:
        return self.min <= value <= self.max

    def __str__(self) -> str:
        span = f"{self.min} to {self.max}"
        return span if not self.unit else f"{span} {self.unit}"
