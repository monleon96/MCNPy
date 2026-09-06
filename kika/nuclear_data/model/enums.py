"""GNDS-2.1 enumerated types, with the allowed values copied from the spec.

The strings are the specification's, verbatim, hyphens and camelCase included
(``'lin-lin'``, ``'centerOfMass'``, ``'correspondingEnergies'``). They are what
the GNDS reader and writer will match on, so they are not tidied.

**The trap in the interpolation names.** §3.4.4 footnote 11 and Table 3.3 are
explicit: *"The first string in a name such as 'log-lin' refers to the dependent
axis (the y-axis) and the second to the independent axis (the x-axis)."* So
``'lin-log'`` means the **independent** axis is logarithmic and the dependent
axis linear — the opposite of what the name reads like to most people.

kika's flat classes already use the same convention (``_ENDF_INTERP_TO_NAME``
maps ENDF ``INT=3`` to ``"linlog"`` and ``INT=4`` to ``"loglin"``, and
``kika.processing.interpolation`` logs *x* for code 3 and *y* for code 4), so
:data:`ENDF_INT_TO_INTERPOLATION` is a straight relabelling and not a
reinterpretation. That was checked against the kernel, not assumed.
"""
from __future__ import annotations

from enum import Enum

__all__ = [
    "Frame",
    "Interpolation",
    "InterpolationQualifier",
    "GridStyle",
    "ValueType",
    "ENDF_INT_TO_INTERPOLATION",
    "INTERPOLATION_TO_ENDF_INT",
    "ENDF_DECADE_TO_QUALIFIER",
    "QUALIFIER_TO_ENDF_DECADE",
    "splitEndfTab2Code",
    "joinEndfTab2Code",
]


class _GNDSEnum(str, Enum):
    """A string enum whose ``str()`` is the GNDS token itself.

    Subclassing ``str`` means an instance compares equal to, and serialises as,
    the specification's own spelling — so a caller may pass either the enum or
    the literal string and the model behaves identically.
    """

    def __str__(self) -> str:
        return self.value


class Frame(_GNDSEnum):
    """§3.4.2. Reference frame of a projectile, product or distribution."""

    lab = "lab"
    centerOfMass = "centerOfMass"


class Interpolation(_GNDSEnum):
    """§3.4.4 and Table 3.3. Rule for interpolating between two points."""

    flat = "flat"
    chargedParticle = "charged-particle"
    linlin = "lin-lin"
    linlog = "lin-log"
    loglin = "log-lin"
    loglog = "log-log"


class InterpolationQualifier(_GNDSEnum):
    """§3.4.5. How to interpolate between two N-dimensional functions."""

    direct = "direct"
    unitBase = "unitBase"
    correspondingEnergies = "correspondingEnergies"
    correspondingPoints = "correspondingPoints"


class GridStyle(_GNDSEnum):
    """§5.1.3. What the values on a ``grid`` axis mean."""

    none = "none"
    points = "points"
    boundaries = "boundaries"
    parameters = "parameters"


class ValueType(_GNDSEnum):
    """§5.2.1. Element type of a ``values`` node; ``Float64`` is the default."""

    Float64 = "Float64"
    Integer32 = "Integer32"


#: ENDF-6 INT code → GNDS interpolation. ENDF is where kika's data comes from
#: and §3.4.4 says GNDS adopted the ENDF-6 scheme wholesale, so this is a
#: relabelling with no change of meaning. INT=6 is ``charged-particle``.
ENDF_INT_TO_INTERPOLATION = {
    1: Interpolation.flat,
    2: Interpolation.linlin,
    3: Interpolation.linlog,
    4: Interpolation.loglin,
    5: Interpolation.loglog,
    6: Interpolation.chargedParticle,
}

#: The inverse. Used wherever the model has to hand a code back to ENDF.
INTERPOLATION_TO_ENDF_INT = {v: k for k, v in ENDF_INT_TO_INTERPOLATION.items()}


#: ENDF-6 §0.5.2.1 puts the *qualifier* of a two-dimensional interpolation in
#: the tens digit of the INT code: 1-6 plain, 11-16 the corresponding-point
#: scheme, 21-26 the unit-base scheme. GNDS §3.4.5 states the same thing as two
#: attributes — ``interpolation`` and ``interpolationQualifier`` — so the
#: conversion is a split and a join and not a lookup.
#:
#: **This is not a curiosity.** MF5's TAB2 uses ``INT=22`` in 44 of the 487
#: LF=1 sections of ENDF/B-VIII.1 (N-15, Mg-24 and 42 others), so a reader that
#: only knows 1-6 raises ``KeyError`` on them. It is also why the committed
#: fixture ``micro_fe56.gnds.xml`` carries ``interpolationQualifier="unitbase"``
#: — the GNDS side of this was already built; only the ENDF mapping was absent.
ENDF_DECADE_TO_QUALIFIER = {
    0: None,
    1: InterpolationQualifier.correspondingPoints,
    2: InterpolationQualifier.unitBase,
}

#: The inverse, without the ``None``. ``direct`` and ``correspondingEnergies``
#: are deliberately absent: GNDS admits them and ENDF has no decade for either,
#: so :func:`joinEndfTab2Code` refuses rather than picking a near miss.
QUALIFIER_TO_ENDF_DECADE = {
    qualifier: decade
    for decade, qualifier in ENDF_DECADE_TO_QUALIFIER.items()
    if qualifier is not None
}


def splitEndfTab2Code(code: int):
    """One ENDF two-dimensional INT → ``(interpolation, qualifier)``."""
    code = int(code)
    decade, base = divmod(code, 10)
    if decade not in ENDF_DECADE_TO_QUALIFIER or base not in ENDF_INT_TO_INTERPOLATION:
        raise KeyError(
            f"INT={code} is not an ENDF-6 §0.5.2.1 two-dimensional "
            f"interpolation code: the units digit must be 1-6 and the tens "
            f"digit 0 (plain), 1 (corresponding points) or 2 (unit base)"
        )
    return ENDF_INT_TO_INTERPOLATION[base], ENDF_DECADE_TO_QUALIFIER[decade]


def joinEndfTab2Code(interpolation, qualifier=None) -> int:
    """The inverse of :func:`splitEndfTab2Code`.

    Raises on a qualifier ENDF cannot write. ``direct`` and
    ``correspondingEnergies`` are GNDS's and have no decade, and silently
    dropping one would write a file that states a different interpolation
    scheme from the one the model holds.
    """
    if qualifier is None:
        return INTERPOLATION_TO_ENDF_INT[Interpolation(interpolation)]
    qualifier = InterpolationQualifier(qualifier)
    if qualifier not in QUALIFIER_TO_ENDF_DECADE:
        raise ValueError(
            f"interpolationQualifier={qualifier.value!r} has no ENDF-6 "
            f"§0.5.2.1 decade; only {sorted(q.value for q in QUALIFIER_TO_ENDF_DECADE)} "
            f"can be written back into an INT code"
        )
    return (10 * QUALIFIER_TO_ENDF_DECADE[qualifier]
            + INTERPOLATION_TO_ENDF_INT[Interpolation(interpolation)])
