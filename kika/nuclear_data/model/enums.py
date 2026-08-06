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
