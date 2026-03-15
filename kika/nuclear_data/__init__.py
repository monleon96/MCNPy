"""
Format-agnostic canonical representations of nuclear data.

These classes serve as the hub in a hub-and-spoke architecture:
format-specific I/O (ENDF, GNDS, ...) converts to/from these types,
and processing operates on them directly.

The design follows the same pattern as ``kika.cov.CovMat``:
concrete dataclasses with ``from_endf`` / ``to_endf`` classmethods.
"""

from .cross_section import CrossSection
from .angular_distribution import AngularDistribution
from .resonance_parameters import (
    ResonanceParameters, ResonanceRecord, LGroup,
    UnresolvedResonanceParameters, URR_LGroup, URR_JGroup,
)
from .nuclide_info import NuclideInfo

__all__ = [
    "CrossSection",
    "AngularDistribution",
    "ResonanceParameters",
    "ResonanceRecord",
    "LGroup",
    "UnresolvedResonanceParameters",
    "URR_LGroup",
    "URR_JGroup",
    "NuclideInfo",
]
