"""
Nuclear data processing — kika's "homemade NJOY".

Format-independent processing that operates on canonical types from
``kika.nuclear_data``.  Each submodule implements one processing step:

- ``reconr``          — resonance reconstruction (MF2 → pointwise σ)
- ``linearization``   — adaptive energy grid generation
- ``resonance_formulas`` — SLBW / MLBW / Reich-Moore cross-section formulas
- ``penetration``     — penetrability and shift factors
"""

from .reconr import reconr
from .multigroup import (
    WeightingFunction,
    compute_rebin_operator,
    collapse_covariance,
    relative_to_absolute,
    absolute_to_relative,
)

__all__ = [
    "reconr",
    "WeightingFunction",
    "compute_rebin_operator",
    "collapse_covariance",
    "relative_to_absolute",
    "absolute_to_relative",
]
