"""
Nuclear data processing — kika's "homemade NJOY".

Format-independent processing that operates on canonical types from
``kika.nuclear_data``.  Each submodule implements one processing step:

- ``reconstruct``     — resonance reconstruction (MF2 → pointwise σ)
- ``linearization``   — adaptive energy grid generation
- ``resonance_formulas`` — SLBW / MLBW / Reich-Moore cross-section formulas
- ``penetration``     — penetrability and shift factors
"""

from .reconstruct import reconstruct
from .njoy_reconstruct import (
    NjoyReconstructError,
    njoy_reconstruct,
    njoy_reconstruct_stream,
)
from .multigroup import (
    WeightingFunction,
    compute_rebin_operator,
    collapse_covariance,
    relative_to_absolute,
    absolute_to_relative,
)
from .group_averaging import resonance_group_average
from .derived_covariance import resolve_derived_covariance
from .njoy_pendf_cache import (
    DEFAULT_PENDF_CACHE_DIR,
    mf33_needs_pendf,
    get_or_create_pendf,
    read_pendf_mf3_sections,
)

__all__ = [
    "reconstruct",
    "njoy_reconstruct",
    "njoy_reconstruct_stream",
    "NjoyReconstructError",
    "WeightingFunction",
    "compute_rebin_operator",
    "collapse_covariance",
    "relative_to_absolute",
    "absolute_to_relative",
    "resonance_group_average",
    "resolve_derived_covariance",
    "DEFAULT_PENDF_CACHE_DIR",
    "mf33_needs_pendf",
    "get_or_create_pendf",
    "read_pendf_mf3_sections",
]
