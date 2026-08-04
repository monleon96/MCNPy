"""Utility functions for KIKA."""
from kika.utils.logging_utils import configure_ace_debug_logging, configure_endf_debug_logging, get_endf_logger
from kika.utils.numerics import (
    gauss_hermite_nodes,
    fold_tabulated,
    average_over_intervals,
)
from kika.utils.energy_folding import (
    EnergyFoldingConfig,
    FWHM_TO_SIGMA,
    tof_energy_resolution,
    compute_energy_resolution_tof,
    fold_cross_section,
    fold_angular_distribution,
    endf_angular_distribution,
    compute_folded_differential_xs,
    compute_unfolded_differential_xs,
)

__all__ = [
    # Logging utilities
    'configure_ace_debug_logging',
    'configure_endf_debug_logging',
    'get_endf_logger',
    # Numerical primitives
    'gauss_hermite_nodes',
    'fold_tabulated',
    'average_over_intervals',
    # Energy folding utilities
    'EnergyFoldingConfig',
    'FWHM_TO_SIGMA',
    'tof_energy_resolution',
    'compute_energy_resolution_tof',
    'fold_cross_section',
    'fold_angular_distribution',
    'endf_angular_distribution',
    'compute_folded_differential_xs',
    'compute_unfolded_differential_xs',
]
