"""
Uncertainty Quantification (UQ) module for KIKA.

This module provides tools for uncertainty propagation and analysis
in Monte Carlo neutron transport calculations.
"""

from .fastTMC import fastTMC, create_summary_table
from .sandwich import (
    sandwich_uncertainty_propagation,
    UncertaintyResult,
    UncertaintyContribution,
    filter_reactions_by_nuclide,
    filter_reactions_by_type
)
from .convergence import convergence_analysis
from .normality import normality_tests, histogram_data, qq_plot_data

__all__ = [
    'fastTMC',
    'create_summary_table',
    'sandwich_uncertainty_propagation',
    'UncertaintyResult',
    'UncertaintyContribution',
    'filter_reactions_by_nuclide',
    'filter_reactions_by_type',
    'convergence_analysis',
    'normality_tests',
    'histogram_data',
    'qq_plot_data',
]
