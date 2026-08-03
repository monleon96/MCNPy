from .sensitivity_processing import compute_sensitivity, compute_total_sensitivity, create_sdf_data, plot_sens_comparison
from .sdf import SDFData, SDFReactionData, sensitivity_to_plot_data
from .sensitivity import SensitivityData, TaylorCoefficients, Coefficients
from .profile import SensitivityProfile, SensitivityReaction
from .condensation import (
    CondensationError,
    CondensationReport,
    CondensationResult,
    condense_sensitivity_profile,
)

__all__ = [
    # Core processing
    'compute_sensitivity', 'compute_total_sensitivity', 'plot_sens_comparison',
    'create_sdf_data',

    # Data classes
    'SDFData', 'SDFReactionData', 'SensitivityProfile', 'SensitivityReaction',
    'SensitivityData', 'TaylorCoefficients', 'Coefficients',

    # Explicit multigroup condensation
    'CondensationError', 'CondensationReport', 'CondensationResult',
    'condense_sensitivity_profile',

    # Plotting adapter
    'sensitivity_to_plot_data',
]
