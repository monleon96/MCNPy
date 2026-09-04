"""
EXFOR (Experimental Nuclear Reaction Data) utilities for KIKA.

This module provides classes and functions for loading, processing, and plotting
experimental angular distribution data from EXFOR.

Quick Start:
    >>> import kika.exfor as exfor
    >>>
    >>> # Configure database path once at the start
    >>> exfor.configure(db_path="C:/Data/x4sqlite1.db")
    >>>
    >>> # Now load data without specifying path each time
    >>> data = exfor.read_all_exfor(target="Fe56", mt=2)
    >>> print(f"Found {len(data)} energy groups")

Primary API:
    - configure(): Set default database path for the session
    - read_all_exfor(): Load EXFOR data from database or JSON files
    - read_exfor(): Load a single EXFOR JSON file
    - ExforAngularDistribution: Main class for angular distribution data

Example - Loading from Database:
    >>> import kika.exfor as exfor
    >>> exfor.configure(db_path="/path/to/x4sqlite1.db")
    >>>
    >>> # Load Fe-56 elastic scattering data
    >>> data = exfor.read_all_exfor(target="Fe56", mt=2)
    >>>
    >>> # Load with energy filter
    >>> data = exfor.read_all_exfor(target="U235", mt=18, energy_range=(1.0, 5.0))

Example - Loading from JSON:
    >>> exfor_data = exfor.read_exfor('/path/to/27673002.json')
    >>> print(exfor_data.label)
    Gkatis et al. (2025)

Legacy API:
    The functions from AD_utils are still available but deprecated.
    Please migrate to the new class-based API.
"""

# Configuration (call configure() to set defaults for the session)
from kika.exfor.config import configure, get_config

# Constants
from kika.exfor._constants import (
    EXFOR_QUANTITY_CODES,
    EXFOR_QUANTITY_WILDCARDS,
    EXFOR_URL_BASE,
    QUANTITY_FAMILIES,
)

# Primary API - Main exports
from kika.exfor.io import read_exfor, read_all_exfor
from kika.exfor.exfor_entry import ExforEntry
from kika.exfor.angular_distribution import ExforAngularDistribution
from kika.exfor.cross_section import ExforCrossSection
from kika.exfor.experiment import ExforExperiment

# Database API
from kika.exfor.database import (
    X4ProDatabase,
    X4ProDataset,
    read_exfor_from_database,
    parse_reacode_fields,
    exfor_quantity,
    canonical_unit_factor,
)

# Transform functions
from kika.exfor.transforms import (
    cos_cm_from_cos_lab,
    cos_lab_from_cos_cm,
    jacobian_cm_to_lab,
    jacobian_lab_to_cm,
    transform_lab_to_cm,
    transform_cm_to_lab,
    angle_deg_to_cos,
    cos_to_angle_deg,
)

# Plotting functions
from kika.exfor.plotting import (
    plot_exfor_angular,
    plot_exfor_ace_comparison,
    plot_multiple_energies,
)

# Legacy API - Deprecated (imported for backward compatibility)
from kika.exfor.AD_utils import (
    # ACE Data Processing
    extract_angular_distribution,
    calculate_differential_cross_section,
    cosine_to_angle_degrees,
    angle_degrees_to_cosine,
    # EXFOR Data Processing (deprecated - use read_exfor instead)
    load_exfor_data,
    extract_experiment_info,
    load_all_exfor_data,
    # Plotting Functions (deprecated - use new plotting module)
    plot_angular_distribution,
    plot_combined_angular_distribution,
    plot_individual_energy_comparisons,
    plot_combined_angular_distribution_multi_ace,
    plot_all_energies_comparison,
)

__all__ = [
    # Configuration
    "configure",
    "get_config",
    # Constants
    "EXFOR_QUANTITY_CODES",
    "EXFOR_QUANTITY_WILDCARDS",
    "EXFOR_URL_BASE",
    "QUANTITY_FAMILIES",
    # Primary API
    "read_exfor",
    "read_all_exfor",
    "ExforEntry",
    "ExforAngularDistribution",
    "ExforCrossSection",
    "ExforExperiment",
    # Database API
    "X4ProDatabase",
    "parse_reacode_fields",
    "exfor_quantity",
    "canonical_unit_factor",
    "X4ProDataset",
    "read_exfor_from_database",
    # Transforms
    "cos_cm_from_cos_lab",
    "cos_lab_from_cos_cm",
    "jacobian_cm_to_lab",
    "jacobian_lab_to_cm",
    "transform_lab_to_cm",
    "transform_cm_to_lab",
    "angle_deg_to_cos",
    "cos_to_angle_deg",
    # Plotting
    "plot_exfor_angular",
    "plot_exfor_ace_comparison",
    "plot_multiple_energies",
    # Legacy API (deprecated)
    "extract_angular_distribution",
    "calculate_differential_cross_section",
    "cosine_to_angle_degrees",
    "angle_degrees_to_cosine",
    "load_exfor_data",
    "extract_experiment_info",
    "load_all_exfor_data",
    "plot_angular_distribution",
    "plot_combined_angular_distribution",
    "plot_individual_energy_comparisons",
    "plot_combined_angular_distribution_multi_ace",
    "plot_all_energies_comparison",
]
