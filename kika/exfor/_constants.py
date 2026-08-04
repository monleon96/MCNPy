"""
EXFOR-specific constants for the kika library.
"""

# Valid angle frames
FRAME_LAB = "LAB"
FRAME_CM = "CM"
VALID_FRAMES = {FRAME_LAB, FRAME_CM}

# Valid angle units
ANGLE_UNIT_DEG = "deg"
ANGLE_UNIT_COS = "cos"
VALID_ANGLE_UNITS = {ANGLE_UNIT_DEG, ANGLE_UNIT_COS}

# Valid energy units
ENERGY_UNIT_EV = "eV"
ENERGY_UNIT_KEV = "keV"
ENERGY_UNIT_MEV = "MeV"
VALID_ENERGY_UNITS = {ENERGY_UNIT_EV, ENERGY_UNIT_KEV, ENERGY_UNIT_MEV}

# Valid cross section units
XS_UNIT_B_SR = "b/sr"
XS_UNIT_MB_SR = "mb/sr"
XS_UNIT_UB_SR = "ub/sr"
VALID_XS_UNITS = {XS_UNIT_B_SR, XS_UNIT_MB_SR, XS_UNIT_UB_SR}

# Unit conversion factors (to base units: MeV, b/sr, deg)
ENERGY_TO_MEV = {
    "eV": 1e-6,
    "keV": 1e-3,
    "MeV": 1.0,
}

XS_TO_B_SR = {
    "b/sr": 1.0,
    "mb/sr": 1e-3,
    "ub/sr": 1e-6,
}

# Schema version for standardized JSON format
SCHEMA_VERSION = "1.0.0"

# Neutron mass in atomic mass units.
# Re-exported from kika._constants so there is one definition; this used to
# carry the rounded 1.008665, which differs from the CODATA value by 8e-7
# relative. Callers that need the historical value should pass it explicitly
# (the EXFOR pipeline does, via its M_PROJ_U config knob).
from kika._constants import NEUTRON_MASS_AMU  # noqa: F401

# Energy matching tolerance (MeV) for grouping data by energy
ENERGY_MATCH_ABS_TOL = 1e-6

# Default absolute energy tolerance for plotting (MeV)
# Used in to_plot_data() for matching experimental data to requested energy
PLOTTING_ENERGY_TOLERANCE = 0.01  # 10 keV

# =============================================================================
# X4Pro Database Configuration
# =============================================================================

# Environment variable for database path
DB_PATH_ENV_VAR = "KIKA_X4PRO_DB_PATH"

# Default database path (None = must be set via env var or passed explicitly)
# Users should set KIKA_X4PRO_DB_PATH environment variable or pass db_path to functions
DB_DEFAULT_PATH = None

# Database unit conversion factors
# Maps database unit strings to conversion factors (to base unit)
DB_UNIT_MAPPINGS = {
    "energy": {
        # Energy units -> eV conversion factor
        "EV": 1.0,
        "KEV": 1e3,
        "MEV": 1e6,
        "GEV": 1e9,
    },
    "cross_section": {
        # Cross section units -> b/sr conversion factor
        "B/SR": 1.0,
        "MB/SR": 1e-3,
        "MUB/SR": 1e-6,  # microbarns
        "UB/SR": 1e-6,
    },
}

# Database family code mappings
DB_FAMILY_MAPPINGS = {
    "energy": ["EN", "E", "EN-CM"],
    "angle": ["ANG", "ANG-CM"],
    "cosine": ["COS", "COS-CM"],
    "cross_section": ["Data", "DATA"],
    "uncertainty": ["dData", "DATA-ERR", "ERR-T", "ERR-S"],
}

# Database quantity codes for angular distributions
DB_DA_QUANTITIES = ["DA", "DA,,RTH", "DA/DA", "DA/DE"]

# =============================================================================
# EXFOR Quantity Codes and Descriptions
# =============================================================================

# EXFOR database quantity codes (used in quant1 column) to descriptions
# These are the category codes used for filtering in the X4Pro database
EXFOR_QUANTITY_CODES = {
    # Cross section data
    "CS": "Cross section data",
    "CSP": "Partial cross section data",
    "CST": "Temperature dependent cross section data",
    "ALF": "Alpha (capture-to-fission ratio)",
    "ETA": "Eta (neutrons per non-elastic event)",

    # Differential data
    "DA": "Differential data with respect to angle",
    "DAE": "Differential data with respect to angle and energy",
    "DAP": "Partial differential data with respect to angle",
    "DAA": "Double differential angular distribution",
    "D3A": "Triple differential angular distribution",
    "D3E": "Triple differential energy distribution",
    "D4A": "Quadruple differential angular distribution",
    "DE": "Differential data with respect to energy",
    "DEP": "Partial differential data with respect to energy",
    "DT": "Time differential data",

    # Fission data
    "FY": "Fission product yields",
    "E": "Fission fragment energies",
    "MFQ": "Miscellaneous fission quantities",

    # Neutron data
    "NU": "Neutron multiplicity (nubar)",
    "NUD": "Delayed neutron data",
    "NUF": "Fission neutron data",
    "MLT": "Multiplicity",

    # Product yields
    "PY": "Product yields",
    "TT": "Thick target yields",
    "TTD": "Differential thick target yields",
    "TTP": "Partial thick target yields",

    # Other quantities
    "AMP": "Scattering amplitude",
    "CHG": "Charge distribution",
    "COR": "Secondary particle correlations",
    "DP": "Momentum distribution",
    "INT": "Integral quantities",
    "KE": "Kinetic energy",
    "KER": "KERMA factors",
    "L": "Scattering amplitudes",
    "MAS": "Mass distribution",
    "NQ": "Nuclear quantities",
    "POD": "Polarization differential data",
    "POL": "Polarization data",
    "POT": "Potential scattering",
    "RI": "Resonance integrals",
    "RP": "Resonance parameters",
    "RR": "Reaction rates",
    "SP": "Gamma spectra",
    "SPC": "Gamma spectra (detailed)",
    "TSL": "Thermal scattering law",
}

# Wildcard/combined quantity codes (match multiple types)
EXFOR_QUANTITY_WILDCARDS = {
    "CS*": "Cross section data (all types: CS, CSP, CST)",
    "DA*": "Differential data (all types: DA, DAE, DAP)",
    "TT*": "Thick target yields (all types: TT, TTD, TTP)",
}

# EXFOR URL base for linking to IAEA database
EXFOR_URL_BASE = "https://www-nds.iaea.org/exfor/"

# Quantity families for grouping related types
QUANTITY_FAMILIES = {
    "cross_section": ["CS", "CSP", "CST", "ALF", "ETA"],
    "angular": ["DA", "DAE", "DAP", "DAA", "D3A", "D4A"],
    "energy_spectrum": ["DE", "DEP", "D3E", "DT"],
    "fission": ["FY", "E", "MFQ"],
    "neutron_data": ["NU", "NUD", "NUF", "MLT"],
    "product_yields": ["PY", "TT", "TTD", "TTP"],
    "resonance": ["RI", "RP"],
    "polarization": ["POL", "POD"],
    "other": [
        "AMP", "CHG", "COR", "DP", "INT", "KE", "KER",
        "L", "MAS", "NQ", "POT", "RR", "SP", "SPC", "TSL",
    ],
}

# Family code to variable name mapping (for general data parsing)
EXFOR_FAMILY_TO_VARIABLE = {
    "EN": "energy",
    "E": "energy",
    "EN-CM": "energy_cm",
    "ANG": "angle",
    "ANG-CM": "angle_cm",
    "COS": "cosine",
    "COS-CM": "cosine_cm",
    "E2": "secondary_energy",
    "LVL": "level",
    "HL": "half_life",
    "MASS": "mass",
    "ZAM": "product_zam",
    "ELEM": "element",
    "ISOMER": "isomer",
}
