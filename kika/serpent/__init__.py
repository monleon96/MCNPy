# ──────────────────────────────────────────────────────────────────────────────
# File: serpent/__init__.py
# Public API for Serpent Monte Carlo code support.
# ──────────────────────────────────────────────────────────────────────────────

# Sensitivity output (.sens files)
from .sens import (
    PertCategory,
    Material,
    Nuclide,
    Perturbation,
    Response,
    SensitivitySet,
    SensitivityFile,
)
from .parse_sens import read_sensitivity_file, parse_sensitivity_text

# Input file parsing
from .input import (
    DetectorDefinition,
    EnergyGrid,
    SensitivityConfig,
    SensPerturbation,
    SensResponse,
    SerpentInput,
    TimeGrid,
)
from .parse_input import read_serpent_input

# Detector output (_det.m files)
from .detector import DetectorFile, DetectorResult
from .parse_det import read_detector_file, parse_detector_text

# Results output (_res.m files)
from .results import ResultsFile
from .parse_res import read_results_file, parse_results_text

# History output (_his.m files)
from .history import HistoryVariable, HistoryFile
from .parse_his import read_history_file, parse_history_text

# Depletion output (_dep.m files)
from .depletion import DepletionMaterial, DepletionFile
from .parse_dep import read_depletion_file, parse_depletion_text

__all__ = [
    # Sensitivity
    "PertCategory",
    "Material",
    "Nuclide",
    "Perturbation",
    "Response",
    "SensitivitySet",
    "SensitivityFile",
    "read_sensitivity_file",
    "parse_sensitivity_text",
    # Input
    "DetectorDefinition",
    "EnergyGrid",
    "SensitivityConfig",
    "SensPerturbation",
    "SensResponse",
    "SerpentInput",
    "TimeGrid",
    "read_serpent_input",
    # Detector output
    "DetectorFile",
    "DetectorResult",
    "read_detector_file",
    "parse_detector_text",
    # Results output
    "ResultsFile",
    "read_results_file",
    "parse_results_text",
    # History output
    "HistoryVariable",
    "HistoryFile",
    "read_history_file",
    "parse_history_text",
    # Depletion output
    "DepletionMaterial",
    "DepletionFile",
    "read_depletion_file",
    "parse_depletion_text",
]
