"""Data models for a parsed Serpent input file.

Provides dataclasses that represent the key cards of a Serpent input:
materials, detectors, energy grids, time grids, and sensitivity configuration.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnergyGrid:
    """Energy binning definition (``ene`` card)."""
    name: str
    grid_type: int  # 1=arbitrary, 2=uniform linear, 3=uniform log, 4=predefined
    boundaries: np.ndarray  # energy bin edges in MeV

    @property
    def n_bins(self) -> int:
        return max(0, len(self.boundaries) - 1)

    def __repr__(self) -> str:
        return f"EnergyGrid('{self.name}', type={self.grid_type}, bins={self.n_bins})"


@dataclass
class TimeGrid:
    """Time binning definition (``tme`` card)."""
    name: str
    grid_type: int  # 1=arbitrary, 2=uniform linear, 3=uniform log
    boundaries: np.ndarray  # seconds

    @property
    def n_bins(self) -> int:
        return max(0, len(self.boundaries) - 1)

    def __repr__(self) -> str:
        return f"TimeGrid('{self.name}', type={self.grid_type}, bins={self.n_bins})"


@dataclass
class DetectorDefinition:
    """Detector definition (``det`` card) — core options only."""
    name: str
    particle: str = "n"
    energy_grid: Optional[str] = None   # name of ``ene`` grid (de option)
    time_grid: Optional[str] = None     # name of ``tme`` grid (di option)
    volume: Optional[float] = None      # dv option
    materials: List[str] = field(default_factory=list)   # dm options
    cells: List[int] = field(default_factory=list)       # dc options
    response_mt: List[int] = field(default_factory=list) # dr option MT numbers
    extra_options: Dict[str, Any] = field(default_factory=dict)  # unparsed options

    def __repr__(self) -> str:
        parts = [f"DetectorDefinition('{self.name}'"]
        if self.energy_grid:
            parts.append(f"ene='{self.energy_grid}'")
        if self.time_grid:
            parts.append(f"tme='{self.time_grid}'")
        if self.materials:
            parts.append(f"mats={self.materials}")
        return ", ".join(parts) + ")"


@dataclass
class SensResponse:
    """A single ``sens resp`` sub-card."""
    response_type: str  # "keff", "leff", "beff", "lambda", "void", "ratio"
    name: Optional[str] = None       # user-given name (for ratio)
    det1: Optional[str] = None       # detector 1 (for ratio)
    det2: Optional[str] = None       # detector 2 (for ratio)
    material: Optional[str] = None   # material name (for void)

    def __repr__(self) -> str:
        if self.response_type == "ratio" and self.det1 and self.det2:
            return f"SensResponse('ratio', {self.name}: {self.det1}/{self.det2})"
        if self.response_type == "void" and self.material:
            return f"SensResponse('void', mat='{self.material}')"
        return f"SensResponse('{self.response_type}')"


@dataclass
class SensPerturbation:
    """A single ``sens pert`` sub-card."""
    pert_type: str  # "xs", "chi", "nubar", "eleg", "temperature", "elamu", "inlmu"
    mode: Optional[str] = None        # for xs: "all", "allmt", "realist"
    nuclides: List[str] = field(default_factory=list)
    legendre_order: Optional[int] = None  # for eleg

    def __repr__(self) -> str:
        if self.pert_type == "xs":
            return f"SensPerturbation('xs', mode='{self.mode}', nuclides={len(self.nuclides)})"
        return f"SensPerturbation('{self.pert_type}')"


@dataclass
class SensitivityConfig:
    """All ``sens`` cards combined."""
    responses: List[SensResponse] = field(default_factory=list)
    perturbations: List[SensPerturbation] = field(default_factory=list)
    energy_grid: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"SensitivityConfig(responses={len(self.responses)}, "
            f"perturbations={len(self.perturbations)})"
        )


@dataclass
class SerpentInput:
    """Top-level container for a parsed Serpent input file."""
    materials: Any = None  # MaterialCollection from kika.materials
    detectors: Dict[str, DetectorDefinition] = field(default_factory=dict)
    energy_grids: Dict[str, EnergyGrid] = field(default_factory=dict)
    time_grids: Dict[str, TimeGrid] = field(default_factory=dict)
    sensitivity: Optional[SensitivityConfig] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = ["SerpentInput("]
        if self.materials:
            parts.append(f"  materials={len(self.materials)}")
        parts.append(f"  detectors={len(self.detectors)}")
        parts.append(f"  energy_grids={len(self.energy_grids)}")
        parts.append(f"  time_grids={len(self.time_grids)}")
        if self.sensitivity:
            parts.append(f"  sensitivity={self.sensitivity}")
        parts.append(")")
        return "\n".join(parts)

    def summary(self) -> str:
        """Compact text summary."""
        lines = []
        if self.title:
            lines.append(f"Title: {self.title}")
        if self.materials:
            lines.append(f"Materials: {len(self.materials)}")
        if self.detectors:
            lines.append(f"Detectors: {', '.join(self.detectors.keys())}")
        if self.energy_grids:
            for name, eg in self.energy_grids.items():
                lines.append(f"Energy grid '{name}': {eg.n_bins} bins, type {eg.grid_type}")
        if self.time_grids:
            for name, tg in self.time_grids.items():
                lines.append(f"Time grid '{name}': {tg.n_bins} bins, type {tg.grid_type}")
        if self.sensitivity:
            lines.append(f"Sensitivity: {len(self.sensitivity.responses)} responses, "
                         f"{len(self.sensitivity.perturbations)} perturbation types")
        return "\n".join(lines)
