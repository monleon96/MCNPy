"""Data models for parsed Serpent depletion output files (_dep.m).

Serpent writes burnup-dependent isotope inventories in MATLAB format::

    ZAI = [ 922350 922380 ... 0 ];
    NAMES = [ 'U-235  ' 'U-238  ' ... 'total' ];

    MAT_fuel_ADENS = [
      val1 val2 ... valN   %% U-235
      val1 val2 ... valN   %% U-238
      ...
      val1 val2 ... valN   %% lost
      val1 val2 ... valN   %% total
    ];

    BU = [ bu1 bu2 ... buN ];
    DAYS = [ d1 d2 ... dN ];

Each material property array has shape ``(n_isotopes + 2, n_burnup_steps)``
where the last two rows are "lost" and "total".
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Standard depletion properties written by Serpent
DEP_PROPERTIES = [
    "VOLUME", "BURNUP", "ADENS", "MDENS",
    "A", "H", "SF", "GSRC", "ING_TOX", "INH_TOX",
]

# Human-readable labels
DEP_PROPERTY_LABELS = {
    "VOLUME": "Volume",
    "BURNUP": "Burnup",
    "ADENS": "Atom density",
    "MDENS": "Mass density",
    "A": "Activity",
    "H": "Decay heat",
    "SF": "Spontaneous fission",
    "GSRC": "Gamma source",
    "ING_TOX": "Ingestion toxicity",
    "INH_TOX": "Inhalation toxicity",
}


@dataclass
class DepletionMaterial:
    """Burnup-dependent data for a single material."""
    name: str
    properties: Dict[str, np.ndarray] = field(default_factory=dict)
    # Each property → shape (n_isotopes + 2, n_steps)
    # Last two rows: "lost" and "total"

    @property
    def available_properties(self) -> List[str]:
        return sorted(self.properties.keys())

    def get_isotope_evolution(
        self, prop: str, isotope_idx: int
    ) -> Optional[np.ndarray]:
        """Time evolution of a single isotope, shape ``(n_steps,)``."""
        arr = self.properties.get(prop)
        if arr is None or isotope_idx >= arr.shape[0]:
            return None
        return arr[isotope_idx, :]

    def get_total(self, prop: str) -> Optional[np.ndarray]:
        """Total (last row) for a property, shape ``(n_steps,)``."""
        arr = self.properties.get(prop)
        if arr is None:
            return None
        return arr[-1, :]

    def get_lost(self, prop: str) -> Optional[np.ndarray]:
        """Lost (second-to-last row), shape ``(n_steps,)``."""
        arr = self.properties.get(prop)
        if arr is None:
            return None
        return arr[-2, :]

    def __repr__(self) -> str:
        props = ", ".join(self.available_properties)
        return f"DepletionMaterial('{self.name}', properties=[{props}])"


@dataclass
class DepletionFile:
    """All depletion data from a Serpent ``_dep.m`` file."""
    zai: List[int] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    materials: Dict[str, DepletionMaterial] = field(default_factory=dict)
    totals: Dict[str, np.ndarray] = field(default_factory=dict)
    burnup: np.ndarray = field(default_factory=lambda: np.array([]))
    days: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_isotopes(self) -> int:
        """Number of tracked isotopes (excluding 'lost' and 'total')."""
        return len(self.zai)

    @property
    def n_burnup_steps(self) -> int:
        return len(self.burnup)

    @property
    def material_names(self) -> List[str]:
        return sorted(self.materials.keys())

    @property
    def n_materials(self) -> int:
        return len(self.materials)

    def zai_to_symbol(self, zai: int) -> str:
        """Convert ZAI code to element symbol (e.g., 922350 → U-235)."""
        if zai == 0:
            return "total"
        z = zai // 10000
        a = (zai % 10000) // 10
        # Simple Z → symbol mapping for common elements
        symbols = {
            1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
            8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al",
            14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
            20: "Ca", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
            29: "Cu", 30: "Zn", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb",
            38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
            44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In",
            50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs",
            56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm",
            62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho",
            68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta",
            74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au",
            80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 88: "Ra",
            89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93: "Np", 94: "Pu",
            95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
        }
        sym = symbols.get(z, f"Z{z}")
        return f"{sym}-{a}"

    def __repr__(self) -> str:
        return (
            f"DepletionFile({self.n_materials} materials, "
            f"{self.n_isotopes} isotopes, "
            f"{self.n_burnup_steps} burnup steps)"
        )
