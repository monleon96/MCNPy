"""
Format-agnostic nuclide identity and evaluation metadata.

Wraps the information in ENDF MF1/MT451 (general information).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kika.ace.classes.ace import Ace
    from kika.endf.classes.mf1.mf1mt451 import MF1MT451


@dataclass
class NuclideInfo:
    """Format-agnostic nuclide identity and evaluation metadata.

    Attributes
    ----------
    nuclide_id : int
        ZA = 1000*Z + A.
    atomic_weight_ratio : float
        Ratio of nucleus mass to neutron mass.
    temperature : float
        Target temperature in Kelvin (0 if unbroadened).
    evaluation_info : dict
        Human-readable evaluation metadata: ``"laboratory"``,
        ``"authors"``, ``"eval_date"``, ``"reference"``, etc.
    metadata : dict
        Format-specific extras for lossless round-trips.
    """

    nuclide_id: int = 0
    atomic_weight_ratio: float = 0.0
    temperature: float = 0.0
    evaluation_info: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # ENDF adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_endf(cls, mt451: "MF1MT451") -> "NuclideInfo":
        """Create from an ENDF ``MT451`` object.

        Parameters
        ----------
        mt451 : MT451
            Parsed MF1/MT451 section.

        Returns
        -------
        NuclideInfo
        """
        return cls(
            nuclide_id=int(mt451.zaid) if mt451.zaid is not None else 0,
            atomic_weight_ratio=mt451.atomic_weight_ratio or 0.0,
            temperature=mt451.temperature or 0.0,
            evaluation_info={
                "laboratory": mt451.laboratory,
                "authors": mt451.authors,
                "eval_date": mt451.eval_date,
                "reference": mt451.reference,
                "dist_date": mt451.dist_date,
                "revision_date": mt451.revision_date,
                "material_id": mt451.material_id,
            },
            metadata={
                "mat": getattr(mt451, "_mat", None),
                "lrp": getattr(mt451, "_lrp", None),
                "lfi": getattr(mt451, "_lfi", None),
                "nlib": mt451.library_id,
                "nmod": mt451.mod_number,
                "elis": mt451.excitation_energy,
                "sta": getattr(mt451, "_sta", None),
                "lis": mt451.state_number,
                "liso": mt451.isomer_number,
                "nfor": mt451.format_version,
                "awi": mt451.projectile_mass,
                "emax": mt451.energy_max,
                "lrel": mt451.library_release,
                "nsub": mt451.sublibrary,
                "nver": mt451.library_version,
                "ldrv": getattr(mt451, "_ldrv", None),
            },
        )

    # ------------------------------------------------------------------
    # ACE adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_ace(cls, ace: "Ace") -> "NuclideInfo":
        """Create from an ACE object.

        Parameters
        ----------
        ace : Ace
            Parsed ACE file.

        Returns
        -------
        NuclideInfo
        """
        from kika._constants import BOLTZMANN_CONSTANT

        kT_MeV = ace.header.temperature or 0.0
        temperature = kT_MeV / BOLTZMANN_CONSTANT if kT_MeV else 0.0

        return cls(
            nuclide_id=ace.header.zaid or 0,
            atomic_weight_ratio=ace.header.atomic_weight_ratio or 0.0,
            temperature=temperature,
            evaluation_info={
                "date": ace.header.date,
                "source": getattr(ace.header, "source", None),
                "reference": ace.header.comment,
            },
            metadata={
                "source_format": "ace",
                "ace_extension": ace.header.extension,
                "ace_matid": ace.header.matid,
                "ace_format_version": ace.header.format_version,
            },
        )

    def __repr__(self) -> str:
        return (
            f"NuclideInfo(ZA={self.nuclide_id}, "
            f"AWR={self.atomic_weight_ratio:.4f}, "
            f"T={self.temperature:.1f} K)"
        )
