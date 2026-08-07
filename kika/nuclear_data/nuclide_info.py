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

        **Phase 3d: this reads the file through the GNDS model.** The section is
        decoded into ``PoPs`` + an ``evaluated`` style + ``EndfProvenance``, and
        ``model.interop`` projects that back into the five fields this class has
        always had. The fields, their order and their defaults are unchanged —
        only the body is — which is what makes ``test_flat_class_surface.py``
        the proof that nothing downstream can tell.

        One thing does change, and it is a fix: ZA is **rounded** rather than
        truncated. ENDF's fixed-format floats do not round-trip exactly, so
        Th-232's ``9.023200+4`` reads back as ``90231.99999999999`` and the old
        ``int()`` named Ac-231. See ``docs/library-gaps.md`` D1.

        Parameters
        ----------
        mt451 : MT451
            Parsed MF1/MT451 section.

        Returns
        -------
        NuclideInfo
        """
        from kika.endf.model_adapter import decodeMF1MT451
        from kika.nuclear_data.model.interop import flatNuclideInfo

        _, _, provenance, _ = decodeMF1MT451(mt451)
        return cls(**flatNuclideInfo(provenance))

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
