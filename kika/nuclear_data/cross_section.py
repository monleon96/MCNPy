"""
Format-agnostic pointwise cross section.

A ``CrossSection`` stores σ(E) as parallel energy and value arrays,
plus minimal physics metadata.  Format-specific details survive
round-trips via the ``metadata`` dict.

Pattern follows ``kika.cov.CrossSectionCovariance``: concrete dataclass with
``from_endf`` / ``to_endf`` classmethods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kika.ace.classes.ace import Ace
    from kika.endf.classes.endf import ENDF
    from kika.endf.classes.mf3.mf3mt import MF3MT
    from kika.plotting.plot_data import CrossSectionPlotData


# ENDF interpolation code → simplified name
_ENDF_INTERP_TO_NAME = {
    1: "histogram",
    2: "linlin",
    3: "linlog",
    4: "loglin",
    5: "loglog",
}

_NAME_TO_ENDF_INTERP = {v: k for k, v in _ENDF_INTERP_TO_NAME.items()}


@dataclass
class CrossSection:
    """Format-agnostic pointwise cross section σ(E).

    Attributes
    ----------
    energies : np.ndarray
        Energy grid in eV, sorted ascending.
    values : np.ndarray
        Cross-section values in barns.
    reaction : int
        Reaction identifier (MT number — shared by ENDF and GNDS).
    nuclide_id : int
        ZA = 1000*Z + A.
    temperature : float
        Temperature in Kelvin (0 = unbroadened / room-temperature).
    interpolation : str
        Simplified interpolation scheme: ``"linlin"``, ``"loglog"``,
        ``"linlog"``, ``"loglin"``, ``"histogram"``.
        When multiple regions use different schemes the dominant one is
        stored here; full ENDF region data lives in ``metadata``.
    metadata : dict
        Format-specific extras preserved for lossless round-trips.
        ENDF keys: ``"mat"``, ``"awr"``, ``"qm"``, ``"qi"``, ``"lr"``,
        ``"interpolation_regions"`` (list of ``(NBT, INT)`` tuples).
    """

    energies: np.ndarray
    values: np.ndarray
    reaction: int
    nuclide_id: int
    temperature: float = 0.0
    interpolation: str = "linlin"
    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # ENDF adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_endf(cls, mf3mt: "MF3MT") -> "CrossSection":
        """Create a ``CrossSection`` from an ENDF ``MF3MT`` object.

        Parameters
        ----------
        mf3mt : MF3MT
            Parsed MF3/MT section.

        Returns
        -------
        CrossSection
        """
        energies = np.asarray(mf3mt.energies, dtype=float)
        values = np.asarray(mf3mt.cross_sections, dtype=float)

        # Determine dominant interpolation scheme
        interp_regions: List[Tuple[int, int]] = list(mf3mt.energy_interpolation)
        if interp_regions:
            # Use the scheme that covers the most points
            dominant_int = max(interp_regions, key=lambda x: x[0])[1]
            interp_name = _ENDF_INTERP_TO_NAME.get(dominant_int, "linlin")
        else:
            interp_name = "linlin"

        return cls(
            energies=energies,
            values=values,
            reaction=mf3mt.number,
            nuclide_id=int(mf3mt.zaid) if mf3mt.zaid is not None else 0,
            interpolation=interp_name,
            metadata={
                "mat": getattr(mf3mt, "_mat", None),
                "awr": mf3mt.atomic_weight_ratio,
                "qm": mf3mt.q_mass_difference,
                "qi": mf3mt.q_reaction,
                "lr": mf3mt.breakup_flag,
                "interpolation_regions": interp_regions,
            },
        )

    def to_endf(self, mat: Optional[int] = None) -> "MF3MT":
        """Convert back to an ENDF ``MF3MT`` object.

        Parameters
        ----------
        mat : int, optional
            MAT number.  If *None*, uses value from ``metadata``.

        Returns
        -------
        MF3MT
        """
        from kika.endf.classes.mf3.mf3mt import MF3MT

        mat = mat if mat is not None else self.metadata.get("mat")
        awr = self.metadata.get("awr", 0.0)
        qm = self.metadata.get("qm", 0.0)
        qi = self.metadata.get("qi", 0.0)
        lr = self.metadata.get("lr", 0)
        interp_regions = self.metadata.get("interpolation_regions")

        if not interp_regions:
            int_code = _NAME_TO_ENDF_INTERP.get(self.interpolation, 2)
            interp_regions = [(len(self.energies), int_code)]

        mf3mt = MF3MT(number=self.reaction)
        mf3mt._za = float(self.nuclide_id)
        mf3mt._awr = awr
        mf3mt._mat = mat
        mf3mt._qm = qm
        mf3mt._qi = qi
        mf3mt._lr = lr
        mf3mt._energies = list(self.energies)
        mf3mt._cross_sections = list(self.values)
        mf3mt._np = len(self.energies)
        mf3mt._nr = len(interp_regions)
        mf3mt._interpolation = list(interp_regions)

        return mf3mt

    # ------------------------------------------------------------------
    # ENDF file-level adapter (with optional resonance reconstruction)
    # ------------------------------------------------------------------

    @classmethod
    def from_endf_file(
        cls,
        endf: "ENDF",
        mt: int,
        reconstruct: bool = True,
        tolerance: float = 1e-3,
    ) -> "CrossSection":
        """Create a ``CrossSection`` from an ENDF object, optionally reconstructing
        resonance cross sections from MF2 parameters.

        Parameters
        ----------
        endf : ENDF
            Parsed ENDF file (must have MF3; MF2 needed for reconstruction).
        mt : int
            Reaction MT number.
        reconstruct : bool
            If True and MF2 data is available, reconstruct pointwise cross
            sections via ``endf.reconstruct_xs()``.  Reconstructed data
            includes resonance contributions added to the MF3 background.
        tolerance : float
            Linearization tolerance passed to ``reconstruct_xs()``.

        Returns
        -------
        CrossSection
        """
        if reconstruct and 2 in endf.files:
            if endf.pendf is None:
                endf.reconstruct_xs(tolerance=tolerance)
            if mt in endf.pendf:
                return cls.from_endf(endf.pendf[mt])
        # Fall back to raw MF3 (threshold reactions, or reconstruct=False)
        return cls.from_endf(endf.mf[3].mt[mt])

    @classmethod
    def all_from_endf_file(
        cls,
        endf: "ENDF",
        reconstruct: bool = True,
        tolerance: float = 1e-3,
    ) -> Dict[int, "CrossSection"]:
        """Extract all available cross sections from an ENDF file.

        Parameters
        ----------
        endf : ENDF
            Parsed ENDF file.
        reconstruct : bool
            If True, reconstruct resonance cross sections from MF2.
        tolerance : float
            Linearization tolerance for reconstruction.

        Returns
        -------
        Dict[int, CrossSection]
            Mapping of MT number to ``CrossSection``.
        """
        if reconstruct and 2 in endf.files:
            if endf.pendf is None:
                endf.reconstruct_xs(tolerance=tolerance)

        result: Dict[int, "CrossSection"] = {}
        for mt_num in endf.mf[3].mt:
            try:
                result[mt_num] = cls.from_endf_file(
                    endf, mt_num, reconstruct=reconstruct, tolerance=tolerance
                )
            except Exception:
                continue
        return result

    # ------------------------------------------------------------------
    # ACE adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_ace(cls, ace: "Ace", mt: int) -> "CrossSection":
        """Create a ``CrossSection`` from an ACE object for a given MT.

        Parameters
        ----------
        ace : Ace
            Parsed ACE file.
        mt : int
            Reaction MT number (supports composites like MT=4, 18, 101).

        Returns
        -------
        CrossSection

        Raises
        ------
        ValueError
            If the requested MT is not available.
        """
        from kika._constants import BOLTZMANN_CONSTANT

        reaction = ace.cross_section._get_or_compute_reaction(mt)
        if reaction is None:
            raise ValueError(
                f"MT={mt} not available in ACE file "
                f"(ZAID={ace.header.zaid})"
            )

        energies = np.asarray(reaction.energies, dtype=float) * 1e6  # MeV → eV
        values = np.asarray(reaction.xs_values, dtype=float)

        kT_MeV = ace.header.temperature or 0.0
        temperature = kT_MeV / BOLTZMANN_CONSTANT if kT_MeV else 0.0

        return cls(
            energies=energies,
            values=values,
            reaction=mt,
            nuclide_id=ace.header.zaid or 0,
            temperature=temperature,
            interpolation="linlin",
            metadata={
                "source_format": "ace",
                "ace_zaid": ace.header.zaid,
                "ace_extension": ace.header.extension,
                "awr": ace.header.atomic_weight_ratio,
                "ace_comment": ace.header.comment,
                "ace_date": ace.header.date,
            },
        )

    @classmethod
    def all_from_ace(
        cls, ace: "Ace", include_composites: bool = True
    ) -> Dict[int, "CrossSection"]:
        """Extract all available cross sections from an ACE file.

        Parameters
        ----------
        ace : Ace
            Parsed ACE file.
        include_composites : bool
            If True, also attempt to compute composite MTs (4, 18, 101, etc.).

        Returns
        -------
        Dict[int, CrossSection]
            Mapping of MT number to ``CrossSection``.
        """
        from kika._constants import MT_COMPOSITE_ORDER

        result: Dict[int, "CrossSection"] = {}

        # Direct reactions
        for mt in ace.cross_section.mt_numbers:
            try:
                result[mt] = cls.from_ace(ace, mt)
            except (ValueError, Exception):
                continue

        # Composite reactions
        if include_composites:
            for mt in MT_COMPOSITE_ORDER:
                if mt in result:
                    continue
                try:
                    result[mt] = cls.from_ace(ace, mt)
                except (ValueError, Exception):
                    continue

        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def to_plot_data(
        self, label: Optional[str] = None, **styling_kwargs
    ) -> "CrossSectionPlotData":
        """Create a ``CrossSectionPlotData`` for this cross section.

        Parameters
        ----------
        label : str, optional
            Plot label.
        **styling_kwargs
            Passed to ``CrossSectionPlotData`` (color, linestyle, ...).

        Returns
        -------
        CrossSectionPlotData
        """
        from kika.plotting import CrossSectionPlotData
        from kika._utils import zaid_to_symbol

        isotope = zaid_to_symbol(self.nuclide_id) if self.nuclide_id else None

        if label is None:
            label = f"MT={self.reaction}"
            if isotope:
                label = f"{isotope} {label}"

        return CrossSectionPlotData(
            x=self.energies,
            y=self.values,
            isotope=isotope,
            mt=self.reaction,
            energy_range=(
                (float(self.energies.min()), float(self.energies.max()))
                if len(self) > 0
                else None
            ),
            data_source="canonical",
            label=label,
            **styling_kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.energies)

    def __repr__(self) -> str:
        return (
            f"CrossSection(MT={self.reaction}, ZA={self.nuclide_id}, "
            f"n_points={len(self)}, "
            f"E=[{self.energies[0]:.4g}, {self.energies[-1]:.4g}] eV)"
            if len(self) > 0
            else f"CrossSection(MT={self.reaction}, ZA={self.nuclide_id}, empty)"
        )
