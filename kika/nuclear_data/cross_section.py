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
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

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


def _dominant_interpolation(regions: List[Tuple[int, int]]) -> str:
    """The scheme covering the most points, as a simplified name.

    ``regions`` is the raw ENDF ``(NBT, INT)`` list, where NBT is the 1-based
    index of the *last* point of the region — cumulative, not a per-region
    count (see ``_regionize`` in ``kika/endf/utils.py``). Region *i* therefore
    spans ``NBT[i] - NBT[i-1]`` points, with ``NBT[-1] = 0``.

    Taking ``max`` on NBT directly, as this used to, always picked the last
    region: NBT increases by construction. On a grid that is 99% lin-lin with
    a two-point histogram tail, that returned ``histogram``.

    Ties go to the earlier region. An empty list gives ``linlin``.
    """
    if not regions:
        return "linlin"

    widest_int, widest_span, previous_nbt = regions[0][1], -1, 0
    for nbt, int_code in regions:
        span = nbt - previous_nbt
        if span > widest_span:
            widest_span, widest_int = span, int_code
        previous_nbt = nbt

    return _ENDF_INTERP_TO_NAME.get(widest_int, "linlin")


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

        # Single scheme standing in for the whole grid, used only when the
        # per-region detail is unavailable — the regions themselves survive in
        # metadata below and win wherever they are present.
        interp_regions: List[Tuple[int, int]] = list(mf3mt.energy_interpolation)
        interp_name = _dominant_interpolation(interp_regions)

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

    def to_endf(
        self,
        mat: Optional[int] = None,
        *,
        qm: Optional[float] = None,
        qi: Optional[float] = None,
        lr: Optional[int] = None,
    ) -> "MF3MT":
        """Convert back to an ENDF ``MF3MT`` object.

        Parameters
        ----------
        mat : int, optional
            MAT number.  If *None*, uses value from ``metadata``.
        qm, qi : float, optional
            Mass-difference and reaction Q values, in eV. Override
            ``metadata``. Required when ``metadata`` carries neither.
        lr : int, optional
            Complex-breakup flag. Overrides ``metadata``.

        Returns
        -------
        MF3MT

        Raises
        ------
        ValueError
            If ``qm``/``qi``/``lr`` are neither given nor in ``metadata``.
            ACE records no reaction Q values, so a section built by
            :meth:`from_ace` always lands here — it used to default them to
            zero instead, writing a physically wrong MF3 header for every
            threshold reaction without saying so.
        """
        from kika.endf.classes.mf3.mf3mt import MF3MT

        mat = mat if mat is not None else self.metadata.get("mat")
        awr = self.metadata.get("awr", 0.0)  # ACE does carry this one

        overrides = {"qm": qm, "qi": qi, "lr": lr}
        missing = [
            key for key, value in overrides.items()
            if value is None and key not in self.metadata
        ]
        if missing:
            source = self.metadata.get("source_format", "unknown")
            raise ValueError(
                f"CrossSection for MT{self.reaction} (source_format={source!r}) "
                f"carries no {'/'.join(missing)}, so an ENDF MF3 header cannot "
                f"be written for it. ACE stores no reaction Q values. Pass them "
                f"explicitly — to_endf(qm=..., qi=..., lr=...) — or build the "
                f"section from ENDF, where they come from the file."
            )
        qm = qm if qm is not None else self.metadata["qm"]
        qi = qi if qi is not None else self.metadata["qi"]
        lr = lr if lr is not None else self.metadata["lr"]

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
        use_reconstructed: bool = True,
    ) -> "CrossSection":
        """Create a ``CrossSection`` from an ENDF object.

        Parameters
        ----------
        endf : ENDF
            Parsed ENDF file (must have MF3).
        mt : int
            Reaction MT number.
        use_reconstructed : bool
            Prefer ``endf.pendf[mt]`` when the caller has populated it, which
            is what carries resonance contributions on top of the MF3
            background. Falls back to raw MF3 when it is absent — for a
            threshold reaction the two are the same thing.

        Returns
        -------
        CrossSection

        Notes
        -----
        This used to reconstruct on demand, via an in-Python reconstructor
        documented as producing incorrect cross sections. It now only consumes
        what the caller chose::

            endf.pendf = kika.processing.njoy_reconstruct(path, njoy_executable=...)
        """
        if use_reconstructed and endf.pendf and mt in endf.pendf:
            return cls.from_endf(endf.pendf[mt])
        return cls.from_endf(endf.mf[3].mt[mt])

    @classmethod
    def all_from_endf_file(
        cls,
        endf: "ENDF",
        use_reconstructed: bool = True,
    ) -> Dict[int, "CrossSection"]:
        """Extract all available cross sections from an ENDF file.

        Parameters
        ----------
        endf : ENDF
            Parsed ENDF file.
        use_reconstructed : bool
            Prefer ``endf.pendf`` per MT where the caller has populated it.

        Returns
        -------
        Dict[int, CrossSection]
            Mapping of MT number to ``CrossSection``. An MT whose MF3 section
            cannot be converted is skipped.
        """
        result: Dict[int, "CrossSection"] = {}
        for mt_num in endf.mf[3].mt:
            try:
                result[mt_num] = cls.from_endf_file(
                    endf, mt_num, use_reconstructed=use_reconstructed
                )
            except (KeyError, AttributeError, ValueError, TypeError):
                # A section this converter cannot read, not a reason to lose
                # the rest. Narrower than the bare `except Exception` this
                # replaced, which also swallowed KeyboardInterrupt-adjacent
                # bugs in from_endf itself.
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
    # Evaluation
    # ------------------------------------------------------------------

    def get_cross_section(
        self,
        energy: Union[float, ArrayLike],
        out_of_range: str = "zero",
    ) -> Union[float, np.ndarray]:
        """Interpolate σ(E) at one or more energies.

        Symmetric with :meth:`MF3MT.get_cross_section`. When the underlying
        ENDF interpolation regions are preserved in
        ``metadata['interpolation_regions']`` (set by :meth:`from_endf`),
        the full ENDF interpolation law is honoured. Otherwise the
        ``interpolation`` attribute is used as a single-region scheme.

        Parameters
        ----------
        energy : float or array-like
            Query energy / energies in eV.
        out_of_range : str
            ``'zero'`` (default) or ``'hold'``.

        Returns
        -------
        float or np.ndarray
            Cross section(s) in barns.
        """
        from kika.endf.utils import interpolate_1d_endf  # local import: avoid cycle

        target = np.asarray(energy, dtype=float)
        scalar_input = np.ndim(target) == 0

        regions = self.metadata.get("interpolation_regions") if self.metadata else None
        if regions:
            result = interpolate_1d_endf(
                list(self.energies),
                list(self.values),
                list(regions),
                target,
                out_of_range=out_of_range,
            )
        else:
            # Fallback: single-region scheme from self.interpolation
            scheme = self.interpolation or "linlin"
            result = self._interp_single_scheme(target, scheme, out_of_range)

        return float(result) if scalar_input and np.ndim(result) == 0 else np.asarray(result, dtype=float)

    def _interp_single_scheme(
        self,
        target: np.ndarray,
        scheme: str,
        out_of_range: str,
    ) -> np.ndarray:
        """Interpolate using a single global scheme (no per-region INT codes)."""
        e = np.asarray(self.energies, dtype=float)
        v = np.asarray(self.values, dtype=float)
        out = np.zeros_like(np.atleast_1d(target), dtype=float)

        if scheme == "loglog":
            positive = (e > 0) & (v > 0)
            if np.all(positive):
                log_e = np.log(e)
                log_v = np.log(v)
                out = np.exp(np.interp(np.log(np.maximum(target, 1e-30)), log_e, log_v))
            else:
                out = np.interp(target, e, v)
        elif scheme == "linlog":  # σ linear in log E
            out = np.interp(np.log(np.maximum(target, 1e-30)), np.log(np.maximum(e, 1e-30)), v)
        elif scheme == "loglin":  # log σ linear in E
            positive = v > 0
            if np.all(positive):
                out = np.exp(np.interp(target, e, np.log(v)))
            else:
                out = np.interp(target, e, v)
        elif scheme == "histogram":
            idx = np.searchsorted(e, target, side='right') - 1
            idx = np.clip(idx, 0, len(v) - 1)
            out = v[idx]
        else:  # linlin or unknown → linear
            out = np.interp(target, e, v)

        # out_of_range handling for linear interp (np.interp clamps by default)
        if out_of_range == "zero":
            below = np.asarray(target) < e[0]
            above = np.asarray(target) > e[-1]
            out = np.where(below | above, 0.0, out)
        return out

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
