"""
Format-agnostic resonance parameters.

Stores resolved resonance data in a structure that can be consumed
by processing functions (e.g. ``kika.processing.reconstruct``) without
knowledge of the original file format.

The ``ResonanceRecord`` dataclass mirrors the physics of a single
resonance level and uses the same field naming conventions as
``kika.endf.classes.mf2.mf2mt151.Resonance`` so that existing
formula functions work with either type via duck typing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kika.endf.classes.mf2.mf2mt151 import MF2MT151


# Mapping from ENDF LRF code to formalism name
_LRF_TO_NAME = {1: "SLBW", 2: "MLBW", 3: "RM", 7: "RML"}
_NAME_TO_LRF = {v: k for k, v in _LRF_TO_NAME.items()}


@dataclass
class ResonanceRecord:
    """Single resonance level.

    Field semantics match the ENDF ``Resonance`` dataclass so that
    formula functions (``slbw_cross_sections``, etc.) accept either type.

    ========  ===============  ===============  ===================
    Field     SLBW (LRF=1)     MLBW (LRF=2)     Reich-Moore (LRF=3)
    ========  ===============  ===============  ===================
    c3        GT (total Γ)     GT (total Γ)     GN (neutron Γ)
    c4        GN (neutron Γ)   GN (neutron Γ)   GG (gamma Γ)
    c5        GG (gamma Γ)     GG (gamma Γ)     GFA (fission A Γ)
    c6        GF (fission Γ)   GF (fission Γ)   GFB (fission B Γ)
    ========  ===============  ===============  ===================
    """

    energy: float  # Resonance energy (eV)
    spin: float  # J value
    c3: float
    c4: float
    c5: float
    c6: float


@dataclass
class LGroup:
    """Resonances grouped by orbital angular momentum *l*.

    Mirrors ``kika.endf.classes.mf2.mf2mt151.LValueBlock``.
    """

    awri: float  # Isotope mass ratio
    l: int  # Orbital angular momentum
    resonances: List[ResonanceRecord] = field(default_factory=list)
    #: Scattering radius for this *l* alone, in fm, when the evaluation gives
    #: one (ENDF's APL, Reich-Moore only). *None* means "use the range-level
    #: radius". Already resolved against the formalism by the adapter, so
    #: unlike ``LValueBlock.apl_or_qx`` this is never a Q value.
    ap: Optional[float] = None


@dataclass
class ResonanceParameters:
    """Format-agnostic resonance parameters for one nuclide.

    Attributes
    ----------
    nuclide_id : int
        ZA = 1000*Z + A.
    spin : float
        Target spin (I).
    scattering_radius : float
        Scattering radius in ENDF units (10^{-12} cm).
    formalism : str
        ``"SLBW"``, ``"MLBW"``, ``"RM"``, or ``"RML"``.
    energy_range : Tuple[float, float]
        ``(E_lo, E_hi)`` in eV.
    l_groups : List[LGroup]
        Resonances grouped by orbital angular momentum.
    metadata : dict
        Format-specific extras (``"mat"``, ``"awr"``, ``"nlsc"``, etc.).
    """

    nuclide_id: int
    spin: float
    scattering_radius: float
    formalism: str
    energy_range: Tuple[float, float]
    l_groups: List[LGroup] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    scattering_radius_table: Optional[Tuple[np.ndarray, np.ndarray, List]] = None
    # (energies_eV, ap_values, interp_regions) when NRO=1; None = constant AP

    # ------------------------------------------------------------------
    # ENDF adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_endf(cls, mf2: "MF2MT151") -> List[Union["ResonanceParameters", "UnresolvedResonanceParameters"]]:
        """Create ``ResonanceParameters`` from an ENDF ``MF2MT151``.

        **Phase 3d: this reads the file through the GNDS model.** The section is
        decoded once, into ``BreitWigner``/``RMatrix``/``TabulatedWidths`` nodes
        by formalism, and ``model.interop`` projects each region back into the
        fields this class has always had. Same fields, same order, same
        defaults; ``test_flat_class_surface.py`` is the proof and
        ``test_facade_reproduces_the_old_bodies.py`` compares the values.

        **LRF=7 still yields nothing, and still warns.** ``ResonanceRecord`` has
        four width columns; an R-Matrix-Limited spin group has one width per
        *channel*, five of them for Fe-57 in JEFF-4.0. There is nothing to
        project into, so the range is omitted — but loudly, which is the half of
        ``docs/library/library-gaps.md`` D3 that is fixed. The model keeps the region;
        ask ``kika.endf.model_adapter.decodeMF2MT151`` for it.

        Returns a *list* because MF2 may contain multiple isotopes and energy
        ranges, each with a different formalism. Unresolved ranges (LRU=2) come
        back as ``UnresolvedResonanceParameters`` entries in the same list.

        Parameters
        ----------
        mf2 : MF2MT151
            Parsed MF2/MT151 section.

        Returns
        -------
        List[Union[ResonanceParameters, UnresolvedResonanceParameters]]
            One entry per (isotope, energy range) combination.
        """
        from kika.endf.model_adapter import decodeMF2MT151
        from kika.nuclear_data.model.interop import (
            flatResonanceParameters, flatUnresolvedResonanceParameters,
        )

        resonances, provenance, _ = decodeMF2MT151(mf2)
        regions = provenance.headerFields.get("regions", [])
        resolved = iter(resonances.resolved)

        results: List[Union[ResonanceParameters, UnresolvedResonanceParameters]] = []
        for fields in regions:
            if fields["kind"] in ("radiusOnly", "unsupported"):
                # Neither contributes a resolved region, so neither may consume
                # one from the iterator below. Since B1a both kinds *appear* in
                # ``regions`` — an LRU=0 range used to be dropped before the
                # append, which is why this skip did not exist before and why
                # its absence was invisible.
                continue
            if fields["kind"] == "unresolved":
                if resonances.unresolved is None:
                    continue
                results.append(UnresolvedResonanceParameters(
                    **_urr_groups(flatUnresolvedResonanceParameters(
                        resonances.unresolved, fields, provenance))
                ))
                continue

            region = next(resolved, None)
            if region is None:
                continue
            projected = flatResonanceParameters(region, fields, provenance)
            if projected is None:
                warnings.warn(
                    f"MF2/151 LRU=1 LRF={fields.get('lrf')}: this resolved range "
                    f"uses a formalism ResonanceParameters cannot represent, so it "
                    f"is omitted and the result describes a nuclide with fewer "
                    f"resonance regions than the file has. Use "
                    f"kika.endf.model_adapter.decodeMF2MT151 for LRF=7.",
                    UserWarning, stacklevel=2,
                )
                continue
            projected["l_groups"] = [
                LGroup(awri=g["awri"], l=g["l"], ap=g["ap"],
                       resonances=[ResonanceRecord(*record) for record in g["records"]])
                for g in projected["l_groups"]
            ]
            results.append(cls(**projected))

        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_resonances(self) -> int:
        """Total number of resonance levels across all l-groups."""
        return sum(len(lg.resonances) for lg in self.l_groups)

    def __repr__(self) -> str:
        return (
            f"ResonanceParameters(ZA={self.nuclide_id}, "
            f"formalism={self.formalism!r}, "
            f"E=[{self.energy_range[0]:.4g}, {self.energy_range[1]:.4g}] eV, "
            f"n_resonances={self.num_resonances})"
        )


# ======================================================================
# Unresolved resonance region (URR) — canonical types
# ======================================================================

@dataclass
class URR_JGroup:
    """Average parameters for one J-state in the URR."""
    j: float
    amun: float = 1.0                      # degrees of freedom, neutron
    d: Union[float, np.ndarray] = 0.0      # level spacing (eV)
    gn0: Union[float, np.ndarray] = 0.0    # average reduced neutron width (eV)
    gg: Union[float, np.ndarray] = 0.0     # average gamma width (eV)
    gf: Union[float, np.ndarray] = 0.0     # average fission width (eV)
    gx: Union[float, np.ndarray] = 0.0     # average competitive width (eV)


@dataclass
class URR_LGroup:
    """URR parameters for one l-value."""
    awri: float
    l: int
    j_groups: List[URR_JGroup] = field(default_factory=list)


@dataclass
class UnresolvedResonanceParameters:
    """Format-agnostic URR parameters for one nuclide/energy range.

    For energy-independent parameters (Case A), d/gn0/gg/gf/gx are scalars.
    For energy-dependent parameters (Cases B, C), they are arrays that must
    be interpolated to the evaluation energy grid before calling formulas.
    The energy_grid field stores the tabulation energies for interpolation.
    """
    nuclide_id: int
    spin: float
    scattering_radius: float
    energy_range: Tuple[float, float]
    lssf: int                                    # 0=compute XS, 1=self-shielding only
    l_groups: List[URR_LGroup] = field(default_factory=list)
    energy_grid: Optional[np.ndarray] = None     # None for Case A (energy-independent)
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"UnresolvedResonanceParameters(ZA={self.nuclide_id}, "
            f"LSSF={self.lssf}, "
            f"E=[{self.energy_range[0]:.4g}, {self.energy_range[1]:.4g}] eV)"
        )


def _urr_groups(projected: Dict) -> Dict:
    """Turn ``interop``'s plain dicts into ``URR_LGroup``/``URR_JGroup``.

    ``model.interop`` may not import these classes — it is below them — so it
    returns dicts and the conversion happens here, where they are in scope.
    """
    projected = dict(projected)
    projected["l_groups"] = [
        URR_LGroup(
            awri=group["awri"], l=group["l"],
            j_groups=[URR_JGroup(**state) for state in group["j_groups"]],
        )
        for group in projected["l_groups"]
    ]
    return projected


# ======================================================================
# URR ENDF converter (module-level, called from ResonanceParameters.from_endf)
# ======================================================================

def _urr_from_endf(er, za, awr, mat, abn):
    """Convert an ENDF unresolved energy range to UnresolvedResonanceParameters."""
    from kika.endf.classes.mf2.mf2mt151 import (
        UnresolvedCaseA, UnresolvedCaseB, UnresolvedCaseC,
    )
    p = er.parameters
    if p is None:
        return None

    energy_grid = None
    l_groups = []

    if isinstance(p, UnresolvedCaseA):
        for lv in p.l_values:
            jgs = [
                URR_JGroup(j=js.aj, amun=js.amun,
                           d=js.d, gn0=js.gn0, gg=js.gg)
                for js in lv.j_states
            ]
            l_groups.append(URR_LGroup(awri=lv.awri, l=lv.l, j_groups=jgs))

    elif isinstance(p, UnresolvedCaseB):
        energy_grid = np.asarray(p.energies, dtype=float)
        for lv in p.l_values:
            jgs = []
            for js in lv.j_states:
                gf_arr = np.asarray(js.gf, dtype=float) if js.gf else np.zeros(len(energy_grid))
                jgs.append(URR_JGroup(
                    j=js.aj, amun=js.amun,
                    d=js.d, gn0=js.gn0, gg=js.gg,
                    gf=gf_arr,
                ))
            l_groups.append(URR_LGroup(awri=lv.awri, l=lv.l, j_groups=jgs))

    elif isinstance(p, UnresolvedCaseC):
        for lv in p.l_values:
            jgs = []
            for js in lv.j_states:
                eps = js.energy_points
                es = np.array([ep.es for ep in eps], dtype=float)
                jgs.append(URR_JGroup(
                    j=js.aj, amun=js.amun,
                    d=np.array([ep.d for ep in eps]),
                    gn0=np.array([ep.gn0 for ep in eps]),
                    gg=np.array([ep.gg for ep in eps]),
                    gf=np.array([ep.gf for ep in eps]),
                    gx=np.array([ep.gx for ep in eps]),
                ))
                energy_grid = es  # same grid for all J-states in Case C
            l_groups.append(URR_LGroup(awri=lv.awri, l=lv.l, j_groups=jgs))
    else:
        return None

    return UnresolvedResonanceParameters(
        nuclide_id=za,
        spin=p.spi,
        scattering_radius=p.ap,
        energy_range=(er.el, er.eh),
        lssf=p.lssf,
        l_groups=l_groups,
        energy_grid=energy_grid,
        metadata={"mat": mat, "awr": awr, "abundance": abn},
    )
