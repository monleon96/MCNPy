"""
Classes for MF2/MT151 (Resonance Parameters) in ENDF files.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..mt import MT
from ...utils import (
    format_endf_data_line,
    format_data_values,
    format_tab1,
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    ENDF_FORMAT_INT_ZERO,
    ENDF_FORMAT_BLANK,
)


@dataclass
class Resonance:
    """Single resonance entry.

    Field meanings depend on the formalism (LRF) of the parent EnergyRange:

    ========  ===============  ===============  ===================
    Position  SLBW (LRF=1)     MLBW (LRF=2)     Reich-Moore (LRF=3)
    ========  ===============  ===============  ===================
    c3        GT (total width)  GT (total width)  GN (neutron width)
    c4        GN (neutron)      GN (neutron)      GG (gamma width)
    c5        GG (gamma)        GG (gamma)        GFA (fission A)
    c6        GF (fission)      GF (fission)      GFB (fission B)
    ========  ===============  ===============  ===================
    """
    energy: float       # ER (eV)
    spin: float         # AJ (J value)
    c3: float           # GT or GN  (see table above)
    c4: float           # GN or GG
    c5: float           # GG or GFA
    c6: float           # GF or GFB


@dataclass
class LValueBlock:
    """Resonances for a single orbital angular momentum *l*."""
    awri: float         # Isotope mass ratio
    l: int              # Orbital angular momentum quantum number
    num_resonances: int # NRS
    resonances: List[Resonance] = field(default_factory=list)


@dataclass
class ScatteringRadiusOnly:
    """Parameters for an energy range with LRU=0 (scattering radius only)."""
    spi: float          # Target spin
    ap: float           # Scattering radius (fm)


@dataclass
class ResolvedResonanceRange:
    """Parameters for an energy range with LRU=1 (resolved resonances)."""
    spi: float          # Target spin
    ap: float           # Scattering radius (fm)
    nls: int            # Number of l-values
    nlsc: int           # Number of l-values for convergence
    l_values: List[LValueBlock] = field(default_factory=list)


@dataclass
class EnergyDependentScatteringRadius:
    """TAB1 record for energy-dependent scattering radius (NRO=1)."""
    interpolation: List[Tuple[int, int]]  # (NBT, INT) pairs
    energies: List[float]                  # E values (eV)
    ap_values: List[float]                 # AP(E) (fm)


# ======================================================================
# Unresolved resonance dataclasses (LRU=2)
# ======================================================================

# --- Case A: LFW=0, LRF=1 (all energy-independent widths) ---

@dataclass
class URR_JState_CaseA:
    """J-state record for unresolved Case A."""
    d: float            # Average level spacing (eV)
    aj: float           # J value
    amun: float         # Degrees of freedom for neutron width
    gn0: float          # Average reduced neutron width (eV)
    gg: float           # Average gamma width (eV)


@dataclass
class URR_LValue_CaseA:
    """L-value block for unresolved Case A."""
    awri: float         # Isotope mass ratio
    l: int              # Orbital angular momentum
    j_states: List[URR_JState_CaseA] = field(default_factory=list)


@dataclass
class UnresolvedCaseA:
    """Unresolved resonance parameters: LFW=0, LRF=1 (energy-independent)."""
    spi: float          # Target spin
    ap: float           # Scattering radius (fm)
    lssf: int           # Self-shielding only/all flag
    nls: int            # Number of l-values
    l_values: List[URR_LValue_CaseA] = field(default_factory=list)


# --- Case B: LFW=1, LRF=1 (fission widths energy-dependent) ---

@dataclass
class URR_JState_CaseB:
    """J-state record for unresolved Case B."""
    d: float            # Average level spacing (eV)
    aj: float           # J value
    amun: float         # Degrees of freedom for neutron width
    gn0: float          # Average reduced neutron width (eV)
    gg: float           # Average gamma width (eV)
    muf: int            # Degrees of freedom for fission width
    gf: List[float] = field(default_factory=list)  # GF(E) values


@dataclass
class URR_LValue_CaseB:
    """L-value block for unresolved Case B."""
    awri: float         # Isotope mass ratio
    l: int              # Orbital angular momentum
    j_states: List[URR_JState_CaseB] = field(default_factory=list)


@dataclass
class UnresolvedCaseB:
    """Unresolved resonance parameters: LFW=1, LRF=1 (energy-dependent fission)."""
    spi: float          # Target spin
    ap: float           # Scattering radius (fm)
    lssf: int           # Self-shielding only/all flag
    ne: int             # Number of energy points
    nls: int            # Number of l-values
    energies: List[float] = field(default_factory=list)  # Energy grid (eV)
    l_values: List[URR_LValue_CaseB] = field(default_factory=list)


# --- Case C: LRF=2 (all energy-dependent) ---

@dataclass
class URR_EnergyPoint:
    """Single energy point in unresolved Case C."""
    es: float           # Energy (eV)
    d: float            # Average level spacing (eV)
    gx: float           # Average competitive width (eV)
    gn0: float          # Average reduced neutron width (eV)
    gg: float           # Average gamma width (eV)
    gf: float           # Average fission width (eV)


@dataclass
class URR_JState_CaseC:
    """J-state record for unresolved Case C."""
    aj: float           # J value
    int_code: int       # Interpolation code
    amux: float         # Degrees of freedom for competitive width
    amun: float         # Degrees of freedom for neutron width
    amug: float         # Degrees of freedom for gamma width
    amuf: float         # Degrees of freedom for fission width
    energy_points: List[URR_EnergyPoint] = field(default_factory=list)


@dataclass
class URR_LValue_CaseC:
    """L-value block for unresolved Case C."""
    awri: float         # Isotope mass ratio
    l: int              # Orbital angular momentum
    j_states: List[URR_JState_CaseC] = field(default_factory=list)


@dataclass
class UnresolvedCaseC:
    """Unresolved resonance parameters: LRF=2 (all energy-dependent)."""
    spi: float          # Target spin
    ap: float           # Scattering radius (fm)
    lssf: int           # Self-shielding only/all flag
    nls: int            # Number of l-values
    l_values: List[URR_LValue_CaseC] = field(default_factory=list)


# ======================================================================
# R-Matrix Limited dataclasses (LRU=1, LRF=7)
# ======================================================================

@dataclass
class RML_ParticlePair:
    """Particle pair definition for R-Matrix Limited."""
    ma: float           # Mass of particle a (neutron mass units)
    mb: float           # Mass of particle b
    za: float           # Charge of particle a
    zb: float           # Charge of particle b
    ia: float           # Spin of particle a
    ib: float           # Spin of particle b
    q: float            # Q-value (eV)
    pnt: int            # Penetrability flag
    shf: int            # Shift factor flag
    mt: int             # Reaction MT number
    pa: float           # Parity of particle a
    pb: float           # Parity of particle b


@dataclass
class RML_Channel:
    """Channel definition within a spin group."""
    ipp: int            # Particle pair index (1-based)
    l: int              # Orbital angular momentum
    sch: float          # Channel spin
    bnd: float          # Boundary condition
    ape: float          # Effective channel radius (fm)
    apt: float          # True channel radius (fm)


@dataclass
class RML_Resonance:
    """Single resonance in R-Matrix Limited."""
    er: float           # Resonance energy (eV)
    widths: List[float] # NCH channel widths


# --- Background R-Matrix (KBK) ---

@dataclass
class NoBackgroundRMatrix:
    """LBK=0: no background R-matrix contribution."""
    lch: int            # Channel index (1-based)


@dataclass
class TabulatedBackgroundRMatrix:
    """LBK=1: tabulated complex background R-matrix."""
    lch: int
    rbr_interp: List[Tuple[int, int]]  # (NBT,INT) for real part
    rbr_energies: List[float]
    rbr_values: List[float]             # RBR(E)
    rbi_interp: List[Tuple[int, int]]  # (NBT,INT) for imaginary part
    rbi_energies: List[float]
    rbi_values: List[float]             # RBI(E)


@dataclass
class SammyBackgroundRMatrix:
    """LBK=2: SAMMY parametrization background R-matrix."""
    lch: int
    ed: float           # Lower energy limit
    eu: float           # Upper energy limit
    r0: float           # Radius parameter 0
    r1: float           # Radius parameter 1
    r2: float           # Radius parameter 2
    s0: float           # Scattering parameter 0
    s1: float           # Scattering parameter 1


@dataclass
class FrohnerBackgroundRMatrix:
    """LBK=3: Frohner parametrization background R-matrix."""
    lch: int
    ed: float           # Lower energy limit
    eu: float           # Upper energy limit
    r0: float           # Radius parameter
    s0: float           # Scattering parameter
    ga: float           # Gamma parameter


BackgroundRMatrix = Union[
    NoBackgroundRMatrix,
    TabulatedBackgroundRMatrix,
    SammyBackgroundRMatrix,
    FrohnerBackgroundRMatrix,
]


# --- Tabulated Phase Shifts (KPS) ---

@dataclass
class TabulatedPhaseShift:
    """Tabulated hard-sphere phase shift (LPS=1)."""
    psr_interp: List[Tuple[int, int]]
    psr_energies: List[float]
    psr_values: List[float]             # Real part
    psi_interp: List[Tuple[int, int]]
    psi_energies: List[float]
    psi_values: List[float]             # Imaginary part


# --- Spin Group ---

@dataclass
class RML_SpinGroup:
    """Spin group for R-Matrix Limited."""
    aj: float           # Spin
    pj: float           # Parity
    kbk: int            # Number of background R-matrix channels
    kps: int            # Phase shift flag
    channels: List[RML_Channel] = field(default_factory=list)
    resonances: List[RML_Resonance] = field(default_factory=list)
    background: List[BackgroundRMatrix] = field(default_factory=list)
    phase_shift: Optional[TabulatedPhaseShift] = None


# --- Top-level R-Matrix Limited ---

@dataclass
class RMatrixLimited:
    """R-Matrix Limited resonance parameters (LRU=1, LRF=7)."""
    ifg: int            # 0=widths(eV), 1=reduced-width amplitudes
    krm: int            # Formalism: 1=SLBW, 2=MLBW, 3=RM, 4=full R-matrix
    krl: int            # 0=non-relativistic, 1=relativistic
    spi: float          # Target spin
    ap: float           # Scattering radius (fm)
    particle_pairs: List[RML_ParticlePair] = field(default_factory=list)
    spin_groups: List[RML_SpinGroup] = field(default_factory=list)


@dataclass
class EnergyRange:
    """Single energy range within an isotope."""
    el: float           # Lower energy boundary (eV)
    eh: float           # Upper energy boundary (eV)
    lru: int            # 0=radius only, 1=resolved, 2=unresolved
    lrf: int            # 1=SLBW, 2=MLBW, 3=Reich-Moore, 7=RML
    nro: int            # Energy-dependent scattering radius flag
    naps: int           # Channel radius flag
    parameters: Union[
        ScatteringRadiusOnly,
        ResolvedResonanceRange,
        UnresolvedCaseA,
        UnresolvedCaseB,
        UnresolvedCaseC,
        RMatrixLimited,
        None,
    ] = None
    ap_e: Optional[EnergyDependentScatteringRadius] = None


@dataclass
class Isotope:
    """Isotope record within MF2/MT151."""
    za: float           # ZA identifier for this isotope
    abn: float          # Abundance fraction
    lfw: int            # Average fission widths flag
    num_energy_ranges: int  # NER
    energy_ranges: List[EnergyRange] = field(default_factory=list)


@dataclass
class MF2MT151(MT):
    """
    MT151 section of MF2 — Resonance Parameters.

    Stores resolved resonance parameters (SLBW, MLBW, Reich-Moore)
    and scattering-radius-only ranges as read from an ENDF file.
    """

    _za: float = None
    _awr: float = None
    _mat: int = None
    _nis: int = 0
    _isotopes: List[Isotope] = field(default_factory=list)

    # Line count
    num_lines: int = 0

    # ------------------------------------------------------------------
    # Properties — convenience accessors for the common single-isotope case
    # ------------------------------------------------------------------

    @property
    def zaid(self) -> float:
        """ZA identifier (1000*Z + A)."""
        return self._za

    @property
    def atomic_weight_ratio(self) -> float:
        """Atomic weight ratio (nucleus mass / neutron mass)."""
        return self._awr

    @property
    def isotopes(self) -> List[Isotope]:
        """List of isotope records."""
        return self._isotopes

    @property
    def scattering_radius(self) -> Optional[float]:
        """AP from the first energy range (fm)."""
        er = self._first_energy_range()
        if er is None or er.parameters is None:
            return None
        return er.parameters.ap

    @property
    def target_spin(self) -> Optional[float]:
        """SPI from the first energy range."""
        er = self._first_energy_range()
        if er is None or er.parameters is None:
            return None
        return er.parameters.spi

    @property
    def energy_range(self) -> Optional[Tuple[float, float]]:
        """(EL, EH) tuple from the first energy range."""
        er = self._first_energy_range()
        if er is None:
            return None
        return (er.el, er.eh)

    @property
    def formalism(self) -> str:
        """Human-readable formalism name from the first energy range."""
        er = self._first_energy_range()
        if er is None:
            return "unknown"
        return _formalism_name(er.lru, er.lrf)

    @property
    def num_resonances(self) -> int:
        """Total number of resonances across all isotopes and l-values."""
        total = 0
        for iso in self._isotopes:
            for er in iso.energy_ranges:
                if isinstance(er.parameters, ResolvedResonanceRange):
                    for lv in er.parameters.l_values:
                        total += lv.num_resonances
                elif isinstance(er.parameters, RMatrixLimited):
                    for sg in er.parameters.spin_groups:
                        total += len(sg.resonances)
        return total

    @property
    def num_l_values(self) -> Optional[int]:
        """NLS from the first resolved energy range."""
        er = self._first_energy_range()
        if er is not None and isinstance(er.parameters, ResolvedResonanceRange):
            return er.parameters.nls
        return None

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def resonance_energies(self) -> np.ndarray:
        """Sorted array of all resonance energies (eV)."""
        energies = []
        for iso in self._isotopes:
            for er in iso.energy_ranges:
                if isinstance(er.parameters, ResolvedResonanceRange):
                    for lv in er.parameters.l_values:
                        for res in lv.resonances:
                            energies.append(res.energy)
        return np.sort(np.array(energies, dtype=float))

    def resonance_table(self) -> List[Dict]:
        """List of dicts with formalism-aware column names.

        Columns always include ``energy``, ``spin``, ``l``.
        Remaining columns depend on the formalism (LRF):
        - SLBW/MLBW: GT, GN, GG, GF
        - Reich-Moore: GN, GG, GFA, GFB
        """
        rows: List[Dict] = []
        for iso in self._isotopes:
            for er in iso.energy_ranges:
                if not isinstance(er.parameters, ResolvedResonanceRange):
                    continue
                names = _column_names(er.lrf)
                for lv in er.parameters.l_values:
                    for res in lv.resonances:
                        row = {
                            "energy": res.energy,
                            "spin": res.spin,
                            "l": lv.l,
                            names[0]: res.c3,
                            names[1]: res.c4,
                            names[2]: res.c5,
                            names[3]: res.c6,
                        }
                        rows.append(row)
        return rows

    def resonances_dataframe(self):
        """Return a ``pandas.DataFrame`` of the resonance table."""
        import pandas as pd
        return pd.DataFrame(self.resonance_table())

    def unresolved_parameters_table(self) -> List[Dict]:
        """List of dicts for unresolved resonance parameters (for DataFrame export)."""
        rows: List[Dict] = []
        for iso in self._isotopes:
            for er in iso.energy_ranges:
                p = er.parameters
                if isinstance(p, UnresolvedCaseA):
                    for lv in p.l_values:
                        for js in lv.j_states:
                            rows.append({
                                "l": lv.l, "J": js.aj, "D": js.d,
                                "AMUN": js.amun, "GN0": js.gn0, "GG": js.gg,
                            })
                elif isinstance(p, UnresolvedCaseB):
                    for lv in p.l_values:
                        for js in lv.j_states:
                            base = {
                                "l": lv.l, "J": js.aj, "D": js.d,
                                "AMUN": js.amun, "GN0": js.gn0, "GG": js.gg,
                                "MUF": js.muf,
                            }
                            for i, e in enumerate(p.energies):
                                row = dict(base)
                                row["energy"] = e
                                row["GF"] = js.gf[i] if i < len(js.gf) else 0.0
                                rows.append(row)
                elif isinstance(p, UnresolvedCaseC):
                    for lv in p.l_values:
                        for js in lv.j_states:
                            for ep in js.energy_points:
                                rows.append({
                                    "l": lv.l, "J": js.aj, "energy": ep.es,
                                    "D": ep.d, "GX": ep.gx, "GN0": ep.gn0,
                                    "GG": ep.gg, "GF": ep.gf,
                                })
        return rows

    def unresolved_dataframe(self):
        """Return a ``pandas.DataFrame`` of unresolved parameters."""
        import pandas as pd
        return pd.DataFrame(self.unresolved_parameters_table())

    def summary(self) -> str:
        """Human-readable summary string."""
        parts = [
            f"MF2/MT151  ZA={self._za}  AWR={self._awr}  NIS={self._nis}",
        ]
        for iso in self._isotopes:
            parts.append(
                f"  Isotope ZA={iso.za}  ABN={iso.abn}  LFW={iso.lfw}  NER={iso.num_energy_ranges}"
            )
            for er in iso.energy_ranges:
                parts.append(
                    f"    Range [{er.el:.6g}, {er.eh:.6g}] eV  "
                    f"LRU={er.lru} LRF={er.lrf} ({_formalism_name(er.lru, er.lrf)})"
                )
                if er.ap_e is not None:
                    parts.append(
                        f"      NRO=1: {len(er.ap_e.energies)} energy-dependent AP points"
                    )
                if isinstance(er.parameters, ScatteringRadiusOnly):
                    parts.append(
                        f"      SPI={er.parameters.spi}  AP={er.parameters.ap}"
                    )
                elif isinstance(er.parameters, ResolvedResonanceRange):
                    p = er.parameters
                    parts.append(
                        f"      SPI={p.spi}  AP={p.ap}  NLS={p.nls}  NLSC={p.nlsc}"
                    )
                    for lv in p.l_values:
                        parts.append(
                            f"        l={lv.l}  NRS={lv.num_resonances}"
                        )
                elif isinstance(er.parameters, UnresolvedCaseA):
                    p = er.parameters
                    nj = sum(len(lv.j_states) for lv in p.l_values)
                    parts.append(
                        f"      SPI={p.spi}  AP={p.ap}  LSSF={p.lssf}  NLS={p.nls}  NJ_total={nj}"
                    )
                    for lv in p.l_values:
                        parts.append(
                            f"        l={lv.l}  NJS={len(lv.j_states)}"
                        )
                elif isinstance(er.parameters, UnresolvedCaseB):
                    p = er.parameters
                    nj = sum(len(lv.j_states) for lv in p.l_values)
                    parts.append(
                        f"      SPI={p.spi}  AP={p.ap}  LSSF={p.lssf}  NE={p.ne}  NLS={p.nls}  NJ_total={nj}"
                    )
                    for lv in p.l_values:
                        parts.append(
                            f"        l={lv.l}  NJS={len(lv.j_states)}"
                        )
                elif isinstance(er.parameters, UnresolvedCaseC):
                    p = er.parameters
                    nj = sum(len(lv.j_states) for lv in p.l_values)
                    parts.append(
                        f"      SPI={p.spi}  AP={p.ap}  LSSF={p.lssf}  NLS={p.nls}  NJ_total={nj}"
                    )
                    for lv in p.l_values:
                        parts.append(
                            f"        l={lv.l}  NJS={len(lv.j_states)}"
                        )
                elif isinstance(er.parameters, RMatrixLimited):
                    p = er.parameters
                    nres = sum(len(sg.resonances) for sg in p.spin_groups)
                    parts.append(
                        f"      SPI={p.spi}  AP={p.ap}  IFG={p.ifg}  KRM={p.krm}  KRL={p.krl}"
                    )
                    parts.append(
                        f"      NPP={len(p.particle_pairs)}  NJS={len(p.spin_groups)}  NRS_total={nres}"
                    )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # ENDF round-trip serialization
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Serialize back to ENDF format."""
        mat = self._mat if self._mat is not None else 0
        mf = 2
        mt = 151
        lines: List[str] = []
        ln = 1  # line counter

        # HEAD record
        lines.append(_cont(
            [self._za, self._awr, 0, 0, self._nis, 0],
            mat, mf, mt, ln,
            [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
             ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
        ))
        ln += 1

        for iso in self._isotopes:
            # Isotope CONT
            lines.append(_cont(
                [iso.za, iso.abn, 0, iso.lfw, iso.num_energy_ranges, 0],
                mat, mf, mt, ln,
                [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                 ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
            ))
            ln += 1

            for er in iso.energy_ranges:
                # Energy range CONT
                lines.append(_cont(
                    [er.el, er.eh, er.lru, er.lrf, er.nro, er.naps],
                    mat, mf, mt, ln,
                    [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
                ))
                ln += 1

                # NRO=1: energy-dependent scattering radius TAB1
                if er.ap_e is not None:
                    ap_e = er.ap_e
                    tab_lines, ln = format_tab1(
                        0.0, 0.0, 0, 0,
                        ap_e.interpolation, ap_e.energies, ap_e.ap_values,
                        mat, mf, mt, ln,
                    )
                    lines.extend(tab_lines)

                if isinstance(er.parameters, ScatteringRadiusOnly):
                    p = er.parameters
                    lines.append(_cont(
                        [p.spi, p.ap, 0, 0, 0, 0],
                        mat, mf, mt, ln,
                        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
                    ))
                    ln += 1

                elif isinstance(er.parameters, ResolvedResonanceRange):
                    p = er.parameters
                    lines.append(_cont(
                        [p.spi, p.ap, 0, 0, p.nls, p.nlsc],
                        mat, mf, mt, ln,
                        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT],
                    ))
                    ln += 1

                    for lv in p.l_values:
                        lines.append(_cont(
                            [lv.awri, 0, lv.l, 0, 6 * lv.num_resonances, lv.num_resonances],
                            mat, mf, mt, ln,
                            [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT],
                        ))
                        ln += 1
                        for res in lv.resonances:
                            lines.append(_cont(
                                [res.energy, res.spin, res.c3, res.c4, res.c5, res.c6],
                                mat, mf, mt, ln,
                            ))
                            ln += 1

                elif isinstance(er.parameters, UnresolvedCaseA):
                    ln = _serialize_urr_case_a(er.parameters, lines, mat, mf, mt, ln)

                elif isinstance(er.parameters, UnresolvedCaseB):
                    ln = _serialize_urr_case_b(er.parameters, lines, mat, mf, mt, ln)

                elif isinstance(er.parameters, UnresolvedCaseC):
                    ln = _serialize_urr_case_c(er.parameters, lines, mat, mf, mt, ln)

                elif isinstance(er.parameters, RMatrixLimited):
                    ln = _serialize_rml(er.parameters, lines, mat, mf, mt, ln)

        # SEND record
        lines.append(format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, mf, 0, 99999,
            formats=[ENDF_FORMAT_INT] * 6,
        ))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _first_energy_range(self) -> Optional[EnergyRange]:
        """Return the first energy range across all isotopes, or None."""
        for iso in self._isotopes:
            if iso.energy_ranges:
                return iso.energy_ranges[0]
        return None


# ======================================================================
# Module-level helpers
# ======================================================================

_FORMALISM_NAMES = {
    (0, 0): "none",
    (1, 1): "SLBW",
    (1, 2): "MLBW",
    (1, 3): "Reich-Moore",
    (1, 7): "R-Matrix Limited",
    (2, 1): "unresolved (case A)",
    (2, 2): "unresolved (case C)",
}


def _formalism_name(lru: int, lrf: int) -> str:
    return _FORMALISM_NAMES.get((lru, lrf), f"LRU={lru}/LRF={lrf}")


def _column_names(lrf: int) -> Tuple[str, str, str, str]:
    """Return (c3_name, c4_name, c5_name, c6_name) for a given LRF."""
    if lrf == 3:
        return ("GN", "GG", "GFA", "GFB")
    # SLBW (1) and MLBW (2) share the same ordering
    return ("GT", "GN", "GG", "GF")


def _cont(values, mat, mf, mt, ln, formats=None):
    """Shorthand for ``format_endf_data_line`` with all-float default."""
    if formats is None:
        formats = [ENDF_FORMAT_FLOAT] * 6
    return format_endf_data_line(values, mat, mf, mt, ln, formats=formats)


# ======================================================================
# Serialization helpers for new parameter types
# ======================================================================

def _serialize_urr_case_a(p, lines, mat, mf, mt, ln):
    """Serialize UnresolvedCaseA to ENDF lines."""
    # CONT: SPI, AP, LSSF, 0, NLS, 0
    lines.append(_cont(
        [p.spi, p.ap, p.lssf, 0, p.nls, 0],
        mat, mf, mt, ln,
        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
    ))
    ln += 1
    for lv in p.l_values:
        njs = len(lv.j_states)
        # LIST header: AWRI, 0, L, 0, 6*NJS, NJS
        lines.append(_cont(
            [lv.awri, 0, lv.l, 0, 6 * njs, njs],
            mat, mf, mt, ln,
            [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        ))
        ln += 1
        # LIST body: [D, AJ, AMUN, GN0, GG, 0] per J-state
        body = []
        for js in lv.j_states:
            body.extend([js.d, js.aj, js.amun, js.gn0, js.gg, 0.0])
        body_lines, ln = format_data_values(body, mat, mf, mt, ln)
        lines.extend(body_lines)
    return ln


def _serialize_urr_case_b(p, lines, mat, mf, mt, ln):
    """Serialize UnresolvedCaseB to ENDF lines."""
    ne = p.ne
    nls = p.nls
    # LIST header: SPI, AP, LSSF, 0, NE, NLS
    lines.append(_cont(
        [p.spi, p.ap, p.lssf, 0, ne, nls],
        mat, mf, mt, ln,
        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
         ENDF_FORMAT_INT, ENDF_FORMAT_INT],
    ))
    ln += 1
    # LIST body: NE energy values
    body_lines, ln = format_data_values(list(p.energies), mat, mf, mt, ln)
    lines.extend(body_lines)
    for lv in p.l_values:
        njs = len(lv.j_states)
        # CONT: AWRI, 0, L, 0, NJS, 0
        lines.append(_cont(
            [lv.awri, 0, lv.l, 0, njs, 0],
            mat, mf, mt, ln,
            [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
        ))
        ln += 1
        for js in lv.j_states:
            nw = ne + 6
            # LIST header: 0, 0, L, MUF, NE+6, 0
            lines.append(_cont(
                [0, 0, lv.l, js.muf, nw, 0],
                mat, mf, mt, ln,
                [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
            ))
            ln += 1
            # LIST body: D, AJ, AMUN, GN0, GG, 0, GF_1..GF_NE
            body = [js.d, js.aj, js.amun, js.gn0, js.gg, 0.0] + list(js.gf)
            body_lines, ln = format_data_values(body, mat, mf, mt, ln)
            lines.extend(body_lines)
    return ln


def _serialize_urr_case_c(p, lines, mat, mf, mt, ln):
    """Serialize UnresolvedCaseC to ENDF lines."""
    # CONT: SPI, AP, LSSF, 0, NLS, 0
    lines.append(_cont(
        [p.spi, p.ap, p.lssf, 0, p.nls, 0],
        mat, mf, mt, ln,
        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
    ))
    ln += 1
    for lv in p.l_values:
        njs = len(lv.j_states)
        # CONT: AWRI, 0, L, 0, NJS, 0
        lines.append(_cont(
            [lv.awri, 0, lv.l, 0, njs, 0],
            mat, mf, mt, ln,
            [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
        ))
        ln += 1
        for js in lv.j_states:
            ne = len(js.energy_points)
            nw = 6 * ne + 6
            # LIST header: AJ, 0, INT, 0, 6*NE+6, NE
            lines.append(_cont(
                [js.aj, 0, js.int_code, 0, nw, ne],
                mat, mf, mt, ln,
                [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT],
            ))
            ln += 1
            # LIST body: [0, 0, AMUX, AMUN, AMUG, AMUF, ES, D, GX, GN0, GG, GF] * NE
            body = [0.0, 0.0, js.amux, js.amun, js.amug, js.amuf]
            for ep in js.energy_points:
                body.extend([ep.es, ep.d, ep.gx, ep.gn0, ep.gg, ep.gf])
            body_lines, ln = format_data_values(body, mat, mf, mt, ln)
            lines.extend(body_lines)
    return ln


def _serialize_rml(p, lines, mat, mf, mt, ln):
    """Serialize RMatrixLimited to ENDF lines."""
    njs = len(p.spin_groups)
    # CONT: 0, 0, IFG, KRM, NJS, KRL
    lines.append(_cont(
        [0, 0, p.ifg, p.krm, njs, p.krl],
        mat, mf, mt, ln,
        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
         ENDF_FORMAT_INT, ENDF_FORMAT_INT,
         ENDF_FORMAT_INT, ENDF_FORMAT_INT],
    ))
    ln += 1

    # Particle pairs LIST
    npp = len(p.particle_pairs)
    lines.append(_cont(
        [p.spi, p.ap, 0, 0, 12 * npp, 2 * npp],
        mat, mf, mt, ln,
        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
         ENDF_FORMAT_INT, ENDF_FORMAT_INT],
    ))
    ln += 1
    pp_body = []
    for pp in p.particle_pairs:
        pp_body.extend([pp.ma, pp.mb, pp.za, pp.zb, pp.ia, pp.ib,
                        pp.q, pp.pnt, pp.shf, pp.mt, pp.pa, pp.pb])
    if pp_body:
        pp_lines, ln = format_data_values(pp_body, mat, mf, mt, ln)
        lines.extend(pp_lines)

    # Spin groups
    for sg in p.spin_groups:
        nch = len(sg.channels)
        # Channel LIST header
        lines.append(_cont(
            [sg.aj, sg.pj, sg.kbk, sg.kps, 6 * nch, nch],
            mat, mf, mt, ln,
            [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        ))
        ln += 1
        ch_body = []
        for ch in sg.channels:
            ch_body.extend([ch.ipp, ch.l, ch.sch, ch.bnd, ch.ape, ch.apt])
        if ch_body:
            ch_lines, ln = format_data_values(ch_body, mat, mf, mt, ln)
            lines.extend(ch_lines)

        # Resonances LIST
        nrs = len(sg.resonances)
        # NX: number of 6-wide entries per resonance line
        nx = (nch + 1 + 5) // 6  # ceil((NCH+1)/6), each row holds 6 values
        lines.append(_cont(
            [0, 0, 0, nrs, 6 * nx * nrs, nrs],
            mat, mf, mt, ln,
            [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
             ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT,
             ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        ))
        ln += 1
        res_body = []
        for res in sg.resonances:
            row = [res.er] + list(res.widths)
            # Pad to 6*NX
            while len(row) < 6 * nx:
                row.append(0.0)
            res_body.extend(row)
        if res_body:
            res_lines, ln = format_data_values(res_body, mat, mf, mt, ln)
            lines.extend(res_lines)

        # Background R-matrix (KBK)
        if sg.kbk > 0:
            for bg in sg.background:
                if isinstance(bg, NoBackgroundRMatrix):
                    lines.append(_cont(
                        [0, 0, bg.lch, 0, 0, 0],
                        mat, mf, mt, ln,
                        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
                    ))
                    ln += 1
                elif isinstance(bg, TabulatedBackgroundRMatrix):
                    lines.append(_cont(
                        [0, 0, bg.lch, 1, 0, 0],
                        mat, mf, mt, ln,
                        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
                    ))
                    ln += 1
                    # TAB1 for RBR
                    rbr_lines, ln = format_tab1(
                        0.0, 0.0, 0, 0,
                        bg.rbr_interp, bg.rbr_energies, bg.rbr_values,
                        mat, mf, mt, ln,
                    )
                    lines.extend(rbr_lines)
                    # TAB1 for RBI
                    rbi_lines, ln = format_tab1(
                        0.0, 0.0, 0, 0,
                        bg.rbi_interp, bg.rbi_energies, bg.rbi_values,
                        mat, mf, mt, ln,
                    )
                    lines.extend(rbi_lines)
                elif isinstance(bg, SammyBackgroundRMatrix):
                    lines.append(_cont(
                        [0, 0, bg.lch, 2, 0, 0],
                        mat, mf, mt, ln,
                        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
                    ))
                    ln += 1
                    # LIST: ED, EU, R0, R1, R2, S0, S1
                    lines.append(_cont(
                        [bg.ed, bg.eu, 0, 0, 7, 0],
                        mat, mf, mt, ln,
                        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
                    ))
                    ln += 1
                    body = [bg.r0, bg.r1, bg.r2, bg.s0, bg.s1, 0.0, 0.0]
                    body_lines, ln = format_data_values(body, mat, mf, mt, ln)
                    lines.extend(body_lines)
                elif isinstance(bg, FrohnerBackgroundRMatrix):
                    lines.append(_cont(
                        [0, 0, bg.lch, 3, 0, 0],
                        mat, mf, mt, ln,
                        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
                    ))
                    ln += 1
                    # LIST: ED, EU, R0, S0, GA
                    lines.append(_cont(
                        [bg.ed, bg.eu, 0, 0, 5, 0],
                        mat, mf, mt, ln,
                        [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
                    ))
                    ln += 1
                    body = [bg.r0, bg.s0, bg.ga, 0.0, 0.0]
                    body_lines, ln = format_data_values(body, mat, mf, mt, ln)
                    lines.extend(body_lines)

        # Phase shifts (KPS)
        if sg.kps > 0:
            lps = 1 if sg.phase_shift is not None else 0
            lines.append(_cont(
                [0, 0, 0, 0, lps, 0],
                mat, mf, mt, ln,
                [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                 ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
            ))
            ln += 1
            if lps == 1 and sg.phase_shift is not None:
                ps = sg.phase_shift
                psr_lines, ln = format_tab1(
                    0.0, 0.0, 0, 0,
                    ps.psr_interp, ps.psr_energies, ps.psr_values,
                    mat, mf, mt, ln,
                )
                lines.extend(psr_lines)
                psi_lines, ln = format_tab1(
                    0.0, 0.0, 0, 0,
                    ps.psi_interp, ps.psi_energies, ps.psi_values,
                    mat, mf, mt, ln,
                )
                lines.extend(psi_lines)

    return ln
