"""
Parser for MF2 (Resonance Parameters) sections in ENDF files.

MF2/MT151 stores resonance parameter data for neutron-induced reactions:
- LRU=0: Scattering radius only
- LRU=1, LRF=1/2/3: Resolved resonances (SLBW, MLBW, Reich-Moore)
- LRU=1, LRF=7: R-Matrix Limited
- LRU=2, LRF=1: Unresolved (Case A: LFW=0, Case B: LFW=1)
- LRU=2, LRF=2: Unresolved (Case C: all energy-dependent)
"""
import warnings
from typing import List, Tuple

from ..classes.mf import MF
from ..classes.mf2.mf2mt151 import (
    MF2MT151,
    Isotope,
    EnergyRange,
    EnergyDependentScatteringRadius,
    ResolvedResonanceRange,
    ScatteringRadiusOnly,
    LValueBlock,
    Resonance,
    # URR
    UnresolvedCaseA, URR_LValue_CaseA, URR_JState_CaseA,
    UnresolvedCaseB, URR_LValue_CaseB, URR_JState_CaseB,
    UnresolvedCaseC, URR_LValue_CaseC, URR_JState_CaseC, URR_EnergyPoint,
    # RML
    RMatrixLimited, RML_ParticlePair, RML_Channel, RML_Resonance,
    RML_SpinGroup,
    NoBackgroundRMatrix, TabulatedBackgroundRMatrix,
    SammyBackgroundRMatrix, FrohnerBackgroundRMatrix,
    TabulatedPhaseShift,
)
from ..utils import parse_line, parse_endf_id, group_lines_by_mt_with_positions, parse_tab1, parse_data_values
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def parse_mf2(lines: List[str]) -> MF:
    """
    Parse MF2 (Resonance Parameters) data.

    Parameters
    ----------
    lines : list of str
        Lines belonging to MF2 (all MT sections concatenated).

    Returns
    -------
    MF
    """
    logger.debug(f"Parsing MF2 with {len(lines)} lines")
    mf = MF(number=2)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            continue
        if mt != 151:
            logger.warning(f"Unexpected MT{mt} in MF2 (only MT151 expected), skipping")
            continue
        try:
            logger.debug(f"Parsing MT{mt} with {len(mt_lines)} lines")
            mt_section = parse_mf2_mt151(mt_lines, mt)
            mf.add_section(mt_section)
            if mt in line_counts:
                mt_section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as e:
            logger.warning(f"Error parsing MT{mt} in MF2: {e}")

    logger.debug("Finished parsing MF2")
    return mf


def parse_mf2_mt151(lines: List[str], mt: int) -> MF2MT151:
    """
    Parse MT151 (Resonance Parameters) from MF2.

    Parameters
    ----------
    lines : list of str
        Lines for this MT section.
    mt : int
        MT number (should be 151).

    Returns
    -------
    MF2MT151
    """
    if not lines:
        raise ValueError("No data provided for MT151 section in MF2")

    # HEAD record (line 0): ZA, AWR, 0, 0, NIS, 0
    head = parse_line(lines[0])
    za = head.get("C1")
    awr = head.get("C2")
    nis = int(head.get("C5", 0) or 0)
    mat, _, _ = parse_endf_id(lines[0])

    logger.debug(f"MF2/MT151: ZA={za}, AWR={awr}, NIS={nis}")

    isotopes: List[Isotope] = []
    idx = 1

    for _ in range(nis):
        isotope, idx = _parse_isotope(lines, idx)
        isotopes.append(isotope)

    return MF2MT151(
        number=mt,
        _za=za,
        _awr=awr,
        _mat=mat,
        _nis=nis,
        _isotopes=isotopes,
    )


# ------------------------------------------------------------------
# Internal parsers
# ------------------------------------------------------------------

def _parse_isotope(lines: List[str], idx: int) -> Tuple[Isotope, int]:
    """Parse one isotope record (CONT + NER energy ranges)."""
    cont = parse_line(lines[idx])
    idx += 1

    za_i = cont.get("C1")
    abn = cont.get("C2", 0.0) or 0.0
    lfw = int(cont.get("C4", 0) or 0)
    ner = int(cont.get("C5", 0) or 0)

    logger.debug(f"  Isotope ZA={za_i}, ABN={abn}, LFW={lfw}, NER={ner}")

    energy_ranges: List[EnergyRange] = []
    for _ in range(ner):
        er, idx = _parse_energy_range(lines, idx, lfw)
        energy_ranges.append(er)

    return Isotope(
        za=za_i,
        abn=abn,
        lfw=lfw,
        num_energy_ranges=ner,
        energy_ranges=energy_ranges,
    ), idx


def _parse_energy_range(lines: List[str], idx: int, lfw: int = 0) -> Tuple[EnergyRange, int]:
    """Parse one energy range record."""
    cont = parse_line(lines[idx])
    idx += 1

    el = cont.get("C1", 0.0)
    eh = cont.get("C2", 0.0)
    lru = int(cont.get("C3", 0) or 0)
    lrf = int(cont.get("C4", 0) or 0)
    nro = int(cont.get("C5", 0) or 0)
    naps = int(cont.get("C6", 0) or 0)

    logger.debug(
        f"    Range [{el:.6g}, {eh:.6g}] LRU={lru} LRF={lrf} NRO={nro} NAPS={naps}"
    )

    # NRO=1: energy-dependent scattering radius (TAB1)
    ap_e = None
    if nro == 1:
        ap_e, idx = _parse_energy_dependent_ap(lines, idx)

    parameters = None

    if lru == 0:
        parameters, idx = _parse_scattering_radius_only(lines, idx)
    elif lru == 1 and lrf in (1, 2, 3):
        parameters, idx = _parse_resolved_resonances(lines, idx, lrf)
    elif lru == 1 and lrf == 7:
        parameters, idx = _parse_rml(lines, idx)
    elif lru == 2:
        if lrf == 1 and lfw == 0:
            parameters, idx = _parse_unresolved_case_a(lines, idx)
        elif lrf == 1 and lfw == 1:
            parameters, idx = _parse_unresolved_case_b(lines, idx)
        elif lrf == 2:
            parameters, idx = _parse_unresolved_case_c(lines, idx)
        else:
            warnings.warn(
                f"MF2: unrecognized LRU=2/LRF={lrf}/LFW={lfw} "
                f"in range [{el:.6g}, {eh:.6g}]"
            )
    else:
        warnings.warn(
            f"MF2: unrecognized LRU={lru}/LRF={lrf} in range [{el:.6g}, {eh:.6g}]"
        )

    return EnergyRange(
        el=el,
        eh=eh,
        lru=lru,
        lrf=lrf,
        nro=nro,
        naps=naps,
        parameters=parameters,
        ap_e=ap_e,
    ), idx


# ------------------------------------------------------------------
# NRO=1: Energy-dependent scattering radius
# ------------------------------------------------------------------

def _parse_energy_dependent_ap(
    lines: List[str], idx: int
) -> Tuple[EnergyDependentScatteringRadius, int]:
    """Parse TAB1 record for energy-dependent scattering radius."""
    header, interp_pairs, x_data, y_data, idx = parse_tab1(lines, idx)
    logger.debug(f"      NRO=1: {len(x_data)} energy-dependent AP points")
    return EnergyDependentScatteringRadius(
        interpolation=interp_pairs,
        energies=x_data,
        ap_values=y_data,
    ), idx


# ------------------------------------------------------------------
# LRU=0: Scattering radius only
# ------------------------------------------------------------------

def _parse_scattering_radius_only(
    lines: List[str], idx: int
) -> Tuple[ScatteringRadiusOnly, int]:
    """Parse LRU=0 — one CONT line with SPI and AP."""
    cont = parse_line(lines[idx])
    idx += 1

    spi = cont.get("C1", 0.0)
    ap = cont.get("C2", 0.0)

    logger.debug(f"      LRU=0: SPI={spi}, AP={ap}")

    return ScatteringRadiusOnly(spi=spi, ap=ap), idx


# ------------------------------------------------------------------
# LRU=1, LRF=1/2/3: Resolved resonances
# ------------------------------------------------------------------

def _parse_resolved_resonances(
    lines: List[str], idx: int, lrf: int
) -> Tuple[ResolvedResonanceRange, int]:
    """Parse LRU=1, LRF=1/2/3 — SPI/AP header + NLS l-value blocks."""
    cont = parse_line(lines[idx])
    idx += 1

    spi = cont.get("C1", 0.0)
    ap = cont.get("C2", 0.0)
    nls = int(cont.get("C5", 0) or 0)
    nlsc = int(cont.get("C6", 0) or 0)

    logger.debug(f"      LRU=1/LRF={lrf}: SPI={spi}, AP={ap}, NLS={nls}, NLSC={nlsc}")

    l_values: List[LValueBlock] = []
    for _ in range(nls):
        lv, idx = _parse_l_value_block(lines, idx)
        l_values.append(lv)

    return ResolvedResonanceRange(
        spi=spi,
        ap=ap,
        nls=nls,
        nlsc=nlsc,
        l_values=l_values,
    ), idx


def _parse_l_value_block(
    lines: List[str], idx: int
) -> Tuple[LValueBlock, int]:
    """Parse one l-value block: CONT header + NRS resonance lines."""
    cont = parse_line(lines[idx])
    idx += 1

    awri = cont.get("C1", 0.0)
    # C2 and C4 were dropped entirely, and the writer re-emitted both as 0.
    # C2 is APL under LRF=3 (Reich-Moore) — the l-dependent scattering radius,
    # 0.5002 fm on the L=1 block of JEFF-4.0 Fe-56 against AP = 0.5444 — and QX,
    # the competitive-reaction Q, under LRF=1/2. C4 is LRX. Both are kept raw
    # here because their meaning depends on the parent range's LRF, which this
    # function does not see; ResolvedResonanceRange resolves it.
    apl_or_qx = cont.get("C2", 0.0)
    l = int(cont.get("C3", 0) or 0)
    lrx = int(cont.get("C4", 0) or 0)
    nrs = int(cont.get("C6", 0) or 0)

    logger.debug(f"        l={l}, NRS={nrs}, AWRI={awri}, C2={apl_or_qx}, LRX={lrx}")

    resonances: List[Resonance] = []
    for _ in range(nrs):
        ld = parse_line(lines[idx])
        idx += 1
        resonances.append(Resonance(
            energy=ld.get("C1", 0.0),
            spin=ld.get("C2", 0.0),
            c3=ld.get("C3", 0.0),
            c4=ld.get("C4", 0.0),
            c5=ld.get("C5", 0.0),
            c6=ld.get("C6", 0.0),
        ))

    return LValueBlock(
        awri=awri,
        l=l,
        num_resonances=nrs,
        resonances=resonances,
        apl_or_qx=apl_or_qx,
        lrx=lrx,
    ), idx


# ------------------------------------------------------------------
# LRU=2, LFW=0, LRF=1: Unresolved Case A (energy-independent)
# ------------------------------------------------------------------

def _parse_unresolved_case_a(
    lines: List[str], idx: int
) -> Tuple['UnresolvedCaseA', int]:
    """Parse unresolved resonances Case A (LFW=0, LRF=1)."""
    cont = parse_line(lines[idx])
    idx += 1

    spi = cont.get("C1", 0.0)
    ap = cont.get("C2", 0.0)
    lssf = int(cont.get("C3", 0) or 0)
    nls = int(cont.get("C5", 0) or 0)

    logger.debug(f"      URR Case A: SPI={spi}, AP={ap}, LSSF={lssf}, NLS={nls}")

    l_values = []
    for _ in range(nls):
        # LIST header: AWRI, 0, L, 0, 6*NJS, NJS
        hdr = parse_line(lines[idx])
        idx += 1
        awri = hdr.get("C1", 0.0)
        l = int(hdr.get("C3", 0) or 0)
        njs = int(hdr.get("C6", 0) or 0)

        # LIST body: [D, AJ, AMUN, GN0, GG, 0] * NJS
        vals, idx = parse_data_values(lines, idx, 6 * njs)
        j_states = []
        for j in range(njs):
            off = j * 6
            j_states.append(URR_JState_CaseA(
                d=vals[off],
                aj=vals[off + 1],
                amun=vals[off + 2],
                gn0=vals[off + 3],
                gg=vals[off + 4],
            ))
        l_values.append(URR_LValue_CaseA(awri=awri, l=l, j_states=j_states))

    return UnresolvedCaseA(
        spi=spi, ap=ap, lssf=lssf, nls=nls, l_values=l_values,
    ), idx


# ------------------------------------------------------------------
# LRU=2, LFW=1, LRF=1: Unresolved Case B (energy-dependent fission)
# ------------------------------------------------------------------

def _parse_unresolved_case_b(
    lines: List[str], idx: int
) -> Tuple['UnresolvedCaseB', int]:
    """Parse unresolved resonances Case B (LFW=1, LRF=1)."""
    # LIST header: SPI, AP, LSSF, 0, NE, NLS
    hdr = parse_line(lines[idx])
    idx += 1

    spi = hdr.get("C1", 0.0)
    ap = hdr.get("C2", 0.0)
    lssf = int(hdr.get("C3", 0) or 0)
    ne = int(hdr.get("C5", 0) or 0)
    nls = int(hdr.get("C6", 0) or 0)

    logger.debug(f"      URR Case B: SPI={spi}, AP={ap}, LSSF={lssf}, NE={ne}, NLS={nls}")

    # LIST body: NE energy values
    energies, idx = parse_data_values(lines, idx, ne)

    l_values = []
    for _ in range(nls):
        # CONT: AWRI, 0, L, 0, NJS, 0
        cont = parse_line(lines[idx])
        idx += 1
        awri = cont.get("C1", 0.0)
        l = int(cont.get("C3", 0) or 0)
        njs = int(cont.get("C5", 0) or 0)

        j_states = []
        for _ in range(njs):
            # LIST header: 0, 0, L, MUF, NE+6, 0
            jhdr = parse_line(lines[idx])
            idx += 1
            muf = int(jhdr.get("C4", 0) or 0)
            nw = int(jhdr.get("C5", 0) or 0)

            # LIST body: D, AJ, AMUN, GN0, GG, 0, GF_1..GF_NE
            vals, idx = parse_data_values(lines, idx, nw)
            j_states.append(URR_JState_CaseB(
                d=vals[0],
                aj=vals[1],
                amun=vals[2],
                gn0=vals[3],
                gg=vals[4],
                muf=muf,
                gf=vals[6:6 + ne],  # skip position 5 (zero)
            ))
        l_values.append(URR_LValue_CaseB(awri=awri, l=l, j_states=j_states))

    return UnresolvedCaseB(
        spi=spi, ap=ap, lssf=lssf, ne=ne, nls=nls,
        energies=energies, l_values=l_values,
    ), idx


# ------------------------------------------------------------------
# LRU=2, LRF=2: Unresolved Case C (all energy-dependent)
# ------------------------------------------------------------------

def _parse_unresolved_case_c(
    lines: List[str], idx: int
) -> Tuple['UnresolvedCaseC', int]:
    """Parse unresolved resonances Case C (LRF=2, all energy-dependent)."""
    # CONT: SPI, AP, LSSF, 0, NLS, 0
    cont = parse_line(lines[idx])
    idx += 1

    spi = cont.get("C1", 0.0)
    ap = cont.get("C2", 0.0)
    lssf = int(cont.get("C3", 0) or 0)
    nls = int(cont.get("C5", 0) or 0)

    logger.debug(f"      URR Case C: SPI={spi}, AP={ap}, LSSF={lssf}, NLS={nls}")

    l_values = []
    for _ in range(nls):
        # CONT: AWRI, 0, L, 0, NJS, 0
        lcont = parse_line(lines[idx])
        idx += 1
        awri = lcont.get("C1", 0.0)
        l = int(lcont.get("C3", 0) or 0)
        njs = int(lcont.get("C5", 0) or 0)

        j_states = []
        for _ in range(njs):
            # LIST header: AJ, 0, INT, 0, 6*NE+6, NE
            jhdr = parse_line(lines[idx])
            idx += 1
            aj = jhdr.get("C1", 0.0)
            int_code = int(jhdr.get("C3", 0) or 0)
            nw = int(jhdr.get("C5", 0) or 0)
            ne = int(jhdr.get("C6", 0) or 0)

            # LIST body: [0, 0, AMUX, AMUN, AMUG, AMUF,
            #             ES, D, GX, GN0, GG, GF] * NE
            vals, idx = parse_data_values(lines, idx, nw)
            amux = vals[2] if len(vals) > 2 else 0.0
            amun = vals[3] if len(vals) > 3 else 0.0
            amug = vals[4] if len(vals) > 4 else 0.0
            amuf = vals[5] if len(vals) > 5 else 0.0

            energy_points = []
            for k in range(ne):
                off = 6 + k * 6
                energy_points.append(URR_EnergyPoint(
                    es=vals[off],
                    d=vals[off + 1],
                    gx=vals[off + 2],
                    gn0=vals[off + 3],
                    gg=vals[off + 4],
                    gf=vals[off + 5],
                ))

            j_states.append(URR_JState_CaseC(
                aj=aj, int_code=int_code,
                amux=amux, amun=amun, amug=amug, amuf=amuf,
                energy_points=energy_points,
            ))
        l_values.append(URR_LValue_CaseC(awri=awri, l=l, j_states=j_states))

    return UnresolvedCaseC(
        spi=spi, ap=ap, lssf=lssf, nls=nls, l_values=l_values,
    ), idx


# ------------------------------------------------------------------
# LRU=1, LRF=7: R-Matrix Limited
# ------------------------------------------------------------------

def _parse_rml(
    lines: List[str], idx: int
) -> Tuple['RMatrixLimited', int]:
    """Parse R-Matrix Limited (LRU=1, LRF=7)."""
    # CONT: 0, 0, IFG, KRM, NJS, KRL
    cont = parse_line(lines[idx])
    idx += 1

    ifg = int(cont.get("C3", 0) or 0)
    krm = int(cont.get("C4", 0) or 0)
    njs = int(cont.get("C5", 0) or 0)
    krl = int(cont.get("C6", 0) or 0)

    logger.debug(f"      RML: IFG={ifg}, KRM={krm}, NJS={njs}, KRL={krl}")

    # Particle pairs LIST: SPI, AP, 0, 0, 12*NPP, 2*NPP
    pp_hdr = parse_line(lines[idx])
    idx += 1
    spi = pp_hdr.get("C1", 0.0)
    ap = pp_hdr.get("C2", 0.0)
    npp2 = int(pp_hdr.get("C6", 0) or 0)
    npp = npp2 // 2 if npp2 > 0 else 0
    nw_pp = int(pp_hdr.get("C5", 0) or 0)

    vals, idx = parse_data_values(lines, idx, nw_pp)

    particle_pairs = []
    for i in range(npp):
        off = i * 12
        particle_pairs.append(RML_ParticlePair(
            ma=vals[off], mb=vals[off + 1],
            za=vals[off + 2], zb=vals[off + 3],
            ia=vals[off + 4], ib=vals[off + 5],
            q=vals[off + 6],
            pnt=int(vals[off + 7] or 0),
            shf=int(vals[off + 8] or 0),
            mt=int(vals[off + 9] or 0),
            pa=vals[off + 10], pb=vals[off + 11],
        ))

    logger.debug(f"      RML: {npp} particle pairs")

    # Spin groups
    spin_groups = []
    for _ in range(njs):
        sg, idx = _parse_rml_spin_group(lines, idx)
        spin_groups.append(sg)

    return RMatrixLimited(
        ifg=ifg, krm=krm, krl=krl,
        spi=spi, ap=ap,
        particle_pairs=particle_pairs,
        spin_groups=spin_groups,
    ), idx


def _parse_rml_spin_group(
    lines: List[str], idx: int
) -> Tuple['RML_SpinGroup', int]:
    """Parse one spin group within RML."""
    # Channel LIST header: AJ, PJ, KBK, KPS, 6*NCH, NCH
    hdr = parse_line(lines[idx])
    idx += 1

    aj = hdr.get("C1", 0.0)
    pj = hdr.get("C2", 0.0)
    kbk = int(hdr.get("C3", 0) or 0)
    kps = int(hdr.get("C4", 0) or 0)
    nch = int(hdr.get("C6", 0) or 0)

    # Channel body: 6 values per channel
    ch_vals, idx = parse_data_values(lines, idx, 6 * nch)
    channels = []
    for i in range(nch):
        off = i * 6
        channels.append(RML_Channel(
            ipp=int(ch_vals[off] or 0),
            l=int(ch_vals[off + 1] or 0),
            sch=ch_vals[off + 2],
            bnd=ch_vals[off + 3],
            ape=ch_vals[off + 4],
            apt=ch_vals[off + 5],
        ))

    # Resonances LIST header: 0, 0, 0, NRS, 6*NX, NRS_again
    # Actually: 0, 0, 0, NRS, 6*NX*NRS, NRS  (but C4=NRS, C6=NRS)
    # The ENDF manual says: C4=NRS (number of resonances), C6=NX (lines per resonance)
    # Wait - let me re-check. The format is:
    # LIST: 0, 0, 0, NRS, 6*NX, NRS
    # where NX = ceiling((NCH+1)/6) * NRS ... no.
    # Actually per ENDF manual: C4=NRS, C5=6*NX, C6=NX
    # where NX is the total number of lines = NRS * ceil((NCH+1)/6)
    # But some evaluations use C5=6*NX*NRS and C6=NRS.
    # Let me just read the header carefully.
    res_hdr = parse_line(lines[idx])
    idx += 1
    nrs = int(res_hdr.get("C4", 0) or 0)
    nw_res = int(res_hdr.get("C5", 0) or 0)  # total words in body

    # Read all resonance data
    res_vals, idx = parse_data_values(lines, idx, nw_res)

    # NX = number of 6-value lines per resonance
    nx = (nch + 1 + 5) // 6  # ceil((NCH+1)/6)
    row_len = 6 * nx

    resonances = []
    for i in range(nrs):
        off = i * row_len
        er = res_vals[off]
        widths = []
        for c in range(nch):
            widths.append(res_vals[off + 1 + c])
        resonances.append(RML_Resonance(er=er, widths=widths))

    logger.debug(f"        Spin group J={aj}, P={pj}: {nch} channels, {nrs} resonances")

    # Background R-matrix
    background = []
    if kbk > 0:
        for _ in range(kbk):
            bg, idx = _parse_rml_background(lines, idx)
            background.append(bg)

    # Phase shifts
    phase_shift = None
    if kps > 0:
        phase_shift, idx = _parse_rml_phase_shift(lines, idx)

    return RML_SpinGroup(
        aj=aj, pj=pj, kbk=kbk, kps=kps,
        channels=channels, resonances=resonances,
        background=background, phase_shift=phase_shift,
    ), idx


def _parse_rml_background(
    lines: List[str], idx: int
) -> Tuple['BackgroundRMatrix', int]:
    """Parse one background R-matrix entry."""
    cont = parse_line(lines[idx])
    idx += 1
    lch = int(cont.get("C3", 0) or 0)
    lbk = int(cont.get("C4", 0) or 0)

    if lbk == 0:
        return NoBackgroundRMatrix(lch=lch), idx

    elif lbk == 1:
        # TAB1 for RBR (real part)
        _, rbr_interp, rbr_e, rbr_v, idx = parse_tab1(lines, idx)
        # TAB1 for RBI (imaginary part)
        _, rbi_interp, rbi_e, rbi_v, idx = parse_tab1(lines, idx)
        return TabulatedBackgroundRMatrix(
            lch=lch,
            rbr_interp=rbr_interp, rbr_energies=rbr_e, rbr_values=rbr_v,
            rbi_interp=rbi_interp, rbi_energies=rbi_e, rbi_values=rbi_v,
        ), idx

    elif lbk == 2:
        # LIST: ED, EU, 0, 0, 7, 0
        lhdr = parse_line(lines[idx])
        idx += 1
        ed = lhdr.get("C1", 0.0)
        eu = lhdr.get("C2", 0.0)
        nw = int(lhdr.get("C5", 0) or 0)
        vals, idx = parse_data_values(lines, idx, nw)
        return SammyBackgroundRMatrix(
            lch=lch, ed=ed, eu=eu,
            r0=vals[0], r1=vals[1], r2=vals[2],
            s0=vals[3], s1=vals[4],
        ), idx

    elif lbk == 3:
        # LIST: ED, EU, 0, 0, 5, 0
        lhdr = parse_line(lines[idx])
        idx += 1
        ed = lhdr.get("C1", 0.0)
        eu = lhdr.get("C2", 0.0)
        nw = int(lhdr.get("C5", 0) or 0)
        vals, idx = parse_data_values(lines, idx, nw)
        return FrohnerBackgroundRMatrix(
            lch=lch, ed=ed, eu=eu,
            r0=vals[0], s0=vals[1], ga=vals[2],
        ), idx

    else:
        warnings.warn(f"MF2/RML: unknown LBK={lbk} for channel {lch}")
        return NoBackgroundRMatrix(lch=lch), idx


def _parse_rml_phase_shift(
    lines: List[str], idx: int
) -> Tuple['TabulatedPhaseShift', int]:
    """Parse tabulated phase shift data."""
    # LIST: 0, 0, 0, 0, LPS, 0
    cont = parse_line(lines[idx])
    idx += 1
    lps = int(cont.get("C5", 0) or 0)

    if lps == 1:
        # TAB1 for PSR (real part)
        _, psr_interp, psr_e, psr_v, idx = parse_tab1(lines, idx)
        # TAB1 for PSI (imaginary part)
        _, psi_interp, psi_e, psi_v, idx = parse_tab1(lines, idx)
        return TabulatedPhaseShift(
            psr_interp=psr_interp, psr_energies=psr_e, psr_values=psr_v,
            psi_interp=psi_interp, psi_energies=psi_e, psi_values=psi_v,
        ), idx

    # LPS != 1: no tabulated data
    return None, idx
