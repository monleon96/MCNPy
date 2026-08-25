"""The committed micro-tapes: how they are built, and that they stay valid.

Phase 0 of the GNDS roadmap needs ENDF fixtures that live in the repo, so the
read-write goldens run on a machine with no access to the shared data tree.
Two fixtures, built two different ways for two different reasons:

``micro_fe56_structural.endf`` — real JEFF-4.0 Fe-56, cut down to
MF1/451, MF2/151, MF3/MT1,2,102, MF4/MT2 and MF34/MT2. The cut is done
**section by section with** :func:`remove_sections`: whole (MF, MT) blocks are
dropped and the MF1/451 directory is rebuilt, but not one record is
reformatted. Every surviving byte is a byte JEFF-4.0 wrote. That is what makes
it a fair stand-in for the real tape — it carries the real interpolation
regions, the real resonance parameters and the real TAB1 layout, quirks
included. Trimming it further would mean rewriting TAB1 records, which would
make the fixture kika's output rather than JEFF's.

``micro_fe56_mf33.endf`` — real JEFF-4.0 Fe-56 again, cut the same verbatim way,
keeping MF1/451, MF3 and MF33 for MT4 and MT16. It exists because the synthetic
covariance fixture below is one MT wide, so nothing committed could exercise a
**cross-reaction** assembly, and because the pair it keeps has no MT4×MT16 block
— which is what makes the carrier's matrix half NaN and pins
``docs/library/library-gaps.md`` D11.

``micro_fe56_cov.endf`` — synthetic, written by kika's own MF33/MF34 writers on
a four-point grid. There is no cheap honest slice of the real MF33: Fe-56
MT2 alone is 33 029 lines (~2.7 MB), and dropping energy bins from it means
rewriting records. So the covariance fixture is openly generated rather than
quietly truncated, and the real MF33 stays covered by the ``tape``-marked tests
that read the full file.

Regenerate both after an intentional change::

    REGEN_MICRO_TAPES=1 pytest kika/endf/tests/test_micro_tape_regen.py --deep

and commit the result. Without the variable, this module only checks that what
is committed still parses and still describes itself correctly.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.classes.mf5.base import MF5MT
from kika.endf.classes.mf5.partials import MF5PartialTabulated
from kika.endf.classes.mf35.mf35 import MF35MT, MF35SubSection
from kika.endf.utils import (
    ENDF_FORMAT_INT,
    format_endf_data_line,
    format_endf_fend_record,
    format_endf_mend_record,
    format_endf_send_record,
    format_endf_tend_record,
    parse_endf_id,
)
from kika.endf.writers import (
    create_mf33_from_covariance,
    create_mf34_from_covariance,
    remove_sections,
    update_mf1_directory,
    write_mf33_to_file,
    write_mf34_to_file,
)

DATA = Path(__file__).resolve().parent / "data"
STRUCTURAL = DATA / "micro_fe56_structural.endf"
COV = DATA / "micro_fe56_cov.endf"
PFNS = DATA / "micro_cf252_pfns.endf"
PFNS_COV = DATA / "micro_pfns_cov.endf"
NUBAR = DATA / "micro_u235_nubar.endf"
MF33 = DATA / "micro_fe56_mf33.endf"

#: MF6 fixtures, keyed by the tape they are cut from. Each carries a law the
#: rest of the library barely has -- see ``KEEP_MF6`` for what and why.
MF6_FIXTURES = {"be9": "be9_b81", "li6": "li6_b81", "c12": "c12_b81",
                "u235": "u235_b81"}

#: The charged-particle MF6 fixtures, copied **whole** rather than cut. Each is
#: a complete ENDF/B-VIII.0 evaluation of 7-56 kB; all five together are 138 kB,
#: an ninth of ``micro_be9_mf6.endf`` alone. Cutting them would cost the
#: property that makes an MF6 fixture worth having -- that the law bodies are
#: the evaluator's bytes -- and save nothing. Same reason ``tsl-s-CH4.endf`` is
#: copied whole by :func:`build_tsl`.
MF6_CP_FIXTURES = {"p_he3": "p_he3_b80", "d_h2": "d_h2_b80",
                   "h3_he4": "h3_he4_b80", "a_he4": "a_he4_b80",
                   "t_li7": "t_li7_b80"}


def mf6_fixture_path(key: str) -> Path:
    """Committed MF6 micro-tape cut from the *key* evaluation."""
    return DATA / f"micro_{key}_mf6.endf"


def mf32_fixture_path(key: str) -> Path:
    """Committed MF32 micro-tape for sub-format *key*."""
    return DATA / f"micro_{key}_mf32.endf"


def tsl_fixture_path(key: str) -> Path:
    """Committed MF7 micro-tape for *key*."""
    return DATA / f"micro_tsl_{key}.endf"

REGEN = bool(os.environ.get("REGEN_MICRO_TAPES"))

#: What the structural micro-tape keeps. Everything else is removed.
KEEP = {1: {451}, 2: {151}, 3: {1, 2, 102}, 4: {2}, 34: {2}}

#: What the MF33 micro-tape keeps, cut the same verbatim way from JEFF-4.0
#: Fe-56. ``micro_fe56_cov.endf``'s MF33 is synthetic and one MT wide, so
#: nothing committed could exercise a **cross-reaction** assembly — the case
#: `cross_section_covariance_blocks` exists for.
#:
#: MT4 (inelastic) and MT16 (n,2n) are the two reactions on that tape whose
#: MF33 is small: 1316 lines each, against 33 029 for MT2 and 33 032 for MT102.
#: Keeping capture would mean a 3 MB fixture for the same lesson. MF3 comes
#: along for the same reason it does in the PFNS cut — a covariance of a cross
#: section wants the cross section — and it is what lets the absolute→relative
#: conversion be exercised without a PENDF.
#:
#: The pair is also, by luck rather than design, the case that matters: the
#: evaluation states **no** MT4×MT16 cross block, so the carrier's matrix comes
#: back 50 % NaN and the fixture pins `docs/library/library-gaps.md` D11.
KEEP_MF33 = {1: {451}, 3: {4, 16}, 33: {4, 16}}

#: What the PFNS micro-tape keeps, cut the same verbatim way from
#: ENDF/B-VIII.1 Cf-252. MF3/MT18 comes along because the fission cross section
#: is what MT18's spectrum belongs to, and it costs 34 lines.
#:
#: **MF4 is dropped, and the MF5 adapter (2026-08-24) made that visible.** The
#: real Cf-252 states MF4/MT2 and MF4/MT18 — the latter two records, LTT=0/LI=1,
#: 152 bytes. Without it the fixture is the one shape the distributed library
#: does not contain: *MF5 present, MF4 absent*, which measured zero times across
#: ENDF/B-VIII.1's 487 modellable MF5 sections. So this fixture exercises the
#: adapter's **inference** branch — the isotropic angular half kika supplies
#: when ENDF states none — and the branch that covers 487 of 487, MF4 and MF5
#: merged into one `uncorrelated`, has no committed witness and is reached only
#: under ``--deep`` or by a hand-built test.
#:
#: Adding ``4: {18}`` here would fix that for 152 bytes, and it is **left alone
#: deliberately**: it would also *remove* the only committed witness for the
#: inference branch, and it changes a fixture shared with `kika/sampling/tests/`
#: and `kika/cov/tests/`. Its own change, with its own before and after.
KEEP_PFNS = {1: {451}, 3: {18}, 5: {18, 455}, 35: {18}}

#: What the nu-bar micro-tape keeps, cut the same verbatim way from
#: ENDF/B-VIII.1 U-235. MF3/MT18 comes along for the same reason it does in
#: the PFNS cut and for one more: GNDS hangs the fission multiplicities off the
#: fission *reaction*, so without MF3/MT18 there is no node for them and
#: `attachNubar` correctly refuses. It costs 270 lines.
KEEP_NUBAR = {1: {451, 452, 455, 456}, 3: {18}, 31: {452, 455, 456}}

#: What the MF7 elastic micro-tapes keep: MF7/MT2 and, where the evaluation has
#: one, MF7/MT451.
#:
#: **MF7/MT4 is dropped, and that is the whole trick.** Every other fixture here
#: is small because the sections it does not need are the bulk of the tape. A
#: TSL evaluation is the other way round — MF7/MT4 is 98-99 % of it (62 411
#: lines of 67 126 for ``tsl-Be-metal``; 1 144 410 of 1 144 539 for
#: ``tsl-HinH2O``) and it cannot be trimmed without rewriting records, which is
#: the one thing these fixtures may not do. So MT4 is cut away entirely and the
#: elastic section is what stays: a real MT2 of each ``LTHR`` branch for a few
#: hundred kB. MT4 coverage comes instead from ``micro_tsl_sch4.endf``, which is
#: a whole tape at 161 kB, and from the ``tape``-marked full-size tests.
KEEP_TSL_ELASTIC = {1: {451}, 7: {2, 451}}

#: The MF7 fixtures and the tape each is cut from. ``sch4`` is copied whole
#: rather than cut, because it is already the smallest TSL evaluation published
#: and dropping anything from it would cost the only committed MT4.
#:
#: ``sch4``             LTHR=2 (incoherent elastic), and an MT4 with LAT=0, a
#:                      single temperature and NS=1 — the secondary-scatterer
#:                      branch of the B array.
#: ``bemetal_elastic``  LTHR=1: 2306 Bragg edges over 11 temperatures, so the
#:                      LT stack and its histogram interpolation are real. Plus
#:                      an MF7/451 mapping the pseudo-ZA 126 to ZAI 4009.
#: ``un_elastic``       LTHR=3, the branch a ``lthr == 1`` test silently gets
#:                      wrong: a coherent block of 911 edges over 8
#:                      temperatures *followed by* an incoherent one.
#: ``jeff_be_elastic``  JEFF-4.0's own dialect rather than an adopted ENDF/B
#:                      evaluation: 80-column records with sequence numbers,
#:                      elemental ZA 4000, and LIST bodies padded with explicit
#:                      zeros while the TAB1 body beside them is blank-padded.
#:                      The only committed witness for ``PadStyle``.
TSL_FIXTURES = {
    "sch4": "tsl_s_ch4",
    "bemetal_elastic": "tsl_be_metal",
    "un_elastic": "tsl_un",
    "jeff_be_elastic": "tsl_jeff_be",
}

#: What each MF32 micro-tape keeps. File 2 comes along rather than being cut:
#: §32.3 makes File 32 meaningless without it — every resonance a covariance
#: names must be present in File 2, to a relative tolerance of 1e-5 — and it is
#: the cheapest part of all four (31 to 1339 lines).
KEEP_MF32 = {1: {451}, 2: {151}, 32: {151}}

#: The four MF32 fixtures and the evaluation each is cut from. One sub-format
#: apiece, chosen from the eleven tapes on this machine that carry an MF32:
#:
#: ``na23``    LCOMP=2 with no INTG block at all — a diagonal covariance, and
#:             the smallest MF32 anywhere at 52 lines.
#: ``cm244``   LCOMP=0, the ENDF/B-V-compatible per-L format. Only actinides
#:             still use it.
#: ``th232``   LCOMP=2 with INTG, **and** a second range with the unresolved
#:             LRU=2 covariance of §32.2.4. The only URR coverage that exists.
#: ``cl35``    LCOMP=2 for R-Matrix Limited. Chosen over W-186 and Cu-63
#:             because it is the one whose NJS (8) differs from its NJSX (7),
#:             which is what forces the spin-group loop to count NJS.
#:
#: LCOMP=1 has no fixture: the smallest evaluation using it is Mn-55 at 26 467
#: MF32 lines, ~2 MB, so it stays on the ``tape``-marked tests.
MF32_FIXTURES = {
    "na23": "na23_b81",
    "cm244": "cm244_b81",
    "th232": "th232_b81",
    "cl35": "cl35_b81",
}

#: Fe-56 identity, shared by both fixtures.
ZA, AWR, MAT, MT = 26056.0, 55.36735, 2631, 2

#: Identity of the synthetic PFNS tape. Not a real material: the numbers are
#: Cf-252's so that anything keyed on ZA behaves, but every record is generated.
PFNS_ZA, PFNS_AWR, PFNS_MAT, PFNS_MT = 98252.0, 249.916, 9861, 18

#: Grid and matrices of the synthetic covariance tape. Small, deterministic,
#: and asymmetric enough that a transposed matrix would be caught.
COV_GRID = np.array([0.85e6, 1.5e6, 2.5e6, 4.0e6])
COV_MAX_ORDER = 2


def _mf33_matrix() -> np.ndarray:
    n = len(COV_GRID) - 1
    base = np.arange(1, n + 1, dtype=float)
    m = 0.001 * np.outer(base, base) + np.diag(0.02 * base)
    return 0.5 * (m + m.T)


def _mf34_matrix() -> np.ndarray:
    n = (len(COV_GRID) - 1) * COV_MAX_ORDER
    base = np.arange(1, n + 1, dtype=float)
    m = 0.0005 * np.outer(base, base) + np.diag(0.01 * base)
    return 0.5 * (m + m.T)


# ---------------------------------------------------------------------------
# Inventory helper
# ---------------------------------------------------------------------------

def section_inventory(text: str) -> dict[int, dict[int, int]]:
    """Map ``{MF: {MT: number of data lines}}`` for an ENDF text."""
    inv: dict[int, dict[int, int]] = {}
    for line in text.splitlines():
        if len(line) < 75:
            continue
        _, mf, mt = parse_endf_id(line)
        if mf and mt:
            inv.setdefault(mf, {}).setdefault(mt, 0)
            inv[mf][mt] += 1
    return inv


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

#: What each MF6 fixture keeps. The MT choices are the census's, not a sample:
#: over all 557 ENDF/B-VIII.1 neutron tapes, **LAW=7 occurs twice and both are
#: Be-9 MT16**; LAW=6 occurs five times, three of them Li-6 MT41; LAW=0 and the
#: negative LAWs live in the actinide fission sections, of which U-235's MT18 is
#: by far the smallest at 591 records; and C-12 MT5 is the readiest LCT=3.
#: Between them the four fixtures carry every LAW that occurs in a *neutron*
#: sublibrary. LAW=5 is on none of them because it cannot be: charged-particle
#: elastic scattering needs a charged projectile. Its witnesses are the five
#: whole tapes in ``MF6_CP_FIXTURES``, cut from ENDF/B-VIII.0's charged-particle
#: sublibraries -- ``docs/library/mf6_witness_hunt.md``.
KEEP_MF6 = {
    # LAW=7 (both occurrences in the library) and LCT=1, plus LAW=1/2/4.
    "be9": {1: {451}, 6: {16, 600, 650, 700, 701, 800}},
    # LAW=6, and LAW=1/LAW=2 sections small enough to be free.
    "li6": {1: {451}, 6: {41, 52, 103}},
    # LCT=3 throughout, and LANG=2 Kalbach-Mann.
    "c12": {1: {451}, 6: {5}},
    # LAW=0, LAW=-5 and LAW=-15 in one 591-record section; LAW=3 in MT800.
    "u235": {1: {451}, 6: {18, 800}},
}


def build_mf6(source: Path, dest: Path, keep: dict) -> None:
    """Cut *source* down to *keep* and write it to *dest*, verbatim.

    Same machinery and same reason as :func:`build_structural`: the point of an
    MF6 fixture is that the law bodies are the evaluator's bytes. Two of these
    tapes are the *only* carriers of their law in the library, so a fixture
    regenerated by kika's own writer would leave those laws with no independent
    witness at all.
    """
    content = source.read_text()
    inventory = section_inventory(content)

    to_remove: list[tuple[int, int | None]] = []
    for mf, mts in sorted(inventory.items()):
        if mf not in keep:
            to_remove.append((mf, None))
            continue
        for mt in sorted(set(mts) - keep[mf]):
            to_remove.append((mf, mt))

    trimmed, n_removed = remove_sections(content, to_remove)
    assert n_removed, f"nothing was removed from {source.name} — wrong source?"
    dest.write_text(trimmed)


def build_structural(source: Path, dest: Path, keep: dict = None) -> None:
    """Cut *source* down to *keep* and write it to *dest*, verbatim.

    *keep* defaults to ``KEEP``; ``KEEP_MF33`` cuts the second Fe-56 fixture
    from the same host tape by the same rule, which is why this takes a
    parameter rather than there being two near-identical builders.
    """
    keep = KEEP if keep is None else keep
    content = source.read_text()
    inventory = section_inventory(content)

    to_remove: list[tuple[int, int | None]] = []
    for mf, mts in sorted(inventory.items()):
        if mf not in keep:
            to_remove.append((mf, None))
            continue
        for mt in sorted(set(mts) - keep[mf]):
            to_remove.append((mf, mt))

    trimmed, n_removed = remove_sections(content, to_remove)
    assert n_removed, "nothing was removed — is the source the full Fe-56 tape?"
    dest.write_text(trimmed)


def build_tsl(source: Path, dest: Path, whole: bool = False) -> None:
    """Cut *source* (a TSL tape) down to ``KEEP_TSL_ELASTIC``, verbatim.

    With *whole*, the tape is copied instead. ``tsl-s-CH4.endf`` is 161 kB
    entire, and it is the only committed fixture carrying an MF7/MT4 — cutting
    anything out of it would be paying its size and losing its point.

    The copy goes through ``read_text``/``write_text`` rather than
    ``shutil.copy`` so a CRLF source lands as LF like every other fixture; the
    CRLF path itself is covered by the ``tape``-marked tests on the real JEFF
    tapes.
    """
    content = source.read_text()
    if whole:
        dest.write_text(content)
        return

    inventory = section_inventory(content)
    to_remove: list[tuple[int, int | None]] = []
    for mf, mts in sorted(inventory.items()):
        if mf not in KEEP_TSL_ELASTIC:
            to_remove.append((mf, None))
            continue
        for mt in sorted(set(mts) - KEEP_TSL_ELASTIC[mf]):
            to_remove.append((mf, mt))

    trimmed, n_removed = remove_sections(content, to_remove)
    assert n_removed, f"nothing was removed from {source.name} — is it a TSL tape?"
    dest.write_text(trimmed)


def build_cov(dest: Path) -> None:
    """Write a small synthetic tape carrying MF33/MT2 and MF34/MT2."""
    def _line(values, mat, mf, mt):
        return format_endf_data_line(
            values, mat, mf, mt, 0, formats=[ENDF_FORMAT_INT] * 6
        )

    # A host with just enough structure for the covariance writers to splice
    # into: one MF3/MT2 stub, its SEND, the MF FEND and the tape MEND.
    # The terminators go through the shared emitters: writing them by hand with
    # six integer formats is the defect this fixture would otherwise re-bake
    # into itself on every regeneration.
    stub = "\n".join([
        _line([int(ZA), 0, 0, 0, 0, 0], MAT, 3, MT),
        format_endf_send_record(MAT, 3),
        format_endf_fend_record(MAT),
        format_endf_mend_record(),
    ]) + "\n"

    tmp_host = dest.with_suffix(".host.tmp")
    tmp_mid = dest.with_suffix(".mid.tmp")
    try:
        tmp_host.write_text(stub)
        mf33 = create_mf33_from_covariance(_mf33_matrix(), COV_GRID, ZA, AWR, MAT, MT)
        write_mf33_to_file(str(tmp_host), mf33, str(tmp_mid), update_directory=False)

        mf34 = create_mf34_from_covariance(
            _mf34_matrix(), COV_GRID, max_order=COV_MAX_ORDER,
            za=ZA, awr=AWR, mat=MAT, mt=MT,
        )
        write_mf34_to_file(str(tmp_mid), mf34, str(dest), update_directory=False)
    finally:
        for tmp in (tmp_host, tmp_mid):
            if tmp.exists():
                tmp.unlink()


# ---------------------------------------------------------------------------
# The PFNS fixtures
# ---------------------------------------------------------------------------

def build_pfns(source: Path, dest: Path) -> None:
    """Cut *source* (ENDF/B-VIII.1 Cf-252) down to ``KEEP_PFNS``, verbatim.

    Same machinery as :func:`build_structural`, and for the same reason: the
    point of the fixture is that MF5/MT18's LF=1 tables, MT455's LF=5 laws and
    MF35's LB=7 bands are the evaluator's bytes, quirks included. Regenerating
    them with kika's own writers would make the round-trip test circular.
    """
    content = source.read_text()
    inventory = section_inventory(content)

    to_remove: list[tuple[int, int | None]] = []
    for mf, mts in sorted(inventory.items()):
        if mf not in KEEP_PFNS:
            to_remove.append((mf, None))
            continue
        for mt in sorted(set(mts) - KEEP_PFNS[mf]):
            to_remove.append((mf, mt))

    trimmed, n_removed = remove_sections(content, to_remove)
    assert n_removed, "nothing was removed — is the source the full Cf-252 tape?"
    dest.write_text(trimmed)


def build_nubar(source: Path, dest: Path) -> None:
    """Cut *source* down to ``KEEP_NUBAR``, verbatim.

    Same machinery and same reason as :func:`build_structural`. It matters more
    here than elsewhere that nothing is reformatted: the MF1 nu-bar sections are
    held to **byte identity** through the model
    (``model_adapter/tests/test_nubar_round_trip.py``), so a fixture kika had
    rewritten would be a gate against kika's own output.
    """
    content = source.read_text()
    inventory = section_inventory(content)

    to_remove: list[tuple[int, int | None]] = []
    for mf, mts in sorted(inventory.items()):
        if mf not in KEEP_NUBAR:
            to_remove.append((mf, None))
            continue
        for mt in sorted(set(mts) - KEEP_NUBAR[mf]):
            to_remove.append((mf, mt))

    trimmed, n_removed = remove_sections(content, to_remove)
    assert n_removed, f"nothing was removed from {source} — wrong source tape?"
    dest.write_text(trimmed)


def build_mf32(source: Path, dest: Path) -> None:
    """Cut *source* down to ``KEEP_MF32``, verbatim.

    Same machinery and same reason as :func:`build_structural`: MF32 is exactly
    the file where a reformatted record would be a false pass, since the whole
    gate is that kika gives the bytes back. Two of these four evaluations
    disagree with ENDF-102 §32 in fields kika reproduces rather than corrects,
    and a regenerated fixture would quietly agree with kika instead.
    """
    content = source.read_text()
    inventory = section_inventory(content)

    to_remove: list[tuple[int, int | None]] = []
    for mf, mts in sorted(inventory.items()):
        if mf not in KEEP_MF32:
            to_remove.append((mf, None))
            continue
        for mt in sorted(set(mts) - KEEP_MF32[mf]):
            to_remove.append((mf, mt))

    trimmed, n_removed = remove_sections(content, to_remove)
    assert n_removed, f"nothing was removed from {source} — wrong source tape?"
    dest.write_text(trimmed)


#: Outgoing-energy group boundaries of the synthetic MF35, in eV. Eight groups,
#: spanning a fission spectrum's range.
PFNS_COV_BOUNDARIES = np.array(
    [1.0e-5, 1.0e5, 5.0e5, 1.0e6, 2.0e6, 3.0e6, 5.0e6, 8.0e6, 2.0e7]
)

#: Incident-energy bands. The edges 1e-5, 1e6 and 2e7 are all MF5 incident
#: nodes below — the property measured on every real tape, reproduced here so
#: the fast lane exercises the same geometry as the slow one.
PFNS_COV_BANDS = [(1.0e-5, 1.0e6), (1.0e6, 2.0e7)]

#: MF5 incident nodes. Two inside the first band, three in the second.
PFNS_COV_INCIDENT = [1.0e-5, 5.0e5, 1.0e6, 1.0e7, 2.0e7]


def _pfns_chi(grid: np.ndarray, temperature: float) -> np.ndarray:
    """A Maxwellian ``sqrt(E')exp(-E'/T)``, normalised on the lin-lin table.

    Normalised against the *trapezoid* integral rather than the analytic one,
    so the table as written integrates to 1 in the representation ENDF will
    read it back in. A fixture that is 0.3 % off before anything touches it
    makes every normalisation assertion downstream negotiable.
    """
    values = np.sqrt(grid) * np.exp(-grid / temperature)
    return values / np.trapezoid(values, grid)


def _pfns_outgoing_grids() -> list[np.ndarray]:
    """Per-incident-node outgoing grids, deliberately not all the same.

    Nodes 0-1 use a 21-point grid and nodes 2-4 a 26-point one, and neither
    contains every MF35 boundary. ENDF/B-VIII.1 U-235 varies its outgoing grid
    between incident points and its MF35 grid matches neither; a fixture with
    one shared grid would let a whole class of reconciliation bug through.
    """
    coarse = np.unique(np.concatenate([
        [1.0e-5], np.linspace(2.0e5, 2.0e7, 20),
    ]))
    fine = np.unique(np.concatenate([
        [1.0e-5], np.linspace(1.5e5, 2.0e7, 25),
    ]))
    return [coarse, coarse, fine, fine, fine]


def _pfns_cov_matrix(n: int, seed: int) -> np.ndarray:
    """A PSD matrix that is exactly zero-sum and genuinely rank-deficient.

    ``C = P M Pᵀ`` with ``P = I - 11ᵀ/n`` projects the constant vector out, so
    ``C·1 = 0`` to machine precision — the property MF35 has because group
    probabilities sum to one — and the rank drops by at least one. ``M`` is
    built from four random modes rather than a full-rank draw, so the fixture
    also reproduces the severe null space of the real bands (roughly two thirds
    of every one of them).
    """
    rng = np.random.default_rng(seed)
    modes = rng.normal(size=(n, 4))
    m = modes @ modes.T
    projector = np.eye(n) - np.ones((n, n)) / n
    c = projector @ m @ projector.T
    return 0.5 * (c + c.T)


def build_pfns_cov(dest: Path) -> None:
    """Write the synthetic MF5/MT18 + MF35/MT18 tape."""
    grids = _pfns_outgoing_grids()
    temperatures = [1.2e6, 1.25e6, 1.3e6, 1.5e6, 1.6e6]

    partial = MF5PartialTabulated(
        u=0.0, lf=1,
        p_interp=[(2, 2)],
        p_energies=[1.0e-5, 2.0e7],
        p_values=[1.0, 1.0],
        tab2_interp=[(len(PFNS_COV_INCIDENT), 2)],
        incident_energies=list(PFNS_COV_INCIDENT),
        outgoing_grids=[[float(v) for v in g] for g in grids],
        chi=[[float(v) for v in _pfns_chi(g, t)]
             for g, t in zip(grids, temperatures)],
        outgoing_interp=[[(len(g), 2)] for g in grids],
    )
    mf5 = MF5MT(number=PFNS_MT, _za=PFNS_ZA, _awr=PFNS_AWR, _nk=1,
                _mat=PFNS_MAT, partials=[partial])

    # Scale each band so sqrt(diag C)/P0 lands in the few-per-cent range the
    # real evaluations state, using the spectrum at the band's lower edge.
    n_groups = len(PFNS_COV_BOUNDARIES) - 1
    bands = []
    for index, (e1, e2) in enumerate(PFNS_COV_BANDS):
        node = PFNS_COV_INCIDENT.index(e1)
        p0 = partial.group_integrals(node, PFNS_COV_BOUNDARIES)
        raw = _pfns_cov_matrix(n_groups, seed=index)
        scale = np.sqrt(np.mean(np.diag(raw)))
        matrix = raw * (0.05 * np.mean(p0) / scale) ** 2
        upper = np.concatenate([matrix[row, row:] for row in range(n_groups)])
        values = list(PFNS_COV_BOUNDARIES) + list(upper)
        bands.append(MF35SubSection(
            e1=e1, e2=e2, ls=1, lb=7,
            nt=len(values), ne=len(PFNS_COV_BOUNDARIES),
            boundaries=list(PFNS_COV_BOUNDARIES),
            upper_triangle=list(upper),
            raw_list_values=values,
        ))

    mf35 = MF35MT(number=PFNS_MT, _za=PFNS_ZA, _awr=PFNS_AWR,
                  _nk=len(bands), _mat=PFNS_MAT, subsections=bands)

    dest.write_text("\n".join([
        str(mf5),
        format_endf_fend_record(PFNS_MAT),
        str(mf35),
        format_endf_fend_record(PFNS_MAT),
        format_endf_mend_record(),
        format_endf_tend_record(),
    ]) + "\n")


@pytest.mark.skipif(not REGEN, reason="set REGEN_MICRO_TAPES=1 to rebuild the fixtures")
def test_regenerate_micro_tapes(fe56_host_tape, cf252_b81_tape, u235_b81_tape, request):
    """Rebuild every fixture from its source. Opt-in, then commit the diff."""
    DATA.mkdir(parents=True, exist_ok=True)
    build_structural(Path(fe56_host_tape), STRUCTURAL)
    build_structural(Path(fe56_host_tape), MF33, keep=KEEP_MF33)
    build_cov(COV)
    build_pfns(Path(cf252_b81_tape), PFNS)
    build_pfns_cov(PFNS_COV)
    build_nubar(Path(u235_b81_tape), NUBAR)
    for key, tape in MF32_FIXTURES.items():
        source = request.getfixturevalue(f"{tape}_tape")
        build_mf32(Path(source), mf32_fixture_path(key))
    for key, tape in TSL_FIXTURES.items():
        source = request.getfixturevalue(f"{tape}_tape")
        build_tsl(Path(source), tsl_fixture_path(key), whole=(key == "sch4"))
    for key, tape in MF6_FIXTURES.items():
        source = request.getfixturevalue(f"{tape}_tape")
        build_mf6(Path(source), mf6_fixture_path(key), KEEP_MF6[key])
    assert all(p.stat().st_size > 0
               for p in (STRUCTURAL, MF33, COV, PFNS, PFNS_COV, NUBAR))
    assert all(mf32_fixture_path(k).stat().st_size > 0 for k in MF32_FIXTURES)
    assert all(tsl_fixture_path(k).stat().st_size > 0 for k in TSL_FIXTURES)
    assert all(mf6_fixture_path(k).stat().st_size > 0 for k in MF6_FIXTURES)
    assert all(mf6_fixture_path(k).stat().st_size > 0 for k in MF6_CP_FIXTURES)


@pytest.mark.skipif(not REGEN, reason="set REGEN_MICRO_TAPES=1 to rebuild the fixtures")
def test_regenerate_mf6_micro_tapes(be9_b81_tape, li6_b81_tape, c12_b81_tape,
                                    u235_b81_tape, request):
    """Rebuild just the MF6 fixtures.

    Separate from :func:`test_regenerate_micro_tapes` on purpose. The six older
    fixtures still carry the 80-column MF1/451 that ``update_mf1_directory``
    wrote before it was fixed (``docs/library/mf7_tsl_notes.md``), so a full regen
    rewrites them — a change worth making, and not worth making *as a side
    effect* of adding a new MF. This test builds the new ones and touches
    nothing else.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    for key, tape in MF6_FIXTURES.items():
        source = request.getfixturevalue(f"{tape}_tape")
        build_mf6(Path(source), mf6_fixture_path(key), KEEP_MF6[key])
    assert all(mf6_fixture_path(k).stat().st_size > 0 for k in MF6_FIXTURES)


@pytest.mark.skipif(not REGEN, reason="set REGEN_MICRO_TAPES=1 to rebuild the fixtures")
def test_regenerate_mf6_charged_particle_micro_tapes(
        p_he3_b80_tape, d_h2_b80_tape, h3_he4_b80_tape, a_he4_b80_tape,
        t_li7_b80_tape, request):
    """Copy the five charged-particle tapes verbatim.

    No builder and no ``KEEP`` dict: these are whole evaluations, and the copy
    goes through ``read_text``/``write_text`` only so a CRLF source would land
    as LF like every other fixture. See ``MF6_CP_FIXTURES`` for why whole.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    for key, tape in MF6_CP_FIXTURES.items():
        source = Path(request.getfixturevalue(f"{tape}_tape"))
        mf6_fixture_path(key).write_text(source.read_text())
    assert all(mf6_fixture_path(k).stat().st_size > 0 for k in MF6_CP_FIXTURES)


# ---------------------------------------------------------------------------
# What the committed fixtures must satisfy
# ---------------------------------------------------------------------------

def test_mf33_inventory_is_exactly_what_we_kept():
    """The cross-reaction fixture kept MF3 and MF33 for MT4 and MT16, nothing else."""
    inventory = section_inventory(MF33.read_text())
    assert set(inventory) == set(KEEP_MF33), (
        f"MF set drifted: {sorted(inventory)} != {sorted(KEEP_MF33)}"
    )
    for mf, expected_mts in KEEP_MF33.items():
        assert set(inventory[mf]) == expected_mts, f"MF{mf} MT set drifted"


def test_mf33_parses_and_states_no_cross_block():
    """Two reactions, and no covariance between them — which is the point.

    The fixture earns its place by being the case the carrier cannot express:
    `CrossSectionCovariance.covariance_matrix` leaves the unstated MT4xMT16
    block as NaN. If a future JEFF revision started stating it, this fixture
    would stop exercising `docs/library/library-gaps.md` D11 and would say so here
    rather than by a distant test quietly passing for a new reason.
    """
    endf = read_endf(str(MF33))
    assert endf.zaid == 26056
    assert endf.mat == MAT
    assert sorted(endf.files) == [1, 3, 33]

    stated = set()
    for mt, section in endf.files[33].sections.items():
        covmat = section.to_xs_covmat()
        for row, col in zip(covmat.reaction_rows, covmat.reaction_cols):
            stated.add((int(row), int(col)))
    assert (4, 4) in stated and (16, 16) in stated
    assert (4, 16) not in stated and (16, 4) not in stated


def test_structural_inventory_is_exactly_what_we_kept(micro_tape):
    """No section survived the cut that should not have, and none was lost."""
    inventory = section_inventory(micro_tape.read_text())
    assert set(inventory) == set(KEEP), (
        f"MF set drifted: {sorted(inventory)} != {sorted(KEEP)}"
    )
    for mf, expected_mts in KEEP.items():
        assert set(inventory[mf]) == expected_mts, f"MF{mf} MT set drifted"


def test_structural_parses_and_carries_the_real_identity(micro_tape):
    """The cut tape is still Fe-56 as JEFF-4.0 wrote it."""
    endf = read_endf(str(micro_tape))
    assert endf.zaid == 26056
    assert endf.mat == MAT
    assert sorted(endf.files) == [1, 2, 3, 4, 34]

    # Full energy range preserved — the cut drops sections, never records.
    mf3mt2 = endf.files[3].sections[2]
    assert mf3mt2._energies[0] == pytest.approx(1e-5)
    assert mf3mt2._energies[-1] == pytest.approx(1.5e8)
    assert np.all(np.diff(np.asarray(mf3mt2._energies)) >= 0)


def test_structural_directory_already_agrees_with_its_content(micro_tape, tmp_path):
    """Rebuilding the MF1/451 directory must be a no-op on the committed file.

    ``remove_sections`` refreshes the directory as it cuts. If a later edit
    ever leaves the two out of step, this is what catches it.
    """
    work = tmp_path / micro_tape.name
    shutil.copy2(micro_tape, work)
    before = work.read_bytes()
    assert update_mf1_directory(str(work)) is True
    assert work.read_bytes() == before, "MF1/451 directory does not match content"


def test_cov_tape_round_trips_both_covariances(micro_cov_tape):
    """The synthetic tape gives back the matrices it was built from."""
    endf = read_endf(str(micro_cov_tape))
    assert sorted(endf.files) == [3, 33, 34]

    got33 = np.asarray(endf.files[33].sections[MT].to_xs_covmat().matrices[0])
    np.testing.assert_allclose(got33, _mf33_matrix(), atol=1e-9)

    mf34 = endf.files[34].sections[MT]
    assert mf34 is not None, "MF34/MT2 did not survive the write-read cycle"


def test_pfns_inventory_is_exactly_what_we_kept(micro_pfns_tape):
    """No section survived the PFNS cut that should not have, and none was lost."""
    inventory = section_inventory(micro_pfns_tape.read_text())
    assert set(inventory) == set(KEEP_PFNS), (
        f"MF set drifted: {sorted(inventory)} != {sorted(KEEP_PFNS)}"
    )
    for mf, expected_mts in KEEP_PFNS.items():
        assert set(inventory[mf]) == expected_mts, f"MF{mf} MT set drifted"


def test_nubar_inventory_is_exactly_what_we_kept(micro_nubar_tape):
    """No section survived the nu-bar cut that should not have, and none was lost."""
    inventory = section_inventory(micro_nubar_tape.read_text())
    assert set(inventory) == set(KEEP_NUBAR), (
        f"MF set drifted: {sorted(inventory)} != {sorted(KEEP_NUBAR)}"
    )
    for mf, expected_mts in KEEP_NUBAR.items():
        assert set(inventory[mf]) == expected_mts, f"MF{mf} MT set drifted"


def test_nubar_tape_carries_the_representations_the_fixture_exists_for(
    micro_nubar_tape,
):
    """A real LNU=2 total and prompt, six real precursor families, and an
    NC-type MF31/452.

    The last of those is why this is U-235 and not something smaller: the total
    nu-bar covariance is *declared* as the sum of MT455 and MT456 (``LTY=0``)
    rather than stored, so it is the only fixture on which the NC resolution
    path runs at all. A cut that dropped MT455 or MT456 would leave the total
    unresolvable and the test that checks its weighting vacuous.
    """
    endf = read_endf(str(micro_nubar_tape))
    assert endf.zaid == 92235

    assert endf.files[1].sections[452].lnu == 2
    assert endf.files[1].sections[456].lnu == 2

    mt455 = endf.files[1].sections[455]
    assert mt455.ldg == 0
    assert len(mt455.decay_constants) == 6

    mf31mt452 = endf.files[31].sections[452]
    derived = [
        nc.lty
        for subsection in mf31mt452._subsections
        for nc in subsection.nc_records
    ]
    assert derived == [0], "MF31/452 is no longer the NC-type case"


def test_pfns_tape_carries_the_laws_the_fixture_exists_for(micro_pfns_tape):
    """The cut kept a real LF=1, a real LF=5 and four real LB=7 bands.

    The point of paying 770 kB is that these are the evaluator's records. If a
    future cut trimmed MT455 or dropped a band, the fixture would still parse
    and would silently stop covering the verbatim path and the band geometry.
    """
    endf = read_endf(str(micro_pfns_tape))
    assert endf.zaid == 98252

    mt18 = endf.files[5].sections[18]
    assert [p.lf for p in mt18.partials] == [1]
    assert len(mt18.tabulated_partials()) == 1
    assert mt18.report_gaps() == []

    mt455 = endf.files[5].sections[455]
    assert [p.lf for p in mt455.partials] == [5] * 6
    assert mt455.tabulated_partials() == []
    assert len(mt455.report_gaps()) == 6

    mf35 = endf.files[35].sections[18]
    assert mf35.num_bands == 4
    assert [b.ne for b in mf35.subsections] == [123] * 4
    assert all((b.ls, b.lb) == (1, 7) for b in mf35.subsections)


def test_pfns_cov_tape_is_the_geometry_the_sampler_fast_lane_needs(
    micro_pfns_cov_tape,
):
    """Small, but with every structural property of the real thing.

    Eight groups over two bands, band edges that are MF5 incident nodes,
    outgoing grids that differ between incident nodes and match no MF35
    boundary set, a rank-deficient covariance, and vanishing row sums. A
    fixture that dropped any one of those would let the corresponding bug
    through while still looking like a PFNS tape.
    """
    endf = read_endf(str(micro_pfns_cov_tape))
    partial = endf.files[5].sections[18].partials[0]
    bands = endf.files[35].sections[18].subsections

    assert partial.lf == 1
    assert partial.incident_energies == PFNS_COV_INCIDENT
    assert len(bands) == 2

    for (e1, e2), band in zip(PFNS_COV_BANDS, bands):
        assert band.e1 == pytest.approx(e1) and band.e2 == pytest.approx(e2)
        assert e1 in partial.incident_energies, "band edge is not an MF5 node"

        matrix = band.matrix()
        assert matrix.shape == (8, 8)
        np.testing.assert_allclose(matrix, matrix.T, atol=0)
        assert band.row_sum_residual() < 1e-6, "C·1 is not ~0"

        eigenvalues = np.linalg.eigvalsh(matrix)
        n_null = int((eigenvalues <= 1e-10 * eigenvalues.max()).sum())
        assert n_null >= 1, "the fixture is full rank; the real bands are not"

    # Outgoing grids differ between incident nodes, and neither is the MF35 grid.
    grids = {tuple(g) for g in partial.outgoing_grids}
    assert len(grids) == 2, "every incident node shares one outgoing grid"
    for grid in grids:
        assert not set(PFNS_COV_BOUNDARIES[1:-1]).issubset(grid)

    for k in range(len(partial.incident_energies)):
        assert partial.normalisation(k) == pytest.approx(1.0, abs=1e-6)


def test_micro_tapes_stay_small(micro_tape, micro_cov_tape):
    """A fixture that grows without anyone noticing stops being a fixture.

    The structural tape is ~2 MB of ENDF text, which git stores in ~0.5 MB.
    The ceiling is deliberately close to the current size: crossing it should
    be a decision, not an accident.
    """
    assert micro_tape.stat().st_size < 2_500_000
    assert micro_cov_tape.stat().st_size < 100_000


def test_tsl_micro_tapes_stay_small(micro_tsl_tape):
    """Same ceiling logic, and it bites harder here than anywhere else.

    A TSL evaluation is 160 kB at its smallest and 93 MB at its largest, so the
    difference between a fixture and an accident is one wrong source file. The
    four together are ~880 kB; the bound is per-tape and close.
    """
    assert micro_tsl_tape.stat().st_size < 400_000


def test_charged_particle_micro_tapes_stay_small():
    """These five are copied whole, so the ceiling is what keeps that honest.

    Whole-copy is defensible only while the tapes are small. The largest
    charged-particle evaluation in ENDF/B-VIII.0 is 3.6 MB; if a fixture is ever
    re-pointed at one of those, whole-copy stops being free and the cut has to
    come back. 64 kB is just above ``t_li7`` at 56 kB.
    """
    for key in MF6_CP_FIXTURES:
        size = mf6_fixture_path(key).stat().st_size
        assert size < 64_000, f"{key} is {size} bytes -- copy it whole no longer"


@pytest.mark.parametrize("key,expected", [
    ("sch4", {1: {451}, 7: {2, 4}}),
    ("bemetal_elastic", {1: {451}, 7: {2, 451}}),
    ("un_elastic", {1: {451}, 7: {2, 451}}),
    ("jeff_be_elastic", {1: {451}, 7: {2}}),
])
def test_tsl_inventories_are_exactly_what_we_kept(key, expected):
    """The cut kept the elastic sections and dropped MF7/MT4, nothing else.

    Worth asserting per fixture rather than as one set: ``jeff_be_elastic`` has
    no MF7/451 to keep and ``sch4`` was not cut at all, so a single shared
    expectation would have to be loose enough to miss a bad cut.
    """
    inventory = section_inventory(tsl_fixture_path(key).read_text())
    assert {mf: set(mts) for mf, mts in inventory.items()} == expected


def test_the_tsl_fixtures_cover_every_lthr_branch():
    """One fixture per elastic representation, so none is left untested.

    LTHR=3 is the one this exists to protect. It is 11 files of 114, it is the
    branch a ``lthr == 1`` test silently mishandles, and without a committed
    fixture it would only ever be exercised by a ``tape``-marked test that skips
    on a machine without the share.
    """
    seen = {}
    for key in TSL_FIXTURES:
        endf = read_endf(str(tsl_fixture_path(key)), mf_numbers=[7])
        mt2 = endf.files[7].sections.get(2)
        if mt2 is not None:
            seen[key] = mt2.lthr
    assert sorted(set(seen.values())) == [1, 2, 3], seen


def test_pfns_micro_tapes_stay_small(micro_pfns_tape, micro_pfns_cov_tape):
    """Same ceiling logic as above: growth should be a decision.

    The PFNS slice is ~770 kB, nearly all of it MF5/MT18 and MF35/MT18 — the
    two sections the fixture exists for. Adding another MT would roughly double
    it, which is why the bound is close.
    """
    assert micro_pfns_tape.stat().st_size < 1_000_000
    assert micro_pfns_cov_tape.stat().st_size < 50_000
