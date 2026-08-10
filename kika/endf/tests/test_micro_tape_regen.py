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

REGEN = bool(os.environ.get("REGEN_MICRO_TAPES"))

#: What the structural micro-tape keeps. Everything else is removed.
KEEP = {1: {451}, 2: {151}, 3: {1, 2, 102}, 4: {2}, 34: {2}}

#: What the PFNS micro-tape keeps, cut the same verbatim way from
#: ENDF/B-VIII.1 Cf-252. MF3/MT18 comes along because the fission cross section
#: is what MT18's spectrum belongs to, and it costs 34 lines.
KEEP_PFNS = {1: {451}, 3: {18}, 5: {18, 455}, 35: {18}}

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

def build_structural(source: Path, dest: Path) -> None:
    """Cut *source* down to ``KEEP`` and write it to *dest*, verbatim."""
    content = source.read_text()
    inventory = section_inventory(content)

    to_remove: list[tuple[int, int | None]] = []
    for mf, mts in sorted(inventory.items()):
        if mf not in KEEP:
            to_remove.append((mf, None))
            continue
        for mt in sorted(set(mts) - KEEP[mf]):
            to_remove.append((mf, mt))

    trimmed, n_removed = remove_sections(content, to_remove)
    assert n_removed, "nothing was removed — is the source the full Fe-56 tape?"
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
def test_regenerate_micro_tapes(fe56_host_tape, cf252_b81_tape):
    """Rebuild every fixture from its source. Opt-in, then commit the diff."""
    DATA.mkdir(parents=True, exist_ok=True)
    build_structural(Path(fe56_host_tape), STRUCTURAL)
    build_cov(COV)
    build_pfns(Path(cf252_b81_tape), PFNS)
    build_pfns_cov(PFNS_COV)
    assert all(p.stat().st_size > 0 for p in (STRUCTURAL, COV, PFNS, PFNS_COV))


# ---------------------------------------------------------------------------
# What the committed fixtures must satisfy
# ---------------------------------------------------------------------------

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


def test_pfns_micro_tapes_stay_small(micro_pfns_tape, micro_pfns_cov_tape):
    """Same ceiling logic as above: growth should be a decision.

    The PFNS slice is ~770 kB, nearly all of it MF5/MT18 and MF35/MT18 — the
    two sections the fixture exists for. Adding another MT would roughly double
    it, which is why the bound is close.
    """
    assert micro_pfns_tape.stat().st_size < 1_000_000
    assert micro_pfns_cov_tape.stat().st_size < 50_000
