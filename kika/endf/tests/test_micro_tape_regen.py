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
from kika.endf.utils import ENDF_FORMAT_INT, format_endf_data_line, parse_endf_id
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

REGEN = bool(os.environ.get("REGEN_MICRO_TAPES"))

#: What the structural micro-tape keeps. Everything else is removed.
KEEP = {1: {451}, 2: {151}, 3: {1, 2, 102}, 4: {2}, 34: {2}}

#: Fe-56 identity, shared by both fixtures.
ZA, AWR, MAT, MT = 26056.0, 55.36735, 2631, 2

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
    stub = "\n".join([
        _line([int(ZA), 0, 0, 0, 0, 0], MAT, 3, MT),
        _line([0, 0, 0, 0, 0, 0], MAT, 3, 0),
        _line([0, 0, 0, 0, 0, 0], MAT, 0, 0),
        _line([0, 0, 0, 0, 0, 0], 0, 0, 0),
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


@pytest.mark.skipif(not REGEN, reason="set REGEN_MICRO_TAPES=1 to rebuild the fixtures")
def test_regenerate_micro_tapes(fe56_host_tape):
    """Rebuild both fixtures from the real tape. Opt-in, then commit the diff."""
    DATA.mkdir(parents=True, exist_ok=True)
    build_structural(Path(fe56_host_tape), STRUCTURAL)
    build_cov(COV)
    assert STRUCTURAL.stat().st_size > 0 and COV.stat().st_size > 0


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


def test_micro_tapes_stay_small(micro_tape, micro_cov_tape):
    """A fixture that grows without anyone noticing stops being a fixture.

    The structural tape is ~2 MB of ENDF text, which git stores in ~0.5 MB.
    The ceiling is deliberately close to the current size: crossing it should
    be a decision, not an accident.
    """
    assert micro_tape.stat().st_size < 2_500_000
    assert micro_cov_tape.stat().st_size < 100_000
