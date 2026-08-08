"""Can the chi2 read the cross term out of MF34's a_0 blocks? (roadmap §10.7-4 step 5)

The cross term has never been scored. Its only source was a `.npy` sidecar
declaring `is_relative=False` against a relative MF34 family, which
`build_mf33_mf34_cross_block` refuses (§L13) -- correctly, because the sidecar
was never a draw from the same distribution as the marginals.

Reading it from the FILE fixes that at the root: one ENDF, one convention, one
shape grid, and a magnitude axis that is the file's own MF33 grid. These tests
pin the reader that makes it possible, and -- more importantly -- pin the four
guards, because reading the cross term separately from the marginals is exactly
the shape of §L, §L3 and §L9, where a joint was certified that was not the joint
being shipped.

The round trip is asserted as an IDENTITY at ENDF's precision, not as a
tolerance: every value here survives one 11-character serialisation and nothing
else touches it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kika.endf.writers.mf34_writer import (  # noqa: E402
    _make_lb5_record,
    _make_lb6_record,
    create_mf34_from_covariance,
    write_mf34_to_file,
)

from scripts.eval_covariance import build_mf33_mf34_cross_block  # noqa: E402
from scripts.mf34_cross_reader import read_mf34_split  # noqa: E402

ZA, AWR, MAT, MT = 26056.0, 55.454, 2631, 2
ISO = 26056
L_MAX = 3
# The magnitude axis is DELIBERATELY finer than the shape axis, because that is
# the shipped geometry: MF33 fine (1738 bins inside a 2317-bin grid), MF34
# grouped (703). A square fixture would hide the whole problem.
SHAPE_GRID = np.array([0.85e6, 1.6e6, 2.7e6, 4.0e6])          # 3 shape bins
MAG_GRID = np.array([0.85e6, 1.2e6, 1.6e6, 2.2e6, 2.7e6, 4.0e6])  # 5 mag bins
N_SH, N_MAG = SHAPE_GRID.size - 1, MAG_GRID.size - 1


def _shape_cov() -> np.ndarray:
    base = np.arange(1, N_SH * L_MAX + 1, dtype=float)
    cov = 0.0005 * np.outer(base, base) + np.diag(0.01 * base)
    return 0.5 * (cov + cov.T)


def _cross_dict() -> dict:
    rng = np.random.default_rng(20260808)
    return {l1: np.round(rng.uniform(-0.02, 0.02, size=(N_MAG, N_SH)), 6)
            for l1 in range(1, L_MAX + 1)}


def _find(mf34, l, l1):
    for ss in mf34._subsections[0].sub_subsections:
        if ss.l == l and ss.l1 == l1:
            return ss
    raise AssertionError(f"no ({l}, {l1}) sub-subsection")


def _build(cross=None, mag_grid=MAG_GRID):
    return create_mf34_from_covariance(
        _shape_cov(), SHAPE_GRID, L_MAX, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=cross, cross_energy_grid_ev=mag_grid,
    )


def _write(mf34, micro_cov_tape, tmp_path, name="with_a0.endf") -> Path:
    out = tmp_path / name
    write_mf34_to_file(str(micro_cov_tape), mf34, str(out))
    return out


@pytest.fixture
def file_with_a0(micro_cov_tape, tmp_path):
    return _write(_build(_cross_dict()), micro_cov_tape, tmp_path)


# ── the round trip ────────────────────────────────────────────────────────────

def test_the_cross_blocks_come_back_on_their_native_rectangular_grids(file_with_a0):
    """⚑ THE POINT. `to_ang_covmat` squares an LB=6 block onto union(row, col);
    this reader does not, so the magnitude axis survives."""
    want = _cross_dict()
    res = read_mf34_split(file_with_a0, isotope=ISO, mt=MT, l_max=L_MAX,
                          mf33_grid_ev=MAG_GRID)

    assert [b["l"] for b in res.cross] == [1, 2, 3]
    assert res.info["n_mag_bins"] == N_MAG
    assert res.info["n_shape_bins"] == N_SH
    for b in res.cross:
        assert b["matrix"].shape == (N_MAG, N_SH), "the block must stay RECTANGULAR"
        np.testing.assert_array_equal(b["shape_grid_ev"], SHAPE_GRID)
        # The values are pre-rounded to 6 significant digits, so the ONLY thing
        # between write and read is float64's representation of that decimal --
        # one ULP, ~2e-16 relative. Anything above that is the reader altering
        # the block, which is what this test is for. Not a drift tolerance.
        np.testing.assert_allclose(b["matrix"], want[b["l"]], rtol=1e-15, atol=0)
        assert b["is_relative"] is True


def test_the_marginals_lose_a0_and_are_not_lifted_onto_a_union_grid(file_with_a0):
    """The 440 MB claim. Left in, each a_0 block is projected onto
    union(mag, shape) and retained for the life of the load, consumed by
    nothing -- `build_mf34_block` skips `l_r < 1`."""
    res = read_mf34_split(file_with_a0, isotope=ISO, mt=MT, l_max=L_MAX,
                          mf33_grid_ev=MAG_GRID)
    mf34 = res.mf34
    assert 0 not in set(mf34.l_rows) | set(mf34.l_cols), "a_0 must not reach the family"
    assert mf34.num_matrices == L_MAX * (L_MAX + 1) // 2
    for i in range(mf34.num_matrices):
        g = np.asarray(mf34.energy_grids[i], dtype=float)
        assert g.size == SHAPE_GRID.size, (
            f"block {i} came back on {g.size} edges; the union with the "
            f"{MAG_GRID.size}-edge magnitude grid would give "
            f"{np.union1d(g, MAG_GRID).size}"
        )


def test_a_file_without_a0_reads_as_before_and_yields_no_cross(micro_cov_tape, tmp_path):
    """JEFF-4.0 and JENDL-5 are this case, and they must stay untouched."""
    path = _write(_build(None), micro_cov_tape, tmp_path, "plain.endf")
    res = read_mf34_split(path, isotope=ISO, mt=MT, l_max=L_MAX)
    assert res.cross == []
    assert res.info["n_a0_subsubsections"] == 0
    assert res.mf34.num_matrices == L_MAX * (L_MAX + 1) // 2

    with pytest.raises(ValueError, match="no MF34 a_0 blocks"):
        read_mf34_split(path, isotope=ISO, mt=MT, l_max=L_MAX, require_cross=True)


def test_orders_above_l_max_are_dropped_like_the_self_blocks(file_with_a0):
    res = read_mf34_split(file_with_a0, isotope=ISO, mt=MT, l_max=2,
                          mf33_grid_ev=MAG_GRID)
    assert [b["l"] for b in res.cross] == [1, 2]


# ── the four guards ───────────────────────────────────────────────────────────

def test_a_magnitude_grid_that_is_not_the_files_mf33_grid_is_refused(file_with_a0):
    """A cross term is Cauchy-Schwarz-compatible only with the marginals it was
    built from (§10.1). Regridding it here is what runs 89/90 did."""
    wrong = np.array([0.85e6, 2.0e6, 4.0e6])
    with pytest.raises(ValueError, match="magnitude grid"):
        read_mf34_split(file_with_a0, isotope=ISO, mt=MT, l_max=L_MAX,
                        mf33_grid_ev=wrong)


def test_a_shape_grid_the_self_blocks_do_not_share_is_refused(micro_cov_tape, tmp_path):
    """⚑ THE GUARD ROUTING THE CROSS THROUGH A SECOND READER OWES.

    The fold applies `PointMap.nearest(col_grid)` to the cross and
    `PointMap.nearest(block_grid)` to the self blocks. Different grids, no
    single M, no congruence -- §L18's four-grid problem, which `merge_mf34`
    produces for real by taking a per-pair union.
    """
    mf34 = _build(_cross_dict())
    coarse = np.array([0.85e6, 2.7e6, 4.0e6])
    ss = _find(mf34, 0, 1)
    ss.records[0] = _make_lb6_record(np.zeros((N_MAG, coarse.size - 1)),
                                     MAG_GRID, coarse)
    path = _write(mf34, micro_cov_tape, tmp_path, "bad_col.endf")
    with pytest.raises(ValueError, match="column grid"):
        read_mf34_split(path, isotope=ISO, mt=MT, l_max=L_MAX,
                        mf33_grid_ev=MAG_GRID)


def test_self_blocks_on_several_grids_are_refused_before_the_cross_is_read(
        micro_cov_tape, tmp_path):
    """The same defect seen from the other side: it is the shipped MF34 that
    has four grids today (673/676/684/703), not the cross block."""
    mf34 = _build(_cross_dict())
    coarse = np.array([0.85e6, 2.7e6, 4.0e6])
    ss = _find(mf34, L_MAX, L_MAX)
    ss.records[0] = _make_lb5_record(np.zeros((coarse.size - 1,) * 2), coarse)
    path = _write(mf34, micro_cov_tape, tmp_path, "four_grids.endf")
    with pytest.raises(ValueError, match="four-grid|sits on a grid"):
        read_mf34_split(path, isotope=ISO, mt=MT, l_max=L_MAX,
                        mf33_grid_ev=MAG_GRID)


def test_a_non_null_magnitude_self_block_is_refused(micro_cov_tape, tmp_path):
    """(0,0) belongs to MF33 (manual Sec. 34.3). Folded as well as the MF33
    self block it would double the magnitude variance, silently."""
    mf34 = _build(_cross_dict())
    ss = _find(mf34, 0, 0)
    ss.records[0] = _make_lb5_record(np.full((N_MAG, N_MAG), 1e-3), MAG_GRID)
    path = _write(mf34, micro_cov_tape, tmp_path, "fat_00.endf")
    with pytest.raises(ValueError, match=r"\(0,0\) block is not null"):
        read_mf34_split(path, isotope=ISO, mt=MT, l_max=L_MAX,
                        mf33_grid_ev=MAG_GRID)


def test_a0_blocks_with_the_wrong_ltt_are_refused(micro_cov_tape, tmp_path):
    """LTT=3 is the format's own flag for 'either L or L1=0 anywhere in the
    Section'. A file disagreeing with itself is misread by everyone else."""
    mf34 = _build(_cross_dict())
    mf34._ltt = 1
    path = _write(mf34, micro_cov_tape, tmp_path, "bad_ltt.endf")
    with pytest.raises(ValueError, match="LTT"):
        read_mf34_split(path, isotope=ISO, mt=MT, l_max=L_MAX,
                        mf33_grid_ev=MAG_GRID)


# ── it reaches the fold ───────────────────────────────────────────────────────

def test_the_blocks_are_accepted_by_the_fold_and_move_sigma(file_with_a0):
    """End of the chain: the file's own blocks pass both §10.7-2 guards.

    `is_relative=True` from the reader against `a_is_relative=True` from the
    family is §L13 satisfied STRUCTURALLY -- one file, one convention -- rather
    than by a flag someone set.
    """
    res = read_mf34_split(file_with_a0, isotope=ISO, mt=MT, l_max=L_MAX,
                          mf33_grid_ev=MAG_GRID)
    rng = np.random.default_rng(7)
    n = 40
    e_mev = rng.uniform(0.9, 3.9, n)
    mf4_mev = MAG_GRID / 1e6
    kw = dict(mf33_grid_ev=MAG_GRID, energies_mf4_mev=mf4_mev, a_is_relative=True)
    args = (e_mev, rng.uniform(-1, 1, n), rng.uniform(0.8, 1.2, n),
            rng.uniform(0.3, 0.9, (n, L_MAX)), rng.uniform(0.5, 1.5, n))

    sigma = build_mf33_mf34_cross_block(res.cross, *args, **kw)
    assert sigma.shape == (n, n)
    assert np.abs(sigma).max() > 0.0, "the cross term must actually contribute"
    np.testing.assert_allclose(sigma, sigma.T, rtol=0, atol=1e-15)

    zero = build_mf33_mf34_cross_block([], *args, **kw)
    np.testing.assert_array_equal(zero, np.zeros((n, n)))


def test_the_fold_still_refuses_the_units_mismatch_on_file_borne_blocks(file_with_a0):
    """The §L13 guard is not weakened by the source changing."""
    res = read_mf34_split(file_with_a0, isotope=ISO, mt=MT, l_max=L_MAX,
                          mf33_grid_ev=MAG_GRID)
    rng = np.random.default_rng(3)
    n = 20
    with pytest.raises(ValueError, match="two units"):
        build_mf33_mf34_cross_block(
            res.cross, rng.uniform(0.9, 3.9, n), rng.uniform(-1, 1, n),
            rng.uniform(0.8, 1.2, n), rng.uniform(0.3, 0.9, (n, L_MAX)),
            rng.uniform(0.5, 1.5, n),
            mf33_grid_ev=MAG_GRID, energies_mf4_mev=MAG_GRID / 1e6,
            a_is_relative=False,
        )
