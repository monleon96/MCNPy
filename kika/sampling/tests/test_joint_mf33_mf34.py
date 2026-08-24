"""The MF33xMF34 joint: assembly, the a₀ cross, and "the marginals did not move".

The fixture is a synthetic ``_a0cross`` tape built the way the pipeline builds
one — a single sample covariance of paired (c0, a_l) replicas, split into MF33,
MF34 and the (0,l) cross blocks — so the joint is PSD by construction and the
cross sits inside Cauchy-Schwarz. A cross drawn independently of its marginals
is not a covariance, and a test on one would measure the fixture rather than
the code: an early version of this file did exactly that and read
``max|rho| = 3.6``.

Per-order meshes are deliberate and are the case that matters: order 1 has 4
groups, order 2 has 2 and order 3 has 3, so the cross columns really do have to
be lifted onto each order's union grid rather than copied.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.writers import (
    create_mf33_from_covariance,
    create_mf34_from_covariance,
    write_mf33_to_file,
    write_mf34_to_file,
)
from kika.sampling.base_tape import COVARIANCE_MF, build_base_tape
from kika.sampling.core import draw_samples
from kika.sampling.endf_perturbation import load_mf34_suite
from kika.sampling.joint_mf33_mf34 import (
    JOINT_SETS,
    load_joint_mf33_mf34,
    restrict_joint,
)
from kika.sampling.model_blocks import (
    _lift_matrix,
    assemble_mf33_mf34_joint,
    legendre_covariance_blocks,
)

ZA, AWR, MAT, MT, ISO = 26056.0, 55.454437, 2631, 2, 26056
LMAX = 3
MAG = np.array([1e-5, 1.0e6, 2.0e6, 3.0e6, 2.0e7])
MESH = {1: np.array([1e-5, 1.0e6, 2.0e6, 3.0e6, 2.0e7]),
        2: np.array([1e-5, 2.0e6, 2.0e7]),
        3: np.array([1e-5, 1.0e6, 3.0e6, 2.0e7])}


def _build_tape(src, out, *, n_rep=4000, seed=7):
    rng = np.random.default_rng(seed)
    n0 = MAG.size - 1
    sizes = {l: MESH[l].size - 1 for l in range(1, LMAX + 1)}
    ntot = n0 + sum(sizes.values())

    lat = rng.normal(size=(n_rep, 3))
    load = rng.normal(size=(3, ntot))
    z = 0.05 * (lat @ load + 0.7 * rng.normal(size=(n_rep, ntot)))
    truth = np.cov(z, rowvar=False)

    mf33 = create_mf33_from_covariance(truth[:n0, :n0], MAG, ZA, AWR, MAT, MT)
    tmp = str(out) + ".mf33"
    write_mf33_to_file(str(src), mf33, tmp)

    # The a₀ ROW grid must be the file's own floats: that round trip is
    # idempotent, a reconstruction is not, and `read_mf34_split` checks it.
    back = read_endf(tmp, mf_numbers=[33])
    cov = back.get_file(33).sections[MT].to_xs_covmat(energy_unit="eV")
    i = next(k for k in range(len(cov.matrices))
             if cov.reaction_rows[k] == MT and cov.reaction_cols[k] == MT)
    mag_file = np.asarray(cov.energy_grids[i], dtype=float)
    c33_file = np.asarray(cov.matrices[i], dtype=float)

    off, k = {}, n0
    for l in sorted(sizes):
        off[l], k = k, k + sizes[l]
    shape = {(l, l1): truth[off[l]:off[l] + sizes[l], off[l1]:off[l1] + sizes[l1]]
             for l in sorted(sizes) for l1 in sorted(sizes) if l1 >= l}
    cross = {l: truth[:n0, off[l]:off[l] + sizes[l]] for l in sorted(sizes)}

    mf34 = create_mf34_from_covariance(
        shape, {l: MESH[l] for l in sorted(sizes)}, LMAX, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=cross, cross_energy_grid_ev=mag_file)
    write_mf34_to_file(tmp, mf34, str(out))
    return {"c33": c33_file, "mag_grid": mag_file, "cross": cross,
            "sizes": sizes, "n0": n0}


@pytest.fixture(scope="module")
def a0cross_tape(tmp_path_factory, micro_tape):
    out = tmp_path_factory.mktemp("a0cross") / "fe56_a0cross.endf"
    meta = _build_tape(micro_tape, out)
    return str(out), meta


@pytest.fixture(scope="module")
def joint(a0cross_tape):
    tape, meta = a0cross_tape
    blocks, index = load_joint_mf33_mf34(tape, mt=MT, isotope=ISO, l_max=LMAX)
    return blocks, index, meta, tape


# ── the layout ────────────────────────────────────────────────────────────────

def test_the_magnitude_block_is_the_file_s_own_mf33(joint):
    blocks, index, meta, _ = joint
    (_, m), = blocks
    m0 = index["n_sigma"]
    assert m0 == meta["n0"]
    np.testing.assert_array_equal(m[:m0, :m0], meta["c33"])


def test_the_shape_block_is_bit_for_bit_what_the_shipped_sampler_assembles(joint):
    """The gate the whole design turns on.

    ``perturb_ENDF_files`` assembles MF34 with ``legendre_covariance_blocks``
    and an ``orders`` filter that drops a₀. If the joint's shape block were even
    one ULP away from that, every existing MF34 ensemble would move the day this
    landed, and no comparison between an old run and a new one would mean
    anything. It is not a tolerance: the joint reaches the same object by
    stripping a₀ from the parsed section *before* the suite is decoded.
    """
    blocks, index, _, tape = joint
    (_, m), = blocks
    suite = load_mf34_suite(tape)
    (_, shipped), = legendre_covariance_blocks(
        suite, mt=[MT], orders=list(range(1, LMAX + 1)), relative=True)
    m0 = index["n_sigma"]
    assert np.array_equal(m[m0:, m0:], shipped)


def test_every_order_gets_its_cross_block_on_its_own_union_grid(joint):
    blocks, index, meta, _ = joint
    (_, m), = blocks
    m0, stride = index["n_sigma"], index["stride"]
    assert index["cross_orders"] == sorted(meta["cross"])
    for l in index["cross_orders"]:
        slot = next(i for i, t in enumerate(index["triplets"]) if t[-1] == l)
        t = index["triplets"][slot]
        c0, w = m0 + slot * stride, index["widths"][t]
        got = m[:m0, c0:c0 + w]
        lift = _lift_matrix(MESH[l], np.asarray(index["grids"][t], float))
        want = meta["cross"][l] @ lift.T
        # 7 significant digits is what ENDF-6 writes, and that is the only
        # difference this comparison is allowed to see.
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-8)


def test_the_padding_columns_of_a_short_order_stay_zero():
    idx = {"triplets": [(1, 2, 1), (1, 2, 2)], "stride": 3,
           "widths": {(1, 2, 1): 3, (1, 2, 2): 2},
           "grids": {(1, 2, 1): np.array([0., 1., 2., 3.]),
                     (1, 2, 2): np.array([0., 1.5, 3.])}}
    m, i = assemble_mf33_mf34_joint(
        np.eye(2), np.array([0., 1., 2.]), np.eye(6), idx,
        {2: (np.full((2, 2), 0.2), np.array([0., 1.5, 3.]))})
    # order 2 occupies 3 slots but spans 2 bins; the third is inert.
    assert m[:2, -1].tolist() == [0.0, 0.0]
    assert i["cross_orders"] == [2]


def test_a_cross_block_for_an_order_with_no_marginal_is_refused():
    idx = {"triplets": [(1, 2, 1)], "stride": 2, "widths": {(1, 2, 1): 2},
           "grids": {(1, 2, 1): np.array([0., 1., 2.])}}
    with pytest.raises(ValueError, match="negative eigenvalue"):
        assemble_mf33_mf34_joint(
            np.eye(2), np.array([0., 1., 2.]), np.eye(2), idx,
            {5: (np.full((2, 2), 0.1), np.array([0., 1., 2.]))})


# ── what the object is ────────────────────────────────────────────────────────

def test_the_assembled_joint_is_psd_and_inside_cauchy_schwarz(joint):
    blocks, _, _, _ = joint
    (_, m), = blocks
    w = np.linalg.eigvalsh(0.5 * (m + m.T))
    assert w.min() / w.max() > -1e-10
    d = np.sqrt(np.diag(m))
    r = m / np.outer(d, d)
    # 5e-6 is the ASCII round trip: six significant digits, independently
    # rounded covariance and variance entries.
    assert np.abs(r).max() <= 1.0 + 5e-6


@pytest.mark.parametrize("which", JOINT_SETS)
def test_restriction_is_a_principal_submatrix(joint, which):
    blocks, index, _, _ = joint
    (_, full), = blocks
    sub_blocks, sub_index = restrict_joint(blocks, index, which)
    (_, sub), = sub_blocks
    m0 = index["n_sigma"]
    expected = {"joint": full, "mf33": full[:m0, :m0], "mf34": full[m0:, m0:]}
    np.testing.assert_array_equal(sub, expected[which])
    assert sub_index["dimension"] == sub.shape[0]
    if which != "joint":
        assert not sub_index["has_cross"]


def test_the_cross_reaches_the_ensemble(joint):
    """The only test that says the cross term survived the draw.

    Everything upstream can be right and the ensemble still carry no
    magnitude-shape correlation — that is precisely what today's two-draw
    pipeline produces — so the realised covariance is checked, not the declared.
    """
    blocks, index, _, _ = joint
    (key, m), = blocks
    samples, _ = draw_samples(blocks, 131_072, space="linear",
                              returns="factors", decomposition_method="svd",
                              sampling_method="sobol", seed=1,
                              psd_method="none", null_tol=None, verbose=False)
    realised = np.cov(samples[key] - 1.0, rowvar=False)
    m0 = index["n_sigma"]
    declared_cross = m[:m0, m0:]
    assert np.abs(declared_cross).max() > 1e-4          # the fixture asserts one
    np.testing.assert_allclose(realised[:m0, m0:], declared_cross,
                               rtol=0.02, atol=1e-6)


def test_a_tape_with_marginals_but_no_a0_blocks_is_refused_by_default(
        micro_tape, tmp_path):
    """`require_cross=True` is this entry point's default on purpose.

    The tape here is the shape of every RELEASED evaluation — MF33 and MF34
    both present, no a₀ sections — which is exactly the case that must not pass
    quietly. Handing back a block-diagonal matrix would be a silent downgrade to
    the factorisation the joint exists to replace, and the run would look like
    it had succeeded.
    """
    out = tmp_path / "no_cross.endf"
    rng = np.random.default_rng(3)
    n0 = MAG.size - 1
    a = rng.normal(size=(n0, n0))
    mf33 = create_mf33_from_covariance(
        0.01 * (a @ a.T) / n0 + np.diag(np.full(n0, 4e-3)),
        MAG, ZA, AWR, MAT, MT)
    tmp = str(out) + ".mf33"
    write_mf33_to_file(str(micro_tape), mf33, tmp)

    grid = MESH[1]
    n = grid.size - 1
    b = rng.normal(size=(n * LMAX, n * LMAX))
    mf34 = create_mf34_from_covariance(
        0.02 * (b @ b.T) / (n * LMAX) + np.diag(np.full(n * LMAX, 1e-2)),
        grid, LMAX, ZA, AWR, MAT, MT, ltt=1)
    write_mf34_to_file(tmp, mf34, str(out))

    with pytest.raises(ValueError, match="no MF34 a_0 blocks"):
        load_joint_mf33_mf34(str(out), mt=MT, isotope=ISO, l_max=LMAX)

    # ...and it is available deliberately, as the zero-cross control.
    blocks, index = load_joint_mf33_mf34(
        str(out), mt=MT, isotope=ISO, l_max=LMAX, require_cross=False)
    assert index["cross_orders"] == [] and not index["has_cross"]
    (_, m), = blocks
    assert not m[:index["n_sigma"], index["n_sigma"]:].any()


# ── the base tape ─────────────────────────────────────────────────────────────

def test_the_base_tape_drops_the_covariance_and_nothing_else(a0cross_tape, tmp_path):
    tape, _ = a0cross_tape
    out = tmp_path / "base.endf"
    report = build_base_tape(tape, str(out))
    assert report["bytes_after"] < report["bytes_before"]

    def mfs(path):
        seen = set()
        for line in open(path):
            if len(line) >= 75:
                try:
                    seen.add(int(line[70:72]))
                except ValueError:
                    pass
        return seen

    assert not (mfs(str(out)) & set(COVARIANCE_MF))
    assert {1, 2, 3, 4} <= mfs(str(out))
    # MF3 and MF4 are byte-identical, which is what makes the ensemble centred
    # on the same evaluation the covariance describes.
    assert set(report["verified_mf"]) == {3, 4}
    read_endf(str(out))            # and it still parses
