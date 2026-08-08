"""`write_consistent_mf34` must emit a file the pipeline's own reader recovers.

WHY THIS FILE EXISTS. Run 89 shipped the group-space cross block as a sidecar
next to the MF34 already in the `_mg.endf` -- a pairing nothing had diagnosed,
measuring sigma_max(K) = 41.27 and lam_min/scale = -0.447. The repair is to ship
MF34 rebuilt from the same collapsed Pass-1 replicas, so the cross block sits
next to the marginals it was built with. That makes `write_consistent_mf34` the
step the whole run now depends on, and it had no test: its first execution was
against the 205 MB production file, where a shape or currency error costs a
cluster job to discover and looks like a covariance failure when it lands.

WHAT IS ACTUALLY PINNED HERE, on 4 shape groups instead of 703:

1. The section is structurally legal -- LTT=3 with NL = max_order + 1, and
   NSS = NL(NL+1)/2 sub-subsections. Under the old NL convention (highest index,
   not count) `parse_mf34_mt` loops NSS times against an under-declared NL and
   **silently truncates the tail of the section** rather than raising, so a
   structural check has to count blocks, not just parse without error.
2. The absolute -> relative conversion: shape blocks divide by outer(a, a) and
   the cross blocks by `a` on the shape axis only. Getting one of the two
   denominators wrong produces a plausible file, not an exception.
3. The (0, 0) magnitude self-block is null (manual Sec. 34.3) -- the magnitude
   self-covariance belongs to MF33 and must not be repeated here.
4. Everything comes back RELATIVE. MF34 admits no absolute matrix form at all
   (only LB=0 is absolute and it carries no off-diagonal structure), so a block
   read back as absolute would mean the writer picked an LB it must not use.
5. The reader's union-grid lift, which is the surprising part. LB=6 carries
   independent row and column grids, but `MF34CovMat` keeps ONE square matrix
   per (L, L1), so it unions the magnitude and shape grids and expands the
   rectangle onto that union by duplication. In production that turns a
   188 x 703 cross block into a matrix on the union of the two grids. This is
   the container's convention, not a defect -- but a downstream consumer sees
   the lifted object, so the lift is pinned as faithful.

The read-back deliberately goes through `MF34CovMat.from_endf` rather than
`parse_mf34_mt` on the section text: that is the entry point both
`build_group_cross.mf34_group_edges_ev` and the chi2 covariance assembly use, so
a file that only round-trips via `str(section)` would still be useless.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.cov import MF34CovMat
from kika.endf.writers.mf34_writer import (
    create_mf34_from_covariance,
    write_mf34_to_file,
)
from scripts.build_group_cross import (
    L_MAX,
    shipped_mg_endf,
    write_consistent_mf34,
)

ZA, AWR, MAT, MT = 26056.0, 55.454, 2631, 2
N_GS, N_GM = 4, 3
SHAPE_EV = np.array([0.85e6, 1.5e6, 2.5e6, 3.2e6, 4.0e6])
MAG_EV = np.array([0.85e6, 1.8e6, 3.0e6, 4.0e6])

_TEMPLATE = Path(__file__).resolve().parents[2] / "kika" / "endf" / "tests" / "data" / "micro_fe56_cov.endf"


def _inputs():
    """A PSD absolute joint in exactly the layout `main()` hands over.

    `a_nom_group` is drawn away from zero on purpose: the relative cross block
    divides by `a_L`, and the zero-crossing case is a separate concern (Gate 2
    measured zero bins at any order within 1e-9 of the serialisation floor).
    """
    rng = np.random.default_rng(0)
    a_nom_group = rng.uniform(0.2, 0.9, (N_GS, L_MAX))
    a_flat = a_nom_group.reshape(-1)
    p = N_GS * L_MAX
    j = np.cov(rng.normal(size=(400, N_GM + p)), rowvar=False)
    c34_post = j[N_GM:, N_GM:] * np.outer(a_flat, a_flat) * 1e-3
    cx_post = j[:N_GM, N_GM:] * a_flat[None, :] * 1e-3
    return a_nom_group, a_flat, c34_post, cx_post


def _rel_ship(fill: float = 0.25) -> np.ndarray:
    """Stand-in for the shipped RELATIVE MF34 on the base shape grid.

    Only consulted where a_l is exactly zero, so a constant is enough to tell
    "preserved" from "overwritten" apart.
    """
    p = N_GS * L_MAX
    return np.full((p, p), fill)


def _source_endf(tmp_path: Path) -> Path:
    """A template carrying an MF34 MT=2, which `_za_awr_mat_from_endf` requires."""
    src = tmp_path / "source.endf"
    seed = create_mf34_from_covariance(
        np.eye(N_GS * L_MAX) * 1e-4, SHAPE_EV, L_MAX, ZA, AWR, MAT, MT, ltt=1,
    )
    write_mf34_to_file(str(_TEMPLATE), seed, str(src))
    return src


def _read_blocks(path: Path) -> dict:
    m = MF34CovMat.from_endf(str(path), energy_unit="MeV")
    m = m.filter_by_isotope_reaction(int(ZA), MT)
    return {(int(m.l_rows[k]), int(m.l_cols[k])):
            (np.asarray(m.matrices[k], float), bool(m.is_relative[k]))
            for k in range(m.num_matrices)}


def _lift(native, row_ev, col_ev):
    """Reproduce `to_ang_covmat`'s piecewise-constant lift onto the union grid.

    Comparing against this rather than against the native rectangle is what
    makes the assertion a test of the FILE instead of a test of the container's
    convention.  A block asserts nothing outside its own grid, so the lift
    zero-fills there rather than clamping to the edge group -- which only bites
    for a restricted cross window, where the row grid is narrower than the union.
    """
    u = np.unique(np.concatenate([row_ev, col_ev]))
    c = 0.5 * (u[:-1] + u[1:])
    gi = np.clip(np.searchsorted(row_ev, c, side="right") - 1, 0, len(row_ev) - 2)
    gj = np.clip(np.searchsorted(col_ev, c, side="right") - 1, 0, len(col_ev) - 2)
    out = native[np.ix_(gi, gj)].astype(float)
    out[(c < row_ev[0]) | (c > row_ev[-1]), :] = 0.0
    out[:, (c < col_ev[0]) | (c > col_ev[-1])] = 0.0
    return out


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("group_cross_endf")
    a_nom_group, a_flat, c34_post, cx_post = _inputs()
    src = _source_endf(tmp)
    out = tmp / "consistent_mg.endf"
    write_consistent_mf34(out, src, c34_post, cx_post, a_nom_group,
                          SHAPE_EV, MAG_EV, _rel_ship())
    return {"src": src, "out": out, "blocks": _read_blocks(out),
            "a_flat": a_flat, "c34_post": c34_post, "cx_post": cx_post}


def test_section_is_structurally_complete(written):
    """NSS = NL(NL+1)/2 blocks survive the round trip, a_0 row included.

    An under-declared NL truncates silently, so counting is the assertion.
    """
    blocks = written["blocks"]
    nl = L_MAX + 1
    assert len(blocks) == nl * (nl + 1) // 2 == 28
    assert sorted(blocks) == [(l, l1) for l in range(nl) for l1 in range(l, nl)]


def test_every_block_is_relative(written):
    """MF34 has no absolute matrix form; an absolute block means a wrong LB."""
    assert all(rel for _, rel in written["blocks"].values())


def test_magnitude_self_block_is_null(written):
    """Manual Sec. 34.3 -- that covariance belongs to MF33, not here."""
    assert np.allclose(written["blocks"][(0, 0)][0], 0.0)


def test_the_null_self_block_is_written_as_one_interval(written):
    """⚑ It is null either way; the question is how many zeros it costs.

    `_create_mf34_with_cross` emits the (0,0) block as a full `zeros((n0, n0))`
    upper triangle -- n0(n0+1)/2 numbers. At the 188-group magnitude axis that
    was 17 k numbers and nobody looked; on the shipped 2317-bin MF33 grid it is
    2.685 M, about 36 MB of ASCII zeros in the deliverable, decoded and
    projected as a 2317^2 matrix by every reader on the way in.

    One interval spanning the same range asserts the identical zero. Pinned on
    the RECORD rather than on the matrix, because the lifted matrix looks the
    same either way -- which is exactly why this was free to get wrong.
    """
    from kika.endf.parsers.parse_mf34 import parse_mf34_mt
    from kika.endf import read_endf

    sec = read_endf(str(written["out"]), mf_numbers=[34]).get_file(34).sections[MT]
    ss = next(s for s in sec._subsections[0].sub_subsections
              if (s.l, s.l1) == (0, 0))
    rec = ss.records[0]
    assert rec.lb == 5, "square blocks are LB=5; LB=6 would need two grids"
    assert len(rec.energies) == 2, (
        f"the null block spans {len(rec.energies) - 1} intervals; one is "
        f"enough and {N_GM} is already more than needed"
    )
    assert list(rec.energies) == [MAG_EV[0], MAG_EV[-1]], (
        "the single interval must still span the magnitude range, so the "
        "block asserts zero over the same energies it did before"
    )
    assert np.allclose(rec.matrix, 0.0)


def test_cross_blocks_carry_cov_over_a_l(written):
    """The cross axis divides by `a_L` on the shape axis only, and lifts."""
    cx_rel = (written["cx_post"] / written["a_flat"][None, :]
              ).reshape(N_GM, N_GS, L_MAX)
    for l1 in range(1, L_MAX + 1):
        got, _ = written["blocks"][(0, l1)]
        want = _lift(cx_rel[:, :, l1 - 1], MAG_EV, SHAPE_EV)
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max()


def test_shape_blocks_carry_cov_over_outer_a(written):
    """The shape blocks divide by outer(a, a) and stay on the shape grid."""
    c34_rel = written["c34_post"] / np.outer(written["a_flat"], written["a_flat"])
    for l in range(1, L_MAX + 1):
        got, _ = written["blocks"][(l, l)]
        want = c34_rel[l - 1::L_MAX, l - 1::L_MAX]
        assert got.shape == (N_GS, N_GS)
        assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max()


def test_shape_blocks_are_independent_of_the_cross_term(written, tmp_path):
    """Adding the a_0 row must not perturb the L >= 1 shape covariance.

    This is what lets run 90 be read against run 86 as one change: MF34's shape
    content differs only by the correlations that came with the rebuild, never
    by the act of carrying a cross block.
    """
    a_nom_group, a_flat, c34_post, cx_post = _inputs()
    out = tmp_path / "zero_cross_mg.endf"
    write_consistent_mf34(out, written["src"], c34_post,
                          np.zeros_like(cx_post), a_nom_group, SHAPE_EV,
                          MAG_EV, _rel_ship())
    zero = _read_blocks(out)
    for l in range(1, L_MAX + 1):
        for l1 in range(l, L_MAX + 1):
            assert np.array_equal(zero[(l, l1)][0], written["blocks"][(l, l1)][0])
        assert np.allclose(zero[(0, l)][0], 0.0)


def test_energy_window_selects_whole_magnitude_groups(written, tmp_path):
    """The window flags exist but are off by default, and must stay exact.

    Run 90 ships the FULL magnitude grid: the chi2 reads the cross term from the
    unrestricted sidecar, so a windowed file would put a different block in the
    file from the one being scored -- the same diagnose-one-thing/ship-another
    failure that produced run 89, one layer down.
    """
    a_nom_group, a_flat, c34_post, cx_post = _inputs()
    out = tmp_path / "windowed_mg.endf"
    write_consistent_mf34(out, written["src"], c34_post, cx_post, a_nom_group,
                          SHAPE_EV, MAG_EV, _rel_ship(),
                          cross_emin_ev=MAG_EV[1], cross_emax_ev=MAG_EV[-1])
    cx_rel = (cx_post / a_flat[None, :]).reshape(N_GM, N_GS, L_MAX)
    got, _ = _read_blocks(out)[(0, 1)]
    want = _lift(cx_rel[1:, :, 0], MAG_EV[1:], SHAPE_EV)
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max()


def test_a_second_mg_endf_is_refused_rather_than_silently_preferred(tmp_path):
    """The self-comparison trap: `--write-endf` leaves a second `*_mg.endf`.

    It sorts BEFORE the shipped one ("consistent" < "nominal"), so a first-wins
    glob would read the shape grids and `c34_ship` off our own output on the
    next run -- diagnosing the candidate against itself, which makes every PSD
    row perfect for the one reason that renders it meaningless. The single-file
    case must keep working; the ambiguous one must stop.
    """
    (tmp_path / "26-Fe-56g_nominal_mg.endf").write_text("")
    assert shipped_mg_endf(tmp_path).name == "26-Fe-56g_nominal_mg.endf"

    (tmp_path / "26-Fe-56g_nominal_consistent_mg.endf").write_text("")
    with pytest.raises(SystemExit, match="--source-endf"):
        shipped_mg_endf(tmp_path)


def test_no_mg_endf_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no \\*_mg.endf"):
        shipped_mg_endf(tmp_path)


@pytest.mark.parametrize("null_fill,expected", [("zero", 0.0), ("ship", 0.25)])
def test_null_parameter_fill_is_selectable(written, tmp_path, null_fill, expected):
    """a_l = 0 makes the round trip 0/0, and BOTH answers must be reachable.

    `row_aggregator` zeroes a whole (group, order) row when no fine bin in the
    group is valid at that order — 1542 of 4218 slots in production. There the
    collapsed replicas are identically zero, so `c34_post` is zero too and the
    relative form is genuinely undefined rather than merely ill-conditioned.

    ⚠ "ship" was adopted first on the argument that it was free, because a
    consumer converting back multiplies by a_l = 0. That is false for the chi2,
    which scales relative MF34 blocks by `a_l_per_pt` — the MF4 coefficients
    interpolated onto each EXFOR energy, which are NOT zero here. Run 90 then
    reproduced run 89 almost exactly (roadmap Sec. 10.1.8-L11). "zero" is the
    default because a dead parameter with no covariance is a true null
    direction and cannot break PSD; "ship" is kept so the two can be measured
    against each other rather than argued about.
    """
    a_nom_group, a_flat, c34_post, cx_post = _inputs()
    a_nom_group[2, 3] = 0.0                      # kill one (group, order) slot
    dead = 2 * L_MAX + 3
    a_flat = a_nom_group.reshape(-1)
    c34_post[dead, :] = 0.0                      # as the collapse would leave it
    c34_post[:, dead] = 0.0
    cx_post[:, dead] = 0.0

    out = tmp_path / f"null_{null_fill}_mg.endf"
    write_consistent_mf34(out, written["src"], c34_post, cx_post, a_nom_group,
                          SHAPE_EV, MAG_EV, _rel_ship(0.25),
                          null_fill=null_fill)
    blocks = _read_blocks(out)

    # The dead slot is group 2 at order 4.
    assert blocks[(4, 4)][0][2, 2] == pytest.approx(expected)
    # ... while a live slot in the same block is the rebuilt value either way.
    live_rel = c34_post[3 * L_MAX + 3, 3 * L_MAX + 3] / a_flat[3 * L_MAX + 3] ** 2
    assert blocks[(4, 4)][0][3, 3] == pytest.approx(live_rel, rel=1e-5)
    # The cross block is new, so a dead parameter gets zero, not 0.25. Compare
    # the whole lifted block rather than hand-indexing the union grid.
    cx_rel = np.zeros((N_GM, N_GS, L_MAX))
    np.divide(cx_post, a_flat[None, :], out=cx_rel.reshape(N_GM, -1),
              where=(np.abs(a_flat) > 0)[None, :])
    got = blocks[(0, 4)][0]
    want = _lift(cx_rel[:, :, 3], MAG_EV, SHAPE_EV)
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-5 * max(np.abs(want).max(), 1e-12)
    # and the dead shape group's column really is zero in that expectation
    assert np.allclose(cx_rel[:, 2, 3], 0.0)


def test_nonzero_covariance_on_a_zero_mean_parameter_is_rejected(written):
    """The one case that is infinite rather than undefined, and must stop.

    Zero mean with zero covariance is a null direction and is handled. Zero mean
    with NONZERO covariance is an infinite relative uncertainty that no written
    value could represent, and it means the collapse and the nominal means were
    built from different validity masks — a real inconsistency, not a corner.
    """
    a_nom_group, _, c34_post, cx_post = _inputs()
    a_nom_group[2, 3] = 0.0                      # but leave c34_post populated
    with pytest.raises(SystemExit, match="where a_l is exactly zero"):
        write_consistent_mf34(Path("unused.endf"), written["src"], c34_post,
                              cx_post, a_nom_group, SHAPE_EV, MAG_EV,
                              _rel_ship())


def test_assemble_c34_rel_is_the_one_definition():
    """`--check` and the writer must build the SAME array, not two like ones.

    Diagnosing one object and shipping another is what produced runs 89 and 90,
    so the diagnostic rows call this function rather than re-deriving the
    quotient. Live slots are the rebuilt value under both fills; only the null
    slots differ, and only in the way the flag names.
    """
    from scripts.build_group_cross import assemble_c34_rel

    a = np.array([2.0, 0.0, 4.0])
    c34 = np.array([[8.0, 0.0, 8.0], [0.0, 0.0, 0.0], [8.0, 0.0, 32.0]])
    ship = np.full((3, 3), 0.5)

    z = assemble_c34_rel(c34, a, ship, "zero")
    s = assemble_c34_rel(c34, a, ship, "ship")

    live = np.ix_([0, 2], [0, 2])
    want_live = np.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(z[live], want_live)
    np.testing.assert_allclose(s[live], want_live)

    assert np.allclose(z[1, :], 0.0) and np.allclose(z[:, 1], 0.0)
    assert np.allclose(s[1, :], 0.5) and np.allclose(s[:, 1], 0.5)

    with pytest.raises(SystemExit, match="unknown null_fill"):
        assemble_c34_rel(c34, a, ship, "keep")


def test_assemble_does_not_mutate_the_shipped_matrix():
    """`out=` aliasing would silently corrupt c34_rel_ship for later callers.

    main() reuses the same array for the absolute diagnosis AND both fills, so
    an in-place write would make the second call see the first call's output.
    """
    from scripts.build_group_cross import assemble_c34_rel

    a = np.array([2.0, 0.0])
    c34 = np.array([[8.0, 0.0], [0.0, 0.0]])
    ship = np.full((2, 2), 0.5)
    before = ship.copy()
    assemble_c34_rel(c34, a, ship, "ship")
    assemble_c34_rel(c34, a, ship, "zero")
    np.testing.assert_array_equal(ship, before)


# ── the fine magnitude axis (roadmap §10.7-4 step 5, §10.7-10) ────────────────

# A stand-in for the file's own MF33 grid: our magnitude range sits INSIDE it,
# with host bins below and above, exactly as run 86 has 431 JEFF bins below our
# 1738 and 148 above.
MF33_FILE_EV = np.array([1.0e5, 4.0e5, 0.85e6, 1.8e6, 3.0e6, 4.0e6, 9.0e6, 2.0e7])


@pytest.fixture(scope="module")
def written_padded(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("group_cross_padded")
    a_nom_group, a_flat, c34_post, cx_post = _inputs()
    src = _source_endf(tmp)
    out = tmp / "padded_mg.endf"
    write_consistent_mf34(out, src, c34_post, cx_post, a_nom_group,
                          SHAPE_EV, MAG_EV, _rel_ship(),
                          mf33_file_grid=MF33_FILE_EV)
    return {"out": out, "src": src, "a_flat": a_flat, "cx_post": cx_post}


def test_the_cross_rows_are_embedded_in_the_files_own_mf33_grid(written_padded):
    """⚑ The magnitude leg must be `_mf33_magnitude_map` on the MF33 grid,
    unmodified — so the a_0 rows have to BE that grid, zero-padded, not our
    narrower one. Padding adds parameters with no variance and no covariance;
    the shipped MF33 has `max|cov[in, out]| = 0` exactly, so they are genuinely
    separable rather than merely small.
    """
    from scripts.mf34_cross_reader import read_mf34_split

    res = read_mf34_split(written_padded["out"], isotope=int(ZA), mt=MT,
                          l_max=L_MAX, mf33_grid_ev=MF33_FILE_EV)
    assert len(res.cross) == L_MAX
    i0 = int(np.searchsorted(MF33_FILE_EV, MAG_EV[0]))
    want = (written_padded["cx_post"] / written_padded["a_flat"][None, :]
            ).reshape(N_GM, N_GS, L_MAX)
    for b in res.cross:
        m = b["matrix"]
        assert m.shape == (MF33_FILE_EV.size - 1, N_GS)
        np.testing.assert_allclose(m[i0:i0 + N_GM], want[:, :, b["l"] - 1],
                                   rtol=2e-6, atol=0)
        assert not m[:i0].any(), "host bins below our range must be zero"
        assert not m[i0 + N_GM:].any(), "host bins above our range must be zero"


def test_a_cross_grid_that_is_not_a_sub_sequence_of_the_files_grid_is_refused(tmp_path):
    """Regridding here is the run-89 mistake: Cx folded against marginals it was
    never built with."""
    a_nom_group, _, c34_post, cx_post = _inputs()
    src = _source_endf(tmp_path)
    bogus = np.array([1.0e5, 1.0e6, 2.0e6, 4.0e6, 2.0e7])   # our edges absent
    with pytest.raises(SystemExit, match="contiguous sub-sequence"):
        write_consistent_mf34(tmp_path / "bad.endf", src, c34_post, cx_post,
                              a_nom_group, SHAPE_EV, MAG_EV, _rel_ship(),
                              mf33_file_grid=bogus)


def test_without_the_file_grid_the_rows_stay_on_the_magnitude_grid(written):
    """The group-axis path is untouched, so `--mag-grid group` still writes what
    it always wrote."""
    from scripts.mf34_cross_reader import read_mf34_split

    res = read_mf34_split(written["out"], isotope=int(ZA), mt=MT, l_max=L_MAX,
                          mf33_grid_ev=MAG_EV)
    assert all(b["matrix"].shape == (N_GM, N_GS) for b in res.cross)


def test_the_c33_matrix_gate_passes_on_the_fine_construction_and_fails_off_diagonal():
    """⚑ The diagonal-only marginal check cannot see the failure this catches.

    Reproduce the fine-axis construction in miniature: one replica set, a
    correlation matrix, and the two-pass rescale as a positive diagonal
    congruence. `c33_post` then equals the shipped matrix EXACTLY, which is what
    makes the fold a congruence — measured on run 86 at 4.58e-16.

    Then perturb an OFF-DIAGONAL element only. The marginals are untouched, so
    `marginal-identity` still passes; the gate must not.
    """
    from scripts.build_group_cross import c33_matrix_gate

    rng = np.random.default_rng(11)
    m = 12
    z = rng.normal(size=(500, m))
    cov_mc = np.cov(z, rowvar=False)
    d_mc = np.sqrt(np.diag(cov_mc))
    d_ship = rng.uniform(0.02, 0.2, m)          # Pass-2 sigmas, unrelated to MC
    c33_ship = (cov_mc / np.outer(d_mc, d_mc)) * np.outer(d_ship, d_ship)

    j33 = d_ship / d_mc
    c33_post = cov_mc * np.outer(j33, j33)
    assert c33_matrix_gate(c33_post, c33_ship, fatal=True) < 1e-12

    bad = c33_post.copy()
    bump = 0.05 * np.abs(c33_ship).max()
    bad[0, 3] += bump
    bad[3, 0] += bump
    np.testing.assert_allclose(np.diag(bad), np.diag(c33_post))  # marginals intact
    with pytest.raises(SystemExit, match="same matrix"):
        c33_matrix_gate(bad, c33_ship, fatal=True)
    # ... and on the group axis the same disagreement is reported, not fatal.
    assert c33_matrix_gate(bad, c33_ship, fatal=False) > 1e-3
