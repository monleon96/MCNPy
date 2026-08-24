"""The shape parameters the diagonal congruence cannot reach.

WHY THIS FILE EXISTS. `build_group_cross` publishes the MC joint rescaled to the
pipeline's sigmas: `jj = d_tar / d_mc`, applied as a diagonal congruence to all
three blocks at once. That is what makes the cross block Cauchy-Schwarz
compatible with the marginals it ships next to -- and it is also what silently
deletes every parameter the replicas never move, because `jj` is zero there and
a congruence cannot give variance to a direction it annihilates.

On run 99 that was 1542 of 4218 (group, order) slots, nearly all of a_4..a_6,
and the tape then declared them with `rel = 0` while `eval_covariance` scaled
that zero by an `a_l` interpolated out of the file's own MF4, which is NOT zero.
A tape asserting perfect knowledge of a parameter it does declare.

`carry_dead_parameters` puts them back as a direct summand. These tests pin the
two properties that make the direct sum the right shape for it: the augmented
matrix stays block diagonal, so PSD is `min` of two PSD pieces, and the cross
term's whitened norm -- the gate the whole file is certified on -- cannot move.
"""
import numpy as np
import pytest

from scripts.build_group_cross import (DEAD_BLOCK_CLIP_MASS_MAX,
                                       DEAD_BLOCK_D_SHIFT_MAX,
                                       DEAD_BLOCK_PSD_RTOL,
                                       carry_dead_parameters, diagnose)


def _case(n_live=4, n_dead=3, seed=0):
    """A joint with `n_dead` parameters the MC froze and the pipeline declares."""
    rng = np.random.default_rng(seed)
    n = n_live + n_dead
    dead = np.zeros(n, bool)
    dead[n_live:] = True

    # the pipeline's own MF34: PSD by construction, and it declares every slot
    B = rng.normal(size=(n, n))
    c34_ship = B @ B.T + 0.5 * np.eye(n)
    d_tar = np.sqrt(np.diag(c34_ship))

    # the congruence's product: the live block only, dead rows AND columns zero
    L = rng.normal(size=(n_live, n_live))
    c34_post = np.zeros((n, n))
    c34_post[:n_live, :n_live] = L @ L.T
    # ...rescaled so its diagonal already matches the target, as `jj` leaves it
    s = d_tar[:n_live] / np.sqrt(np.diag(c34_post[:n_live, :n_live]))
    c34_post[:n_live, :n_live] *= np.outer(s, s)

    d_mc = np.concatenate([np.sqrt(np.diag(c34_post[:n_live, :n_live])),
                           np.zeros(n_dead)])
    return c34_post, c34_ship, d_mc, d_tar, dead


def test_the_carried_slots_are_the_ones_the_mc_froze():
    c34_post, c34_ship, d_mc, d_tar, dead = _case()
    _, got, orphan, _ = carry_dead_parameters(c34_post.copy(), c34_ship, d_mc, d_tar)
    np.testing.assert_array_equal(got, dead)
    assert orphan == 0


def test_a_slot_neither_side_declares_is_left_alone_and_counted():
    """d_mc = 0 AND d_tar = 0 is a genuinely absent parameter, not a loss."""
    c34_post, c34_ship, d_mc, d_tar, dead = _case()
    i = int(np.flatnonzero(dead)[0])
    c34_ship = c34_ship.copy()
    c34_ship[i, :] = c34_ship[:, i] = 0.0
    d_tar = d_tar.copy()
    d_tar[i] = 0.0
    out, got, orphan, _ = carry_dead_parameters(c34_post.copy(), c34_ship, d_mc, d_tar)
    assert not got[i] and orphan == 1
    assert out[i, i] == 0.0


def test_the_marginal_comes_back_exactly():
    """The gate `build_group_cross` prints -- sqrt(diag) must equal d_tar."""
    c34_post, c34_ship, d_mc, d_tar, _ = _case()
    out, _, _, _ = carry_dead_parameters(c34_post.copy(), c34_ship, d_mc, d_tar)
    np.testing.assert_allclose(np.sqrt(np.diag(out)), d_tar, rtol=0, atol=1e-14)


def test_the_result_is_block_diagonal():
    """No coupling is invented between what the MC saw and what it did not."""
    out, dead, _, _ = carry_dead_parameters(*_case()[:4])
    assert np.abs(out[np.ix_(dead, ~dead)]).max() == 0.0
    assert np.abs(out[np.ix_(~dead, dead)]).max() == 0.0


def test_psd_is_the_min_of_the_two_pieces():
    c34_post, c34_ship, d_mc, d_tar, dead = _case()
    out, _, _, _ = carry_dead_parameters(c34_post.copy(), c34_ship, d_mc, d_tar)
    lam = np.linalg.eigvalsh(out)[0]
    lam_live = np.linalg.eigvalsh(c34_post[np.ix_(~dead, ~dead)])[0]
    lam_dead = np.linalg.eigvalsh(c34_ship[np.ix_(dead, dead)])[0]
    assert lam == pytest.approx(min(lam_live, lam_dead), rel=1e-10)


def test_the_cross_terms_whitened_norm_cannot_move():
    """Row F's gate. `cx` is exactly zero on the dead columns and the whitener
    of a block-diagonal matrix is block diagonal, so ||W33 cx W34||_2 is blind
    to the augmentation. If this ever fails, the augmentation is not a direct
    sum and the tape must not ship."""
    c34_post, c34_ship, d_mc, d_tar, dead = _case()
    rng = np.random.default_rng(7)
    n_mag = 5
    A = rng.normal(size=(n_mag, n_mag))
    c33 = A @ A.T + 0.5 * np.eye(n_mag)
    cx = rng.normal(size=(n_mag, dead.size)) * 0.01
    cx[:, dead] = 0.0

    before = diagnose("before", c33, c34_post, cx)
    out, _, _, _ = carry_dead_parameters(c34_post.copy(), c34_ship, d_mc, d_tar)
    after = diagnose("after", c33, out, cx)
    assert after["sigma_max(K)"] == pytest.approx(before["sigma_max(K)"], rel=1e-9)


def test_a_dead_row_that_is_not_identically_zero_is_refused():
    """The mask and the congruence have to agree, or `block_diag` is a lie and
    the PSD argument silently stops holding."""
    c34_post, c34_ship, d_mc, d_tar, dead = _case()
    i = int(np.flatnonzero(dead)[0])
    c34_post[i, 0] = c34_post[0, i] = 1e-3
    with pytest.raises(SystemExit, match="direct-sum"):
        carry_dead_parameters(c34_post, c34_ship, d_mc, d_tar)


def test_nothing_happens_when_the_mc_moved_everything():
    """The `drop` path and a fully live run must be untouched, bit for bit."""
    c34_post, c34_ship, _, d_tar, _ = _case(n_live=6, n_dead=0)
    d_mc = np.sqrt(np.diag(c34_post))
    out, dead, orphan, _ = carry_dead_parameters(c34_post.copy(), c34_ship, d_mc, d_tar)
    assert not dead.any() and orphan == 0
    np.testing.assert_array_equal(out, c34_post)


# ---------------------------------------------------------------------------
# The PSD projection, and the bar it is only allowed to act below.
#
# `c34_ship` restricted to the carried slots came back slightly indefinite on
# run 99, so the direct sum inherits a negative eigenvalue. Clipping it is
# admissible ONLY while it is numerical dust; above `DEAD_BLOCK_PSD_RTOL` the
# code has to refuse, because a projection that absorbs a real negative
# direction hides a defect in the deliverable instead of fixing it.
# ---------------------------------------------------------------------------

def _case_indefinite(ratio, n_live=4, n_dead=3, seed=0):
    """As `_case`, but the pipeline's block on the dead slots has
    `lam_min / lam_max = -ratio` exactly."""
    c34_post, c34_ship, d_mc, d_tar, dead = _case(n_live, n_dead, seed)
    D = np.ix_(dead, dead)
    w, V = np.linalg.eigh(0.5 * (c34_ship[D] + c34_ship[D].T))
    w[0] = -ratio * w[-1]
    blk = (V * w) @ V.T
    c34_ship = c34_ship.copy()
    c34_ship[D] = 0.5 * (blk + blk.T)
    # d_tar has to keep quoting the block the function will actually read
    d_tar = d_tar.copy()
    d_tar[dead] = np.sqrt(np.maximum(np.diag(c34_ship[D]), 0.0))
    return c34_post, c34_ship, d_mc, d_tar, dead


def test_dust_is_projected_away_and_the_result_is_psd():
    c34_post, c34_ship, d_mc, d_tar, dead = _case_indefinite(
        0.01 * DEAD_BLOCK_PSD_RTOL)
    out, _, _, info = carry_dead_parameters(c34_post.copy(), c34_ship,
                                            d_mc, d_tar)
    assert info["n_clipped"] == 1
    assert info["ratio"] == pytest.approx(-0.01 * DEAD_BLOCK_PSD_RTOL, rel=1e-6)
    assert np.linalg.eigvalsh(out[np.ix_(dead, dead)])[0] >= -1e-18
    # and the projection is dust: it moves no declared variance detectably
    assert info["d_shift_rel"] < 1e-6


def test_a_real_negative_direction_is_refused_not_projected():
    """The condition on the whole projection. 100x over the bar must raise."""
    c34_post, c34_ship, d_mc, d_tar, _ = _case_indefinite(
        100 * DEAD_BLOCK_PSD_RTOL)
    with pytest.raises(SystemExit, match="INDEFINITE"):
        carry_dead_parameters(c34_post, c34_ship, d_mc, d_tar)


def test_the_bar_is_a_ratio_not_an_absolute_size():
    """Scaling the whole problem by 1e6 must not change the verdict, or the
    bar would be a unit accident rather than a statement about conditioning."""
    for scale in (1.0, 1e6):
        c34_post, c34_ship, d_mc, d_tar, _ = _case_indefinite(
            100 * DEAD_BLOCK_PSD_RTOL)
        with pytest.raises(SystemExit, match="INDEFINITE"):
            carry_dead_parameters(c34_post * scale**2, c34_ship * scale**2,
                                  d_mc * scale, d_tar * scale)


def test_the_projection_cannot_move_the_cross_terms_whitened_norm():
    """Row F's gate again, this time with the clip active. Same argument: the
    carried block is a separate summand and `cx` is zero on it."""
    c34_post, c34_ship, d_mc, d_tar, dead = _case_indefinite(
        0.01 * DEAD_BLOCK_PSD_RTOL)
    rng = np.random.default_rng(11)
    A = rng.normal(size=(5, 5))
    c33 = A @ A.T + 0.5 * np.eye(5)
    cx = rng.normal(size=(5, dead.size)) * 0.01
    cx[:, dead] = 0.0

    before = diagnose("before", c33, c34_post, cx)
    out, _, _, info = carry_dead_parameters(c34_post.copy(), c34_ship,
                                            d_mc, d_tar)
    assert info["n_clipped"] == 1
    after = diagnose("after", c33, out, cx)
    assert after["sigma_max(K)"] == pytest.approx(before["sigma_max(K)"],
                                                  rel=1e-9)


def test_a_psd_pipeline_block_is_not_touched_at_all():
    """No clip when there is nothing to clip -- the marginals stay exact."""
    c34_post, c34_ship, d_mc, d_tar, _ = _case()
    out, _, _, info = carry_dead_parameters(c34_post.copy(), c34_ship,
                                            d_mc, d_tar)
    assert info["n_clipped"] == 0 and info["d_shift_rel"] == 0.0
    assert info["lam_min"] > 0.0
    np.testing.assert_allclose(np.sqrt(np.diag(out)), d_tar, rtol=0, atol=1e-14)


def test_many_small_negatives_WARN_but_are_judged_by_the_consequence():
    """⚑ LA POLITICA CAMBIO EL 2026-08-22 (Juan), y este test la fija.

    Bar 2 aplicaba el suelo de las 7 cifras -- derivado para la PEOR direccion
    -- a la SUMA sobre cientos de autovalores, y esa parte no se sostiene: la
    masa negativa que el formato puede fabricar escala con el numero de
    direcciones. Aqui cada autovalor esta 100x dentro de bar 1 y la masa pasa de
    bar 2, y eso ya NO rechaza: avisa, y el veredicto lo da ``d_shift_rel`` --
    cuanto mueve el recorte una varianza DECLARADA, que es la unica lectura que
    distingue redondeo acumulado de un defecto repartido.
    """
    n_dead = 60
    c34_post, c34_ship, d_mc, d_tar, dead = _case(n_live=4, n_dead=n_dead)
    D = np.ix_(dead, dead)
    w, V = np.linalg.eigh(0.5 * (c34_ship[D] + c34_ship[D].T))
    # half the spectrum pushed just barely negative, each one 100x inside bar 1
    w[: n_dead // 2] = -0.9 * DEAD_BLOCK_PSD_RTOL * w[-1]
    blk = (V * w) @ V.T
    c34_ship = c34_ship.copy()
    c34_ship[D] = 0.5 * (blk + blk.T)
    d_tar = d_tar.copy()
    d_tar[dead] = np.sqrt(np.maximum(np.diag(c34_ship[D]), 0.0))

    _, _, _, probe = _spectrum(c34_ship[D])
    assert probe["ratio"] > -DEAD_BLOCK_PSD_RTOL, "bar 1 must PASS here"
    assert probe["clip_mass"] > DEAD_BLOCK_CLIP_MASS_MAX, (
        "bar 2 tiene que dispararse aqui, o el test no prueba el cambio")

    out, _, _, info = carry_dead_parameters(c34_post.copy(), c34_ship.copy(),
                                            d_mc, d_tar)
    assert info["clip_mass"] > DEAD_BLOCK_CLIP_MASS_MAX
    assert info["d_shift_rel"] < DEAD_BLOCK_D_SHIFT_MAX, (
        "el caso sintetico tenia que ser inocuo en la consecuencia")
    assert np.linalg.eigvalsh(out[np.ix_(dead, dead)])[0] >= -1e-12

    # y el rechazo SIGUE VIVO: lo decide d_shift_rel, no la masa
    with pytest.raises(SystemExit, match="varianza DECLARADA"):
        carry_dead_parameters(c34_post.copy(), c34_ship.copy(), d_mc, d_tar,
                              d_shift_max=0.1 * info["d_shift_rel"])


def test_la_masa_por_si_sola_ya_no_rechaza_pero_sigue_reportandose():
    """Que bar 2 avise y no rechace no puede significar perder la senal: el
    diagnostico tiene que seguir trayendo la masa, o el defecto se vuelve
    invisible en vez de tolerado."""
    n_dead = 60
    c34_post, c34_ship, d_mc, d_tar, dead = _case(n_live=4, n_dead=n_dead)
    D = np.ix_(dead, dead)
    w, V = np.linalg.eigh(0.5 * (c34_ship[D] + c34_ship[D].T))
    w[: n_dead // 2] = -0.9 * DEAD_BLOCK_PSD_RTOL * w[-1]
    c34_ship = c34_ship.copy()
    c34_ship[D] = 0.5 * (((V * w) @ V.T) + ((V * w) @ V.T).T)
    d_tar = d_tar.copy()
    d_tar[dead] = np.sqrt(np.maximum(np.diag(c34_ship[D]), 0.0))
    _, _, _, info = carry_dead_parameters(c34_post.copy(), c34_ship, d_mc, d_tar)
    assert info["n_clipped"] == n_dead // 2
    assert info["clip_mass"] > 0.0 and info["trace"] > 0.0
    assert info["d_shift_rel"] >= 0.0


def _spectrum(blk):
    """The two bar quantities, computed the way the function computes them."""
    w = np.linalg.eigvalsh(0.5 * (blk + blk.T))
    neg = w[w < 0.0]
    tr = float(np.sum(np.abs(w)))
    return w, neg, tr, {"ratio": float(w[0] / w[-1]),
                        "clip_mass": float(-neg.sum() / tr) if tr > 0 else 0.0}


def test_the_two_bars_are_independent():
    """A block with ONE eigenvalue over bar 1 but negligible total mass still
    raises -- on bar 1. Neither bar is redundant."""
    c34_post, c34_ship, d_mc, d_tar, _ = _case_indefinite(
        10 * DEAD_BLOCK_PSD_RTOL, n_dead=40)
    with pytest.raises(SystemExit, match="INDEFINITE"):
        carry_dead_parameters(c34_post, c34_ship, d_mc, d_tar)
