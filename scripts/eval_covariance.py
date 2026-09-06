"""Dense per-experiment evaluation covariance for the chi^2 pipelines.

Builds the N x N block

    Sigma_eval = Sigma_eval^MF34 + Sigma_eval^MF33

for one experiment, where N is the number of data points (E_j, mu_j) in that
experiment. This replaces the per-row scalar `sigma_eval` previously folded
into the diagonal of the chi^2 covariance, so that cross-energy MF34
correlations (read from the full mat[k_j, k_k] block instead of just the
energy-diagonal mat[k, k]), cross-order MF34 correlations propagated through
shared a_L, and cross-energy MF33 correlations are all carried into the
chi^2.

Sensitivity model (matches numpy.polynomial.legendre.legval applied to
[c_0, c_0*(2L+1)*a_L, ...] in the precompute scripts):

    y(E, mu) = c_0(E) * [P_0(mu) + sum_{L>=1} (2L+1) a_L(E) P_L(mu)]
    dy/da_L(E)  = c_0(E) * (2L+1) * P_L(mu)         (L >= 1)
    dy/dsigma(E) at relative scaling: y(E, mu) * dsigma_rel(E)

For relative MF34 blocks, ENDF stores Cov[a_L/a_L_nom, a_L'/a_L'_nom];
we recover the absolute Cov[a_L, a_L'] by multiplying with the *interpolated*
nominal a_L(E_j), a_L'(E_k) (same convention as the legacy
`propagate_mf34_to_dy`). Off-diagonal energy entries use the same dispatch:
two points in MF34 bins k_j, k_k contribute mat[k_j, k_k]; two points in the
same bin share mat[k, k] (perfect intra-bin correlation, ENDF
piecewise-constant semantics).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.polynomial.legendre import legval

try:                                    # module, or script on sys.path
    from scripts.point_map import PointMap
except ImportError:                     # pragma: no cover - direct execution
    from point_map import PointMap

# Blocks already reported as partly off-grid, so the message is printed once per
# (l_row, l_col, group size) instead of once per experiment. Reporting matters;
# 67 identical lines per library do not.
_OFF_GRID_SEEN: set = set()


# ── MF34: dense Σ_eval contribution from Legendre coefficient covariance ──────

def _legendre_base_sens(mu: np.ndarray, c0: np.ndarray, L_max: int) -> np.ndarray:
    """base_sens[L-1, j] = c_0(E_j) * (2L+1) * P_L(mu_j) for L = 1..L_max.

    This is dy/da_L(E_j, mu_j) before the relative-a_L scaling; shared by the
    MF34 self block and the MF33↔MF34 cross block so the two use one sensitivity
    definition.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    c0 = np.asarray(c0, dtype=float).ravel()
    N = mu.size
    P_arr = np.zeros((L_max, N), dtype=float)
    for L in range(1, L_max + 1):
        coef = np.zeros(L + 1, dtype=float)
        coef[L] = 1.0
        P_arr[L - 1] = legval(mu, coef)
    twoLp1 = np.arange(3, 2 * L_max + 2, 2, dtype=float)  # 3, 5, 7, ... for L=1..
    return c0[None, :] * twoLp1[:, None] * P_arr  # (L_max, N)


def build_mf34_block(
    mf34,
    e_mev: np.ndarray,
    mu: np.ndarray,
    c0: np.ndarray,
    a_l_per_pt: np.ndarray,
    drop: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dense N×N covariance contribution from MF34 for one experiment.

    Parameters
    ----------
    mf34 : MF34CovMat or None
    e_mev : (N,) energy of each data point in MeV.
    mu : (N,) cos(scattering angle).
    c0 : (N,) c_0(E_j) used by the library at each point.
    a_l_per_pt : (N, L_max) interpolated a_L(E_j) for L = 1..L_max.
    drop : (N, L_max) bool, optional
        True where the (point, order) draws on an MF34 parameter the evaluation
        never determined, so its row AND column are to be excluded. A term
        contributes only if BOTH its orders are kept — dropping only the
        variance would leave the correlations behind and is not the same thing.

        WHY THIS EXISTS (roadmap §10.6-1). `row_aggregator` returns an all-zero
        row for any (shape group, order) with no valid fine bin, so the MC
        constrains nothing at ~37 % of MF34's parameters — yet the shipped file
        declares relative variance up to 7.6 (σ_rel ≈ 276 %) at ~95 % of them.
        Those are NOT invisible here: this function scales a relative block by
        `a_l_per_pt`, and the file's own MF4 is not zero there. Measured on run
        86: 2.14 % of the Σ_eval diagonal, reaching 82.9 % of points, with two
        small datasets above 44 %. It inflates `This_work` alone — JEFF and
        JENDL carry no such term — so it flatters us. §10.1.8-L14.2.

        Default None keeps every existing number unchanged.

    Returns
    -------
    (N, N) ndarray.  Zero if mf34 is None or N == 0.
    """
    e_mev = np.asarray(e_mev, dtype=float).ravel()
    N = e_mev.size
    if N == 0:
        return np.zeros((0, 0), dtype=float)
    if mf34 is None:
        return np.zeros((N, N), dtype=float)

    a_l_per_pt = np.asarray(a_l_per_pt, dtype=float)
    if a_l_per_pt.ndim == 1:
        a_l_per_pt = a_l_per_pt.reshape(N, -1)
    L_max = a_l_per_pt.shape[1]
    if L_max == 0:
        return np.zeros((N, N), dtype=float)

    mu = np.asarray(mu, dtype=float).ravel()
    c0 = np.asarray(c0, dtype=float).ravel()
    e_ev = e_mev * 1e6

    # base_sens[L-1, j] = c_0(E_j) * (2L+1) * P_L(mu_j); a_L scaling added per
    # block when relative.
    base_sens = _legendre_base_sens(mu, c0, L_max)  # (L_max, N)

    sigma = np.zeros((N, N), dtype=float)
    for idx in range(len(mf34.matrices)):
        l_r = int(mf34.l_rows[idx])
        l_c = int(mf34.l_cols[idx])
        if l_r < 1 or l_c < 1 or l_r > L_max or l_c > L_max:
            continue
        grid = np.asarray(mf34.energy_grids[idx], dtype=float)
        mat = np.asarray(mf34.matrices[idx], dtype=float)
        M = mat.shape[0]
        if M == 0:
            continue
        # THE ONE MAP for the Legendre family (roadmap §10.7-7). Containing-bin,
        # because ENDF LB=5/6 is piecewise-constant ON its grid; all-zero row
        # off-grid, because the file declares nothing there. Both properties
        # live in `PointMap.nearest` now, so the JEFF (2,6) case below cannot be
        # fixed in one copy of the fold and missed in another -- which is
        # exactly what happened when there were five copies.
        pm = PointMap.nearest(grid, e_ev)
        # ⚑ THE OFF-GRID CASE IS REAL AND IT FIRES ON JEFF-4.0. §L18.7 said it
        # never did; that was checked against This_work and is WRONG in general
        # (§10.7-6). JEFF publishes 20 of its 21 blocks from 1e-05 eV but
        # **(2,6) only from 1 MeV**, while the EXFOR points run down to
        # 0.85 MeV -- so every chi^2 from run 82 to 90 folded a fabricated
        # Cov(a_2, a_6) for the points below 1 MeV, for JEFF and JEFF alone.
        # This_work's own (2,6) starts at 0.846822 MeV and covers every point,
        # which is why it was invisible. Loud, because the failure mode this
        # whole section exists to prevent is silence.
        if pm.n_off_grid:
            _key = (l_r, l_c, int(N))
            if _key not in _OFF_GRID_SEEN:
                _OFF_GRID_SEEN.add(_key)
                print(
                    f"  [MF34 off-grid] block (L={l_r}, L1={l_c}) spans "
                    f"[{grid[0]:.6g}, {grid[-1]:.6g}] eV; {pm.n_off_grid} of "
                    f"{e_ev.size} points lie outside it and contribute ZERO "
                    f"through this block (the file declares nothing there). "
                    f"Roadmap §10.7-6.",
                    flush=True,
                )
        block = pm.sandwich(mat)  # (N, N), block[j,k] = mat[bin_j, bin_k], masked

        sens_r = base_sens[l_r - 1].copy()
        sens_c = base_sens[l_c - 1].copy()
        if mf34.is_relative[idx]:
            sens_r *= a_l_per_pt[:, l_r - 1]
            sens_c *= a_l_per_pt[:, l_c - 1]
        if drop is not None:
            # Zeroing the SENSITIVITY, not the block, is what removes the
            # parameter's row and its column: a point that draws on a dropped
            # (group, order) contributes nothing through it, in this term or in
            # its companion transpose below.
            sens_r = np.where(drop[:, l_r - 1], 0.0, sens_r)
            sens_c = np.where(drop[:, l_c - 1], 0.0, sens_c)

        sigma += sens_r[:, None] * block * sens_c[None, :]
        if l_r != l_c:
            # ENDF stores one of (l_r, l_c) / (l_c, l_r). The companion block's
            # cov at energy pair (E_j, E_k) is mat[bin_k, bin_j] = block.T[j,k]
            # by symmetry of the joint distribution.
            sigma += sens_c[:, None] * block.T * sens_r[None, :]

    # Symmetrize against rounding asymmetries from float ops.
    return 0.5 * (sigma + sigma.T)


# ── MF33: dense Σ_eval contribution from cross-section magnitude covariance ───

def _mf4_bin_edges_for_points(
    energies_mf4_mev: np.ndarray, e_mev: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point (E_lo, E_hi) of the MF4 bin containing each E_j (clamped).

    Mirrors `find_mf4_bin` from precompute_chi2_library_c0.py. Returns arrays
    of length N in MeV.
    """
    grid = np.asarray(energies_mf4_mev, dtype=float)
    e = np.asarray(e_mev, dtype=float).ravel()
    n = grid.size
    e_lo = np.empty_like(e)
    e_hi = np.empty_like(e)
    for j, val in enumerate(e):
        if val <= grid[0]:
            e_lo[j], e_hi[j] = grid[0], grid[1]
        elif val >= grid[-1]:
            e_lo[j], e_hi[j] = grid[-2], grid[-1]
        else:
            i_hi = int(np.searchsorted(grid, val, side="right"))
            e_lo[j], e_hi[j] = grid[i_hi - 1], grid[i_hi]
    return e_lo, e_hi


def _mf33_magnitude_map(
    mf33_grid_ev: np.ndarray, energies_mf4_mev: np.ndarray, e_mev: np.ndarray,
) -> PointMap:
    """THE ONE MAP for the magnitude family (roadmap §10.7-7).

    Overlap-average over the MF4 bin containing each point — the same
    length-weighted averaging as `avg_mf33_rel_var_over_bin`, in 2-D.

    ⚑ Every leg that touches the magnitude parameter must come through here:
    the MF33 self block AND the magnitude side of the MF33↔MF34 cross term.
    That is not tidiness. `Sigma_eval = M J M^T` is a congruence — hence PSD
    whenever `J` is — only if one `M` exists, and while the cross term used a
    nearest-bin lookup against the self block's `W`, none did.
    """
    e_lo_mev, e_hi_mev = _mf4_bin_edges_for_points(energies_mf4_mev, e_mev)
    return PointMap.overlap(
        np.asarray(mf33_grid_ev, dtype=float), e_lo_mev * 1e6, e_hi_mev * 1e6,
    )


def _mf33_overlap_weights(
    grid_ev: np.ndarray, e_lo_ev: np.ndarray, e_hi_ev: np.ndarray,
) -> np.ndarray:
    """Row-normalised (N, M) overlap weights.

    Kept as the name existing tests and callers import; it **delegates** to
    `PointMap.overlap` rather than reimplementing it, so this cannot become a
    sixth copy.
    """
    return PointMap.overlap(grid_ev, e_lo_ev, e_hi_ev).dense()


def build_mf33_block(
    mf33_grid_ev: Optional[np.ndarray],
    mf33_rel_cov: Optional[np.ndarray],
    energies_mf4_mev: np.ndarray,
    e_mev: np.ndarray,
    y_eval: np.ndarray,
) -> np.ndarray:
    """Dense N×N covariance contribution from MF33 (cross-section magnitude).

    Returns zero block if mf33 inputs are missing.

    Sigma_eval^MF33[j, k] = y_eval(j) * y_eval(k) * (W[j] @ rel_cov @ W[k]^T)

    where W is the MF4-bin-averaged overlap of each data point with the MF33
    grid (same averaging convention as the legacy diagonal path).
    """
    e_mev = np.asarray(e_mev, dtype=float).ravel()
    N = e_mev.size
    if N == 0:
        return np.zeros((0, 0), dtype=float)
    if mf33_grid_ev is None or mf33_rel_cov is None:
        return np.zeros((N, N), dtype=float)

    pm = _mf33_magnitude_map(mf33_grid_ev, energies_mf4_mev, e_mev)
    rho = pm.sandwich(np.asarray(mf33_rel_cov, dtype=float))
    y = np.asarray(y_eval, dtype=float).ravel()
    sigma = (y[:, None] * y[None, :]) * rho
    return 0.5 * (sigma + sigma.T)


# ── MF33 ↔ MF34: dense Σ_eval cross contribution (sigma ↔ a_L) ────────────────

def build_mf33_mf34_cross_block(
    cross,
    e_mev: np.ndarray,
    mu: np.ndarray,
    c0: np.ndarray,
    a_l_per_pt: np.ndarray,
    y_eval: np.ndarray,
    *,
    mf33_grid_ev: Optional[np.ndarray] = None,
    energies_mf4_mev: Optional[np.ndarray] = None,
    a_is_relative: Optional[bool] = None,
    drop: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dense N×N MF33↔MF34 cross contribution (magnitude ↔ shape).

    This is the chi^2 counterpart of the MF34 (L=0, L1) cross blocks written by
    ``create_mf34_from_covariance``.  It is **opt-in and defaults to a zero
    block** when ``cross`` is falsy, so today's Σ_eval numbers are unchanged
    (the factorization assumption Cov(sigma, a_L) = 0 stays in force until a
    joint evaluation provides these blocks).

    Parameters
    ----------
    cross : sequence of dict, or None
        Per-order cross-covariance blocks.  Each entry::

            {"l": L1,                       # Legendre order (>= 1)
             "shape_grid_ev": (Nsh+1,),     # a_L shape grid
             "matrix": (M33, Nsh),          # Cov(sigma bin, a_L1 bin)
             "is_relative": bool}           # default True

        Mirrors item-2's LB=6 (L=0, L1) block layout.  Missing/empty → zeros.
        The magnitude axis is **``mf33_grid_ev``**, not a grid of the block's
        own choosing — see below.
    e_mev, mu, c0, a_l_per_pt, y_eval
        Same per-point arrays as :func:`build_mf34_block` /
        :func:`build_mf33_block`.
    mf33_grid_ev, energies_mf4_mev
        The MF33 self block's grid and the MF4 grid, i.e. exactly what
        :func:`build_mf33_block` is given.  Required whenever ``cross`` is
        non-empty; the magnitude leg is built from them through
        :func:`_mf33_magnitude_map`, the same call the self block makes.
    a_is_relative
        The **MF34 family's** convention, i.e. ``mf34.is_relative``.  Required
        whenever ``cross`` is non-empty.  The shape leg is scaled by it — not by
        the cross block's own flag — and a block that disagrees is refused.
        See §L13 below.
    drop : (N, L_max) bool, optional
        Exactly :func:`build_mf34_block`'s ``drop``, and it **must be the same
        array**.  Removing a (point, order) from the self block but not from
        here leaves ``M_d`` in one term of Σ_eval and ``M`` in the other, so no
        single ``M`` reproduces the sum and the congruence is gone — §L13's
        failure mode wearing different clothes.  Default None, so every
        existing number is unchanged.

    Notes
    -----
    ⚑⚑ **ONE MAP PER FAMILY** (roadmap §10.7-2(a), §10.7-7).  The magnitude leg
    goes through ``_mf33_magnitude_map`` — the MF4-bin overlap average ``W`` —
    because that is what :func:`build_mf33_block` uses; the shape leg goes
    through ``PointMap.nearest`` on the block's own grid because that is what
    :func:`build_mf34_block` uses.  Then

        Σ_eval = M J M^T,  M = [ diag(y)·W | sens·S ]

    is a congruence and PSD transfers from ``J`` by algebra rather than by luck.

    ⚠ **This is a behaviour change, and it is the point.**  This function used
    to take a per-block ``mag_grid_ev`` and place points on it by nearest-bin.
    Against a self block averaging over ``W``, no single ``M`` existed, and that
    is why a certified-PSD joint kept folding indefinite (runs 87–90, four runs
    with no χ²).  A sidecar whose magnitude axis is *not* the shipped MF33 grid
    is now rejected loudly instead of folded wrongly — which is the honest
    reading of §10.1: ``Cx`` is Cauchy–Schwarz-compatible only with the
    marginals it was built from.

    ⚑⚑ **§L13, THE UNITS, and why the fix is a guard rather than a conversion.**
    The second of the two independent fold defects (§10.7-2(b)).  A *relative*
    MF34 declares ``Cov(a_L/a_L_nom, ...)``, and ENDF's reading of that is a
    multiplicative perturbation of the pointwise MF4 — so the parameter is
    ``delta_L(g) = a_L(g)/a_L_nom(g) - 1`` and the sensitivity carries
    ``a_L(E_j)``, which is what :func:`build_mf34_block` applies.  This function
    used to apply the same factor **keyed on the cross block's own
    ``is_relative``**, which the stored sidecar sets to ``False``.  Two
    conventions for one parameter, so ``Sigma_eval`` was not ``M J M^T`` and the
    joint's PSD certificate meant nothing.

    It is now keyed on ``a_is_relative`` — the MF34 family's flag — and a cross
    block declaring the other convention is **refused**.  Refused rather than
    converted on purpose: converting an absolute cross into the relative frame
    means dividing by ``a_L_nom``, which passes through zero, so the correct
    repair is upstream — build the cross from the SAME replicas as the
    marginals, in their coordinates (§10.7-8 does exactly that and gets a PSD
    joint).  A fold cannot rescue blocks that were never draws from one
    distribution.

    Sensitivities: dy/dsigma = ``y_eval`` (magnitude side), dy/da_L1 =
    ``c0*(2L1+1)*P_L1`` (shape side; scaled by the nominal a_L1 for relative
    blocks, matching :func:`build_mf34_block`).  The block is assembled
    symmetrically::

        Σ_cross[j,k] = ydsig(j)·C[j,k]·sensL(k) + sensL(j)·C.T[j,k]·ydsig(k)
    """
    e_mev = np.asarray(e_mev, dtype=float).ravel()
    N = e_mev.size
    if N == 0:
        return np.zeros((0, 0), dtype=float)
    if not cross:
        return np.zeros((N, N), dtype=float)

    a_l_per_pt = np.asarray(a_l_per_pt, dtype=float)
    if a_l_per_pt.ndim == 1:
        a_l_per_pt = a_l_per_pt.reshape(N, -1)
    L_max = a_l_per_pt.shape[1]
    if L_max == 0:
        return np.zeros((N, N), dtype=float)

    if mf33_grid_ev is None or energies_mf4_mev is None:
        raise ValueError(
            "a non-empty cross term needs `mf33_grid_ev` and "
            "`energies_mf4_mev`: its magnitude leg must reach the points "
            "through the SAME map as `build_mf33_block`, or Sigma_eval is not "
            "a congruence and PSD does not transfer (roadmap §10.7-2(a))."
        )
    if a_is_relative is None:
        raise ValueError(
            "a non-empty cross term needs `a_is_relative`, the MF34 family's "
            "convention. Keying the shape leg on the cross block's own flag is "
            "§L13: two units for one parameter (roadmap §10.7-2(b))."
        )
    a_is_relative = bool(a_is_relative)

    base_sens = _legendre_base_sens(mu, c0, L_max)  # (L_max, N)
    y = np.asarray(y_eval, dtype=float).ravel()
    e_ev = e_mev * 1e6

    # THE magnitude map — one call, identical to the self block's.
    pm_mag = _mf33_magnitude_map(mf33_grid_ev, energies_mf4_mev, e_mev)

    sigma = np.zeros((N, N), dtype=float)
    for blk in cross:
        L = int(blk["l"])
        if L < 1 or L > L_max:
            continue
        shape_grid = np.asarray(blk["shape_grid_ev"], dtype=float)
        mat = np.asarray(blk["matrix"], dtype=float)
        if mat.size == 0:
            continue
        if mat.shape[0] != pm_mag.n_bins:
            raise ValueError(
                f"cross block l={L} has {mat.shape[0]} magnitude bins but the "
                f"shipped MF33 grid has {pm_mag.n_bins}. A cross term is "
                f"Cauchy-Schwarz-compatible only with the marginals it was "
                f"built from (§10.1); rebuild it on the MF33 grid rather than "
                f"regridding it here."
            )
        # §L13. The shape leg belongs to the MF34 family, so it is scaled by
        # THAT family's convention. A block declaring the other one is not a
        # unit to be converted — it is evidence the cross was not drawn from the
        # same distribution as the marginals, and folding it anyway is what
        # produced a certified-PSD joint with an indefinite Sigma_eval.
        blk_rel = bool(blk.get("is_relative", True))
        if blk_rel != a_is_relative:
            raise ValueError(
                f"cross block l={L} declares is_relative={blk_rel} but the "
                f"MF34 family is is_relative={a_is_relative}. One parameter "
                f"cannot carry two units (§L13). Rebuild the cross in the "
                f"marginals' coordinates -- from the same replicas -- rather "
                f"than rescaling it here, which would divide by an a_L_nom "
                f"that passes through zero."
            )
        # THE shape map — identical to `build_mf34_block`'s, off-grid masking
        # included, so a block that does not span a point contributes zero
        # there on this axis too.
        pm_shape = PointMap.nearest(shape_grid, e_ev)
        C = pm_mag.sandwich(mat, pm_shape)  # (N, N): Cov(sigma_j, a_L k)

        sens_L = base_sens[L - 1].copy()
        if a_is_relative:
            sens_L *= a_l_per_pt[:, L - 1]
        if drop is not None:
            # THE SAME REMOVAL `build_mf34_block` MAKES, on the same axis.
            # Dropping a (point, order) means deleting that parameter's row and
            # column from the joint — a principal submatrix, which preserves
            # PSD. Deleting it from the self block but not from here would
            # leave `M_d` in one term and `M` in the other, so Sigma_eval would
            # stop being `M J M^T` for any single M. That is §L13's failure
            # mode in a new place, and it fires the moment the null mask and a
            # cross term are both on.
            sens_L = np.where(drop[:, L - 1], 0.0, sens_L)

        # IN PLACE, and it is not micro-optimisation. At the shipped geometry
        # (2317 magnitude bins x 703 shape groups x 6 orders) each (N, N) is
        # 6.6 GB for Cierjacks' 28631 points; writing this as
        # `y[:,None] * C * sens[None,:]` allocates two more of them per order
        # on top of `C` and `sigma`. `C` is a fresh array out of `sandwich`
        # (fancy indexing copies), so scaling it in place is safe.
        C *= y[:, None]
        C *= sens_L[None, :]
        sigma += C
        sigma += C.T

    return 0.5 * (sigma + sigma.T)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def _mf34_family_is_relative(mf34) -> Optional[bool]:
    """The one convention the whole Legendre family is folded under (§L13).

    ``None`` when there is no MF34 at all, which is fine — a cross term is
    meaningless without one and the caller only needs the flag when it has one.

    Raises when the file mixes conventions across blocks: guessing which one the
    cross term belongs to is exactly the class of mistake §L13 is.
    """
    if mf34 is None:
        return None
    flags = {bool(f) for f in mf34.is_relative} if len(mf34.is_relative) else set()
    if not flags:
        return None
    if len(flags) != 1:
        raise ValueError(
            f"MF34 mixes relative and absolute blocks {flags}; the Legendre "
            f"family is folded under ONE convention and there is no safe guess "
            f"(roadmap §L13)."
        )
    return flags.pop()

def build_eval_cov_for_groups(
    df,
    libraries: Dict[str, Dict],
    a_l_lookup: Callable[[str, Dict, float], np.ndarray],
    *,
    l_max: int,
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
) -> Dict[Tuple, np.ndarray]:
    """Group df by `group_cols`, build one dense Σ_eval per group.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-row chi^2 frame with at least: library, experiment_id, energy_mev,
        mu, c0, y_eval.
    libraries : dict[str, dict]
        Library data keyed by library label (must include the same keys used
        by `a_l_lookup` and the optional MF33 fields `mf33_grid_ev` and
        `mf33_rel_cov`, plus `mf34`).
    a_l_lookup : callable (lib_key, lib_dict, e_mev) -> a_L vector of length l_max
        Interpolates the library's nominal Legendre a_L at one energy.
    l_max : int
        Truncation order.

    Returns
    -------
    dict mapping group key tuple -> (N, N) np.float32 covariance block.
    """
    import pandas as pd  # local import to keep this module import-light

    eval_cov: Dict[Tuple, np.ndarray] = {}
    for key, sub in df.groupby(list(group_cols), observed=True, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        lib_key = key[group_cols.index("library")] if "library" in group_cols else None
        if lib_key is None or lib_key not in libraries:
            n = len(sub)
            eval_cov[key] = np.zeros((n, n), dtype=np.float32)
            continue
        lib = libraries[lib_key]

        e_mev = sub["energy_mev"].to_numpy(dtype=float)
        mu = sub["mu"].to_numpy(dtype=float)
        c0 = sub["c0"].to_numpy(dtype=float)
        y_eval = sub["y_eval"].to_numpy(dtype=float)
        N = e_mev.size

        a_l_per_pt = np.zeros((N, l_max), dtype=float)
        for i, e_i in enumerate(e_mev):
            a_l_per_pt[i] = a_l_lookup(lib_key, lib, float(e_i))

        # OPT-IN removal of MF34 parameters the evaluation never determined
        # (roadmap §10.6-1). Absent keys → drop is None → every existing number
        # is unchanged, and only the library carrying the mask is affected.
        #
        # Placed per POINT, by the same nearest-bin convention `build_mf34_block`
        # uses to place data (`searchsorted(grid, e) - 1`, clamped), so the
        # re-score removes exactly what `null_slot_exposure.py` sized.
        #
        # ⚠ Conservative where a block's grid is coarser than the mask's. The
        # shipped MF34 sits on four nested grids (673/676/684/703) and the mask
        # is on the finest, so a point in a dropped base group inside a
        # partially supported coarse bin is dropped anyway. That over-removes
        # rather than under-removes, which is the right direction for a term
        # that inflates our own Sigma_eval.
        drop = None
        _mask = lib.get("mf34_null_mask")
        if _mask is not None:
            _grid = np.asarray(lib["mf34_null_grid_ev"], dtype=float)
            _mask = np.asarray(_mask, dtype=bool)
            g_pt = np.clip(
                np.searchsorted(_grid, e_mev * 1e6, side="right") - 1,
                0, _mask.shape[0] - 1,
            )
            drop = _mask[g_pt][:, :l_max]

        sigma_mf34 = build_mf34_block(
            lib.get("mf34"), e_mev, mu, c0, a_l_per_pt, drop=drop)
        sigma_mf33 = build_mf33_block(
            lib.get("mf33_grid_ev"), lib.get("mf33_rel_cov"),
            lib.get("energies_mf4_mev"), e_mev, y_eval,
        )
        # Opt-in MF33↔MF34 cross term; None → zero block → numbers unchanged.
        # The MF33 grid and the MF4 grid go in because the cross term's
        # magnitude leg is built from them by the SAME call the self block
        # makes — that is the whole of §10.7-2(a).
        # §L13: the shape leg is scaled by the MF34 family's convention, so it
        # has to be read off the family rather than off the cross block.
        # ⚑ A cross term without its magnitude self block is never PSD. The
        # joint is [[C33, Cx], [Cx.T, C34]]; with C33 absent it is
        # [[0, Cx], [Cx.T, C34]], and any nonzero Cx then gives that matrix a
        # negative eigenvalue — a zero variance cannot carry a covariance. The
        # scenarios that fold MF34 alone (`exfor_c0`, `folded_c0`,
        # `folded_al_c0`, `representation`) simply never set the key, and
        # dropping the whole family is a principal submatrix and therefore
        # fine; what must not happen is keeping Cx and losing C33.
        if lib.get("mf33_mf34_cross") and lib.get("mf33_rel_cov") is None:
            raise ValueError(
                f"library {lib_key!r} carries an MF33<->MF34 cross term but no "
                f"MF33 self block. [[0, Cx], [Cx.T, C34]] has a negative "
                f"eigenvalue for every nonzero Cx, so this cannot be folded. "
                f"Either supply `mf33_grid_ev`/`mf33_rel_cov`, or drop the "
                f"cross term with them."
            )
        sigma_cross = build_mf33_mf34_cross_block(
            lib.get("mf33_mf34_cross"), e_mev, mu, c0, a_l_per_pt, y_eval,
            mf33_grid_ev=lib.get("mf33_grid_ev"),
            energies_mf4_mev=lib.get("energies_mf4_mev"),
            a_is_relative=_mf34_family_is_relative(lib.get("mf34")),
            drop=drop,
        )
        eval_cov[key] = (sigma_mf34 + sigma_mf33 + sigma_cross).astype(np.float32)
    return eval_cov


# ── Sidecar I/O ───────────────────────────────────────────────────────────────

_KEY_SEP = "@@"


def _encode_key(key: Tuple) -> str:
    return _KEY_SEP.join(str(p) for p in key)


def _decode_key(s: str) -> Tuple:
    return tuple(s.split(_KEY_SEP))


def save_eval_cov(path: str, eval_cov: Dict[Tuple, np.ndarray]) -> None:
    """Write all per-group blocks to a single compressed .npz file."""
    arrays = {_encode_key(k): np.asarray(v) for k, v in eval_cov.items()}
    np.savez_compressed(path, **arrays)


def load_eval_cov(path: str) -> Dict[Tuple, np.ndarray]:
    """Inverse of save_eval_cov. Returns {(library, experiment_id): array}."""
    with np.load(path) as data:
        return {_decode_key(k): np.asarray(data[k]) for k in data.files}
