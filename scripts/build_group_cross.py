#!/usr/bin/env python
"""Build the MF33<->MF34 cross block and write it into an ENDF tape.

Cov(c0, a_l) is measured across the run's own MC replicas, collapsed onto the
grids the shipped _mg tape already carries, and written into MF34's (L=0, L1)
blocks with the (0,0) self-block null -- that covariance belongs to MF33 and
must not be repeated. Everything goes out RELATIVE: of the MF34 forms only
LB=0 is absolute and it carries no off-diagonal structure.

The cross term cannot ship as a sidecar and cannot be bolted onto a foreign
MF34: it is Cauchy-Schwarz-compatible only with the marginals it was built
with, so shipping it means rewriting MF34.

Both the magnitude and shape channels are collapsed from the SAME replica set
and the two-pass rescaling is applied AFTER the collapse, as one positive
diagonal congruence on the whole joint. That is what makes the result PSD by
construction, and collapsing replicas is exactly collapsing the covariance:

    C = Z^T Z / (n-1)   =>   A C A^T = (A Z^T)(Z A^T) / (n-1)

so nothing large is ever formed.

Called by exfor_to_endf_research.py as Step 10b via
``build_cross_and_write_endf``. Standalone:

    python build_group_cross.py --run-dir <dir> --check      # diagnose only
    python build_group_cross.py --run-dir <dir> --write      # .npy sidecars
    python build_group_cross.py --run-dir <dir> --mag-grid fine --null-fill zero \
        --source-endf <mg.endf> --write-endf <out.endf>
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_kika_root = Path(__file__).resolve().parent.parent
if str(_kika_root) not in sys.path:
    sys.path.insert(0, str(_kika_root))

from scripts.point_map import PointMap  # noqa: E402  (needs the sys.path above)

L_MAX = 6
A_COLS = [f"a_{l}" for l in range(1, L_MAX + 1)]
ZA, MT = 26056, 2
NULL_TOL = 1e-10
CHUNK = 250
SPREAD_EPS = 1e-12

# ⚑ THE BAR THE PIPELINE'S OWN MF34 HAS TO CLEAR ON THE CARRIED SLOTS.
#
# `--dead-parameters carry` takes those slots straight out of `c34_ship`, and
# on run 99 that block came back with `lam_min = -8.60e-11` against a joint
# scale of 4.17e-2 -- the shipped object is very slightly not PSD there, and
# the direct sum inherits it by construction (that is the point: it MEASURES
# the pipeline instead of hiding it behind a singular congruence).
#
# Projecting that away is legitimate ONLY while it is numerical dust. The bar
# is a RATIO to the block's own largest eigenvalue, so it does not drift with
# units or with the run:
#
#   * LAPACK's own floor for a symmetric eigendecomposition is ~n*eps*lam_max,
#     i.e. ~3e-13 at n = 1307. Anything at that level is the solver, not the
#     matrix.
#   * ENDF-6 writes 11-character fields, so a round-tripped covariance carries
#     relative perturbations several orders above that floor.
#   * 1e-8 sits between the two: ~5 decades above the solver's noise and ~8
#     decades below any eigenvalue that could carry physical variance.
#
# A block that FAILS this is not dust and must not be quietly projected -- it
# means the pipeline's MF34 is genuinely indefinite on those parameters, which
# is a finding about the deliverable, not a rounding problem. `carry` raises
# there instead of clipping, and BOTH numbers below are printed on every run so
# that an object drifting toward either bar is visible BEFORE it trips it.
#
# ── BAR 1, CONDITIONING: you cannot assert PSD more tightly than the format ──
#
# `c34_ship` is `c34_rel_ship * outer(a_nom, a_nom)` and `c34_rel_ship` is READ
# BACK OUT OF THE ENDF TAPE, whose MF34 carries 7 significant digits
# (`1.665047-2`; measured on run 99's `_mg`: 582060 fields with 6 decimals,
# 39291 with 5 -- the short ones are the NEGATIVE entries, where the sign eats
# a column and only 6 digits fit). Half a unit in the last place is a relative
# perturbation of 5e-8 to 5e-7 per positive entry depending on the leading
# digit, and up to 5e-6 on a negative one; by Weyl a
# symmetric perturbation moves every eigenvalue by at most its 2-norm. So a
# block whose TRUE lam_min is exactly zero comes back off the tape at
# lam_min ~ -1e-7 * lam_max, and no gate can tell that apart from a real
# violation of the same size. Anything at or below this is the text format,
# not the matrix.
#
# ⚠ THIS BAR IS A PROPERTY OF THE FILE, NOT A TASTE. If MF34 is ever written
# with more digits, tighten it; the derivation, not the number, is the rule.
# 5e-7 is deliberately the TIGHT end of that range -- it errs toward refusing,
# since the negative entries only carry 6 digits and could justify 5e-6.
# Run 99 measured -1.514e-08: a factor 33 inside even the tight end.
DEAD_BLOCK_PSD_RTOL = 5e-7

# ── BAR 2, MASS: how much variance the projection actually invents ──────────
#
# The ratio above is a CONDITIONING number: it says the most negative direction
# is small next to the largest positive one. It does NOT say the projection is
# harmless -- a block could clear it with thousands of small negative
# eigenvalues whose total is not small at all. This is the number that answers
# "how much am I hiding": the negative spectral mass the clip has to
# manufacture, against the block's own trace, i.e. the fraction of the declared
# variance that the projection is inventing.
#
# It is the bar that actually protects the deliverable, and it is independent
# of the first: a genuinely indefinite object fails HERE even when its worst
# eigenvalue happens to look small.
#
# SAME NUMBER, SAME DERIVATION, DIFFERENT FUNCTIONAL -- and that is the point.
# Bar 1 applies the tape's floor to the WORST direction, bar 2 applies it to
# the TOTAL. Neither implies the other: one eigenvalue at 10x the floor trips
# bar 1 with negligible mass, and a thousand eigenvalues each at a tenth of the
# floor trip bar 2 while bar 1 sees nothing wrong. Both must hold. Picking two
# unrelated numbers here would have made the pair a matter of taste; deriving
# both from the file's 7 digits keeps it a statement about the object.
DEAD_BLOCK_CLIP_MASS_MAX = 5e-7


def shipped_mg_endf(run_dir: Path) -> Path:
    """The ONE shipped multigroup ENDF in a run dir. Ambiguity is fatal.

    `--write-endf` puts a SECOND *_mg.endf in the directory and it sorts BEFORE
    the shipped one, so a first-wins glob would read the shape grids and
    `c34_ship` off our own output and diagnose the candidate against itself --
    every PSD number perfect, for the one reason that makes it meaningless.
    Refuse and make the caller name the file.
    """
    hits = sorted(run_dir.glob("*_mg.endf"))
    if not hits:
        raise SystemExit(f"no *_mg.endf in {run_dir}")
    if len(hits) > 1:
        raise SystemExit(
            f"{len(hits)} *_mg.endf files in {run_dir}:\n  "
            + "\n  ".join(h.name for h in hits)
            + "\nPass --source-endf to say which one is the SHIPPED file. An "
              "earlier --write-endf output living here is the usual cause, and "
              "it sorts first, so first-wins would silently self-compare."
        )
    return hits[0]


def mf34_group_edges_ev(mg: Path, cache: Path) -> dict:
    """Per-(l_row,l_col) MF34 group edges and relative matrices, from the file.

    Takes the resolved path rather than globbing again: the grids and
    `c34_ship` MUST come from the same file `--write-endf` uses as its template,
    and two independent globs are two chances to disagree.

    Cached because the ENDF parse is the slow step and does not depend on
    anything else here.
    """
    raw = cache / f"mf34_groups__{_tape_key(mg)}.npz"
    if not raw.exists():
        from kika.cov import MF34CovMat

        m = MF34CovMat.from_endf(str(mg), energy_unit="MeV")
        m = m.filter_by_isotope_reaction(ZA, MT)
        store = {}
        for k in range(m.num_matrices):
            lr, lc = m.l_rows[k], m.l_cols[k]
            if not (1 <= lr <= L_MAX and 1 <= lc <= L_MAX):
                continue
            if not bool(m.is_relative[k]):
                raise SystemExit(f"MF34 block ({lr},{lc}) is absolute; expected relative")
            store[f"e_{lr}_{lc}"] = np.asarray(m.energy_grids[k], float)
            store[f"m_{lr}_{lc}"] = np.asarray(m.matrices[k], float)
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw, **store)
    return dict(np.load(raw))


C33_GATE_RTOL = 1e-10

def _tape_key(mg: Path) -> str:
    """A cache key that cannot name two different tapes the same thing.

    ⚠ `name + size` IS NOT ENOUGH, and the reason is a property we rely on
    elsewhere: the writer emits fixed-width ENDF records, so two runs that
    reproduce each other produce tapes of **exactly** the same size under
    exactly the same name -- which is what makes `cmp` a cheap gate. Run 92's
    `26-Fe-56g_nominal_a0cross_mg.endf` and the analytic deliverable are both
    346 800 528 B; the `_mg` pair are both 205 838 091 B. Keyed on name + size,
    the cache served run 92's MF34 for the deliverable and reported max relative
    sigma 99.7 % where the file says 6337 % (library-gaps D16).

    So the key carries the resolved parent directory AND a digest of the head
    and tail of the file. The digest is 2 MB of reads, not 346, because a cache
    lookup must stay cheap; the parent directory is what actually separates
    reruns, and the digest is what catches an in-place rewrite.
    """
    import hashlib

    st = mg.stat()
    h = hashlib.sha1()
    h.update(str(mg.resolve().parent).encode())
    h.update(str(st.st_size).encode())
    with open(mg, "rb") as fh:
        h.update(fh.read(1 << 20))
        if st.st_size > (2 << 20):
            fh.seek(-(1 << 20), 2)
            h.update(fh.read(1 << 20))
    return f"{mg.name}__{st.st_size}__{h.hexdigest()[:16]}"


def c33_matrix_gate(c33_post, c33_ship, *, fatal: bool,
                    rtol: float = C33_GATE_RTOL) -> float:
    """Is the joint's magnitude block the SAME MATRIX the chi2 folds?

    The marginal-identity check pins only the DIAGONAL, and `Sigma_eval = M J M^T`
    transfers J's PSD to the chi2 only if J's c33 block is the matrix
    `build_mf33_block` folds -- not merely one with the same variances.

    On the FINE axis it is, by algebra: A_mag = I, so both sides are
    corr1 (*) sigma sigma^T. On the GROUP axis it is not, because
    corr(A C A^T) != A corr(C) A^T -- hence `fatal` rather than two functions.
    """
    num = float(np.abs(np.asarray(c33_post) - np.asarray(c33_ship)).max())
    den = float(np.abs(np.asarray(c33_ship)).max())
    rel = num / den if den > 0 else float("inf")
    print(f"  c33 matrix gate: max|c33_post - c33_ship| = {num:.6e}   "
          f"relative = {rel:.6e}"
          + ("" if fatal else "   [group axis: informational]"))
    if fatal and rel > rtol:
        raise SystemExit(
            f"c33_post and the shipped MF33 differ by {rel:.3e} relative, "
            f"above the {rtol:.0e} gate. On the fine axis they must be the "
            f"same matrix; run 86 measures 4.6e-16. Something broke the "
            f"two-pass identity -- check for NaN Pass-1 c0 (the "
            f"pairwise-complete np.ma.corrcoef in `combine_c0_covariance` then "
            f"disagrees with this script's drop-incomplete rule) or for fine "
            f"bins with sigma_mc = 0 (they get j33 = 0 while the file declares "
            f"sigma > 0). Nothing downstream is a congruence until this "
            f"passes, so nothing has been written."
        )
    return rel


def mf33_file_grid_ev(mg: Path, cache: Path) -> np.ndarray:
    """The MF33 energy grid as the chi2 reads it back, cached.

    ⚑ Read through `load_library_lib_c0`, the chi2's own entry point, and NOT
    reconstructed from the run's `.npy` sidecars. The a_0 blocks' row grid has
    to equal this array element-wise or `read_mf34_split` refuses to fold them,
    and the two are not interchangeable: the ENDF parser evaluates
    `mantissa * 10**exp`, so `2.000500+6` comes back as 2000500.0000000002
    while the in-memory grid holds 2000499.9999999998 -- one ULP, on 613 of
    1739 edges (roadmap §10.7-10, 0.7).

    Writing the file's own floats back out IS idempotent, because they
    re-format to the same 11 characters. Reconstructing them is not. Hence this
    function rather than `np.load(... mf33_energy_grid_ev.npy)`.
    """
    raw = cache / f"mf33_file_grid__{_tape_key(mg)}.npy"
    if not raw.exists():
        from scripts.precompute_chi2_library_c0 import load_library_lib_c0
        lib = load_library_lib_c0(str(mg), "source (for the MF33 grid)")
        g = lib.get("mf33_grid_ev")
        if g is None:
            raise SystemExit(
                f"{mg} has no readable MF33/MT=2, so the a_0 blocks have no "
                f"magnitude axis to be written on."
            )
        cache.mkdir(parents=True, exist_ok=True)
        np.save(raw, np.asarray(g, float))
    return np.load(raw)


def base_shape_grid(blocks: dict) -> np.ndarray:
    """A grid every MF34 block's grid is a subset of, so duplication is well defined.

    With one mesh shared by all orders — every tape written before roadmap
    §10.8 — the blocks are nested and this is the finest of them, which is what
    it used to return outright.

    ⚑ WITH A MESH PER ORDER THEY NEED NOT BE NESTED, and refusing was wrong.
    ENDF-6 states the grids inside each (L, L1) sub-subsection precisely so that
    a_1 and a_5 can be written at different resolutions; two coarsenings of the
    same parent mesh are then both legal and neither contains the other. What
    the expansion actually needs is a common refinement, and the union of the
    block grids IS one — every block's grid is a subset of it by construction,
    so `shipped_c34_rel_on_base`'s duplication stays exactly as well defined as
    it was.

    The old guard survives only as a postcondition, and it can no longer fire:
    every block grid is a subset of the union BY CONSTRUCTION. The case it used
    to catch — a boundary falling strictly inside another grid's interval — is
    not an error any more, because the union splits that interval and both
    blocks expand onto it unambiguously. It is kept as a cheap invariant on how
    `base` is built, not as a check on the data.
    """
    keys = [k[2:] for k in blocks if k.startswith("m_")]
    grids = [np.asarray(blocks[f"e_{k}"], dtype=float) for k in keys]
    # `np.unique` also DEDUPLICATES, so a block grid carrying a repeated
    # boundary — a zero-width group — would come back one group shorter than the
    # old `max(..., key=len)` returned, silently changing n_gs. Refuse instead.
    for k, g in zip(keys, grids):
        if np.any(np.diff(g) <= 0):
            raise SystemExit(
                f"MF34 block {k}'s grid is not strictly increasing; it has "
                f"{int((np.diff(g) <= 0).sum())} zero-width or reversed groups.")
    base = np.unique(np.concatenate(grids)) if grids else np.array([])
    for k, g in zip(keys, grids):
        if not np.isin(g, base).all():
            raise SystemExit(
                f"MF34 block {k} sits on a grid that is not a subset of the "
                f"union of all block grids; expanding by duplication would be "
                f"wrong."
            )
    return base


def fine_to_group(fine_edges_ev: np.ndarray, target_edges_ev: np.ndarray) -> np.ndarray:
    centres = 0.5 * (fine_edges_ev[:-1] + fine_edges_ev[1:])
    return np.clip(np.searchsorted(target_edges_ev, centres, side="right") - 1,
                   0, len(target_edges_ev) - 2)


def row_aggregator(g, widths, n_groups, valid):
    """A[group, fine] = w_i / sum_{j in group, valid} w_j.

    Width weights, one block per order — the same definition as
    `multigroup_collapse.build_aggregation_matrix` and its scalar form
    `build_l0_row_aggregator`. Groups with no valid bin at this order stay
    all-zero, contributing a genuine null direction rather than a fabricated
    value.
    """
    A = np.zeros((n_groups, len(g)))
    A[g, np.arange(len(g))] = np.where(valid, widths, 0.0)
    tot = A.sum(axis=1, keepdims=True)
    np.divide(A, tot, out=A, where=tot > 0)
    return A


def load_pass1_c0(run_dir: Path, n_fine: int):
    t = pq.read_table(run_dir / "mf33_c0_samples.parquet",
                      columns=["sample_idx", "energy_index", "pass", "c0"],
                      filters=[("pass", "=", "pass1")])
    s = t.column("sample_idx").to_numpy()
    ids = np.unique(s)
    out = np.full((ids.size, n_fine), np.nan)
    out[np.searchsorted(ids, s), t.column("energy_index").to_numpy()] = \
        t.column("c0").to_numpy()
    return out, ids


def _read_a_chunk(run_dir, i0, i1, ids):
    """Pass-1 a_l for fine bins [i0, i1), rows aligned to `ids`.

    THE JOIN KEY, WHICH IS THE ONE THING THAT SILENTLY DESTROYS THE ANSWER.
    `mf33_c0_samples` writes the internal replica index (exfor_utils.py:2776);
    `save_all_legendre_coefficients` reserves sample_idx 0 for the nominal row
    and writes MC draw s as s+1 (exfor_utils.py:5087). Joining on the raw column
    pairs c0 replica k with a_l replica k-1 — a random pairing, which forces rho
    to the MC noise floor whatever the truth is, and equally forces it below a
    permutation null because both are then null. Sec 10.1.8's retraction is
    exactly this bug.
    """
    t = pq.read_table(
        run_dir / "legendre_samples_tmc.parquet",
        columns=["sample_idx", "energy_index", "is_nominal", *A_COLS],
        filters=[("energy_index", ">=", i0), ("energy_index", "<", i1),
                 ("is_nominal", "=", False)],
    )
    s = t.column("sample_idx").to_numpy() - 1
    e = t.column("energy_index").to_numpy() - i0
    pos = np.searchsorted(ids, s)
    blk = np.zeros((ids.size, i1 - i0, L_MAX))
    for k, col in enumerate(A_COLS):
        blk[pos, e, k] = t.column(col).to_numpy()
    return blk


def valid_and_collapsed(run_dir, n_fine, ids, g_shape, widths, n_gs):
    """Two passes over the TMC parquet: the validity mask, then the collapse.

    The mask has to exist before the aggregator weights can be built, and the
    weights before the replicas can be collapsed, so the file is read twice
    rather than held in memory (10000 x 1738 x 6 would be 834 MB).
    """
    valid = np.zeros((n_fine, L_MAX), dtype=bool)
    for i0 in range(0, n_fine, CHUNK):
        i1 = min(i0 + CHUNK, n_fine)
        blk = _read_a_chunk(run_dir, i0, i1, ids)
        sd = blk.std(axis=0, ddof=1)
        scale = np.maximum(np.abs(blk).max(axis=0), np.finfo(float).tiny)
        valid[i0:i1] = sd > SPREAD_EPS * scale
        print(f"    mask {i0}:{i1}", flush=True)

    A = [row_aggregator(g_shape, widths, n_gs, valid[:, l]) for l in range(L_MAX)]

    acc = np.zeros((ids.size, n_gs, L_MAX))
    for i0 in range(0, n_fine, CHUNK):
        i1 = min(i0 + CHUNK, n_fine)
        blk = _read_a_chunk(run_dir, i0, i1, ids)
        for l in range(L_MAX):
            acc[:, :, l] += blk[:, :, l] @ A[l][:, i0:i1].T
        print(f"    collapse {i0}:{i1}", flush=True)
    return valid, A, acc


def carry_dead_parameters(c34_post, c34_ship, d_mc34, d_tar34,
                          psd_rtol=DEAD_BLOCK_PSD_RTOL,
                          clip_mass_max=DEAD_BLOCK_CLIP_MASS_MAX):
    """Put back the shape parameters the diagonal congruence cannot reach.

    ``jj = d_tar / d_mc`` is a diagonal congruence, and a congruence cannot give
    variance to a direction the Monte Carlo never explored: where ``d_mc = 0``
    the whole row and column of ``c34_post`` go to zero and take the pipeline's
    own sigma with them. That is not a modelling decision, it is the rescaling
    deleting what it cannot rescale -- 1542 of 4218 slots on run 99, nearly all
    of a_4..a_6, and the count does not shrink with more replicas because what
    freezes them is the AIC mixture's BETWEEN-model variance, which the replicas
    do not carry (the order cap and the analytic substitution both bite at fixed
    n).

    Those slots come back as a DIRECT SUMMAND out of ``c34_ship``, never as a
    patch of the full block, and that choice is what keeps the diagnosis honest:

    * ``c34_post`` is exactly zero on D in both its rows and its columns, so the
      result is ``block_diag(live, c34_ship[D, D])`` and its smallest eigenvalue
      is ``min`` of the two. A principal submatrix of a PSD matrix is PSD, so
      the PSD gate becomes a real measurement of the pipeline's own MF34 instead
      of a tautology about the chain being a congruence.
    * the whitener of a block-diagonal matrix is block diagonal, and the cross
      block is exactly zero on D, so ``||W33 Cx W34||_2`` cannot move. Row F has
      to land on the CONTROL row exactly as before.

    The alternative -- keeping the pipeline's ``corr(dead, live)`` as well -- has
    neither property, which is why the caller prices it as a diagnostic row
    rather than choosing between them on argument.

    THE CARRIED BLOCK IS PROJECTED ONTO THE PSD CONE, AND ONLY WHILE THAT
    PROJECTION IS DUST. ``c34_ship`` restricted to the dead slots is not quite
    PSD (run 99: ``lam_min/lam_max`` a few times 1e-9), so the direct sum
    inherits a negative eigenvalue that the old ``drop`` path never showed --
    ``drop`` reached machine zero only because the singular congruence had
    ANNIHILATED those very directions. Clipping the negative eigenvalues to
    zero is safe here for the same reason the direct sum is: the block is
    separate and ``cx`` is exactly zero on it, so ``sigma_max(K)`` provably
    cannot move. Above ``psd_rtol`` it is not dust any more, and this raises
    instead: an indefinite MF34 on real parameters is a finding about the
    deliverable, not something a projection should absorb.

    Returns
    -------
    (c34_post, dead, orphan, info)
        ``c34_post`` modified in place, the boolean mask of what was carried,
        the count of slots neither the replicas nor the pipeline declare, and
        a dict with the carried block's spectrum and what the projection cost.
    """
    d_mc34 = np.asarray(d_mc34, float)
    d_tar34 = np.asarray(d_tar34, float)
    dead = (d_mc34 <= 0.0) & (d_tar34 > 0.0)
    orphan = int(((d_mc34 <= 0.0) & (d_tar34 <= 0.0)).sum())
    info = {"lam_min": 0.0, "lam_max": 0.0, "ratio": 0.0, "n_clipped": 0,
            "d_shift_rel": 0.0, "clip_mass": 0.0, "trace": 0.0}
    if dead.any():
        D = np.ix_(dead, dead)
        if np.abs(c34_post[dead, :]).max() > 0.0:
            raise SystemExit(
                "a row with zero MC spread is not identically zero in c34_post; "
                "the congruence and the mask disagree and the direct-sum "
                "argument does not hold.")
        blk = 0.5 * (c34_ship[D] + c34_ship[D].T)
        w, V = np.linalg.eigh(blk)
        lam_min, lam_max = float(w[0]), float(w[-1])
        neg = w[w < 0.0]
        tr = float(np.sum(np.abs(w)))
        mass = float(-neg.sum() / tr) if tr > 0 else 0.0
        info.update(lam_min=lam_min, lam_max=lam_max,
                    ratio=(lam_min / lam_max if lam_max > 0 else -np.inf),
                    n_clipped=int(neg.size), clip_mass=mass, trace=tr)
        if lam_min < 0.0:
            if lam_min < -psd_rtol * lam_max:
                raise SystemExit(
                    f"the pipeline's own MF34 is INDEFINITE on the "
                    f"{int(dead.sum())} carried parameters: lam_min "
                    f"{lam_min:.6e}, lam_max {lam_max:.6e}, ratio "
                    f"{lam_min / lam_max:.3e} against the bar "
                    f"-{psd_rtol:.0e}. That is {abs(lam_min / lam_max) / psd_rtol:.1f}x "
                    f"too negative to be the tape's own 7-digit rounding, so it "
                    f"is NOT projected away: a real negative direction in the "
                    f"shipped object would be hidden, not fixed. Find out why "
                    f"c34_ship is indefinite there before carrying it.")
            if mass > clip_mass_max:
                raise SystemExit(
                    f"the PSD projection would have to invent "
                    f"{mass:.3e} of the carried block's variance "
                    f"({neg.size} negative eigenvalues summing to "
                    f"{-neg.sum():.6e} against a trace of {tr:.6e}), over the "
                    f"bar {clip_mass_max:.0e}. The worst eigenvalue is small "
                    f"but the total is not, which is exactly the case the "
                    f"ratio bar cannot see. This is a defect in the pipeline's "
                    f"MF34, not rounding.")
            d0 = np.diag(blk).copy()
            blk = (V * np.maximum(w, 0.0)) @ V.T
            blk = 0.5 * (blk + blk.T)
            info["d_shift_rel"] = float(np.max(
                np.abs(np.diag(blk) - d0)
                / np.maximum(np.abs(d0), np.finfo(float).tiny)))
        c34_post[D] = blk
    return c34_post, dead, orphan, info


def assemble_c34_rel(c34_post, a_flat, c34_rel_ship, null_fill="zero"):
    """The RELATIVE MF34 the writer emits. One definition, used by both paths.

    Rebuilt as `c34_post / outer(a, a)` wherever a_l is nonzero. Where a_l is
    exactly zero -- the (group, order) pairs the aggregator never populated, ~37 %
    of slots -- the quotient is 0/0 and `null_fill` decides:

      "zero"  the parameter is dead, so it carries no covariance. PSD-safe.
      "ship"  keep the file's declared value. Preserves the published relative
              sigma but injects the old construction into the softest directions
              of the new one; this reproduced the run-89 failure.
    """
    if null_fill not in ("zero", "ship"):
        raise SystemExit(f"unknown null_fill {null_fill!r}; use zero or ship")
    live = np.abs(a_flat) >= np.finfo(float).tiny
    both = np.outer(live, live)
    base = (np.array(c34_rel_ship, float) if null_fill == "ship"
            else np.zeros_like(c34_post))
    return np.divide(c34_post, np.outer(a_flat, a_flat), out=base, where=both)


def _snap_to_base(g_run, base_ev, label, tol_rel=5e-5):
    """The mesh's boundaries, replaced by the base grid's own values for them.

    ⚑ THE NPZ IS FLOAT64; THE TAPE IS SIX SIGNIFICANT DIGITS. The mesh is
    chosen on the multigroup boundaries and saved as it stands
    (``847499.9999999999``); by the time the same boundary comes back off the
    tape it has been through ``format_endf_number`` and reads ``8.475000+5``.
    That moves 287 of a_1's 661 boundaries, so exact set membership called our
    own mesh foreign and run 97 died on the guard below with every other
    artefact already on disk.

    The tolerance is what separates a text round trip from a real difference.
    Six significant digits move a value by at most 5e-6 relative -- half a unit
    in the last place, worst case at mantissa 1.0; measured maximum over the
    six meshes 2.0e-6 -- while the narrowest group in any of them is 4.0e-4
    relative wide. 5e-5 sits an order above the round trip and an order below
    the narrowest group's half width, so it can only ever snap a boundary onto
    itself.

    Anything outside that band is NOT a round trip and is refused: it means the
    npz and the tape did not come from the same run, and silently accepting it
    would emit the cross term on a mesh the shape blocks are not written on.
    """
    base_ev = np.asarray(base_ev, float)
    out = np.asarray(g_run, float)
    if base_ev.size < 2:
        raise SystemExit(f"{label}: the base shape grid has no groups to snap onto.")
    j = np.clip(np.searchsorted(base_ev, out), 1, base_ev.size - 1)
    lo, hi = base_ev[j - 1], base_ev[j]
    near = np.where(np.abs(out - lo) <= np.abs(hi - out), lo, hi)
    off = np.abs(near - out) / np.maximum(np.abs(out), 1.0)
    if np.any(off > tol_rel):
        k = int(np.argmax(off))
        raise SystemExit(
            f"{label}: mesh boundary {out[k]!r} eV is not in the base shape "
            f"grid and is not a rounding of anything in it (nearest "
            f"{near[k]!r}, {off[k]:.2e} relative, tolerance {tol_rel:.0e}). "
            f"The npz and the tape do not come from the same run.")
    if np.any(np.diff(near) <= 0):
        raise SystemExit(
            f"{label}: snapping put two mesh boundaries on the same base "
            f"boundary, which would emit a zero-width group.")
    return near


def per_order_shape_grids(blocks: dict, base_ev: np.ndarray, run_dir=None,
                          l_max: int = L_MAX) -> dict:
    """``{l: e_l}`` — the mesh each Legendre order's parameters actually live on.

    ENDF-6 states the grids inside each (L, L1) sub-subsection, so order l's
    parameters are the ones its DIAGONAL block declares: the off-diagonal
    (l, l1) records are LB=6 on the pair (e_l, e_l1), and reading them back
    squares them onto the union of the two, which is why they must not be used
    to infer either mesh.

    ⚑ THE RUN'S OWN `mf34_per_order_mesh.npz` WINS when it is there. The tape's
    grids have been through `merge_mf34` with the host by the time this reads
    them, and a host boundary landing inside our window would re-split a group
    the mesh deliberately merged. Deriving the mesh from the tape is the
    fallback for a foreign file, not the primary source.
    """
    base_ev = np.asarray(base_ev, float)
    from_npz = None
    if run_dir is not None:
        p = Path(run_dir) / "mf34_per_order_mesh.npz"
        if p.exists():
            z = np.load(p)
            from_npz = {l: np.asarray(z[f"e_{l}"], float)
                        for l in range(1, l_max + 1) if f"e_{l}" in z}
            if len(from_npz) != l_max:
                from_npz = None

    grids = {}
    for l in range(1, l_max + 1):
        key = f"e_{l}_{l}"
        if key not in blocks:
            raise SystemExit(
                f"MF34 has no (L={l}, L1={l}) block, so order {l}'s mesh is not "
                f"stated anywhere; the cross term cannot be put on it.")
        g = np.asarray(blocks[key], float)
        if from_npz is not None:
            # Snap FIRST: everything below compares mesh boundaries against
            # tape boundaries, and on raw float64 every one of those
            # comparisons is wrong in the same way -- `extra` would report our
            # own rounded boundaries as the host's, and the closing guard would
            # refuse the run.
            g_run = _snap_to_base(from_npz[l], base_ev, f"a_{l}")
            # ⚑ THE MESH SPANS OUR WINDOW; THE BASE GRID SPANS THE HOST'S RANGE.
            # `merge_mf34` gave the tape the host's coverage outside our fits,
            # where a_l is exactly zero and the emission writes zeros. Splice
            # those boundaries back so the per-order grids cover exactly what
            # the shared-mesh emission covered: then the ONLY thing the mesh
            # changes is the resolution INSIDE the window, and the outside
            # groups carry the same nothing they carry today.
            outside = np.concatenate([base_ev[base_ev < g_run[0] - 1e-3],
                                      base_ev[base_ev > g_run[-1] + 1e-3]])
            g_run = np.unique(np.concatenate([g_run, outside]))
            extra = g[(g > g_run[0]) & (g < g_run[-1]) & ~np.isin(g, g_run)]
            if extra.size:
                # The host's own MF34 boundaries inside our window (JEFF-4.0
                # declares 14, on a 0.2 MeV grid) come back through the union
                # the merge takes. Re-splitting a group the DP merged is not
                # fatal -- it costs resolution the mesh said was not supported,
                # nothing more -- but the emission must then use the mesh, not
                # the tape, or shape and cross would sit on different grids.
                print(f"    a_{l}: the tape has {extra.size} in-window "
                      f"boundaries the mesh merged away (host merge); emitting "
                      f"on the mesh")
            g = g_run
        if not np.isin(g, base_ev).all():
            raise SystemExit(
                f"a_{l}'s mesh is not a subset of the base shape grid; the "
                f"collapse onto it would not be a congruence.")
        grids[l] = g
    return grids


def _fixed_cut_points(order_ev, base_ev, u):
    """``order_ev`` refined with the boundaries the CENTRAL VALUES force.

    Two of them, and both are properties of ``u = w * a`` alone -- no covariance
    is involved, so the rule reads the same here as it does inside the DP.

    ⚑ A SIGN CHANGE OF ``a_l`` IS A BOUNDARY. ``rel_S = Var_S / mean_S^2``, and
    the mean of a segment straddling a zero crossing is a DIFFERENCE, not an
    average: it can be arbitrarily small while its terms are not. Run 98 merged
    across crossings and the emission blew ``max|rel|`` from 0.9945 to 260.4,
    with ``(sum|u| / |sum u|)^2`` reaching 2.6e5 at a_5. With every coarse group
    of one sign, ``sum|u| == |sum u|``, so ``|rel_S| <= max|rel|`` termwise and
    the collapse is a CONTRACTION rather than something the guard below has to
    catch. a_2 has no crossing, ratio exactly 1, and it is the one order the DP
    never touched.

    ⚑ ENFORCED HERE AS WELL AS IN THE DP because the two see different values.
    The DP works on the pipeline's multigroup mesh with the mixture's means; this
    works on the tape's base grid with the means the MC replicas actually
    support (a_1 1738 fine bins, a_4 727, a_6 106). The theorem holds for
    whichever ``u`` the collapse is done with, so it has to be checked with THIS
    one.

    ⚑ A dead base group keeps both edges.

    ⚑ THE BASE GRID HAS GROUPS THE MESH NEVER SAW. The DP chooses the mesh on
    the pipeline's 660-group multigroup mesh, but what comes back off the tape
    has been through `merge_mf34` with the host, which splits our groups at the
    host's own boundaries -- 14 of them inside the analysis window. A sliver
    between a host boundary and ours can contain no fine bin at all, and then
    its ``a_l`` is exactly zero. The DP could not make those cut points because
    on its own grid they do not exist: run 98 died with 10 of them at a_1
    absorbed into live coarse groups.

    Keeping BOTH edges of every dead base group is the conservative repair. Each
    one then stays exactly one coarse group and is written zero -- which is what
    the shared-mesh emission declared for it -- and the merging happens only
    among live groups, which is what the mesh is for. It costs at most one cut
    point per dead group, against meshes of 550-660.

    ⚠ Not the same thing as letting them merge. The aggregator weights are
    ``w * a``, so a dead member would contribute zero weight and nothing would
    be *fabricated* -- but the coarse group would then declare the neighbour's
    relative covariance over an interval where we fitted nothing, which is the
    same objection that keeps the out-of-window groups untouched.
    """
    base_ev = np.asarray(base_ev, float)
    order_ev = np.asarray(order_ev, float)
    u = np.asarray(u, float)
    if u.size != base_ev.size - 1:
        raise SystemExit(
            f"the weight vector has {u.size} entries for {base_ev.size - 1} base "
            f"groups; it is not on the base grid.")
    keep = np.isin(base_ev, order_ev)
    idx = np.flatnonzero(u == 0.0)
    keep[idx] = True
    keep[idx + 1] = True
    # Between two adjacent LIVE groups of opposite sign. Dead groups already
    # separate what lies either side of them, so they need no second rule.
    sgn = np.sign(u)
    both = (sgn[:-1] != 0) & (sgn[1:] != 0)
    keep[1:-1] |= both & (sgn[:-1] != sgn[1:])
    out = base_ev[keep]
    if not np.isin(order_ev, out).all():
        raise SystemExit("refining the mesh dropped one of its own boundaries.")
    return out


def order_emission_weights(base_ev, order_ev, base_valid_width, a_base):
    """``U[coarse, base]``: the map that collapses RELATIVE MF34 entries exactly.

    The absolute collapse is width-weighted over the bins the aggregator counts
    as valid, and the group central value is the same width-weighted mean, so

        rel_S = abs_S / m_S^2
              = sum_ij (w_i a_i)(w_j a_j) rel_ij / (sum_i w_i a_i)^2

    i.e. the relative entries average with weights ``u = w * a``.  Absolute
    never appears, which matters because what reaches this point has been
    through the near-zero regularisation, the between-experiment floor and the
    order smoothing — steps that put content exactly where ``a`` is zero and an
    absolute round trip would delete it.

    ⚑ EXACTLY TRANSITIVE, which is why the collapse can happen here instead of
    on the replicas: a base group's ``w_g * a_g`` IS the sum of ``w_i * a_i``
    over the fine bins it covers, so composing fine -> base -> coarse gives the
    same weights as fine -> coarse.  Everything upstream — the marginal-identity
    gate, the PSD diagnosis, the Cauchy-Schwarz margin — therefore stays on the
    base grid where it is comparable with every previous run, and the emission
    is a congruence of the object that was diagnosed, so PSD survives it.

    ⚠ ``a`` CAN CHANGE SIGN, and then ``sum_i w_i a_i`` can approach zero and the
    relative form explodes.  That is a true property of the relative form, not
    an artefact — and it is exactly what the mesh's SNR >= 1 criterion and its
    no-cancellation constraint exist to prevent.  A mesh that did not come from
    that DP can violate it, so the caller checks the result.
    """
    base_ev = np.asarray(base_ev, float)
    order_ev = np.asarray(order_ev, float)
    g = fine_to_group(base_ev, order_ev)
    U = np.zeros((len(order_ev) - 1, len(base_ev) - 1))
    U[g, np.arange(len(g))] = np.asarray(base_valid_width, float) * np.asarray(a_base, float)
    tot = U.sum(axis=1, keepdims=True)
    np.divide(U, tot, out=U, where=tot != 0)
    return U


def shipped_c34_rel_on_base(blocks, base_ev):
    """Shipped RELATIVE MF34 expanded onto the base group grid.

    Coarser blocks are mapped by duplication, which is what the file asserts.

    Out-of-range must be ZERO, not pinned: the blocks come back on four nested
    grids because `merge_mf34` builds each (L, L1) pair on that pair's own union
    of the host grid and the overlay, so not all of them span the base grid.
    Clipping a `searchsorted` index would silently reuse the first or last
    interval instead.
    """
    n_g = len(base_ev) - 1
    centres = 0.5 * (base_ev[:-1] + base_ev[1:])
    out = np.zeros((n_g * L_MAX, n_g * L_MAX))
    for lr in range(1, L_MAX + 1):
        for lc in range(1, L_MAX + 1):
            key = f"{min(lr, lc)}_{max(lr, lc)}"
            if f"m_{key}" not in blocks:
                continue
            edges, mat = blocks[f"e_{key}"], blocks[f"m_{key}"]
            if lr > lc:
                mat = mat.T
            pm = PointMap.nearest(edges, centres)
            out[np.ix_(np.arange(n_g) * L_MAX + lr - 1,
                       np.arange(n_g) * L_MAX + lc - 1)] = pm.sandwich(mat)
    return 0.5 * (out + out.T)


def shipped_c34_on_base(blocks, base_ev, a_nom_group):
    """Absolute shipped MF34 expanded onto the base group grid.

    Relative -> absolute via outer(a_nom, a_nom), never the reverse.

    ⚠ THIS CONVERSION IS LOSSY WHERE a_nom IS ZERO, and that is not a corner
    case: `row_aggregator` returns an all-zero row for any (group, order) with
    no valid fine bin, so a_nom is EXACTLY zero at ~37 % of the parameters
    (rising with order: 39 of 703 groups at l=1, 572 at l=6). The file declares
    nonzero RELATIVE variance at ~95 % of those, and this multiply sends all of
    it to zero. Nothing is wrong with that for the joint — the collapsed
    replicas are identically zero there too, so both marginals agree at zero and
    the direction is a genuine null one — but it does mean the
    marginal-identity gate is VACUOUS at those slots (it compares 0 with 0), and
    the relative content cannot be recovered by dividing back out.
    """
    a = np.asarray(a_nom_group, float).reshape(-1)
    out = shipped_c34_rel_on_base(blocks, base_ev) * np.outer(a, a)
    return 0.5 * (out + out.T)


def whiten(A, tol=NULL_TOL):
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    keep = w > tol * max(w.max(), 0.0)
    return (V[:, keep] @ np.diag(w[keep] ** -0.5) @ V[:, keep].T,
            V[:, ~keep] @ V[:, ~keep].T, int(keep.sum()))


def spectrum(name, A):
    """What a matrix's eigenvalues say about whether whitening it means anything.

    The three numbers that matter are the smallest RETAINED eigenvalue (it sets
    the `w^-1/2` amplification), the most NEGATIVE one (it is the noise floor --
    a covariance cannot have one, so its size is how much garbage the round trip
    left), and their ratio. If the retained minimum sits below the noise floor,
    the pseudo-inverse is inverting noise. Sec. 10.1.8-L16.1.
    """
    w = np.linalg.eigvalsh(0.5 * (np.asarray(A, float) + np.asarray(A, float).T))
    keep = w > NULL_TOL * w.max()
    wmin_keep = float(w[keep].min()) if keep.any() else float("nan")
    wneg = float(w.min())
    return {"matrix": name, "n": A.shape[0], "rank@NULL_TOL": int(keep.sum()),
            "lam_max": float(w.max()), "lam_min_kept": wmin_keep,
            "amplif_w^-0.5": wmin_keep ** -0.5 if wmin_keep > 0 else np.inf,
            "cond_kept": float(w.max()) / wmin_keep if wmin_keep > 0 else np.inf,
            "n_negative": int((w < 0).sum()), "lam_most_negative": wneg,
            "kept_min/|most_neg|": abs(wmin_keep / wneg) if wneg < 0 else np.inf}


def diagnose(name, c33, c34, cx, tol=NULL_TOL):
    """PSD diagnostics for the joint [[c33, cx], [cx.T, c34]].

    ⚠ `lam_min_norm` IS NOT COMPARABLE ACROSS ROWS THAT DIFFER IN THEIR
    DIAGONAL, which is why `scale` and the raw `lam_min` are both reported now
    (roadmap Sec. 10.1.8-L12). The normaliser is `max|diag(J)|`, and in the
    relative-space table that maximum is a NULL-SLOT relative variance: the
    shipped file reaches 7.6 there, so the `null=ship` row is divided by 7.6 and
    the `null=zero` row is not. Read that way, -3.462 -> -26.45 is a factor 7.64
    against a removed diagonal of 7.6 -- i.e. the violation did not move at all,
    and the scale-free `sigma_max(K)` column agrees (1544.7 -> 1561.5, 1 %).

    Compare `sigma_max(K)` across rows. Compare `lam_min_norm` only against rows
    with the same `scale`.

    ⚠ `sigma_max(K)` AND `leak_null34` ARE TOLERANCE-DEPENDENT; `lam_min` IS
    NOT. `whiten` keeps `w > tol*w_max` and inverts the square root, so both
    columns are set by the smallest RETAINED eigenvalue. Against the file's
    2317-bin MF33 that is 6.37e-10 while the matrix's own most negative
    eigenvalue is -1.27e-07 -- 200x larger in magnitude -- so at NULL_TOL the
    whitening inverts directions below the file's numerical noise floor and
    sigma_max(K) measures nothing (18837 vs 1560 for the same Cx, roadmap
    Sec. 10.1.8-L16.1). Use `--tol-sweep` before reading either column off a
    matrix that has been through an ENDF round trip.
    """
    w33, p33, r33 = whiten(c33, tol)
    w34, p34, r34 = whiten(c34, tol)
    nx = float(np.linalg.norm(cx, "fro"))
    # `0.5 * (J + J.T)` block by block instead of on the assembled joint. Same
    # matrix to the last bit -- the symmetrised off-diagonal is
    # `0.5 * (cx + (cx.T).T) = cx` -- but the temporaries are the BLOCKS, not
    # two more copies of the joint. On the fine magnitude grid the joint is
    # 5956 x 5956 = 271 MB, so the naive form asks for 1.4 GB per row of the
    # table and dies on a 12 GB box before the CONTROL row prints.
    m = c33.shape[0]
    j = np.empty((m + c34.shape[0],) * 2)
    j[:m, :m] = 0.5 * (c33 + c33.T)
    j[m:, m:] = 0.5 * (c34 + c34.T)
    j[:m, m:] = cx
    j[m:, :m] = cx.T
    lam = float(np.linalg.eigvalsh(j)[0])
    sc = float(np.max(np.abs(np.diag(j))))
    del j
    return {"case": name, "rank33": f"{r33}/{c33.shape[0]}",
            "rank34": f"{r34}/{c34.shape[0]}",
            "sigma_max(K)": float(np.linalg.norm(w33 @ cx @ w34, 2)) if nx else 0.0,
            "leak_null34": float(np.linalg.norm(cx @ p34, "fro")) / nx if nx else 0.0,
            "lam_min": lam,
            "scale": sc,
            "lam_min_norm": lam / sc if sc > 0 else np.nan}


def _za_awr_mat_from_endf(path: Path):
    """ZA, AWR and MAT off the source file's MF34 MT=2 HEAD record."""
    from kika.endf.utils import parse_line, parse_endf_id

    with open(path) as fh:
        for line in fh:
            try:
                mat, mf, mt = parse_endf_id(line)
            except Exception:
                continue
            if mf == 34 and mt == 2:
                h = parse_line(line)
                return float(h["C1"]), float(h["C2"]), int(mat)
    raise SystemExit(f"no MF34 MT=2 HEAD record found in {path}")


def _compact_null_self_block(sub, mag_grid_ev) -> Tuple[int, int]:
    """Rewrite MF34's (0,0) null block as one interval. ``(before, after)`` values.

    The magnitude self-covariance belongs to MF33 and must not be repeated in
    MF34 (manual Sec. 34.3), so the (0,0) sub-subsection has to exist and has to
    be null. It does not have to be null on every bin: an LB=5 record over a
    single interval spanning the same range asserts exactly the same zero.

    Returns (0, 0) when there is no (0,0) block, so a section without a_0 is
    untouched and the frozen pipeline stays bit-identical.
    """
    from kika.endf.writers.mf34_writer import _make_lb5_record

    for ss in sub.sub_subsections:
        if int(ss.l or 0) != 0 or int(ss.l1 or 0) != 0:
            continue
        before = sum(len(r.energies) + len(r.matrix) for r in ss.records)
        edges = [float(mag_grid_ev[0]), float(mag_grid_ev[-1])]
        ss.records = [_make_lb5_record(np.zeros((1, 1)), edges)]
        after = sum(len(r.energies) + len(r.matrix) for r in ss.records)
        return before, after
    return 0, 0


def write_consistent_mf34(out_path: Path, source_endf: Path, c34_post, cx_post,
                          a_nom_group, shape_ev, mag_ev, c34_rel_ship,
                          null_fill="zero",
                          cross_emin_ev=None, cross_emax_ev=None,
                          mf33_file_grid=None,
                          order_grids_ev=None, base_valid_width=None):
    """Write the _mg tape whose MF34 is the joint that was just diagnosed.

    `cx_post` is Cauchy-Schwarz-compatible with `c34_post`, not with the MF34
    already in the file, so shipping the cross term means rewriting MF34.

    Everything goes out relative: of MF34's forms (LB = 0, 1, 2, 5, 6) only LB=0
    is absolute and it carries no off-diagonal structure. The (L=0, L1) blocks
    are the format's own mechanism for magnitude-shape covariance, with the (0,0)
    self-block null because that covariance belongs to MF33.

    Parameters
    ----------
    c34_rel_ship : np.ndarray
        The shipped RELATIVE MF34 on the base shape grid, used only where a_l is
        exactly zero and the absolute round trip is 0/0. Pass the same array the
        diagnosis used, never a re-mapping.
    """
    from kika.endf.writers.mf34_writer import (
        create_mf34_from_covariance, write_mf34_to_file,
    )

    n_gs = len(shape_ev) - 1
    a_flat = a_nom_group.reshape(-1)

    live = np.abs(a_flat) >= np.finfo(float).tiny
    both_live = np.outer(live, live)

    def _check_null_block(m, mask, name, extra=""):
        scale = float(np.abs(m).max())
        atol = 1e-15 * scale
        bad = (np.abs(m) > atol) & mask
        if not bad.any():
            return
        idx = np.argwhere(bad)
        i, j = idx[0]
        raise SystemExit(
            f"{int(bad.sum())} entries of {name} exceed {atol:.3g} where a_l is "
            f"exactly zero (worst |{name}| = {np.abs(m[bad]).max():.6g}, first "
            f"at [{i}, {j}]); the relative form is INFINITE there, not "
            f"undefined. The absolute covariance and the aggregator's validity "
            f"mask disagree, i.e. the collapse and the nominal group means were "
            f"not built from the same mask.{extra}"
        )

    _check_null_block(c34_post, ~both_live, "c34_post")
    _check_null_block(cx_post, ~live[None, :], "cx_post",
                      extra=" Same inconsistency, on the cross block.")

    c34_rel = assemble_c34_rel(c34_post, a_flat, c34_rel_ship, null_fill)
    cx_rel = np.divide(cx_post, a_flat[None, :], out=np.zeros_like(cx_post),
                       where=live[None, :])
    n_null = int((~live).sum())
    print(f"\n  null parameters: {n_null} of {a_flat.size} group-mean a_l are "
          f"exactly zero  [--null-fill {null_fill}]")
    if n_null:
        kept = np.abs(c34_rel[~both_live]).max()
        print(f"    shape blocks there: "
              + ("shipped relative values PRESERVED"
                 if null_fill == "ship" else "written ZERO")
              + f" (max |rel| {kept:.4g})")
        print(f"    cross blocks there: written ZERO")

    for name, arr in (("c34_rel", c34_rel), ("cx_rel", cx_rel)):
        if not np.all(np.isfinite(arr)):
            raise SystemExit(f"{name} is not finite after the relative conversion")
    print(f"\n  relative conversion: min |a_l group mean| = {np.abs(a_flat).min():.3e}")
    print(f"    max |c34_rel| = {np.abs(c34_rel).max():.4g}   "
          f"max |cx_rel| = {np.abs(cx_rel).max():.4g}")

    n_gm = len(mag_ev) - 1
    lo = mag_ev[0] if cross_emin_ev is None else cross_emin_ev
    hi = mag_ev[-1] if cross_emax_ev is None else cross_emax_ev
    g0 = int(np.searchsorted(mag_ev, lo, side="left"))
    g1 = int(np.searchsorted(mag_ev, hi, side="right")) - 1
    if g1 <= g0:
        raise SystemExit(f"cross window [{lo}, {hi}] eV selects no whole "
                         f"magnitude group")
    cross_grid = mag_ev[g0:g1 + 1]
    cx_win = cx_rel.reshape(n_gm, n_gs, L_MAX)[g0:g1]
    tag = "FULL magnitude grid" if (g1 - g0) == n_gm else "RESTRICTED"
    print(f"    cross rows: {tag} — groups {g0}:{g1} ({g1 - g0} of {n_gm}), "
          f"{cross_grid[0] / 1e6:.4g}-{cross_grid[-1] / 1e6:.4g} MeV")

    if mf33_file_grid is not None:
        g_file = np.asarray(mf33_file_grid, float)
        i0 = int(np.searchsorted(g_file, cross_grid[0]))
        seg = g_file[i0:i0 + cross_grid.size]
        if seg.shape != cross_grid.shape or not np.allclose(
                seg, cross_grid, rtol=1e-9, atol=0.0):
            raise SystemExit(
                f"the cross magnitude grid ({cross_grid.size} edges, "
                f"{cross_grid[0]:.7g}-{cross_grid[-1]:.7g} eV) is not a "
                f"contiguous sub-sequence of the file's MF33 grid "
                f"({g_file.size} edges). It has to be, or the a_0 blocks and "
                f"the MF33 self block describe different parameters and the "
                f"fold is not a congruence."
            )
        padded = np.zeros((g_file.size - 1, n_gs, L_MAX), dtype=float)
        padded[i0:i0 + cx_win.shape[0]] = cx_win
        print(f"    cross rows embedded in the file's MF33 grid: "
              f"{cx_win.shape[0]} live rows at [{i0}, {i0 + cx_win.shape[0]}) "
              f"of {g_file.size - 1}; the rest written zero")
        cx_win = padded
        cross_grid = g_file

    cross_cov = {l1: cx_win[:, :, l1 - 1] for l1 in range(1, L_MAX + 1)}
    shape_arg = np.asarray(shape_ev, float)

    # ── a mesh per Legendre order, as one congruence at the point of emission ──
    #
    # Shape and cross are collapsed HERE, together, with the SAME U. Handing
    # them meshes separately is the failure this whole step exists to avoid: a
    # cross block whose columns sit on a different mesh than the shape block it
    # correlates with stays PSD, passes every marginal gate, and is wrong, with
    # nothing downstream able to detect it.
    if order_grids_ev is not None:
        if null_fill != "zero":
            raise SystemExit(
                f"--null-fill {null_fill!r} cannot be combined with a mesh per "
                f"order: it writes the file's own relative values where a_l is "
                f"exactly zero, and those slots are not what U was built to "
                f"average. Use --null-fill zero.")
        # The base grid carries groups the DP never saw, and the DP's means are
        # not these means -- see `_fixed_cut_points`. Refine here, before
        # anything is built from the grids, so U, the blocks and what gets
        # written all use the same one.
        _n_before = {l: len(np.asarray(order_grids_ev[l], float)) - 1
                     for l in range(1, L_MAX + 1)}
        order_grids_ev = {
            l: _fixed_cut_points(
                order_grids_ev[l], shape_ev,
                np.asarray(base_valid_width[:, l - 1], float)
                * np.asarray(a_nom_group[:, l - 1], float))
            for l in range(1, L_MAX + 1)}
        n_ord = {l: len(np.asarray(order_grids_ev[l], float)) - 1
                 for l in range(1, L_MAX + 1)}
        _added = {l: n_ord[l] - _n_before[l] for l in n_ord}
        if any(_added.values()):
            print("    cortes que fuerzan los centrales (muertos + signo): "
                  + "  ".join(f"a_{l} +{_added[l]}" for l in range(1, L_MAX + 1)))
        U = {}
        for l in range(1, L_MAX + 1):
            U[l] = order_emission_weights(shape_ev, order_grids_ev[l],
                                          base_valid_width[:, l - 1],
                                          a_nom_group[:, l - 1])
            # A coarse group must be entirely live or entirely dead.
            # `_fixed_cut_points` above makes that true by construction, so
            # this is now an assertion on that refinement rather than a check on
            # the mesh. It is kept because the failure it describes is silent:
            # the merged entry would carry a central value the file does not
            # declare, and every downstream gate would still pass.
            dead = np.abs(a_nom_group[:, l - 1]) < np.finfo(float).tiny
            g_of = fine_to_group(np.asarray(shape_ev, float),
                                 np.asarray(order_grids_ev[l], float))
            n_dead = np.bincount(g_of, weights=dead.astype(float),
                                 minlength=n_ord[l])
            n_tot = np.bincount(g_of, minlength=n_ord[l])
            mixed = int(((n_dead > 0) & (n_dead < n_tot)).sum())
            if mixed:
                raise SystemExit(
                    f"a_{l}: {mixed} coarse groups merge an absent group with a "
                    f"live one; the mesh was not built with absent groups as cut "
                    f"points, so the merged entry would carry a fabricated "
                    f"central value.")
        # ── how much room the mesh leaves the relative form to explode ────────
        #
        # rel_S = u^T R u / (sum u)^2 with u = w * a, so
        #
        #     |rel_S| <= max|R| * (sum|u| / |sum u|)^2
        #
        # and the whole budget is the SIGN CANCELLATION of u inside each coarse
        # group. It depends on the mesh and the central values only -- not on the
        # covariance -- so it can be read off here, before a single block is
        # built, and it says which merge is responsible rather than leaving the
        # guard below to guess.
        amp = {}
        for l in range(1, L_MAX + 1):
            u = (np.asarray(base_valid_width[:, l - 1], float)
                 * np.asarray(a_nom_group[:, l - 1], float))
            g_of = fine_to_group(np.asarray(shape_ev, float),
                                 np.asarray(order_grids_ev[l], float))
            s_abs = np.bincount(g_of, weights=np.abs(u), minlength=n_ord[l])
            s_net = np.abs(np.bincount(g_of, weights=u, minlength=n_ord[l]))
            live = s_abs > 0
            r = np.ones(n_ord[l])
            r[live] = (s_abs[live] / np.maximum(s_net[live],
                                                np.finfo(float).tiny)) ** 2
            amp[l] = r
        _worst_l = max(amp, key=lambda l: amp[l].max())
        _worst_g = int(np.argmax(amp[_worst_l]))
        print("    cancelacion en u = w*a, (sum|u|/|sum u|)^2 por orden: "
              + "  ".join(f"a_{l} {amp[l].max():.3g}" for l in range(1, L_MAX + 1)))
        _ge = np.asarray(order_grids_ev[_worst_l], float)
        print(f"    peor: a_{_worst_l} grupo {_worst_g} "
              f"[{_ge[_worst_g] / 1e6:.4f}, {_ge[_worst_g + 1] / 1e6:.4f}] MeV, "
              f"factor {amp[_worst_l].max():.4g} sobre max|rel| base")

        a_ord = {l: U[l] @ a_nom_group[:, l - 1] for l in range(1, L_MAX + 1)}

        rows_of = {l: np.arange(n_gs) * L_MAX + (l - 1) for l in range(1, L_MAX + 1)}
        blocks_rel = {}
        for l in range(1, L_MAX + 1):
            for l1 in range(l, L_MAX + 1):
                blocks_rel[(l, l1)] = (
                    U[l] @ c34_rel[np.ix_(rows_of[l], rows_of[l1])] @ U[l1].T)
        cross_cov = {l1: cross_cov[l1] @ U[l1].T for l1 in range(1, L_MAX + 1)}

        before = float(np.abs(c34_rel).max())
        worst = max(float(np.abs(b).max()) for b in blocks_rel.values())
        print(f"\n  per-order mesh: groups "
              + "/".join(str(n_ord[l]) for l in range(1, L_MAX + 1))
              + f" of {n_gs}")
        print(f"    MF34 entries {L_MAX * L_MAX * n_gs * n_gs:,d} -> "
              f"{sum(n_ord[a] * n_ord[b] for a in n_ord for b in n_ord):,d} "
              f"({100 * (sum(n_ord[a] * n_ord[b] for a in n_ord for b in n_ord) / (L_MAX * L_MAX * n_gs * n_gs) - 1):+.1f} %)")
        # ⚑ THE MESH IMPROVES THE RELATIVE FORM, IT DOES NOT STRAIN IT, and the
        # right guard is a comparison rather than an absolute bar. The SNR >= 1
        # criterion is exactly "the group mean is not small against its sigma",
        # so merging removes the near-cancelling groups where rel explodes:
        # measured on the mixture object, max |rel| falls from 1.35e7 to 5.5 at
        # a_3 and from 3.86e7 to 6.4 at a_6. A fixed threshold would have
        # aborted a run whose object is thousands of times better behaved than
        # the shared-mesh one that ships today.
        print(f"    max |c34_rel|: {before:.4g} -> {worst:.4g} "
              f"({worst / max(before, 1e-300):.2e}x)")
        if not np.isfinite(worst):
            raise SystemExit(
                "the per-order collapse produced non-finite relative entries: a "
                "merged group's central value cancelled to exactly zero.")
        if worst > 10.0 * max(before, np.finfo(float).tiny):
            _pred = before * amp[_worst_l].max()
            raise SystemExit(
                f"the per-order collapse made the relative form {worst / before:.1f}x "
                f"WORSE (max |rel| {before:.4g} -> {worst:.4g}).\n"
                f"    La cota de cancelacion predice {_pred:.4g} "
                f"(= {before:.4g} x {amp[_worst_l].max():.4g}), asi que "
                f"{'ES' if worst <= 1.5 * _pred else 'NO ES'} eso: "
                f"{'algun grupo grueso fusiona valores centrales que se cancelan' if worst <= 1.5 * _pred else 'la explosion viene de la covarianza, no de la malla'}.\n"
                f"    rel_S = 1/SNR_S^2 exactamente, asi que un grupo con rel > 1 "
                f"es un grupo que el DP habria rechazado si lo hubiera evaluado "
                f"sobre ESTA rejilla y ESTAS medias.")
        for l in range(1, L_MAX + 1):
            live_b = np.abs(a_nom_group[:, l - 1]) >= np.finfo(float).tiny
            live_c = np.abs(a_ord[l]) >= np.finfo(float).tiny
            print(f"    a_{l}: {int(live_b.sum())} live of {n_gs} -> "
                  f"{int(live_c.sum())} live of {n_ord[l]}")
        c34_rel = blocks_rel
        shape_arg = {l: np.asarray(order_grids_ev[l], float)
                     for l in range(1, L_MAX + 1)}

    za, awr, mat = _za_awr_mat_from_endf(source_endf)
    mf34 = create_mf34_from_covariance(
        c34_rel, shape_arg, L_MAX, za, awr, mat, 2,
        ltt=1, cross_cov=cross_cov, cross_energy_grid_ev=cross_grid,
    )
    sub = mf34._subsections[0]

    n_before, n_after = _compact_null_self_block(sub, cross_grid)
    if n_before:
        print(f"    (0,0) null block: {n_before} -> {n_after} values "
              f"(~{(n_before - n_after) * 11 / 6 * 81 / 80 / 1e6:.1f} MB of "
              f"zeros not written)")

    print(f"    MF34: LTT={mf34._ltt}, NL={sub.nl}, "
          f"{len(sub.sub_subsections)} sub-subsections "
          f"(NL(NL+1)/2 = {sub.nl * (sub.nl + 1) // 2})")

    n_expect = L_MAX + 1
    problems = []
    if int(mf34._ltt) != 3:
        problems.append(f"LTT={mf34._ltt}, expected 3 (manual Sec. 34.2: "
                        f"'LTT=3 if either L or L1=0 anywhere in the Section')")
    if int(sub.nl) != n_expect or int(sub.nl1) != n_expect:
        problems.append(f"NL/NL1={sub.nl}/{sub.nl1}, expected {n_expect} "
                        f"(NL is the NUMBER of coefficients a_0..a_{L_MAX}, "
                        f"not the highest index)")
    if len(sub.sub_subsections) != n_expect * (n_expect + 1) // 2:
        problems.append(f"{len(sub.sub_subsections)} sub-subsections, expected "
                        f"NSS = NL(NL+1)/2 = {n_expect * (n_expect + 1) // 2}")
    if problems:
        raise SystemExit(
            "the installed kika writes a non-conforming MF34:\n  - "
            + "\n  - ".join(problems)
            + "\n\nThis script's fixes are in kika/endf/{writers/mf34_writer,"
              "parsers/parse_mf34}.py, which is a LIBRARY change: it does not "
              "reach the cluster by copying scripts/. Update the installed "
              "kika, then re-run. Nothing has been written."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mf34_to_file(str(source_endf), mf34, str(out_path))
    print(f"  wrote {out_path} ({out_path.stat().st_size} bytes; "
          f"source {source_endf.stat().st_size})")


def build_cross_and_write_endf(run_dir, source_endf, out_endf, *,
                               mag_grid="fine", null_fill="zero", cache=None,
                               dead_parameters="drop"):
    """Build the joint and write the cross-term ENDF. Library entry point.

    Goes through ``main`` on a constructed argv so the pipeline runs the exact
    command that produced the shipped run-91 tape, byte for byte.
    """
    argv = ["--run-dir", str(run_dir),
            "--source-endf", str(source_endf),
            "--write-endf", str(out_endf),
            "--mag-grid", str(mag_grid),
            "--null-fill", str(null_fill),
            "--dead-parameters", str(dead_parameters)]
    if cache is not None:
        argv += ["--cache", str(cache)]
    rc = main(argv)
    if rc != 0:
        raise RuntimeError(f"build_group_cross failed (rc={rc})")
    return Path(out_endf)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="Pipeline run directory: supplies the replicas, the grids and nominal_fits")
    ap.add_argument("--cache", default=None,
                    help="Parse cache for the source ENDF (default: <run-dir>/.group_cross_cache)")
    ap.add_argument(
        "--mag-grid", choices=["group", "fine"], default="group",
        help="Energy axis for the magnitude family. 'fine': the analysis mesh, "
             "which is what the shipped file carries and the only choice that "
             "makes the fold a congruence (A_mag = I, so the joint's c33 block "
             "IS the shipped MF33). 'group': the adaptive MF33 grid, where "
             "corr(A C A^T) != A corr(C) A^T and the certified joint is not the "
             "matrix the chi2 folds.",
    )
    ap.add_argument(
        "--null-fill", choices=["zero", "ship"], default="zero",
        help="What the rewritten MF34 carries where a_l is exactly zero (~37 %% "
             "of slots). 'zero': dead parameter, no covariance, PSD-safe. "
             "'ship': keep the file's declared value -- this reproduced the "
             "run-89 failure.",
    )
    ap.add_argument(
        "--dead-parameters", choices=["drop", "carry"], default="drop",
        help="What to do with the (group, order) slots whose REPLICAS never "
             "move -- 1542 of 4218 on run 99, almost all of a_4..a_6. 'drop' "
             "(what every run so far did): the nominal group mean is averaged "
             "over the MC-valid bins only, comes out exactly zero there, and "
             "the tape declares the parameter with NO uncertainty. 'carry': "
             "the group mean is averaged over ALL fine bins -- the functional "
             "the consumer interpolates back out of MF4 -- and the shape block "
             "on those slots is taken from the pipeline's own MF34 as a DIRECT "
             "SUMMAND. The cross block stays zero there, because the MC cannot "
             "inform it.",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="Report the PSD diagnostics and write nothing")
    mode.add_argument("--write", action="store_true",
                      help="Write the .npy cross-block sidecars")
    mode.add_argument(
        "--write-endf", metavar="OUT",
        help="Write a _mg.endf whose MF34 is the consistent joint: shape blocks "
             "from c34_post, (L=0, L1) cross blocks from cx_post. MF3/MF4/MF33 "
             "are copied from --source-endf untouched.",
    )
    ap.add_argument("--source-endf", default=None,
                    help="Template for --write-endf; defaults to the run dir's *_mg.endf")
    ap.add_argument(
        "--c33-from-file", action="store_true",
        help="Also diagnose against the MF33 read from the source ENDF -- the "
             "one the chi2 folds -- not only the adaptive sidecar the joint is "
             "certified against. Costs one extra ENDF parse.",
    )
    ap.add_argument(
        "--tol-sweep", action="store_true",
        help="With --c33-from-file: print the eigenvalue spectra and re-diagnose "
             "over NULL_TOL = 1e-10 .. 1e-4. lam_min must be FLAT across the "
             "sweep; sigma_max(K) and leak_null34 will not be.",
    )
    ap.add_argument(
        "--c33-blend-ref", metavar="ENDF",
        help="With --c33-from-file: a second tape whose MF33 is the s = 0 end of "
             "a blend C33(s) = (1-s)*ref + s*source. Sweeps row F' over s and "
             "prints where the joint stops being compatible with the cross term. "
             "Both MF33s must sit on the same grid and carry the same diagonal.",
    )
    ap.add_argument(
        "--c33-blend-s", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated blend fractions for --c33-blend-ref.",
    )
    ap.add_argument(
        "--cx-j33-exp", metavar="T1,T2,...", default=None,
        help="With --c33-from-file: sweep the exponent t in "
             "cx(t) = cx_mc * outer(j33**t, j34) and re-run row F' at each t. "
             "t = 1 is what ships today (the cross block scaled to the DECLARED "
             "MF33 diagonal); t = 0 leaves the magnitude side at the Pass-1 "
             "scale the split-combine's C1' is written in. Give "
             "--c33-blend-ref as well to get the unsplit anchor column. Writes "
             "no tape and scores no chi2 -- it is a PSD table (roadmap Sec. 10.3).",
    )
    ap.add_argument(
        "--split-forensics", metavar="C_SPLIT_ABS.npy", default=None,
        help="With --c33-from-file: the P1.2 split-combine product "
             "(mf33_absolute_covariance.npy, ABSOLUTE units, on the run's fine "
             "grid). Answers whether the split/cross incompatibility is PHYSICAL "
             "or a REPRESENTATION artefact: (1) is C1' = c33_mc, (2) is the joint "
             "PSD on the MC's OWN grid with no file and no m_of, (3) where the "
             "Cauchy-Schwarz load lives by eigen-direction. Roadmap Sec. 10.6. "
             "Writes no tape and scores no chi2.",
    )
    ap.add_argument(
        "--write-null-mask", metavar="OUT.npz",
        help="Emit the (n_shape_groups, L_MAX) mask of slots the MC never "
             "populated, with the shape grid and a_nom_group. Works in --check.",
    )
    ap.add_argument("--cross-emin-ev", type=float, default=None,
                    help="Clip the cross block's magnitude axis (default: the whole grid)")
    ap.add_argument("--cross-emax-ev", type=float, default=None)
    ap.add_argument(
        "--per-order-mesh", choices=["auto", "on", "off"], default="auto",
        help="Emit each Legendre order on its own mesh. 'auto': on exactly when "
             "the run wrote mf34_per_order_mesh.npz, so a run that did not "
             "choose a mesh is untouched byte for byte. 'on' also accepts the "
             "tape's own diagonal grids, for a foreign file.",
    )
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    cache = Path(args.cache) if args.cache else run_dir / ".group_cross_cache"

    fine_ev = np.load(run_dir / "mf33_energy_grid_ev.npy").astype(float)
    n_fine = len(fine_ev) - 1
    widths = np.diff(fine_ev) / 1e6

    _fine_mag = args.mag_grid == "fine"
    if _fine_mag:
        mag_ev = fine_ev
        c33_ship = np.load(run_dir / "mf33_relative_covariance.npy")
        if args.null_fill != "zero":
            raise SystemExit(
                "--mag-grid fine requires --null-fill zero. 'ship' keeps the "
                "file's declared relative variance at the ~1542 slots where "
                "a_l is exactly zero, which injects the old construction into "
                "the softest directions of the new one and reproduced run 89 "
                "almost exactly (§L11). There is nothing to preserve there: a "
                "parameter with no variance has no covariance either."
            )
        if args.cross_emin_ev is not None or args.cross_emax_ev is not None:
            raise SystemExit(
                "--cross-emin-ev/--cross-emax-ev are refused with --mag-grid "
                "fine. Zero-PADDING the magnitude axis out to the file's MF33 "
                "grid is safe -- it adds separable parameters with no variance "
                "and no covariance. WINDOWING inside our own block is not: it "
                "zeroes rows of a PSD matrix's off-diagonal while leaving the "
                "marginals, which is not PSD-preserving and is precisely what "
                "`check_endf_roundtrip_psd.py`'s header warns about."
            )
    else:
        mag_ev = np.load(run_dir / "mf33_multigroup_grid_ev.npy").astype(float)
        c33_ship = np.load(run_dir / "mf33_multigroup_relative_covariance.npy")

    source_endf = (Path(args.source_endf) if args.source_endf
                   else shipped_mg_endf(run_dir))
    print(f"shipped MF34 source: {source_endf}")

    blocks = mf34_group_edges_ev(source_endf, cache)
    shape_ev = base_shape_grid(blocks)
    n_gm, n_gs = len(mag_ev) - 1, len(shape_ev) - 1
    print(f"fine {n_fine} bins  ->  MF33 magnitude {n_gm} groups, "
          f"MF34 shape {n_gs} groups ({n_fine / n_gs:.2f} fine bins per shape group)")

    nom = pd.read_parquet(run_dir / "nominal_fits.parquet",
                          columns=["energy_index"] + [f"c_{l}" for l in range(1, L_MAX + 1)])
    a_nom_fine = nom.sort_values("energy_index")[
        [f"c_{l}" for l in range(1, L_MAX + 1)]].to_numpy(float)

    g_mag = fine_to_group(fine_ev, mag_ev)
    g_shape = fine_to_group(fine_ev, shape_ev)

    print("  reading Pass-1 c0 ...", flush=True)
    c0, ids = load_pass1_c0(run_dir, n_fine)

    print("  mask + collapse of the a_l replicas ...", flush=True)
    valid, A_shape, a_g = valid_and_collapsed(run_dir, n_fine, ids, g_shape, widths, n_gs)
    print("  valid fine bins per order: "
          + "  ".join(f"a_{l+1} {int(valid[:, l].sum())}" for l in range(L_MAX)))

    A_mag = row_aggregator(g_mag, widths, n_gm, np.ones(n_fine, bool))
    c0_g = c0 @ A_mag.T

    # ── the denominator has to be the number the CONSUMER multiplies back ─────
    #
    # `A_shape[l]` is masked by the MC's validity, so for a group whose replicas
    # never move at order l it is an all-zero row and the group mean comes out
    # EXACTLY ZERO -- 1542 of 4218 slots on run 99. The tape then declares
    # `rel = 0` there, while `eval_covariance.build_mf34_block` scales that
    # relative block by `a_l_per_pt` INTERPOLATED FROM THE FILE'S OWN MF4, which
    # is not zero: the file asserts perfect knowledge of a parameter it does
    # declare. The run's own log already prints the contradiction in one line --
    # `a_pt` has 1 exact zero of 4218 against `a_nom_group`'s 1542.
    #
    # `carry` averages over every fine bin instead, which is the same functional
    # the consumer interpolates. `drop` keeps the masked aggregator, so the
    # default reproduces every earlier run byte for byte.
    _carry = args.dead_parameters == "carry"
    A_nom = ([row_aggregator(g_shape, widths, n_gs, np.ones(n_fine, bool))]
             * L_MAX if _carry else A_shape)
    a_nom_group = np.stack(
        [A_nom[l] @ a_nom_fine[:, l] for l in range(L_MAX)], axis=-1)
    if _carry:
        _tiny = np.finfo(float).tiny
        _was = np.stack([A_shape[l] @ a_nom_fine[:, l] for l in range(L_MAX)],
                        axis=-1)
        print(f"  --dead-parameters carry: group means over ALL fine bins  "
              f"(exact zeros {int((np.abs(_was) < _tiny).sum())} -> "
              f"{int((np.abs(a_nom_group) < _tiny).sum())} of {_was.size})")

    if args.write_null_mask:
        mask = np.abs(a_nom_group) < np.finfo(float).tiny
        outm = Path(args.write_null_mask)
        np.savez(outm, null_mask=mask, shape_grid_ev=shape_ev,
                 a_nom_group=a_nom_group)
        print(f"\n  wrote {outm.name}: {int(mask.sum())} of {mask.size} "
              f"(group, order) slots unsupported")
        print("    by order: " + "  ".join(
            f"a_{l+1} {int(mask[:, l].sum())}/{mask.shape[0]}"
            for l in range(L_MAX)))

    cen = 0.5 * (shape_ev[:-1] + shape_ev[1:])
    bin_at = np.clip(np.searchsorted(fine_ev, cen, side="right") - 1,
                     0, n_fine - 1)
    a_pt = a_nom_fine[bin_at].reshape(-1)
    zero_pt = np.abs(a_pt) < np.finfo(float).tiny
    print(f"  consumer-convention group coefficients a_pt: "
          f"{int(zero_pt.sum())} exact zeros of {a_pt.size}")
    if zero_pt.any():
        print(f"    excluded from the second table; the consumer annihilates "
              f"them as well, so this is exact, not a truncation")
    keep_pt = ~zero_pt

    keep = np.isfinite(c0_g).all(1) & np.isfinite(a_g).all((1, 2))
    c0_g, a_g = c0_g[keep], a_g[keep]
    n = int(keep.sum())
    a_flat = a_g.reshape(n, n_gs * L_MAX)
    print(f"  replicas: {n}")

    joint = np.cov(np.hstack([c0_g, a_flat]), rowvar=False)
    c33_mc = joint[:n_gm, :n_gm]
    c34_mc = joint[n_gm:, n_gm:]
    cx_mc = joint[:n_gm, n_gm:]

    # The replica arrays die here: `joint` is their only consumer, and the
    # three blocks above are VIEWS into it, so `joint` is what has to stay.
    # On the fine magnitude grid they are ~950 MB that would otherwise sit
    # resident under the whole diagnostics table, where each row builds its
    # own 5956 x 5956 joint. Dropping them is the difference between the table
    # printing and the box swapping.
    del c0, c0_g, a_g, a_flat, nom
    gc.collect()
    # `gc.collect()` returns the pages to glibc's arenas, NOT to the kernel, and
    # the arenas are badly fragmented by here: the two parquet passes allocate
    # and free hundreds of MB in 250-bin chunks. Measured on run 99, the process
    # sat ~2 GB above the sum of its live arrays until this call. That is the
    # difference between the diagnostics table fitting on a 12 GB box and not.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # not glibc; the free lists just stay resident

    print("  mapping the shipped MF34 onto the base shape grid ...", flush=True)
    c34_rel_ship = shipped_c34_rel_on_base(blocks, shape_ev)
    a_flat_nom = a_nom_group.reshape(-1)
    c34_ship = c34_rel_ship * np.outer(a_flat_nom, a_flat_nom)
    c34_ship = 0.5 * (c34_ship + c34_ship.T)

    d_tar = np.concatenate([np.sqrt(np.maximum(np.diag(c33_ship), 0.0)),
                            np.sqrt(np.maximum(np.diag(c34_ship), 0.0))])
    d_mc = np.sqrt(np.maximum(np.diag(joint), 0.0))
    jj = np.divide(d_tar, d_mc, out=np.zeros_like(d_tar), where=d_mc > 0)
    j33, j34 = jj[:n_gm], jj[n_gm:]
    c33_post = c33_mc * np.outer(j33, j33)
    c34_post = c34_mc * np.outer(j34, j34)
    cx_post = cx_mc * np.outer(j33, j34)

    # ── the parameters the congruence cannot reach ────────────────────────────
    #
    # `jj = d_tar / d_mc` is a diagonal congruence, and a congruence cannot give
    # variance to a direction the MC never explored: where `d_mc = 0` the whole
    # row and column go to zero and take the pipeline's own sigma with them.
    # That is not a modelling decision, it is the rescaling deleting what it
    # cannot rescale.
    #
    # `carry` puts those slots back as a DIRECT SUMMAND out of the pipeline's
    # own MF34. Direct sum, not a patch of the full block, and the reason is
    # that it keeps the diagnosis honest:
    #   * PSD becomes min(lam_min(congruence), lam_min(c34_ship[D, D])), and a
    #     principal submatrix of a PSD matrix is PSD -- so the gate turns into a
    #     real measurement of the pipeline object instead of a tautology about
    #     the chain being a congruence;
    #   * row F must still land on the CONTROL row's sigma_max(K), because
    #     `cx_post` is exactly zero on D and the whitener of a block-diagonal
    #     matrix is block diagonal, so the augmented directions are invisible to
    #     ||W33 Cx W34||_2. The live block is still a positive diagonal
    #     congruence of the CONTROL -- the denominator moves with `carry` on the
    #     MIXED groups, but a congruence is a congruence. If row F pulls away
    #     from CONTROL, the chain is not one and the tape must not ship.
    # NOTE `lam_min_norm` IS NOT comparable across drop/carry: the normaliser is
    # max|diag(J)| and `carry` restores relative variances up to ~7.6 at slots
    # `drop` wrote as zero. Read sigma_max(K) and the raw lam_min.
    # The alternative -- taking the pipeline's corr(dead, live) as well -- is
    # priced as row G below rather than asserted.
    dead34 = np.zeros(n_gs * L_MAX, bool)
    if _carry:
        if (d_mc[:n_gm] <= 0.0).any():
            raise SystemExit(
                f"{int((d_mc[:n_gm] <= 0.0).sum())} magnitude groups have zero "
                f"MC spread. The c0 replicas always move, so that is a join or "
                f"a masking bug, not a dead parameter.")
        c34_post, dead34, _orphan, _pinfo = carry_dead_parameters(
            c34_post, c34_ship, d_mc[n_gm:], d_tar[n_gm:])
        _lam = (float(np.linalg.eigvalsh(c34_post[np.ix_(dead34, dead34)])[0])
                if dead34.any() else 0.0)
        _dg = dead34.reshape(n_gs, L_MAX)
        print(f"\n  carried back: {int(dead34.sum())} of {dead34.size} shape "
              f"parameters that the replicas never move and the pipeline does "
              f"declare\n    ({_orphan} declared by neither, left at zero)")
        print("    by order: " + "  ".join(
            f"a_{l + 1} {int(_dg[:, l].sum())}/{n_gs}" for l in range(L_MAX)))
        print(f"    the pipeline's MF34 on those slots: lam_min "
              f"{_pinfo['lam_min']:.6e}  lam_max {_pinfo['lam_max']:.6e}")
        print(f"    bar 1, conditioning: ratio {_pinfo['ratio']:.3e} vs "
              f"-{DEAD_BLOCK_PSD_RTOL:.0e}  -- using "
              f"{abs(_pinfo['ratio']) / DEAD_BLOCK_PSD_RTOL:.3f} of the tape's "
              f"own 7-digit floor")
        print(f"    bar 2, invented mass: {_pinfo['clip_mass']:.3e} vs "
              f"{DEAD_BLOCK_CLIP_MASS_MAX:.0e}  -- using "
              f"{_pinfo['clip_mass'] / DEAD_BLOCK_CLIP_MASS_MAX:.3f} of the bar "
              f"(trace {_pinfo['trace']:.6e})")
        print(f"    PSD projection: {_pinfo['n_clipped']} negative eigenvalues "
              f"clipped to zero, worst relative shift of a declared variance "
              f"{_pinfo['d_shift_rel']:.3e}")
        print(f"    lam_min of the carried block AFTER the projection: "
              f"{_lam:.6e}")

    e33 = np.max(np.abs(np.sqrt(np.maximum(np.diag(c33_post), 0)) - d_tar[:n_gm]))
    e34 = np.max(np.abs(np.sqrt(np.maximum(np.diag(c34_post), 0)) - d_tar[n_gm:]))
    print(f"\n  marginal-identity gate: max abs sigma error  "
          f"MF33 {e33:.3e}   MF34 {e34:.3e}")

    c33_matrix_gate(c33_post, c33_ship, fatal=_fine_mag)

    rows = [
        diagnose("CONTROL: all blocks from the collapsed replicas", c33_mc, c34_mc, cx_mc),
        diagnose("shipped marginals, Cx = 0", c33_ship, c34_ship, np.zeros_like(cx_post)),
        diagnose("CANDIDATE: rescaling after the collapse", c33_post, c34_post, cx_post),
        diagnose("LEGACY (run 89): shipped marginals + cx_post",
                 c33_ship, c34_ship, cx_post),
    ]
    print("\n=== PSD on the run's own adaptive grids (a_nom space) ===")
    print(pd.DataFrame(rows).set_index("case").to_string())
    print(f"\n  ||Cx||_F = {np.linalg.norm(cx_post):.4g}")
    print("\n  ⚠ EVERY ROW ABOVE IS TAKEN AGAINST a_nom, WHICH IS ZERO AT "
          f"{int((np.abs(a_flat_nom) < np.finfo(float).tiny).sum())} OF "
          f"{a_flat_nom.size} SLOTS.\n"
          "  That congruence is SINGULAR, so it annihilates the null directions "
          "and cannot\n  see anything living in them. Run 90 passed here and "
          "still died in the chi2.\n  The table below is the one that "
          "corresponds to what the chi2 folds.")

    K = keep_pt
    cxA = (cx_post / a_pt[None, :])[:, K]
    rel_post = assemble_c34_rel(c34_post, a_flat_nom, c34_rel_ship, "zero")
    rel_ship_fill = assemble_c34_rel(c34_post, a_flat_nom, c34_rel_ship, "ship")
    KK = np.ix_(K, K)
    rel_ship_K, rel_post_K = c34_rel_ship[KK], rel_post[KK]
    rel_shipfill_K = rel_ship_fill[KK]

    rows2 = [
        diagnose("A  shipped MF34, Cx = 0   (the run-86 baseline)",
                 c33_ship, rel_ship_K, np.zeros_like(cxA)),
        diagnose("B  shipped MF34 + Cx      (run 89)",
                 c33_ship, rel_ship_K, cxA),
        diagnose("C  rebuilt + null=ship    (RUN 90, AS SHIPPED)",
                 c33_ship, rel_shipfill_K, cxA),
        diagnose("D  rebuilt + null=zero    (the proposed fix)",
                 c33_ship, rel_post_K, cxA),
        diagnose("E  rebuilt + null=zero + c33_post (MF33 rebuilt too)",
                 c33_post, rel_post_K, cxA),
        diagnose("F  a_0 blocks: cx/a_nom, ONE convention (WHAT WE NOW SHIP)",
                 c33_ship, rel_post_K,
                 np.divide(cx_post, a_flat_nom[None, :],
                           out=np.zeros_like(cx_post),
                           where=(np.abs(a_flat_nom) >= np.finfo(float).tiny
                                  )[None, :])[:, K]),
    ]
    if _carry and dead34.any():
        # Not a direct sum: the pipeline's correlation between the dead
        # parameters and the live ones, kept. PSD is not guaranteed and
        # sigma_max(K) is no longer protected, which is the whole point of
        # measuring it next to F instead of choosing between them on argument.
        _cf = c34_post.copy()
        _m = dead34[None, :] | dead34[:, None]
        _cf[_m] = (0.5 * (c34_ship + c34_ship.T))[_m]
        _rel_full = assemble_c34_rel(_cf, a_flat_nom, c34_rel_ship, "zero")
        rows2.append(diagnose(
            "G  carry, FULL pipeline block on the dead rows (NOT a direct sum)",
            c33_ship, _rel_full[KK],
            np.divide(cx_post, a_flat_nom[None, :], out=np.zeros_like(cx_post),
                      where=(np.abs(a_flat_nom) >= np.finfo(float).tiny
                             )[None, :])[:, K]))

    print("\n=== PSD in the space the chi2 folds (relative MF34, a_pt) ===")
    print("  ⚠ ROWS A-E MODEL THE SIDECAR ROUTE AND ARE HISTORY. They divide "
          "the cross\n  block by `a_pt` and the shape blocks by `a_nom` -- two "
          "denominators for one\n  parameter, i.e. §L13 itself -- because when "
          "they were written the cross term\n  arrived absolute and there was "
          "no single convention. Their sigma_max(K) ~ 1000\n  measures that "
          "mismatch, NOT the file. **Read row F.**")
    print(pd.DataFrame(rows2).set_index("case").to_string())
    n_null_nom = int((np.abs(a_flat_nom) < np.finfo(float).tiny).sum())
    n_seen = int((np.abs(a_flat_nom) < np.finfo(float).tiny)[K].sum())
    print(f"\n  a_nom is zero at {n_null_nom} slots; {n_seen} of them survive "
          f"into this table,\n  i.e. the consumer does NOT annihilate them and "
          f"whatever the fill puts there is real.")
    print("\n  ⚠ COMPARE `sigma_max(K)`, NOT `lam_min_norm`. The normaliser is\n"
          "  max|diag(J)|, and rows C and D differ in exactly that diagonal --\n"
          "  the shipped null-slot relative variance, up to 7.6 -- so their\n"
          "  lam_min_norm are divided by different numbers and cannot be read\n"
          "  against each other. sigma_max(K) is free of it. Sec. 10.1.8-L12.")
    print("\n  How to read it:\n"
          "    A            -> the run-86 baseline, cross term off.\n"
          "    B, C, D, E   -> the SIDECAR route. They land on one number\n"
          "                    because they share one defect: the cross is\n"
          "                    divided by a_pt and the shape blocks by a_nom.\n"
          "                    Two denominators for one parameter, so no single\n"
          "                    M exists and Sigma_eval is not M J M^T. That is\n"
          "                    Sec. 10.1.8-L13, and it is why runs 87-90 died on\n"
          "                    potrf. The null fill (C vs D) is not the lever.\n"
          "                    D == E because on the FINE axis c33_post IS\n"
          "                    c33_ship -- see the c33 matrix gate above -- so\n"
          "                    rebuilding MF33 changes nothing; on the group\n"
          "                    axis it would.\n"
          "    F            -> the a_0 blocks, ONE denominator. The published\n"
          "                    joint is then a positive diagonal congruence of\n"
          "                    the CANDIDATE row above, and the fold is another,\n"
          "                    so sigma_max(K) must land on the CONTROL's\n"
          "                    0.999993 and lam_min_norm at machine zero. If it\n"
          "                    does not, something in the chain is not a\n"
          "                    congruence and the file must not ship.")

    if args.c33_from_file:
        from scripts.precompute_chi2_library_c0 import load_library_lib_c0

        print("\n=== re-certifying against the MF33 the chi2 folds ===")
        lib = load_library_lib_c0(str(source_endf), "This work")
        g33 = np.asarray(lib.get("mf33_grid_ev"), float)
        c33_file = np.asarray(lib.get("mf33_rel_cov"), float)
        if c33_file.ndim != 2 or g33.size != c33_file.shape[0] + 1:
            raise SystemExit(
                f"MF33 from the file is {c33_file.shape} against {g33.size} "
                f"edges; cannot re-certify.")
        print(f"  file MF33: {c33_file.shape[0]} bins   "
              f"sidecar magnitude axis: {n_gm} groups")

        cen33 = 0.5 * (g33[:-1] + g33[1:])
        m_of = np.searchsorted(mag_ev, cen33, side="right") - 1
        inside = (m_of >= 0) & (m_of < n_gm)
        cx_rel_full = cx_post / a_pt[None, :]
        cx_file = np.zeros((c33_file.shape[0], cx_post.shape[1]))
        cx_file[inside] = cx_rel_full[m_of[inside]]
        print(f"  file bins inside the sidecar's span: "
              f"{int(inside.sum())} of {inside.size}; the rest carry no cross "
              f"covariance and are written zero")

        # ⚑ F' — the a_0 convention against the FILE's MF33, added 2026-08-14.
        #
        # WHY IT DID NOT EXIST. Rows A'-C' were written when the cross term still
        # arrived through the sidecar, so all three divide it by `a_pt` and are
        # about a route that is retired. Row F below divides by `a_nom` -- the
        # one convention we actually ship -- but it is taken against `c33_ship`,
        # which is `run_dir/mf33_relative_covariance.npy`, i.e. the MF33 THE RUN
        # PRODUCED. When --source-endf carries a DIFFERENT MF33 (the P1.2
        # split-combine tape: same diagonal, coh 0.900 -> 0.391, PR 1.78 -> 8.06)
        # nothing in this script tested the pair that would actually be folded.
        # A' cannot stand in for it: at Cx = 0 the binding eigenvalue comes from
        # the MF34 block, so A' reproduces A to the digit and is blind to MF33.
        #
        # WHAT IT PRICES. §10.1: the cross block is Cauchy-Schwarz-compatible
        # only with the marginals it was built from. The split preserves the
        # DIAGONAL byte-for-byte, so `jj = d_tar / d_mc` and hence `cx_post` are
        # unchanged -- the cross block itself is not the risk. The risk is the
        # joint: the split SHRINKS the coherent variance (sigma_coh 0.0554 ->
        # 0.0241 for split-loc, the shipped variant -- 0.0225 is split-diag, the
        # bound) that the cross block leans on, while Cx stays put. F' is the
        # only row that sees that.
        #
        # ⚑ 2026-08-14, AFTER F' MEASURED 84.72: read the note above with §10.
        # "the cross block itself is not the risk" is true of its DIAGONAL and
        # false of its SCALE -- `jj = d_tar/d_mc` inflates it by exactly the
        # ratio the split removes. --cx-j33-exp prices that.
        cx_a0_full = np.divide(
            cx_post, a_flat_nom[None, :], out=np.zeros_like(cx_post),
            where=(np.abs(a_flat_nom) >= np.finfo(float).tiny)[None, :])
        cx_a0_file = np.zeros((c33_file.shape[0], cx_post.shape[1]))
        cx_a0_file[inside] = cx_a0_full[m_of[inside]]

        rows3 = [
            diagnose("A' file MF33, Cx = 0   (the real run-86 baseline)",
                     c33_file, rel_ship_K, np.zeros_like(cx_file[:, K])),
            diagnose("B' file MF33 + Cx      (what run 89 really folded)",
                     c33_file, rel_ship_K, cx_file[:, K]),
            diagnose("C' file MF33 + Cx, rebuilt MF34 (run 90)",
                     c33_file, rel_shipfill_K, cx_file[:, K]),
            diagnose("F' file MF33 + a_0 blocks  (THE PAIR THAT WOULD SHIP)",
                     c33_file, rel_post_K, cx_a0_file[:, K]),
        ]
        print("\n=== PSD against the FILE's MF33, in the chi2's own space ===")
        print(pd.DataFrame(rows3).set_index("case").to_string())
        print("\n  Read A' against A, and B' against B. A difference means the "
              "188-group\n  object the joint is certified against is not "
              "interchangeable with the\n  MF33 the chi2 reads, and every PSD "
              "number in this document predating\n  2026-08-06 inherits that. "
              "Sec. 10.6-3 item 5.")
        print("\n  ⚑ AND READ F' AGAINST F, WHICH IS THE ONLY ROW THAT PRICES\n"
              "  THE FILE'S MF33 NEXT TO THE CROSS TERM WE ACTUALLY SHIP.\n"
              "  Criterion, fixed before measuring: F' must land where F does --\n"
              "  sigma_max(K) on the CONTROL's 0.999993 and lam_min_norm at\n"
              "  machine zero. Compare sigma_max(K), NOT lam_min_norm: the\n"
              "  normaliser is max|diag(J)| and the two rows do not share it.\n"
              "  If sigma_max(K) pulls away from 1, the cross block is NOT\n"
              "  compatible with these marginals and the tape must not ship --\n"
              "  the cross block has to be rebuilt against them instead (§8\n"
              "  step 12(c)). When --source-endf is the run's own tape, F' and F\n"
              "  are the same pair and agreeing proves nothing; it is only a\n"
              "  measurement when the source carries a rewritten MF33.")

        # ⚑ THE BLEND SWEEP (option D) — added 2026-08-14.
        #
        # WHY. F' measured that the FULL split is incompatible with the cross
        # term (84.72 against the control's 1.020), and the cross term ships by
        # physics. So the question is no longer "split or cross" but "HOW MUCH of
        # the split survives next to the cross". C33(s) = (1-s)*ref + s*source is
        # the honest way to ask it:
        #   * both ends carry the SAME diagonal (the split redistributes, it does
        #     not declare), so C33(s) does too -- no extra sigma at ANY s;
        #   * a convex combination of two PSD matrices is PSD, so MF33's own
        #     block never breaks and only the JOINT can fail;
        #   * s = 0 reproduces the shipped tape and s = 1 reproduces the split,
        #     so the endpoints are controls, not extrapolations.
        # Same shape as the cross term's own s_max scan, §10.8-5 step 4.
        #
        # ⛔ s IS CHOSEN AGAINST THE PSD LIMIT, NEVER AGAINST V4. Picking the
        # blend that scores best is rule §10.8-6 -- the thing this work holds
        # against JENDL -- and it would void the whole argument. Read s_max off
        # this table BEFORE any tape is written or scored.
        c33_ref = None
        if args.c33_blend_ref:
            print("\n=== BLEND SWEEP: how much of the split survives the cross "
                  "term? ===")
            ref = load_library_lib_c0(str(args.c33_blend_ref), "This work")
            g33r = np.asarray(ref.get("mf33_grid_ev"), float)
            c33_ref = np.asarray(ref.get("mf33_rel_cov"), float)
            if c33_ref.shape != c33_file.shape or g33r.size != g33.size:
                raise SystemExit(
                    f"--c33-blend-ref MF33 is {c33_ref.shape} on {g33r.size} "
                    f"edges against the source's {c33_file.shape} on {g33.size}; "
                    f"the blend needs one grid.")
            d_grid = float(np.abs(g33r - g33).max())
            d_diag = float(np.abs(np.diag(c33_ref) - np.diag(c33_file)).max())
            d_off = float(np.abs(c33_ref - c33_file).max())
            print(f"  ref    : {args.c33_blend_ref}")
            print(f"  max |grid_ref - grid_src|       = {d_grid:.6e}")
            print(f"  max |diag(ref) - diag(src)|     = {d_diag:.6e}   "
                  f"(must be ~0: the split redistributes, it declares nothing)")
            print(f"  max |ref - src| overall         = {d_off:.6e}   "
                  f"(must NOT be 0, or the two tapes carry the same MF33)")
            if d_off == 0.0:
                raise SystemExit(
                    "--c33-blend-ref carries the SAME MF33 as --source-endf; "
                    "the sweep would be constant. Point it at the unsplit tape.")

            try:
                svals = [float(x) for x in args.c33_blend_s.split(",") if x.strip()]
            except ValueError:
                raise SystemExit(f"--c33-blend-s not parseable: {args.c33_blend_s!r}")

            rows4 = []
            for sv in svals:
                c33_s = (1.0 - sv) * c33_ref + sv * c33_file
                r = diagnose(f"s = {sv:.3f}", c33_s, rel_post_K, cx_a0_file[:, K])
                r["s"] = sv
                rows4.append(r)
            df4 = pd.DataFrame(rows4)
            cols = ["s", "rank33", "rank34", "sigma_max(K)", "leak_null34",
                    "lam_min", "scale", "lam_min_norm"]
            print(pd.DataFrame(df4)[cols].to_string(index=False))
            print("\n  HOW TO READ IT. s = 0 must reproduce the control's F'\n"
                  "  (sigma_max(K) ~ 1.02, lam_min at round-off) and s = 1 must\n"
                  "  reproduce this run's F'. If either endpoint misses, the\n"
                  "  blend is not between the two objects you think it is.\n"
                  "  s_max = the largest s whose lam_min stays at round-off and\n"
                  "  whose sigma_max(K) has not pulled away from the s = 0 value.\n"
                  "  ⛔ Choose s from THIS table only. Scoring several s and\n"
                  "  keeping the best V4 is §10.8-6 and voids the result.\n"
                  "  ⚠ A small s_max is a real answer too: it would say the\n"
                  "  cross term and the redistribution are incompatible at any\n"
                  "  useful amplitude, which is worth writing down as it stands.")

        # ⚑ THE PASS-1 SCALE SWEEP — added 2026-08-14, roadmap §10.
        #
        # WHY. The cross block ships as `cx_post = cx_mc * outer(j33, j34)` with
        # `j33 = sqrt(diag(c33_ship)) / sqrt(diag(joint))`, i.e. it is scaled up
        # to the DECLARED MF33 diagonal. But that diagonal is Pass 2's, and the
        # inflation of the coherent mode by exactly that ratio IS the defect the
        # split-combine repairs. `cx_mc` itself is honest: it comes out of the
        # same `np.cov` call, over the same Pass-1 replicas, as `c33_mc`.
        #
        # And the split is written in that same Pass-1 scale:
        #     C_split = C1' + D_exc R_loc D_exc ,  C1' = Cov(Pass-1) if s1 <= s2
        # so the consistent joint decomposes as
        #     [[C33_split, cx_mc],[cx_mc^T, C34_ship]]
        #       = [[c33_mc, cx_mc],[cx_mc^T, c34_mc]]              <- a SAMPLE cov, PSD
        #       + [[D_exc R_loc D_exc, 0],[0, C34_ship - c34_mc]]  <- block diagonal
        # PSD as soon as both added blocks are, and the control F' already prices
        # the MF34 one at ~2 % (sigma_max 1.0205).
        #
        # THE COUNTER-ARGUMENT, which is why this is measured and not asserted:
        # sigma_max(K) is exactly LINEAR in cx, so a UNIFORM rescale would have
        # to divide the cross block by >= 1/0.011805 = 84.7 to be admissible next
        # to the split -- and j33 is typically ~2.5, not ~85. t = 0 can only pass
        # if the inflation is strongly NON-uniform and sits where cx_mc has mass.
        # (`W33` and `diag(j33)` do not commute, so that is an expectation, not a
        # bound.) Hence the j33 percentiles below: they are the predictor.
        #
        # ⛔ t IS NOT CHOSEN FROM THIS TABLE. Only t = 1 (declared scale) and
        # t = 0 (Pass-1 scale) have a justification; the rest show where it
        # crosses. Picking an intermediate t because it reads better is §10.8-6.
        # ⛔ And nothing here is decided against V4. This is a PSD gate.
        if args.cx_j33_exp:
            print("\n=== PASS-1 SCALE SWEEP on the cross block's MF33 side ===")
            try:
                tvals = [float(x) for x in args.cx_j33_exp.split(",") if x.strip()]
            except ValueError:
                raise SystemExit(f"--cx-j33-exp not parseable: {args.cx_j33_exp!r}")

            live33 = j33 > 0
            q = np.percentile(j33[live33], [0, 5, 25, 50, 75, 95, 100])
            print(f"  j33 = sigma_declared / sigma_Pass1 over {int(live33.sum())} "
                  f"live bins of {j33.size}")
            print("    min {:.4g}  p5 {:.4g}  p25 {:.4g}  MEDIAN {:.4g}  "
                  "p75 {:.4g}  p95 {:.4g}  MAX {:.4g}".format(*q))
            print(f"    bins with j33 > 84.7 : {int((j33 > 84.7).sum())}   "
                  f"(a uniform j33 would have to reach ~85 for t = 0 to pass)")
            print(f"    bins with j33 < 1    : {int((j33[live33] < 1).sum())}   "
                  f"(there Pass 1 is WIDER than the declared sigma, so C1' is "
                  f"capped by d1 = min(s1,s2) and C1' != c33_mc)")

            rows5 = []
            for tv in tvals:
                jt = np.where(live33, np.power(np.where(live33, j33, 1.0), tv), 0.0)
                cx_t = cx_mc * np.outer(jt, j34)
                cx_a0_t = np.divide(
                    cx_t, a_flat_nom[None, :], out=np.zeros_like(cx_t),
                    where=(np.abs(a_flat_nom) >= np.finfo(float).tiny)[None, :])
                cx_t_file = np.zeros((c33_file.shape[0], cx_t.shape[1]))
                cx_t_file[inside] = cx_a0_t[m_of[inside]]

                r = diagnose(f"t = {tv:.3f}  vs SPLIT MF33 (--source-endf)",
                             c33_file, rel_post_K, cx_t_file[:, K])
                r["t"], r["MF33"] = tv, "split"
                rows5.append(r)
                if c33_ref is not None:
                    r2 = diagnose(f"t = {tv:.3f}  vs UNSPLIT MF33 (--c33-blend-ref)",
                                  c33_ref, rel_post_K, cx_t_file[:, K])
                    r2["t"], r2["MF33"] = tv, "unsplit"
                    rows5.append(r2)

            cols5 = ["t", "MF33", "rank33", "rank34", "sigma_max(K)",
                     "leak_null34", "lam_min", "scale", "lam_min_norm"]
            print(pd.DataFrame(rows5)[cols5].to_string(index=False))
            print("\n  ANCHORS -- if either misses, the table is not measuring\n"
                  "  what it claims and NOTHING below it may be read:\n"
                  "    t = 1, split    -> sigma_max(K) = 84.715790  (F', job 8488793)\n"
                  "    t = 1, unsplit  -> sigma_max(K) =  1.020488  (F', job 8488801)\n"
                  "\n  CRITERION, FIXED BEFORE MEASURING (roadmap §10.3), read at\n"
                  "  t = 0 against the SPLIT MF33:\n"
                  "    sigma_max(K) <= 1.05 and lam_min >= -1e-6*scale\n"
                  "        -> the repair and the cross term are compatible after\n"
                  "           all; write the tape with cx(0) and score it ONCE.\n"
                  "    1.05 < sigma_max(K) <= 2\n"
                  "        -> right mechanism, not sufficient alone. §10.5.\n"
                  "    sigma_max(K) > 2\n"
                  "        -> the inflation was not the dominant cause. §10.6,\n"
                  "           starting with whether C1' really is c33_mc.\n"
                  "\n  ⚠ The t = 0 joint carries a DECLARED assumption, not a\n"
                  "  measurement: the split's D_exc R_loc D_exc term is taken to\n"
                  "  have zero cross covariance with a_l, because Pass 2 emits no\n"
                  "  a_l replicas. There is nothing to measure there -- but it is\n"
                  "  an assumption and it goes in the text.")

        # ⚑ SPLIT FORENSICS — added 2026-08-14 after job 8488882, roadmap §10.6.
        #
        # WHAT THE SWEEP ACTUALLY SETTLED. --cx-j33-exp failed (sigma_max 7.61 at
        # t = 0, criterion was <= 1.05), and the UNSPLIT control failed with it:
        # 1.0205 at t = 1 -> 96.39 at t = 0.75. De-scaling breaks a pair that
        # WORKED, so t < 1 measures a broken congruence, not the split. The
        # premise was wrong: sigma_max(K) is INVARIANT under a diagonal
        # congruence -- the CONTROL and CANDIDATE rows of the first table have
        # both read 0.999993 all along -- so "the cross block is inflated" is not
        # a Cauchy-Schwarz defect. c33 is inflated by the same factor.
        #
        # ⚠ AND j33's MEDIAN 13.77 IS NOT AN INFLATION FACTOR. `load_pass1_c0`
        # reads c0 ABSOLUTE and c33_ship is RELATIVE, so j33 carries 1/c0_nom.
        # The physical ratio is j33 * c0_nom, median 13.77 * 0.192845 = 2.66 --
        # consistent with sigma_coh 0.0554 -> 0.0241. For the same reason the
        # printed "bins with j33 < 1" does NOT test sigma_1 <= sigma_2; the
        # correct per-bin condition is j33 * c0_nom >= 1, measured below.
        #
        # WHAT IS STILL OPEN, and it is binary. The algebra of §10.1 lives on the
        # MC's own grid:
        #     [[C33_split, cx_mc],[cx_mc^T, c34_mc]]
        #       = [[c33_mc, cx_mc],[cx_mc^T, c34_mc]]  <- a SAMPLE cov, PSD always
        #       + [[D_exc R_loc D_exc, 0],[0, 0]]      <- PSD always
        # PSD by construction IF C1' = c33_mc. The sweep never tested that -- it
        # tested the FILE's representation (2317 bins, the m_of injection, and
        # c34_ship substituted for c34_mc). If the MC-grid joint passes, the
        # incompatibility is REPRESENTATIONAL and there may be a route; if it
        # fails, it is physical and the line closes.
        #
        # ⛔ THIS IS THE LAST PROBE ON THIS LINE. Two hypotheses have already
        # died here in one day. If it does not come back clean, ship A or B and
        # report 84.72 as it stands. Do not chain a fourth.
        if args.split_forensics:
            print("\n=== SPLIT FORENSICS: physical, or representation? ===")
            c_split = np.load(args.split_forensics).astype(float)
            if c_split.shape != c33_mc.shape:
                raise SystemExit(
                    f"--split-forensics is {c_split.shape}, the MC magnitude "
                    f"block is {c33_mc.shape}; they must be the same grid.")
            # ⚑ THE DENOMINATOR IS c0_HOST, NOT c0_nominal -- fixed 2026-08-14
            # after job 8488889. `mf33_relative_covariance.npy` is
            # cov_abs / outer(c0_host, c0_host), the folded-PENDF File-3
            # denominator that `mf33_build.build_mf33_denominator` computes;
            # `mf33_c0_nominal.npy` is a DIFFERENT array. Using the wrong one
            # made this gate report max|ratio-1| = 4.319790, which is EXACTLY
            # max|c0_host/c0_nom - 1| = 4.319790 -- the gate measured the ratio
            # between the two denominators and nothing else.
            # ⚠ It did NOT void (1), (2) or (3): those use only the ABSOLUTE
            # matrices C_split, c33_mc, c34_mc, cx_mc, and A_mag is exactly the
            # identity on the fine grid (row_aggregator gives w_i/w_i for
            # one-bin groups), so they were unit-consistent throughout.
            _host = run_dir / "mf33_c0_host.npy"
            if not _host.exists():
                raise SystemExit(
                    f"{_host} is missing; it is the denominator MF33's relative "
                    f"covariance is divided by, and mf33_c0_nominal.npy is NOT "
                    f"interchangeable with it (they differ by up to 5.3x here).")
            c0nom = np.load(_host).astype(float)

            # --- UNIT GATE. Both matrices claim ABSOLUTE c0^2. The split
            # preserves the DECLARED sigma exactly, so
            #     sqrt(diag(C_split)) = sigma_abs_declared = j33 * c0_host * sqrt(diag(c33_mc))
            # must hold bin by bin. If it does not, the two objects are not in
            # the same units and NOTHING below may be read.
            s_split = np.sqrt(np.maximum(np.diag(c_split), 0.0))
            s_mc = np.sqrt(np.maximum(np.diag(c33_mc), 0.0))
            phys = j33 * c0nom                 # the PHYSICAL sigma_2/sigma_1 (c0_host)
            pred = phys * s_mc
            ok = pred > 0
            ratio = np.divide(s_split, pred, out=np.ones_like(pred), where=ok)
            print(f"  UNIT GATE  max |sqrt(diag(C_split)) / (j33*c0_nom*sigma_mc) "
                  f"- 1| = {float(np.abs(ratio - 1).max()):.6e}   (must be ~0; "
                  f"c0_host, NOT c0_nominal -- see the note above)")
            q = np.percentile(phys, [0, 5, 25, 50, 75, 95, 100])
            print("  PHYSICAL inflation j33*c0_host = sigma_declared/sigma_Pass1:")
            print("    min {:.4g}  p5 {:.4g}  p25 {:.4g}  MEDIAN {:.4g}  "
                  "p75 {:.4g}  p95 {:.4g}  MAX {:.4g}".format(*q))
            n_bad = int((phys < 1.0).sum())
            print(f"    bins with j33*c0_host < 1 : {n_bad}   <-- THE CORRECT "
                  f"TEST of sigma_1 <= sigma_2. Must be 0, or d1 = min(s1,s2) "
                  f"bites and C1' != c33_mc.")

            # --- (1) IS THE PREMISE TRUE? C_split - c33_mc must be PSD.
            delta = c_split - c33_mc
            delta = 0.5 * (delta + delta.T)
            lam_d = float(np.linalg.eigvalsh(delta)[0])
            nrm = float(np.abs(c_split).max())
            print(f"\n  (1) min eig(C_split - c33_mc) = {lam_d:.6e}   "
                  f"max|C_split| = {nrm:.6e}   ratio = {lam_d / nrm:.3e}")
            print("      PASS if ratio >= -1e-10: then C_split >= c33_mc and the "
                  "joint below is\n      PSD by algebra. FAIL closes the line "
                  "(roadmap §10.6).")

            # --- (2) THE PURE MC JOINT. No file, no m_of, no c34_ship.
            rows6 = [
                diagnose("ANCHOR  c33_mc + cx_mc + c34_mc (must be 0.999993)",
                         c33_mc, c34_mc, cx_mc),
                diagnose("SPLIT   C_split + cx_mc + c34_mc  (THE ALGEBRA)",
                         c_split, c34_mc, cx_mc),
            ]
            print("\n  (2) THE JOINT ON THE MC's OWN GRID, absolute units")
            print(pd.DataFrame(rows6).set_index("case").to_string())
            print("      PASS if SPLIT sigma_max(K) <= 1.001 AND the anchor is on\n"
                  "      0.999993. Then the incompatibility is REPRESENTATIONAL,\n"
                  "      not physical. If the anchor misses, the table is void.")

            # --- (3) WHERE DOES THE 84.72 LIVE?
            # Decompose C33 from the FILE, project the shipped cross block into
            # its eigenbasis and split the Cauchy-Schwarz load by direction.
            # sum_i ||(V^T Cx W34)_i||^2 / w_i = ||W33 Cx W34||_F^2, so the loads
            # are an exact decomposition of the Frobenius norm sigma_max bounds.
            print("\n  (3) WHERE THE LOAD LIVES, by eigen-direction of the "
                  "file's MF33")
            w34m, _, _ = whiten(rel_post_K)
            cxK = cx_a0_file[:, K]
            eig = {}
            for tag, C in (("split", c33_file),
                           ("unsplit", c33_ref if c33_ref is not None else None)):
                if C is None:
                    continue
                w, V = np.linalg.eigh(0.5 * (C + C.T))
                keep = w > NULL_TOL * max(w.max(), 0.0)
                Vk, wk = V[:, keep], w[keep]
                M = (Vk.T @ cxK) @ w34m
                load = (M ** 2).sum(1) / wk
                eig[tag] = (Vk, wk, load)
                order = np.argsort(load)[::-1]
                tot = float(load.sum())
                print(f"\n    {tag}: rank {int(keep.sum())}/{C.shape[0]}   "
                      f"total load ||W33 Cx W34||_F^2 = {tot:.6e}   "
                      f"(sqrt = {np.sqrt(tot):.4f})")
                print("      top 8 directions:  " + "  ".join(
                    f"[w={wk[i]:.3e} load={load[i]:.3e} "
                    f"({100 * load[i] / tot:.1f}%)]" for i in order[:8]))
                cum = np.cumsum(load[order]) / tot
                for f in (0.5, 0.8, 0.95):
                    print(f"      directions carrying {100 * f:.0f}% of the load: "
                          f"{int(np.searchsorted(cum, f) + 1)}")

            # The headline: how much of the split's load lives where the UNSPLIT
            # MF33 has no variance at all -- i.e. in the directions the
            # redistribution OPENS, which come from Pass 2 and where the measured
            # cross covariance with a_l is exactly zero.
            if "unsplit" in eig and "split" in eig:
                Vs, ws, load_s = eig["split"]
                Vu, _, _ = eig["unsplit"]
                proj = Vs - Vu @ (Vu.T @ Vs)
                newness = (proj ** 2).sum(0)          # in [0, 1] per direction
                frac = float((load_s * newness).sum() / load_s.sum())
                print(f"\n    ⚑ FRACTION OF THE SPLIT'S LOAD IN DIRECTIONS THE "
                      f"UNSPLIT MF33 DOES NOT SPAN: {100 * frac:.2f} %")
                print(f"      (split rank {Vs.shape[1]}, unsplit rank "
                      f"{Vu.shape[1]}, opened {Vs.shape[1] - Vu.shape[1]})")
                print("      PASS if >= 80 %: the load sits where Pass 2 opened\n"
                      "      variance and Pass 1 measured NO cross covariance, so\n"
                      "      zeroing it there declares LESS, not more -- a\n"
                      "      projection with a physical justification, not a fit.\n"
                      "      If the load is spread, no projection removes it and\n"
                      "      the line closes (roadmap §10.6).")

        if args.tol_sweep:
            print("\n=== SPECTRA: is whitening these matrices meaningful? ===")
            print(pd.DataFrame([
                spectrum("MF33 188-group (what we certify against)", c33_ship),
                spectrum("MF33 from the file (what the chi2 folds)", c33_file),
                spectrum("MF34 shipped, relative (a_pt space)", rel_ship_K),
            ]).set_index("matrix").to_string())
            print("\n  `kept_min/|most_neg|` < 1 means the smallest RETAINED\n"
                  "  eigenvalue is below the matrix's own noise floor, so\n"
                  "  w^-1/2 amplifies garbage and sigma_max(K) is not a\n"
                  "  measurement. ENDF-6 is a 6-significant-digit ASCII\n"
                  "  format, so an O(1e-7) relative floor is the round trip,\n"
                  "  not a defect in the MC. Sec. 10.1.8-L16.1.")

            print("\n=== TOLERANCE SWEEP: which columns move, and which do not ===")
            sweep = []
            for t in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
                for nm, c33x, c34x, cxx in (
                        ("188grp B", c33_ship, rel_ship_K, cxA),
                        ("file   B'", c33_file, rel_ship_K, cx_file[:, K])):
                    r = diagnose(nm, c33x, c34x, cxx, tol=t)
                    r["tol"] = t
                    sweep.append(r)
            df = pd.DataFrame(sweep)[
                ["tol", "case", "rank33", "rank34", "sigma_max(K)",
                 "leak_null34", "lam_min"]]
            print(df.to_string(index=False))
            print("\n  EXPECTED, and it is the whole point: `lam_min` is a\n"
                  "  property of the joint and must be FLAT across every row --\n"
                  "  if it is not, something other than the tolerance is\n"
                  "  moving. `sigma_max(K)` and `leak_null34` will move, and\n"
                  "  the tolerance at which the two `case` blocks stop\n"
                  "  disagreeing is the one above both noise floors. Read\n"
                  "  sigma_max(K) THERE, not at NULL_TOL. Sec. 10.1.8-L16.2.")

    if args.check:
        print("\n--check: nothing written.")
        return 0

    if _fine_mag:
        print("\n  no .npy sidecar written (--mag-grid fine): the cross term "
              "ships INSIDE\n  the ENDF's MF34 a_0 blocks, and §L8 inverted "
              "says the sidecar must be absent.\n  Score with "
              "KIKA_MF33_MF34_CROSS_FROM_FILE=1.")
    else:
        out = cx_post.reshape(n_gm, n_gs, L_MAX)
        np.save(run_dir / "mf33_mf34_cross_group_covariance.npy", out)
        np.save(run_dir / "mf33_mf34_cross_group_mag_grid_ev.npy", mag_ev)
        np.save(run_dir / "mf33_mf34_cross_group_shape_grid_ev.npy", shape_ev)
        print(f"\n  wrote mf33_mf34_cross_group_covariance.npy {out.shape}")
        print("  wrote mf33_mf34_cross_group_mag_grid_ev.npy, "
              "mf33_mf34_cross_group_shape_grid_ev.npy")

    if args.write_endf:
        _cen = 0.5 * (shape_ev[:-1] + shape_ev[1:])
        _outside = (_cen < fine_ev[0]) | (_cen > fine_ev[-1])
        _dead = np.abs(a_nom_group) < np.finfo(float).tiny
        print(f"\n  ⚑ what this rewrite REMOVES relative to {source_endf.name}:")
        print(f"    dead (group, order) slots written zero: "
              f"{int(_dead.sum())} of {_dead.size} "
              f"({100.0 * _dead.sum() / _dead.size:.1f} %) — the SAME set the "
              f"null mask removes, so score against 086_nonull, not run 086")
        print(f"    shape groups outside the fine range: "
              f"{int(_outside.sum())} of {_outside.size} — these carried the "
              f"host's merged MF34 and now carry nothing")

        # ── the mesh per order, if this run has one ───────────────────────────
        #
        # `auto` is what makes this switch safe to leave in: with one shared
        # mesh every e_l IS the base grid, the branch is not entered at all, and
        # the emission is the pre-existing code path untouched.
        # ⚑ `auto` KEYS ON THE RUN'S OWN mf34_per_order_mesh.npz, NOT on whether
        # the tape's grids differ. They already differ on every tape shipped so
        # far -- run 94's blocks carry 703/703/703/684/673/673 groups -- because
        # `merge_mf34` unions each pair with the host's, which adds ~30
        # boundaries in 0.05-0.55 and 14.5-19 MeV. Those sit entirely OUTSIDE
        # the fits, where a_l is zero and the emission writes zeros either way;
        # collapsing onto them would still move those zeros to a coarser grid
        # and change the file. Firing only on the npz means a run that did not
        # choose a mesh is untouched, byte for byte.
        _order_grids = _valid_width = None
        _has_mesh = (run_dir / "mf34_per_order_mesh.npz").exists()
        if args.per_order_mesh == "on" or (args.per_order_mesh == "auto" and _has_mesh):
            _g = per_order_shape_grids(blocks, shape_ev, run_dir=run_dir)
            if all(np.array_equal(_g[l], shape_ev) for l in _g):
                if args.per_order_mesh == "on":
                    raise SystemExit(
                        "--per-order-mesh on, but every order is already on the "
                        "base grid: there is no mesh to emit.")
            else:
                _order_grids = _g
                # Same width convention as the nominal aggregator above, or
                # the emission weights u = w * a would average over a different
                # set of fine bins than the group mean they normalise against.
                _valid_width = np.stack(
                    [np.bincount(g_shape,
                                 weights=np.where(valid[:, l] | _carry,
                                                  widths, 0.0),
                                 minlength=n_gs)
                     for l in range(L_MAX)], axis=-1)
                print(f"\n  per-order mesh: "
                      + ("mf34_per_order_mesh.npz" if _has_mesh
                         else "the tape's own diagonal blocks"))

        write_consistent_mf34(
            out_path=Path(args.write_endf),
            source_endf=source_endf,
            c34_post=c34_post, cx_post=cx_post, a_nom_group=a_nom_group,
            shape_ev=shape_ev, mag_ev=mag_ev, c34_rel_ship=c34_rel_ship,
            null_fill=args.null_fill,
            cross_emin_ev=args.cross_emin_ev, cross_emax_ev=args.cross_emax_ev,
            mf33_file_grid=(mf33_file_grid_ev(source_endf, cache)
                            if _fine_mag else None),
            order_grids_ev=_order_grids, base_valid_width=_valid_width,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
