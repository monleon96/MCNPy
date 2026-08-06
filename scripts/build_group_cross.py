#!/usr/bin/env python
"""Build the MF33<->MF34 cross block in the pipeline's OWN adaptive group space.

This is both the measurement and the production writer. `--check` reports the PSD
diagnostics and writes nothing; `--write` emits the sidecars that
`precompute_chi2_predictive.load_mf33_mf34_cross` consumes.

WHY THIS EXISTS (roadmap Sec 10.1.8). Runs 87-88 estimated the cross block by
pairing replica s across energies in PASS 2, whose RNG is seeded per BIN
(`exfor_to_endf_research.py:1535`), and glued it to marginals whose correlations
come from PASS 1, seeded per REPLICA (`exfor_utils.py:2002`). Measured: the
Pass-2 cross-energy correlation equals its own permutation null exactly, so that
block carried no information; and under the Pass-1 law the correlation is large
(median |rho| 0.18-0.59 within bin, 0.08-0.55 across energy).

WHY GROUP SPACE. The pipeline does collapse-then-assemble, and the aggregation A
mixes bins, so corr(A C A^T) != A corr(C) A^T. Rescaling to Pass-2 sigma on the
fine grid and collapsing afterwards leaves the cross block nothing consistent to
attach to; on the fine grid the shipped MF34 also spans only ~41*6 of 960
directions per window, purely because several fine bins share a group. Doing the
rescaling AFTER the collapse makes it a single positive diagonal congruence on
the whole joint, which is PSD by construction. And collapsing the REPLICAS is
exactly equivalent to collapsing the covariance --

    C = Z^T Z / (n-1)   =>   A C A^T = (A Z^T)(Z A^T) / (n-1)

-- so nothing large is ever formed.

THE GRIDS ARE NOT OURS TO CHOOSE. The magnitude axis uses the run's MF33
adaptive grid (`mf33_multigroup_grid_ev.npy`) and the shape axis the run's MF34
adaptive grid, read back from the shipped _mg.endf. The shipped MF34 blocks sit
on four nested grids (673 / 676 / 684 / 703 groups, all subsets of the
703-group one); this builds on the 703-group base so the cross block has the same
resolution as the MF34 it must satisfy Cauchy-Schwarz against. Coarser blocks are
expanded onto it by duplication, which is what the file already means.

Run (check only, reads a run dir, writes nothing):
  <venv>/bin/python build_group_cross.py --run-dir <dir> --check
Write the sidecars:
  <venv>/bin/python build_group_cross.py --run-dir <dir> --write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

L_MAX = 6
A_COLS = [f"a_{l}" for l in range(1, L_MAX + 1)]
ZA, MT = 26056, 2
NULL_TOL = 1e-10
CHUNK = 250
SPREAD_EPS = 1e-12          # relative; matches exfor_to_endf_research.py:1393


# --------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------

def shipped_mg_endf(run_dir: Path) -> Path:
    """The ONE shipped multigroup ENDF in a run dir. Ambiguity is fatal.

    THIS IS NOT PEDANTRY, IT IS THE SELF-COMPARISON TRAP. `--write-endf` puts a
    SECOND *_mg.endf in this directory, and it sorts BEFORE the shipped one
    ("..._consistent_mg.endf" < "..._nominal_mg.endf"). A first-wins glob would
    therefore, on the very next run, read the shape grids and `c34_ship` off our
    OWN OUTPUT instead of the shipped file -- and then the CANDIDATE and LEGACY
    rows would be diagnosing the candidate against itself, so every PSD number
    would come out perfect for the one reason that makes it meaningless.

    Refuse and make the caller name the file. Sec. 10.1.8-L exists because a
    joint was verified that was not the joint being shipped; this is the same
    mistake with the two files swapped.
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
    # Fingerprint the source in the cache name. A cache built from a DIFFERENT
    # run's MF34 would be silently wrong -- same shapes, wrong grids -- and the
    # result would look entirely plausible. Name and size are enough to stop a
    # cross-run reuse; the cache is a speed-up, not a store of record.
    raw = cache / f"mf34_groups__{mg.name}__{mg.stat().st_size}.npz"
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


def base_shape_grid(blocks: dict) -> np.ndarray:
    """The finest MF34 grid; every other block's grid must be a subset of it."""
    keys = [k[2:] for k in blocks if k.startswith("m_")]
    cand = max((blocks[f"e_{k}"] for k in keys), key=len)
    for k in keys:
        if not np.isin(blocks[f"e_{k}"], cand).all():
            raise SystemExit(
                f"MF34 block {k} sits on a grid that is not a subset of the "
                f"finest one; expanding by duplication would be wrong."
            )
    return cand


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


# --------------------------------------------------------------------------
# replicas
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# shipped MF34 on the base group grid
# --------------------------------------------------------------------------

def shipped_c34_rel_on_base(blocks, base_ev):
    """Shipped RELATIVE MF34 expanded onto the base group grid.

    Coarser blocks are mapped by duplication onto the base grid — that is what
    the file asserts, not an approximation.

    Kept separate from the absolute form because the relative matrix is needed
    in its own right: it is what the rewrite must fall back on wherever the
    absolute round trip destroys the file's content (see `write_consistent_mf34`
    on null parameters).
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
            gi = np.clip(np.searchsorted(edges, centres, side="right") - 1,
                         0, mat.shape[0] - 1)
            gj = np.clip(np.searchsorted(edges, centres, side="right") - 1,
                         0, mat.shape[1] - 1)
            out[np.ix_(np.arange(n_g) * L_MAX + lr - 1,
                       np.arange(n_g) * L_MAX + lc - 1)] = mat[np.ix_(gi, gj)]
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


# --------------------------------------------------------------------------

def whiten(A, tol=NULL_TOL):
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    keep = w > tol * max(w.max(), 0.0)
    return (V[:, keep] @ np.diag(w[keep] ** -0.5) @ V[:, keep].T,
            V[:, ~keep] @ V[:, ~keep].T, int(keep.sum()))


def diagnose(name, c33, c34, cx):
    w33, p33, r33 = whiten(c33)
    w34, p34, r34 = whiten(c34)
    nx = float(np.linalg.norm(cx, "fro"))
    j = np.block([[c33, cx], [cx.T, c34]])
    lam = float(np.linalg.eigvalsh(0.5 * (j + j.T))[0])
    sc = float(np.max(np.abs(np.diag(j))))
    return {"case": name, "rank33": f"{r33}/{c33.shape[0]}",
            "rank34": f"{r34}/{c34.shape[0]}",
            "sigma_max(K)": float(np.linalg.norm(w33 @ cx @ w34, 2)) if nx else 0.0,
            "leak_null34": float(np.linalg.norm(cx @ p34, "fro")) / nx if nx else 0.0,
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


def write_consistent_mf34(out_path: Path, source_endf: Path, c34_post, cx_post,
                          a_nom_group, shape_ev, mag_ev, c34_rel_ship,
                          cross_emin_ev=None, cross_emax_ev=None):
    """Emit the _mg.endf whose MF34 *is* the joint that was just diagnosed.

    WHY THIS EXISTS. `cx_post` is Cauchy-Schwarz-compatible with `c34_post`, not
    with the MF34 already in the file -- the two share a diagonal but not their
    correlations, and the violation that fact produces is first order (Sec.
    10.1.8-L: sigma_max(K) 41.27). So the cross block can only be shipped
    alongside the marginals it was built with, which means rewriting MF34.

    EVERYTHING GOES OUT RELATIVE, because the format leaves no choice: MF34
    admits LB = 0, 1, 2, 5 and 6 (manual Sec. 34.2), of which only LB=0 is
    absolute and LB=0 carries no off-diagonal structure at all. The matrix forms
    LB=5/LB=6 are defined as Cov(X_i, Y_j) = sum P F X_i Y_j, i.e. relative. (MF35
    had to invent LB=7 to get an absolute *matrix*, and LB=7 is not allowed here.)
    So there is no absolute option and hence nothing to mix -- the whole section
    stays relative, exactly as the file already is.

    The (L=0, L1) blocks are the format's own mechanism for this: "one expresses
    covariances with the a0 Legendre coefficients even though a0 = 1 in the ENDF
    system" (Sec. 34.1), with the (0,0) magnitude self-block written null because
    that covariance belongs to MF33 and must not be repeated (Sec. 34.3).

    Parameters
    ----------
    c34_rel_ship : np.ndarray
        The shipped RELATIVE MF34 on the base shape grid. Used only where a_l is
        exactly zero, i.e. where the absolute round trip is 0/0 and this rebuild
        spans no direction; see the null-parameter note in the body. Pass the
        same array the diagnosis derived `c34_ship` from, never a re-mapping.
    """
    from kika.endf.writers.mf34_writer import (
        create_mf34_from_covariance, write_mf34_to_file,
    )

    n_gs = len(shape_ev) - 1
    a_flat = a_nom_group.reshape(-1)          # (n_gs*L,), (group, order) order

    # --- absolute -> relative -----------------------------------------------
    # The shape blocks divide by outer(a, a) and the cross blocks by a on the
    # shape axis only (cx_post is already relative in sigma). Where a_l is
    # nonzero, c34_post's absolute diagonal equals the shipped one to 2.8e-17
    # and both are divided by the same a^2, so the RELATIVE sigmas the file
    # publishes are reproduced -- the rewrite moves correlations and nothing
    # else.
    #
    # NULL PARAMETERS, WHICH ARE 37 % OF THE MATRIX AND NOT A CORNER CASE.
    # `row_aggregator` gives an all-zero row to any (group, order) with no valid
    # fine bin, so a_l is EXACTLY zero at ~1542 of 4218 slots (39 of 703 groups
    # at l=1, 572 at l=6). There the collapsed replicas are identically zero, so
    # c34_post and c34_ship are BOTH exactly zero and the relative form is 0/0 --
    # genuinely undefined, not merely ill-conditioned.
    #
    # Sec. 4.3's "zero bins at any order" does NOT cover this: it measured
    # FITTED coefficients approaching the serialisation floor, a different
    # population from parameters the aggregator never populated at all.
    #
    # DECISION (Juan, 2026-08-06): rewrite only where the Pass-1 covariance has
    # support, and carry the shipped file's own relative values through
    # untouched at the null slots. The rebuild has nothing to say about a
    # direction it does not span, and the alternative -- writing zero -- would
    # DELETE declared uncertainty at 1542 parameters (the file publishes up to
    # 7.6 relative variance there), which would make "run 90 moves correlations
    # only" false. Absolute space is zero under either choice, so the chi2, the
    # PSD verdict and Sigma_eval are all unaffected: a consumer converting back
    # multiplies by a_l = 0 and annihilates whatever is written.
    live = np.abs(a_flat) >= np.finfo(float).tiny
    both_live = np.outer(live, live)

    # The one case that must still stop the run: a MATERIALLY nonzero absolute
    # covariance sitting on a zero-mean parameter. That is not a null direction,
    # it is an infinite relative uncertainty, and no written value represents it.
    #
    # The threshold is scaled, not exact. The collapse should give exactly 0.0
    # there (an all-zero aggregator row makes the matmul, the mean and every
    # deviation exactly zero), but tying a 40-minute job to bit-exact zero would
    # let one denormal abort it. Anything at double-precision roundoff of the
    # matrix scale is dust and is discarded with the rest of the null slot.
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

    denom = np.outer(a_flat, a_flat)
    c34_rel = np.divide(c34_post, denom, out=np.array(c34_rel_ship, float),
                        where=both_live)
    # The cross blocks are NEW, so there is no shipped value to preserve and
    # nothing to decide: a parameter with no variance has no covariance either,
    # and zero is the correct entry.
    cx_rel = np.divide(cx_post, a_flat[None, :], out=np.zeros_like(cx_post),
                       where=live[None, :])
    n_null = int((~live).sum())
    print(f"\n  null parameters: {n_null} of {a_flat.size} group-mean a_l are "
          f"exactly zero")
    if n_null:
        print(f"    shape blocks there: shipped relative values PRESERVED "
              f"(max |rel| {np.abs(c34_rel_ship[~both_live]).max():.4g})")
        print(f"    cross blocks there: written ZERO")

    for name, arr in (("c34_rel", c34_rel), ("cx_rel", cx_rel)):
        if not np.all(np.isfinite(arr)):
            raise SystemExit(f"{name} is not finite after the relative conversion")
    print(f"\n  relative conversion: min |a_l group mean| = {np.abs(a_flat).min():.3e}")
    print(f"    max |c34_rel| = {np.abs(c34_rel).max():.4g}   "
          f"max |cx_rel| = {np.abs(cx_rel).max():.4g}")

    # --- energy support of the cross blocks ----------------------------------
    # Sub-subsections "need not be given for all values of L and L1" and LB=6
    # carries its own row grid, so the cross term CAN be restricted in energy.
    # By default it is not: the magnitude grid already spans exactly the range
    # the evaluation covers, and the sidecar the chi2 reads is unrestricted, so
    # anything narrower would put a different block in the file from the one
    # being scored -- and it is the windowed version this script's PSD gate
    # would then be certifying. Keep file and sidecar the same object.
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

    cross_cov = {l1: cx_win[:, :, l1 - 1] for l1 in range(1, L_MAX + 1)}

    za, awr, mat = _za_awr_mat_from_endf(source_endf)
    mf34 = create_mf34_from_covariance(
        c34_rel, np.asarray(shape_ev, float), L_MAX, za, awr, mat, 2,
        ltt=1, cross_cov=cross_cov, cross_energy_grid_ev=cross_grid,
    )
    sub = mf34._subsections[0]
    print(f"    MF34: LTT={mf34._ltt}, NL={sub.nl}, "
          f"{len(sub.sub_subsections)} sub-subsections "
          f"(NL(NL+1)/2 = {sub.nl * (sub.nl + 1) // 2})")

    # THE INSTALLED kika IS NOT VERIFIABLE FROM WHERE THIS RUNS, SO CHECK THE
    # ARTEFACT INSTEAD. The NL-is-a-count and LTT=3 fixes (Sec. 10.1.8-L6) live
    # in the LIBRARY, which the cluster gets as an installed wheel, not by file
    # copy -- so this script can be up to date while the kika under it is not.
    # Both old defects are SELF-CONSISTENT within kika (the old parser derives
    # its sub-subsection count from the old NL convention and reads the file
    # back happily), so nothing downstream would fail; the file would simply be
    # wrong for every other code that reads it, and silently. This file is meant
    # to leave our tooling, so refuse rather than ship it.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cache", default=None)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument(
        "--write-endf", metavar="OUT",
        help="Also emit a _mg.endf whose MF34 IS the consistent joint: shape "
             "blocks from c34_post and (L=0, L1) cross blocks from cx_post. "
             "MF3/MF4/MF33 are copied from --source-endf untouched.",
    )
    ap.add_argument("--source-endf", default=None,
                    help="Template for --write-endf; defaults to the run dir's *_mg.endf")
    # Default: the WHOLE magnitude grid. The evaluation only exists over
    # 0.8468-4.075 MeV in the first place, so the grid already IS the range the
    # data constrain -- clipping it to a nominal 0.85-4.0 would drop real groups
    # at both ends AND, worse, leave the file asserting a different cross block
    # from the sidecar the chi2 reads, which is exactly the diagnose-one-thing/
    # ship-another mistake that produced run 89. Narrow these only deliberately.
    ap.add_argument("--cross-emin-ev", type=float, default=None)
    ap.add_argument("--cross-emax-ev", type=float, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cache = Path(args.cache) if args.cache else run_dir / ".group_cross_cache"

    fine_ev = np.load(run_dir / "mf33_energy_grid_ev.npy").astype(float)
    mag_ev = np.load(run_dir / "mf33_multigroup_grid_ev.npy").astype(float)
    c33_ship = np.load(run_dir / "mf33_multigroup_relative_covariance.npy")
    n_fine = len(fine_ev) - 1
    widths = np.diff(fine_ev) / 1e6

    # ONE resolved source, used for BOTH the shipped grids/c34_ship and the
    # --write-endf template. They must be the same file or the diagnosis is not
    # about the thing being written.
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
    a_nom_group = np.stack(
        [A_shape[l] @ a_nom_fine[:, l] for l in range(L_MAX)], axis=-1)

    keep = np.isfinite(c0_g).all(1) & np.isfinite(a_g).all((1, 2))
    c0_g, a_g = c0_g[keep], a_g[keep]
    n = int(keep.sum())
    a_flat = a_g.reshape(n, n_gs * L_MAX)
    print(f"  replicas: {n}")

    # Every block from ONE collapsed replica set: PSD by construction.
    joint = np.cov(np.hstack([c0_g, a_flat]), rowvar=False)
    c33_mc = joint[:n_gm, :n_gm]
    c34_mc = joint[n_gm:, n_gm:]
    cx_mc = joint[:n_gm, n_gm:]

    print("  mapping the shipped MF34 onto the base shape grid ...", flush=True)
    # Mapped ONCE and reused: the absolute form drives the diagnosis, the
    # relative form is what the rewrite falls back on at null parameters. Two
    # separate mappings would be two chances to disagree about the grids.
    c34_rel_ship = shipped_c34_rel_on_base(blocks, shape_ev)
    _a = a_nom_group.reshape(-1)
    c34_ship = c34_rel_ship * np.outer(_a, _a)
    c34_ship = 0.5 * (c34_ship + c34_ship.T)

    # The two-pass rescaling, done AFTER the collapse: one positive diagonal
    # congruence on the WHOLE joint, so PSD is inherited from the control and
    # both marginals come out equal to the shipped ones.
    d_tar = np.concatenate([np.sqrt(np.maximum(np.diag(c33_ship), 0.0)),
                            np.sqrt(np.maximum(np.diag(c34_ship), 0.0))])
    d_mc = np.sqrt(np.maximum(np.diag(joint), 0.0))
    jj = np.divide(d_tar, d_mc, out=np.zeros_like(d_tar), where=d_mc > 0)
    j33, j34 = jj[:n_gm], jj[n_gm:]
    c33_post = c33_mc * np.outer(j33, j33)
    c34_post = c34_mc * np.outer(j34, j34)
    cx_post = cx_mc * np.outer(j33, j34)

    e33 = np.max(np.abs(np.sqrt(np.maximum(np.diag(c33_post), 0)) - d_tar[:n_gm]))
    e34 = np.max(np.abs(np.sqrt(np.maximum(np.diag(c34_post), 0)) - d_tar[n_gm:]))
    print(f"\n  marginal-identity gate: max abs sigma error  "
          f"MF33 {e33:.3e}   MF34 {e34:.3e}")

    rows = [
        diagnose("CONTROL: all blocks from the collapsed replicas", c33_mc, c34_mc, cx_mc),
        diagnose("shipped marginals, Cx = 0", c33_ship, c34_ship, np.zeros_like(cx_post)),
        diagnose("CANDIDATE: rescaling after the collapse", c33_post, c34_post, cx_post),
        # THE ROW RUN 89 DID NOT HAVE, AND THE ONE THAT MATTERS. The CANDIDATE
        # is PSD as a TRIPLE; its MF34 leg is c34_post, not the MF34 in the
        # _mg.endf. Run 89 shipped cx_post as a sidecar next to the FILE's
        # marginals -- a joint nothing had ever diagnosed. Measured: sigma_max(K)
        # 41.27 and lam_min/scale -0.447, i.e. worse than every construction in
        # Sec. 10.1.8-J. The marginal-identity gate above only pins the DIAGONAL,
        # so it cannot see this. Keep this row: it is the one that says whether
        # the thing we are about to ship is a covariance.
        diagnose("LEGACY (run 89): shipped marginals + cx_post",
                 c33_ship, c34_ship, cx_post),
    ]
    print("\n=== PSD on the run's own adaptive grids ===")
    print(pd.DataFrame(rows).set_index("case").to_string())
    print(f"\n  ||Cx||_F = {np.linalg.norm(cx_post):.4g}")
    print("\n  Read the CANDIDATE row against the LEGACY row: the cross block is\n"
          "  only a covariance next to the marginals it was built with. Shipping\n"
          "  it therefore means shipping c34_post as MF34 too (--write-endf).")

    if args.check:
        print("\n--check: nothing written.")
        return 0

    # Sidecars. The magnitude axis is RELATIVE (delta_sigma/sigma) because
    # c33_ship is the shipped relative MF33 and the chi2 builder's magnitude
    # sensitivity is y_eval; the shape axis is ABSOLUTE because the relative
    # form would divide by a_l_nom, which crosses zero.
    out = cx_post.reshape(n_gm, n_gs, L_MAX)
    np.save(run_dir / "mf33_mf34_cross_group_covariance.npy", out)
    np.save(run_dir / "mf33_mf34_cross_group_mag_grid_ev.npy", mag_ev)
    np.save(run_dir / "mf33_mf34_cross_group_shape_grid_ev.npy", shape_ev)
    print(f"\n  wrote mf33_mf34_cross_group_covariance.npy {out.shape}")
    print("  wrote mf33_mf34_cross_group_mag_grid_ev.npy, "
          "mf33_mf34_cross_group_shape_grid_ev.npy")

    if args.write_endf:
        write_consistent_mf34(
            out_path=Path(args.write_endf),
            source_endf=source_endf,
            c34_post=c34_post, cx_post=cx_post, a_nom_group=a_nom_group,
            shape_ev=shape_ev, mag_ev=mag_ev, c34_rel_ship=c34_rel_ship,
            cross_emin_ev=args.cross_emin_ev, cross_emax_ev=args.cross_emax_ev,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
