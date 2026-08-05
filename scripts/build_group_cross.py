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

def mf34_group_edges_ev(run_dir: Path, cache: Path) -> dict:
    """Per-(l_row,l_col) MF34 group edges and relative matrices, from the file.

    Cached because the ENDF parse is the slow step and does not depend on
    anything else here.
    """
    mg = next(iter(sorted(run_dir.glob("*_mg.endf"))), None)
    if mg is None:
        raise SystemExit(f"no *_mg.endf in {run_dir}")
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

def shipped_c34_on_base(blocks, base_ev, a_nom_group):
    """Absolute shipped MF34 expanded onto the base group grid.

    Coarser blocks are mapped by duplication onto the base grid — that is what
    the file asserts, not an approximation. Relative -> absolute via
    outer(a_nom, a_nom), never the reverse.
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
            sub = mat[np.ix_(gi, gj)] * np.outer(a_nom_group[:, lr - 1],
                                                 a_nom_group[:, lc - 1])
            out[np.ix_(np.arange(n_g) * L_MAX + lr - 1,
                       np.arange(n_g) * L_MAX + lc - 1)] = sub
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cache", default=None)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cache = Path(args.cache) if args.cache else run_dir / ".group_cross_cache"

    fine_ev = np.load(run_dir / "mf33_energy_grid_ev.npy").astype(float)
    mag_ev = np.load(run_dir / "mf33_multigroup_grid_ev.npy").astype(float)
    c33_ship = np.load(run_dir / "mf33_multigroup_relative_covariance.npy")
    n_fine = len(fine_ev) - 1
    widths = np.diff(fine_ev) / 1e6

    blocks = mf34_group_edges_ev(run_dir, cache)
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
    c34_ship = shipped_c34_on_base(blocks, shape_ev, a_nom_group)

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
    ]
    print("\n=== PSD on the run's own adaptive grids ===")
    print(pd.DataFrame(rows).set_index("case").to_string())
    print(f"\n  ||Cx||_F = {np.linalg.norm(cx_post):.4g}")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
