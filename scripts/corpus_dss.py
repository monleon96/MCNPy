"""Compute a Gaussian Dawid--Sebastiani score for the folded EXFOR corpus.

For one experiment and library, let E be the experimental covariance, S the
folded evaluated covariance, and r the residual.  The comparable score is

    DSS_rel = r.T @ (E + S)^-1 @ r + log det(E + S) - log det(E)
            = V4 + log det(I + E^-1/2 S E^-1/2).

Subtracting log det(E) removes a term common to all libraries on the same
experiment and makes the reported score dimensionless.  Lower is better for
the same observations.  DSS has no universal numerical target; V4/N retains
its separate calibration target of approximately one.

The 91_cross sidecar already contains each dense folded S block.  This script
uses the same exact Cholesky as ``corpus_absorbed.py``; after factorization, the
log determinant costs only a sum over the Cholesky diagonal.

The output is checkpointed after every library/experiment block.  ``--resume``
skips completed blocks, so a requeued cluster job does not repeat them.
"""

import argparse
import gc
import os
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


CH = os.environ.get("KIKA_CHI2_DIR", "/share_snc/snc/JuanMonleon/chi2")
SIGMA_STAT_FLOOR_REL = 0.01
DIAGONAL_FLOOR_REL = 0.001


def components(sub):
    """Build the same E and residual used by the published V2/V4 results."""
    y = sub.y_exp.to_numpy()
    sstat = np.maximum(
        np.nan_to_num(sub.sigma_exp_stat.to_numpy()),
        SIGMA_STAT_FLOOR_REL * np.abs(y),
    )
    D = np.maximum(sstat**2, (DIAGONAL_FLOOR_REL * np.abs(y)) ** 2)
    u = np.nan_to_num(sub.sigma_sys_indep_rel.to_numpy()) * y
    v = np.nan_to_num(sub.sigma_sys_dep_rel.to_numpy()) * y
    r = y - sub.y_eval.to_numpy()
    return D, u, v, r


def stream_block(npz_path, key, tmpdir):
    """Extract one NPZ member to a memory-mapped local NPY file."""
    dst = os.path.join(tmpdir, f"{key.replace('@@', '_')}.npy")
    with zipfile.ZipFile(npz_path).open(f"{key}.npy") as src, open(dst, "wb") as out:
        while True:
            chunk = src.read(1 << 24)
            if not chunk:
                break
            out.write(chunk)
    return np.load(dst, mmap_mode="r"), dst


def logdet_experimental(D, u, v):
    """Return log det(diag(D) + u u.T + v v.T) by determinant lemma."""
    B = np.column_stack([u, v])
    small = np.eye(2) + B.T @ (B / D[:, None])
    sign, value = np.linalg.slogdet(small)
    if sign <= 0:
        raise np.linalg.LinAlgError("experimental covariance is not positive definite")
    return float(np.log(D).sum() + value)


def dss(block, D, u, v, r, pos=None):
    """Return V4, the relative log-volume penalty, and relative DSS."""
    N = D.size
    total = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        src = block[i] if pos is None else block[pos[i]]
        row = np.asarray(src, dtype=np.float64)
        if pos is not None:
            row = row[pos]
        row += u[i] * u
        row += v[i] * v
        row[i] += D[i]
        total[i] = row
    total += total.T
    total *= 0.5

    cf, low = cho_factor(total, lower=True, overwrite_a=True, check_finite=False)
    del total
    gc.collect()

    chi2_v4 = float(r @ cho_solve((cf, low), r, check_finite=False))
    logdet_total = float(2.0 * np.log(np.diag(cf)).sum())
    logdet_ratio = logdet_total - logdet_experimental(D, u, v)
    return chi2_v4, logdet_ratio, chi2_v4 + logdet_ratio


def checkpoint(rows, dst):
    """Atomically write the accumulated rows in stable order."""
    out = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["library", "experiment_id"], keep="last")
        .sort_values(["experiment_id", "library"])
    )
    tmp = f"{dst}.tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--max-n", type=int, default=16000)
    ap.add_argument("--only", default=None, help="single experiment_id")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    pq = f"{CH}/chi2_data_predictive_{args.tag}.parquet"
    npz = pq + ".eval_cov.npz"
    suffix = f"_{args.only}" if args.only else ""
    dst = args.out or f"{CH}/dss_{args.tag}{suffix}.csv"

    rows = []
    completed = set()
    if args.resume and os.path.exists(dst):
        previous = pd.read_csv(dst)
        rows = previous.to_dict("records")
        completed = {
            (str(row.library), str(row.experiment_id))
            for row in previous.itertuples(index=False)
        }
        print(f"resuming {dst}: {len(completed)} completed blocks", flush=True)

    df = pd.read_parquet(pq)
    names = {name[:-4] for name in zipfile.ZipFile(npz).namelist()}
    skipped = []
    tmpdir = tempfile.mkdtemp(prefix="corpus_dss_")
    try:
        for (lib, eid), sub in df.groupby(
            ["library", "experiment_id"], observed=True
        ):
            identity = (str(lib), str(eid))
            if identity in completed:
                continue
            if args.only and str(eid) != args.only:
                continue
            key = f"{lib}@@{eid}"
            if key not in names:
                continue
            sub = sub.sort_values("_eval_pos")
            N = len(sub)
            if N > args.max_n:
                skipped.append((lib, eid, N))
                continue

            started = time.time()
            block, extracted = stream_block(npz, key, tmpdir)
            p = sub._eval_pos.to_numpy(int)
            pos = None if (p == np.arange(N)).all() else p
            D, u, v, r = components(sub)
            chi2_v4, logdet_ratio, score = dss(block, D, u, v, r, pos)
            del block
            gc.collect()
            os.unlink(extracted)

            rows.append(
                dict(
                    library=lib,
                    experiment_id=eid,
                    author=sub.author.iloc[0],
                    N=N,
                    chi2_v4=chi2_v4,
                    chi2_v4_per_N=chi2_v4 / N,
                    logdet_ratio=logdet_ratio,
                    logdet_ratio_per_N=logdet_ratio / N,
                    dss_relative=score,
                    dss_relative_per_N=score / N,
                )
            )
            checkpoint(rows, dst)
            print(
                f"  {lib:10s} {eid} {str(sub.author.iloc[0])[:14]:14s} "
                f"N={N:6d} V4/N={chi2_v4/N:9.4f} "
                f"logdet/N={logdet_ratio/N:9.4f} "
                f"DSSrel/N={score/N:9.4f} [{time.time()-started:.0f}s]",
                flush=True,
            )
    finally:
        for name in os.listdir(tmpdir):
            try:
                os.unlink(os.path.join(tmpdir, name))
            except OSError:
                pass
        os.rmdir(tmpdir)

    if not rows:
        print("no rows produced", file=sys.stderr)
        return 1
    checkpoint(rows, dst)
    print(f"\nwrote {dst} ({len(rows)} accumulated rows)")
    if skipped:
        print("SKIPPED (block larger than --max-n):")
        for lib, eid, N in skipped:
            print(f"  {lib:10s} {eid} N={N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
