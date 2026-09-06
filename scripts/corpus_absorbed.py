"""The informativeness column: how many directions does each library's Sigma_eval
actually declare anything in, measured in the experiment's own metric?

    n_absorbed = N - tr( (I + E^-1/2 Sigma_eval E^-1/2)^-1 ),
    E = diag(D) + u u^T + v v^T          (the EXFOR budget, identical across libraries)

A covariance coarse enough to absorb anything spends a handful of directions; a
fine one spends hundreds. Without this column a chi2 table reads JENDL's 0.034 on
Cierjacks and 0.059 on Kinney as accuracy. Roadmap §10.8-19, §10.8-6.

Exact Cholesky; only the trace is stochastic (Hutchinson, `--probes`).

Memory. Sigma is built ROW BY ROW so no `np.outer` temporary is ever allocated,
and the Cholesky overwrites it in place, so peak is ONE float64 N x N plus the
memory-mapped float32 block:

    Kinney    13208^2 ->  1.4 GB   (fine on the 12 GB workstation)
    Cierjacks 28631^2 ->  6.6 GB   (cluster only: sbatch --mem=32G)

Blocks above `--max-n` are reported as skipped, never silently dropped.

  usage:
    corpus_absorbed.py 91_cross                          # whole corpus
    corpus_absorbed.py 91_cross --only 20743002 --max-n 30000   # Cierjacks alone
"""
import argparse, gc, io, os, sys, tempfile, time, zipfile
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

CH = os.environ.get("KIKA_CHI2_DIR", "/share_snc/snc/JuanMonleon/chi2")
SIGMA_STAT_FLOOR_REL = 0.01
DIAGONAL_FLOOR_REL = 0.001


def components(sub):
    y = sub.y_exp.to_numpy()
    sstat = np.maximum(sub.sigma_exp_stat.to_numpy(), SIGMA_STAT_FLOOR_REL * np.abs(y))
    D = np.maximum(sstat ** 2, (DIAGONAL_FLOOR_REL * np.abs(y)) ** 2)
    u = np.nan_to_num(sub.sigma_sys_indep_rel.to_numpy()) * y
    v = np.nan_to_num(sub.sigma_sys_dep_rel.to_numpy()) * y
    r = y - sub.y_eval.to_numpy()
    return D, u, v, r


def stream_block(npz_path, key, tmpdir):
    """Copy one member of the sidecar to a local .npy and mmap it, so the
    decompressed array is never held twice."""
    dst = os.path.join(tmpdir, f"{key.replace('@@','_')}.npy")
    with zipfile.ZipFile(npz_path).open(f"{key}.npy") as fh, open(dst, "wb") as out:
        while True:
            chunk = fh.read(1 << 24)
            if not chunk:
                break
            out.write(chunk)
    return np.load(dst, mmap_mode="r"), dst


def n_absorbed(block, D, u, v, r, probes, rng, pos=None):
    N = D.size
    ds = np.sqrt(D)
    B = np.column_stack([u / ds, v / ds])
    G = B.T @ B
    wg, Vg = np.linalg.eigh(G)
    keep = wg > 1e-14 * max(wg.max(), 1e-300)
    if keep.any():
        Qs = B @ (Vg[:, keep] / np.sqrt(wg[keep]))
        Ms = np.eye(int(keep.sum())) + Qs.T @ (B @ (B.T @ Qs))
        wm, Vm = np.linalg.eigh(Ms)
        corr_h = Vm @ np.diag(np.sqrt(wm) - 1.0) @ Vm.T
    else:
        Qs = corr_h = None

    def Ehalf(X):
        Y = X if Qs is None else X + Qs @ (corr_h @ (Qs.T @ X))
        return ds[:, None] * Y if Y.ndim == 2 else ds * Y

    sigma = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        src = block[i] if pos is None else block[pos[i]]
        row = np.asarray(src, dtype=np.float64)
        if pos is not None:
            row = row[pos]
        row += u[i] * u
        row += v[i] * v
        row[i] += D[i]
        sigma[i] = row
    sigma += sigma.T
    sigma *= 0.5

    cf, low = cho_factor(sigma, lower=True, overwrite_a=True, check_finite=False)
    del sigma
    gc.collect()

    chi2 = float(r @ cho_solve((cf, low), r, check_finite=False))
    m = min(probes, max(16, N))
    est = np.empty(m)
    step = 32                                   # probe in chunks: N x 32 doubles
    done = 0
    while done < m:
        k = min(step, m - done)
        Z = rng.integers(0, 2, size=(N, k)).astype(np.float64) * 2 - 1
        GZ = Ehalf(Z)
        est[done:done+k] = np.einsum(
            "ij,ij->j", GZ, cho_solve((cf, low), GZ, check_finite=False))
        done += k
        del Z, GZ
    return chi2, float(est.mean()), float(est.std(ddof=1) / np.sqrt(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--probes", type=int, default=128)
    ap.add_argument("--max-n", type=int, default=16000)
    ap.add_argument("--only", default=None, help="single experiment_id")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pq = f"{CH}/chi2_data_predictive_{a.tag}.parquet"
    npz = pq + ".eval_cov.npz"
    df = pd.read_parquet(pq)
    names = {n[:-4] for n in zipfile.ZipFile(npz).namelist()}
    rng = np.random.default_rng(7)

    rows, skipped = [], []
    tmpdir = tempfile.mkdtemp(prefix="corpus_absorbed_")
    try:
        for (lib, eid), sub in df.groupby(["library", "experiment_id"], observed=True):
            if a.only and str(eid) != a.only:
                continue
            key = f"{lib}@@{eid}"
            if key not in names:
                continue
            sub = sub.sort_values("_eval_pos")
            N = len(sub)
            if N > a.max_n:
                skipped.append((lib, eid, N))
                continue
            t = time.time()
            block, dst = stream_block(npz, key, tmpdir)
            p = sub._eval_pos.to_numpy(int)
            pos = None if (p == np.arange(N)).all() else p
            D, u, v, r = components(sub)
            chi2, n_free, se = n_absorbed(block, D, u, v, r, a.probes, rng, pos)
            del block
            gc.collect()
            os.unlink(dst)
            rows.append(dict(library=lib, experiment_id=eid,
                             author=sub.author.iloc[0], N=N,
                             chi2_v4_per_N=chi2 / N,
                             n_absorbed=N - n_free, n_absorbed_se=se,
                             frac_absorbed=(N - n_free) / N))
            print(f"  {lib:10s} {eid} {str(sub.author.iloc[0])[:14]:14s} N={N:6d} "
                  f"chi2/N={chi2/N:9.4f}  n_absorbed={N-n_free:9.1f} "
                  f"({100*(N-n_free)/N:5.2f} %)  [{time.time()-t:.0f}s]", flush=True)
    finally:
        for f in os.listdir(tmpdir):
            try:
                os.unlink(os.path.join(tmpdir, f))
            except OSError:
                pass
        os.rmdir(tmpdir)

    if not rows:
        print("no rows produced", file=sys.stderr)
        return 1
    out = pd.DataFrame(rows).sort_values(["experiment_id", "library"])
    suffix = f"_{a.only}" if a.only else ""
    dst = a.out or f"{CH}/n_absorbed_{a.tag}{suffix}.csv"
    out.to_csv(dst, index=False)
    print(f"\nwrote {dst}  ({len(out)} rows)")
    if skipped:
        print("SKIPPED (block larger than --max-n; not silently dropped):")
        for lib, eid, N in skipped:
            print(f"  {lib:10s} {eid}  N={N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
