"""R2 step 7 — the gate that killed runs 89/90: diagnose what the FILE says.

Between "PSD in memory" and "PSD to a consumer" sit ENDF's 11-character floats
(~1e-7 relative on every entry) and the null slots the writer fills with zeros.
The joint sits ON the Cauchy-Schwarz boundary by construction, so that slop is
the same order as the headroom and has to be measured.

Read back with the pipeline's own reader (`mf34_cross_reader.read_mf34_split`,
the one the chi2 uses when `KIKA_MF33_MF34_CROSS_FROM_FILE=1`), rebuilt in
ABSOLUTE units on the live sub-block — `check_endf_roundtrip_psd.py`'s own
header explains why the relative-space verdict over-reports where a_l is zero.

    python r2_readback_gate.py --endf <a0cross tape> --joint <r2g dir> --c0-host <npy>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

AP = argparse.ArgumentParser()
AP.add_argument("--endf", required=True)
AP.add_argument("--joint", required=True)
AP.add_argument("--c0-host", required=True)
AP.add_argument("--kika", default="/home/MONLEON-JUAN/kika",
                help="On the cluster this is the deployed scripts' PARENT, i.e. "
                     "/share_snc/snc/JuanMonleon/EXFOR -- the gate imports "
                     "scripts.mf34_cross_reader and scripts.build_group_cross "
                     "from it, and those must be the deployed copies.")
AP.add_argument("--lmax", type=int, default=6)
args = AP.parse_args()

sys.path.insert(0, args.kika)
from scripts.mf34_cross_reader import read_mf34_split                # noqa: E402
from scripts.build_group_cross import diagnose                       # noqa: E402

J = Path(args.joint)
L = args.lmax
a_nom = np.load(J / "r2g_a_nom_group.npy")            # (n_gs, L)
shape_ev = np.load(J / "r2g_shape_ev.npy")
mag_ev = np.load(J / "r2g_mag_ev.npy")
C34_mem = np.load(J / "r2g_c34.npy")
Cx_mem = np.load(J / "r2g_cx.npy")
# ⚑ `r2_stage_tape_inputs.py` writes the SAME array under the pipeline's name,
# and a stage directory is what you have on the cluster -- not the r2g dir the
# joint was built in. Same bytes, two names; take whichever is there.
_c33 = J / "r2g_c33.npy"
if not _c33.exists():
    _c33 = J / "mf33_absolute_covariance.npy"
    if not _c33.exists():
        raise SystemExit(
            f"{J} has neither r2g_c33.npy nor mf33_absolute_covariance.npy; "
            f"one of them is the analytic magnitude block and the gate cannot "
            f"diagnose the joint without it.")
    print(f"[gate] magnitude block from {_c33.name} (the stage's name for it)")
C33_mem = np.load(_c33)
host = np.load(args.c0_host)
n_gs, n_gm = len(shape_ev) - 1, len(mag_ev) - 1
a_flat = a_nom.reshape(-1)
live = np.abs(a_flat) > 0
print(f"[gate] {n_gm} magnitude bins, {n_gs} shape groups, "
      f"{int(live.sum())}/{a_flat.size} live (group, order) slots")

print("[gate] parsing the written tape ...", flush=True)
res = read_mf34_split(args.endf, l_max=L, require_cross=True)
cross = {int(b['l']): b for b in res.cross}
print(f"[gate] a0 cross blocks found: {sorted(cross)}")

# ── the cross block, back on our layout: rows = magnitude, cols = (group, l) ──
# The writer embeds our magnitude rows inside the FILE's MF33 grid and writes
# the rest zero, so the block comes back taller than n_gm. Recover the offset by
# matching against what we wrote -- self-validating: a wrong offset cannot agree
# to 1e-7, and the search costs nothing.
Cx_file = np.zeros((n_gm, n_gs * L))
raw = {l1: np.asarray(b["matrix"], float) for l1, b in cross.items()
       if 1 <= l1 <= L}
n_rows_file = {m.shape[0] for m in raw.values()}
assert len(n_rows_file) == 1, n_rows_file
n_rows = n_rows_file.pop()

# what we handed the writer, in the same (relative-magnitude, relative-shape) form
Cx_w = np.divide(Cx_mem, host[:, None], out=np.zeros_like(Cx_mem),
                 where=host[:, None] > 0)
Cx_w_rel = np.zeros_like(Cx_w)
Cx_w_rel[:, live] = Cx_w[:, live] / a_flat[live]
sc = max(np.abs(Cx_w_rel).max(), 1e-300)
if n_rows == n_gm:
    off = 0
else:
    probe = Cx_w_rel[:, 0::L]
    cands = range(n_rows - n_gm + 1)
    off = min(cands, key=lambda i: np.abs(raw[1][i:i + n_gm] - probe).max())
    print(f"[gate] magnitude rows found at offset {off} of {n_rows} "
          f"(the write log said 431)")
for l1, m in raw.items():
    Cx_file[:, l1 - 1::L] = m[off:off + n_gm]
d = np.abs(Cx_file - Cx_w_rel).max() / sc
print(f"[gate] cross block, file vs memory: max|diff|/scale = {d:.3e}"
      f"   ({'serialisation only' if d < 1e-5 else 'LOOK AT THIS'})")

# ── diagnose in absolute units on the live sub-block ────────────────────────
keep = np.where(live)[0]
D = a_flat[keep]
print("[gate] rebuilding MF34's L>=1 marginals from the file ...", flush=True)
ang = res.mf34
# assemble the (group, order) square block from the per-(l,l') matrices
C34_file = np.zeros((n_gs * L, n_gs * L))
got = 0
for k in range(ang.num_matrices):
    lr, lc = int(ang.l_rows[k]), int(ang.l_cols[k])
    if not (1 <= lr <= L and 1 <= lc <= L):
        continue
    m = np.asarray(ang.matrices[k], float)
    if m.shape != (n_gs, n_gs):
        continue
    C34_file[lr - 1::L, lc - 1::L] = m
    if lr != lc:
        C34_file[lc - 1::L, lr - 1::L] = m.T
    got += 1
print(f"[gate] MF34 blocks assembled: {got}")

C34_abs = C34_file * np.outer(a_flat, a_flat)
Cx_abs_relmag = Cx_file * a_flat[None, :]          # magnitude side stays relative
C33_rel_mem = np.divide(C33_mem, np.outer(host, host),
                        out=np.zeros_like(C33_mem), where=np.outer(host, host) > 0)

rows = [
    diagnose("MEMORY   (what we handed the writer)",
             C33_rel_mem, C34_mem[np.ix_(keep, keep)],
             Cx_w[:, keep]),
    diagnose("FILE     (what a consumer reads)",
             C33_rel_mem, C34_abs[np.ix_(keep, keep)],
             Cx_abs_relmag[:, keep]),
]
cols = ["rank33", "rank34", "sigma_max(K)", "leak_null34", "lam_min",
        "lam_min_norm"]
print()
print(f"{'':40s}" + "".join(f"{c:>16s}" for c in cols))
for r in rows:
    print(f"{r['case'][:39]:40s}"
          + "".join(f"{r[c]:16.6g}" if isinstance(r.get(c), float)
                    else f"{str(r.get(c)):>16s}" for c in cols))
print("""
  The FILE row is the one that matters: it is the object the chi2 folds and the
  one an evaluator would receive. sigma_max(K) <= 1 and lam_min at round-off is
  the pass. MF33 is taken from memory in both rows on purpose -- it goes onto
  the tape through `rebuild_mf33`, whose own splice reports -1.27e-07 for the
  shipped file too, so mixing it in here would price someone else's step.""")
