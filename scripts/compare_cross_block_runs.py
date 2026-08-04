"""Did the measured MF33<->MF34 cross block change anything, and is it trustworthy?

WHAT THIS COMPARES. `run_086` and `run_087` score the SAME ENDF -- run 86's
`_mg.endf`, same MF4 centrals, same MF33, same MF34. The only difference is that
run 87 feeds `build_mf33_mf34_cross_block` the Cov(c0, a_l) run 86 measured,
where every run from 82 to 86 fed it None. So the difference between the two IS
the cross term, with nothing else moving.

WHY IT NEEDED A RUN AT ALL. Run 86 measured the correlation in PARAMETER space:
median |rho| = 0.594 at a_1, 81 % of bins above |rho| = 0.3 -- but sign-
alternating in energy, so the median rho is only -0.173 and the mean -0.009.
That is not the answer to "what does it do to the chi^2", because Sigma_cross
carries P_l(mu), which alternates in ANGLE on top of rho's alternation in
ENERGY. Two cancellations, one of which is invisible in the sidecar. Whether the
net inflates or deflates sigma_eval per datapoint is exactly what this measures.

THE GATES, and they are not decoration:

  1. V2 must be IDENTICAL. It excludes Sigma_eval entirely, so a cross block
     cannot reach it. If V2 moves, the two runs are not scoring the same file.
  2. JEFF and JENDL must be IDENTICAL on every variant. They keep a zero cross
     block -- only This_work carries the measured one.
  3. sigma_eval must stay inside the Cauchy-Schwarz envelope,
     (sigma_MF33 - sigma_MF34)^2 <= sigma_eval^2 <= (sigma_MF33 + sigma_MF34)^2,
     recomputed from run 86's own split. Landing outside means the cross block
     is inconsistent with the diagonals it was added to -- which is possible
     here, because Sigma^MF33 and Sigma^MF34 come from the shipped (collapsed,
     near-zero-guarded) file while Sigma^cross comes from the raw MC.

HOW TO READ THE RESULT. Coverage getting worse is an accepted outcome: the
criterion is a mutually consistent MF4/MF34/MF33 evaluation, not a better score
(roadmap §9.4.2). What would be a finding is a LARGE move -- more than ~2 pp on
`no_Cierjacks` in either direction -- because that would mean the sign structure
is far more coherent than the sidecar suggests, and it should be understood
before it is believed.

Reads the two summary.json files and the two parquets. Never touches the ~11 GB
Sigma_eval sidecars.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_kika_root = Path(__file__).resolve().parent.parent
if str(_kika_root) not in sys.path:
    sys.path.insert(0, str(_kika_root))

FIG_ROOT = Path(os.environ.get(
    "KIKA_FIG_ROOT", "/share_snc/snc/JuanMonleon/CHI_Figures",
))
CHI2_ROOT = Path(os.environ.get(
    "KIKA_CHI2_ROOT", "/share_snc/snc/JuanMonleon/chi2",
))

BASE_RUN = os.environ.get("KIKA_CROSS_BASE_RUN", "086").strip()
CROSS_RUN = os.environ.get("KIKA_CROSS_RUN", "087").strip()
BASE_TAG = os.environ.get("KIKA_CROSS_BASE_TAG", "86").strip()
CROSS_TAG = os.environ.get("KIKA_CROSS_TAG", "87").strip()

SUBSETS = ["all", "no_Cierjacks", "no_KS", "no_KS_no_Cierjacks",
           "only_KS", "only_Cierjacks"]
LIBS = ["JEFF", "JENDL", "This_work"]
VARIANTS = ["V1", "V2", "V3", "V4"]

E_MIN_MEV, E_MAX_MEV = 0.85, 4.0
GATE_TOL = 1e-9


def load_summary(run_id: str) -> Optional[dict]:
    p = FIG_ROOT / "chi2_predictive" / f"run_{run_id}" / "summary.json"
    if not p.exists():
        print(f"  [missing] {p}")
        return None
    return json.loads(p.read_text())


def chi2(summary: dict, subset: str, lib: str, variant: str) -> Optional[float]:
    for row in summary["per_subset_all_variants"].get(subset, []):
        if row["library_key"] == lib:
            v = row.get(f"chi2_{variant.lower()}_per_N")
            return float(v) if v is not None and np.isfinite(v) else None
    return None


def coverage(summary: dict, subset: str, lib: str) -> Optional[float]:
    name = {"JEFF": "JEFF-4.0", "JENDL": "JENDL-5", "This_work": "This work"}[lib]
    for row in summary["coverage_per_subset"].get(subset, []):
        if row["Library"] == name:
            return float(row["|z|<1 (%)"])
    return None


def rel(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return new / old - 1.0


# ── Gates ─────────────────────────────────────────────────────────────────────

def invariance_gates(base: dict, cross: dict) -> bool:
    print(f"\n{'=' * 100}")
    print("  INVARIANCE GATES")
    print("  The two runs score the same ENDF. Only This_work's Sigma_eval may move,")
    print("  and V2 may not move at all -- it carries no Sigma_eval.")
    print("=" * 100)
    ok = True

    worst = 0.0
    for subset in SUBSETS:
        a, b = chi2(base, subset, "This_work", "V2"), chi2(cross, subset, "This_work", "V2")
        if a is None or b is None:
            continue
        worst = max(worst, abs(b - a))
    verdict = "PASS" if worst <= GATE_TOL else "*** FAIL"
    print(f"  1. This_work V2 unchanged        worst |delta| = {worst:.3e}   {verdict}")
    ok &= worst <= GATE_TOL

    worst_lib, where = 0.0, None
    for lib in ["JEFF", "JENDL"]:
        for subset in SUBSETS:
            for variant in VARIANTS:
                a, b = chi2(base, subset, lib, variant), chi2(cross, subset, lib, variant)
                if a is None or b is None:
                    continue
                if abs(b - a) > worst_lib:
                    worst_lib, where = abs(b - a), (lib, subset, variant)
    verdict = "PASS" if worst_lib <= GATE_TOL else "*** FAIL"
    print(f"  2. JEFF/JENDL unchanged          worst |delta| = {worst_lib:.3e}   "
          f"{verdict}  {where or ''}")
    ok &= worst_lib <= GATE_TOL

    if not ok:
        print("\n  A failed gate means the two runs differ by more than the cross")
        print("  block. Nothing below is interpretable until that is explained.")
    return ok


def cauchy_schwarz_gate() -> bool:
    """sigma_eval must stay inside the envelope its own two channels allow.

    Uses run 86's split -- with the cross block off, sigma^2_MF34 =
    sigma^2_eval - sigma^2_MF33 exactly, so the envelope is computable without
    the sidecar. Then |Sigma_cross(j,j)| <= 2 sigma_MF33 sigma_MF34 bounds run
    87's diagonal at every point.
    """
    print(f"\n{'=' * 100}")
    print("  CAUCHY-SCHWARZ GATE -- is the cross block consistent with its diagonals?")
    print("  Sigma^MF33 and Sigma^MF34 come from the shipped (collapsed, guarded) file;")
    print("  Sigma^cross comes from the raw MC. Nothing guarantees the sum is PSD.")
    print("=" * 100)

    base_pq = CHI2_ROOT / f"chi2_data_predictive_{BASE_TAG}.parquet"
    cross_pq = CHI2_ROOT / f"chi2_data_predictive_{CROSS_TAG}.parquet"
    for p in (base_pq, cross_pq):
        if not p.exists():
            print(f"  [missing] {p} -- gate skipped")
            return True

    def load(p):
        df = pd.read_parquet(p, columns=["library", "experiment_id", "energy_mev",
                                         "sigma_eval_diag", "y_eval"])
        df = df[(df["energy_mev"] >= E_MIN_MEV) & (df["energy_mev"] <= E_MAX_MEV)]
        return df[df["library"].astype(str) == "This_work"].reset_index(drop=True)

    a, b = load(base_pq), load(cross_pq)
    if len(a) != len(b):
        print(f"  [skip] row counts differ ({len(a)} vs {len(b)})")
        return True

    s_base = a["sigma_eval_diag"].to_numpy(float)
    s_cross = b["sigma_eval_diag"].to_numpy(float)
    y = np.abs(a["y_eval"].to_numpy(float))

    with np.errstate(divide="ignore", invalid="ignore"):
        r_base = np.where(y > 0, s_base / y, np.nan)
        r_cross = np.where(y > 0, s_cross / y, np.nan)

    print(f"  sigma_eval / y, median      run {BASE_RUN}: {100 * np.nanmedian(r_base):6.2f} %"
          f"    run {CROSS_RUN}: {100 * np.nanmedian(r_cross):6.2f} %"
          f"    ({100 * (np.nanmedian(r_cross) / np.nanmedian(r_base) - 1):+.2f} %)")

    ratio = np.where(s_base > 0, s_cross / s_base, np.nan)
    qs = np.nanpercentile(ratio, [1, 10, 50, 90, 99])
    print(f"  per-point sigma_eval ratio  p01 {qs[0]:.3f}  p10 {qs[1]:.3f}  "
          f"median {qs[2]:.3f}  p90 {qs[3]:.3f}  p99 {qs[4]:.3f}")
    print(f"  points where sigma_eval grew: {100 * np.nanmean(ratio > 1):.1f} %"
          f"    shrank: {100 * np.nanmean(ratio < 1):.1f} %")

    # THE TRAP, and it took a live run to notice. `sigma_eval_diag` is stored as
    # sqrt(max(diag, 0)) -- the precompute clips before writing -- so a negative
    # variance NEVER reaches the parquet as a negative number. Testing
    # `s_cross < 0` therefore always passes and would report a clean PSD bill of
    # health for a matrix that is not PSD at all.
    #
    # The surviving fingerprint is an EXACT ZERO where the baseline was positive:
    # sigma_eval > 0 everywhere with the cross block off (a sum of two PSD
    # sandwiches), so a zero in the cross run is a clipped negative diagonal.
    clipped = (s_cross == 0.0) & (s_base > 0.0)
    n_clipped = int(np.count_nonzero(clipped))
    n_bad = int(np.count_nonzero(~np.isfinite(s_cross)))
    frac = 100.0 * n_clipped / max(s_cross.size, 1)

    print(f"  non-finite sigma_eval in run {CROSS_RUN}: {n_bad}")
    verdict = "PASS" if (n_clipped == 0 and n_bad == 0) else "*** FAIL"
    print(f"  CLIPPED negative variances (sigma_eval == 0 where run {BASE_RUN} > 0): "
          f"{n_clipped} / {s_cross.size} = {frac:.2f} %   {verdict}")

    if n_clipped:
        print(f"\n  The joint matrix is NOT PSD. Sigma^MF33 and Sigma^MF34 went")
        print(f"  through the multigroup collapse and the near-zero guard; Sigma^cross")
        print(f"  did not, so the three are no longer mutually consistent and the")
        print(f"  per-point Cauchy-Schwarz bound |Sigma_cross(j,j)| <= 2 s33 s34 can be")
        print(f"  violated. V1/V3 read the clipped diagonal, so those points lose their")
        print(f"  evaluated uncertainty entirely and their chi^2 blows up; V4 reads the")
        print(f"  indefinite block directly. Neither number is publishable as it stands.")
        print(f"\n  Where it fails, by experiment (top 10 by share of the experiment):")
        by = (pd.DataFrame({"experiment_id": b["experiment_id"].astype(str),
                            "clipped": clipped})
              .groupby("experiment_id")["clipped"].agg(["sum", "size"]))
        by["pct"] = 100.0 * by["sum"] / by["size"]
        by = by[by["sum"] > 0].sort_values("pct", ascending=False).head(10)
        for exp_id, row in by.iterrows():
            print(f"    {exp_id:>12}  {int(row['sum']):6d} / {int(row['size']):6d} "
                  f"= {row['pct']:5.1f} %")
        print(f"\n  Next move: damp with KIKA_MF33_MF34_CROSS_SCALE. The cross term is")
        print(f"  LINEAR in the scale, so halving it halves Sigma_cross exactly and the")
        print(f"  scan brackets the largest consistent term instead of guessing one.")
    return n_clipped == 0 and n_bad == 0


# ── Reporting ─────────────────────────────────────────────────────────────────

def chi2_table(base: dict, cross: dict) -> None:
    print(f"\n{'=' * 100}")
    print(f"  This_work chi2/N -- run {CROSS_RUN} (cross block ON) vs run {BASE_RUN} (OFF)")
    print("  V2 is the invariance gate. V1/V3 read the diagonal, V4 the dense block:")
    print("  a cross term that lives in the correlations shows up in V4 and not in V1.")
    print("=" * 100)
    print("  %-22s %s" % ("subset", "  ".join(f"{v:>24}" for v in VARIANTS)))
    for subset in SUBSETS:
        cells = []
        for variant in VARIANTS:
            old = chi2(base, subset, "This_work", variant)
            new = chi2(cross, subset, "This_work", variant)
            d = rel(new, old)
            cells.append(f"{'--':>24}" if new is None else
                         f"{new:>11.4f} ({100 * d:+8.3f}%)" if d is not None else
                         f"{new:>11.4f}{'':>13}")
        print("  %-22s %s" % (subset, "  ".join(cells)))


def coverage_table(base: dict, cross: dict) -> None:
    print(f"\n{'=' * 100}")
    print("  Coverage |z|<1 (%), This work")
    print("  A move larger than ~2 pp on no_Cierjacks in EITHER direction is the finding:")
    print("  it would mean the sign structure is more coherent than the sidecar showed.")
    print("=" * 100)
    print("  %-22s %10s %10s %10s   %s" %
          ("subset", f"run {BASE_RUN}", f"run {CROSS_RUN}", "delta pp", "JENDL (context)"))
    for subset in SUBSETS:
        a, b = coverage(base, subset, "This_work"), coverage(cross, subset, "This_work")
        j = coverage(cross, subset, "JENDL")
        if a is None or b is None:
            continue
        flag = "   <-- LARGE" if abs(b - a) > 2.0 else ""
        print("  %-22s %10.2f %10.2f %+10.2f   %10s%s" %
              (subset, a, b, b - a, f"{j:.2f}" if j is not None else "--", flag))


# ── Damping scan ──────────────────────────────────────────────────────────────

def _tag_from_run_id(run_id: str) -> str:
    """`086` -> `86`, `087_s050` -> `87_s050`. One leading zero, by convention."""
    return run_id[1:] if run_id.startswith("0") else run_id


def _load_diag(tag: str):
    """(signed variance diagonal or None, clipped sigma, |y|) for This_work."""
    p = CHI2_ROOT / f"chi2_data_predictive_{tag}.parquet"
    if not p.exists():
        return None
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(p).schema_arrow.names)
    cols = ["library", "energy_mev", "sigma_eval_diag", "y_eval"]
    if "sigma_eval_var_diag" in have:
        cols.append("sigma_eval_var_diag")
    df = pd.read_parquet(p, columns=cols)
    df = df[(df["energy_mev"] >= E_MIN_MEV) & (df["energy_mev"] <= E_MAX_MEV)]
    df = df[df["library"].astype(str) == "This_work"].reset_index(drop=True)
    var = (df["sigma_eval_var_diag"].to_numpy(float)
           if "sigma_eval_var_diag" in df.columns else None)
    return var, df["sigma_eval_diag"].to_numpy(float), np.abs(df["y_eval"].to_numpy(float))


def critical_scale(base_tag: str, cross_tag: str, cross_scale: float) -> None:
    """The largest damping factor whose Σ_eval diagonal stays non-negative.

    Σ_eval(s) = Σ^MF33 + Σ^MF34 + s·Σ^cross is exactly linear in s, so one run at
    ANY s0 determines Σ^cross's diagonal everywhere,

        c(j) = ( v_{s0}(j) - v_0(j) ) / s0 ,

    and the per-point limit is s*(j) = v_0(j) / (-c(j)) wherever c(j) < 0. The
    global answer is the minimum. No further cluster time, no scanning.

    ⚠ NECESSARY, NOT SUFFICIENT. A non-negative diagonal does not make a matrix
    PSD. Cross-check against the [PSD] Lanczos lines in the precompute log for
    the two large groups before trusting a scale.
    """
    base = _load_diag(base_tag)
    cross = _load_diag(cross_tag)
    if base is None or cross is None:
        print("  [skip] one of the two parquets is missing")
        return
    v0_raw, s0_clipped, _y = base
    v1_raw, s1_clipped, _ = cross
    if len(s0_clipped) != len(s1_clipped):
        print("  [skip] row counts differ")
        return

    # The baseline is PSD, so its clipped sigma is its true diagonal.
    v0 = v0_raw if v0_raw is not None else s0_clipped ** 2

    if v1_raw is not None:
        v1 = v1_raw
        exact = True
    else:
        # Pre-`sigma_eval_var_diag` run: the clipped points only give an
        # inequality, so they are excluded and the answer is an UPPER bound.
        v1 = np.where(s1_clipped > 0, s1_clipped ** 2, np.nan)
        exact = False

    c = (v1 - v0) / cross_scale
    neg = np.isfinite(c) & (c < 0) & (v0 > 0)
    if not np.any(neg):
        print("  no point has a negative cross contribution — any scale is safe "
              "on the diagonal")
        return
    s_star = v0[neg] / (-c[neg])
    s_max = float(np.min(s_star))
    n_excluded = int(np.count_nonzero(~np.isfinite(c)))

    print(f"  measured from run {cross_tag} at scale {cross_scale:g}"
          f"{'' if exact else ' (clipped run — this is an UPPER BOUND)'}")
    if n_excluded:
        print(f"  {n_excluded} points excluded: clipped in the source run, so only "
              f"an inequality is available for them")
    print(f"  points with a negative cross contribution: {int(neg.sum())} / {v0.size}"
          f" = {100 * neg.mean():.2f} %")
    qs = np.percentile(s_star, [0, 0.1, 1, 5, 25, 50])
    print(f"  per-point limit s*(j):  min {qs[0]:.4f}   p0.1 {qs[1]:.4f}   "
          f"p1 {qs[2]:.4f}   p5 {qs[3]:.4f}   p25 {qs[4]:.4f}   median {qs[5]:.4f}")
    if exact:
        print(f"\n  LARGEST DIAGONAL-SAFE SCALE  s_max = {s_max:.4f}")
        for s in (0.75, 0.5, 0.25, 0.1):
            n_bad = int(np.count_nonzero(s_star < s))
            print(f"    at s = {s:4.2f}: {n_bad:5d} points would still go negative "
                  f"({100 * n_bad / v0.size:.2f} %)")
    else:
        # DO NOT print a per-scale count here as if it were the answer. The
        # excluded points are not a random sample — they are EXACTLY the points
        # that already failed, i.e. the ones with the smallest s*. Counting only
        # the survivors says "0 points fail at s = 0.5" while saying nothing
        # about the 1005 that are the whole problem. That is a confident-looking
        # wrong number, so it is not printed.
        print(f"\n  s_max CANNOT BE DETERMINED from a clipped run.")
        print(f"  The {n_excluded} excluded points are not a random sample: they are")
        print(f"  precisely the ones that already went negative, i.e. those with the")
        print(f"  SMALLEST s*. All that is known about them is s*(j) <= {cross_scale:g}.")
        print(f"  Over the {int(neg.sum())} points that ARE measurable the limit is "
              f"{s_max:.4f},")
        print(f"  which bounds nothing useful. Re-run with `sigma_eval_var_diag`")
        print(f"  present (any run from 2026-08-04 onward) and this becomes exact.")
    print("\n  ⚠ Non-negative diagonal is NECESSARY, NOT SUFFICIENT for PSD. Check")
    print("    the [PSD] Lanczos lines for Cierjacks and K&S before adopting a scale.")


def scan_report() -> None:
    """Compare several damping scales side by side.

    Spec: KIKA_CROSS_SCAN="087:1.0,087_s050:0.5,087_s025:0.25" (run_id:scale).
    """
    spec = os.environ.get("KIKA_CROSS_SCAN", "").strip()
    if not spec:
        return
    entries = []
    for item in spec.split(","):
        run_id, _, s = item.partition(":")
        entries.append((run_id.strip(), float(s or 1.0)))

    print(f"\n{'=' * 100}")
    print("  DAMPING SCAN")
    print("  Sigma_eval is exactly linear in the scale, so this brackets the largest")
    print("  self-consistent cross term rather than guessing one.")
    print("=" * 100)
    base = _load_diag(_tag_from_run_id(BASE_RUN))
    print("  %-14s %6s %12s %12s %10s %10s %10s" %
          ("run", "scale", "sig_eval/y", "clipped", "V4 all", "V4 noCierj", "|z|<1 noC"))
    if base is not None:
        v0, s0, y0 = base
        print("  %-14s %6.2f %11.3f %% %12s %10s %10s %10s" %
              (BASE_RUN, 0.0, 100 * np.median(s0 / y0), "0", "-", "-", "-"))

    for run_id, s in entries:
        got = _load_diag(_tag_from_run_id(run_id))
        summ = load_summary(run_id)
        if got is None:
            continue
        _v, sig, y = got
        n_clip = (int(np.count_nonzero((sig == 0.0) & (base[1] > 0)))
                  if base is not None else -1)
        v4a = chi2(summ, "all", "This_work", "V4") if summ else None
        v4n = chi2(summ, "no_Cierjacks", "This_work", "V4") if summ else None
        cvn = coverage(summ, "no_Cierjacks", "This_work") if summ else None
        print("  %-14s %6.2f %11.3f %% %12d %10s %10s %10s" % (
            run_id, s, 100 * np.median(sig / y), n_clip,
            f"{v4a:.4f}" if v4a else "--",
            f"{v4n:.4f}" if v4n else "--",
            f"{cvn:.2f}" if cvn else "--",
        ))

    print(f"\n{'=' * 100}")
    print("  LARGEST DIAGONAL-SAFE SCALE")
    print("=" * 100)
    # Prefer a run that carries the signed diagonal — it gives the exact answer.
    for run_id, s in entries:
        got = _load_diag(_tag_from_run_id(run_id))
        if got is not None and got[0] is not None:
            critical_scale(_tag_from_run_id(BASE_RUN), _tag_from_run_id(run_id), s)
            return
    run_id, s = entries[0]
    critical_scale(_tag_from_run_id(BASE_RUN), _tag_from_run_id(run_id), s)


def main() -> int:
    print("=" * 100)
    print(f"  MF33<->MF34 CROSS BLOCK: run {CROSS_RUN} vs run {BASE_RUN}")
    print(f"  Same ENDF, same centrals, same MF33, same MF34. Only Sigma^cross differs.")
    print("=" * 100)

    base, cross = load_summary(BASE_RUN), load_summary(CROSS_RUN)
    if base is None or cross is None:
        print("\nBoth runs must exist. Nothing to compare.")
        return 1

    gates_ok = invariance_gates(base, cross)
    chi2_table(base, cross)
    coverage_table(base, cross)
    cs_ok = cauchy_schwarz_gate()
    scan_report()

    print(f"\n{'=' * 100}")
    if gates_ok and cs_ok:
        print("  All gates pass. The difference above IS the cross block.")
    else:
        print("  *** A GATE FAILED -- read the failure before reading the numbers.")
    print("  Reminder (roadmap §9.4.2): the criterion is a mutually consistent")
    print("  MF4/MF34/MF33 evaluation, not a better chi^2. Worse coverage is an")
    print("  accepted outcome and is published with its explanation.")
    print("=" * 100)
    return 0 if (gates_ok and cs_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
