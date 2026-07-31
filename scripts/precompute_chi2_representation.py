"""
Precompute chi^2 data for the MF34 REPRESENTATION study — fine vs grouped, and
how many Legendre orders the file actually needs to carry.

This is `precompute_chi2_predictive.py` with two knobs bolted on and nothing
else touched. The forward model, the resolution window, the fold convention,
the MF33 handling and the Sigma_eval assembly are *literally the same code* —
this module imports them rather than restating them, so any difference in the
numbers is attributable to the two knobs alone.

Run it like any other precompute script (one entry in run_pyscript.sh); it
sweeps every requested mode itself. Set REPR_MODES to change the set:

    REPR_MODES="mg fine fine_cov3 fine_eval3"     # the default
    REPR_MODES=all                                # all eleven

Each ENDF file is parsed ONCE and reused across every mode that needs it. That
matters: the fine This_work MF34 is ~10428 parameters (1738 bins x 6 orders,
~734 MB on disk) and re-reading it per mode would dominate the runtime. The
truncations are applied to the already-parsed objects, which costs nothing.

The two questions
-----------------

**Q1 — what does multigroup collapse cost?**  The pipeline ships two files.
The fine one carries MF34 on the full ~1738-bin evaluation grid; the `_mg` one
carries it collapsed to ~598 groups chosen from the `l=1` correlation structure
(`scripts/multigroup_collapse.py`).  Collapsing is *irreversible* and it is a
modification of the result: it width-averages the coefficient means and then
applies a percentile variance compensation driven by `l=1` heterogeneity to
every order.  Where a high order changes sign inside a group, the grouped mean
can approach zero while the compensated variance stays finite.  Grouping can
always be redone downstream by a consumer; it can never be undone.  So the
burden of proof is on the collapse, and this study is that proof: run the same
chi^2 against both files and read the difference in V4.

Note the two files carry the *same* MF4 central values and, since
`MF33_MG_REPRESENTATION="fine"` (2026-07-28), the same MF33.  The fine-vs-mg
difference is therefore purely MF34, which is what makes the comparison clean.

**Q2 — do orders 4-6 earn their place?**  The fine MF34 is large because it is
quadratic in (bins x orders).  Dropping orders 5 and 6 is a 2.25x reduction in
the matrix; dropping 4-6 is 4x.  That is only worth discussing if the high
orders do not carry chi^2.  Two separate versions of the question, and they are
not the same:

    cov{3,4,5}    MF4 central stays at L=6, MF34 truncated to l <= L.
                  "we evaluated to 6 but only ship covariance for the first L"
                  -> moves V4 only; V2 is untouched by construction.

    eval{3,4,5}   BOTH the MF4 central and MF34 truncated to l <= L.
                  "we only evaluate to L at all"
                  -> moves V2 and V4.

`cov` is the file-size question.  `eval` is the physics question, and it is the
more interesting one: if V2 barely moves between eval3 and the L=6 reference,
the high orders are not doing evaluation work either.

Only This_work is truncated.  JEFF and JENDL stay at L=6 throughout as fixed
context, because the decision being made is about *our* file.  Set
TRUNCATE_LIBS=all to truncate every library instead, which answers the
different and more academic question of how much the high orders matter in
general.

How the truncation is done
--------------------------
Not by lowering L_MAX — that would change the forward model's array widths and
make the comparison partly an artifact of the code path.  Instead the *library
dictionaries* are transformed, which is exactly what shipping a smaller file
would mean:

    MF4 central   each per-energy Legendre coefficient array is sliced to
                  l <= L.  `interp_a_l_to_energy` zero-pads back to 6, so the
                  forward model runs at full width with a_4..a_6 = 0.

    MF34          sub-blocks with l_row > L or l_col > L are dropped from the
                  LegendreCovariance object.  `build_mf34_block` already skips
                  what is not there.

Both produce shallow views that share the underlying matrices with the parsed
original, so a mode costs no extra memory and no extra parsing.

`eval{L}` truncates both together on purpose.  A file cannot carry covariance
for a coefficient it does not ship, and with a *relative* MF34 the sandwich
multiplies each block by a_l — so keeping MF34 order 5 while zeroing a_5 would
contribute exactly nothing anyway.  Requesting MF34_L_MAX > MF4_L_MAX is
therefore rejected rather than silently producing the `eval` answer under a
`cov` label.

Output
------
`chi2_data_repr_82_<mode>.parquet` + `.eval_cov.npz` per mode, consumed by
`chi2_analysis_cluster.py` under the `repr_<mode>` methodology keys.  Modes
whose parquet already exists are skipped; delete it to force a rebuild.

Validation gate: mode `mg` is the run-82 `predictive` configuration reached by
a different code path.  Its V2/V4 must reproduce
`CHI_Figures/chi2_predictive/run_082/summary.json`.  If it does not, this
script is wrong and none of the other modes mean anything.  It is first in the
default set for that reason.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make BLAS single-threaded BEFORE numpy/scipy load.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_kika_root = Path(__file__).resolve().parent.parent
if str(_kika_root) not in sys.path:
    sys.path.insert(0, str(_kika_root))

import scripts.precompute_chi2_predictive as pred
from scripts.precompute_chi2_exfor_c0 import load_exfor
from scripts.precompute_chi2_library_c0 import (
    interp_a_l_to_energy,
    load_library_lib_c0,
)
from scripts.eval_covariance import build_eval_cov_for_groups, save_eval_cov
from scripts.tof_parameters import load_tof_parameters_file


# ── Mode table ────────────────────────────────────────────────────────────────
#
# mode -> (grid, mf4_l_max, mf34_l_max).  grid selects which This_work ENDF is
# read; it has no meaning for JEFF/JENDL, which ship one file each.

MODES: Dict[str, Tuple[str, int, int]] = {
    # Reference pair — Q1, the cost of grouping. Identical in every respect
    # except which This_work file is read.
    "mg":        ("mg",   6, 6),   # == run-82 `predictive`; the validation gate
    "fine":      ("fine", 6, 6),

    # Q2a — covariance orders only, central held at 6.
    "fine_cov5": ("fine", 6, 5),
    "fine_cov4": ("fine", 6, 4),
    "fine_cov3": ("fine", 6, 3),
    "mg_cov5":   ("mg",   6, 5),
    "mg_cov4":   ("mg",   6, 4),
    "mg_cov3":   ("mg",   6, 3),

    # Q2b — the whole evaluation truncated. Central and covariance together.
    "fine_eval5": ("fine", 5, 5),
    "fine_eval4": ("fine", 4, 4),
    "fine_eval3": ("fine", 3, 3),
}

# Default set: brackets both questions at their extremes for ~4 x 11 GB of
# sidecar. Fill in cov4/cov5/eval4/eval5 afterwards only if the extreme shows
# an effect worth resolving. `mg` is first because it is the validation gate.
DEFAULT_MODES = ["mg", "fine", "fine_cov3", "fine_eval3"]

_raw = os.environ.get("REPR_MODES", "").replace(",", " ").split()
if not _raw:
    REQUESTED_MODES: List[str] = list(DEFAULT_MODES)
elif len(_raw) == 1 and _raw[0].lower() == "all":
    REQUESTED_MODES = list(MODES)
else:
    REQUESTED_MODES = [m.strip().lower() for m in _raw]

_unknown = [m for m in REQUESTED_MODES if m not in MODES]
if _unknown:
    raise SystemExit(
        f"unknown repr mode(s) {_unknown} — pick from {sorted(MODES)} or 'all'"
    )

# "this_work" (default) or "all".
TRUNCATE_LIBS = os.environ.get("TRUNCATE_LIBS", "this_work").strip().lower()
if TRUNCATE_LIBS not in ("this_work", "all"):
    raise SystemExit(f"TRUNCATE_LIBS={TRUNCATE_LIBS!r} must be this_work|all")

for _m in REQUESTED_MODES:
    _g, _l4, _l34 = MODES[_m]
    if _l34 > _l4:
        raise SystemExit(
            f"mode {_m}: MF34_L_MAX={_l34} > MF4_L_MAX={_l4}. A relative MF34 "
            f"block for order l is multiplied by a_l in the sandwich, so "
            f"covariance above the central truncation contributes nothing and "
            f"the run would silently be an `eval` run."
        )

# This_work ENDF files. Both live in the MT1-repaired run-82 directory and
# carry the same MF4 and the same (fine) MF33; they differ only in MF34.
THIS_WORK_FILES = {
    "fine": f"{pred.THIS_WORK_DIR}/26-Fe-56g_nominal.endf",
    "mg":   f"{pred.THIS_WORK_DIR}/26-Fe-56g_nominal_mg.endf",
}


def output_parquet(mode: str) -> str:
    return (
        "/share_snc/snc/JuanMonleon/chi2/"
        f"chi2_data_repr_82_{mode}.parquet"
    )


# ── Truncation ────────────────────────────────────────────────────────────────

def truncate_mf4(library: Dict, l_max: int) -> Dict:
    """Return a shallow copy whose MF4 carries only orders 1..l_max.

    Each entry of ``coefficients`` is the per-energy a_l vector starting at
    a_1. Slicing it is exactly what shipping a lower-order MF4 would produce;
    ``interp_a_l_to_energy`` zero-pads back to whatever width the caller asks
    for, so downstream array shapes are unchanged.
    """
    if l_max >= max((len(c) for c in library["coefficients"]), default=0):
        return library
    out = dict(library)
    out["coefficients"] = [
        np.asarray(c, dtype=float)[:l_max] for c in library["coefficients"]
    ]
    return out


def truncate_mf34(mf34, l_max: int):
    """Return a LegendreCovariance keeping only blocks with both l <= l_max.

    ``None`` in, ``None`` out. Blocks are shared by reference with the input —
    the result is a view, so truncating a parsed fine MF34 costs no memory.
    """
    if mf34 is None:
        return None

    keep = [
        i for i in range(len(mf34.matrices))
        if int(mf34.l_rows[i]) <= l_max and int(mf34.l_cols[i]) <= l_max
    ]
    if len(keep) == len(mf34.matrices):
        return mf34

    out = type(mf34)()
    for i in keep:
        out.isotope_rows.append(mf34.isotope_rows[i])
        out.reaction_rows.append(mf34.reaction_rows[i])
        out.l_rows.append(mf34.l_rows[i])
        out.isotope_cols.append(mf34.isotope_cols[i])
        out.reaction_cols.append(mf34.reaction_cols[i])
        out.l_cols.append(mf34.l_cols[i])
        out.energy_grids.append(mf34.energy_grids[i])
        out.matrices.append(mf34.matrices[i])
        out.is_relative.append(mf34.is_relative[i])
        out.frame.append(mf34.frame[i])
    out.energy_unit = mf34.energy_unit
    out.legendre_coefficients = dict(
        getattr(mf34, "legendre_coefficients", {}) or {}
    )
    return out


def truncate_library(library: Dict, mf4_l_max: int, mf34_l_max: int) -> Dict:
    """Apply both truncations, returning a new dict that shares its arrays.

    The input is never mutated, so the same parsed library can back every mode.
    """
    out = dict(truncate_mf4(library, mf4_l_max))
    out["mf34"] = truncate_mf34(out.get("mf34"), mf34_l_max)
    return out


def _describe_mf34(mf34) -> str:
    if mf34 is None:
        return "none"
    n = len(mf34.matrices)
    if n == 0:
        return "0 blocks"
    orders = sorted({int(l) for l in mf34.l_rows} | {int(l) for l in mf34.l_cols})
    sizes = {int(np.asarray(m).shape[0]) for m in mf34.matrices}
    return (f"{n} blocks, orders {min(orders)}..{max(orders)}, "
            f"grid {'/'.join(str(s) for s in sorted(sizes))}")


# ── Loading, once ─────────────────────────────────────────────────────────────

def _resolve_this_work_file(grid: str) -> str:
    path = THIS_WORK_FILES[grid]
    if Path(path).is_file():
        return path
    d = Path(pred.THIS_WORK_DIR)
    listing = (
        "\n".join(f"    {p.name}" for p in sorted(d.glob("*.endf")))
        if d.is_dir() else "    <directory does not exist>"
    )
    raise SystemExit(
        f"grid={grid!r} needs {path}, which is not there.\n"
        f"  present in {d}:\n{listing}\n"
        f"  Adjust THIS_WORK_FILES at the top of this file."
    )


def _write_one_mode(
    mode: str,
    libraries: Dict[str, Dict],
    exfor_cache: Dict,
    tof_cache: Dict,
    energies_in_range: List[float],
) -> None:
    """Build and save the parquet + sidecar for one already-truncated set."""
    out_path = output_parquet(mode)

    all_rows: List[dict] = []
    for e_mev in energies_in_range:
        all_rows.extend(pred.build_rows_at_energy(
            e_mev, exfor_cache[e_mev], libraries, tof_cache,
        ))

    grid, mf4_l, mf34_l = MODES[mode]
    df = pd.DataFrame(all_rows)
    df["repr_mode"]     = mode
    df["repr_grid"]     = grid
    df["mf4_l_max"]     = mf4_l
    df["mf34_l_max"]    = mf34_l
    df["truncate_libs"] = TRUNCATE_LIBS

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def _a_l_lookup(lib_key: str, lib: dict, e_mev: float) -> np.ndarray:
        return interp_a_l_to_energy(lib, e_mev, pred.L_MAX)

    eval_cov = build_eval_cov_for_groups(
        df, libraries, _a_l_lookup, l_max=pred.L_MAX,
    )

    sigma_eval_diag = np.zeros(len(df), dtype=float)
    eval_pos = np.full(len(df), -1, dtype=np.int32)
    for (lib_key, exp_id), block in eval_cov.items():
        mask = (df["library"].astype(str) == lib_key) & \
               (df["experiment_id"].astype(str) == exp_id)
        idx = np.flatnonzero(mask.to_numpy())
        if idx.size != block.shape[0]:
            raise RuntimeError(
                f"Row count mismatch for ({lib_key}, {exp_id}): "
                f"{idx.size} parquet rows vs {block.shape[0]} cov rows"
            )
        sigma_eval_diag[idx] = np.sqrt(np.maximum(np.diag(block), 0.0))
        eval_pos[idx] = np.arange(idx.size, dtype=np.int32)
    df["sigma_eval_diag"] = sigma_eval_diag
    df["_eval_pos"] = eval_pos

    df.to_parquet(out_path, index=False)
    sidecar = out_path + ".eval_cov.npz"
    save_eval_cov(sidecar, eval_cov)

    print(f"  saved: {out_path}")
    print(f"    shape: {df.shape}; sidecar {len(eval_cov)} blocks")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    todo = [m for m in REQUESTED_MODES if not Path(output_parquet(m)).exists()]
    skipped = [m for m in REQUESTED_MODES if m not in todo]

    print("\n=== MF34 representation study ===")
    print(f"  requested : {REQUESTED_MODES}")
    if skipped:
        print(f"  skipping  : {skipped}  (parquet exists — delete to rebuild)")
    print(f"  to build  : {todo}")
    print(f"  truncating: {TRUNCATE_LIBS}")
    print(f"  fold mode : {pred.FOLD_MODE} (inherited, not a variable here)")
    if not todo:
        print("\nNothing to do.")
        return

    # ── Parse every input exactly once ──
    # JEFF and JENDL never change across modes. This_work is parsed once per
    # *grid* actually needed, not once per mode: the fine MF34 is the expensive
    # one (~734 MB, ~10428 parameters) and re-reading it per mode would
    # dominate the runtime.
    print("\nLoading libraries (once each)...")
    base_libraries: Dict[str, Dict] = {
        "JEFF":  load_library_lib_c0(pred.JEFF_FILE,  "JEFF-4.0"),
        "JENDL": load_library_lib_c0(pred.JENDL_FILE, "JENDL-5"),
    }

    grids_needed = sorted({MODES[m][0] for m in todo})
    this_work_by_grid: Dict[str, Dict] = {}
    for grid in grids_needed:
        path = _resolve_this_work_file(grid)
        this_work_by_grid[grid] = load_library_lib_c0(path, f"This work ({grid})")
        print(f"  This_work[{grid}] MF34: "
              f"{_describe_mf34(this_work_by_grid[grid].get('mf34'))}")

    # The fine-vs-mg comparison is only clean if MF33 is identical in both
    # files. It should be: MF33_MG_REPRESENTATION defaults to "fine" since
    # 2026-07-28, so the _mg file carries the ungrouped MF33. Check rather than
    # assume — if it ever flips, Q1 silently becomes MF33+MF34 instead of MF34.
    if len(this_work_by_grid) > 1:
        counts = {
            g: (len(lib["mf33_grid_ev"]) if lib.get("mf33_grid_ev") is not None
                else None)
            for g, lib in this_work_by_grid.items()
        }
        print(f"\n  This_work MF33 grid sizes by grid: {counts}")
        if len(set(counts.values())) > 1:
            print("  *** WARNING — the two files carry DIFFERENT MF33. Q1 is "
                  "then measuring MF33+MF34, not MF34 alone. Check "
                  "MF33_MG_REPRESENTATION in the run that produced them.")

    missing_mf33 = [
        k for k, v in list(base_libraries.items()) + list(this_work_by_grid.items())
        if v.get("mf33_rel_cov") is None
    ]
    if missing_mf33:
        print(f"\n[WARN] No MF33 for {missing_mf33} — their magnitude term will "
              f"be zero while the others carry one. That is an unfair "
              f"comparison; check the ENDF files before using the result.")

    # ── EXFOR and TOF, once ──
    try:
        tof_cache = load_tof_parameters_file(pred.TOF_PARAMETERS_FILE)
        print(f"\nLoaded TOF parameters: {len(tof_cache)} entries")
    except FileNotFoundError:
        tof_cache = {}
        print(f"\n[WARN] TOF parameters file not found: "
              f"{pred.TOF_PARAMETERS_FILE} — all experiments use defaults")

    exfor_cache, _sorted_e = load_exfor(
        db_path=pred.EXFOR_DB_PATH,
        target_zaids=pred.TARGET_ZAIDS,
        mt=pred.MT_NUMBER,
        supplementary_json_files=pred.SUPPLEMENTARY_JSON_FILES,
        exclude_experiments=pred.EXCLUDE_EXPERIMENTS,
    )
    energies_in_range = sorted(
        e for e in exfor_cache if pred.E_MIN_MEV <= e <= pred.E_MAX_MEV
    )
    print(f"Iterating {len(energies_in_range)} EXFOR energies in "
          f"[{pred.E_MIN_MEV}, {pred.E_MAX_MEV}] MeV")

    # ── Sweep. Truncation is a view over the parsed data, so a mode costs one
    #    forward-model pass and nothing else. ──
    for i, mode in enumerate(todo, 1):
        grid, mf4_l, mf34_l = MODES[mode]
        print(f"\n--- [{i}/{len(todo)}] {mode}: grid={grid}, "
              f"MF4 l<={mf4_l}, MF34 l<={mf34_l} ---")

        libraries = dict(base_libraries)
        libraries["This_work"] = this_work_by_grid[grid]

        targets = list(libraries) if TRUNCATE_LIBS == "all" else ["This_work"]
        for key in targets:
            before = _describe_mf34(libraries[key].get("mf34"))
            libraries[key] = truncate_library(libraries[key], mf4_l, mf34_l)
            after = _describe_mf34(libraries[key].get("mf34"))
            n_a = max((len(c) for c in libraries[key]["coefficients"]), default=0)
            if before != after or n_a < 6:
                print(f"  {key}: MF4 max L={n_a}; MF34 {before} -> {after}")

        _write_one_mode(mode, libraries, exfor_cache, tof_cache,
                        energies_in_range)

    built = [m for m in REQUESTED_MODES]
    print("\n=== done ===")
    print("Next, through the same runner:")
    print(f"  KIKA_CHI2_METHODOLOGIES="
          f"{','.join('repr_' + m for m in built)} python chi2_analysis_cluster.py")
    print("  python compare_representation_modes.py")
    if "mg" in built:
        print("\nCHECK THE VALIDATION GATE FIRST: repr_mg must reproduce "
              "CHI_Figures/chi2_predictive/run_082/summary.json. "
              "compare_representation_modes.py prints this automatically.")


if __name__ == "__main__":
    main()
