#!/usr/bin/env python
"""Three ACE ensembles from the frozen Fe-56 deliverable, one covariance apart.

    joint  : sigma and a_l drawn together, MF33<->MF34 cross term included
    mf34   : the shape block alone   (sigma frozen at the host's MF3)
    mf33   : the magnitude block alone (a_l frozen at the deliverable's MF4)

All three are principal submatrices of the SAME assembled joint, drawn by the
same code with the same seed policy, so ``joint`` against ``mf34 (+) mf33`` is a
one-variable contrast and the variable is the cross term. That is the whole
point of running three: the cross term is what no released library carries and
what this evaluation ships, and a propagated difference has to be attributable.

WHAT THIS SCRIPT IS FOR
-----------------------
`docs/handoffs/handoff_2026-08-24.md` §7.2: the evaluation is closed and the
open item is the ensemble -- "generar el ensemble ACE desde la cinta elegida y
correr VENUS / ASPIS88 / PWR900". This produces that ensemble.

TWO TAPES, AND THE DIFFERENCE IS 20x THE DISK
---------------------------------------------
The deliverable is 570 MB, of which MF34 alone is 84 % and MF33 another 12 %.
ACER reads none of it. `perturb_ENDF_files` writes a whole copy of its source
per replica, so pointing this at the deliverable would cost 292 GB per ensemble
against 589 GB free.

So step 0 builds a BASE TAPE: same MF1/MF2/MF3/MF4/MF6..., no MF31-35, ~27 MB.
The covariance is read once from the deliverable; the base tape is what gets
written 512 times. `build_base_tape` proves MF3 and MF4 came through byte for
byte, and writes the report next to the tape.

RUN (cluster) -- prepare once, then the three CONCURRENTLY
----------------------------------------------------------
    JID=$(sbatch --parsable run_deliverable_samples.sh prepare)
    sbatch --dependency=afterok:$JID run_deliverable_samples.sh joint
    sbatch --dependency=afterok:$JID run_deliverable_samples.sh mf34
    sbatch --dependency=afterok:$JID run_deliverable_samples.sh mf33

``prepare`` builds the base tape and caches the assembled joint (~1.4 GB npz,
keyed on the tape's sha256) and takes the spectrum once. The three ensembles
then read those and never write them: two jobs creating one path is how a gate
comes to compare a half-written file. The dependency is what makes "run them in
parallel" safe rather than merely fast.

Smoke first, same shape, with ``8`` as the replica count.

⚠ Not a local script. Edits under `kika/scripts` do not reach the cluster by
themselves (deploy-cluster skill), and the library half needs a fresh WHEEL --
`/work` is not mounted, so copying files cannot deliver it. The assembled joint
is ~1.4 GB with an eigendecomposition on top, against a 12 GB shared box.

⚠ Output goes to `/SCRATCH`, which is NOT mounted in WSL. Nothing about the
output path can be verified from the workstation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kika.sampling.base_tape import COVARIANCE_MF, build_base_tape   # noqa: E402
from kika.sampling.joint_perturbation import (                       # noqa: E402
    JOINT_SETS,
    perturb_joint_mf33_mf34,
)

SHARE = "/share_snc/snc/JuanMonleon"
DEFAULT_ENDF = (f"{SHARE}/ENDF_samples/ENTREGABLE_Fe56_MF4_MF34_20260824/"
                "26-Fe-56_thiswork_MF4_MF33_MF34_a0cross.endf")
#: The ACE ensembles land on SCRATCH, beside the other MCNP libraries. The name
#: carries what a directory of 1 536 ACE files has to say for itself two years
#: from now: the nuclide, WHICH tape (``a0cross`` is the deliverable and the only
#: Fe-56 tape that ships the MF33<->MF34 cross term), how many replicas, and the
#: date the evaluation was frozen. ``/SCRATCH`` is not mounted in WSL, so nothing
#: here can be checked from the workstation — only from the cluster.
DEFAULT_OUT = ("/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/"
               "Fe56_a0cross_ens512_20260824")
DEFAULT_PENDF_CACHE = f"{SHARE}/cache/kika_pendf_cache"
DEFAULT_NJOY = "/soft_snc/NJOY/2016.78/bin/njoy"
DEFAULT_XSDIR = f"{SHARE}/xsdir_MCNPy/xsdir40-irdff2"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="which", choices=(*JOINT_SETS, "prepare"),
                   required=True,
                   help="which ensemble, or 'prepare' to build the base tape "
                        "and the joint cache and stop (run that first, alone)")
    p.add_argument("-n", "--n", type=int, default=512,
                   help="replicas (default 512)")
    p.add_argument("--endf", default=DEFAULT_ENDF,
                   help="the tape the COVARIANCE is read from (the deliverable)")
    p.add_argument("--base-tape", default=None,
                   help="the covariance-free tape the replicas are written "
                        "onto; built next to --out if omitted")
    p.add_argument("--rebuild-base", action="store_true",
                   help="rebuild the base tape even if it is already there")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"output root; one subdirectory per set. Budget ~14 GB "
                        f"of ENDF and/or ~25-50 GB of PENDF plus ~20-40 GB of "
                        f"ACE per 512-replica set (default {DEFAULT_OUT})")
    p.add_argument("--joint-cache", default=None,
                   help="npz to cache the assembled joint in, keyed on the "
                        "tape's sha256 (default <out>/joint_cache.npz)")

    g = p.add_argument_group("physics")
    g.add_argument("--mt", type=int, default=2)
    g.add_argument("--isotope", type=int, default=26056)
    g.add_argument("--l-max", type=int, default=6)

    g = p.add_argument_group("the draw")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--sampling-method", default="sobol",
                   choices=["sobol", "lhs", "random"])
    g.add_argument("--decomposition", default="svd", choices=["svd", "eigen"])
    g.add_argument("--psd-method", default="none",
                   choices=["none", "auto", "clip", "higham"],
                   help="'none' matches perturb_ENDF_files, so the mf34 set "
                        "reproduces today's draw exactly; the spectrum is "
                        "reported either way")
    g.add_argument("--null-tol", default="none",
                   help="'none' retains every direction (today's setting) or a "
                        "float. Changing it moves EVERY drawn column, so it "
                        "gets its own run and its own before/after -- see "
                        "docs/chi2-mf4/null_direction_truncation_decision.md")
    g.add_argument("--no-positivity", action="store_true",
                   help="skip the f(mu)>=0 projection on MF4 (do not: a "
                        "negative angular distribution is not a distribution)")

    g = p.add_argument_group("NJOY / ACE")
    g.add_argument("--no-ace", action="store_true",
                   help="stop after the tapes; for the smoke run only")
    g.add_argument("--njoy-exe", default=DEFAULT_NJOY)
    g.add_argument("--xsdir", default=DEFAULT_XSDIR)
    g.add_argument("--temperatures", type=float, nargs="+", default=[293.6])
    g.add_argument("--library-name", default="jeff40")
    g.add_argument("--pendf-cache-dir", default=DEFAULT_PENDF_CACHE,
                   help="a miss costs ~6 min of RECONR; never let this default "
                        "to the WSL temp dir")
    g.add_argument("--pendf-tolerance", type=float, default=1.0e-3)

    g = p.add_argument_group("run")
    g.add_argument("--nprocs", type=int, default=24)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--quiet", action="store_true")
    return p


def resolve_base_tape(args, out_root: Path, log=print) -> str:
    base = Path(args.base_tape) if args.base_tape else out_root / "base_tape.endf"
    report_path = base.with_suffix(".base_report.json")
    if base.exists() and not args.rebuild_base:
        log(f"[BASE] reusing {base} ({base.stat().st_size / 1e6:.1f} MB)")
        if not report_path.exists():
            # The report is the evidence that MF3/MF4 survived. A base tape
            # without one cannot be told apart from a tape someone edited.
            raise SystemExit(
                f"{base} exists but {report_path} does not, so nothing says "
                f"what it was built from or that its MF3/MF4 are the "
                f"deliverable's. Pass --rebuild-base, or point --base-tape at a "
                f"tape that has its report beside it."
            )
        return str(base)

    base.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"[BASE] stripping MF{list(COVARIANCE_MF)} from {args.endf} ...")
    report = build_base_tape(args.endf, str(base))
    report["seconds"] = round(time.time() - t0, 1)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"[BASE] {report['bytes_before'] / 1e6:.1f} MB -> "
        f"{report['bytes_after'] / 1e6:.1f} MB (x{report['shrink_factor']:.1f}) "
        f"in {report['seconds']}s; MF3/MF4 byte-identical")
    return str(base)


def prepare(args, out_root: Path, base_tape: str, cache: str, log=print) -> int:
    """Build what the three ensembles share, and report the object once.

    Separated out because the three sets run **concurrently**: they read the
    base tape and the joint cache, and two jobs creating the same path is how a
    gate ends up comparing a file that is still being written. It is also the
    only place the full joint's spectrum is worth taking — the cache is keyed on
    the tape's sha256, so all three would be measuring the same matrix.
    """
    from kika.sampling.joint_perturbation import joint_report, load_or_build_joint

    t0 = time.time()
    log(f"[PREP] assembling the joint from {args.endf}")
    blocks, index = load_or_build_joint(
        args.endf, mt=args.mt, isotope=args.isotope, l_max=args.l_max,
        require_cross=True, cache_path=cache)
    (_, joint), = blocks
    report = joint_report(joint, index)
    report["seconds"] = round(time.time() - t0, 1)
    report["base_tape"] = base_tape
    report["joint_cache"] = cache
    (out_root / "joint_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    log(f"[PREP] dim={report['dimension']} "
        f"(sigma {report['n_sigma']} + a_l {report['n_a']}), "
        f"cross orders {report['cross_orders']}")
    log(f"[PREP] lam_min/lam_max={report['lam_min_over_max']:.3e}  "
        f"max|rho|={report['max_abs_rho']:.9f}  "
        f"negative mass={report['negative_mass_fraction']}")
    log(f"[PREP] done in {report['seconds']}s -> {out_root}/joint_report.json")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out)
    out_dir = out_root if args.which == "prepare" else out_root / args.which
    out_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if args.quiet else print

    base_tape = resolve_base_tape(args, out_root, log=log)
    cache = args.joint_cache or str(out_root / "joint_cache.npz")
    null_tol = None if str(args.null_tol).lower() == "none" else float(args.null_tol)

    if args.which == "prepare":
        return prepare(args, out_root, base_tape, cache, log=log)

    # The three ensembles run concurrently, so they may NOT build the shared
    # inputs: two jobs writing one path is how a gate comes to compare a
    # half-written file. `prepare` builds them and the three depend on it.
    for path, what in ((base_tape, "base tape"), (cache, "joint cache")):
        if not Path(path).exists():
            raise SystemExit(
                f"the {what} ({path}) is missing. Run --set prepare first and "
                f"let it finish; the three ensembles share these and must not "
                f"race to create them."
            )

    log(f"[RUN ] set={args.which} n={args.n} seed={args.seed} -> {out_dir}")
    t0 = time.time()
    summary = perturb_joint_mf33_mf34(
        args.endf, base_tape, int(args.n),
        which=args.which, mt=args.mt, isotope=args.isotope, l_max=args.l_max,
        space="linear", decomposition_method=args.decomposition,
        sampling_method=args.sampling_method, psd_method=args.psd_method,
        null_tol=null_tol, seed=args.seed, joint_cache=cache,
        enforce_positivity=not args.no_positivity,
        generate_ace=not args.no_ace,
        pendf_cache_dir=args.pendf_cache_dir,
        pendf_tolerance=args.pendf_tolerance,
        njoy_exe=(None if args.no_ace else args.njoy_exe),
        ace_temperatures=args.temperatures,
        ace_library_name=args.library_name, xsdir_file=args.xsdir,
        output_dir=str(out_dir), nprocs=args.nprocs, dry_run=args.dry_run,
        verbose_diagnostics=0 if args.quiet else 1,
    )
    summary["argv"] = sys.argv[1:]
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["base_tape"] = base_tape
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    r = summary.get("joint_report_restricted") or summary["joint_report_full"]
    log(f"[RUN ] {args.which}: dim={r['dimension']} "
        f"cross={r['cross_orders']} lam_min/lam_max={r.get('lam_min_over_max')} "
        f"max|rho|={r.get('max_abs_rho')}")
    log(f"[RUN ] ACE {summary.get('n_ace_ok', 0)}/{summary.get('n_attempted', 0)}"
        f" in {summary['elapsed_s']}s -> {out_dir}/run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
