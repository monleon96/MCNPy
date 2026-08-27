#!/usr/bin/env python3
"""Re-run the NJOY half of a perturbation set at a different temperature.

The perturbation and the NJOY processing are two separate halves of the same
pipeline, and only the second one carries a temperature. When a set was
processed at the wrong temperature, the samples themselves are still good: the
perturbed ENDF (MF4/MF34) and the perturbed PENDF (MF3/MF33) are
temperature-independent tapes, and the draws are already on disk. What has to
be redone is the NJOY chain -- broadr/heatr/purr/gaspr/acer -- and the xsdir
entries that point at its output. Nothing is re-sampled, so the ensemble stays
the one that was drawn.

WHICH ROUTE. A set is processed the way it was built, and that is read off the
directory rather than assumed, because feeding the wrong tape to NJOY produces
an ACE that looks perfectly healthy and silently carries no perturbation at
all:

    endf/ and pendf/   joint MF33+MF34   perturbed ENDF + perturbed PENDF
    pendf/ only        MF33 (XS)         original ENDF + perturbed PENDF
    endf/ only         MF34 (LEG)        perturbed ENDF, RECONR from it

The MF33 routes MUST go through the perturbed PENDF: the perturbation lives in
the PENDF, and running RECONR on the ENDF instead would reconstruct the
unperturbed cross sections. The cached PENDF is RECONR output (0 K), so
broadening it to the target temperature is the ordinary NJOY path and not an
approximation.

THE TEMPERATURE IS READ FROM THE FILE, NOT FROM ITS NAME. An ACE's extension
(`.02c`, `.06c`) is a label the writer chooses; the temperature that MCNP
transports with is the kT on the header's first line. A set processed at the
wrong temperature and labelled with the right extension is exactly the failure
this script exists to undo, and it is invisible from a directory listing. The
inventory therefore stats and reads every ACE, and says how many sit at which
temperature.

Idempotent and resumable: a sample whose ACE is already at the target
temperature is skipped unless `--force`, so an interrupted pass is continued by
running it again. The ACE and its xsdir entry are overwritten in place, under
the same file names the set already uses, so nothing downstream has to be
repointed.

Usage
-----
    python reprocess_ace.py <set>                      # inventario, no toca nada
    python reprocess_ace.py <set> --apply --limit 2    # dos muestras, para mirar
    python reprocess_ace.py <set> --apply              # las 512
    python reprocess_ace.py <set> --verify-only        # solo comprueba lo que hay

`<set>` is the perturbation output directory -- the one holding `endf/`,
`pendf/`, `ace/`, `xsdir/` -- not the MCNP run directory.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Boltzmann in MeV/K, the constant NJOY writes kT with on the ACE header.
K_BOLTZMANN_MEV = 8.617333262e-11
#: Two ACE temperatures are the same one when they agree to this many kelvin.
#: NJOY round-trips kT through a 5-digit field, which is ~0.01 K at 600 K.
TEMP_TOL_K = 1.0

DEFAULT_NJOY = "/soft_snc/NJOY/2016.78/bin/njoy"
DEFAULT_NJOY_VERSION = "NJOY 2016.78"

#: `<zaid>.<ext>` -- what the pipelines write when they are given an extension.
ACE_NAME_NEW = re.compile(r"^(\d+)\.(\d{2}[a-z])$")
#: `<zaid*10>_<lib>_<NNNN>.<ext>` -- the older, extension-less naming.
ACE_NAME_LEGACY = re.compile(r"^(\d+)_(\d+)_(\d{4})\.(\d{2}[a-z])$")
SAMPLE_DIR = re.compile(r"^\d{4}$")


# ---------------------------------------------------------------------------
# reading what is on disk
# ---------------------------------------------------------------------------

def ace_temperature_K(path: str) -> Optional[float]:
    """Kelvin from an ACE file's header, or None when it cannot be read.

    ACE 1.0 puts `zaid.ext awr kT date` on line 1. ACE 2.0 moves that to line 2
    behind a version token. Only the first two lines are read: the point is to
    do this for every sample in a set without paying for the XSS array.
    """
    try:
        with open(path, "r", errors="ignore") as fh:
            first = fh.readline()
            second = fh.readline()
    except OSError:
        return None

    parts = first.split()
    if parts and parts[0][:1].isdigit() and "." in parts[0] and len(parts) >= 3:
        kt_token = parts[2]                       # ACE 1.0
    elif parts and parts[0].startswith("2."):
        second_parts = second.split()             # ACE 2.0
        if len(second_parts) < 2:
            return None
        kt_token = second_parts[1]
    else:
        return None

    try:
        return float(kt_token) / K_BOLTZMANN_MEV
    except ValueError:
        return None


def sample_dirs(root: str) -> List[str]:
    """The `NNNN` directories under `root`, sorted."""
    try:
        return sorted(d for d in os.listdir(root) if SAMPLE_DIR.match(d))
    except OSError:
        return []


def find_zaid_dir(set_dir: str, kind: str) -> Optional[Tuple[str, int]]:
    """`(<set>/<kind>/<zaid>, zaid)` when the set has exactly one isotope.

    The sets this script is for are single-isotope by construction. More than
    one is not handled rather than half-handled: the caller is told to pass
    --zaid instead of the script picking one.
    """
    root = os.path.join(set_dir, kind)
    if not os.path.isdir(root):
        return None
    zaids = sorted(d for d in os.listdir(root) if d.isdigit())
    if len(zaids) != 1:
        return None
    return os.path.join(root, zaids[0]), int(zaids[0])


def tape_for_sample(zaid_root: str, sample: str, suffix: Optional[str] = None) -> Optional[str]:
    """The single tape inside `<zaid_root>/<sample>/`, whatever it is called.

    The extension is not assumed: the ENDF sets carry `.endf` or `.txt`
    depending on what the source file was called, and the PENDF sets carry
    `.pendf`. Summary sidecars written next to the tape are excluded.
    """
    d = os.path.join(zaid_root, sample)
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return None
    for n in names:
        p = os.path.join(d, n)
        if not os.path.isfile(p):
            continue
        if n.endswith((".parquet", ".csv", ".json", ".log")):
            continue
        if suffix is not None and not n.endswith(suffix):
            continue
        return p
    return None


def detect_ace_naming(set_dir: str, samples: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """`(style, extension)` of the ACE files a set already carries.

    style is 'new' (`<zaid>.<ext>`, written when the pipeline is given an
    extension) or 'legacy' (`<zaid*10>_<lib>_<NNNN>.<ext>`). Reusing the style
    the set already has is what keeps the rewrite in place, so the master xsdir
    lines and the MCNP inputs stay valid.
    """
    for sample in samples[:8]:
        d = os.path.join(set_dir, "ace", sample)
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names:
            m = ACE_NAME_LEGACY.match(n)
            if m:
                return "legacy", m.group(4)
            m = ACE_NAME_NEW.match(n)
            if m:
                return "new", m.group(2)
    return None, None


def ace_files_for_sample(set_dir: str, sample: str) -> List[str]:
    d = os.path.join(set_dir, "ace", sample)
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return []
    return [os.path.join(d, n) for n in names
            if (ACE_NAME_NEW.match(n) or ACE_NAME_LEGACY.match(n))
            and os.path.isfile(os.path.join(d, n))]


def master_xsdir_base(set_dir: str) -> Optional[str]:
    """The per-replica master xsdir template, with its `_NNNN` stripped.

    `create_xsdir_files_for_ace` appends the sample tag back on, and prefers the
    replica's existing master as the source when there is one -- so passing the
    stripped path updates each replica's own file in place and keeps whatever
    else was done to it (the DPA-600 and IRDFF-II repairs, for one).
    """
    d = os.path.join(set_dir, "xsdir")
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return None
    for n in names:
        parts = n.split("_")
        if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
            return os.path.join(d, n.rsplit("_", 1)[0])
    return None


def xsdir_entry(master_path: str, zaid: int, ext: str) -> Optional[str]:
    """The `<zaid>.<ext>` line of a master xsdir, or None."""
    want = f"{zaid}.{ext}"
    try:
        with open(master_path, "r", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.split(None, 1)[:1] == [want]:
                    return stripped
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# the set, as discovered
# ---------------------------------------------------------------------------

class SetLayout:
    def __init__(self, set_dir: str, zaid: Optional[int]):
        self.set_dir = os.path.abspath(set_dir)
        self.endf_root = self.endf_zaid = None
        self.pendf_root = self.pendf_zaid = None

        found = find_zaid_dir(self.set_dir, "endf")
        if found:
            self.endf_root, self.endf_zaid = found
        found = find_zaid_dir(self.set_dir, "pendf")
        if found:
            self.pendf_root, self.pendf_zaid = found

        self.zaid = zaid or self.pendf_zaid or self.endf_zaid

        if self.endf_root and self.pendf_root:
            self.route = "pendf-pair"
        elif self.pendf_root:
            self.route = "pendf-only"
        elif self.endf_root:
            self.route = "endf-only"
        else:
            self.route = None

        endf_samples = sample_dirs(self.endf_root) if self.endf_root else []
        pendf_samples = sample_dirs(self.pendf_root) if self.pendf_root else []
        if self.route == "pendf-pair":
            self.samples = sorted(set(endf_samples) & set(pendf_samples))
            self.unpaired = sorted(set(endf_samples) ^ set(pendf_samples))
        else:
            self.samples = pendf_samples or endf_samples
            self.unpaired = []

        self.ace_style, self.ace_ext_on_disk = detect_ace_naming(self.set_dir, self.samples)
        self.master_base = master_xsdir_base(self.set_dir)


def describe(layout: SetLayout, target_K: float, ext: str) -> Counter:
    """Print the inventory and return the temperature histogram of the ACE."""
    print(f"\n-- el conjunto: {layout.set_dir}")
    if layout.route is None:
        print("   ⛔ no hay ni endf/ ni pendf/ aqui. Esto no es un directorio de")
        print("      salida de perturbacion; sera el directorio de la run de MCNP.")
        return Counter()

    route_label = {
        "pendf-pair": "MF33+MF34 conjunto  (ENDF perturbado + PENDF perturbado)",
        "pendf-only": "MF33 / XS           (ENDF original + PENDF perturbado)",
        "endf-only":  "MF34 / LEG          (ENDF perturbado, RECONR desde el)",
    }[layout.route]
    print(f"   ruta               : {route_label}")
    print(f"   ZAID               : {layout.zaid}")
    print(f"   muestras           : {len(layout.samples)}")
    if layout.unpaired:
        print(f"   ⚠ sin pareja       : {len(layout.unpaired)} muestra(s) tienen "
              f"solo una de las dos cintas: {layout.unpaired[:5]}")
    if layout.ace_style:
        print(f"   ACE en disco       : estilo {layout.ace_style}, extension .{layout.ace_ext_on_disk}")
    else:
        print("   ACE en disco       : ninguno encontrado (se creara)")
    print(f"   xsdir maestro      : {layout.master_base or '⛔ no encontrado en <set>/xsdir/'}")

    # The temperature audit. This is the measurement the whole job rests on:
    # if these already read 600 K there is nothing to redo.
    print(f"\n-- temperatura REAL de los ACE (leida de la cabecera, no del nombre)")
    hist: Counter = Counter()
    unreadable = []
    for sample in layout.samples:
        files = ace_files_for_sample(layout.set_dir, sample)
        if not files:
            hist["(sin ACE)"] += 1
            continue
        for path in files:
            T = ace_temperature_K(path)
            if T is None:
                unreadable.append(path)
                hist["(ilegible)"] += 1
            else:
                hist[f"{T:.1f} K"] += 1
    for key, n in sorted(hist.items()):
        mark = "  ← el objetivo" if key.startswith(f"{target_K:.1f} ") else ""
        print(f"   {key:>12} : {n:5d} fichero(s){mark}")
    if unreadable:
        print(f"   primeros ilegibles : {unreadable[:3]}")

    if layout.ace_ext_on_disk and layout.ace_ext_on_disk != ext:
        print(f"\n   ⚠ LA EXTENSION CAMBIA: en disco .{layout.ace_ext_on_disk}, "
              f"a {target_K:g} K corresponde .{ext}.")
        print("     El ACE nuevo se llamara distinto y el xsdir declarara otro ZAID,")
        print(f"     asi que los inputs de MCNP que pidan {layout.zaid}.{layout.ace_ext_on_disk}")
        print(f"     hay que cambiarlos a {layout.zaid}.{ext} — o pasar")
        print(f"     --extension {layout.ace_ext_on_disk} para conservar el nombre.")
    return hist


# ---------------------------------------------------------------------------
# doing it
# ---------------------------------------------------------------------------

def _worker(job: Dict) -> Tuple[str, Dict]:
    """One sample, one temperature. Runs in a Pool worker."""
    from kika.sampling.endf_perturbation import _process_njoy_for_sample
    from kika.sampling.pendf_perturbation import _stage_b_run_njoy_for_pair

    sample = job["sample"]
    try:
        if job["route"] == "endf-only":
            res = _process_njoy_for_sample(
                out_endf=job["endf_path"],
                sample_index=job["sample_index"],
                njoy_exe=job["njoy_exe"],
                temperatures=[job["temperature"]],
                library_name=job["library_name"],
                njoy_version=job["njoy_version"],
                output_dir=job["set_dir"],
                xsdir_file=job["xsdir_file"],
                extensions=job["extensions"],
            )
        else:
            res = _stage_b_run_njoy_for_pair(
                endf_path=job["endf_path"],
                pendf_path=job["pendf_path"],
                sample_index=job["sample_index"],
                zaid=job["zaid"],
                njoy_exe=job["njoy_exe"],
                ace_temperatures=[job["temperature"]],
                ace_extensions=job["extensions"],
                ace_library_name=job["library_name"],
                ace_njoy_version=job["njoy_version"],
                xsdir_file=job["xsdir_file"],
                output_dir=job["set_dir"],
                keep_njoy_io=job["keep_njoy_io"],
            )
        return sample, res
    except Exception as e:                       # a worker must not take the pass down
        return sample, {"success": False, "temperatures_processed": [],
                        "errors": [f"{type(e).__name__}: {e}"], "warnings": []}


def build_jobs(layout: SetLayout, args, ext: str, source_endf: Optional[str]) -> List[Dict]:
    jobs = []
    for sample in layout.samples:
        idx = int(sample) - 1

        if not args.force:
            # Resume on the temperature, not on the file name: the whole point
            # is that a file with the right name can hold the wrong data.
            done = any(
                (T := ace_temperature_K(p)) is not None
                and abs(T - args.temperature) <= TEMP_TOL_K
                for p in ace_files_for_sample(layout.set_dir, sample)
            )
            if done:
                continue

        if layout.route == "pendf-only":
            endf_path = source_endf
        else:
            endf_path = tape_for_sample(layout.endf_root, sample)
        pendf_path = (tape_for_sample(layout.pendf_root, sample, ".pendf")
                      if layout.route != "endf-only" else None)

        if endf_path is None or (layout.route != "endf-only" and pendf_path is None):
            print(f"   ⚠ {sample}: falta una cinta, se salta")
            continue

        jobs.append({
            "route": layout.route,
            "sample": sample,
            "sample_index": idx,
            "zaid": layout.zaid,
            "endf_path": endf_path,
            "pendf_path": pendf_path,
            "set_dir": layout.set_dir,
            "njoy_exe": args.njoy,
            "temperature": args.temperature,
            "extensions": [ext] if layout.ace_style != "legacy" else None,
            "library_name": args.library,
            "njoy_version": args.njoy_version,
            "xsdir_file": layout.master_base,
            "keep_njoy_io": args.keep_njoy_io,
        })
    return jobs


def verify(layout: SetLayout, target_K: float, ext: str, samples: List[str]) -> int:
    """Check what landed, not what was asked for. Returns the number of faults.

    Two things are checked per sample, because either one alone lets a bad run
    through: the ACE really carries the target temperature, and the replica's
    master xsdir points at that file. MCNP resolves a relative xsdir entry
    against the directory the xsdir sits in, so that is how it is resolved here.
    """
    print(f"\n-- comprobacion de lo que hay en disco")
    bad_ace = bad_xsdir = 0
    first_bad: List[str] = []

    for sample in samples:
        files = ace_files_for_sample(layout.set_dir, sample)
        at_target = [p for p in files
                     if (T := ace_temperature_K(p)) is not None
                     and abs(T - target_K) <= TEMP_TOL_K]
        if not at_target:
            bad_ace += 1
            if len(first_bad) < 5:
                first_bad.append(f"{sample}: ningun ACE a {target_K:g} K")
            continue

        if not layout.master_base:
            continue
        master = f"{layout.master_base}_{sample}"
        entry = xsdir_entry(master, layout.zaid, ext)
        if entry is None:
            bad_xsdir += 1
            if len(first_bad) < 5:
                first_bad.append(f"{sample}: el maestro no declara {layout.zaid}.{ext}")
            continue
        ref = entry.split()[2]
        resolved = os.path.normpath(os.path.join(os.path.dirname(master), ref))
        T = ace_temperature_K(resolved) if os.path.exists(resolved) else None
        if T is None or abs(T - target_K) > TEMP_TOL_K:
            bad_xsdir += 1
            if len(first_bad) < 5:
                first_bad.append(f"{sample}: el maestro apunta a {ref} "
                                 f"({'no existe' if T is None else f'{T:.1f} K'})")

    label = f"con ACE a {target_K:g} K"
    print(f"   muestras comprobadas               : {len(samples)}")
    print(f"   {label:<34} : {len(samples) - bad_ace}")
    if bad_ace:
        print(f"   ⛔ sin ACE a la temperatura        : {bad_ace}")
    if layout.master_base:
        print(f"   con el maestro apuntando bien      : {len(samples) - bad_ace - bad_xsdir}")
        if bad_xsdir:
            print(f"   ⛔ maestro mal o sin la entrada    : {bad_xsdir}")
    for line in first_bad:
        print(f"      {line}")
    return bad_ace + bad_xsdir


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("set_dir", help="directorio de salida de la perturbacion "
                                   "(el que tiene endf/ pendf/ ace/ xsdir/)")
    p.add_argument("--temperature", type=float, default=600.0,
                   help="temperatura objetivo en K (por defecto 600)")
    p.add_argument("--extension", default=None,
                   help="extension ACE a escribir; por defecto la que corresponde "
                        "a la temperatura (600 K -> 06c)")
    p.add_argument("--zaid", type=int, default=None,
                   help="ZAID ENDF (26056 para Fe-56); por defecto se deduce")
    p.add_argument("--library", default="jeff40",
                   help="nombre de biblioteca para el titulo y el sufijo (jeff40)")
    p.add_argument("--source-endf", default=None,
                   help="ENDF original; obligatorio en la ruta MF33 sin endf/")
    p.add_argument("--njoy", default=DEFAULT_NJOY, help=f"ejecutable NJOY ({DEFAULT_NJOY})")
    p.add_argument("--njoy-version", default=DEFAULT_NJOY_VERSION)
    p.add_argument("--nprocs", type=int, default=None,
                   help="procesos en paralelo; por defecto SLURM_CPUS_PER_TASK")
    p.add_argument("--limit", type=int, default=None,
                   help="procesar solo las N primeras muestras pendientes")
    p.add_argument("--force", action="store_true",
                   help="rehacer tambien las que ya estan a la temperatura objetivo")
    p.add_argument("--keep-njoy-io", action="store_true",
                   help="conservar los input/output de NJOY por muestra")
    p.add_argument("--apply", action="store_true",
                   help="ejecutar; sin esto solo se inventaria")
    p.add_argument("--verify-only", action="store_true",
                   help="solo comprobar lo que ya hay, sin lanzar NJOY")
    args = p.parse_args()

    if not os.path.isdir(args.set_dir):
        print(f"⛔ no existe: {args.set_dir}")
        return 2

    ext = args.extension
    if ext is None:
        from kika._constants import K_TO_SUFFIX
        match = min(K_TO_SUFFIX, key=lambda k: abs(k - args.temperature))
        if abs(match - args.temperature) > TEMP_TOL_K:
            print(f"⛔ {args.temperature} K no tiene extension conocida "
                  f"({sorted(K_TO_SUFFIX)}). Pasa --extension.")
            return 2
        ext = K_TO_SUFFIX[match].lstrip(".") + "c"

    layout = SetLayout(args.set_dir, args.zaid)
    describe(layout, args.temperature, ext)
    if layout.route is None:
        return 2

    if args.verify_only:
        return 1 if verify(layout, args.temperature, ext, layout.samples) else 0

    if layout.route == "pendf-only" and not args.source_endf:
        print("\n⛔ esta ruta (solo pendf/) necesita el ENDF ORIGINAL: la "
              "perturbacion vive en el PENDF y NJOY lo empareja con la cinta "
              "de partida. Pasa --source-endf.")
        return 2
    if args.source_endf and not os.path.isfile(args.source_endf):
        print(f"⛔ --source-endf no existe: {args.source_endf}")
        return 2
    if not os.path.isfile(args.njoy):
        print(f"⛔ NJOY no esta en {args.njoy}")
        return 2
    if not layout.master_base:
        print("\n⚠ no hay maestro por replica en <set>/xsdir/: se escribiran los "
              "fragmentos por muestra pero ningun xsdir maestro se actualizara.")

    jobs = build_jobs(layout, args, ext, args.source_endf)
    if args.limit is not None:
        jobs = jobs[:args.limit]

    print(f"\n-- trabajo")
    skipped = len(layout.samples) - len(jobs) if args.limit is None else "—"
    label = f"ya a {args.temperature:g} K, se saltan"
    print(f"   muestras pendientes                : {len(jobs)}"
          f"{' (limitado)' if args.limit is not None else ''}")
    print(f"   {label:<34} : {skipped}")
    print(f"   NJOY                               : {args.njoy}")
    print(f"   escribe                            : "
          f"<set>/ace/NNNN/*.{ext}  +  <set>/xsdir/*_NNNN")

    if not args.apply:
        print("\n   Ensayo. Nada ejecutado. Repite con --apply "
              "(y con --limit 2 la primera vez).")
        return 0
    if not jobs:
        print("\n   Nada que hacer.")
        return 0

    nprocs = args.nprocs or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    nprocs = max(1, min(nprocs, len(jobs)))
    print(f"   procesos                           : {nprocs}")

    t0 = time.time()
    done = failed = 0
    errors: List[str] = []
    if nprocs > 1:
        with Pool(processes=nprocs) as pool:
            for i, (sample, res) in enumerate(pool.imap_unordered(_worker, jobs), 1):
                ok = bool(res.get("temperatures_processed"))
                done += ok
                failed += not ok
                if not ok and len(errors) < 10:
                    errors.append(f"{sample}: {'; '.join(res.get('errors') or ['?'])}")
                if i % 25 == 0 or i == len(jobs):
                    print(f"   {i}/{len(jobs)}  ok={done} fallo={failed}  "
                          f"({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            sample, res = _worker(job)
            ok = bool(res.get("temperatures_processed"))
            done += ok
            failed += not ok
            if not ok and len(errors) < 10:
                errors.append(f"{sample}: {'; '.join(res.get('errors') or ['?'])}")
            print(f"   {i}/{len(jobs)}  {sample}  {'ok' if ok else 'FALLO'}", flush=True)

    print(f"\n-- resumen NJOY")
    print(f"   procesadas                         : {done}")
    print(f"   fallidas                           : {failed}")
    print(f"   reloj                              : {time.time() - t0:.0f} s")
    for e in errors:
        print(f"      {e}")

    touched = [j["sample"] for j in jobs]
    faults = verify(layout, args.temperature, ext, touched)
    if failed or faults:
        print("\n❌ quedan muestras mal. Vuelve a lanzar: las buenas se saltan solas.")
        return 1
    print("\n✅ todo a la temperatura pedida y con el maestro apuntando a ello.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
