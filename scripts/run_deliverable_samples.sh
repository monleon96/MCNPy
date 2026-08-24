#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -p par_IB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#SBATCH --job-name=kika-ens

# Los tres ensembles ACE del entregable Fe-56. PREPARE PRIMERO, LOS TRES DESPUES
# EN PARALELO:
#
#     JID=$(sbatch --parsable run_deliverable_samples.sh prepare)
#     sbatch --dependency=afterok:$JID run_deliverable_samples.sh joint
#     sbatch --dependency=afterok:$JID run_deliverable_samples.sh mf34
#     sbatch --dependency=afterok:$JID run_deliverable_samples.sh mf33
#
# ⚑ LA DEPENDENCIA NO ES OPCIONAL, Y NO ES POR ORDEN SINO POR CORRECCION.
# Los tres leen la cinta base y la cache del conjunto; `prepare` es el unico que
# las ESCRIBE. Lanzados los tres a pelo, los tres construyen la misma cinta base
# sobre la misma ruta, y dos jobs escribiendo un fichero es exactamente como una
# puerta acaba comparando algo a medio escribir
# ([[duplicate-submission-makes-the-gate-lie]]). El driver lo comprueba tambien:
# un set sin cinta base o sin cache aborta en vez de construirlas.
#
# `prepare` ademas paga una sola vez lo caro: los ~6 min de leer 570 MB, el
# ensamblado, y la eigendescomposicion de 13 003 filas. Los tres sets la saltan.
#
# COLA. 1 job + luego 3 simultaneos, contra el tope de 4 en par_IB
# ([[queue-limits-4-jobs-and-par-ib-250g]]). 96 G esta muy por debajo de los
# 250 G que par_IB admite; no lo subas por encima o el job se va a xlarge.
#
# ⚑ --mem ESTA PUESTO Y EL PREFLIGHT LO COMPRUEBA, no lo avisa. La run 100 murio
# por OOM a las 6 h 24 porque `--mem` era un comentario y SLURM_MEM_PER_NODE
# viene VACIO en este cluster ([[mem-directive-was-a-comment-not-a-directive]],
# [[slurm-mem-per-node-is-empty-here]]). Se lee por tres vias y se aborta.
#
# MEMORIA. El conjunto es (2317 + 6*stride)^2 en float64: con stride ~1781 son
# 13 003 filas = 1,35 GB, y tanto eigvalsh como la SVD copian. Pico ~12-15 GB.
# Los 40 workers de ACER NO lo multiplican: el driver LIBERA la matriz antes de
# que nada bifurque, porque fork da a cada worker una vista copy-on-write y los
# refcounts de CPython escriben en las cabeceras.
#
# DISCO, por ensemble de 512 replicas, EN /SCRATCH:
#     joint  ENDF 14 GB + PENDF 25-50 GB + ACE 20-40 GB
#     mf34   ENDF 14 GB +      -         + ACE 20-40 GB   (PENDF nominal, 1 fichero)
#     mf33        -      + PENDF 25-50 GB + ACE 20-40 GB   (ENDF base, 1 fichero)
# ~150-250 GB los tres. El preflight mide el sistema de ficheros de SALIDA, que
# es /SCRATCH y NO /share_snc. El PENDF se puede borrar en cuanto ACER ha pasado.

set -o pipefail

# ⚑ El argumento se comprueba a mano, y no con `${1:?...}`. Ese atajo estuvo aqui
# y ERA UN FALLO: bash cierra la expansion en el PRIMER `}` del mensaje, asi que
# `{prepare|joint|...}` la cortaba en `{prepare`, trataba los `|` como tuberias y
# dejaba `WHICH` VACIO aunque el argumento estuviese ahi. Los jobs 8556726/27
# murieron por eso, con el preflight informando `set=` en blanco.
WHICH=$1
N=${2:-512}
case "$WHICH" in
    prepare|joint|mf34|mf33) ;;
    *)
        echo "uso: sbatch run_deliverable_samples.sh MODO [n]"
        echo "  MODO = prepare | joint | mf34 | mf33"
        echo "  prepare va SOLO y primero; los tres sets encadenan con"
        echo "  --dependency=afterok:<jobid del prepare>"
        echo "recibido: [$WHICH]"
        exit 1
        ;;
esac

source /work/monleon-de-la-jan/myenv/bin/activate
SCRIPTS=/share_snc/snc/JuanMonleon/EXFOR/newcode/scripts
cd "$SCRIPTS" || exit 1

OUT=/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/Fe56_a0cross_ens512_20260824
ENDF=/share_snc/snc/JuanMonleon/ENDF_samples/ENTREGABLE_Fe56_MF4_MF34_20260824/26-Fe-56_thiswork_MF4_MF33_MF34_a0cross.endf
# El sha256 que va en el SHA256SUMS del entregable. Se comprueba: la cache del
# conjunto se indexa por el, y muestrear otra cinta creyendola esta es un fallo
# silencioso.
ENDF_SHA=d7054ac382ad4a26d84e1ccbdc1241170d1f3b0b6e4bc93285bfb882d9b7a7d5

echo "[preflight] set=$WHICH n=$N cpus=${SLURM_CPUS_PER_TASK:-?} out=$OUT"

# 1. Que el kika INSTALADO traiga el camino conjunto. Va por wheel y /work no se
#    ve desde WSL, asi que el script comprueba su propia importacion en vez del
#    entorno ([[cluster-venv-is-not-inspectable]]).
python - <<'PRE' || exit 1
import inspect, sys
try:
    from kika.sampling.joint_perturbation import perturb_joint_mf33_mf34
    from kika.sampling.joint_mf33_mf34 import load_joint_mf33_mf34
    from kika.sampling.base_tape import build_base_tape
    from kika.sampling.mf34_cross import read_mf34_split
    from kika.sampling.endf_perturbation import perturb_ENDF_files
    from kika.sampling.pendf_perturbation import perturb_PENDF_files
    for f in (perturb_ENDF_files, perturb_PENDF_files):
        assert "precomputed" in inspect.signature(f).parameters, f.__name__
except Exception as e:
    print(f"[preflight] ⛔ el kika instalado NO trae el camino conjunto: {e}")
    print("[preflight]    Instala el wheel de EXFOR/kika_dist/ con")
    print("[preflight]    pip install --force-reinstall <wheel>")
    sys.exit(1)
print("[preflight] kika instalado: OK (joint + precomputed presentes)")
PRE

# 2. La cinta y su hash, la reserva de memoria, el disco de SALIDA, y las
#    entradas compartidas segun el modo.
python - "$ENDF" "$ENDF_SHA" "$OUT" "$WHICH" "$N" <<'PRE' || exit 1
import hashlib, os, shutil, sys
from pathlib import Path
endf, want_sha, out, which, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

p = Path(endf)
if not p.exists():
    print(f"[preflight] ⛔ no existe {endf}"); sys.exit(1)
h = hashlib.sha256()
with open(p, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 22), b""):
        h.update(b)
if h.hexdigest() != want_sha:
    print(f"[preflight] ⛔ sha256 del entregable no coincide\n"
          f"             esperado {want_sha}\n             leido    {h.hexdigest()}")
    sys.exit(1)
print(f"[preflight] entregable OK ({p.stat().st_size/1e6:.0f} MB, sha256 verificado)")

def reserva_mb():
    v = os.environ.get("SLURM_MEM_PER_NODE")
    if v and v.strip().isdigit():
        return int(v), "SLURM_MEM_PER_NODE"
    v, c = os.environ.get("SLURM_MEM_PER_CPU"), os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.strip().isdigit() and c and c.strip().isdigit():
        return int(v) * int(c), "SLURM_MEM_PER_CPU x cpus"
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
    except OSError:
        rel = ""
    base = Path("/sys/fs/cgroup"); cand = base / rel.lstrip("/")
    for d in [cand, *cand.parents]:
        if d != base and base not in d.parents:
            continue
        for name in ("memory.max", "memory/memory.limit_in_bytes"):
            try:
                s = (d / name).read_text().strip()
            except OSError:
                continue
            if s.isdigit() and int(s) < (1 << 50):
                return int(s) // (1024 * 1024), f"cgroup {d/name}"
    return 0, None

mem_mb, src = reserva_mb()
peak_gb = 4 * (13003 ** 2 * 8) / 1e9 + 8      # eigvalsh y SVD copian
print(f"[preflight] pico esperado ~{peak_gb:.0f} GB")
if src is None:
    print("[preflight] ⛔ no puedo leer la reserva de memoria por ninguna via.")
    print("[preflight]    Es exactamente lo que dejo morir la run 100 a las 6 h.")
    sys.exit(1)
print(f"[preflight] reserva {mem_mb/1024:.0f} GB (via {src})")
if mem_mb / 1024 < peak_gb:
    print("[preflight] ⛔ la reserva no cubre el pico. Sube --mem (<=250G en par_IB).")
    sys.exit(1)

outp = Path(out)
outp.mkdir(parents=True, exist_ok=True)
base, cache = outp / "base_tape.endf", outp / "joint_cache.npz"

if which == "prepare":
    need = 2.0                                   # cinta base 27 MB + cache ~1.4 GB
else:
    need = {"joint": 100, "mf34": 60, "mf33": 95}[which] * n / 512
# El disco que importa es el de SALIDA (/SCRATCH), no /share_snc.
free = shutil.disk_usage(outp).free / 1e9
print(f"[preflight] {outp} libre {free:.0f} GB, este job pide ~{need:.0f} GB")
if free < need + 60:
    print("[preflight] ⛔ margen insuficiente en el sistema de ficheros de salida.")
    sys.exit(1)

if which == "prepare":
    if base.exists() and cache.exists():
        print("[preflight] ⚠ la cinta base y la cache YA estan. Este job las"
              " reutiliza; usa --rebuild-base si quieres rehacerlas.")
else:
    faltan = [str(x) for x in (base, cache) if not x.exists()]
    if faltan:
        print(f"[preflight] ⛔ faltan {faltan}.")
        print("[preflight]    Los tres sets NO las construyen a proposito: dos")
        print("[preflight]    jobs escribiendo la misma ruta hacen mentir a la")
        print("[preflight]    puerta. Lanza --set prepare y encadena con")
        print("[preflight]    --dependency=afterok:<jobid>.")
        sys.exit(1)
    print("[preflight] cinta base y cache del conjunto presentes: reutilizadas")
print("[preflight] OK")
PRE

COMMON=(--endf "$ENDF" --out "$OUT" --seed 42
        --nprocs "${SLURM_CPUS_PER_TASK:-40}"
        --pendf-cache-dir /share_snc/snc/JuanMonleon/cache/kika_pendf_cache
        --njoy-exe /soft_snc/NJOY/2016.78/bin/njoy
        --xsdir /share_snc/snc/JuanMonleon/xsdir_MCNPy/xsdir40-irdff2
        --temperatures 293.6 --library-name jeff40)

if [ "$WHICH" = "prepare" ]; then
    python run_deliverable_samples.py --set prepare -n "$N" "${COMMON[@]}" || exit 1
    echo "[prepare] listo. Ahora los tres, en paralelo:"
    echo "  JID=\$(sbatch --parsable run_deliverable_samples.sh prepare)   # ya hecho"
    echo "  sbatch run_deliverable_samples.sh joint"
    echo "  sbatch run_deliverable_samples.sh mf34"
    echo "  sbatch run_deliverable_samples.sh mf33"
    ls -la "$OUT"
    exit 0
fi

python run_deliverable_samples.py --set "$WHICH" -n "$N" "${COMMON[@]}" || exit 1

# --- LAS PUERTAS ------------------------------------------------------------
python - "$OUT/$WHICH" "$WHICH" "$N" <<'GATE' || exit 1
import json, sys
from pathlib import Path
d, which, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
s = json.loads((d / "run_summary.json").read_text())
bad = 0

def check(ok, msg):
    global bad
    print(f"   {'OK  ' if ok else '⛔  '} {msg}")
    bad += 0 if ok else 1

print("\n-- G3. el objeto que se ha muestreado")
r = s.get("joint_report_restricted") or s["joint_report_full"]
print(f"   dim {r['dimension']}, sigma {r['n_sigma']}, a_l {r['n_a']}, "
      f"cruzado {r['cross_orders']}")
print(f"   lam_min/lam_max {r.get('lam_min_over_max')}, "
      f"max|rho| {r.get('max_abs_rho')}, "
      f"masa negativa {r.get('negative_mass_fraction')}")
check(r["asymmetry"] < 1e-12, "el conjunto es simetrico")
check(r["n_non_finite"] == 0, "sin entradas no finitas")
# 5e-6 es el viaje ASCII a 6 cifras, medido. Mas que eso es el fichero, no la
# aritmetica, y hay que mirarlo ANTES de propagar nada.
check(r.get("max_abs_rho", 9) <= 1 + 5e-6,
      "max|rho| dentro de Cauchy-Schwarz + el ruido del formato")
check((which == "joint") == bool(r["has_cross"]),
      f"el cruzado {'ESTA' if which == 'joint' else 'NO esta'}, como toca")

print("\n-- el sorteo")
dr = s["draw"]
print(f"   rank {dr['rank']}, {dr['n_null']} nulas, "
      f"error de covarianza realizada {dr['realised_covariance_error']}")
check(int(dr["n"]) == r["dimension"], "el sorteo cubre todas las filas")

print("\n-- G7/G8. lo producido")
check(s.get("n_ace_ok", 0) == n, f"{s.get('n_ace_ok', 0)}/{n} ACE generados")
check(s.get("n_missing_inputs", 0) == 0,
      f"{s.get('n_missing_inputs', 0)} replicas sin par (ENDF, PENDF)")

if bad:
    print(f"\n❌ el ensemble '{which}' FALLA en {bad} comprobacion(es).")
    sys.exit(1)
print(f"\n✅ ensemble '{which}' OK: {n} ACE en {d}/ace/")
print("   Cuando los tres esten: borrar los pendf/ y quedarse con ace/ + xsdir/.")
GATE

du -sh "$OUT/$WHICH"/* 2>/dev/null
df -h "$OUT" | tail -1
