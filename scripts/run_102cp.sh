#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
# ⚑ 2 dias, no 1: tres runs a 24 cpus (101 fueron 40) contra un limite de
#    24 h iban justas, y que el reloj mate la tercera es el peor final.
#    Si xlarge NO admite 2 dias, sbatch lo RECHAZA al submitir -- lo veras
#    al instante, no de madrugada -- y el arreglo es una linea:
#        sbatch -t 1-00:00:00 run_pyscript.sh
#    (el -t de la linea de comandos pisa la directiva #SBATCH). Con 1 dia,
#    si el reloj se come 102c queda `sbatch run_102c.sh` para rematarla.
#SBATCH -p xlarge
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#SBATCH --job-name=kika-102cp
#
# ============================================================================
#  RUN 102c EN PARALELO con la cadena 102a/102b -- NO es el plan B
# ============================================================================
#
# ⛔ POR QUE ESTE FICHERO EXISTE Y NO SE USA `run_102c.sh` A SECAS.
#
# `run_102c.sh` escribe en `new_test_102c_mixhi_floor`, que es EXACTAMENTE el
# directorio en el que la cadena `run_pyscript.sh` (job `kika-mixhi`, EN VUELO)
# va a escribir su propia 102c en cuanto 102b termine. Lanzar `run_102c.sh`
# ahora pone DOS procesos escribiendo los mismos ficheros, y la puerta acaba
# comparando un fichero a medio escribir. Eso ya paso el 21 y el 22 de agosto.
#
# La cadena esta corriendo y su `KIKA_SKIP_102C` se fijo al submitir, asi que no
# se puede apagar desde fuera. La unica solucion a prueba de tiempos es que esta
# run escriba en SU PROPIO directorio. Es lo que hace.
#
# ⚑ QUE HACER CON LA 102c DE LA CADENA. En cuanto `new_test_102b_mixhi_on` este
#   completa (el log dice "Cadena 102 terminada" o aparece la cinta
#   `26-Fe-56g_nominal.endf`), o bien:
#     - `scancel` del job `kika-mixhi`  -> te ahorras 6 h de maquina; o
#     - dejarlo correr           -> te sale una segunda 102c, y entonces
#       `cmp` de las tres cintas contra las de aqui es la puerta de
#       reproducibilidad mas barata que hay (runs que reproducen escriben
#       cintas byte a byte).
#   Lo que NO vale es lanzar `run_102c.sh` ademas de este.
#
# ⚑ RESERVA DE MEMORIA: 128 GB, no 450. El preflight de 101b/102b midio el pico
#   en 25 GB para 10k muestras, y `--mem=450G` reserva un nodo entero -- con eso
#   esta run NO se programaria en paralelo, que es justo lo que se busca. 128 GB
#   son 5x el pico calculado, y el preflight aborta si no cubre. El reloj baja a
#   1 dia porque aqui hay UNA run, no tres.
#
# ⚑ SANDBOX PROPIO, Y EL NOMBRE `scripts` NO ES NEGOCIABLE.
#
#   EXFOR/scripts/        <- la RUN 100 (100k), EN VUELO. NO SE TOCA.
#   EXFOR/bexp/scripts/   <- las runs 101a/101b. Cerradas.
#   EXFOR/mixhi/scripts/  <- esto.
#
# `exfor_to_endf_research.py:57` hace `sys.path.insert(0, parent.parent)`, asi
# que el paquete `scripts` es el hermano del script. Con `scripts_mixhi/` el
# import habria resuelto contra EXFOR/scripts/ -- el codigo de la run 100 -- en
# silencio. El preflight lo comprueba y lo imprime.
#
# QUE MIDE. Opcion 3 de docs/chi2-mf4/between_experiment_floor_design.md §7.7,
# un solo interruptor `KIKA_RESTORE_MIXTURE_HIGH_ORDER`:
#   (a) el regularizador near-zero deja de coger la sigma de los bins CONGELADOS
#       (por encima de mc_order_cap las replicas llevan todas el nominal, asi que
#       su sigma es jitter de c0) y trasplantarsela a los que SI se muestrearon;
#   (b) en los bins congelados vuelve el termino entre-modelos ANALITICO, exacto,
#       calculado de los ajustes nominales por grado.
# NO TOCA MF4: el central se queda como esta, y eso es una de las puertas.
#
# EL INJERTO, que es lo que 102a existe para comprobar. Este sandbox NO es una
# copia de mi copia local: es `bexp/scripts` + SOLO los 13+15 hunks de este
# cambio. El refactor de splice (`merge_spliced_grid`/`split_original_grid`) que
# hay en HEAD se ha dejado FUERA a proposito -- necesita su propia puerta, y
# meter dos cambios a la vez destruye la atribucion. Verificado: 0 apariciones.
#
# ¿HACE FALTA 102a? Es la pregunta que hizo Juan, y la respuesta es SI, por una
# razon concreta y no por ceremonia: el injerto de arriba es una operacion
# MANUAL. Los tests unitarios ya prueban que con el flag apagado los caminos son
# identicos (`test_apagado_es_la_identidad_bit_a_bit`), pero no prueban que yo
# haya injertado bien. La puerta byte a byte contra 101a si. Y cuesta 4 h de una
# maquina que esta parada de noche. Si aun asi lo quieres saltar:
# `sbatch --export=ALL,KIKA_SKIP_102A=1 run_pyscript.sh`  -- pero entonces una
# diferencia en 102b NO es atribuible al flag.
#
# ¿POR QUE 10k? Porque el control emparejado es 101a, que es de 10k. A 100k se
# medirian dos cambios a la vez. Pico ~25 GB, no compite con la run 100.
# ============================================================================

set -o pipefail

R101A=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_101a_bexp_off
R102A=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_102a_mixhi_off
R102B=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_102b_mixhi_on
R101B=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_101b_bexp_on
# ⚑ DIRECTORIO PROPIO. Ver la cabecera: `new_test_102c_mixhi_floor` es de la
#   cadena en vuelo y escribir ahi es la colision que este fichero evita.
R102C=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_102cp_mixhi_floor
# ⚠ Y NO se hace mkdir de $R102A/$R102B: son de la cadena. Aqui solo el propio.
mkdir -p "$R102C"

if [ -e "$R102C/26-Fe-56g_nominal.endf" ]; then
  echo "⛔ $R102C ya tiene una cinta. O esta run ya se hizo, o hay otro job"
  echo "   escribiendo ahi. Borra el directorio a mano si quieres rehacerla."
  exit 1
fi

# Cronometraje por etapa: a 24 cpus no sabemos el coste real (101 fueron 40), y
# tres runs contra un limite de 24 h es justo. Que quede medido para la proxima.
_t0=$SECONDS
_stamp(){ echo "[tiempo] $1: $(( (SECONDS-_t0)/60 )) min desde el arranque"; }

# El entorno COMUN -- identico al de 101a. Las dos runs difieren en UNA variable.
export KIKA_N_SAMPLES=10000
export KIKA_MF34_PER_ORDER_MESH=1
export KIKA_SAVE_FINE_MESH_INPUTS=1
export KIKA_SAVE_MF34_COV_SIDECARS=1
export KIKA_STOP_AFTER_NOMINAL_FITS=0
export KIKA_DEGREE_WEIGHT_FLOOR=0.01
export KIKA_BETWEEN_EXP_FLOOR_ONLY_BLIND=1
export KIKA_BETWEEN_EXP_POOL_MIN_BINS=30
# ⚑ El suelo entre experimentos va APAGADO en las DOS. Es un cambio distinto y
#   sin decidir; dejarlo encendido mezclaria dos efectos en una sola medida.
export KIKA_BETWEEN_EXP_FLOOR_POOLED=0

echo "[preflight] cpus=${SLURM_CPUS_PER_TASK:-?}  mem_per_cpu=${SLURM_MEM_PER_CPU:-?} MB"
python - <<'PRE' || exit 1
import os, sys, shutil
from pathlib import Path

here = Path.cwd()
if not (here / "exfor_to_endf_research.py").exists():
    print(f"[preflight] ⛔ cwd={here} no es el directorio del sandbox.")
    print("[preflight]    Lanza con: cd .../EXFOR/mixhi/scripts && sbatch run_pyscript.sh")
    sys.exit(1)

sys.path.insert(0, str(here.parent))
import scripts.exfor_utils as eu
resolved = Path(eu.__file__).resolve()
print(f"[preflight] scripts.exfor_utils -> {resolved}")
if "/mixhi/" not in str(resolved):
    print("[preflight] ⛔ el paquete `scripts` NO resuelve dentro del sandbox.")
    print("[preflight]    Este job estaria corriendo el codigo de la run 100 o de bexp.")
    sys.exit(1)

# 1. Que el codigo nuevo este, y que el refactor de splice NO este.
import scripts.exfor_to_endf_research as er
for tok in ("analytic_between_block", "mixture_regularisation_inputs"):
    if not hasattr(er, tok):
        print(f"[preflight] ⛔ falta {tok} en el sandbox."); sys.exit(1)
import inspect
sig = inspect.signature(eu.regularize_near_zero_relative_covariance).parameters
for tok in ("frozen_mask", "mixture_abs_std"):
    if tok not in sig:
        print(f"[preflight] ⛔ regularize_near_zero... no acepta {tok}."); sys.exit(1)
src_u = (here / "exfor_utils.py").read_text()
if "def merge_spliced_grid" in src_u or "def split_original_grid" in src_u:
    print("[preflight] ⛔ el refactor de splice se ha colado en el injerto.")
    print("[preflight]    Eso son DOS cambios a la vez y la atribucion se pierde.")
    sys.exit(1)
print("[preflight] sandbox OK: codigo nuevo presente, splice fuera")

# 2. Que el flag sea LEIBLE POR ENTORNO. Si fuera constante, las dos runs
#    correrian lo mismo y la comparacion saldria plana sin avisar.
src = (here / "exfor_to_endf_research.py").read_text()
if 'KIKA_RESTORE_MIXTURE_HIGH_ORDER' not in src:
    print("[preflight] ⛔ RESTORE_MIXTURE_HIGH_ORDER no se lee del entorno.")
    sys.exit(1)
print("[preflight] flag por entorno OK")

# 3. Memoria. Misma formula MEDIDA que la run 100 (751 B por pareja).
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
        if base not in d.parents and d != base:
            continue
        for name in ("memory.max", "memory/memory.limit_in_bytes"):
            try:
                s = (d / name).read_text().strip()
            except OSError:
                continue
            if s.isdigit() and int(s) < (1 << 50):
                return int(s) // (1024 * 1024), f"cgroup {d/name}"
    return 0, None

N, NBINS = int(os.environ["KIKA_N_SAMPLES"]), 1738
peak_gb = (751 * NBINS * N + 2 * N * 10428 * 8) / 1e9 + 10
mem_mb, fuente = reserva_mb()
print(f"[preflight] N = {N:,} x {NBINS} bins -> pico MEDIDO {peak_gb:.0f} GB")
if fuente is None:
    print("[preflight] ⛔ no puedo leer la reserva por ninguna via. Arregla el --mem.")
    if os.environ.get("KIKA_ALLOW_UNKNOWN_MEM") != "1":
        sys.exit(1)
else:
    print(f"[preflight] reserva = {mem_mb/1024:.0f} GB  (via {fuente})")
    if mem_mb / 1024 < peak_gb:
        print("[preflight] ⛔ la reserva no cubre el pico. Sube cpus o pon --mem.")
        sys.exit(1)

free_gb = shutil.disk_usage("/share_snc").free / 1e9
need_gb = 2 * (12 + 10 * N / 100000)
print(f"[preflight] disco libre {free_gb:.0f} GB, las dos runs piden ~{need_gb:.0f} GB")
if free_gb < need_gb + 50:
    print("[preflight] ⛔ margen insuficiente. Borra sidecars de chi2 antes.")
    sys.exit(1)
print("[preflight] OK")
PRE

# ============ 102c : FLAG ENCENDIDO + SUELO ENTRE EXPERIMENTOS =============
#
# La especulativa, y la razon de que este aqui: si 102b sale como se espera,
# ESTA es la run que se pediria mañana, y ya estaria hecha. Si 102b NO sale como
# se espera, 102c sigue siendo informacion -- mide el suelo agrupado, que es una
# decision abierta aparte, y ademas es la PRIMERA vez que el tope por orden
# (max_floor_order=4) se ejerce sobre el objeto real y no en un test.
#
# ⚑ No hace falta codigo nuevo: los dos flags existen y estan probados. Es una
#   variable de entorno mas, o sea el riesgo de dejarla encolada es ~0.
#
# ⚠ Y lo que 102c NO puede medir: el suelo agrupado no llega a la cinta `_mg`
#   (§7.6 de las notas, confirmado con cmp sobre 101a/101b). Asi que aqui se
#   vera en la cinta FINA y no en el entregable. Eso es un hueco conocido del
#   suelo, no de esta run.
# ===========================================================================
  echo; echo "=========== 102c-paralela  (flag ENCENDIDO + suelo agrupado) ==========="
  KIKA_OUTPUT_DIR=$R102C/ \
  KIKA_RESTORE_MIXTURE_HIGH_ORDER=1 \
  KIKA_BETWEEN_EXP_FLOOR_POOLED=1 \
      python exfor_to_endf_research.py || exit 1

  # ⚑ EL CONTROL ES 102a, NO 102b. 102b puede no haber terminado cuando esto
  #   acabe, y una puerta contra un directorio a medio escribir MIENTE. 102a
  #   esta cerrada (RESTORE=0, suelo=0) y es el control emparejado que existe
  #   HOY; 101b (suelo=1, RESTORE=0) es el segundo, y entre los dos el efecto
  #   queda atribuido: 102cp vs 102a = los dos cambios, 102cp vs 101b = solo
  #   RESTORE. La comparacion con 102b se hace DESPUES, a mano, cuando exista.
  python - "$R102C" "$R102A" "$R101B" <<'GATE_C' || exit 1
import json, sys, filecmp
from pathlib import Path
import numpy as np
import pandas as pd

c, b, fl = (Path(p) for p in sys.argv[1:4])
if not (b / "run_metadata.json").exists():
    alt = Path(str(b).replace("102b_mixhi_on", "102a_mixhi_off"))
    print(f"  ⚠ 102b no esta; comparo contra {alt.name}")
    b = alt
bad = 0
cfg = json.loads((c / "run_metadata.json").read_text())["config"]
for k, want in [("RESTORE_MIXTURE_HIGH_ORDER", True),
                ("APPLY_BETWEEN_EXP_FLOOR_POOLED", True),
                ("BETWEEN_EXP_FLOOR_MAX_ORDER", 4),
                ("BASE_SEED", 42)]:
    got = cfg.get(k)
    print(f"  {k:34s} = {got}   (esperado {want})")
    if got != want:
        print("    ⛔ NO coincide"); bad += 1

# PUERTA 1 -- el central sigue sin moverse. Ninguno de los dos cambios puede.
for ref, etiq in ((b, "102a"), (fl, "101b")):
    if not (ref / "26-Fe-56g_nominal.endf").exists():
        print(f"  ⚠ {etiq} incompleta, no la uso como control"); continue
    for name in ("nominal_fits.parquet", "mf34_mean_perbin.npy"):
        pc, pb = c / name, ref / name
        if pc.exists() and pb.exists():
            if filecmp.cmp(pc, pb, shallow=False):
                print(f"  ✅ {name} identica a {etiq}: el central no se ha movido")
            else:
                print(f"  ⛔ {name} DIFIERE de {etiq} -- ni el suelo ni la"
                      f" restauracion pueden tocar el central"); bad += 1

# PUERTA 2 -- EL TOPE POR ORDEN, ejercido por primera vez sobre el objeto real.
# El suelo tiene que morder en a_1..a_4 y NO en a_5/a_6.
try:
    log = sorted(c.glob("exfor_to_endf_*.log"))[-1].read_text().splitlines()
    fl = [l for l in log if "[Between-exp floor POOLED]" in l]
    for l in fl:
        print("   ", l.strip())
    per = {}
    for l in log:
        s = l.strip()
        if s.startswith("l=") and "floored (median" in s:
            per[int(s[2])] = int(s.split(":")[1].split()[0])
    ret = any("RETENIDO en l>4" in l for l in log)
    print(f"  floored por orden: {per}")
    if per.get(5) or per.get(6):
        print("  ⛔ el suelo ha tocado a_5/a_6: max_floor_order NO esta actuando")
        bad += 1
    elif ret:
        print("  ✅ a_5/a_6 retenidos: el tope por orden actua sobre el objeto real")
    else:
        print("  ⚠ no veo la linea RETENIDO; comprueba a mano")
except Exception as e:
    print(f"  ⚠ no pude leer el log: {type(e).__name__}: {e}")

print()
print("  ⚑ 102cp es la CANDIDATA a entregable si las dos piezas convencen.")
print("    La barra sigue siendo la banda DCS y el conservadurismo, no un umbral.")
sys.exit(1 if bad else 0)
GATE_C
_stamp "102c lista"

echo
echo "✅ 102c-paralela terminada. Siguiente:"
echo "   1. leer la banda DCS contra 102a (between_experiment_floor_design.md §7.6-7.10);"
echo "   2. re-medir la malla sobre su mf34_fine_mesh_inputs.npz --"
echo "      notebooks/mf34_mesh/{extract_fine_mesh_inputs,criteria_report}.py;"
echo "   3. decidir que se hace con la 102c de la cadena (ver cabecera)."
echo "   (obsoleto) leer la banda DCS de 102b y 102c"
echo "   contra 102a (docs/chi2-mf4/between_experiment_floor_design.md §7.6-7.10)."
