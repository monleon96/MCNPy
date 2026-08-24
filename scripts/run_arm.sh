#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH -p xlarge
#SBATCH --mail-type=END,FAIL
# ⚑ LA PARTICION Y LOS CPUS LOS PISA `lanzar_todo.sh` EN CADA SUBMIT, porque
#   los seis brazos se reparten entre dos particiones (3 en `xlarge` a 24 cpus y
#   3 en `par_IB` a 40). Lo de arriba es el fallback para un `sbatch run_arm.sh`
#   suelto. `--mem=128G` NO se pisa: es explicito en las dos particiones a
#   proposito -- un `--mem` implicito ya mato la run 100 por OOM a las 6h24, y
#   128 GB son 5x el pico MEDIDO (25 GB) y caben de sobra en las dos (par_IB da
#   4 G/cpu x 40 = 160 GB por defecto, que es con lo que corrieron 101a/101b).
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#
# ============================================================================
#  LAS SEIS RUNS DE LA NOCHE DEL 22-ago-2026 — un job por brazo, en paralelo
# ============================================================================
#
# USO:  cd .../EXFOR/newcode/scripts && ./lanzar_todo.sh
#       (o a mano:  sbatch --job-name=k-R2 --export=ALL,KIKA_ARM=R2 run_arm.sh)
#
# ⚑ POR QUE UN SOLO FICHERO Y NO SEIS. Los brazos se diferencian EXCLUSIVAMENTE
#   en variables de entorno y todos piden los mismos recursos. Seis copias
#   casi iguales derivan entre si en cuanto se toca una; una tabla no puede.
#   El nombre del job y el directorio salen del brazo, asi que cada submit es
#   un job independiente con su propia salida.
#
# ⚑ SANDBOX NUEVO Y LIMPIO: `EXFOR/newcode/scripts`.
#
#     EXFOR/scripts/        run 100 (100k). Cerrada.
#     EXFOR/bexp/scripts/   runs 101a/101b. Cerradas.
#     EXFOR/mixhi/scripts/  runs 102a/102b/102c/102cp. Cerradas.
#     EXFOR/newcode/scripts <- esto: COPIA LIMPIA de ~/kika/scripts.
#
#   Los tres anteriores eran INJERTOS -- `bexp/scripts` mas unos hunks
#   elegidos a mano -- porque habia runs en vuelo y habia que aislar UN cambio.
#   Ya no hay ninguna en vuelo, asi que este es una copia entera y se acabaron
#   los injertos. Lo que eso arrastra: el refactor de splice
#   (`merge_spliced_grid` / `split_original_grid`), que hasta ahora se dejaba
#   fuera A PROPOSITO y cuyo preflight lo prohibia. Ahora entra, y por eso el
#   brazo R1 existe.
#
#   El nombre `scripts` NO es negociable: `exfor_to_endf_research.py:57` hace
#   `sys.path.insert(0, parent.parent)`, asi que el paquete `scripts` es el
#   hermano del script. Con `scripts_newcode/` el import resolveria contra
#   EXFOR/scripts/ -- el codigo de la run 100 -- en silencio. El preflight lo
#   comprueba y lo imprime.
#
# ============================================================================
#  LOS SEIS BRAZOS
# ============================================================================
#
#   brazo         RESTORE  SUELO  SIG_RATIO  MALLA_1ETAPA  SIGMA_E   que contesta
#   ----------------------------------------------------------------------------
#   R1 inercia       0       0        0            0          0      ¿la copia
#     entera reproduce 102a BYTE A BYTE? Cubre de una vez el refactor de splice
#     y los cuatro cambios de hoy. Es la unica puerta DURA de la noche: si
#     falla, los otros cinco brazos estan construidos sobre algo que no sabemos
#     que es.
#
#   R2 base          1       1        0            0          0      LA NUEVA
#     CANDIDATA, y la referencia de R3-R6. Tres cosas que solo una run puede
#     decir: (a) ¿vuelve a escribirse el `_a0cross` FINO ahora que el rechazo
#     lo decide `d_shift_rel` y no la masa? (b) ¿conserva la cinta `_mg` la
#     sigma de orden alto, ahora que la guarda esta portada? Medido sobre
#     102cp, sin ella a_6 caia de 233 % a 64 %. (c) ¿que cuesta de verdad
#     retirar `BETWEEN_EXP_FLOOR_MAX_ORDER`? Offline dio 400 entradas a x1,62.
#
#   R3 sig-ratio     1       1        1            0          0      lo que
#     compra leer el tope sigma-ratio en los SEIS ordenes y no solo en a_1.
#     Es monotono (mas restriccion => mas grupos), asi que la malla solo puede
#     salir mas fina. ⚠ Mide la ruta que embarca HOY; deja de importar en
#     cuanto R4 se acepte.
#
#   R4 malla 1 etapa 1       1        0            1          0      lo que
#     compra elegir la malla UNA vez, por orden, desde el FINO, con el criterio
#     fisico (sigma_E + consistencia de a_l + tope por orden). Offline sobre
#     102cp: 6134 parametros y sobre-declaracion 3,0/3,0/3,0/3,0/4,5/6,7x,
#     contra 3311 y 87 734x de la malla desplegada.
#     ⚑ DESBLOQUEADA HOY: esta es la ruta que hacia alcanzable el `1e-300` del
#     colapso, y ya esta acotado (suelo de no-cancelacion) y probado de punta a
#     punta (el DP fisico veta el grupo que cancela).
#
#   R5 sigma_E       1       1        0            0          1      la
#     dependencia en ENERGIA del suelo entre experimentos. La constante cubre
#     90,6/79,3/63,1 % por tercios de energia: sub-declara justo donde mas
#     desacuerdo hay. Un flag de diferencia contra R2.
#
#   R6 todo          1       1        1            1          1      el
#     entregable candidato. No es atribuible por si solo -- lo es por R2..R5 --
#     pero existe para que mañana se pueda MIRAR el objeto final sin esperar
#     otras seis horas.
#
# ⚑ POR QUE R2 ES LA REFERENCIA Y NO 102cp. 102cp corrio con el codigo viejo:
#   sin la guarda multigrupo, con el tope del suelo en 4 y con el rechazo del
#   cruzado por masa. Comparar R5 contra 102cp mezclaria cuatro cambios con
#   uno. R2 es el mismo codigo que R3-R6 y difiere en UN flag de cada uno.
#
# ⚑ RESERVA: 128 GB. El pico MEDIDO a 10k son 25 GB y el preflight aborta si la
#   reserva no lo cubre. 450G reservaria un nodo entero e impediria justo lo
#   que se busca esta noche, que es que los seis corran a la vez.
# ============================================================================

set -o pipefail

ARM="${KIKA_ARM:?falta KIKA_ARM (R1..R6). Usa ./lanzar_todo.sh}"
SAMPLES=/share_snc/snc/JuanMonleon/ENDF_samples
R102A=$SAMPLES/new_test_102a_mixhi_off
R102CP=$SAMPLES/new_test_102cp_mixhi_floor

# ── la tabla: RESTORE SUELO SIG_RATIO MALLA_1ETAPA SIGMA_E  K  C ────────────
#
# ⚑ K y C SON NUEVOS (23-ago) y son las dos perillas del Anexo J:
#     K = KIKA_MF34_MESH_A_CONSISTENCY_K  -- nivel de confianza del test de
#         consistencia de a_l (J-6). El 10 con el que corrio la serie 103 esta
#         marcado con ⛔ en el propio anexo: «un test a 10 sigma no rechaza casi
#         nada». k = 3 es el defendible y cuesta +385 parametros.
#     C = KIKA_MF34_MESH_SIGMA_RATIO_MAX -- techo de DESPERDICIO declarado (J-7).
#         No es fisica: con COLLAPSE=max sub-declarar es imposible, asi que C
#         solo acota lo que se sobre-declara.
#   RES_FACTOR se queda en 1.0 y NO se parametriza: J-5 dice que es una
#   definicion (un elemento de resolucion), no un ajuste. Relajarlo hay que
#   declararlo, y entonces se escribe aqui como brazo con su nombre.
#
# ⚑ LA SERIE 104 LLEVA EL ARREGLO DEL SINGLETON. Un bin mas ancho que su propia
#   sigma_E hacia inadmisible su propio singleton y el DP rendia el TRAMO ENTERO
#   a singletons: 82 bins por encima de 3,47 MeV dejaban 3 636 de los 10 428
#   parametros sin agrupar (a_2 entero, porque no cambia de signo nunca).
case "$ARM" in
  R1) FL=(0 0 0 0 0 10 3); TAG=inercia      ; SERIE=103 ;;
  R2) FL=(1 1 0 0 0 10 3); TAG=base         ; SERIE=103 ;;
  R3) FL=(1 1 1 0 0 10 3); TAG=sigratio     ; SERIE=103 ;;
  R4) FL=(1 1 0 1 0 10 3); TAG=malla1etapa  ; SERIE=103 ;;
  R5) FL=(1 1 0 0 1 10 3); TAG=sigmaE       ; SERIE=103 ;;
  R6) FL=(1 1 1 1 1 10 3); TAG=todo         ; SERIE=103 ;;
  S1) FL=(1 1 0 1 0 10 3); TAG=fixsingleton ; SERIE=104 ;;
  S2) FL=(1 1 0 1 0  3 3); TAG=k3c3         ; SERIE=104 ;;
  S3) FL=(1 1 0 1 0  3 2); TAG=k3c2         ; SERIE=104 ;;
  *)  echo "⛔ KIKA_ARM='$ARM' no es R1..R6 ni S1..S3"; exit 1 ;;
esac
OUT=$SAMPLES/new_test_${SERIE}${ARM}_${TAG}

# ── CERROJO ATOMICO. Dos submits del mismo brazo escribirian los mismos
#    ficheros y la puerta acabaria comparando una cinta a medio escribir. Eso
#    paso el 21 y el 22 de agosto y costo dos diagnosticos falsos. `mkdir` es
#    atomico en POSIX: el segundo submit muere aqui, no dentro de la puerta.
mkdir -p "$OUT" || exit 1
if ! mkdir "$OUT/.lock" 2>/dev/null; then
  echo "⛔ $OUT ya tiene cerrojo: hay otro job escribiendo ahi, o uno murio"
  echo "   sin soltarlo. Si estas seguro de que no hay ninguno: rmdir $OUT/.lock"
  exit 1
fi
trap 'rmdir "$OUT/.lock" 2>/dev/null' EXIT
if [ -e "$OUT/26-Fe-56g_nominal.endf" ]; then
  echo "⛔ $OUT ya tiene una cinta. Borra el directorio si quieres rehacerlo."
  exit 1
fi

_t0=$SECONDS
echo "=============================================================="
echo "  BRAZO $ARM ($TAG)   ->  $OUT"
echo "  RESTORE=${FL[0]}  SUELO=${FL[1]}  SIG_RATIO_POR_ORDEN=${FL[2]}"
echo "  MALLA_1_ETAPA=${FL[3]}  SUELO_CON_ENERGIA=${FL[4]}"
echo "  A_CONSISTENCY_K=${FL[5]}  SIGMA_RATIO_MAX=${FL[6]}  (RES_FACTOR=1.0 fijo)"
echo "=============================================================="

# ── el entorno COMUN a los seis. Identico al de 102a/102cp. ──────────────────
export KIKA_N_SAMPLES=10000
export KIKA_MF34_PER_ORDER_MESH=1
export KIKA_SAVE_FINE_MESH_INPUTS=1
export KIKA_SAVE_MF34_COV_SIDECARS=1
export KIKA_STOP_AFTER_NOMINAL_FITS=0
export KIKA_DEGREE_WEIGHT_FLOOR=0.01
export KIKA_BETWEEN_EXP_FLOOR_ONLY_BLIND=1
export KIKA_BETWEEN_EXP_POOL_MIN_BINS=30
# ── lo que distingue este brazo ─────────────────────────────────────────────
export KIKA_RESTORE_MIXTURE_HIGH_ORDER=${FL[0]}
export KIKA_BETWEEN_EXP_FLOOR_POOLED=${FL[1]}
export KIKA_MULTIGROUP_SIGMA_RATIO_PER_ORDER=${FL[2]}
export KIKA_MF34_MESH_SINGLE_STAGE=${FL[3]}
export KIKA_BETWEEN_EXP_FLOOR_ENERGY=${FL[4]}
export KIKA_MF34_MESH_A_CONSISTENCY_K=${FL[5]}
export KIKA_MF34_MESH_SIGMA_RATIO_MAX=${FL[6]}
export KIKA_OUTPUT_DIR=$OUT/

echo "[preflight] cpus=${SLURM_CPUS_PER_TASK:-?}  brazo=$ARM"
python - <<'PRE' || exit 1
import os, sys, shutil, inspect
from pathlib import Path

here = Path.cwd()
if not (here / "exfor_to_endf_research.py").exists():
    print(f"[preflight] ⛔ cwd={here} no es el sandbox."); sys.exit(1)
sys.path.insert(0, str(here.parent))
import scripts.exfor_utils as eu
resolved = Path(eu.__file__).resolve()
print(f"[preflight] scripts.exfor_utils -> {resolved}")
if "/newcode/" not in str(resolved):
    print("[preflight] ⛔ `scripts` NO resuelve dentro de newcode/: este job")
    print("[preflight]    estaria corriendo el codigo de otra run."); sys.exit(1)

# 1. LOS CUATRO CAMBIOS DE HOY, cada uno con su testigo. Sin esto, un brazo
#    podria correr codigo viejo y salir "sin diferencia" sin avisar de nada.
import scripts.exfor_to_endf_research as er
import scripts.build_group_cross as bgc
import scripts.per_order_mesh as pom
faltan = []
if not hasattr(er, "mixture_regularisation_inputs_grouped"):
    faltan.append("guarda multigrupo (mixture_regularisation_inputs_grouped)")
if not hasattr(bgc, "DEAD_BLOCK_D_SHIFT_MAX"):
    faltan.append("rechazo PSD por consecuencia (DEAD_BLOCK_D_SHIFT_MAX)")
if "indep = (W[l] ** 2) @ d_fine" not in (here / "per_order_mesh.py").read_text():
    faltan.append("suelo de no-cancelacion en el colapso")
if 'str(MAX_SAMPLE_ORDER)' not in (here / "exfor_to_endf_research.py").read_text():
    faltan.append("tope del suelo retirado (BETWEEN_EXP_FLOOR_MAX_ORDER)")
if not hasattr(pom, "physical_solve_order"):
    faltan.append("malla fisica (physical_solve_order)")
# ⚑ EL ARREGLO DEL SINGLETON (23-ago). Sin el testigo, un brazo de la serie 104
#   correria el codigo viejo y saldria con la malla de la 103 sin avisar.
if "lim[-1] = max(lim[-1], wid[-1])" not in (here / "per_order_mesh.py").read_text():
    faltan.append("singleton exento de la resolucion (serie 104)")
if faltan:
    for f in faltan:
        print(f"[preflight] ⛔ falta: {f}")
    sys.exit(1)
print("[preflight] los cuatro cambios de hoy estan presentes")

# 2. QUE LOS CINCO FLAGS SE LEAN DEL ENTORNO. Si alguno fuera constante, dos
#    brazos correrian lo mismo y la comparacion saldria plana SIN avisar.
src = (here / "exfor_to_endf_research.py").read_text()
for tok in ("KIKA_RESTORE_MIXTURE_HIGH_ORDER", "KIKA_BETWEEN_EXP_FLOOR_POOLED",
            "KIKA_MULTIGROUP_SIGMA_RATIO_PER_ORDER",
            "KIKA_MF34_MESH_SINGLE_STAGE", "KIKA_BETWEEN_EXP_FLOOR_ENERGY",
            "KIKA_MF34_MESH_A_CONSISTENCY_K", "KIKA_MF34_MESH_SIGMA_RATIO_MAX"):
    if tok not in src:
        print(f"[preflight] ⛔ {tok} no se lee del entorno."); sys.exit(1)
print("[preflight] los siete flags son leibles por entorno")

# 3. MEMORIA. Misma formula MEDIDA que la run 100 (751 B por pareja).
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
    sys.exit(1)
print(f"[preflight] reserva = {mem_mb/1024:.0f} GB  (via {fuente})")
if mem_mb / 1024 < peak_gb:
    print("[preflight] ⛔ la reserva no cubre el pico."); sys.exit(1)

# 4. DISCO. Seis brazos a la vez, ~9 GB cada uno; el share va al 93 %.
free_gb = shutil.disk_usage("/share_snc").free / 1e9
print(f"[preflight] disco libre {free_gb:.0f} GB; los seis brazos piden ~60 GB")
if free_gb < 120:
    print("[preflight] ⛔ margen insuficiente para seis brazos."); sys.exit(1)
print("[preflight] OK")
PRE

python exfor_to_endf_research.py || exit 1
echo "[tiempo] $ARM: $(( (SECONDS-_t0)/60 )) min"

# ============================================================================
#  LA PUERTA. Distinta por brazo, porque miden cosas distintas.
# ============================================================================
python - "$ARM" "$OUT" "$R102A" "$R102CP" <<'GATE' || exit 1
import json, sys, filecmp
from pathlib import Path
import numpy as np

arm, out, r102a, r102cp = sys.argv[1], *(Path(p) for p in sys.argv[2:5])
bad = 0
cfg = json.loads((out / "run_metadata.json").read_text())["config"]

esperado = {"R1": (False, False), "R2": (True, True), "R3": (True, True),
            "R4": (True, True), "R5": (True, True), "R6": (True, True),
            "S1": (True, True), "S2": (True, True), "S3": (True, True)}[arm]
for k, want in (("RESTORE_MIXTURE_HIGH_ORDER", esperado[0]),
                ("APPLY_BETWEEN_EXP_FLOOR_POOLED", esperado[1]),
                ("BASE_SEED", 42)):
    got = cfg.get(k)
    print(f"  {k:34s} = {got}   (esperado {want})")
    if got != want:
        print("    ⛔ NO coincide"); bad += 1

# ── R1: LA PUERTA DURA DE LA NOCHE ──────────────────────────────────────────
if arm == "R1":
    print("\n  ⚑ PUERTA DE INERCIA: la copia entera contra 102a, byte a byte.")
    print("    Cubre el refactor de splice Y los cuatro cambios de hoy de una vez.")
    for name in ("26-Fe-56g_nominal.endf", "26-Fe-56g_nominal_mg.endf"):
        pa, pb = out / name, r102a / name
        if not pb.exists():
            print(f"  ⚠ 102a no tiene {name}"); continue
        if filecmp.cmp(pa, pb, shallow=False):
            print(f"  ✅ {name} IDENTICA a 102a ({pa.stat().st_size:,} B)")
        else:
            print(f"  ⛔ {name} DIFIERE de 102a "
                  f"({pa.stat().st_size:,} vs {pb.stat().st_size:,} B)")
            print("     Con los flags apagados el codigo nuevo TENIA que ser inerte.")
            print("     El sospechoso numero uno es el refactor de splice, que en")
            print("     los injertos anteriores se dejaba fuera a proposito.")
            bad += 1
else:
    # ── R2..R6: el central no se mueve. Ningun cambio de covarianza puede. ──
    for name in ("nominal_fits.parquet", "mf34_mean_perbin.npy"):
        pa, pb = out / name, r102a / name
        if pa.exists() and pb.exists():
            if filecmp.cmp(pa, pb, shallow=False):
                print(f"  ✅ {name} identica a 102a: el central no se ha movido")
            else:
                print(f"  ⛔ {name} DIFIERE de 102a"); bad += 1

# ── lo que hay que MIRAR mañana, impreso aqui para no tener que abrir nada ──
log = sorted(out.glob("exfor_to_endf_*.log"))
if log:
    txt = log[-1].read_text().splitlines()
    print("\n  --- lineas clave del log ---")
    # ⚠ Patrones ESTRECHOS a proposito. Con "floored" a secas entran cientos de
    #   lineas de `[Unc floor]` y lo que importa se pierde en medio.
    pats = ("FG abs→nominal", "MG abs→nominal", "Near-zero regularization:",
            "[Between-exp floor POOLED]", "retenidas", "floored (median",
            "[CROSS] cross-term ENDF", "proyeccion PSD", "varianza DECLARADA",
            "no-cancelacion", "sobre-declaracion", "agrupamiento en UNA etapa",
            "entradas del agrupamiento", "[AVG] model-averaged")
    vistos = []
    for l in txt:
        t = l.strip()
        if any(p in t for p in pats) and t not in vistos:
            vistos.append(t)
    for t in vistos[:60]:
        print("   ", t[:210])
    if len(vistos) > 60:
        print(f"    ... y {len(vistos)-60} lineas mas, en {log[-1].name}")

fino_cross = out / "26-Fe-56g_nominal_a0cross.endf"
print(f"\n  cinta cruzada FINA: {'✅ ESCRITA' if fino_cross.exists() else '⛔ NO escrita'}"
      + (f" ({fino_cross.stat().st_size:,} B)" if fino_cross.exists() else ""))
if arm == "R2" and not fino_cross.exists():
    print("     ⚠ R2 existe en parte para contestar esto: con el rechazo movido a")
    print("       d_shift_rel deberia volver a escribirse. Si no, el recorte SI")
    print("       mueve una varianza declarada y el mensaje lo dice con el numero.")

mesh = out / "mf34_per_order_mesh.npz"
if mesh.exists():
    z = np.load(mesh)
    ns = [len(z[f"e_{l}"]) - 1 for l in range(1, 7)]
    print(f"  malla por orden: {ns}   total {sum(ns)} parametros")
    print("     (102a/101b: [649,660,636,601,618,646] = 3810 grupos;"
          " 102b/102cp: [662,696,631,501,430,391] = 3311;"
          " 103R4: [1215,1738,1215,1255,1198,1199] = 7820)")
    # ⚑ LA PUERTA FUERTE DE LA SERIE 104. La malla se calculo OFFLINE sobre el
    #   `mf34_fine_mesh_inputs.npz` de 103R4 antes de lanzar nada, asi que el
    #   numero de parametros esta PREDICHO. Si la run no lo reproduce, o el
    #   arreglo no llego, o la covarianza de esta run no es la de 103R4 -- y las
    #   dos cosas invalidan la comparacion, no solo el tamano.
    PRED = {"S1": 5964, "S2": 6349, "S3": 6888}
    if arm in PRED:
        want = PRED[arm]
        print(f"  prediccion offline para {arm}: {want} parametros")
        if sum(ns) == want:
            print(f"  ✅ la run reproduce la prediccion EXACTA ({want})")
        else:
            print(f"  ⛔ la run da {sum(ns)} y la prediccion era {want}")
            print("     Sospechoso 1: el arreglo del singleton no llego a este job.")
            print("     Sospechoso 2: la covarianza difiere de la de 103R4.")
            bad += 1

print(f"\n  brazo {arm}: {'⛔ con fallos' if bad else '✅ puertas en verde'}")
sys.exit(1 if bad else 0)
GATE

echo "[tiempo] $ARM completo: $(( (SECONDS-_t0)/60 )) min"
