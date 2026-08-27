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
  # ⚑ T1 (26-ago): EXACTAMENTE los flags de S2, el entregable. La UNICA variable
  #   es sigma_E de entrada -- el JSON corregido (Smith 5.25 m, Perey 200.191 m,
  #   Salnikov como caja, Cox corroborado) mas los dos canales nuevos del
  #   resolver (cuarentena como caja, default relativo 1.31 % de E) y el marco
  #   CM de Becker. Se compara contra 104S2 y contra nada mas.
  T1) FL=(1 1 0 1 0  3 3); TAG=sigmaEfix    ; SERIE=105 ;;
  # ⚑ T2 (27-ago): los flags de T1 — o sea los de S2 — y DOS arreglos aguas
  #   arriba del ajuste, ninguno de ellos una perilla de covarianza:
  #     1. se rechazan los candidatos Legendre imposibles (|a_l| > 1) antes de
  #        pesarlos por AIC. En la 105T1, 2 bins (1,558 y 1,560 MeV) metian un
  #        componente con |a_1| > 1 que volteaba avg_a_1 y dejaba
  #        sigma(a_1) = 1,667, mas que el rango fisico del coeficiente entero.
  #     2. la sintesis de DATA-ERR de Gkatis 27673002, que corria con sigma 1 %
  #        plana porque la guarda era `is None` y el cargador JSON pone `[]`.
  #   Se compara contra 105T1: los dos mueven el CENTRAL, asi que T2 tampoco
  #   tiene puerta de inercia. Su puerta dura es la del RANGO FISICO.
  T2) FL=(1 1 0 1 0  3 3); TAG=admisible    ; SERIE=106 ;;
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
# ⚑ LOS CUATRO CAMBIOS DE sigma_E (26-ago). Sin testigo, T1 correria con los
#   inputs viejos y saldria identico a 104S2 -- que es justo el resultado que
#   la comparacion quiere poder creer.
import scripts.tof_parameters as tp
if not hasattr(tp, "CORPUS_MEDIAN_REL_SIGMA_E"):
    faltan.append("default relativo (CORPUS_MEDIAN_REL_SIGMA_E)")
if "quarantined_as_box" not in inspect.signature(tp.get_tof_parameters).parameters:
    faltan.append("canal caja para anchos en cuarentena (quarantined_as_box)")
_tof = tp.load_tof_parameters_file(
    "/share_snc/snc/JuanMonleon/EXFOR/exfor_tof_parameters.json")
if (_tof.get("10886002", {}).get("tof") or {}).get("flight_path_m") != 5.25:
    faltan.append("camino de vuelo de Smith 1980 en exfor_tof_parameters.json (5.25 m)")
if "13511004" not in _tof or "40372004" not in _tof:
    faltan.append("entradas nuevas de Perey 13511004 / Salnikov 40372004")
# El marco de Becker vive en la LIBRERIA, no en scripts/: va por wheel y por
# eso se comprueba importando, no leyendo el fuente ([[cluster-venv-is-not-inspectable]]).
try:
    from kika.exfor.database import _parse_c5data_json as _pc5
    if _pc5({"c5data": {"x2": {"fam": "ANG", "x2": [30.0], "ifCM": True}}}).get("is_cm") is not True:
        raise RuntimeError("is_cm no se propaga")
except Exception as _e:
    faltan.append(f"marco CM de X4Pro en el kika INSTALADO ({_e}); "
                  f"pip install --force-reinstall "
                  f"/share_snc/snc/JuanMonleon/EXFOR/kika_dist/kika_nd-0.2.10-py3-none-any.whl")

# ⚑ EL RECHAZO DE CANDIDATOS IMPOSIBLES (27-ago). Sin testigo, un brazo correria
#   con el codigo viejo y volveria a embarcar sigma(a_1) = 1,667 sobre un
#   coeficiente acotado en [-1, 1], que es justo lo que se va a rehacer.
if not hasattr(er, "admissible_degrees"):
    faltan.append("filtro de candidatos imposibles (admissible_degrees)")
elif not er.REJECT_INADMISSIBLE_DEGREES:
    faltan.append("REJECT_INADMISSIBLE_DEGREES esta APAGADO")
else:
    # que ademas FUNCIONE, no solo que exista: un candidato que afirma <mu> = 1,4
    _c0 = 0.25
    _probe = {2: {"coeffs": [_c0, 1.4 * 3 * _c0, 0.2 * 5 * _c0]},
              3: {"coeffs": [_c0, 0.3 * 3 * _c0, 0.2 * 5 * _c0, 0.1 * 7 * _c0]}}
    _keep, _drop = er.admissible_degrees(_probe, [2, 3])
    if _keep != [3] or 2 not in _drop:
        faltan.append(f"admissible_degrees no filtra (keep={_keep}, drop={list(_drop)})")

# ⚑ EL ARREGLO DE GKATIS (27-ago): la sintesis de DATA-ERR estaba guardada por
#   `is None`, y `[]` no es `None`, asi que 27673002 corria con sigma = 1 % plana.
import inspect as _insp
import scripts.uncertainty_manifest as _um
if "if uncertainty_components is None:" in _insp.getsource(_um.apply_manifest_to_exfor):
    faltan.append("arreglo de Gkatis en uncertainty_manifest.py (la guarda sigue en `is None`)")

if faltan:
    for f in faltan:
        print(f"[preflight] ⛔ falta: {f}")
    sys.exit(1)
print("[preflight] presentes: los cuatro cambios de sigma_E (26-ago) + el filtro de "
      "candidatos imposibles y el arreglo de Gkatis (27-ago)")

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

# 2b. Los dos canales de sigma_E NO son de entorno: son constantes de modulo, y
#     es su VALOR el que decide la run. Se afirman aqui para que queden en el
#     log junto a los flags, y para que apagarlos por error no pase inadvertido.
if er.DEFAULT_REL_SIGMA_E is None or er.QUARANTINED_AS_BOX is not True:
    print(f"[preflight] ⛔ canales sigma_E apagados: "
          f"DEFAULT_REL_SIGMA_E={er.DEFAULT_REL_SIGMA_E} "
          f"QUARANTINED_AS_BOX={er.QUARANTINED_AS_BOX}")
    sys.exit(1)
print(f"[preflight] sigma_E: default_rel={100*er.DEFAULT_REL_SIGMA_E:.2f}% de E, "
      f"cuarentena como caja=ON")

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
            "S1": (True, True), "S2": (True, True), "S3": (True, True),
            "T1": (True, True), "T2": (True, True)}[arm]
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
elif arm in ("T1", "T2"):
    # ── T1/T2 MUEVEN EL CENTRAL A PROPOSITO, y por eso NO tienen puerta de inercia.
    #   sigma_E no es una perilla de covarianza: entra en el AJUSTE. El marco CM
    #   de Becker reetiqueta los 14 puntos de 11511009 que caen en el bin 1700,
    #   asi que `nominal_fits.parquet` y `mf34_mean_perbin.npy` TIENEN que
    #   diferir de 102a. Compararlos byte a byte es una puerta que T1 esta
    #   obligado a fallar.
    # ⛔ 26-ago-2026: eso fue exactamente lo que paso. La run termino BIEN a las
    #   9h35, escribio la cinta de 598 MB y reprodujo la prediccion, y esta
    #   puerta la marco "⛔ con fallos" -> `exit 1` -> el chi2 que estaba
    #   encolado con `--dependency=afterok` se quedo en DependencyNeverSatisfied
    #   para siempre. Una puerta que el brazo no puede pasar no es una puerta.
    #   La de VERDAD para T1 es la de mas abajo: que se mueva SOLO lo predicho.
    print("\n  ⚑ T1 mueve el central A PROPOSITO (sigma_E entra en el ajuste):")
    print("    sin puerta de inercia contra 102a. La puerta de T1 es la de abajo.")
else:
    # ── R2..R6, S1..S3: el central no se mueve. Ningun cambio de covarianza puede. ──
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
            "entradas del agrupamiento", "[AVG] model-averaged", "[ADMIS]")
    vistos = []
    for l in txt:
        t = l.strip()
        if any(p in t for p in pats) and t not in vistos:
            vistos.append(t)
    for t in vistos[:60]:
        print("   ", t[:210])
    if len(vistos) > 60:
        print(f"    ... y {len(vistos)-60} lineas mas, en {log[-1].name}")

# ── LA PUERTA DEL RANGO FISICO ──────────────────────────────────────────────
# Es la red que la cadena no tenia: todos los guardianes vigilaban el
# denominador (SNR) o la malla (tope sigma-ratio, cambio de signo, anchura), y
# ninguno miraba si la sigma declarada cabe en el rango que el coeficiente puede
# tomar. La 105T1 embarco sigma(a_1) = 1,667 sobre |a_1| <= 1 sin que saltara
# nada. Ahora `build_mixture_blocks` lo comprueba y lo dice en el log; aqui se
# convierte en rojo del brazo.
if log:
    _admis = [l.strip() for l in txt if "[ADMIS]" in l]
    _viol = [l for l in _admis if "RANGO FISICO VIOLADO" in l]
    _sin = [l for l in _admis if "NINGUN candidato admisible" in l and " 0 bin(s) con" not in l]
    if _viol:
        print("\n  ⛔ RANGO FISICO VIOLADO en la mezcla — la cinta declara una sigma")
        print("     que no cabe en |a_l| <= 1. NO se embarca asi.")
        for l in _viol[:5]:
            print("    ", l[:200])
        bad += 1
    elif any("rango fisico:" in l for l in _admis):
        print("\n  ✅ rango fisico: la mezcla cumple between_var <= 1 - abar^2 "
              "y sigma(a_l) <= 1")
    else:
        print("\n  ⚠ no encuentro la linea [ADMIS] del rango fisico en el log:")
        print("    ¿corrio este brazo con el codigo del 27-ago?")
        bad += 1
    if _sin:
        print(f"  ⚠ {len(_sin)} aviso(s) de bins sin NINGUN candidato admisible "
              f"— hay que mirarlos a mano (no suman fallo).")

# ── T2: contra 105T1, y a proposito NO es pasa/no-pasa por conteo ───────────
# Lo que T2 cambia son DOS cosas aguas arriba del ajuste, y sus alcances se
# predijeron con precisiones distintas:
#   * el filtro de imposibles: EXACTO, 2 bins (1,558 y 1,560 MeV);
#   * Gkatis: Gkatis aparece en 98 de los 1738 bins (76 solo-Gkatis, donde el
#     central se mueve, y 22 compartidos, donde cambian los pesos GLS).
# La suma es una COTA, no un numero: por eso esto informa y no suspende. La
# puerta dura de T2 es la del RANGO FISICO de mas arriba, que si es exacta.
# ⛔ No repetir el error del 26-ago: una puerta que el brazo no puede pasar
#    bloquea la cadena entera.
if arm == "T2":
    ref = out.parent / "new_test_105T1_sigmaEfix"
    print("\n  ⚑ T2 vs 105T1 — filtro de imposibles + Gkatis (informativo)")
    if not ref.exists():
        print(f"  ⚠ no encuentro la referencia {ref}")
    else:
        import pandas as pd
        a_, b_ = (pd.read_parquet(d / "nominal_fits.parquet") for d in (out, ref))
        E = b_.energy_mev.to_numpy()
        cols = [f"c_{l}" for l in range(7)] + [f"avg_a_{l}" for l in range(1, 7)]
        moved = np.zeros(len(E), dtype=bool)
        for c in cols:
            if c in a_ and c in b_:
                x, y = a_[c].to_numpy(float), b_[c].to_numpy(float)
                moved |= ~np.isclose(x, y, rtol=1e-9, atol=1e-12)
        idx = np.flatnonzero(moved)
        print(f"  centrales movidos: {len(idx)} de {len(E)} bins "
              f"(cota esperada ~100: 2 del filtro + hasta 98 de Gkatis)")
        for lab, lo_, hi_ in (("1,558/1,560 (filtro)", 1.5575, 1.5605),):
            n = int(moved[(E >= lo_) & (E <= hi_)].sum())
            print(f"    de ellos en {lab}: {n} (se esperan 2)")
        if len(idx) > 140:
            print("    ⚠ mas de 140 bins movidos: mas de lo que las dos causas "
                  "explican. Mirar antes de embarcar.")
        # lo que este brazo existe para arreglar
        for L in (1,):
            sa = np.load(out / "mf34_std_perbin.npy").reshape(-1, 6).T
            sb = np.load(ref / "mf34_std_perbin.npy").reshape(-1, 6).T
            print(f"  sigma por bin de a_{L}: max {sa[L-1].max():.4f} "
                  f"(105T1: {sb[L-1].max():.4f})")

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

# ── T1: LA COMPARACION QUE DECIDE SI HAY QUE RE-MUESTREAR ───────────────────
# T1 lleva los flags de S2 y difiere de el en UNA cosa: los inputs de sigma_E.
# Asi que este bloque no es una puerta pasa/no-pasa sino la MEDIDA que Juan
# necesita para decidir si el UQ del PWR900 hay que rehacerlo. La prediccion
# offline (26-ago, sobre la rejilla de 104S2):
#   * centrales invariantes salvo UN bin -- el 1700 (E=3.2086 MeV), donde
#     Becker 11511009 aporta 14 puntos que cambian 2.1 % de mediana al leerse
#     en CM. El ajuste nominal usa bordes duros y NO ve sigma_E por experimento.
#   * covarianza: los datasets cuyo sigma_E cambia se llevan <= 2 % del peso
#     por debajo de 3.06 MeV y ~19 % por encima. Y por encima de 3.06 MeV vive
#     el 0.24 % de la varianza MF34 embarcada.
# Si la medida contradice la prediccion, la prediccion estaba mal y hay que
# mirar por que ANTES de tocar el PWR900.
if arm == "T1":
    ref = out.parent / "new_test_104S2_k3c3"
    print(f"\n  ⚑ T1 vs 104S2 -- la UNICA variable es sigma_E de entrada")
    if not ref.exists():
        print(f"  ⛔ no encuentro la referencia {ref}"); bad += 1
    else:
        import pandas as pd
        a_, b_ = (pd.read_parquet(d / "nominal_fits.parquet") for d in (out, ref))
        E = b_.energy_mev.to_numpy()
        # 1. centrales
        cols = [f"c_{l}" for l in range(7)] + [f"avg_a_{l}" for l in range(1, 7)]
        moved = np.zeros(len(E), dtype=bool)
        for c in cols:
            if c in a_ and c in b_:
                x, y = a_[c].to_numpy(float), b_[c].to_numpy(float)
                moved |= ~np.isclose(x, y, rtol=1e-9, atol=1e-12)
        idx = np.flatnonzero(moved)
        print(f"  centrales: {len(idx)} de {len(E)} bins se mueven")
        if len(idx):
            print(f"    bins: {idx[:12].tolist()}{' ...' if len(idx) > 12 else ''}")
            print(f"    E MeV: {np.round(E[idx][:12], 4).tolist()}")
        # ⚑ ESTA es la puerta de T1, y es MAS FUERTE que la de inercia: no pide
        #   que el central no se mueva, pide que se mueva EXACTAMENTE donde la
        #   prediccion offline dijo. Un bin de mas significa que sigma_E se ha
        #   colado en algo que no es el marco CM de Becker, y eso hay que
        #   mirarlo antes de creerse nada de esta run.
        PRED = {1700}
        got = set(idx.tolist())
        print("    (prediccion: SOLO el bin 1700, E=3.2086, por el marco CM de Becker)")
        if got == PRED:
            print("  ✅ se mueve EXACTAMENTE lo predicho: 1 bin, el 1700")
        else:
            print(f"  ⛔ el central NO se mueve como se predijo:"
                  f" sobran {sorted(got - PRED)[:12]}, faltan {sorted(PRED - got)}")
            bad += 1

        # 2. covarianza MF34, separada por el corte de Cierjacks
        sa = np.load(out / "mf34_std_perbin.npy").reshape(6, -1)
        sb = np.load(ref / "mf34_std_perbin.npy").reshape(6, -1)
        if sa.shape != sb.shape:
            print(f"  ⚠ formas distintas {sa.shape} vs {sb.shape}: la malla cambio, "
                  f"comparacion por bin no aplicable")
        else:
            rel = np.abs(sa - sb) / np.maximum(np.abs(sb), 1e-30)
            for lab, m in (("<= 3.06 MeV", E <= 3.06), ("> 3.06 MeV", E > 3.06)):
                r = rel[:, m]
                print(f"  sigma MF34 {lab:11s}: max |d| {100*r.max():7.2f} %   "
                      f"p99 {100*np.percentile(r, 99):6.2f} %   "
                      f"mediana {100*np.median(r):5.2f} %")
            v = sb ** 2
            print(f"  peso: el tramo > 3.06 MeV es el {100*v[:, E>3.06].sum()/v.sum():.2f} % "
                  f"de la varianza MF34 total")
            print()
            print("  --- LECTURA PARA EL UQ DEL PWR900 ---")
            # ⚑ EL MAXIMO SIN PONDERAR NO ES LA PREGUNTA, y el 26-ago estuvo a
            #   punto de costar una decision equivocada: salio 18.05 % contra una
            #   prediccion de "< 1 %", pero eran 5 bins de a_6 (2.357-2.362 MeV)
            #   cuyo peso en la varianza MF34 es 0.0000 %. La prediccion hablaba
            #   de PESO ("los datasets que cambian se llevan <= 2 %"), asi que un
            #   max sobre 10 428 parametros contestaba a otra pregunta. a_6 se
            #   lleva ~0.5 % de la varianza de la DCS: un 18 % ahi no mueve nada.
            #   Lo que decide si hay que re-muestrear es el PESO, no el maximo.
            big = rel > 0.01
            w_mov = 100 * v[big].sum() / v.sum()
            worst = 100 * rel[:, E <= 3.06].max()
            n_ord = sorted(int(l) + 1 for l in np.flatnonzero(big.any(axis=1)))
            print(f"  max sin ponderar, <= 3.06 MeV : {worst:.2f} %   "
                  f"({int(big.sum())} de {rel.size} parametros pasan del 1 %, "
                  f"ordenes {n_ord})")
            print(f"  PESO en varianza MF34 de todo lo que se mueve > 1 % : {w_mov:.4f} %")
            if w_mov < 0.1:
                print(f"  ✅ lo que se mueve pesa {w_mov:.4f} % de la varianza MF34 "
                      f"declarada. NO justifica re-muestrear el PWR900.")
            else:
                print(f"  ⚠ lo que se mueve pesa {w_mov:.4f} % de la varianza MF34.")
                print("     Por encima del 0.1 % hay que mirar por que antes de decidir.")

print(f"\n  brazo {arm}: {'⛔ con fallos' if bad else '✅ puertas en verde'}")
sys.exit(1 if bad else 0)
GATE

echo "[tiempo] $ARM completo: $(( (SECONDS-_t0)/60 )) min"
