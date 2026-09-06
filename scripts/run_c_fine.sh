#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH -t 1-12:00:00
#SBATCH -p par_IB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#SBATCH --job-name=kika-cfine
#
# ⚑ SIN `--mem`, A PROPOSITO Y POR DECISION EXPLICITA (2026-08-20). El job se
# queda con el default de mem-per-cpu de par_IB por los 40 cpus, que es
# exactamente para lo que estan esos 40 (`precompute_chi2_predictive.py:109`
# clava OMP/OPENBLAS/MKL a 1 hilo antes de cargar numpy, asi que los cores NO
# compran velocidad en el chi2 -- compran memoria).
# `precompute_chi2_predictive.py:168` avisa de que los 100G que bastan para el
# `_mg` NO bastan para el fino. Si un paso muere por OOM, esta es la linea:
##SBATCH --mem=300G
#   Fallback conocido-bueno si par_IB no concede la reserva: `-p xlarge
#   --mem=500G`, que es lo que uso la run 86 con la cadena entera encadenada.
#
# ⚠ `build_group_cross.py` NO clava los hilos, asi que el paso 1 SI usa los 40
#   cores. En la rejilla fina la tabla de diagnostico hace ~11 eigh de
#   12424 x 12424; con BLAS multihilo son minutos, con 1 hilo serian horas.

# ===========================================================================
# PASOS C DEL PLAN: la rejilla fina pasa a ser la referencia
# Plan: kika-workspace/docs/plan_fine_reference_and_100k.md §C
# ===========================================================================
#
#   sbatch -J kika-c0 run_c_fine.sh c0     # el ancla
#   sbatch -J kika-c1 run_c_fine.sh c1     # aisla el cambio de REJILLA
#   sbatch -J kika-c2 run_c_fine.sh c2     # aisla la RESTITUCION de a_5/a_6
#
# Sobre OTRA run del pipeline, con el directorio como SEGUNDO argumento (el
# prefijo del tag sale del nombre del directorio: new_test_100_100k -> `100`):
#
#   R100=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_100_100k
#   sbatch -J kika-c0-100 run_c_fine.sh c0 $R100
#   sbatch -J kika-c1-100 run_c_fine.sh c1 $R100
#   sbatch -J kika-c2-100 run_c_fine.sh c2 $R100
#
# ===========================================================================
# VARIANTES DE MALLA (m1/m2/m3): cuanto se puede AGRUPAR el objeto fino
# ===========================================================================
#
#   sbatch -J kika-m1 run_c_fine.sh m1     # sigma_E x1, k=3    7949 params
#   sbatch -J kika-m2 run_c_fine.sh m2     # sigma_E x1, k=10   6746 params
#   sbatch -J kika-m3 run_c_fine.sh m3     # sigma_E x2, k=10   6006 params
#
# Los tres son `c2` MAS una malla por orden: cinta FINA de origen, cruzado,
# `--dead-parameters carry`, `--mag-grid fine`. La UNICA variable contra c2 es
# el agrupamiento, asi que c2 es su ancla y la pareja es de una sola cosa.
#
# QUE MALLA ES CADA UNA. Dos restricciones, las dos derivadas del dato y no
# elegidas (`notebooks/mf34_mesh/`, medido sobre `mf34_cov_combined.npy` de la
# run 99, que ES cov_emission en la rejilla fina):
#   (i)  anchura del grupo <= res_factor * sigma_E, la resolucion TOF del propio
#        experimento: por debajo, los bins no son medidas independientes.
#   (ii) a_l constante dentro de k sigmas de su miembro mas preciso. El fichero
#        MF34 es RELATIVO, asi que lo homogeneo tiene que ser el DENOMINADOR.
# El corte por cambio de signo de a_l es el caso limite de (ii) y ya estaba.
#
# ⚑ MEDIDO ANTES DE LANZAR, y por eso NO hay una cuarta variante mas agresiva:
# quitar (i) del todo solo baja de 6746 a 5582 parametros (-17 %, 127 MiB) y a
# cambio la varianza plegada a la resolucion del dato se va a 165x en a_5. Lo
# que ata mas alla de la resolucion es (ii), no (i). El criterio de aceptacion
# es CONSERVADURISMO -- que el fichero no declare MENOS varianza de la que el
# objeto fino tiene a la resolucion a la que se lee:
#
#   malla    params   MiB    sub-declara (peor)   sobre-declara (peor)
#   m1k3       7949   773    0.65  (1.5x menos)   1.28
#   m2k10      6746   621    0.27  (3.7x menos)   1.97
#   m3r2       6006   538    0.34  (2.9x menos)   2.54
#   c2 fino   10428  1145    1.00  (exacto)       1.00
#   c0 hoy     3960   327    0.01  (100x menos)   354
#
# ⚠ Ninguna es estrictamente conservadora; la de HOY es con diferencia la peor.
#
# PREDICCION, FIJADA ANTES DE MEDIR: los tres chi2 caeran dentro de +-2 % de c2
# pese a que la peor sub-declaracion va de 1.5x a 3.7x. Si acierta, queda MEDIDO
# que el chi2 no puede arbitrar la malla -- se sienta en la mediana del cociente
# plegado, que es ~1 en todas -- y §10.8 se cierra con un negativo medido en vez
# de con un argumento. El control agresivo ya esta medido y es `c0`.
#
# ⛔ RIESGO REAL, no tramite: `--per-order-mesh auto` + `--dead-parameters carry`
# + cinta FINA de origen no se han corrido nunca juntos. La malla por orden solo
# se ha ejercitado sobre la cinta de 660 que emite el pipeline. El paso 1 es una
# MEDIDA.
#
# ⚠ POR QUE UN DIRECTORIO POR MALLA. `build_group_cross` toma la malla de
# `<run-dir>/mf34_per_order_mesh.npz` y no acepta una ruta: `auto` dispara sobre
# ese nombre exacto. La run 99 ya tiene el suyo, sobre la rejilla de 660, y
# sobreescribirlo destruiria el registro de la run y cambiaria en silencio lo que
# `auto` hace para cualquier otro. Cada malla vive en su directorio con symlinks
# a lo que se LEE de la run 99, mas su npz de verdad.
# ⚑ En esta configuracion el run-dir es de SOLO LECTURA, verificado en el codigo:
# las tres `np.save(run_dir / "mf33_mf34_cross_group_*.npy")` de
# `build_group_cross` estan en la rama `else` de `if _fine_mag`, y aqui vamos con
# `--mag-grid fine`. Los symlinks no se pueden sobreescribir a traves.
# ⚠ Los symlinks los pone ESTE script, no `make_meshes.py`: el share montado en
# WSL no admite `symlink()`, asi que desde alli solo se escribe el npz.
#
# ✅ CORREGIDO 2026-08-21 — LA RUN 100 SI PRODUCE LA CINTA FINA CON CRUZADO.
# El texto de abajo era cierto hasta el deploy de `exfor_to_endf_research.py` de
# las 17:26 del 20-ago, y la run 100 se relanzo a las 17:31, o sea CON ese
# cambio. El paso 10b recorre ahora `cross_targets = [(mg, "drop", strict),
# (nominal, "carry", strict=False)]`, asi que emite las DOS: el analogo de `c0`
# y el analogo de `c2`. Verificado en el codigo desplegado, no supuesto.
# ⚠ `CROSS_MAG_GRID=fine` sigue SIN significar "la cinta fina": nombra el eje de
# la MAGNITUD (MF33). El eje de la FORMA lo fija `--source-endf`.
# ~~LA RUN 100 NO PRODUCE LA CINTA FINA CON CRUZADO POR SI SOLA~~ (historico):
# `_write_cross_term_endf` llamaba a `build_cross_and_write_endf` con la cinta
# `_mg` y sin `dead_parameters`, o sea solo el analogo de `c0`.
#
# ⚠ SOLO SE LEE UNA PAREJA CRUZANDO RUNS: `99c2fine` vs `100c2fine`, el mismo
# punto del mismo objeto con la misma config salvo N_SAMPLES, o sea p/n de 1.043
# a 0.104. Cualquier otra pareja entre runs mueve la run y la representacion a la
# vez (§L16/§L18).
#
# LOS TRES SON INDEPENDIENTES Y SE LANZAN A LA VEZ. No comparten ni cinta, ni
# cache de parseo, ni parquet, ni sidecar, ni directorio de informe
# (`chi2_analysis_cluster.py` aisla cada uno en `<REPORT_DIR>/run_<RUN_ID>/`).
# Lo unico compartido es la run 99, que se lee y no se toca.
#
# QUE PREGUNTA CADA UNO. Tres puntos del MISMO objeto, leidos en DOS parejas, y
# cada pareja mueve UNA sola cosa:
#
#   c0  mg 660  + cruzado, dead=drop     ─┐ el cambio de REJILLA
#   c1  fino 1738 + cruzado, dead=drop   ─┘
#   c1  fino 1738 + cruzado, dead=drop   ─┐ la RESTITUCION de a_5/a_6
#   c2  fino 1738 + cruzado, dead=carry  ─┘
#
# ⚠⚠ POR QUE HACE FALTA c0 Y NO VALE `predictive_91_cross` NI `98raw`. Aquellas
# son OTRA run del pipeline. Leer c1 contra ellas mueve la run Y la rejilla a la
# vez, que es exactamente el confound por el que se retiraron §L16 y §L18. El
# ancla de estos tres es c0 y nada mas.
#
# ⚠⚠ `--per-order-mesh off` EN LOS TRES C, INCLUIDO EL mg. (En m1/m2/m3 es `auto`
# a proposito y sobre el npz de SU directorio, no sobre el de la run 99.) La run 99 escribio
# `mf34_per_order_mesh.npz` sobre la rejilla de 660 (`w_l` es 649x660 ...
# 646x660) y el default `auto` la aplicaria. Al mg seria legitimo, pero al FINO
# le colapsaria los 1738 bins sobre una malla elegida ENCIMA de los 660 -- el
# regroup-de-un-regroup que este plan elimina -- y lo haria en silencio, porque
# `_snap_to_base` engancha sin quejarse (el rango es el mismo, 846822-4075000
# eV). Con `off` los tres salen en UNA sola rejilla y la unica diferencia entre
# c0 y c1 es fina contra mg.
#
# ⚠ `--mag-grid fine` en los tres: es el eje de la MAGNITUD (MF33), no el de la
# forma. Es la unica eleccion que hace del plegado una congruencia (A_mag = I,
# o sea el bloque c33 del conjunto ES el MF33 que se emite). El eje de la FORMA
# lo fija `--source-endf`, y ahi esta la diferencia c0/c1.
#
# ⚠ Sin `KIKA_MF34_NULL_MASK`, a proposito: la mascara vive en la malla de 703
# de la run 86, `eval_covariance` la coloca POR ENERGIA y una mascara caduca se
# aplicaria en silencio. La retirada ya la lleva la cinta, via `--null-fill
# zero` mas el complemento de `live` en `write_consistent_mf34`.
#
# ⚠ `KIKA_MF33_MF34_CROSS_FROM_FILE=1`, NUNCA `..._CROSS_DIR`: la ruta del
# sidecar declara `is_relative=False` contra una familia MF34 relativa, y es la
# que mato las runs 87-90.
#
# ⚠ `KIKA_THIS_WORK_ENDF` explicito: el default es `_nominal_mg.endf` y
# puntuaria la cinta SIN cruzado en vez del entregable.
#
# VETO, igual que siempre: **JEFF y JENDL al 0.00000 % entre los tres**. Si se
# mueven, ninguna de las dos parejas es de una sola variable y no se lee nada.
# V2 identico a cuatro decimales es el otro control, y es gratis.
#
# PREDICCIONES, fijadas ANTES de medir:
#   c1 vs c0 : la run 86 midio el mismo cambio de rejilla SIN cruzado y V4 se
#              movia hasta un 8 % (only KS -8.0 %, no_KS_no_Cierjacks +6.2 %).
#              Aqui hay ademas cruzado, asi que puede ser mayor.
#   c2 vs c1 : `carry` restituye parametros de forma que hoy se declaran con
#              CERO incertidumbre teniendo la pipeline su sigma. En el mg eran
#              1307 de 4218 y rank34 subia de 2660 a 3481. Mete incertidumbre
#              REAL donde hoy hay ceros ⇒ V4 tiene que BAJAR.
#
# ✅ RIESGO RETIRADO (2026-08-20, run 99). `build_group_cross --mag-grid fine`
# con una cinta FINA como `--source-endf` no se habia corrido nunca; corrio tres
# veces y la fila F aterrizo en el CONTROL en las tres (sigma_max(K) = 1.000000
# fino / 0.999993 mg, lam_min ~ -5e-16), con la puerta c33 en 2.08e-16 relativo.
# El `|| exit 1` del paso 1 se queda: sigue siendo la diferencia entre perder el
# parseo y perder la hora y media del precompute.
#
# ⛔ Y EN c2 EL PASO 1 ES UNA MEDIDA, NO UN TRAMITE. `carry_dead_parameters`
# proyecta al cono PSD solo bajo dos barras DERIVADAS de las 7 cifras
# significativas de MF34 (conditioning `lam_min/lam_max` y masa inventada
# `sum|lam-|/traza`, ambas a 5e-7). Por encima de cualquiera de las dos levanta
# `SystemExit` y el job muere aqui, que es lo que se quiere.
# MEDIDO 2026-08-20 sobre la run 99: en el mg gastaba el 3.0 % y el 1.4 % de cada
# barra; en la rejilla FINA gasta el 10.9 % y el 8.7 % -- pasa, con holgura.
# ⚠ Lo que ese margen NO cubre: el recorte lleva 1278 autovalores negativos a
# cero (282 en el mg) y mueve la peor varianza declarada un 1.317e-02 relativo
# (9.8e-05 en el mg), ~5 ordenes por encima del suelo de 7 cifras del formato.
# Las barras miden el ESPECTRO, no la entrada, asi que ese numero no lo puertea
# nadie. La direccion si esta determinada y es CONSERVATIVA: la proyeccion es
# C' = C + sum|lam-|*vv^T, o sea C mas una matriz PSD, y la diagonal solo puede
# SUBIR. Es un +1.3 %, no un -1.3 %. Queda como decision de procedencia.
#
# DISCO: tres cintas `_a0cross` (~1.8 GB las finas, ~0.35 GB la mg) y tres
# sidecars `eval_cov` de ~11 GB. ~37 GB en total sobre los 673 G libres.
# ⚑ BORRAR LOS TRES SIDECARS en cuanto se hayan leido los parquets.
# ===========================================================================

set -o pipefail

VARIANT="${1:-}"

# ⚑ SEGUNDO ARGUMENTO: el directorio de la run del pipeline. Por defecto la 99,
#   asi que `run_c_fine.sh c2` sigue siendo EXACTAMENTE lo que se corrio el
#   2026-08-20 y las tres cintas de entonces se reproducen sin tocar nada.
#   Para la run 100:  run_c_fine.sh c2 /share_snc/.../new_test_100_100k
RUN_DIR="${2:-/share_snc/snc/JuanMonleon/ENDF_samples/new_test_99_signcut}"
RUN_DIR="${RUN_DIR%/}"

# El prefijo del tag sale del nombre del directorio (`new_test_100_100k` -> 100)
# y NO se pasa a mano: un tag mal tecleado es exactamente el fallo que mato a
# las runs 85, 89 y al job 8497757, y el preflight (c) lo cazaria tarde.
RUN_LABEL="${KIKA_C_RUN_LABEL:-$(basename "$RUN_DIR" | sed -n 's/^new_test_0*\([0-9][0-9]*\)_.*$/\1/p')}"
[ -n "$RUN_LABEL" ] || {
    echo "⛔ no puedo derivar el prefijo del tag de '$(basename "$RUN_DIR")'."
    echo "   Se espera 'new_test_<N>_<algo>'. Si el directorio no sigue el patron,"
    echo "   pasa el prefijo a mano:  KIKA_C_RUN_LABEL=100 sbatch run_c_fine.sh c2 <dir>"
    exit 2; }

MESH=""
case "$VARIANT" in
  c0) SRC_ENDF=26-Fe-56g_nominal_mg.endf ; DEAD=drop  ; TAG=${RUN_LABEL}c0mg   ;;
  c1) SRC_ENDF=26-Fe-56g_nominal.endf    ; DEAD=drop  ; TAG=${RUN_LABEL}c1fine ;;
  c2) SRC_ENDF=26-Fe-56g_nominal.endf    ; DEAD=carry ; TAG=${RUN_LABEL}c2fine ;;
  m1) SRC_ENDF=26-Fe-56g_nominal.endf    ; DEAD=carry ; TAG=${RUN_LABEL}m1k3   ; MESH=m1k3  ;;
  m2) SRC_ENDF=26-Fe-56g_nominal.endf    ; DEAD=carry ; TAG=${RUN_LABEL}m2k10  ; MESH=m2k10 ;;
  m3) SRC_ENDF=26-Fe-56g_nominal.endf    ; DEAD=carry ; TAG=${RUN_LABEL}m3r2   ; MESH=m3r2  ;;
  *)  echo "uso: sbatch run_c_fine.sh {c0|c1|c2|m1|m2|m3} [run-dir]"
      echo "  c0  mg 660,   dead=drop   -- el ancla"
      echo "  c1  fino 1738, dead=drop  -- aisla el cambio de rejilla"
      echo "  c2  fino 1738, dead=carry -- aisla la restitucion de a_5/a_6"
      echo "  m1  fino + malla sigma_E x1, k=3   -- 7949 params"
      echo "  m2  fino + malla sigma_E x1, k=10  -- 6746 params"
      echo "  m3  fino + malla sigma_E x2, k=10  -- 6006 params"
      echo "  run-dir: por defecto new_test_99_signcut"
      exit 2 ;;
esac

source /work/monleon-de-la-jan/myenv/bin/activate
cd /share_snc/snc/JuanMonleon/EXFOR/scripts/ || exit 1

export KIKA_UNCERTAINTY_MANIFEST_PATH=/share_snc/snc/JuanMonleon/EXFOR/uncertainty_manifest_diagonal_cierjacks.yaml

OUT=/share_snc/snc/JuanMonleon/chi2/${TAG}
OUT_ENDF=$OUT/26-Fe-56g_nominal_a0cross_${VARIANT}.endf

# El directorio que `build_group_cross` va a tratar como "la run". Para c0/c1/c2
# es la run de verdad y `off`, o sea el camino de siempre sin tocar. Para m1/m2/m3
# es el directorio de la malla, y `auto` dispara sobre su npz.
MESH_ROOT="${KIKA_MESH_ROOT:-/share_snc/snc/JuanMonleon/ENDF_samples/mesh99}"
if [ -n "$MESH" ]; then
    CROSS_RUN_DIR="$MESH_ROOT/$MESH"
    PER_ORDER=auto
else
    CROSS_RUN_DIR="$RUN_DIR"
    PER_ORDER=off
fi

echo "=========================================================="
echo "  variante   : $VARIANT"
echo "  run-dir    : $RUN_DIR"
echo "  cross-dir  : $CROSS_RUN_DIR  (malla: ${MESH:-ninguna}, --per-order-mesh $PER_ORDER)"
echo "  source     : $CROSS_RUN_DIR/$SRC_ENDF"
echo "  dead params: $DEAD"
echo "  tag        : $TAG"
echo "  salida     : $OUT_ENDF"
echo "  job        : ${SLURM_JOB_ID:-<sin slurm>}  en ${SLURMD_NODENAME:-$(hostname)}"
echo "=========================================================="

# --- PREFLIGHT -------------------------------------------------------------
# Barato, y cada linea es un fallo que ya ha costado un job entero.

# (a0) El directorio de la run existe. Sin esto los siete `[ -f ]` de abajo
#      fallan uno a uno y el mensaje no dice que el problema es el argumento.
[ -d "$RUN_DIR" ] || { echo "⛔ no existe el run-dir: $RUN_DIR"; exit 1; }

# (a0-bis) El directorio de la malla, montado aqui y comprobado por (a). Los
#     symlinks se crean en el CLUSTER porque el share montado en WSL no admite
#     `symlink()`; `make_meshes.py` solo deja el npz. Si `ln -s` fallara tambien
#     aqui, esto muere en segundos en vez de despues del paso 1.
if [ -n "$MESH" ]; then
    [ -f "$CROSS_RUN_DIR/mf34_per_order_mesh.npz" ] || {
        echo "⛔ falta $CROSS_RUN_DIR/mf34_per_order_mesh.npz."
        echo "   Generarlo con kika-workspace/notebooks/mf34_mesh/make_meshes.py"
        exit 1; }
    for f in "$SRC_ENDF" legendre_samples_tmc.parquet mf33_c0_samples.parquet \
             nominal_fits.parquet mf33_energy_grid_ev.npy \
             mf33_relative_covariance.npy mf33_c0_host.npy \
             mf33_multigroup_grid_ev.npy mf33_multigroup_relative_covariance.npy; do
        [ -e "$CROSS_RUN_DIR/$f" ] && continue
        [ -e "$RUN_DIR/$f" ] || continue
        ln -s "$RUN_DIR/$f" "$CROSS_RUN_DIR/$f" || {
            echo "⛔ no puedo enlazar $f en $CROSS_RUN_DIR."
            echo "   Si el filesystem no admite symlinks, copiar a mano."
            exit 1; }
    done
    echo "  malla      : $(ls "$CROSS_RUN_DIR" | wc -l) entradas en $CROSS_RUN_DIR"
fi

# (a) La entrada existe. `build_group_cross --mag-grid fine` abre exactamente
#     estos, verificado contra lo que el script abre de verdad.
for f in "$SRC_ENDF" legendre_samples_tmc.parquet mf33_c0_samples.parquet \
         nominal_fits.parquet mf33_energy_grid_ev.npy \
         mf33_relative_covariance.npy mf33_c0_host.npy; do
    [ -f "$CROSS_RUN_DIR/$f" ] || { echo "⛔ falta $CROSS_RUN_DIR/$f"; exit 1; }
done

# (b) El `build_group_cross.py` desplegado es el del hilo B. Sin esto, c2
#     correria con el codigo viejo y produciria una copia de c1 sin avisar.
python build_group_cross.py --help 2>&1 | grep -q -- "--dead-parameters" \
  || { echo "⛔ el build_group_cross.py desplegado no tiene --dead-parameters:"
       echo "   falta desplegar el hilo B. /deploy-cluster"; exit 1; }

# (c) La entrada del registro existe. Esto, y solo esto, es lo que mato a las
#     runs 85, 89 y al job 8497757: el chi2 corre hora y media y muere al final
#     buscando un sidecar que no es el suyo.
grep -q "\"predictive_${TAG}\"" chi2_analysis_cluster.py \
  || { echo "⛔ 'predictive_${TAG}' no esta en chi2_analysis_cluster.py."
       echo "   Registrarlo ANTES de gastar el precompute."; exit 1; }

# (d) Disco. El sidecar son ~11 GB y la cinta fina ~1.8 GB.
#     Si `df` no da un numero (montaje raro, coreutils viejo) se AVISA y se
#     sigue: una puerta que no sabe medir no debe abortar el job.
FREE_GB=$(df -BG --output=avail /share_snc 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$FREE_GB" ]; then
    echo "  disco libre: ${FREE_GB} G"
    [ "$FREE_GB" -ge 50 ] \
      || { echo "⛔ menos de 50 G libres en /share_snc"; exit 1; }
else
    echo "  ⚠ no he podido medir el disco libre en /share_snc; sigo."
    df -h /share_snc | tail -1
fi

mkdir -p "$OUT" || exit 1

# --- PASO 1: la cinta con los bloques a_0 ----------------------------------
# En la rejilla fina el agregador de la magnitud es la IDENTIDAD: esto no
# agrupa nada, solo anade los bloques cruzados al MF34 que la pipeline calculo.
# `--cache` propio por variante: los tres leen la MISMA cinta de origen en c1 y
# c2, asi que compartir cache seria una carrera de escritura sobre el mismo
# `.npz`. La clave del cache ya lleva directorio + digest, o sea que no colisiona
# por contenido; lo que se evita aqui es la concurrencia.
echo
echo "--- PASO 1: build_group_cross ($VARIANT) ---"
python -u build_group_cross.py \
    --run-dir "$CROSS_RUN_DIR" \
    --source-endf "$CROSS_RUN_DIR/$SRC_ENDF" \
    --write-endf "$OUT_ENDF" \
    --mag-grid fine \
    --null-fill zero \
    --per-order-mesh "$PER_ORDER" \
    --dead-parameters "$DEAD" \
    --cache "$OUT/.group_cross_cache" || exit 1

[ -s "$OUT_ENDF" ] || { echo "⛔ el paso 1 no dejo cinta"; exit 1; }
ls -la "$OUT_ENDF"

# --- PASO 2: el precompute -------------------------------------------------
# ~1.5-2 h y un sidecar de ~11 GB. El parseo del MF34 fino son ~11 min en la
# cinta de 843 MB; en la `_a0cross` fina (~1.8 GB) cuenta ~20 min.
echo
echo "--- PASO 2: precompute_chi2_predictive ($TAG) ---"
KIKA_THIS_WORK_DIR=$OUT \
KIKA_THIS_WORK_ENDF=$(basename "$OUT_ENDF") \
KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
KIKA_RUN_TAG=$TAG \
    python -u precompute_chi2_predictive.py || exit 1

# --- PASO 3: la puntuacion -------------------------------------------------
echo
echo "--- PASO 3: chi2_analysis_cluster ($TAG) ---"
KIKA_CHI2_METHODOLOGIES=predictive_$TAG \
KIKA_CHI2_RUN_ID=$TAG \
    python -u chi2_analysis_cluster.py || exit 1

echo
echo "=========================================================="
echo "  ✅ $VARIANT LISTO"
echo "  informe : /share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive/run_${TAG}/"
echo "  parquet : /share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_${TAG}.parquet"
echo "  ⚑ BORRAR el sidecar en cuanto se lea el parquet:"
echo "    rm /share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_${TAG}.parquet.eval_cov.npz"
echo "=========================================================="
df -h /share_snc | tail -1

unset KIKA_UNCERTAINTY_MANIFEST_PATH
deactivate
