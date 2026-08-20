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
# ⚠⚠ `--per-order-mesh off` EN LOS TRES, INCLUIDO EL mg. La run 99 escribio
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
# ⛔ RIESGO CONOCIDO: `build_group_cross --mag-grid fine` con una cinta FINA
# como `--source-endf` NUNCA SE HA CORRIDO. Las runs 85/86 puntuaron el fino
# pero sin cruzado, y todo el cruzado hasta hoy se ha construido sobre el mg.
# El paso 1 esta encadenado con `|| exit 1` justo por eso: si esa ruta falla,
# cuesta el parseo y nada mas, no la hora y media del precompute.
#
# ⛔ Y EN c2 EL PASO 1 ES UNA MEDIDA, NO UN TRAMITE. `carry_dead_parameters`
# proyecta al cono PSD solo bajo dos barras DERIVADAS de las 7 cifras
# significativas de MF34 (conditioning `lam_min/lam_max` y masa inventada
# `sum|lam-|/traza`, ambas a 5e-7). Por encima de cualquiera de las dos levanta
# `SystemExit` y el job muere aqui, que es lo que se quiere: en el mg gastaba el
# 3.0 % y el 1.4 % de cada barra, pero en la rejilla fina hay ~2.5x mas slots
# muertos y eso no esta medido todavia.
#
# DISCO: tres cintas `_a0cross` (~1.8 GB las finas, ~0.35 GB la mg) y tres
# sidecars `eval_cov` de ~11 GB. ~37 GB en total sobre los 673 G libres.
# ⚑ BORRAR LOS TRES SIDECARS en cuanto se hayan leido los parquets.
# ===========================================================================

set -o pipefail

VARIANT="${1:-}"
case "$VARIANT" in
  c0) SRC_ENDF=26-Fe-56g_nominal_mg.endf ; DEAD=drop  ; TAG=99c0mg   ;;
  c1) SRC_ENDF=26-Fe-56g_nominal.endf    ; DEAD=drop  ; TAG=99c1fine ;;
  c2) SRC_ENDF=26-Fe-56g_nominal.endf    ; DEAD=carry ; TAG=99c2fine ;;
  *)  echo "uso: sbatch run_c_fine.sh {c0|c1|c2}"
      echo "  c0  mg 660,   dead=drop   -- el ancla"
      echo "  c1  fino 1738, dead=drop  -- aisla el cambio de rejilla"
      echo "  c2  fino 1738, dead=carry -- aisla la restitucion de a_5/a_6"
      exit 2 ;;
esac

source /work/monleon-de-la-jan/myenv/bin/activate
cd /share_snc/snc/JuanMonleon/EXFOR/scripts/ || exit 1

export KIKA_UNCERTAINTY_MANIFEST_PATH=/share_snc/snc/JuanMonleon/EXFOR/uncertainty_manifest_diagonal_cierjacks.yaml

R99=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_99_signcut
OUT=/share_snc/snc/JuanMonleon/chi2/${TAG}
OUT_ENDF=$OUT/26-Fe-56g_nominal_a0cross_${VARIANT}.endf

echo "=========================================================="
echo "  variante   : $VARIANT"
echo "  source     : $R99/$SRC_ENDF"
echo "  dead params: $DEAD"
echo "  tag        : $TAG"
echo "  salida     : $OUT_ENDF"
echo "  job        : ${SLURM_JOB_ID:-<sin slurm>}  en ${SLURMD_NODENAME:-$(hostname)}"
echo "=========================================================="

# --- PREFLIGHT -------------------------------------------------------------
# Barato, y cada linea es un fallo que ya ha costado un job entero.

# (a) La entrada existe. `build_group_cross --mag-grid fine` abre exactamente
#     estos, verificado contra lo que el script abre de verdad.
for f in "$SRC_ENDF" legendre_samples_tmc.parquet mf33_c0_samples.parquet \
         nominal_fits.parquet mf33_energy_grid_ev.npy \
         mf33_relative_covariance.npy mf33_c0_host.npy; do
    [ -f "$R99/$f" ] || { echo "⛔ falta $R99/$f"; exit 1; }
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
    --run-dir "$R99" \
    --source-endf "$R99/$SRC_ENDF" \
    --write-endf "$OUT_ENDF" \
    --mag-grid fine \
    --null-fill zero \
    --per-order-mesh off \
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
