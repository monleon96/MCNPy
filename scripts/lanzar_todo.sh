#!/bin/bash
# ============================================================================
#  Lanza los seis brazos de la noche, uno por job, REPARTIDOS EN DOS COLAS.
#
#      cd /share_snc/snc/JuanMonleon/EXFOR/newcode/scripts && ./lanzar_todo.sh
#
#  ⚑ El `cd` NO es opcional: SLURM fija el cwd al de submision, y el preflight
#    aborta si el paquete `scripts` no resuelve dentro de newcode/.
#
#  Para lanzar solo algunos:   ./lanzar_todo.sh R1 R2 R4
#  Para ver que haria:         DRY=1 ./lanzar_todo.sh
#
# ============================================================================
#  EL REPARTO, Y POR QUE ASI
# ============================================================================
#
#   xlarge / 24 cpus  ->  R1  R2  R4
#   par_IB / 40 cpus  ->  R3  R5  R6
#
# ⚑ LAS DOS COLAS ESTAN PROBADAS CON ESTE MISMO PIPELINE. `xlarge` a 24 cpus es
#   donde corrieron 102a/102b/102c/102cp (~6-7 h por run). `par_IB` a 40 cpus es
#   donde corrieron 101a/101b (~4 h), y ahi los cpus estan libres -- por eso 40 y
#   no 24: la misma run sale mas rapida.
#
# ⚑ EL REPARTO NO ES ARBITRARIO. Los tres brazos de los que depende todo lo
#   demas van a la cola con mas historial reciente:
#     R1 es la unica puerta DURA de la noche (byte a byte contra 102a);
#     R2 es la referencia de R3, R5 y R6 -- sin ella los deltas no existen;
#     R4 es la unica ruta que no se ha corrido nunca de punta a punta.
#   Si `par_IB` diera un problema de cola, seguiriamos teniendo la puerta, la
#   referencia y la ruta nueva. Al reves nos quedariamos sin nada que comparar.
#
# ⚑ `--mem=128G` VA EXPLICITO EN LAS DOS y NO se pisa aqui. Un `--mem` implicito
#   (comentado, en realidad) ya mato la run 100 por OOM a las 6h24. 128 GB son
#   5x el pico MEDIDO de 25 GB y caben en las dos colas: el default de par_IB es
#   4 G/cpu, o sea 160 GB a 40 cpus, que es con lo que corrieron 101a/101b.
# ============================================================================
set -o pipefail
cd "$(dirname "$0")" || exit 1

if [ ! -f exfor_to_endf_research.py ]; then
  echo "⛔ no estoy en el sandbox"; exit 1
fi

ARMS=("$@")
# ⚑ POR DEFECTO, LA SERIE 104 (23-ago). La 103 esta cerrada y sus directorios
#   tienen cinta, asi que relanzar R1..R6 muere en la guarda de "ya tiene una
#   cinta" -- que es lo correcto, pero no es lo que se quiere por defecto.
# ⚑ 26-ago: por defecto SOLO T1. La serie 104 esta cerrada y sus directorios
#   tienen cinta, asi que relanzarlos muere en la guarda -- correcto, pero no
#   es lo que se quiere. T1 es el unico brazo pendiente.
# ⚑ 27-ago: por defecto SOLO T2, el re-run con el filtro de candidatos
#   imposibles y el arreglo de Gkatis. T1 ya tiene cinta y moriria en la guarda.
[ ${#ARMS[@]} -eq 0 ] && ARMS=(T2)

declare -A DESC=(
  [S1]="R4 + arreglo del singleton — UNA variable, 5964 params predichos"
  [S2]="k=3 c=3 — EL ENTREGABLE, 6349 params predichos, ~441 MiB"
  [S3]="k=3 c=2 — variante conservadora, 6888 params predichos, ~519 MiB"
  [T1]="sigma_E corregida — flags de S2, la UNICA variable son los inputs"
  [T2]="filtro de candidatos imposibles (|a_l|>1) + Gkatis — re-run del entregable"
  [R1]="inercia — byte a byte contra 102a (LA puerta dura)"
  [R2]="base    — RESTORE + suelo, codigo nuevo (referencia de R3/R5/R6)"
  [R3]="sig-ratio por orden"
  [R4]="malla en UNA etapa desde el fino (ruta nunca corrida)"
  [R5]="suelo con dependencia en energia"
  [R6]="todo encendido — el entregable candidato"
)
declare -A PART=( [R1]=xlarge [R2]=xlarge [R4]=xlarge
                  [R3]=par_IB [R5]=par_IB [R6]=par_IB
                  [S1]=par_IB [S2]=xlarge [S3]=par_IB [T1]=xlarge [T2]=xlarge )
declare -A CPUS=( [R1]=24 [R2]=24 [R4]=24
                  [R3]=40 [R5]=40 [R6]=40
                  [S1]=40 [S2]=24 [S3]=40 [T1]=24 [T2]=24 )
# ⚑ S2, QUE ES EL ENTREGABLE, VA SOLO A `xlarge`. Es la cola donde los tres
#   brazos de anoche entraron sin esperar; en `par_IB` el sexto se quedo 6 h en
#   cola y para cuando entro su directorio ya tenia cerrojo. S1 y S3 son
#   controles y pueden permitirse la espera.
#
# ⛔ LIMITES DE COLA (Juan, 23-ago-2026), y hay que contarlos a mano porque nada
#   los comprueba:
#     * como mucho 4 jobs en `par_IB` y 4 en `xlarge` a la vez;
#     * `par_IB` NO PASA DE 250 GB por job. Los brazos piden 128 G y caben; el
#       chi2 pide 300 G y NO cabe -- por eso `run_chi.sh` esta en `xlarge`.
#   Con este reparto la serie 104 ocupa 1 de xlarge y 2 de par_IB, y deja sitio
#   para los dos chi2 en xlarge sin llegar al tope.

echo "Sandbox : $(pwd)"
echo "Salidas : /share_snc/snc/JuanMonleon/ENDF_samples/new_test_<103|104><ARM>_<tag>"
echo "Reparto : xlarge/24 -> T2 (topes: 4+4 jobs, par_IB <= 250 G)"
echo
declare -A IDS
for A in "${ARMS[@]}"; do
  P=${PART[$A]}; C=${CPUS[$A]}
  if [ -z "$P" ]; then echo "  ⛔ '$A' no es R1..R6, S1..S3, T1 ni T2"; continue; fi
  CMD=(sbatch --parsable --job-name="k-$A" -p "$P" --cpus-per-task="$C"
       --export=ALL,KIKA_ARM="$A" run_arm.sh)
  if [ -n "$DRY" ]; then
    echo "  [DRY] ${CMD[*]}"
    printf "        %-6s %-8s %2s cpus   %s\n" "$A" "$P" "$C" "${DESC[$A]}"
    continue
  fi
  ID=$("${CMD[@]}"); rc=$?
  if [ $rc -ne 0 ] || [ -z "$ID" ]; then
    echo "  ⛔ $A NO submitido en $P (sbatch devolvio $rc)"
    echo "     Reintenta a mano:  ${CMD[*]}"
    continue
  fi
  IDS[$A]=$ID
  printf "  ✅ %-3s job %-9s %-8s %2s cpus   %s\n" "$A" "$ID" "$P" "$C" "${DESC[$A]}"
done

echo
echo "Seguimiento:  squeue -u \$USER"
echo "Salida:       tail -f slurm-<jobid>.out"
echo
echo "⚠ Si una cola rechaza -t 1-00:00:00, relanza ESE brazo con un -t menor:"
echo "  sbatch -t 12:00:00 -p <cola> --cpus-per-task=<n> --job-name=k-RN \\"
echo "         --export=ALL,KIKA_ARM=RN run_arm.sh"
echo "⚠ Si un brazo muere por el cerrojo, es que ya hay otro job en su directorio."
