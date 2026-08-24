#!/bin/bash
# Encadena el chi2 de los tres brazos de la serie 104.  (2026-08-23, noche)
#
# ⚠ SECUENCIAL A PROPOSITO. Cada job pide --mem=300G y un nodo de xlarge tiene
#   ~450 G: dos a la vez no caben. `afterany` (no `afterok`) serializa pase lo
#   que pase, para que un fallo en uno no bloquee a los otros dos.
#
# ⚠ ORDEN: S1 y S3 ya tienen su cinta fina cerrada; S2 la estaba escribiendo
#   cuando se encolo esto, asi que va la ultima. run_chi.sh espera sola por
#   ella (`.lock` + tamano estable, tope 7 h) antes de gastar el precompute.
#
# ⛔ NO SE PUNTUA PARA ELEGIR MALLA: el chi2 baja al engrosar la malla y
#   discrimina AL REVES del conservadurismo. Esto es para REPORTAR el numero.
#
# ⚑ Cada uno deja un sidecar de ~11 GB. Borrarlos en cuanto se lean los
#   parquets (la orden sale impresa al final de cada job).
set -u
cd /share_snc/snc/JuanMonleon/EXFOR/scripts || exit 1

J1=$(sbatch --parsable run_chi.sh S1)                            || exit 1
J3=$(sbatch --parsable --dependency=afterany:$J1 run_chi.sh S3)   || exit 1
J2=$(sbatch --parsable --dependency=afterany:$J3 run_chi.sh S2)   || exit 1

echo "  S1 (k=10 c=3, 5 964 params) -> job $J1"
echo "  S3 (k= 3 c=2, 6 888 params) -> job $J3  (tras $J1)"
echo "  S2 (k= 3 c=3, 6 349 params) -> job $J2  (tras $J3; espera a su cinta fina)"
echo
echo "  seguimiento:  squeue -u \$USER"
echo "  informes:     /share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive/run_104S{1,2,3}/"
