#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH -t 0-12:00:00
#SBATCH -p par_IB
#SBATCH --mem=128G
#SBATCH --job-name=k-S3-10b
# Rehace SOLO el STEP 10b de 104S3, perdido con el NODE_FAIL de node161
# (job 8548705, 23-ago 18:28, tras 4h54). Reproduce las dos llamadas del
# pipeline (exfor_to_endf_research.py:6452) con sus mismos parametros.
set -euo pipefail
RUN=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_104S3_k3c2

source /work/monleon-de-la-jan/myenv/bin/activate
# newcode, NO EXFOR/scripts: las dos copias de build_group_cross.py difieren
# (122141 vs 119470 bytes) y S3 corrio con la de newcode.
cd /share_snc/snc/JuanMonleon/EXFOR/newcode/scripts || exit 1

if ! mkdir "$RUN/.lock" 2>/dev/null; then
  echo "⛔ $RUN ya tiene cerrojo"; exit 1
fi
trap 'rmdir "$RUN/.lock" 2>/dev/null' EXIT

for f in 26-Fe-56g_nominal_a0cross_mg.endf 26-Fe-56g_nominal_a0cross.endf; do
  if [ -e "$RUN/$f" ]; then echo "⛔ $f ya existe, no lo piso"; exit 1; fi
done

echo "--- 10b (1/2): _mg, dead=drop  [estricta] ---"
python -u build_group_cross.py \
  --run-dir "$RUN" \
  --source-endf "$RUN/26-Fe-56g_nominal_mg.endf" \
  --cache "$RUN/.group_cross_cache" \
  --mag-grid fine --null-fill zero --dead-parameters drop \
  --per-order-mesh auto \
  --write-endf "$RUN/26-Fe-56g_nominal_a0cross_mg.endf"

echo "--- 10b (2/2): fina, dead=carry  [adicion: una negativa no tira lo de arriba] ---"
python -u build_group_cross.py \
  --run-dir "$RUN" \
  --source-endf "$RUN/26-Fe-56g_nominal.endf" \
  --cache "$RUN/.group_cross_cache" \
  --mag-grid fine --null-fill zero --dead-parameters carry \
  --per-order-mesh auto \
  --write-endf "$RUN/26-Fe-56g_nominal_a0cross.endf" \
  || echo "⚠ la cinta FINA se nego; la _mg de arriba no se ve afectada"

echo
ls -la "$RUN"/*a0cross*.endf
