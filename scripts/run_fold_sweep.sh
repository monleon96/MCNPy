#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH -t 1-00:00:00
#SBATCH -p xlarge
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#SBATCH --job-name=kika-foldsweep

# Fold-mode sweep for the predictive scenario.
#
# Runs precompute_chi2_predictive.py once per FOLD_MODE, then the analysis over
# all five, then the comparison table. Everything except which part of the
# forward model is resolution-averaged is held fixed, so the differences are
# attributable to the fold convention alone.
#
# `product` is skipped if its parquet already exists (run 82 produced it) —
# delete it to force a rebuild.
#
# Cost: 4 extra precomputes at ~8 GB sidecar each, ~35 GB total.

source /work/monleon-de-la-jan/myenv/bin/activate
cd /share_snc/snc/JuanMonleon/EXFOR/scripts/

export KIKA_UNCERTAINTY_MANIFEST_PATH=/share_snc/snc/JuanMonleon/EXFOR/uncertainty_manifest_diagonal_cierjacks.yaml

CHI2_DIR=/share_snc/snc/JuanMonleon/chi2

declare -A SUFFIX=(
  [product]=""
  [factors]="_factors"
  [sigma]="_foldsigma"
  [al]="_foldal"
  [none]="_nofold"
)

for mode in product factors sigma al none; do
    target="${CHI2_DIR}/chi2_data_predictive_82${SUFFIX[$mode]}.parquet"
    if [[ -f "$target" ]]; then
        echo "=== FOLD_MODE=$mode — parquet already present, skipping precompute ==="
        continue
    fi
    echo "=== FOLD_MODE=$mode — precompute ==="
    FOLD_MODE=$mode python precompute_chi2_predictive.py || exit 1
done

echo "=== analysis over all five fold modes ==="
KIKA_CHI2_METHODOLOGIES="predictive,predictive_factors,predictive_foldsigma,predictive_foldal,predictive_nofold" \
    python chi2_analysis_cluster.py || exit 1

echo "=== comparison table ==="
python compare_fold_modes.py || exit 1

unset KIKA_UNCERTAINTY_MANIFEST_PATH
deactivate
