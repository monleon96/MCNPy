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
# the modes run, then the comparison table. Everything except which part of the
# forward model is resolution-averaged is held fixed, so the differences are
# attributable to the fold convention alone.
#
# Which modes to run is the first argument (or $FOLD_MODES), space separated.
# Default is the reduced set, which answers both questions the sweep is for:
#   product  <sigma . F(a_l)>   the physically correct fold  (run 82; reused)
#   factors  <sigma> . F(<a_l>) what a naive implementation does -> costs how much?
#   none     sigma(E0) F(a_l(E0)) no fold at all -> tests the assumed widths
# Add `sigma al` to also decompose WHICH factor carries the curvature.
#
# Each new mode costs one 11 GB .eval_cov.npz sidecar (measured on run 82), so:
#   sbatch run_fold_sweep.sh                                  # reduced, 2 modes, ~22 GB
#   sbatch run_fold_sweep.sh "product factors sigma al none"  # full,    4 modes, ~44 GB
#
# Any mode whose parquet already exists is skipped (run 82 produced `product`) —
# delete the parquet to force a rebuild.

set -o pipefail

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

declare -A METHODOLOGY=(
  [product]="predictive"
  [factors]="predictive_factors"
  [sigma]="predictive_foldsigma"
  [al]="predictive_foldal"
  [none]="predictive_nofold"
)

MODES="${1:-${FOLD_MODES:-product factors none}}"

# Fail before spending a day of wall clock on a typo.
for mode in $MODES; do
    if [[ -z "${METHODOLOGY[$mode]+x}" ]]; then
        echo "unknown fold mode '$mode' — pick from: ${!METHODOLOGY[*]}" >&2
        exit 2
    fi
done

echo "=== fold-mode sweep over: $MODES ==="

for mode in $MODES; do
    target="${CHI2_DIR}/chi2_data_predictive_82${SUFFIX[$mode]}.parquet"
    if [[ -f "$target" ]]; then
        echo "=== FOLD_MODE=$mode — parquet already present, skipping precompute ==="
        continue
    fi
    echo "=== FOLD_MODE=$mode — precompute ==="
    FOLD_MODE=$mode python precompute_chi2_predictive.py || exit 1
done

# Analyse exactly the modes requested, not all five — chi2_analysis_cluster.py
# would otherwise look for parquets this run never built.
METHS=""
for mode in $MODES; do
    METHS="${METHS:+$METHS,}${METHODOLOGY[$mode]}"
done

echo "=== analysis over: $METHS ==="
KIKA_CHI2_METHODOLOGIES="$METHS" python chi2_analysis_cluster.py || exit 1

# compare_fold_modes.py tabulates whatever summary.json files exist and reports
# the rest as [missing], so it is safe to run on a subset.
echo "=== comparison table ==="
python compare_fold_modes.py || exit 1

unset KIKA_UNCERTAINTY_MANIFEST_PATH
deactivate
