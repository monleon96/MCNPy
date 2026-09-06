#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00
#SBATCH -p xlarge
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#SBATCH --job-name=kika-dss-91

set -euo pipefail

source /work/monleon-de-la-jan/myenv/bin/activate
cd /share_snc/snc/JuanMonleon/EXFOR/scripts/

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python corpus_dss.py 91_cross \
    --max-n 30000 \
    --resume \
    --out /share_snc/snc/JuanMonleon/chi2/dss_91_cross.csv

deactivate
