#!/bin/bash
#SBATCH -J njoy600
#SBATCH -p par_IB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH -t 0-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr

# Re-run the NJOY half of a perturbation set at 600 K.
#
#     sbatch run_reprocess_ace.sh <set> [--limit 4] [--source-endf ...] [...]
#
# <set> is the PERTURBATION output directory (the one with endf/ pendf/ ace/
# xsdir/), NOT the MCNP run directory. Everything after it is passed straight
# to reprocess_ace.py.
#
# ⚑ ONE NODE, 40 PROCESSES, NOT MPI. Each sample is an independent NJOY
# process; the parallelism is a python Pool over samples, so this asks for one
# task with 40 cpus and not 40 tasks. `-n 40` would start 40 copies of the
# whole pass, each racing the others to write the same ACE.
#
# ⚑ MEMORY. NJOY is ~0.5-1 GB per process, so 40 of them fit in 64 G with
# room. par_IB does not take more than 250 G per job in any case
# (kika-workspace memory: queue-limits-4-jobs-and-par-ib-250g).
#
# ⚑ DISK. Each worker opens a temporary directory INSIDE <set> for the NJOY
# tapes and removes it when done; the transient peak is roughly 1 GB per
# process. The preflight below refuses to start without 100 GB free.
#
# ⚑ RESUMABLE. A sample whose ACE already carries the target temperature is
# skipped, and the temperature is read from the ACE header — so re-running
# this after a timeout or a partial failure picks up exactly what is left.

set -u

SET="${1:?uso: sbatch run_reprocess_ace.sh <set> [opciones de reprocess_ace.py]}"
shift

SCRIPTS_DIR="${KIKA_SCRIPTS_DIR:-/share_snc/snc/JuanMonleon/EXFOR/scripts}"
VENV="${KIKA_VENV:-/work/monleon-de-la-jan/myenv}"
NJOY="${KIKA_NJOY:-/soft_snc/NJOY/2016.78/bin/njoy}"

# --- preflight: fallar en los primeros segundos, no a la hora 6 -------------
[[ -d "$SET" ]]                    || { echo "[preflight] ⛔ no existe el conjunto: $SET"; exit 1; }
[[ -d "$SET/endf" || -d "$SET/pendf" ]] \
                                   || { echo "[preflight] ⛔ $SET no tiene endf/ ni pendf/. ¿Es el directorio de la run de MCNP en vez del de la perturbación?"; exit 1; }
[[ -x "$NJOY" ]]                   || { echo "[preflight] ⛔ NJOY no ejecutable: $NJOY"; exit 1; }
[[ -f "$VENV/bin/activate" ]]      || { echo "[preflight] ⛔ no hay venv en $VENV"; exit 1; }
[[ -f "$SCRIPTS_DIR/reprocess_ace.py" ]] \
                                   || { echo "[preflight] ⛔ falta reprocess_ace.py en $SCRIPTS_DIR"; exit 1; }

# df puede no traer --output en cualquier coreutils; si no se puede medir se
# avisa y se sigue, que es distinto de medir y encontrar poco.
free_gb=$(df -BG "$SET" 2>/dev/null | awk 'NR==2 {gsub(/[^0-9]/,"",$4); print $4}')
if [[ -z "$free_gb" ]]; then
    echo "[preflight] ⚠ no puedo leer el disco libre de $SET; sigo sin comprobarlo."
elif (( free_gb < 100 )); then
    echo "[preflight] ⛔ solo ${free_gb} GB libres; los temporales de NJOY caben mal."
    exit 1
else
    echo "[preflight] disco libre en el conjunto: ${free_gb} GB"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$SCRIPTS_DIR" || exit 1
python -c "import kika, sys; print('[preflight] kika', getattr(kika,'__version__','?'), 'desde', kika.__file__)" || exit 1

echo "[preflight] cpus=${SLURM_CPUS_PER_TASK:-1}"
echo

# --- inventario: qué hay, y a qué temperatura está de verdad ---------------
python reprocess_ace.py "$SET" --njoy "$NJOY" "$@" || exit 1

# --- la pasada -------------------------------------------------------------
python reprocess_ace.py "$SET" \
    --njoy "$NJOY" \
    --nprocs "${SLURM_CPUS_PER_TASK:-1}" \
    --apply "$@"
rc=$?

echo
echo "-- el maestro de cada réplica, releído"
# fix_xsdir_dpa es idempotente y en ensayo no escribe: si dice "a corregir: 0"
# y las entradas resuelven, los xsdir siguen enteros después de la reescritura.
python fix_xsdir_dpa.py "$SET/xsdir" --quiet || true

exit $rc
