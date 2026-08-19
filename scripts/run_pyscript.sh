#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH -t 1-12:00:00
#SBATCH -p par_IB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#SBATCH --job-name=kika-research

# Sin --mem a proposito: el job corre el MC solo. Si lo mata el OOM, ahi hay que
# mirar. cpus-per-task y N_PROCS ya no pueden discrepar: el script lee
# SLURM_CPUS_PER_TASK. Detalle en run_pyscript.archive.sh.

source /work/monleon-de-la-jan/myenv/bin/activate
cd /share_snc/snc/JuanMonleon/EXFOR/scripts/
export KIKA_UNCERTAINTY_MANIFEST_PATH=/share_snc/snc/JuanMonleon/EXFOR/uncertainty_manifest_diagonal_cierjacks.yaml

# Trabajos anteriores y por que se tomo cada decision: run_pyscript.archive.sh

# ===========================================================================
# CURRENT JOB: RUN 97 -- una malla por orden Legendre, con el cruzado dentro
# ===========================================================================
# Unico cambio frente a la run 96: KIKA_MF34_PER_ORDER_MESH=1. El resto son los
# env de la 96 letra por letra. No hay que reinstalar nada; lo desplegado hoy es
# solo scripts/. El cruzado va DENTRO: build_group_cross recoge solo la malla de
# mf34_per_order_mesh.npz y colapsa forma y cruzado con la misma U.
#
# Esperado en el log:  679/703/637/472/299/105 grupos, MF34 entries -52.9 %,
#                      max |c34_rel| 3.9e+07 -> 1.0
#
# ⚠ El _mg intermedio NO conserva la compresion (merge_mf34 cuadra los LB=6
#   ragged sobre la union); el entregable _a0cross si, porque se re-emite.
# ⚠ ~4.5 h, ~7.7 GB. Nada escribe en new_test_92/93/94/96.
# Por que cada cosa: docs/handoff_per_order_mesh.md en kika-workspace.

R97=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_97_perordermesh
R96=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_96_wheel029
mkdir -p $R97

# --- PRE-FLIGHT: que los scripts desplegados HOY estan ahi -------------------
# Ejercita el codigo, no lee una etiqueta: el fallo que esto atrapa es enviar el
# job antes de copiar, y entonces la run seria la 96 otra vez con otro nombre.
python - <<'PREFLIGHT97' || exit 1
import sys, numpy as np
try:
    from scripts.per_order_mesh import order_cut_indices, collapse_relative_per_order
except Exception:
    from per_order_mesh import order_cut_indices, collapse_relative_per_order
n = 12
idx = order_cut_indices(np.eye(n), np.full(n, 10.0), np.ones(n),
                        np.ones(n, bool), 3, 9)
ok_dp = np.array_equal(idx, np.arange(n + 1))     # el borde de la ventana vive
print(f"[preflight] DP, borde de ventana : {'OK' if ok_dp else 'NO'}")
try:
    from scripts.build_group_cross import order_emission_weights, per_order_shape_grids
except Exception:
    from build_group_cross import order_emission_weights, per_order_shape_grids
e = np.linspace(1e6, 4e6, 5)
U = order_emission_weights(e, e, np.diff(e), np.full(4, 2.0))
ok_u = np.allclose(U, np.eye(4))                  # misma malla => identidad
print(f"[preflight] colapso por orden    : {'OK' if ok_u else 'NO'}")
if not (ok_dp and ok_u):
    print("[preflight] los scripts de hoy NO llegaron al share. No envies.")
    sys.exit(1)
PREFLIGHT97

KIKA_OUTPUT_DIR=$R97/ \
KIKA_MF34_PER_ORDER_MESH=1 \
KIKA_STOP_AFTER_NOMINAL_FITS=0 \
KIKA_DEGREE_WEIGHT_FLOOR=0.01 \
KIKA_SAVE_RAW_KW_PARQUET=1 \
KIKA_SAVE_MF34_COV_SIDECARS=1 \
KIKA_SAVE_PERBIN_PARQUET=1 \
    python exfor_to_endf_research.py || exit 1

# --- EL GATE: lo que la malla NO puede tocar, y lo que si -------------------
# Escrito antes de la run. La malla es una eleccion de REPRESENTACION aplicada
# despues de los ajustes, asi que no puede mover ni un central ni un byte del
# canal de magnitud. Lo unico que puede cambiar es MF34.
python - "$R97" "$R96" <<'GATE97' || exit 1
import sys, filecmp
from pathlib import Path
import numpy as np
new, ref = Path(sys.argv[1]), Path(sys.argv[2])
bad = 0

print("\n-- 1. LO QUE NO PUEDE MOVERSE: centrales y canal de magnitud")
for f in ["nominal_fits.parquet", "mf33_absolute_covariance.npy",
          "mf33_relative_covariance.npy", "mf33_c0_nominal.npy",
          "mf33_c0_host.npy", "mf33_energy_grid_ev.npy",
          "mf33_multigroup_relative_covariance.npy"]:
    a, b = new / f, ref / f
    if not a.exists() or not b.exists():
        print(f"   FALTA  {f}"); bad += 1; continue
    if filecmp.cmp(a, b, shallow=False):
        print(f"   OK   {f}")
    else:
        print(f"   ⛔ {f} CAMBIO -- la malla toco algo que no es MF34"); bad += 1

print("\n-- 2. LA MALLA: la eligio la MEZCLA, no el estimador ganador")
z = new / "mf34_per_order_mesh.npz"
if not z.exists():
    print("   ⛔ no hay mf34_per_order_mesh.npz: el flag no llego"); bad += 1
else:
    m = np.load(z)
    n = [len(m[f"e_{l}"]) - 1 for l in range(1, 7)]
    print(f"   grupos por orden: {'/'.join(map(str, n))}   (esperado 679/703/637/472/299/105)")
    # El discriminante es a_6: con las medias de la mezcla el DP baja a ~105;
    # alimentado con medias winner-take-all se queda en ~701, porque entonces
    # a_6 solo esta declarado en 74 de 703 grupos y casi todo es corte fijo.
    if n[5] > 200:
        print(f"   ⛔ a_6 tiene {n[5]} grupos, no ~105: el DP esta recibiendo")
        print(f"      medias WINNER-TAKE-ALL, no las de la mezcla. NO PUNTUAR.")
        bad += 1
    else:
        print(f"   OK   a_6 en {n[5]} grupos: la malla viene de la mezcla")

print("\n-- 3. LAS CINTAS: MF34 tiene que haber ENCOGIDO")
for f in ["26-Fe-56g_nominal_mg.endf", "26-Fe-56g_nominal_a0cross_mg.endf"]:
    a, b = new / f, ref / f
    if not a.exists():
        print(f"   ⛔ FALTA {f}"); bad += 1; continue
    sa, sb = a.stat().st_size, b.stat().st_size
    print(f"   {f}: {sa:,} vs {sb:,} bytes ({100*(sa/sb-1):+.1f} %)")
    if f.endswith("a0cross_mg.endf") and sa >= sb:
        print(f"   ⛔ el ENTREGABLE no encogio: la emision ragged no ocurrio")
        bad += 1

if bad:
    print(f"\n❌ RUN 97 FALLA en {bad} comprobacion(es). NO se puntua.")
    sys.exit(1)
print("\n✅ RUN 97 PASA. La malla por orden esta en el entregable, con el")
print("   cruzado colapsado sobre la MISMA U que los bloques de forma.")
GATE97

echo "--- criterios de lectura, no pass/fail ---"
grep -E "per-order mesh|max \|c34_rel\||MF34 entries|Fine bins:|Compression:|Valid mask:|XCORR" $R97/exfor_to_endf_*.log | tail -30
df -h /share_snc | tail -1

# --- SIGUIENTE: la puntuacion va en run_chi.sh, no aqui --------------------
#   KIKA_THIS_WORK_DIR=$R97 KIKA_MF33_MF34_CROSS_DIR=$R97 KIKA_RUN_TAG=97 \
#       python precompute_chi2_predictive.py
# ⚠ La run 97 SI se puntua: no es una reproduccion.
