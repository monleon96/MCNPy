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
# CURRENT JOB: RUN 99 -- un cambio de signo de a_l es una frontera
# ===========================================================================
# La run 98 aplico el criterio donde toca (decision 154 casillas con SNR < 1,
# emision 0) y la malla comprimio de verdad: 648/660/620/554/588/637, -12.4 %.
# Pero murio al emitir el cruzado, con max|rel| 0.9945 -> 260.4.
#
# La causa, y es la mitad que le faltaba al criterio:
#
#     rel_S = Var_S / mean_S^2 = 1 / SNR_S^2
#
# El DP restringia la VARIANZA (Var(media) >= promediado independiente) y nada
# restringia la MEDIA. La media de un segmento que cruza un cero de a_l es una
# DIFERENCIA, no un promedio: puede salir arbitrariamente pequena sin que sus
# terminos lo sean, y rel diverge. Medido sobre la run 98, (sum|u|/|sum u|)^2:
#
#     a_1 4.4   a_2 1.0   a_3 11.3   a_4 62.8   a_5 2.6e5   a_6 6835
#
# a_2 no tiene ningun cruce, su factor es exactamente 1, y es el unico orden que
# el DP no toco. La correlacion es perfecta.
#
# Cambio de la 99: un cambio de signo de a_l es un punto de corte fijo, igual
# que un grupo ausente. Con cada grupo grueso de un solo signo, sum|u| = |sum u|
# y por tanto |rel_S| <= max|rel| termino a termino: el colapso pasa a ser una
# CONTRACCION de la forma relativa, no algo que un guard tenga que cazar.
# Se aplica en los dos sitios -- en el DP y en la emision -- porque ven valores
# distintos (el DP la mezcla sobre 660 grupos, la emision el soporte del MC
# sobre 703), y el teorema hay que cumplirlo con el u con el que se colapsa.
#
# Cuesta poco: los cruces son 26/0/40/128/198/245 de 660 grupos, asi que la
# regla todavia permite bajar a 644 parametros de 3960.
#
# Esperado: entre -8 % y -12 % en entradas MF34, y amp = 1 en los seis ordenes.
# ⚠ ~5 h, ~7.7 GB. Nada escribe en 92/93/94/96/97/98.
# ⚠ La run 98 se puede terminar en 90 s con run_chi.sh, que repara su malla en la
#   emision (-7.9 %). Esta run existe para que lo que se envie sea la malla que
#   el criterio ELIGIO, no una reparada a posteriori.
# Por que cada cosa: docs/handoff_per_order_mesh.md en kika-workspace.

R99=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_99_signcut
R98=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_98_meshraw
R97=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_97_perordermesh
R96=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_96_wheel029
mkdir -p $R99

# --- PRE-FLIGHT: que los scripts desplegados HOY estan ahi -------------------
# Ejercita el codigo, no lee una etiqueta: el fallo que esto atrapa es enviar el
# job antes de copiar, y entonces la run seria la 97 otra vez con otro nombre.
grep -q "MF34_MESH_FROM_RAW" exfor_to_endf_research.py || {
    echo "[preflight] exfor_to_endf_research.py no tiene MF34_MESH_FROM_RAW. No envies."
    exit 1; }
python - <<'PREFLIGHT99' || exit 1
import sys, numpy as np
try:
    from scripts.per_order_mesh import order_cut_indices, per_order_meshes
    from scripts.build_group_cross import order_emission_weights, _snap_to_base
except Exception:
    from per_order_mesh import order_cut_indices, per_order_meshes
    from build_group_cross import order_emission_weights, _snap_to_base

n = 12
idx = order_cut_indices(np.eye(n), np.full(n, 10.0), np.ones(n),
                        np.ones(n, bool), 3, 9)
ok_dp = np.array_equal(idx, np.arange(n + 1))     # el borde de la ventana vive
print(f"[preflight] DP, borde de ventana : {'OK' if ok_dp else 'NO'}")

e = np.linspace(1e6, 4e6, 5)
U = order_emission_weights(e, e, np.diff(e), np.full(4, 2.0))
ok_u = np.allclose(U, np.eye(4))                  # misma malla => identidad
print(f"[preflight] colapso por orden    : {'OK' if ok_u else 'NO'}")

# El fallo de la 97, reproducido: la cinta guarda seis cifras significativas.
raw = np.array([846822.0, 847499.9999999999, 848500.0, 850500.0])
base = np.array([float(f"{v:.5e}") for v in raw])
ok_snap = (not np.isin(raw, base).all()) and np.array_equal(
    _snap_to_base(raw, base, "a_1"), base)
print(f"[preflight] snap de seis cifras   : {'OK' if ok_snap else 'NO'}")

# El cambio de la 98: sobre un objeto capado el DP no puede fusionar nada.
edges, means = np.linspace(1e6, 4e6, 13), np.full(12, 0.2)
crudo = np.outer(np.ones(12), np.ones(12)) * 0.995 + 0.005 * np.eye(12)
s = np.minimum(np.sqrt(np.diag(crudo)), np.abs(means)) / np.sqrt(np.diag(crudo))
capado = crudo * np.outer(s, s)
n_crudo = len(per_order_meshes(edges, crudo, means, 1)[1]) - 1
n_capado = len(per_order_meshes(edges, capado, means, 1)[1]) - 1
ok_raw = n_crudo < 12 and n_capado == 12
print(f"[preflight] crudo {n_crudo} vs capado {n_capado} grupos : "
      f"{'OK' if ok_raw else 'NO'}")

# El cambio de la 99: no se fusiona a traves de un cero de a_l.
edges = np.linspace(1e6, 4e6, 13)
means = np.r_[np.full(6, 0.2), np.full(6, -0.2)]        # un solo cruce, en el 6
crudo = np.outer(np.ones(12), np.ones(12)) * 0.995 + 0.005 * np.eye(12)
m = per_order_meshes(edges, crudo, means, 1)[1]
ok_sgn = float(edges[6]) in set(map(float, m))
print(f"[preflight] frontera en el cambio de signo : {'OK' if ok_sgn else 'NO'}")

if not (ok_dp and ok_u and ok_snap and ok_raw and ok_sgn):
    print("[preflight] los scripts de hoy NO llegaron al share. No envies.")
    sys.exit(1)
PREFLIGHT99

KIKA_OUTPUT_DIR=$R99/ \
KIKA_MF34_PER_ORDER_MESH=1 \
KIKA_MF34_MESH_FROM_RAW=1 \
KIKA_STOP_AFTER_NOMINAL_FITS=0 \
KIKA_DEGREE_WEIGHT_FLOOR=0.01 \
KIKA_SAVE_RAW_KW_PARQUET=1 \
KIKA_SAVE_MF34_COV_SIDECARS=1 \
KIKA_SAVE_PERBIN_PARQUET=1 \
    python exfor_to_endf_research.py || exit 1

# --- EL GATE ---------------------------------------------------------------
# Falla por lo que TIENE que ser cierto; reporta lo que estamos midiendo. El
# numero de grupos es la incognita de esta run, asi que no es un pass/fail.
python - "$R99" "$R98" "$R96" <<'GATE99' || exit 1
import sys, filecmp, re
from pathlib import Path
import numpy as np
new, r97, r96 = (Path(p) for p in sys.argv[1:4])
bad = 0

print("\n-- 1. LO QUE NO PUEDE MOVERSE: nada anterior a la malla")
# La malla se elige despues de los ajustes y del canal de magnitud, y la cinta
# fina se escribe antes de que exista. Byte a byte contra la 97.
for f in ["26-Fe-56g_nominal.endf", "nominal_fits.parquet",
          "mf33_absolute_covariance.npy", "mf33_relative_covariance.npy",
          "mf33_c0_nominal.npy", "mf33_c0_host.npy", "mf33_energy_grid_ev.npy",
          "mf33_multigroup_relative_covariance.npy"]:
    a, b = new / f, r97 / f
    if not a.exists() or not b.exists():
        print(f"   FALTA  {f}"); bad += 1; continue
    if filecmp.cmp(a, b, shallow=False):
        print(f"   OK   {f}")
    else:
        print(f"   ⛔ {f} CAMBIO -- el cambio de la 98 toco algo que no es MF34")
        bad += 1

print("\n-- 2. QUE LA MALLA SE DECIDIO SOBRE EL OBJETO SIN CAPAR")
logs = sorted(new.glob("exfor_to_endf_*.log"))
txt = logs[-1].read_text() if logs else ""
got = dict(re.findall(r"\[mesh\] (decision|emission) matrix: (\d+)/", txt))
if len(got) != 2:
    print("   ⛔ el log no trae las dos lineas [mesh]: el script desplegado es el viejo")
    bad += 1
else:
    d, e = int(got["decision"]), int(got["emission"])
    print(f"   casillas con SNR < 1:  decision {d}   emision {e}")
    if d <= e:
        print("   ⛔ la matriz de decision esta tan capada como la de emision:")
        print("      KIKA_MF34_MESH_FROM_RAW no llego. La malla no puede fusionar.")
        bad += 1
    else:
        print(f"   OK   el criterio ve {d} casillas que sobre la capada no existen")

print("\n-- 3. LA MALLA (se REPORTA, no se puntua)")
z = new / "mf34_per_order_mesh.npz"
if not z.exists():
    print("   ⛔ no hay mf34_per_order_mesh.npz: el flag no llego"); bad += 1
else:
    n = [len(np.load(z)[f"e_{l}"]) - 1 for l in range(1, 7)]
    r98n = [648, 660, 620, 554, 588, 637]
    print(f"   grupos por orden: {'/'.join(map(str, n))}")
    print(f"   la run 98 dio   : {'/'.join(map(str, r98n))}")
    print(f"   entradas MF34   : {sum(n)**2:,} vs {sum(r98n)**2:,} "
          f"({100*(sum(n)**2/sum(r98n)**2 - 1):+.1f} %)")
    if n == r98n:
        print("   ⛔ malla IDENTICA a la de la 97: el cambio no ha tenido efecto")
        bad += 1

print("\n-- 4. EL ENTREGABLE EXISTE (lo que mato a la 97)")
for f in ["26-Fe-56g_nominal_mg.endf", "26-Fe-56g_nominal_a0cross_mg.endf"]:
    a, b = new / f, r96 / f
    if not a.exists():
        print(f"   ⛔ FALTA {f}"); bad += 1; continue
    print(f"   OK   {f}: {a.stat().st_size:,} bytes "
          f"({100*(a.stat().st_size/b.stat().st_size - 1):+.1f} % vs run 96)")

if bad:
    print(f"\n❌ RUN 99 FALLA en {bad} comprobacion(es). NO se puntua.")
    sys.exit(1)
print("\n✅ RUN 99 PASA. Leer los grupos del punto 3 antes de decidir si se puntua:")
print("   si la compresion es marginal, la conclusion es que la resolucion")
print("   enviada SI esta soportada, y eso cierra §10.8 como negativo medido.")
GATE99

echo "--- criterios de lectura, no pass/fail ---"
grep -E "\[mesh\]|per-order mesh|max \|c34_rel\||MF34 entries|Fine bins:|Compression:|Near-zero regular" $R99/exfor_to_endf_*.log | tail -40
df -h /share_snc | tail -1

# --- SIGUIENTE: la puntuacion va en run_chi.sh, no aqui --------------------
# run_chi.sh ya esta preparado; hay que cambiarle R97 por R98 y el tag antes de
# lanzarlo. La run 98 SI se puntua: no es una reproduccion.
