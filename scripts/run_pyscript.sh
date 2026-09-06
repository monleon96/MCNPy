#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=500G
#SBATCH -t 5-00:00:00
#SBATCH -p xlarge
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#SBATCH --job-name=kika-research

# ⛔⛔ 2026-08-20 15:59 -- LA RUN 100 (job 8522071) MURIO POR OOM A LAS 6 h 24 m.
# Murio en el PASO 2 del MC ("Two-pass: running per-bin MC for variance"); el
# paso 1 habia terminado. No dejo nada: el run-dir solo tiene nominal_fits.
#
# POR QUE. Este bloque decia "--mem VUELVE ... Se piden 300G" y LA DIRECTIVA
# `#SBATCH --mem` NO ESTABA EN EL FICHERO. El comentario describia una reserva
# que nadie pidio, asi que el job tomo el default de par_IB. Y el preflight, que
# existia justo para cazar esto, leyo SLURM_MEM_PER_NODE, lo encontro VACIO y
# se limito a AVISAR ([[slurm-mem-per-node-is-empty-here]]). Un aviso no es una
# puerta. Las dos cosas quedan arregladas abajo.
#
# ⚑ 2026-08-20 (original): --mem VUELVE (las runs 93-99 corrian sin el a
# proposito, porque el MC solo cabia de sobra) y -t pasa de 36 h a 5 dias.
#
# 10x muestras no escalan solo el tiempo: `all_samples_perbin` es
# {replica: {bin: array(6)}} y cuesta 196 B/entrada MEDIDOS -- 48 B de datos y
# el resto overhead de dict + ndarray. A 100k x 1738 bins son 34 GB, y hay dos
# stores asi, mas sample_matrix (100k x 10428 x 8 = 8.3 GB) que np.cov copia y
# centra. Pico estimado 80-120 GB, contra los ~3 GB del MC a 10k.
# ⚑ PICO MEDIDO, NO ESTIMADO (2026-08-20, tras el OOM). Construidas las cuatro
# estructuras reales y leido ru_maxrss:
#     Pass 1  kw_samples + c0_samples_kw                     61.8 GB
#     Pass 2  bin_results (lo que devuelve pool.map)         54.0 GB
#     Pass 2  all_samples_perbin + c0_samples_perbin         14.7 GB  (re-indexado)
#     ------------------------------------------------------------
#     PICO en el punto donde murio                          130.5 GB
#     + sample_matrix (N x 10428 x 8) y su copia centrada     16.7 GB
#     TOTAL                                                 147.2 GB
# La formula vieja daba 95 GB porque NO contaba el canal c0 (c0_samples_kw y
# c0_samples_perbin, 173.8 M entradas cada uno). Ahora son 751 B por pareja
# (replica, bin) para los cuatro almacenes juntos, que es lo medido.
# 500 GB dejan 3.4x de margen.
#
# ⚑ LOS CPUS NO MUEVEN LA MEMORIA. El pico esta en el proceso PADRE; cada worker
# solo tiene el bin que calcula (~20 MB) mas el pickle en vuelo. 40 -> 24 cpus
# no ahorra un byte, solo alarga el reloj (~1.7x). Y no puede mover un numero:
# `_mc_one_bin` se siembra con `base_seed + energy_idx`, funcion del bin solo, y
# `pool.map` conserva el orden -- comprobado en run_pyscript.archive.sh.
#
# El preflight aborta en los primeros segundos si no cuadra, en vez de morir por
# OOM a la hora 6. Y si NO PUEDE LEER la reserva, tambien aborta.
#
# ⚑ FALLBACK CONOCIDO-BUENO: `-p xlarge --mem=500G` es lo que uso la run 86
#   (MC + precompute + analisis encadenados). Si par_IB rechaza la reserva,
#   mover ahi, o bajar a KIKA_N_SAMPLES=50000.
#
# cpus-per-task y N_PROCS no pueden discrepar: el script lee
# SLURM_CPUS_PER_TASK. Detalle en run_pyscript.archive.sh.

source /work/monleon-de-la-jan/myenv/bin/activate
cd /share_snc/snc/JuanMonleon/EXFOR/scripts/
export KIKA_UNCERTAINTY_MANIFEST_PATH=/share_snc/snc/JuanMonleon/EXFOR/uncertainty_manifest_diagonal_cierjacks.yaml

# Trabajos anteriores y por que se tomo cada decision: run_pyscript.archive.sh

# ===========================================================================
# CURRENT JOB: RUN 100 -- 100 000 muestras, y el fino pasa a ser la referencia
# ===========================================================================
# Plan completo: kika-workspace/docs/plan_fine_reference_and_100k.md
#
# POR QUE 100k. p = 10 428 parametros (1738 bins x 6 ordenes), n = 10 000
# replicas => p/n = 1.043. En ese regimen la covarianza muestral NO es
# consistente: el soporte Marchenko-Pastur de sus autovalores es
# [(1-sqrt(p/n))^2, (1+sqrt(p/n))^2] = [0, 4.1], o sea degenerado. La DIAGONAL
# esta bien (error relativo sqrt(2/n) ~ 1.4 %, y en a_5/a_6 es analitica), pero
# la ESTRUCTURA DE CORRELACION fina es en buena parte ruido de MC.
#   100k => p/n = 0.104 => soporte [0.46, 1.75].
#
# POR QUE AHORA. Se deja de agrupar sobre lo ya agrupado. El objeto fino pasa a
# ser la referencia y el agrupamiento se elige DESDE esa matriz. Dos hechos que
# lo obligan, los dos ya escritos en el repo:
#   - precompute_chi2_predictive.py:162 -- "scoring the multigroup file measures
#     the mixture's ABSENCE".
#   - history §8.4 -- "§3.4's 'collapse is nearly free' was measured on a
#     winner-take-all covariance ... does not transfer to the Phase-3 product".
#   Y medido en la run 86 (fino vs mg, mismo objeto, sin cruzado) nuestro V4 se
#   movia hasta un 8 %.
#
# ⛔ LO QUE ESTA RUN NO ARREGLA, Y HAY QUE SABERLO ANTES DE MIRARLA.
# `mc_order_cap` es el SOPORTE ANGULAR del bin y no depende de n:
#     cap = 3 -> 385 bins   4 -> 524   5 -> 353   6 -> 476   (de 1738)
# En los 1262 bins con cap < 6, a_6 vale cero exacto en TODAS las replicas, sean
# 10 000 o un millon. La deadness de a_5/a_6 es estructural, no estadistica, y se
# arregla en `build_group_cross` (hilo B del plan), no con muestras.
#
# LO QUE SI COMPRA, Y NO LO DA NADA MAS: mf34_fine_mesh_inputs.npz --
#   cov_decision  la relativa FINA antes del capado near-zero, que es donde el
#                 criterio SNR todavia tiene disparador
#   cov_emission  la que de verdad se escribe en 26-Fe-56g_nominal.endf
#   means         nominal_params, que decide donde existe a_l y donde cambia de
#                 signo
# 1.74 GB. Sin el, cada criterio de agrupamiento nuevo cuesta repetir el
# pipeline entero.
#
# PER-ORDER MESH APAGADA A PROPOSITO. Vive sobre los 660 multigrupos, que es
# justo el regroup-de-un-regroup que este plan elimina. Se quiere un _mg limpio
# como linea base.
#
# ⚠ MEMORIA -- POR ESO ESTE JOB PIDE --mem Y LOS ANTERIORES NO.
# `all_samples_perbin` es {replica: {bin: array(6)}}, medido en 196 B/entrada
# (48 B de datos, el resto overhead de dict + ndarray):
#     all_samples_perbin  10k 3.4 GB  ->  100k  34 GB
#     el mismo store kw   10k 3.4 GB  ->  100k  34 GB
#     sample_matrix + la copia centrada de np.cov      100k  16.7 GB
# Pico realista 80-120 GB. Se piden 300G. Si par_IB rechaza la reserva, mover a
# `-p xlarge` o bajar a KIKA_N_SAMPLES=50000 (p/n = 0.209, soporte [0.30, 2.11]:
# buena parte del beneficio a la mitad del riesgo).
#
# DISCO: ~30 GB (parquets x10 ~ 22 GB, npy ~ 5 GB, mesh inputs 1.74 GB, cintas
# ~1.4 GB). Comprobado abajo antes de arrancar.

R100=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_100_100k
R99=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_99_signcut
mkdir -p $R100

# ⚑ EXPORTADA, no puesta en la linea del python. El preflight y el gate tienen
# que ver el MISMO numero que el pipeline, o comprueban otra run.
export KIKA_N_SAMPLES=100000

# --- PREFLIGHT: que el script desplegado sea el de hoy, y que quepa ---------
# `cross_targets`: que el script desplegado sea el que emite LAS DOS cintas
# con cruzado (mg + fina). Sin el, la run vuelve a dejar solo la _mg.
for tok in SAVE_FINE_MESH_INPUTS mf34_fine_mesh_inputs cross_targets; do
    grep -q "$tok" exfor_to_endf_research.py || {
        echo "[preflight] exfor_to_endf_research.py no tiene $tok. No envies."
        exit 1; }
done

echo "[preflight] cpus=${SLURM_CPUS_PER_TASK:-?}  mem_por_nodo=${SLURM_MEM_PER_NODE:-?} MB"
python - <<'PREFLIGHT100' || exit 1
import os, shutil, sys
from pathlib import Path

# --- LA RESERVA. Tres fuentes, en orden, porque en este cluster
# SLURM_MEM_PER_NODE viene VACIO aun con --mem puesto, y leerlo a secas fue lo
# que dejo correr 6 h 24 m la run 100 antes del OOM.
def reserva_mb():
    v = os.environ.get("SLURM_MEM_PER_NODE")
    if v and v.strip().isdigit():
        return int(v), "SLURM_MEM_PER_NODE"
    v = os.environ.get("SLURM_MEM_PER_CPU")
    c = os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.strip().isdigit() and c and c.strip().isdigit():
        return int(v) * int(c), "SLURM_MEM_PER_CPU x cpus"
    # cgroup: el limite que de verdad mata al proceso.
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
    except OSError:
        rel = ""
    base = Path("/sys/fs/cgroup")
    cand = base / rel.lstrip("/")
    for d in [cand, *cand.parents]:
        if base not in d.parents and d != base:
            continue
        for name, unit in (("memory.max", 1), ("memory/memory.limit_in_bytes", 1)):
            f = d / name
            try:
                s = f.read_text().strip()
            except OSError:
                continue
            if s.isdigit() and int(s) < (1 << 50):        # 'max' o un limite absurdo = sin limite
                return int(s) // (1024 * 1024), f"cgroup {f}"
    return 0, None

mem_mb, fuente = reserva_mb()

# --- EL PICO. MEDIDO tras el OOM de la run 100 (ru_maxrss sobre las cuatro
# estructuras reales), no estimado: 751 B por pareja (replica, bin) para
# kw_samples + c0_samples_kw + bin_results + los dos re-indexados, todos vivos a
# la vez en el punto donde murio. La formula anterior daba 95 GB porque no
# contaba el canal c0 en absoluto.
N = int(os.environ.get("KIKA_N_SAMPLES", "10000"))
NBINS = 1738
peak_gb = (751 * NBINS * N + 2 * N * 10428 * 8) / 1e9 + 10
print(f"[preflight] N = {N:,} replicas x {NBINS} bins -> pico MEDIDO {peak_gb:.0f} GB")
if fuente is None:
    print("[preflight] ⛔ NO PUEDO LEER LA RESERVA DE MEMORIA por ninguna via")
    print("[preflight]    (SLURM_MEM_PER_NODE, SLURM_MEM_PER_CPU, cgroup).")
    print("[preflight]    Esto es exactamente lo que dejo morir la run 100 a las")
    print("[preflight]    6 h. Si sabes que la reserva basta, relanza con")
    print("[preflight]    KIKA_ALLOW_UNKNOWN_MEM=1; si no, arregla el --mem.")
    if os.environ.get("KIKA_ALLOW_UNKNOWN_MEM") != "1":
        sys.exit(1)
    print("[preflight] ⚠ KIKA_ALLOW_UNKNOWN_MEM=1: sigo bajo tu responsabilidad.")
else:
    print(f"[preflight] reserva = {mem_mb/1024:.0f} GB  (via {fuente})")
    if mem_mb / 1024 < peak_gb:
        print("[preflight] ⛔ la reserva NO cubre el pico medido. Sube --mem,")
        print("[preflight]    mueve a -p xlarge, o baja KIKA_N_SAMPLES.")
        sys.exit(1)
free_gb = shutil.disk_usage("/share_snc").free / 1e9
need_gb = 12 + 10 * N / 100000  # ~12 GB fijos (incluye la _a0cross FINA, ~1.8 GB)
                                # + tmc/c0 parquet, que si escalan
print(f"[preflight] disco libre {free_gb:.0f} GB, esta run pide ~{need_gb:.0f} GB")
if free_gb < need_gb + 50:
    print("[preflight] ⛔ margen de disco insuficiente. Borra sidecars de chi2 antes.")
    sys.exit(1)
print("[preflight] OK")
PREFLIGHT100

# --- QUE PRODUCE ESTA RUN, Y QUE NO ---------------------------------------
#
# El objetivo son las CINTAS y la matriz fina. El tamano de una cinta lo fija la
# rejilla, no n, asi que 10x muestras no las engorda: lo que escala con n son las
# replicas en parquet, y de esas solo dos hacen falta.
#
#   SE PRODUCE                         100k     por que
#   ---------------------------------------------------------------------------
#   26-Fe-56g_nominal.endf             843 MB   MF34 fino 1738, SIN cruzado
#   26-Fe-56g_nominal_a0cross.endf     1.8 GB   ⚑ EL ENTREGABLE (fino + cruzado)
#                                               NUEVO en la run 100: hasta hoy el
#                                               pipeline solo emitia la _mg y esta
#                                               habia que montarla a mano
#   26-Fe-56g_nominal_mg.endf          204 MB   referencia, NO entregable (ver abajo)
#   26-Fe-56g_nominal_a0cross_mg.endf  347 MB   la _mg con cruzado; el analogo de c0
#   mf34_fine_mesh_inputs.npz         1.74 GB   desde aqui se elige el agrupado
#   legendre_samples_tmc.parquet       6.9 GB   ⚑ SIN ESTO NO HAY TERMINO CRUZADO
#   mf33_c0_samples.parquet            3.1 GB   ⚑ idem: la otra pata del cruzado
#   sidecars mf34_*.npy + mixture     4.35 GB   fijos en n; incluye el split
#                                               within/between, que es como se
#                                               diagnostico a_5/a_6
#   mf33_*.npy, nominal_fits           ~0.2 GB
#   ---------------------------------------------------------------------------
#   NO se produce                      ahorro
#   legendre_samples_raw_kw.parquet    5.8 GB   era para el split de MF34, que
#                                               esta DESCALIFICADO
#   legendre_samples_perbin.parquet    6.2 GB   replicas propias de Pass 2; nada
#                                               aguas abajo las lee
#
#   Total ~18 GB en vez de ~30 GB. Los dos flags simplemente NO se ponen: su
#   default ya es False y era la run 99 la que los encendia.
#
# ⚑ EL MULTIGRUPO SE DEJA ENCENDIDO A PROPOSITO, Y NO COMO ENTREGABLE.
# El colapso es O(p^2 * g) sobre la correlacion fina: INDEPENDIENTE de n, o sea
# minutos y una cinta de 200 MB. Lo que compra es una medida gratis: el criterio
# de hoy (rho >= 0.85 sobre a_1, sigma-ratio <= 5) aplicado a una matriz de
# correlacion que POR PRIMERA VEZ no es ruido. A 10k dio 1738 -> 660. Si a 100k
# da otro numero, eso es evidencia directa de que los 660 ajustaban ruido, y no
# compromete a nada porque el agrupamiento definitivo se elige despues, desde
# mf34_fine_mesh_inputs.npz.

KIKA_OUTPUT_DIR=$R100/ \
KIKA_MF34_PER_ORDER_MESH=0 \
KIKA_SAVE_FINE_MESH_INPUTS=1 \
KIKA_SAVE_MF34_COV_SIDECARS=1 \
KIKA_STOP_AFTER_NOMINAL_FITS=0 \
KIKA_DEGREE_WEIGHT_FLOOR=0.01 \
    python exfor_to_endf_research.py || exit 1

# --- EL GATE ---------------------------------------------------------------
# Falla por lo que TIENE que ser cierto. Lo que esta run mide -- el efecto de
# 10x muestras sobre la estructura de correlacion -- se lee, no se puertea.
python - "$R100" "$R99" "$KIKA_N_SAMPLES" <<'GATE100' || exit 1
import json, sys
from pathlib import Path
import numpy as np
new, r99 = (Path(p) for p in sys.argv[1:3])
want_n = int(sys.argv[3])
bad = 0

print("\n-- 1. LA CONFIGURACION QUE DE VERDAD CORRIO (run_metadata.json manda)")
md = json.loads((new / "run_metadata.json").read_text())["config"]
for k, want in [("N_SAMPLES", want_n), ("MF34_PER_ORDER_MESH", False),
                ("SAVE_FINE_MESH_INPUTS", True), ("DEGREE_WEIGHT_FLOOR", 0.01)]:
    got = md.get(k)
    ok = got == want
    print(f"   {'OK  ' if ok else '⛔  '} {k} = {got!r} (esperado {want!r})")
    bad += 0 if ok else 1

print("\n-- 2. EL OBJETO DE ESTA RUN: mf34_fine_mesh_inputs.npz")
z = new / "mf34_fine_mesh_inputs.npz"
if not z.exists():
    print("   ⛔ NO EXISTE. Sin el, el agrupamiento vuelve a costar una run entera.")
    bad += 1
else:
    d = np.load(z)
    n_par = d["cov_emission"].shape[0]
    print(f"   OK   {z.stat().st_size/1e9:.2f} GB, {n_par} parametros, "
          f"decision {'SIN capar' if bool(d['decision_is_raw']) else 'CAPADA'}")
    if not bool(d["decision_is_raw"]):
        print("   ⛔ la decision esta capada: el criterio SNR no tendra disparador")
        bad += 1
    if d["means"].shape[0] != n_par:
        print("   ⛔ means y cov_emission no cuadran"); bad += 1
    # La decision tiene que ver sigmas que la emision ya no tiene.
    sd_dec = np.sqrt(np.maximum(np.diag(d["cov_decision"]), 0))
    sd_emi = np.sqrt(np.maximum(np.diag(d["cov_emission"]), 0))
    print(f"   max sigma_rel:  decision {sd_dec.max()*100:.0f} %   "
          f"emision {sd_emi.max()*100:.0f} %")
    if sd_dec.max() <= sd_emi.max():
        print("   ⛔ la decision no es mas cruda que la emision")
        bad += 1

print("\n-- 3. LOS ENTREGABLES Y LAS REPLICAS")
# Exactamente lo que `build_group_cross --mag-grid fine` abre, mas las cintas.
# raw_kw y perbin NO estan aqui: esta run no los produce, a proposito.
for f in ["26-Fe-56g_nominal.endf", "26-Fe-56g_nominal_mg.endf",
          "legendre_samples_tmc.parquet", "mf33_c0_samples.parquet",
          "nominal_fits.parquet", "mf33_energy_grid_ev.npy",
          "mf33_relative_covariance.npy", "mf33_c0_host.npy",
          "mf34_cov_combined.npy"]:
    p = new / f
    if not p.exists():
        print(f"   ⛔ FALTA {f}"); bad += 1
    else:
        r = (r99 / f)
        rel = f" ({p.stat().st_size/r.stat().st_size:.1f}x la run 99)" if r.exists() else ""
        print(f"   OK   {f}: {p.stat().st_size/1e6:,.0f} MB{rel}")

print("\n-- 3b. LAS DOS CINTAS CON CRUZADO (nuevo en la run 100)")
# La _mg es dura: si falla, `_write_cross_term_endf` re-lanza y la run termina
# en no-cero, asi que llegar aqui sin ella seria una incoherencia.
# La FINA es blanda a proposito: va con strict=False para que una negativa de
# build_group_cross no tire una run de dos dias. Si falta, el entregable se
# monta a mano con `run_c_fine.sh c2 $R100`, que es lo que se hizo hasta hoy.
mgx = new / "26-Fe-56g_nominal_a0cross_mg.endf"
fnx = new / "26-Fe-56g_nominal_a0cross.endf"
if not mgx.exists():
    print("   ⛔ FALTA la _a0cross_mg: incoherente, esa es dura"); bad += 1
else:
    print(f"   OK   _a0cross_mg: {mgx.stat().st_size/1e6:,.0f} MB")
if not fnx.exists():
    print("   ⚠⚠ NO ESTA LA CINTA FINA CON CRUZADO: el paso se nego (strict=False).")
    print("      Busca '[CROSS] cross-term ENDF refused' en el log y montala con")
    print("      `sbatch run_c_fine.sh c2 $R100`. NO es motivo de fallo de la run.")
else:
    print(f"   ✅ _a0cross FINA (EL ENTREGABLE): {fnx.stat().st_size/1e6:,.0f} MB")

print("\n-- 4. QUE LA MALLA POR ORDEN NO SE APLICO (se quiere un _mg limpio)")
if (new / "mf34_per_order_mesh.npz").exists():
    print("   ⛔ hay mf34_per_order_mesh.npz: el flag no llego y el _mg esta mallado")
    bad += 1
else:
    print("   OK   sin malla por orden")

if bad:
    print(f"\n❌ RUN 100 FALLA en {bad} comprobacion(es).")
    sys.exit(1)
print("\n✅ RUN 100 PASA. Siguiente: hilo C del plan (chi2 c0/c1/c2), no aqui.")
print("   La cinta fina con cruzado ya la trae la run; el paso 1 de")
print("   run_c_fine.sh la reconstruye, y eso es una comprobacion gratis:")
print("   `cmp` contra la de la run tiene que dar identico.")
GATE100

echo "--- criterios de lectura, no pass/fail ---"
grep -E "PSD CHECK|Fine bins:|Compression:|Near-zero regular|MIX\]|entradas del agrupamiento" $R100/exfor_to_endf_*.log | tail -30
df -h /share_snc | tail -1

# --- SIGUIENTE: la puntuacion va en run_chi.sh, no aqui --------------------
# run_chi.sh ya esta preparado; hay que cambiarle R97 por R98 y el tag antes de
# lanzarlo. La run 98 SI se puntua: no es una reproduccion.
