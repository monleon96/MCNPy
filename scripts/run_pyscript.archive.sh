# Archivo de run_pyscript.sh. NO se ejecuta: es el registro de los trabajos
# anteriores y de por que se tomo cada decision. Para relanzar uno, copiar
# su bloque al runner. Separado 2026-08-19.

#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH -t 1-12:00:00
#SBATCH -p par_IB
# 2026-08-12: moved off `xlarge` to `par_IB`, and --mem removed so the job takes
# the partition default instead of reserving a fixed 500G. Previous header:
#   #SBATCH --mem=500G
#   #SBATCH -p xlarge
# ^ that reservation was sized for RUN 86: full MC (~6 h) + precompute +
#   analysis chained, the same one run 85 used. Run 92 (job 8474763) had it too.
#
# ⚠⚠ RUN 93 IS A FULL MC AND RUNS WITHOUT --mem. Juan's call, taken with the
# risk on the table. Why it should hold: the 500G was sized for MC + precompute
# + analysis CHAINED, and this job runs the MC ONLY -- no precompute, no chi2.
# The largest objects inside the MC are the 10428^2 covariance (870 MB) and the
# 10000 x 10428 sample matrix (834 MB); step 10b's 5956^2 joint is 284 MB. That
# is a handful of GB, not hundreds.
# ⚠ If it is OOM-killed anyway, THAT is where to look, and --mem goes back.
#
# ⚑ cpus-per-task 24 -> 40 (Juan). SAFE FOR THE NUMBERS, and this was checked,
# not assumed: `_mc_one_bin` seeds itself with `bin_seed = base_seed +
# energy_idx`, a function of the bin alone -- not of worker id, not of
# scheduling order -- and `pool.map` preserves input order. Pool size therefore
# cannot move a single value. N_PROCS now reads SLURM_CPUS_PER_TASK itself, so
# the header and the pool can no longer drift apart; they were coupled by hand,
# across two files, until today.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#SBATCH --job-name=kika-research

# 1) Activate your virtual env
source /work/monleon-de-la-jan/myenv/bin/activate

# 2) Go to your project directory
cd /share_snc/snc/JuanMonleon/EXFOR/scripts/

export KIKA_UNCERTAINTY_MANIFEST_PATH=/share_snc/snc/JuanMonleon/EXFOR/uncertainty_manifest_diagonal_cierjacks.yaml

# ---------------------------------------------------------------------------
# CURRENT JOB: RUN 86 — Phase 3 in the product we actually ship, plus the
# first measurement of the magnitude<->shape correlation.
#
# IDENTICAL TO RUN 85 IN EVERY KNOB. The differences are both in the code, and
# that is the point: run 86 minus run 85 isolates the §8.4 fix.
#
# 1) §8.4 — the multigroup mask. Run 85 put the Phase-3 mixture into the FINE
#    MF34 and not the multigroup one, because perform_adaptive_multigroup_collapse
#    rebuilt the legacy winner-take-all mask internally from nr.frozen_degree.
#    Dead parameters at l=6, run 84 -> 85:
#
#        fine MF34        92.3 %  ->   1.0 %      (the mixture arrived)
#        multigroup MF34  81.6 %  ->  81.7 %      (it did not)
#
#    Now overridable via `valid_orders_fn`, defaulting to the legacy rule so
#    exfor_to_endf_sampling_v2 stays bit-identical. The fork passes its q_l rule.
#
# 2) §9.4.2 — COMPUTE_MF33_MF34_CROSS. Measures Cov(c0, a_l) over the shared
#    Pass-2 MC replicas. DIAGNOSTIC ONLY: nothing downstream reads the sidecars,
#    so the ENDF is bit-identical with it on or off, and it costs no MC time.
#    It answers the one thing the Cauchy-Schwarz bound (§9.4.1) cannot: the SIGN.
#
# ⚠ CHECK THESE TWO LINES IN THE LOG BEFORE TRUSTING ANYTHING DOWNSTREAM.
#
#   a) The §8.4 fix took:
#        "Valid mask: N/10428 parameters fitted ... [rule: caller-supplied]"
#      Run 85 read 6523 fitted under "[rule: legacy frozen_degree]" (that tag is
#      new, so run 85's log does not carry it). Expect ~10427 under the new rule.
#      If it still says "legacy frozen_degree", the deploy did not take and
#      run 86 is just run 85 again.
#
#   b) The cross measurement ran:
#        "[XCORR] median rho by order: a_1 ..., a_2 ..., ..."
#      A missing line means it silently fell into its except branch — look for
#      "[XCORR] cross-covariance failed:". A line of all-nan means every order
#      came back frozen, which would itself be a finding worth stopping for.
#
# Read the chi2 against `predictive_85_fine` (the mixture present, fine grid)
# and `predictive_85` (the mixture absent — the bug). Run 86 should land near
# the former. If it lands on the latter, the fix did not reach the product.
#
# `predictive_86` IS already registered in chi2_analysis_cluster.py. That entry
# not existing is exactly what killed job 8434496 on its last line, after the
# 6 h evaluation and the 10.9 GB precompute had both succeeded.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CURRENT JOB: run 88 -- the COMPLETE MF33<->MF34 cross block (roadmap S10.1.6)
#
# WHY. Runs 87 / 87_s050 / 87_s025 produced NO chi2 at all: Sigma_V4 was not
# positive definite and chi2_metrics._solve died on Cholesky at EVERY scale.
# S10.1.4 put the PSD-safe damping at <= 0.052 against a diagonal ceiling of
# 0.276 -- a token, not a measurement -- and S10.1.5 then proved damping and
# representation work are both dead ends: with Cx = 0 the joint parameter-space
# covariance is PSD at -1e-11, and enforcing Cauchy-Schwarz, which is exactly
# what a consistent collapse buys, moves lambda_min by 12 % in the best energy
# window and ~0.01 % in the rest.
#
# The defect was never the representation. We shipped a PARTIAL Level A --
# within-bin Cov(c0(E_i), a_l(E_i)) only -- against COMPLETE MF33/MF34
# diagonals. On run 81's paired replicas the full joint sample covariance is PSD
# at -1e-18, and zeroing ONLY the cross-energy entries breaks it by 13 orders,
# with those entries the same size as the ones kept and ~159x more numerous.
#
# WHAT CHANGED IN THE PIPELINE: SAVE_MF33_C0_SAMPLES = True (False through run
# 86, and that is the ONLY reason run 86's structure cannot be rebuilt offline),
# plus `_cross_full` writing mf33_mf34_cross_covariance_full.npy =
# Cov(c0(E_i), a_l(E_j)) over ONE COMMON REPLICA SET. The common set is not a
# detail: a sample covariance is PSD only if every entry uses the same replicas.
#
# THIS IS A FULL RE-EVALUATION (~6 h MC), not a chi2-only job like run 87.
#
# CHECK IN THE LOG, IN THIS ORDER:
#   [XCORR] FULL cross block written: (1738, 1738, 6) over N replicas ...
#   [XCROSS] ... form=full   <- if it says within_bin_only the sidecar is missing
#                               and run 87 has just happened again
#   [PSD]    lines -- clean means the S10.1.3 congruence theorem is back
#
# `predictive_88` IS registered in chi2_analysis_cluster.py (verified). That
# entry not existing is what killed job 8434496 on its last line.
#
# DISK: the share is at 92 %. This run adds ~300 MB of c0 samples and ~145 MB of
# full cross block on top of the usual output. Fine, but do not queue two.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CURRENT JOB: RUN 92 -- the pipeline produces the DELIVERABLE end to end.
#
# WHY. Every tape shipped so far was assembled from more than one run. The run-91
# deliverable (26-Fe-56g_nominal_a0cross_mg.endf) was built by running
# build_group_cross.py by hand, taking its MC replicas from run 88 and its ENDF
# template from run 86, five days after either finished. There has never been a
# single command that goes from EXFOR to the file we ship.
#
# WHAT CHANGED IN THE CODE, and nothing else did:
#   1. The cross-term write is now STEP 10b inside exfor_to_endf_research.py. It
#      calls build_group_cross's certified path through a constructed argv, so
#      the command run is character-for-character the one that produced run 91:
#          --mag-grid fine --null-fill zero
#   2. A "WHAT THIS RUN PRODUCES" block at the top of the script holds every
#      product switch, and _preflight_products() refuses at STARTUP -- not five
#      hours in -- if the cross term is on while an input it needs is off.
#   3. Comments stripped from both scripts. Gated on AST equality: the code is
#      provably unchanged.
#
# ⚠ THE ONE KNOB THAT WOULD HAVE BROKEN THIS SILENTLY. The script's own default
# for DEGREE_WEIGHT_FLOOR was 0.0, but every shipped run (84, 85, 86, 88) ran at
# 0.01, exported here. It is NOT inert -- it feeds the MC degree draws and hence
# the covariance. The default is now 0.01 to match what shipped; the export below
# is kept as belt and braces.
#
# NO CHI2 IN THIS JOB. Scoring is post-processing and lives in run_chi.sh.
#
# EXPECT IN THE LOG, IN ORDER:
#   "#-- STEP 10b: MF33<->MF34 cross term"
#   "| F  a_0 blocks: cx/a_nom, ONE convention (WHAT WE NOW SHIP)"  with
#        sigma_max(K) ~ 0.999993 and lam_min ~ -1e-16.  If row F does not land on
#        the CONTROL's value, something in the chain is not a congruence and THE
#        FILE MUST NOT SHIP.
#   "| c33 matrix gate: ... relative = ~4.6e-16"   (fatal above 1e-10 on the fine axis)
#   "[CROSS] cross-term ENDF written: ..._a0cross_mg.endf"
#   "[POSITIVITY] projection fired on N/M samples" -- this number has NEVER been
#        produced before; the counter was added after run 88 and never deployed.
#
# VERIFICATION GATE once it lands (from WSL, all three must hold):
#   nominal_fits.parquet          == run 88's        (byte-identical)
#   26-Fe-56g_nominal_mg.endf     == run 86's        (byte-identical)
#   ..._a0cross_mg.endf           == run 91's 86_a0cross tape  (byte-identical)
# The positivity counter changes _mc_one_bin's return tuple from 6 to 7 elements,
# so byte-identity is a real test of its asserted inertness, not a formality.
#
# RESOURCES. 500G as inherited: the MC needs it, and step 10b holds a 5956^2
# joint plus several eigendecompositions on top (run 91 did that step in ~40 min
# under a 300G reservation). Walltime ~6 h MC + ~40 min cross.
# DISK: the share is at 87 %, 1.1 T free. This run writes ~2.1 GB
# (843 MB fine ENDF + 690 MB TMC + 300 MB c0 samples + 206 MB _mg + 347 MB cross).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CURRENT JOB 2026-08-12: P2 — the OUT-OF-SAMPLE REFITS (roadmap §3).
#
# Three nominal-fit-only variants. Each costs ~2 min, NOT a campaign: the 5 h is
# the MC, i.e. the covariance, and V2 needs only the central. With
# KIKA_STOP_AFTER_NOMINAL_FITS=1 each writes nominal_fits.parquet and nothing
# else — no covariance, no ENDF, nothing shippable.
#
# ⚑ MONOVARIABLE BY CONSTRUCTION, and this has to be said in the write-up:
#   · UNION_GRID_SUBENTRIES is NOT touched (Kinney 0.847-2.5 + Pirovano 2.5-4).
#     The mesh is a representation choice carrying no cross-section
#     information, so freezing it even when Kinney is excluded is deliberate.
#   · KIKA_DEGREE_WEIGHT_FLOOR=0.01 and the manifest match run 92 exactly
#     (run_metadata.json is the config authority; script defaults disagree).
#   · c0 comes from the host and does not move in a refit, so only the a_l
#     change; tau, ESS/Kish and the membership window recompute themselves.
#
# ⚠ RECORD, per §3.3: how many bins change AICc-winning degree in each variant.
#   That is the only source of non-linearity that can surprise.
# ⚠ Scoring a variant's central with run 91's Sigma_eval is NOT valid and must
#   not be done even as an approximation — that Sigma_eval was built with the
#   data the variant excludes. V2/N on the held-out set is what is free here.
#
# RESOURCES: nominal-fits-only. 100G / 0-01:00:00 is enough (the header above is
# sized for the full MC); leave the reservation as is or drop it to queue faster.
# Nothing writes into ENDF_samples/: all three land under chi2/p2_refits/.
# ---------------------------------------------------------------------------

# ===========================================================================
# CURRENT JOB — RUN 94: the same MC AGAIN, saving MF34's ASSEMBLY STATE.
# ===========================================================================
# Roadmap §2.7-ter D and §8 step 12(b2). Like run 93 it produces NO new
# evaluation and decides nothing. It exists to close a structural hole.
#
# WHY RUN 93 WAS NOT ENOUGH — the same lesson, paid twice on two artefacts.
# Run 93 recovered sigma_1 for a_l. The MF34 repair also needs the marginal it
# has to PRESERVE, `std_perbin`, and that is in NO parquet:
#   - per-bin MIXTURE blocks replaced the Pass-2 diagonal in 1700/1738 bins
#   - near-zero regularisation moved 2037/10428 relative sigmas
#   - `legendre_samples_tmc.parquet` is an affine rescale of PASS 1, so it
#     carries Pass-2's std only as a scale factor: un-mixed, un-regularised,
#     and in ABSOLUTE units where the tape ships RELATIVE
# and the mixture moments come from `_mc_one_bin`, i.e. from the MC itself, so
# no 2-minute nominal refit reproduces them. MF33 has shipped
# `mf33_absolute_covariance.npy` since run 88; MF34 has shipped NOTHING, and
# that asymmetry is what forces a full re-run every time MF34 needs touching.
#
# ⚑ THE ONLY TWO KNOBS THAT DIFFER FROM RUN 93 ARE SAVES:
#     KIKA_SAVE_MF34_COV_SIDECARS=1   the MF34 assembly state, float64
#     KIKA_SAVE_PERBIN_PARQUET=1      Pass 2's OWN replicas
#   Both guard `np.save`/parquet writes ONLY -- the deploy diff is three pure
#   ADD hunks, no line changed or removed, and no statement they add rebinds a
#   name the pipeline reads. The gate PROVES that instead of asserting it: if a
#   save had a side effect, the tapes would move.
#
# ⚠ RESOURCES — the one open risk. The header is run 93's exactly, deliberately:
#   run 93 was a full MC INCLUDING step 10b and finished in 16 095 s on par_IB
#   with no --mem. What run 94 ADDS is a THIRD 17.4 M-row parquet build
#   (`save_all_legendre_coefficients` materialises a list of dicts before the
#   DataFrame). It is built and freed SEQUENTIALLY, at the same point where
#   raw_kw already peaks, so the peak should not rise -- run 93 already did that
#   twice. If it is OOM-killed, THAT is where to look, and the fix is to put
#   `#SBATCH --mem=500G` back (see the header note).
#
# COST: ~4,5 h. Output ~2,5 GB as run 93, PLUS ~5 GB of sidecars and ~700 MB of
# Pass-2 parquet. Disk was at 87 %, 1,1 T free.
# ⚠ Nothing touches new_test_92_integrated or new_test_93_rawkw: both are READ
#   ONLY here, and run 94 writes to its own directory.

# ===========================================================================
# CURRENT JOB — RUN 95: the MF34 SPLIT, shipped through the pipeline.
# ===========================================================================
# Roadmap §2.7-quater (the split) and §2.7-quinquies (why it enters as a
# CORRELATION).  This is step (c)+(d) of §8 step 12: it produces the tape.
# It does NOT score anything -- that is (e), in run_chi.sh.
#
# ⚑ ONE KNOB, and it is a file:
#     KIKA_MF34_CORR_OVERRIDE=<...>/p16_mf34_split/mf34_corr_split.npy
#
# WHY A CORRELATION AND NOT `cov_combined`.  Read the code, not the sidecar
# names: `cov_combined` is NOT an interface.  STEP 7 rebuilds it from
# `(corr_kw, std_perbin)` for the multigroup collapse
# (MULTIGROUP_USE_RAW_MC_CORR = True, ~:4530) and never reads it back, so a
# covariance sidecar would reach the FINE MF34 and silently miss the
# MULTIGROUP one -- and the _mg tape is what gets scored.  Substituting
# `corr_kw` at its single point of definition reaches both, plus the saved
# sidecars, so the run documents what it actually used.
#
# WHAT THE SPLIT IS.  C_new = C1' + D_exc R_loc D_exc, with the diagonal pinned
# to the file's own `std_perbin`.  It declares NO new uncertainty and moves NO
# central value: it takes the ~95 % of MF34's variance that Pass 1 never
# generated and stops giving it Pass 1's coherent correlation.  On the 6184
# usable parameters: coh 0.209 -> 0.054, PR 15.3 -> 19.7.  On the other 40.7 %
# Pass 1 is exactly constant, so there is no measured correlation to hand and
# those columns are DIAGONAL BY ABSENCE OF DATA.  That goes in the tape write-up.
#
# ⚠ TWO CHANGES TRAVEL TOGETHER, and the write-up must say so.  The split also
# DELETES a +-1 correlation block on 2517 columns that is last-ulp rounding
# divided by last-ulp rounding (2.67 % of the shipped off-diagonal mass, and
# numerically decoupled from the live parameters at 1.7e-12).  If the (e) score
# moves, attribution between the redistribution and that removal is open.
#
# ⚠ THE MULTIGROUP GRID IS NOT FROZEN, deliberately.  It is chosen adaptively
# from the l=1 adjacent correlation, which is exactly what the split changes, so
# the group count WILL move (run 94: 660 groups).  That is downstream of the one
# knob, not a second knob; forcing run 94's boundaries would ship a grid chosen
# for the OLD correlation.  ⚑ Report the new count; do not suppress it.
#
# RESOURCES: run 94's header exactly -- same MC, same step 10b, ~4.5 h.  Output
# ~2.5 GB.  ⚠ Nothing touches new_test_92/93/94: all READ ONLY here.



# --- HECHA 2026-08-19: RUN 96, el gate de inercia. PASO (job 8501764).
# --- Se conserva por si hay que releerlo, NO para relanzarlo.
# # ===========================================================================
# # CURRENT JOB: RUN 96 — la libreria nueva no cambia nada, y hay que probarlo
# # ===========================================================================
# #
# # NO ES UNA MEJORA. Es la mitad "equivalente primero" de un cambio en dos
# # tiempos, y la unica que se puede correr hoy.
# #
# # QUE HA CAMBIADO, y solo esto:
# #   * kika 0.2.8 -> 0.2.9. `create_mf34_from_covariance` acepta ahora una malla
# #     por orden Legendre (commit cf5431a) y `JointCov.to_endf_sections` sabe
# #     colapsar a ella (b563d9a). Las dos rutas nuevas son OPT-IN: sin los
# #     argumentos nuevos el emisor recorre el mismo codigo de antes.
# #   * `joint_covariance.py` desplegado con ese metodo.
# #   * `build_group_cross.py` y `chi2_analysis_cluster.py` NO se han tocado: sus
# #     copias del share iban por delante y son las que mandan.
# #
# # QUE NO HA CAMBIADO: ningun knob, ningun umbral, ninguna entrada. Los env de
# # abajo son los de la run 94, letra por letra.
# #
# # ⚠ POR QUE HACE FALTA IGUAL. La verificacion de la librería se hizo en WSL,
# # contra el commit anterior a la cinta (813141e): seccion y fichero salieron
# # byte a byte identicos, y las 16 revisiones de GNDS/MF7 en vuelo no mueven un
# # byte de lo que escribimos. Pero el venv del cluster no es inspeccionable desde
# # aqui y la instalacion la haces tu a mano, asi que esa prueba NO cubre el
# # cluster. Esta run es la que lo cubre.
# #
# # ⚠ LA MALLA POR ORDEN NO SE EJERCITA AQUI, a proposito. La DP que la elige vive
# # en myworkspace/chi2/r2_mixture_mesh_dp.py y exfor_to_endf_research.py todavia
# # no la calcula ni pasa order_weights/order_grids_ev. Una run que la usara seria
# # dos cambios a la vez y no habria forma de atribuir una diferencia.
# #
# # INSTALAR ANTES (el wheel esta puesto, no instalado):
# #   source /work/monleon-de-la-jan/myenv/bin/activate
# #   pip install --force-reinstall --no-deps \
# #       /share_snc/snc/JuanMonleon/EXFOR/wheels/kika_nd-0.2.9-py3-none-any.whl
# #
# # RESOURCES: la cabecera de la run 94 exactamente -- mismo MC, mismo step 10b,
# # ~4.5 h. Salida ~2.5 GB. ⚠ Nada escribe en new_test_92/93/94/95: solo lectura.
#
# R96=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_96_wheel029
# R94=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_94_mf34state
# mkdir -p $R96
#
# # --- PRE-FLIGHT: fallar AHORA si el wheel no esta instalado, no en 4.5 h -----
# # El venv no se puede inspeccionar desde WSL, asi que el script comprueba su
# # propio artefacto. Sin esto, la run 96 seria la run 94 otra vez y el gate de
# # abajo PASARIA -- diciendo "no cambia nada" sobre un cambio que no llego.
# python - <<'PREFLIGHT' || exit 1
# import sys
# import numpy as np
# import kika
# # ⚠ kika.__version__ esta congelado en "0.1.0" en el __init__.py y no sigue a
# # pyproject, asi que la version se lee de los METADATOS del paquete instalado.
# # Aun asi la version no es la comprobacion vinculante: las dos de abajo lo son,
# # porque ejercitan el codigo en vez de leer una etiqueta.
# try:
#     from importlib.metadata import version as _pkgver
#     v = _pkgver("kika-nd")
# except Exception as e:
#     v = f"? ({type(e).__name__})"
# print(f"[preflight] kika-nd {v} desde {kika.__file__}")
# ok_v = str(v).startswith("0.2.9")
# print(f"[preflight] version 0.2.9      : {'OK' if ok_v else 'aviso, no bloquea'}")
# # No se comprueba por texto ni por nombre de argumento: se EJERCITA la ruta.
# try:
#     from kika.endf.writers.mf34_writer import _normalize_shape_mesh
#     g = {l: np.linspace(1e6, 4e6, 5) for l in (1, 2)}
#     b = {(1, 1): np.eye(4), (1, 2): np.eye(4), (2, 2): np.eye(4)}
#     grids, blocks = _normalize_shape_mesh(b, g, 1, 2)
#     ok_f = sorted(blocks) == [(1, 1), (1, 2), (2, 2)]
# except Exception as e:
#     ok_f = False
#     print(f"[preflight] malla por orden fallo: {type(e).__name__}: {e}")
# print(f"[preflight] malla por orden    : {'OK' if ok_f else 'NO'}")
# try:
#     from scripts.joint_covariance import JointCov
# except Exception:
#     from joint_covariance import JointCov
# ok_j = hasattr(JointCov, "collapse_orders")
# print(f"[preflight] JointCov.collapse_orders : {'OK' if ok_j else 'NO'}")
# if not (ok_f and ok_j):
#     print("[preflight] el despliegue NO llego: instala el wheel y vuelve a enviar.")
#     print("  pip install --force-reinstall --no-deps \\")
#     print("    /share_snc/snc/JuanMonleon/EXFOR/wheels/kika_nd-0.2.9-py3-none-any.whl")
#     sys.exit(1)
# if not ok_v:
#     print("[preflight] la version no cuadra pero el codigo SI esta: sigo.")
# PREFLIGHT
#
# KIKA_OUTPUT_DIR=$R96/ \
# KIKA_STOP_AFTER_NOMINAL_FITS=0 \
# KIKA_DEGREE_WEIGHT_FLOOR=0.01 \
# KIKA_SAVE_RAW_KW_PARQUET=1 \
# KIKA_SAVE_MF34_COV_SIDECARS=1 \
# KIKA_SAVE_PERBIN_PARQUET=1 \
#     python exfor_to_endf_research.py || exit 1
#
# # --- EL GATE: la run 96 tiene que ser la run 94, byte a byte ----------------
# # Escrito antes de la run. Es el gate de la run 94 invertido de intencion: alli
# # habia que probar que un cambio de codigo no movia nada; aqui, que un cambio de
# # LIBRERIA no mueve nada. Las tres cintas son la cobertura buena -- cubren MF33,
# # MF34 y el cruzado de una vez, que es mas de lo que cubre cualquier .npy.
# python - "$R96" "$R94" <<'GATE96' || exit 1
# import sys, filecmp
# from pathlib import Path
# import numpy as np
# new, old = Path(sys.argv[1]), Path(sys.argv[2])
# bad = 0
#
# print("-- 1. LAS TRES CINTAS, byte a byte (MF33 + MF34 + cruzado a la vez)")
# tapes = ("26-Fe-56g_nominal.endf",
#          "26-Fe-56g_nominal_mg.endf",
#          "26-Fe-56g_nominal_a0cross_mg.endf")
# for t in tapes:
#     a, b = new / t, old / t
#     if not a.exists():
#         print(f"   FALTA  {t}"); bad += 1; continue
#     if not b.exists():
#         print(f"   sin referencia en run 94: {t}"); continue
#     same = filecmp.cmp(a, b, shallow=False)
#     print(f"   {'OK  ' if same else 'MAL '} {t}"
#           f"   ({a.stat().st_size:,} vs {b.stat().st_size:,} bytes)")
#     bad += 0 if same else 1
#
# print("-- 2. los .npy de MF33 (1e-12 relativo; 40 hilos BLAS pueden reasociar)")
# for name in ("mf33_absolute_covariance.npy", "mf33_relative_covariance.npy",
#              "mf33_c0_nominal.npy", "mf33_c0_host.npy",
#              "mf33_energy_grid_ev.npy",
#              "mf33_multigroup_relative_covariance.npy"):
#     a, b = new / name, old / name
#     if not (a.exists() and b.exists()):
#         print(f"   ausente en uno de los dos: {name}"); continue
#     x, y = np.load(a), np.load(b)
#     d = float(np.abs(x - y).max()); s = float(np.abs(y).max()) or 1.0
#     ok = d / s < 1e-12
#     print(f"   {'OK  ' if ok else 'MAL '} {name}  rel {d/s:.3e}")
#     bad += 0 if ok else 1
#
# print("-- 3. los centrales")
# a, b = new / "nominal_fits.parquet", old / "nominal_fits.parquet"
# if a.exists() and b.exists():
#     same = filecmp.cmp(a, b, shallow=False)
#     print(f"   {'OK  ' if same else 'MAL '} nominal_fits.parquet")
#     bad += 0 if same else 1
#
# if bad:
#     print(f"\n❌ RUN 96 FALLA en {bad} comprobacion(es).")
#     print("   El wheel 0.2.9 NO es inerte, y hay que averiguar por que ANTES")
#     print("   de usar la malla por orden. La verificacion de WSL decia que si")
#     print("   lo era, asi que una diferencia aqui apunta al venv del cluster")
#     print("   (otra numpy/scipy) mas que al cambio.")
#     sys.exit(1)
# print("\n✅ RUN 96 PASA: kika 0.2.9 es inerte en la ruta de produccion.")
# print("   Queda desbloqueado el paso siguiente -- calcular la malla DP dentro de")
# print("   exfor_to_endf_research.py y pasarla por order_weights/order_grids_ev.")
# GATE96
#
# echo "--- criterios de lectura, no pass/fail ---"
# grep -E "Fine bins:|Compression:|Valid mask:|XCORR" $R96/exfor_to_endf_*.log | tail -20
# echo "  (run 94 agrupo 1738 bins finos en 660 grupos)"
# df -h /share_snc | tail -1
#
# # --- SIGUIENTE: la puntuacion va en run_chi.sh, no aqui --------------------
# #   KIKA_THIS_WORK_DIR=$R96 KIKA_MF33_MF34_CROSS_DIR=$R96 KIKA_RUN_TAG=96 \
# #       python precompute_chi2_predictive.py
# # ⚠ Si el gate pasa, la run 96 NO necesita puntuarse: es la run 94 byte a byte,
# #   asi que su chi2 es el de la run 94 por construccion. Puntuarla seria gastar
# #   6 h en reproducir un numero que ya tenemos.


# ⛔ RUN 95 -- DESCALIFICADA. El split de MF34 no se envia: §6e se dispara
# (n_absorbed sale fuera de Kinney) y el chi2 coincide. Se conserva el bloque
# por si hace falta releerlo, NO para relanzarlo.
# R95=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_95_mf34split
# R94=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_94_mf34state
# SPLIT=/share_snc/snc/JuanMonleon/chi2/p16_mf34_split
# mkdir -p $R95
#
# Fail BEFORE the 4.5 h MC if the override file is not there.
# test -f $SPLIT/mf34_corr_split.npy || { echo "missing $SPLIT/mf34_corr_split.npy"; exit 1; }
#
# KIKA_OUTPUT_DIR=$R95/ \
# KIKA_STOP_AFTER_NOMINAL_FITS=0 \
# KIKA_DEGREE_WEIGHT_FLOOR=0.01 \
# KIKA_SAVE_RAW_KW_PARQUET=1 \
# KIKA_SAVE_MF34_COV_SIDECARS=1 \
# KIKA_MF34_CORR_OVERRIDE=$SPLIT/mf34_corr_split.npy \
#     python exfor_to_endf_research.py || exit 1
#
# --- THE GATE, and it is the INVERSE of run 94's -------------------------
# Run 94 had to prove nothing moved.  Run 95 has to prove that EXACTLY the
# intended thing moved and nothing else did.  Written before the run.
#   1. MF33's six .npy byte-identical to run 94  -- MF33 is untouched, so this
#      is the veto that makes the comparison monovariable.  Same role as the
#      "libraries move 0.00000 %" veto in P1.3 (§2.6-bis).
#   2. nominal_fits.parquet byte-identical       -- no central value moved.
#   3. mf34_std_perbin.npy byte-identical        -- ⚑ THE claim: the declared
#      sigma is unchanged, this is a redistribution and not an inflation.
#   4. mf34_corr_kw.npy == the override file     -- the knob actually took.
#   5. mf34_cov_combined.npy == p16's product    -- the pipeline rebuilt the
#      same (b2) matrix the workstation did, through its own expression.
#   6. the three tapes MUST DIFFER from run 94's -- if any is byte-identical the
#      override did not reach the writer and the run is run 94 again.
# python - "$R95" "$R94" "$SPLIT" <<'GATE95' || exit 1
# import sys, os, filecmp, numpy as np
# new, old, split = sys.argv[1], sys.argv[2], sys.argv[3]
# bad = 0
#
# print("-- 1. MF33 is untouched (the monovariable veto)")
# for name in ("mf33_absolute_covariance.npy", "mf33_relative_covariance.npy",
#              "mf33_c0_nominal.npy", "mf33_c0_host.npy",
#              "mf33_energy_grid_ev.npy", "mf33_multigroup_relative_covariance.npy"):
#     same = filecmp.cmp(f"{new}/{name}", f"{old}/{name}", shallow=False)
#     print(f"  {name:44s} {'byte-identical' if same else '*** DIFFERS ***'}")
#     if not same:
#         print(f"  *** FAIL: MF34's override moved MF33. Not monovariable."); bad += 1
#
# print("\n-- 2. no central value moved")
# same = filecmp.cmp(f"{new}/nominal_fits.parquet", f"{old}/nominal_fits.parquet",
#                    shallow=False)
# print(f"  nominal_fits.parquet  {'byte-identical' if same else '*** DIFFERS ***'}")
# if not same:
#     print("  *** FAIL: a covariance-only change moved the centrals."); bad += 1
#
# print("\n-- 3. the declared sigma is unchanged (this is a REDISTRIBUTION)")
# same = filecmp.cmp(f"{new}/mf34_std_perbin.npy", f"{old}/mf34_std_perbin.npy",
#                    shallow=False)
# print(f"  mf34_std_perbin.npy   {'byte-identical' if same else '*** DIFFERS ***'}")
# if not same:
#     print("  *** FAIL: the split must not declare more uncertainty."); bad += 1
#
# print("\n-- 4. the override took")
# a = np.load(f"{new}/mf34_corr_kw.npy", mmap_mode="r")
# b = np.load(f"{split}/mf34_corr_split.npy", mmap_mode="r")
# if a.shape != b.shape:
#     print(f"  *** FAIL: shape {a.shape} vs {b.shape}"); bad += 1
# else:
#     w = 0.0
#     for i0 in range(0, a.shape[0], 1024):
#         i1 = min(i0 + 1024, a.shape[0])
#         w = max(w, float(np.abs(np.asarray(a[i0:i1]) - np.asarray(b[i0:i1])).max()))
#     print(f"  max |corr_kw(saved) - corr_split(input)| = {w:.3e}")
#     if w != 0.0:
#         print("  *** FAIL: the saved correlation is not the one supplied."); bad += 1
#
# print("\n-- 5. the pipeline rebuilt the (b2) matrix")
# C = np.load(f"{new}/mf34_cov_combined.npy", mmap_mode="r")
# P = np.load(f"{split}/mf34_cov_combined.npy", mmap_mode="r")
# w, sc = 0.0, 0.0
# for i0 in range(0, C.shape[0], 1024):
#     i1 = min(i0 + 1024, C.shape[0])
#     x, y = np.asarray(C[i0:i1]), np.asarray(P[i0:i1])
#     w = max(w, float(np.abs(x - y).max())); sc = max(sc, float(np.abs(y).max()))
#     del x, y
# bar = 8.0 * np.finfo(float).eps * max(sc, 1.0)
# print(f"  max |pipeline - p16| = {w:.3e}   scale {sc:.4g}   bar {bar:.3e}")
# if not np.isfinite(w) or w > bar:
#     print("  *** FAIL: the pipeline's MF34 is not the (b2) product."); bad += 1
#
# print("\n-- 6. the tapes MUST have moved")
# for t in ("26-Fe-56g_nominal.endf", "26-Fe-56g_nominal_mg.endf",
#           "26-Fe-56g_nominal_a0cross_mg.endf"):
#     pn = f"{new}/{t}"
#     if not os.path.exists(pn):
#         print(f"  *** FAIL {t}: not written"); bad += 1; continue
#     same = filecmp.cmp(pn, f"{old}/{t}", shallow=False)
#     print(f"  {t:38s} {os.path.getsize(pn):>13,d} B   "
#           f"{'*** BYTE-IDENTICAL — THE OVERRIDE DID NOT REACH THE WRITER ***' if same else 'differs, as intended'}")
#     if same:
#         bad += 1
#
# if bad:
#     print("\n>>> GATE FAILED. STOP: either the override did not take, or it moved")
#     print("    something it had no business moving. Nothing here is admissible.")
#     sys.exit(1)
# print("\n>>> GATE PASSED. Run 95 is run 94 with MF34's correlation redistributed")
# print("    and nothing else: MF33, the centrals and the declared sigma are all")
# print("    byte-identical, and all three tapes moved.")
# GATE95
#
# echo "--- reading criteria, not pass/fail: what the regrouping did ---"
# grep -E "Fine bins:|Compression:|MF34-OVERRIDE" $R95/exfor_to_endf_*.log | tail -20
# echo "  (run 94 grouped 1738 fine bins into 660 groups)"
# df -h /share_snc | tail -1

# --- NEXT: (e) THE SCORING, and it is in run_chi.sh, not here --------------
#   KIKA_THIS_WORK_DIR=$R95 KIKA_MF33_MF34_CROSS_DIR=$R95 KIKA_RUN_TAG=95 \
#       python precompute_chi2_predictive.py
#   KIKA_CHI2_METHODOLOGIES=predictive_95 KIKA_CHI2_RUN_ID=095 \
#       python chi2_analysis_cluster.py
# ⚠ `predictive_95` must be REGISTERED in chi2_analysis_cluster.py FIRST. That
#   entry not existing is what killed job 8434496 on its last line, after the
#   6 h evaluation and the 10.9 GB precompute had both succeeded.
# ⚠ READING ORDER, fixed before the run (§2.6 P1.3):
#   1 VETO: JEFF and JENDL must move 0.00000 % on all six subsets.
#   2 Kinney V4/N   3 aggregate off-Kinney   4 n_absorbed, which MUST NOT FALL
#     outside Kinney (§6e is disqualifying)   5 headline no_Cierjacks.
# ⚠ It is NOT established that repairing MF34 helps. MF33's split paid 30 %;
#   MF34's is unknown until scored, and it is not anticipated.


# R94=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_94_mf34state
# R93=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_93_rawkw
# R92=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_92_integrated
# # exfor_to_endf_research.py does NOT create KIKA_OUTPUT_DIR itself.
# mkdir -p $R94
#
# KIKA_OUTPUT_DIR=$R94/ \
# KIKA_STOP_AFTER_NOMINAL_FITS=0 \
# KIKA_DEGREE_WEIGHT_FLOOR=0.01 \
# KIKA_SAVE_RAW_KW_PARQUET=1 \
# KIKA_SAVE_MF34_COV_SIDECARS=1 \
# KIKA_SAVE_PERBIN_PARQUET=1 \
#     python exfor_to_endf_research.py || exit 1
#
# # --- THE GATE: did run 94 reproduce run 93, and are the sidecars the file? --
# # Three parts, and the job exits 1 on any of them.
# #   1. the six MF33 .npy, as run 93 (1e-12 relative; 40 BLAS threads can
# #      reassociate reductions at the last ulp, so byte-identity is not demanded
# #      -- though run 93 delivered it anyway)
# #   2. ⚑ the THREE TAPES, byte-identical. This is the MF34 coverage, and it is
# #      what PROVES the new saves are inert. Measured today: run 92's and run
# #      93's tapes are byte-identical to each other, all three, including the
# #      deliverable _a0cross_mg. There are no embedded timestamps.
# #   3. the sidecars exist and rebuild the assembly identity they claim to be
# #      the pieces of:  cov_combined == corr_kw (x) outer(std,std) (x) signs,
# #      and diag(cov_combined) == std_perbin^2.
# python - "$R94" "$R93" <<'GATE' || exit 1
# import sys, os, filecmp, numpy as np
# new, old = sys.argv[1], sys.argv[2]
# bad = 0
#
# print("-- 1. MF33 sidecars vs run 93")
# for name in ("mf33_absolute_covariance.npy", "mf33_relative_covariance.npy",
#              "mf33_c0_nominal.npy", "mf33_c0_host.npy",
#              "mf33_energy_grid_ev.npy", "mf33_multigroup_relative_covariance.npy"):
#     a = np.load(f"{new}/{name}", mmap_mode="r")
#     b = np.load(f"{old}/{name}", mmap_mode="r")
#     if a.shape != b.shape:
#         print(f"  FAIL {name}: shape {a.shape} vs {b.shape}"); bad += 1; continue
#     d = np.abs(np.asarray(a, float) - np.asarray(b, float))
#     s = np.abs(np.asarray(b, float))
#     rel = float((d / np.maximum(s, 1e-300)).max())
#     ident = bool((np.asarray(a) == np.asarray(b)).all())
#     print(f"  {name:44s} max|rel| = {rel:.3e}   {'byte-identical' if ident else ''}")
#     if rel > 1e-12:
#         print(f"  *** FAIL: {name} moved."); bad += 1
#
# print("\n-- 2. the three TAPES, byte-identical (this is the MF34 coverage)")
# for t in ("26-Fe-56g_nominal.endf", "26-Fe-56g_nominal_mg.endf",
#           "26-Fe-56g_nominal_a0cross_mg.endf"):
#     pn, po = f"{new}/{t}", f"{old}/{t}"
#     if not os.path.exists(pn):
#         print(f"  *** FAIL {t}: not written"); bad += 1; continue
#     same = filecmp.cmp(pn, po, shallow=False)
#     print(f"  {t:38s} {os.path.getsize(pn):>13,d} B   "
#           f"{'byte-identical' if same else '*** DIFFERS ***'}")
#     if not same:
#         print(f"  *** FAIL: {t} moved. A 'save-only' change was NOT save-only.")
#         bad += 1
#
# print("\n-- 3. the new MF34 sidecars: present, and the assembly identity holds")
# need = ["mf34_cov_combined.npy", "mf34_cov_kw_rel.npy", "mf34_cov_perbin_rel.npy",
#         "mf34_corr_kw.npy", "mf34_corr_perbin.npy", "mf34_std_perbin.npy",
#         "mf34_mean_perbin.npy", "mf34_mean_kw.npy", "mf34_mean_signs.npy",
#         "mf34_valid_mask_kw.npy", "mf34_energy_indices.npy"]
# missing = [f for f in need if not os.path.exists(f"{new}/{f}")]
# if missing:
#     print(f"  *** FAIL: missing sidecars: {missing}"); bad += 1
# else:
#     for f in need:
#         print(f"  {f:34s} {os.path.getsize(f'{new}/{f}'):>14,d} B")
#     C  = np.load(f"{new}/mf34_cov_combined.npy", mmap_mode="r")
#     R  = np.load(f"{new}/mf34_corr_kw.npy", mmap_mode="r")
#     sd = np.load(f"{new}/mf34_std_perbin.npy")
#     sg = np.load(f"{new}/mf34_mean_signs.npy")
#     print(f"  shapes: cov {C.shape}  corr {R.shape}  std {sd.shape}")
#     dg = np.asarray([C[i, i] for i in range(C.shape[0])])
#     dd = float(np.max(np.abs(dg - sd ** 2)))
#     scale = float(np.max(np.abs(dg)))
#     print(f"  max |diag(cov_combined) - std_perbin^2| = {dd:.3e}   (congruence)")
#     if not np.isfinite(dd) or dd > 1e-12 * max(scale, 1e-300):
#         print("  *** FAIL: the shipped diagonal is not std_perbin^2."); bad += 1
#     # rebuild the assembly formula in row blocks -- the full outer is 870 MB
#     worst = 0.0
#     n = C.shape[0]
#     for i0 in range(0, n, 1024):
#         i1 = min(i0 + 1024, n)
#         blk = (np.asarray(R[i0:i1]) * np.outer(sd[i0:i1], sd)
#                * np.outer(sg[i0:i1], sg))
#         worst = max(worst, float(np.max(np.abs(blk - np.asarray(C[i0:i1])))))
#         del blk
#     print(f"  max |cov_combined - corr_kw (x) std (x) signs| = {worst:.3e}"
#           f"   (scale {scale:.3e})")
#     if not np.isfinite(worst) or worst > 1e-12 * max(scale, 1e-300):
#         print("  *** FAIL: the sidecars do not rebuild the shipped matrix."); bad += 1
#
# pq = f"{new}/legendre_samples_perbin.parquet"
# if os.path.exists(pq):
#     print(f"  legendre_samples_perbin.parquet {os.path.getsize(pq):>14,d} B")
# else:
#     print("  *** FAIL: legendre_samples_perbin.parquet missing"); bad += 1
#
# if bad:
#     print("\n>>> GATE FAILED. Run 94 is NOT run 93, or its sidecars are not the")
#     print("    shipped MF34. NOTHING built on them is admissible. STOP.")
#     sys.exit(1)
# print("\n>>> GATE PASSED. run 94 reproduces run 93 tape for tape, and the MF34")
# print("    sidecars ARE the shipped cov_combined and its two passes. MF34 is now")
# print("    as repairable offline as MF33 has been since run 88.")
# GATE
#
# ls -la $R94/mf34_*.npy $R94/legendre_samples_perbin.parquet
# df -h /share_snc | tail -1

# --- NEXT, and it is NOT in this job (roadmap §8 step 12) -------------------
#  b2) split MF34 offline from the sidecars: C_new = C1' + D_exc R_loc D_exc,
#      with sigma_1 from mf34_cov_kw_rel.npy (already RELATIVE, so there is no
#      unit transfer to get wrong) and the diagonal pinned to mf34_std_perbin.
#  c) cross block recomputed from the NEW marginals -- §10.1 says the a0 block
#     is only Cauchy-Schwarz-compatible with the marginals it was built from,
#     and ignoring that is what killed runs 89-90. Gate of §10.7-10 first.
#  d) ONE tape.  e) ONE scoring.
# ⚠ And declare what §2.7-bis measured: a split MF34 would be MEASURED on
#   59,3 % of its parameters and DIAGONAL BY ABSENCE OF DATA on the other
#   40,7 %, because Pass 1 is exactly constant there.

# ===========================================================================
# --- PREVIOUS JOB: RUN 93, the Pass-1 a_l marginals. DONE 2026-08-12. ------
# ===========================================================================
# GATE PASSED: all six MF33 .npy came back byte-identical (0.000e+00), and
# mf33_c0_samples.parquet has the same md5 as run 92's -- so MF33's step (b)
# closed on a checksum instead of a re-run. Measured on it: sigma_2/sigma_1 for
# a_l = 4.302 (the run-81 proxy said 4.872), 6188 live / 4240 exactly constant,
# and the live-by-degree profile matches run 92 bin for bin. §2.7-ter.
#
# mkdir -p $R93
# KIKA_OUTPUT_DIR=$R93/ \
# KIKA_STOP_AFTER_NOMINAL_FITS=0 \
# KIKA_DEGREE_WEIGHT_FLOOR=0.01 \
# KIKA_SAVE_RAW_KW_PARQUET=1 \
#     python exfor_to_endf_research.py || exit 1

# ===========================================================================
# --- PREVIOUS JOB: P2's three refits. DONE 2026-08-12, do NOT re-enable. ---
# ===========================================================================
# V-KS and V-noCJ came back clean; V-noKIN has 716/1654 bins with n_pts <= 4 and
# 713 interpolated, so it CANNOT be read as a method comparison (§3.3-bis).
# All three produced the 1738-bin union grid after the grid-preservation fix.
#
# export KIKA_STOP_AFTER_NOMINAL_FITS=1
# export KIKA_DEGREE_WEIGHT_FLOOR=0.01
# P2BASE=/share_snc/snc/JuanMonleon/chi2/p2_refits
# mkdir -p $P2BASE/v_ks $P2BASE/v_nocj $P2BASE/v_nokin

# --- V-KS: only the LIBRARIES' fit set (Kinney <=2.5 MeV + Smith 2.5-4) -----
# The one that matters most and is in no document: (a) method against method on
# IDENTICAL data — the only clean test of whether our method beats theirs rather
# than us averaging 67 experiments against their two; (b) all three libraries
# out of sample at once on the other 65. Keeps 10571002 + 10886002; the 65 ids
# are the corpus minus those two, generated from the run-92 parquet.
# KIKA_OUTPUT_DIR=$P2BASE/v_ks \
# KIKA_EXCLUDE_EXPERIMENTS=10332004,11121006,11121007,11121008,11121009,11121010,11121011,11220013,11276009,11316003,11341005,11396011,11496004,11511009,11519016,11572011,11637007,11638003,11706005,11711002,11717002,13511004,14451003,14462002,20008002,20008003,20008004,20008005,20008006,20008007,20008008,20008009,20008010,20008011,20008012,20008013,20008014,20008015,20008016,20008017,20008018,20019099,20020012,20197006,20304002,20379003,20482005,20743002,20761002,21377005,22531007,23365004,23365005,27673002,30076004,30463020,30633002,40042016,400750021,40168003,40336101,40367002,40372004,40532004,40706046 \
#     python exfor_to_endf_research.py || exit 1

# --- V-noCJ: leave-Cierjacks-out -------------------------------------------
# The measurement both reassessment docs ask for: out of sample for ALL THREE.
# KIKA_OUTPUT_DIR=$P2BASE/v_nocj \
# KIKA_EXCLUDE_EXPERIMENTS=20743002 \
#     python exfor_to_endf_research.py || exit 1

# --- V-noKIN: leave-Kinney-out ---------------------------------------------
# The hardest possible test: out of sample for us, IN sample for JEFF/JENDL.
# If we stay close it is a strong result; if not, it gets declared.
# KIKA_OUTPUT_DIR=$P2BASE/v_nokin \
# KIKA_EXCLUDE_EXPERIMENTS=10571002 \
#     python exfor_to_endf_research.py || exit 1

# unset KIKA_STOP_AFTER_NOMINAL_FITS KIKA_DEGREE_WEIGHT_FLOOR

# --- previous job: run 92, THE DELIVERABLE (full MC, ~6 h) ------------------
# Landed 2026-08-11 -> new_test_92_integrated. Uncomment to restore.
#
# export KIKA_OUTPUT_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_92_integrated/
# export KIKA_STOP_AFTER_NOMINAL_FITS=0
# export KIKA_DEGREE_WEIGHT_FLOOR=0.01
#
# python exfor_to_endf_research.py || exit 1
#
# unset KIKA_OUTPUT_DIR KIKA_STOP_AFTER_NOMINAL_FITS KIKA_DEGREE_WEIGHT_FLOOR

# --- previous job: run 88 -- the complete cross block, scored ---------------
# Landed 2026-08-04 -> new_test_88_fullcross / run_088. Uncomment to restore.
#
# export KIKA_OUTPUT_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_88_fullcross/
# export KIKA_STOP_AFTER_NOMINAL_FITS=0
# export KIKA_DEGREE_WEIGHT_FLOOR=0.01
# python exfor_to_endf_research.py || exit 1
# unset KIKA_OUTPUT_DIR KIKA_STOP_AFTER_NOMINAL_FITS KIKA_DEGREE_WEIGHT_FLOOR
# KIKA_THIS_WORK_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_88_fullcross \
# KIKA_MF33_MF34_CROSS_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_88_fullcross \
# KIKA_RUN_TAG=88 \
#     python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_88 KIKA_CHI2_RUN_ID=088 \
#     python chi2_analysis_cluster.py || exit 1

# --- previous job: run 86 -- the multigroup collapse-mask fix ---------------
# Landed 2026-08-03 -> new_test_86_mgfix / run_086. Uncomment to restore.
#
# export KIKA_OUTPUT_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_86_mgfix/
# export KIKA_STOP_AFTER_NOMINAL_FITS=0
# export KIKA_DEGREE_WEIGHT_FLOOR=0.01
# python exfor_to_endf_research.py || exit 1
# unset KIKA_OUTPUT_DIR KIKA_STOP_AFTER_NOMINAL_FITS KIKA_DEGREE_WEIGHT_FLOOR
# KIKA_THIS_WORK_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_86_mgfix \
# KIKA_RUN_TAG=86 \
#     python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_86 KIKA_CHI2_RUN_ID=086 \
#     python chi2_analysis_cluster.py || exit 1

unset KIKA_UNCERTAINTY_MANIFEST_PATH

# 4) Deactivate
deactivate

# --- previous job: GATE A — Phase-3 refactor inertness ----------------------
# A ~2 min nominal-fits-only job (100G / 0-01:00:00) proving that with both
# Phase-3 knobs off, every nominal-stage column reproduces run 84's.
# ⚠ BOTH knobs are needed: SHIP_MIXTURE_MEAN is INDEPENDENT of
# USE_MIXTURE_COVARIANCE, and turning off only the latter would still ship the
# mixture mean, failing the gate for the wrong reason.
# Then: python .../myworkspace/model_order/gate_a_inertness.py (from WSL).
#
# export KIKA_OUTPUT_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/GATE_A_INERTNESS/
# export KIKA_USE_MIXTURE_COVARIANCE=0
# export KIKA_SHIP_MIXTURE_MEAN=0
# export KIKA_DEGREE_WEIGHT_FLOOR=0.01
# export KIKA_MC_CAP_FROM_SUPPORT_ONLY=0
# python exfor_to_endf_research.py || exit 1
# unset KIKA_OUTPUT_DIR KIKA_USE_MIXTURE_COVARIANCE KIKA_SHIP_MIXTURE_MEAN
# unset KIKA_DEGREE_WEIGHT_FLOOR KIKA_MC_CAP_FROM_SUPPORT_ONLY

# --- previous job: RUN 85 — Phase 3, the mixture covariance -----------------
# Produced new_test_85_mixture/ and run_085/. 500G / 1-12:00:00.
#
# export KIKA_OUTPUT_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_85_mixture/
# export KIKA_STOP_AFTER_NOMINAL_FITS=0
# export KIKA_DEGREE_WEIGHT_FLOOR=0.01
# python exfor_to_endf_research.py || exit 1
# unset KIKA_OUTPUT_DIR KIKA_STOP_AFTER_NOMINAL_FITS KIKA_DEGREE_WEIGHT_FLOOR
# KIKA_THIS_WORK_DIR=.../new_test_85_mixture KIKA_RUN_TAG=85 \
#     python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_85 KIKA_CHI2_RUN_ID=085 \
#     python chi2_analysis_cluster.py || exit 1

# --- previous job: RUN 84 — the full tau-GLS re-evaluation, scored ----------
# Produced new_test_84_taugls/ and run_084/. The baseline run 85 is compared
# against. Same script; the difference is that run 84 had no Phase-3 knobs.
#
# export KIKA_OUTPUT_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_84_taugls/
# export KIKA_STOP_AFTER_NOMINAL_FITS=0
# export KIKA_DEGREE_WEIGHT_FLOOR=0.01
# export KIKA_MC_CAP_FROM_SUPPORT_ONLY=0
# python exfor_to_endf_research.py || exit 1
# KIKA_THIS_WORK_DIR=.../new_test_84_taugls KIKA_RUN_TAG=84 python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_84 KIKA_CHI2_RUN_ID=084 python chi2_analysis_cluster.py || exit 1

# --- previous job: the τ-GLS nominal-fits-only diagnostic run ---------------
# Produced NEW_FIT_RESEARCH_TAUGLS/ in ~2 min: the τ and central-value shifts
# and the re-measured Phase-2 chi2_pp_ratio (median 0.9837). No MC, no ENDF.
#
# #SBATCH --mem=100G
# #SBATCH -t 1-00:00:00
#
# export KIKA_OUTPUT_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/NEW_FIT_RESEARCH_TAUGLS/
# python exfor_to_endf_research.py
# unset KIKA_OUTPUT_DIR

# --- previous job: Phase 2, the model-averaged central (WLS τ refit) --------
# Produced NEW_FIT_RESEARCH/ — the baseline this run is compared against.
# Same script; the only difference is TAU_REFIT_USE_GLS was absent (i.e. the
# WLS refit) and the output went to the default OUTPUT_DIR.
#
# python exfor_to_endf_research.py          # with no KIKA_OUTPUT_DIR export

# --- previous job: the v2 thesis pipeline (run 83 and earlier) --------------
# Flip back to this for anything in the v2 lineage. Note the walltime: a full
# run with the MC is ~5 h, and 12 d was the standing reservation.
#
# #SBATCH -t 12-00:00:00
# #SBATCH --job-name=kika
# python exfor_to_endf_sampling_v2.py

# #SBATCH --mem=100G
