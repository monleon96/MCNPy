#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=40
# 2026-08-17: 24 -> 40. par_IB has them free.
# ⚠ THIS BUYS MEMORY, NOT SPEED, AND THAT IS NOT AN OVERSIGHT.
# `precompute_chi2_predictive.py:109` pins OMP/OPENBLAS/MKL/BLIS/VECLIB/NUMEXPR
# to 1 thread BEFORE numpy loads, and nothing in the chain uses multiprocessing
# (checked: precompute, chi2_analysis_cluster, eval_covariance). So the extra
# cores sit idle; what they do buy is the partition's default mem-per-cpu, which
# is what the 11 GB sidecar and the N=28631 eigendecompositions actually need.
# ⛔ To spend them on speed you must export the thread vars in THIS file before
# python starts (`setdefault` will then leave them alone) -- but threaded BLAS
# changes the summation order, and the vetoes in this run are EXACT-MATCH tests
# ("JEFF and JENDL at 0.00000 %", "V2 identical to four decimals"). Do not do it
# on a run whose job is to certify invariance.
#SBATCH -t 0-12:00:00
#SBATCH -p par_IB
# 2026-08-12: moved off `xlarge` to `par_IB`, and --mem removed so the job takes
# the partition default instead of reserving a fixed 300G. Previous header:
#   #SBATCH --mem=300G
#   #SBATCH -p xlarge
# ⚠ If a step is OOM-killed on par_IB, the scoring reads an 11 GB sidecar and
# the MF34 fine-grid work wants ~300G — put --mem back before blaming the code.
# 2026-08-14, run 95: its MF34 is finer than anything scored so far (896 groups
# / 485 MB tape vs 91_cross's 703 / 347 MB), so this is the likeliest job yet to
# need it. Uncomment if it is OOM-killed:
##SBATCH --mem=300G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=juan-antonio.monleondelalluvia@asnr.fr
#SBATCH --job-name=kika-chi2

# 1) Activate your virtual env
source /work/monleon-de-la-jan/myenv/bin/activate

# 2) Go to your project directory
cd /share_snc/snc/JuanMonleon/EXFOR/scripts/

export KIKA_UNCERTAINTY_MANIFEST_PATH=/share_snc/snc/JuanMonleon/EXFOR/uncertainty_manifest_diagonal_cierjacks.yaml

# ---------------------------------------------------------------------------

# XDIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_88_fullcross
# DIR86=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_86_mgfix
# SHIPPED_ENDF=26-Fe-56g_nominal_mg.endf
# NULL_MASK=/share_snc/snc/JuanMonleon/chi2/mf34_null_mask.npz

# ---------------------------------------------------------------------------

# GROUPED_ENDF=26-Fe-56g_nominal_mg_mf33grouped.endf

# ⚑⚑ TWO JOBS, IN THIS ORDER, AND THE ORDER IS THE WHOLE POINT (§10.7-6).
#
# Job 8456291 died on the guard that is now a mask, and what it found was not
# ours: **JEFF-4.0 publishes MF34 (2,6) only from 1 MeV** while the EXFOR points
# run to 0.85 MeV, so `np.clip` pinned 1200 of 13208 points to (2,6)'s first
# interval and runs 82–90 all folded a fabricated Cov(a_2, a_6) FOR JEFF.
# This_work's own (2,6) starts at 0.846822 MeV and covers every point.
#
# That fix moves the BASELINE we compare against. Scoring the MF33 regrouping
# against run_086 now would move two things at once — the exact confound that
# §L16 and §L18 were retracted for. So:
#
#   A. 86_maskfix     same ENDF as run 86, only the mask.  Read against run_086.
#                     PREDICTION: This_work identical to run_086, JEFF moves.
#                     If This_work moves, the mask is reaching something it
#                     should not and B must not be read.
#   B. 86_mf33grouped the regrouped MF33 on top.           Read against A.
#
# Both entries are REGISTERED in chi2_analysis_cluster.py. Runs 85 and 89 died
# after their precompute for want of exactly that (§L1).
# Disk: two ~11 GB sidecars; delete both once the parquets are read.
# ---------------------------------------------------------------------------

# ⚠ The parse cache goes HERE, not in $XDIR. `build_group_cross`'s default is
# the run directory, and the deploy rules keep run outputs read-only. Costs
# two ENDF parses (~10 min) in a 40-minute step; buys not writing into run 88.
# A0DIR=/share_snc/snc/JuanMonleon/ENDF_samples/86_a0cross
# A0ENDF=26-Fe-56g_nominal_a0cross_mg.endf

# mkdir -p $A0DIR

# ===========================================================================
# CURRENT JOB (2026-08-12): P1.3 — price the split-combine MF33.
# docs/post_run92_verification_roadmap.md §2.6
# ===========================================================================
# P1.0/P1.1 measured that the two-pass combine hands Pass-2's variance the
# correlation of Pass-1 -- so the file declares ~5.55 % fully-coherent where the
# shared-draw pass measures 2.08 %, and 2.67 % energy-local where the marginal
# pass measures ~5.7 %. The two parts are nearly swapped. P1.2 rebuilt MF33 as
# "Pass-1 at its own scale + the excess, correlated as MEASURED by Pass-1's
# local residual": PR 1.78 -> 8.06, coh 0.900 -> 0.391, L_corr 1.73 -> 0.098 MeV,
# DIAGONAL UNCHANGED. This job asks what that costs or buys.
#
# ⚠⚠ SINGLE VARIABLE. Only MF33 MT2 (and the MT1 rebuilt from it) differs from
#    run 92. ACCEPTANCE: **JEFF and JENDL must move 0.00 % on every subset.**
#    If they move, the run is not single-variable and nothing in it can be read.
#
# ⚠⚠ BASELINE IS `91_rewrite`, NOT `91_cross`. Changing MF33's marginals breaks
#    the Cauchy-Schwarz compatibility of the a0 cross block (§10.1 -- this killed
#    runs 89/90), so the cross tape cannot be reused. It is also where §10.8-19
#    measured energy-local freedom at -68 %/5 %.
#
# ⚠ DISK: 87 % used, 1.1 T free. The eval_cov sidecar is ~11 GB. Step 4 below.
# ⚠ Nothing writes into ENDF_samples/: run 92 is read-only here and the new tape
#   goes to chi2/p13_split/.
#
# Step 1 is a GATE and costs ~2 min: it reads the written tape back and refuses
# to continue unless PR >= 4 and coh <= 0.60, i.e. unless the redistribution
# survived the near-zero guard and the ENDF writer. It saves the hour below.

# P13RUN=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_92_integrated
# P13SPLIT=/share_snc/snc/JuanMonleon/chi2/p12_split
# P13OUT=/share_snc/snc/JuanMonleon/chi2/p13_split
# P13TAG=92split
# For the rank UPPER BOUND instead, set:
#   P13VARIANT=mf33_absolute_covariance_splitdiag.npy
# P13VARIANT=mf33_absolute_covariance.npy

# ⚠⚠ RESUME 2026-08-12 after job 8480897 — steps 1 and 2 ALREADY SUCCEEDED and
# are deliberately not repeated. The gate passed on the written tape (diagonal
# max |rel diff| = 0.000e+00, coh 0.779 -> 0.344, PR 2.13 -> 9.47, PSD floor
# 1.00x the shipped one) and the precompute wrote both artefacts:
#
#     chi2_data_predictive_92split.parquet                (7.0 MB, 140457 rows)
#     chi2_data_predictive_92split.parquet.eval_cov.npz   (11 GB, 201 blocks)
#
# The chain then died on step 3 because it set KIKA_CHI2_RUN_ID but NOT
# KIKA_CHI2_METHODOLOGIES, so chi2_analysis_cluster.py fell through to its
# default `predictive_82` and went looking for that run's 11 GB sidecar, which
# had already been deleted for disk. `predictive_92split` is now REGISTERED in
# chi2_analysis_cluster.py and selected explicitly below.
#
# ⛔ Do NOT re-enable steps 1-2 unless the tape is rebuilt: re-running the
# precompute costs an hour and rewrites the same 11 GB sidecar byte for byte.
#
#   python p13_build_split_tape.py --run $P13RUN --split-dir $P13SPLIT \
#       --out $P13OUT --variant $P13VARIANT || exit 1
#
#   KIKA_THIS_WORK_DIR=$P13OUT \
#   KIKA_THIS_WORK_ENDF=26-Fe-56g_nominal_mg.endf \
#   KIKA_RUN_TAG=$P13TAG \
#       python precompute_chi2_predictive.py || exit 1

# ⚠⚠ RESUME 2026-08-12 (second time). The scoring AND both n_absorbed CSVs of
# the CANDIDATE are done and on disk. What was never measured is the BASELINE's
# n_absorbed, and §6e cannot be closed without it: `n_absorbed_91_cross.csv`
# exists, but 91_cross is the CROSS-TERM tape -- a different covariance -- so it
# is not the control for a candidate built on 91_rewrite's marginals. Scoring
# 92split against it would be comparing two changes at once.
#
# THIS JOB measures that control and nothing else, with the SAME arguments as
# the candidate so the two CSVs subtract row by row (same --max-n 16000, which
# skips Cierjacks in both; same seed 7 inside corpus_absorbed.py).
#
# ⚠ It reads 91_rewrite's 11 GB sidecar. It does NOT score and does NOT write
# any sidecar, so disk does not move.
#
# ===   the deliverable candidate = MF33's repair + the cross term
# ===========================================================================
#
# WHY THIS CANDIDATE. It is the only combination never scored, and the only one
# that puts a repair which passed BOTH gates next to the term that ships by
# physics:
#     92split    MF33 repaired, MF34 original, cross NOT read    V4 4.4358  ✅
#     91_cross   MF33 original, MF34 original, cross read        V4 8.4437
#     95         MF33 original, MF34 repaired, cross read        V4 8.5445  ⛔
#     THIS ONE   MF33 repaired, MF34 original, cross read        never measured
# Both bases are already on disk, so it buys two readings for one scoring:
# against 92split it prices the cross term on top of the repair, against
# 91_cross it prices the repair with the cross present.
#
# ⚠⚠ WHY ONLY THE GATE RUNS TODAY. §10.1: the a_0 cross block is
# Cauchy-Schwarz-compatible only with the marginals it was built from, and
# skipping that is what killed runs 89-90. p13_build_split_tape.py's own
# docstring refuses to reuse the cross tape for exactly this reason. Reading
# build_group_cross.py narrows what is actually at risk:
#   * `jj = d_tar / d_mc` uses ONLY the diagonals, and the split preserves
#     MF33's diagonal byte-for-byte, so `cx_post` comes out numerically
#     IDENTICAL either way. The cross block itself is NOT the risk.
#   * The risk is the JOINT. The certified object is [c33_post, cx_post; ...]
#     with `c33_post` the MC matrix, while the tape ships the SPLIT MF33
#     (coh 0.900 -> 0.391, PR 1.78 -> 8.06). So what the chi2 folds is not the
#     PSD-by-construction object — and the split moves it the risky way, by
#     SHRINKING the coherent variance (sigma_coh 0.0554 -> 0.0225) the cross
#     block leans on.
# `--c33-from-file --check` diagnoses precisely that, against the MF33 the chi2
# folds, and writes nothing. Roadmap: "puerta de §10.7-10 antes de gastar la
# hora" — so it is a JOB, not a `|| exit 1` in front of the expensive chain.
#
# ⛔ AND IT IS A HUMAN READ, not an exit code. `--check` prints diagnostics; it
# has not been proven to exit non-zero on a bad one, and this repo's own rule is
# that a gate is tested in both directions before anything depends on it.
#
# ⚑⚑ READ ROW F', AND ONLY F'. The first pass of this gate (job 8488778,
# 2026-08-14) passed everything it tested and STILL DID NOT TEST THIS CANDIDATE:
# row F is taken against `c33_ship`, which build_group_cross.py:698 loads from
# `run_dir/mf33_relative_covariance.npy` — the MF33 THE RUN PRODUCED, not the
# split MF33 in --source-endf. The only rows that read the file's MF33 were
# A'-C', and all three model the retired sidecar route. A' cannot stand in
# either: at Cx = 0 the binding eigenvalue is MF34's, so A' came back at
# -5.936e-05, the digit-for-digit twin of A, blind to the split.
# F' was added on 2026-08-14 (pure addition, backup in _backup_pre_fprime_row/)
# and is the pair that would actually ship: file MF33 + the a_0-convention cross.
#
# CRITERION, fixed before measuring: F' must land where F does.
#     sigma_max(K) 0.999993   lam_min_norm ~1e-16    <- healthy (F, and CONTROL)
#     sigma_max(K) 1026-1713  lam_min_norm -15       <- the sidecar route, B-E
#     sigma_max(K) 42113      lam_min      -114.7    <- B'/C', same route on the
#                                                       file's MF33
# ⚠ Compare sigma_max(K), NOT lam_min_norm: the normaliser is max|diag(J)| and
# the rows do not share it (§10.1.8-L12).
#
# If F' is healthy, uncomment STEPS 2-5 below and resubmit. If sigma_max(K)
# pulls away from 1, the cross block is NOT compatible with the split marginals
# and the tape must not ship — it has to be rebuilt against them, which is §8
# step 12(c) and a different job.
#
# COST: two ENDF parses, ~15-20 min. Writes nothing, scores nothing, and the
# disk does not move.

# P17RUN=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_92_integrated
# P17SRC=/share_snc/snc/JuanMonleon/chi2/p13_split/26-Fe-56g_nominal_mg.endf
# P17OUT=/share_snc/snc/JuanMonleon/chi2/p17_mf33split_cross
# P17ENDF=$P17OUT/26-Fe-56g_nominal_a0cross_mg.endf
# P17TAG=92split_cross

# mkdir -p $P17OUT
# ls -la $P17SRC || exit 1

# ===========================================================================
# 2026-08-14 (c): F' FAILED, AND THIS RUN IS THE CONTROL THAT ATTRIBUTES IT.
# ===========================================================================
#
# Job 8488793 measured, with the split tape as --source-endf:
#     F   a_0 blocks, run's own MF33   sigma_max(K)  0.999993   lam_min -5.56e-16
#     F'  a_0 blocks, FILE's MF33      sigma_max(K) 84.7158     lam_min -4.3828
# on the SAME scale (0.994484), so those lam_min are directly comparable. The
# criterion was "F' lands where F does". It does not.
#
# ⚑ The MF34 side is provably not the variable: F and F' returned rank34
# 2660/4217 and leak_null34 4.532378e-09 IDENTICALLY, to all seven digits. Same
# MF34, same cross block. Only MF33 differs.
#
# ⚠ BUT F' CHANGES TWO THINGS AT ONCE, so 84.7 is not yet attributable:
#   (1) the split;
#   (2) the MF33 object and its grid -- the 1738-group sidecar becomes the
#       file's 2317 bins, with the cross mapped nearest-bin and 579 bins zeroed.
# And (2) is NOT inert: on the sidecar route the same swap gives B -> B'
# 1712.8 -> 42112.6 (x24.6) and C -> C' 1026.0 -> 43061.8 (x42.0). Carrying
# x25-42 onto F's 0.999993 would already land at 25-42 without any split; we
# measured 84.7, i.e. roughly another x2-3 -- which is also what the physics
# predicts, since the split cuts the coherent variance to x0.165 and whitening
# amplifies by 1/sqrt(0.165) = 2.46. Suggestive, but that is arithmetic on a
# ratio, not a measurement.
#
# ⚠⚠ AND THE PRIMED ROWS ARE KNOWN TO BE PESSIMISTIC. The shipped tapes FOLD
# PSD: run 95's scoring printed [PSD] Lanczos min eigenvalue (This_work,
# 10571002) = +5.746e-08, positive, and 91_cross scored without incident. If F'
# were literally what the chi2 sees, run 95 would have died in potrf like runs
# 87-90. So these rows model a mapping the chi2 does not perform.
#
# THIS JOB RUNS THE GATE TWICE, control first, under identical code, so both
# tables land in one log:
#     PASS 1  --source-endf = run 92's OWN _mg tape   (MF33 NOT split)
#     PASS 2  --source-endf = the p13_split tape      (MF33 split)
# PASS 1 vs its own F isolates the grid/representation; PASS 2's F' against
# PASS 1's F' isolates the split, same grid and same mapping.
#
# READING CRITERION, FIXED BEFORE MEASURING -- on PASS 1's F':
#   ~84.7   the split is NOT the problem; the primed rows are merely pessimistic
#           and F is the row that adjudicates -> candidate CLEAR, uncomment 2-5.
#   ~25-40  the split costs x2-3 inside a model we know is pessimistic -> the
#           primed rows cannot adjudicate; decide on the real precompute's [PSD]
#           line or rebuild the cross.
#   ~1      the split IS the cause -> CONDEMNED; rebuild the cross block against
#           the split marginals (§8 step 12(c)), do not ship this tape.
# ⛔ If PASS 1's F' is also ~85, then 91_cross -- already scored, and the shape
# of the deliverable -- carries the same number. That would not invalidate it
# (it folds PSD, measured), but it would retire the primed rows as a gate.
#
# COST: four ENDF parses, ~35-40 min. Writes nothing, scores nothing.

# P17CTRL=$P17RUN/26-Fe-56g_nominal_mg.endf
# ls -la $P17CTRL || exit 1

# ⚠ SEPARATE PARSE CACHES, AND IT IS NOT COSMETIC. build_group_cross.py keys the
# cache on `{basename}__{st_size}` (lines 88 and 159), and the two tapes are BOTH
# `26-Fe-56g_nominal_mg.endf` at exactly 205,838,091 bytes -- the split rewrites
# MF33's values without changing a single record count. So the key COLLIDES and
# one cache dir would silently serve pass 1's parse to pass 2.
# It happens to be harmless today: the only cached objects are the MF34 group
# edges and the MF33 energy grid, and the split changes neither -- the covariance
# itself is read fresh by load_library_lib_c0 every time. Job 8488793 was
# therefore not contaminated. But the collision is real, so do not merge these.

# ===========================================================================
# SIGUIENTE: PUNTUAR LA RUN 97 (la malla por orden). Descomentar CUANDO PASE.
# ===========================================================================
# La run 97 SI se puntua: no es una reproduccion. Descomentar solo despues de
# ver "✅ RUN 97 PASA" en el log de run_pyscript.sh -- si su gate falla, el
# numero no significa nada.
#
# R97=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_97_perordermesh
# ls -la $R97/26-Fe-56g_nominal_a0cross_mg.endf || exit 1
#
# KIKA_THIS_WORK_DIR=$R97 \
# KIKA_THIS_WORK_ENDF=26-Fe-56g_nominal_a0cross_mg.endf \
# KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
# KIKA_RUN_TAG=97mesh \
#     python precompute_chi2_predictive.py || exit 1
#
# KIKA_CHI2_METHODOLOGIES=predictive_97mesh \
# KIKA_CHI2_RUN_ID=97mesh \
#     python chi2_analysis_cluster.py || exit 1
#
# ⚠ El baseline es la run 96 = la run 94. Su chi2 ya esta; no se re-puntua.
# ⚠ ~1.5 h y un sidecar de ~11 GB, que se borra en cuanto se lea el parquet.

# ===========================================================================
# CURRENT JOB (2026-08-17): THE READ-BACK GATE ON THE ANALYTIC DELIVERABLE.
#
# The one gate of the r2 chain that has never run. It died with [Errno 5] on
# 2026-08-14 reading the mount, and a local retry on 2026-08-17 was killed after
# an hour still inside the parse -- that is the 9p mount, not the file, which is
# why it belongs here. Expect ~15-30 min, almost all of it in read_endf.
#
# WHAT IT MEASURES, and what it no longer has to. Block-by-block fidelity is
# already settled: the tape's MF34 was decoded from its own TEXT on 2026-08-17
# and reproduces the in-memory diagonals to six figures on all six orders
# (docs/deliverable_tape_audit.md Sec. 2.2). What is left is the JOINT as a
# consumer assembles it -- ENDF's 11-character floats are ~1e-7 on every entry
# and the joint sits ON the Cauchy-Schwarz boundary by construction, so the slop
# is the same order as the headroom.
#
# PASS: the FILE row shows sigma_max(K) <= 1 and lam_min at round-off. The
# MEMORY row is the control and must read sigma_max(K) = 0.99949.
# ⚠ Read sigma_max(K), NOT lam_min_norm -- the two rows do not share a scale.
#
# ⚑ --joint is the STAGE dir: its magnitude block is called
# mf33_absolute_covariance.npy, not r2g_c33.npy, and the gate now takes either.
# ⚑ --c0-host must be the one rebuild_mf33 PRODUCED (in r2_analytic_tape), never
# the reference run's -- they are different arrays.
# ⚠ If it is OOM-killed, uncomment #SBATCH --mem=300G at the top: the a0 blocks
# are 6 x 2317 x 703 and the reader retains them.
R2STAGE=/share_snc/snc/JuanMonleon/chi2/r2_analytic
R2TAPEDIR=/share_snc/snc/JuanMonleon/chi2/r2_analytic_tape
R2ENDF=$R2TAPEDIR/26-Fe-56g_nominal_a0cross_mg.endf

ls -la $R2ENDF || exit 1

# STEP 1 -- DONE, job 8494554 (2026-08-17), PASSED:
#   FILE  sigma_max(K) 0.999488   lam_min +1.38454e-07   leak_null34 0
#   rank33 1738/1738  rank34 2627/2627   cross file-vs-memory 1.643e-07
# i.e. the tape IS the object we built, and the joint stays inside
# Cauchy-Schwarz as a consumer reads it. Re-enable only if the tape is rewritten.
#
# python r2_readback_gate.py \
#     --endf $R2ENDF \
#     --joint $R2STAGE \
#     --c0-host $R2TAPEDIR/mf33_c0_host.npy \
#     --kika /share_snc/snc/JuanMonleon/EXFOR || exit 1

# --- STEP 3: THE CLOSING HALF OF THE EXPOSURE (P1 + P2) --------------------
# Registry entry ALREADY ADDED (that is what killed 8497757, and 85 and 89
# before it). ~1.5 h + an 11 GB sidecar; 711 GB free, and the two existing
# sidecars can go once their parquets are read.
# ⛔ LEE ESTO ANTES DE ENVIAR (2026-08-19). Esto puntua la cinta ANALITICA R2,
# que `r2_group_joint.py` construyo con las cargas del estimador GANADOR. Hoy se
# ha medido que ese no es el estimador que la cinta lleva: a_6 tiene central en
# 74 de 703 grupos por esa ruta y en 694 por la mezcla. Son ~1.5 h y un sidecar
# de 11 GB para un numero construido sobre la covarianza equivocada.
# Antes: rehacer la cadena R2 con `r2_group_joint.py --estimator mixture`.
KIKA_MF34_NULL_MASK=/share_snc/snc/JuanMonleon/chi2/mf34_p1p2_mask_r2analytic.npz \
KIKA_THIS_WORK_DIR=$R2TAPEDIR \
KIKA_THIS_WORK_ENDF=26-Fe-56g_nominal_a0cross_mg.endf \
KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
KIKA_RUN_TAG=r2analytic_nodegp2 \
    python precompute_chi2_predictive.py || exit 1

KIKA_CHI2_METHODOLOGIES=predictive_r2analytic_nodegp2 \
KIKA_CHI2_RUN_ID=r2analytic_nodegp2 \
    python chi2_analysis_cluster.py || exit 1
#
# Follow-up if step 2 moves anything: the same with
# mf34_p1p2_mask_r2analytic.npz (504 slots, 46.3 % of the variance), which adds
# the slots whose denominator MF4 cannot reproduce at all.
# --- DONE (job 8481007). Do NOT re-enable. ---------------------------------
# Scoring of the candidate: veto exact (JEFF/JENDL 0.00000 %), Kinney -43.2 %,
# off-Kinney -37.3 %, no_Cierjacks 6.3075 -> 4.4358. And its two n_absorbed
# CSVs, 66 experiments x 3 libraries:
#   n_absorbed_92split.csv         This_work sum 3465.0 over N=18188
#   n_absorbed_92split_kinney.csv  This_work 1792.2 over N=13208 (13.57 %)
#
#   KIKA_CHI2_METHODOLOGIES=predictive_92split KIKA_CHI2_RUN_ID=$P13TAG \
#       python chi2_analysis_cluster.py || exit 1
#   python corpus_absorbed.py $P13TAG --probes 128 \
#       --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_${P13TAG}.csv || exit 1
#   python corpus_absorbed.py $P13TAG --only 10571002 --max-n 30000 --probes 256 \
#       --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_${P13TAG}_kinney.csv || exit 1
#
# ⚑ The baseline is NOT re-scored either: run_091_rewrite/{report.md,summary.json}
# already exist from 2026-08-08 and criterion 1 reads against them.
#
# ⚠ Version of this block that died at step 3 (job 8480897):
#   KIKA_CHI2_RUN_ID=$P13TAG python chi2_analysis_cluster.py || exit 1

df -h /share_snc | tail -1

# --- PREVIOUS JOB (run 87 damping scan) — flip back by uncommenting ---
#
## ---------------------------------------------------------------------------
## CURRENT JOB: run 87 damping scan — find the largest MF33<->MF34 cross block
## the shipped covariance can actually carry.
##
## WHY. Run 87 at full strength (job 8445747) produced its parquet and its
## 10.9 GB sidecar and then DIED in the analysis:
##
##     numpy.linalg.LinAlgError: Internal potrf return info = [23]
##     chi2_metrics._solve -> cho_factor, and the jitter fallback failed too
##
## That is not a distorted result, it is no result: Sigma_V4 is not positive
## definite for at least one experiment, so chi^2 is undefined. The diagnosis was
## already visible in the parquet — 1005 of 46819 This_work points (2.15 %) came
## out with a NEGATIVE Sigma_eval diagonal. Sigma^MF33 and Sigma^MF34 come from the
## shipped file (multigroup collapse + near-zero guard) while Sigma^cross comes
## from the raw MC, so the three are not mutually consistent and the per-point
## Cauchy-Schwarz bound |Sigma_cross(j,j)| <= 2 s33 s34 is violated.
##
## THE KEY FACT THIS JOB EXPLOITS. Sigma_eval(s) = Sigma^MF33 + Sigma^MF34 +
## s*Sigma^cross is EXACTLY LINEAR in s. The precompute now records
## `sigma_eval_var_diag` — the SIGNED, unclipped diagonal — so ONE run at any
## scale fixes Sigma^cross's diagonal everywhere,
##
##     c(j) = ( v_s(j) - v_0(j) ) / s
##
## and hence the largest diagonal-safe scale s_max = min_j v_0(j)/(-c(j)), for
## free, offline, from a 5 MB parquet. Run 87's own parquet CANNOT give this: its
## 1005 failing points were clipped to zero before being written, and those are
## precisely the points with the smallest s*. Hence the re-runs.
##
## ORDER MATTERS HERE. The precomputes come first and the analyses last, because
## a precompute always succeeds while an analysis crashes on a non-PD block. The
## deliverable — s_max — is produced by step 3, before anything can die. The two
## analyses are deliberately NON-FATAL (`|| echo`), so one non-PD scale does not
## throw away the other.
##
## A non-negative diagonal is NECESSARY, NOT SUFFICIENT for PSD. The precompute
## now also reports the smallest eigenvalue: exactly for every group of N <= 2500,
## and by Lanczos (a few dozen matvecs instead of an O(N^3) decomposition) for
## Cierjacks and K&S, which are 90 % of the points. Read those [PSD] lines before
## adopting any scale.
##
## Resources: 300G. Two precomputes at ~35 min plus two analyses at ~25 min is
## ~2.5 h, well inside the 12 h.
##
## DISK: each sidecar is ~11 GB and the share is at 94 %. If space is short, the
## s=0.25 pair can be dropped — it is a linearity self-check, not a requirement.
## ---------------------------------------------------------------------------
#
#XDIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_86_mgfix
#
## --- Step 1: precompute at scale 0.50 ---------------------------------------
#KIKA_THIS_WORK_DIR=$XDIR \
#KIKA_MF33_MF34_CROSS_DIR=$XDIR \
#KIKA_MF33_MF34_CROSS_SCALE=0.5 \
#KIKA_RUN_TAG=87_s050 \
#    python precompute_chi2_predictive.py || exit 1
#
## --- Step 2: precompute at scale 0.25 ---------------------------------------
## Not needed to find s_max — one scale already fixes it. It is the SELF-CHECK:
## the s=0.25 diagonal must equal the s=0.50 one interpolated linearly, or
## something other than the cross block moved between the two runs.
#KIKA_THIS_WORK_DIR=$XDIR \
#KIKA_MF33_MF34_CROSS_DIR=$XDIR \
#KIKA_MF33_MF34_CROSS_SCALE=0.25 \
#KIKA_RUN_TAG=87_s025 \
#    python precompute_chi2_predictive.py || exit 1
#
## --- Step 3: THE DELIVERABLE — exact s_max, before anything can crash -------
## Needs only the parquets. Prints the per-point limit distribution, s_max, and
## how many points would still go negative at 0.75 / 0.5 / 0.25 / 0.1.
#KIKA_CROSS_BASE_RUN=086 KIKA_CROSS_BASE_TAG=86 \
#KIKA_CROSS_RUN=087_s050 KIKA_CROSS_TAG=87_s050 \
#KIKA_CROSS_SCAN="087:1.0,087_s050:0.5,087_s025:0.25" \
#    python compare_cross_block_runs.py \
#    || echo "[WARN] comparison reported a failed gate — expected while scanning; continuing"
#
## --- Step 4: analyses. NON-FATAL by design ----------------------------------
## `chi2_metrics._solve` raises LinAlgError on a non-PD block and its jitter
## fallback is deliberately too small to paper over a real negative eigenvalue.
## A scale that is still not PD will die here; that is INFORMATION, and it must
## not take the other scale down with it.
#KIKA_CHI2_METHODOLOGIES=predictive_87_s050 KIKA_CHI2_RUN_ID=087_s050 \
#    python chi2_analysis_cluster.py \
#    || echo "[FAIL] scale 0.50 is still not PD — no chi2 at this scale"
#
#KIKA_CHI2_METHODOLOGIES=predictive_87_s025 KIKA_CHI2_RUN_ID=087_s025 \
#    python chi2_analysis_cluster.py \
#    || echo "[FAIL] scale 0.25 is still not PD — no chi2 at this scale"
#
## --- Step 5: the scan table, now with whatever chi2 survived ----------------
#KIKA_CROSS_BASE_RUN=086 KIKA_CROSS_BASE_TAG=86 \
#KIKA_CROSS_RUN=087_s050 KIKA_CROSS_TAG=87_s050 \
#KIKA_CROSS_SCAN="087:1.0,087_s050:0.5,087_s025:0.25" \
#    python compare_cross_block_runs.py \
#    || echo "[WARN] comparison reported a failed gate — read it, do not ignore it"
#

unset KIKA_UNCERTAINTY_MANIFEST_PATH

# 4) Deactivate
deactivate

