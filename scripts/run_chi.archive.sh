# Archivo de run_chi.sh. NO se ejecuta: bloques ya comentados, movidos aqui
# el 2026-08-19. Para relanzar uno, copiar su bloque al runner.

# ---------------------------------------------------------------------------
# CURRENT JOB: re-certify against the file's MF33, then RE-SCORE RUN 86 with the
# unsupported MF34 parameters removed. Roadmap §10.6-3 item 5, and the §10.6-1
# escalation. Step 1 is ~15 min; steps 2-3 are the long part (~1 h + ~30 min)
# and write an 11 GB sidecar. Overnight job.
#
# WHERE THE TRACK STANDS (job 8451762, §10.1.8-L14). lam_min is INVARIANT at
# -26.308 across all four constructions that include Cx -- the MF34
# correlations, the null-slot fill and whether MF33 is rebuilt all move it by
# less than 1e-3. Only the presence of Cx matters (Cx = 0 gives -0.0621). That
# is the signature of a units mismatch in the FOLD, §10.1.8-L13, reproduced
# synthetically in §10.1.8-L13.1: the r = 1 control is PSD at -2.4e-15 and every
# r != 1 row is not. The fix is the a_0 route (§10.6-3) and it is NOT in this job.
#
# --- STEP 1 (§10.6-3 item 5): is the MF33 we certify against the one we fold? --
#
# Every PSD row in §§J, K, L3, L12 and L14.1 takes c33 from
# `mf33_multigroup_relative_covariance.npy` -- 188 adaptive groups over
# 0.8468-4.075 MeV, which is also the cross sidecar's magnitude axis. The chi2
# reads MF33 out of the ENDF instead: 2317 bins, averaged onto each point via W.
# THIS IS §L3'S ERROR WITH THE MAGNITUDE LEG IN PLACE OF THE SHAPE LEG -- a trio
# verified that is not the trio folded -- and nobody had checked it for MF33.
#
# --c33-from-file adds rows A'/B'/C' taken against the file's own MF33, with the
# cross block's magnitude axis expanded onto that grid by duplication. Read A'
# against A and B' against B.
#
# ⚠ `|| echo` NOT `|| exit 1`, deliberately. This step is diagnostic and the
# overnight re-score does not depend on it; an ENDF-read failure here must not
# cost the long job. Steps 2 and 3 keep `|| exit 1`.
#
# --- STEPS 2-3 (§10.6-1 escalation): run 86 without the dead parameters -------
#
# `null_slot_exposure.py` sized it at 2.14 % of the Sigma_eval diagonal, 82.9 %
# of points touched, two small datasets above 44 % and Cierjacks at 3.0 %. A
# diagonal share does NOT map linearly onto a chi2 -- the chi2 inverts the full
# per-experiment block -- so it is re-scored rather than scaled.
#
# THE COMPARISON IS predictive_86_nonull AGAINST predictive_86: same ENDF, same
# centrals, same MF33, same MF34, NO cross block in either. The ONLY difference
# is that the 1542 (group, order) slots `row_aggregator` never populated no
# longer contribute. That difference IS the bias, and it is one-sided in our
# favour because JEFF and JENDL ship no such term.
#
# ⚠ SAME DIRECTORY AS RUN 86 (new_test_86_mgfix), not run 88's. They are
# byte-identical (§10.1.7-A) but using run 86's own path removes any doubt that
# this is a single-variable comparison.
#
# ⚠ THE MASK COMES FROM RUN 88's REPLICAS, and that is sound rather than
# convenient: only run 88's directory carries `mf33_c0_samples.parquet` (§K),
# and the two runs' _mg.endf are byte-identical, so the collapse inputs that
# determine which (group, order) the MC populated were identical too. It is an
# inference from byte-identity, not a direct measurement on run 86.
#
# ⚠ predictive_86_nonull IS REGISTERED in chi2_analysis_cluster.py. Runs 85 and
# 89 both died after their precompute for want of that entry (§10.1.8-L1).
#
# Resources: 300G, as inherited. Run 86 itself needed 100G; step 1 adds one ENDF
# parse and a 6535^2 eigendecomposition on top.
# Disk: /share_snc was at 86 % (1.2 T free) when this was prepared. The new
# sidecar is ~11 GB. Run 90's 11 GB sidecar produced no chi2 and is dead weight
# -- worth deleting, but that is not this job's call.
# === CURRENT JOB, 2026-08-07: PRICE THE MF33 REGROUPING (roadmap §10.7-4/2) ===
#
# WHAT IS BEING DECIDED. §10.7-3 says grouping MF33 in the multigroup file is the
# ENABLING step for the cross term, not a size optimisation. Three reasons, and
# the first two are already proved:
#
#  1. It is what makes the fold a CONGRUENCE. `scripts/tests/test_fold_maps.py`
#     shows Sigma_eval is literally M J M^T when MF33 is folded on the same grid
#     the cross block's magnitude axis uses, and is not when it is not. The
#     deciding quantity is `||C_f - P C_f P^T|| / ||C_f||`, the structure the
#     collapse discards -- measured on run 86 at **42.6 %**, and INDEFINITE
#     (lam in [-1.578, +2.186]). We are deep in the broken regime.
#  2. It is what lets MF33 survive ENDF at all. Condition number **1.6e9** fine
#     against six significant digits -- which is why the shipped MF33 reads back
#     with 237 negative eigenvalues and rank 1916/2317 -- versus **1.0e6**
#     grouped.
#  3. It takes the cross block from **+132 MB** to **+11 MB**, and drops MF33
#     itself from 75.7 MB to ~1 MB. MF33+MF34 are 91 % of the file.
#
# WHAT IT COSTS is the only thing not yet known, and it is what this job buys:
# relative sigma median 0.0616 -> 0.0528 and PEAK **0.204 -> 0.113 (-45 %)**.
# Less evaluated variance means a LARGER chi2 for This_work, so expect us to
# look worse and the JEFF gap to narrow. §10.7-5 puts the kill line at ~5 %.
#
# ⚠ SINGLE VARIABLE. `regroup_mf33.py` copied run 86's shipped file and replaced
# ONLY the MF33/MT2 self-block; MF34, MF4, MF3, the centrals and MT1 are
# byte-identical, and neither file carries a cross block. **JEFF and JENDL must
# move by exactly 0.0.** If they move, something else moved and the comparison
# is void -- that control is what made 086_nonull readable.
#
# ⚠ MT2 ONLY, on purpose: `precompute_chi2_library_c0` reads the MT=2 self-block
# and nothing else, so regrouping MT1 would change the file without changing the
# chi2 -- two variables for the price of one.
#
# ⚠ `predictive_86_mf33grouped` IS REGISTERED in chi2_analysis_cluster.py. Runs
# 85 and 89 both died after their precompute for want of that entry (§L1).
#
# Disk: /share_snc at 86 % (1.2 T free). The new sidecar is ~11 GB; delete it
# once the parquet is read, as run 88's and 90's were.
# Resources: 300G/4h as inherited; run 86 itself needed 100G.
#
# COMPARE run_086_mf33grouped/summary.json against run_086/summary.json.
#   moves < ~1 %  -> grouping is free; take it and move on to §10.7-4 step 3.
#   moves 1-5 %   -> the price of a congruent fold and a valid MF33. Take it,
#                    but re-quote every chi2 delta in the write-up against it.
#   moves > 5 %   -> §10.7-5 fires. Either the fine route (+132 MB) or stop.
# ---------------------------------------------------------------------------

# === CURRENT JOB, 2026-08-08: THE CROSS TERM, IN MF34's a_0 BLOCKS ===========
# Roadmap §10.7-4 step 5. The term has never been scored: its only source was a
# .npy sidecar declaring is_relative=False against a relative MF34 family, which
# the §L13 guard refuses. Runs 87-90 died on potrf and produced no χ².
#
# ONE FILE, TWO SCORINGS, and that is what makes it single-variable. The a_0
# blocks are in the file for both steps; `build_mf34_block` skips l_r < 1, so
# with KIKA_MF33_MF34_CROSS_FROM_FILE unset they are simply not read. The
# marginals are BYTE-IDENTICAL between step 2 and step 3 -- the same bytes, not
# the same construction.
#
# MEASURED BEFORE LAUNCH (§10.7-10 row F, on run 88's replicas against run 86's
# shipped file). In the space the chi2 folds:
#
#     row                              sigma_max(K)   leak_null34   lam_min_norm
#     D  sidecar route (runs 87-90)      1055.41       4.02e-04      -1.13e+02
#     F  a_0 blocks, one convention         0.999993   4.53e-09      -1.42e-16
#
# 0.999993 is the CONTROL's own value, which is what a congruence requires. The
# joint itself is PSD at -5.6e-19 on the fine axis, and the c33 matrix gate
# reads 4.575e-16 -- the joint's magnitude block IS the matrix the chi2 folds.
#
# --- STEP 1: write the file (~40 min) ---------------------------------------
# Replicas come from run 88 (only its directory carries mf33_c0_samples.parquet);
# the template is run 86's _mg.endf, and the two are byte-identical (§10.1.7-A,
# re-verified by md5 on 2026-08-08). --mag-grid fine writes NO .npy sidecar and
# touches no run directory: the cross term ships inside the ENDF, and §L8
# inverted says the sidecar must be absent.
#
# ⚠ `|| exit 1` here, unlike the old step 1. Steps 2 and 3 read the file this
# writes, so there is nothing to salvage if it fails.
#
# --- STEPS 2-3: score it ------------------------------------------------------
# ⚠ READ STEP 2 AGAINST predictive_86_nonull, NOT predictive_86. --null-fill
# zero and the null mask remove the SAME 1542 (group, order) slots, so the
# rewrite has already had §L14.2's 2.14 % of the diagonal removed.
#
# ⚠ STEP 2 IS NOT EXPECTED TO BE A NULL RESULT. It carries two changes against
# 086_nonull -- MF34's correlations now come from the collapsed replicas, and
# the 39 shape groups outside 0.8468-4.075 MeV lose the host's merged MF34
# (178 of 46819 points). Its job is to absorb both, so that step 3 minus step 2
# is the cross term alone. §10.7-5 predicts 1-3 pp for that difference.
#
# ⚠ THE NULL MASK IS SET IN BOTH, and it is a no-op on this file (those slots
# are already zero in the MF34 blocks AND in the cross blocks). Setting it
# anyway keeps the only difference from 086_nonull the file itself.
#
# ⚠ predictive_91_rewrite and predictive_91_cross ARE REGISTERED in
# chi2_analysis_cluster.py. Runs 85 and 89 both died after their precompute for
# want of that entry (§10.1.8-L1).
#
# Resources: 300G, as inherited. Step 1 holds a 5956^2 joint plus several
# eigendecompositions; steps 2-3 fold a 2317-bin magnitude axis against 703
# shape groups over 6 orders, ~256 s for Cierjacks alone after the gather fix.
# Walltime 8 h is enough for all three.
# Disk: ~345 MB for the ENDF, plus ~11 GB per chi2 sidecar. 1.2 T free.
# ===========================================================================
# === CURRENT JOB, 2026-08-09: THE CIERJACKS CORRECTION, AND THE CROSS ======
# ===              TERM'S DOSE-RESPONSE.  Roadmap §10.8-14, §10.8-5 step 4.
# ===========================================================================
#
# TWO INDEPENDENT PARTS, CHEAPEST AND MOST DECISIVE FIRST. Part A needs no
# precompute at all and is done inside the first hour; part B costs a precompute
# and an 11 GB sidecar per point. If the walltime kills anything it will be B's
# tail, and A is already written.
#
# --- PART A: score run 91 with Cierjacks' backward points CORRECTED ---------
#
# §10.8-13's fourth option, through §10.8-14's gate. The integral test passed:
# Cierjacks' angle-integrated sigma_el is +14.3 % above JEFF's MF3 against a
# corpus consensus of +4.0 %, and a backward step moves it TOWARD the consensus,
# landing on it at A = 0.275 (median of six determinations). The form is a STEP
# at 90 deg, not a P_1 tilt: it reproduces §10.8-10's measured mode ratio
# Delta a_2 / Delta a_1 = 0.276 to three decimals, where the tilt gives 0.19-0.22
# and pushes the integral the wrong way. Neither test used the residual.
#
# ⚑ NO PRECOMPUTE AND NO NEW SIDECAR. Sigma_eval is folded from (mu, E, c_0,
# a_l) and never sees y_exp, so the 11 GB sidecar that 91_cross already has is
# still exactly right. KIKA_CHI2_EVAL_COV points at it. That is why this part
# costs ~25 min a row instead of ~3 h.
#
# The parquets were written and control-checked from WSL by
# `myworkspace/chi2/correct_cierjacks_backward.py`: untouched rows byte-
# identical, the declared relative systematics byte-identical (Cierjacks keeps
# its 5 % normalization and 7 % ERR-T untouched -- nothing is overridden), and
# `_eval_pos` preserved so the sidecar's blocks still line up.
#
# ⚠ THE CORRECTION IS APPLIED TO ALL THREE LIBRARIES' ROWS. It corrects the
# measurement, not our evaluation, so JEFF's and JENDL's chi2 move too. That is
# what makes it non-self-serving, and BOTH tables get reported.
#
# ⚠⚠ THE FREE CONTROL, AND IT IS THE FIRST THING TO READ: `no_Cierjacks` MUST
# come back EXACTLY unchanged -- it excludes every row this touches. If it
# moves, the wrong rows were modified and neither row may be read.
#
# WHAT TO EXPECT, recorded in advance:
#   only_Cierjacks and all   improve, and JEFF improves at least as much as we
#                            do (§10.8-12 M5's qualitative half, which survived)
#   no_Cierjacks             0.000 %, exactly
#   A = 0.30 vs A = 0.55     the dose-response. 0.30 lands the integral on the
#                            consensus; 0.55 takes it ~5 % below, and buys 46 %
#                            of the shape displacement instead of 31 %.
#
# ⚠ WHAT PART A CANNOT DO, so it is not over-read: the headline subset is
# `no_Cierjacks`, and no change to Cierjacks' own data can move it through the
# DATA. Reaching the headline needs the corrected data to go back through the
# MC and restructure MF34 -- a re-evaluation, §10.8-12 M6, not a re-score.
# Part A prices the data-side effect and gates whether that is worth spending.

# DONE (job 8470630 part A) and read — §10.8-17. The correction fixes the data
# and does not buy chi2: only_Cierjacks V1 falls -31 % (us), -24 % (JEFF),
# -26 % (JENDL), but at V4 it is a +8.6 % penalty and the headline does not move.
# Kept so it can be flipped back; do not re-run.
#
# CJSIDE=/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_91_cross.parquet.eval_cov.npz
#
# ls -la $CJSIDE || exit 1
#
# KIKA_CHI2_EVAL_COV=$CJSIDE \
# KIKA_CHI2_METHODOLOGIES=predictive_91_cj030 KIKA_CHI2_RUN_ID=091_cj030 \
#     python chi2_analysis_cluster.py || exit 1
#
# KIKA_CHI2_EVAL_COV=$CJSIDE \
# KIKA_CHI2_METHODOLOGIES=predictive_91_cj055 KIKA_CHI2_RUN_ID=091_cj055 \
#     python chi2_analysis_cluster.py || exit 1

# --- PART B: is the cross term's +33.9 % linear in the cross block? ---------
#
# §10.8-5 step 4, cheap insurance against a sixth retraction. The endpoints are
# already measured -- s = 0 is 91_rewrite (no_Cierjacks V4 6.308) and s = 1 is
# 91_cross (8.444) -- so ONE interior point tests linearity.
#
#   LINEAR PREDICTION AT s = 0.50: no_Cierjacks V4 = 7.376.
#
# Materially above that and the chain amplifies: §10.8-1's certificate has to be
# re-read before +33.9 % is quoted again. s = 0.25 (predicted 6.842) is the
# second point and is the one to drop if the queue runs short.
#
# ⚠ s is read inside precompute_chi2_predictive.py, NOT in the analysis, so each
# point costs a full precompute and its own ~11 GB sidecar. The roadmap's
# "re-scoring only" meant "no re-evaluation". Delete both sidecars once the
# parquets are read.
#
# ⚠ Same ENDF, same everything else, as steps 2-3 of the run-91 job below.

# ===========================================================================
# CURRENT JOB — the informativeness column, Cierjacks' block. Roadmap §10.8-19.
# ===========================================================================
#
# `n_absorbed` = the effective number of an experiment's directions in which a
# library's Sigma_eval declares anything, measured in that experiment's own
# metric. It is the column the chi2 table cannot ship without: JENDL scores
# 0.034 on Cierjacks and 0.059 on Kinney with SEVEN effective directions out of
# 13208, and without this number that reads as accuracy rather than as a
# covariance coarse enough to absorb anything.
#
# 66 of 67 experiments are ALREADY DONE on the workstation
# (myworkspace/chi2/n_absorbed_91_cross.csv). Only Cierjacks was skipped: its
# block is 28631^2 = 6.6 GB in float64 against a 12 GB laptop. This job is that
# one block, three libraries, and it completes the column.
#
# READ-ONLY. It reads the existing 91_cross sidecar and writes one small CSV.
# No precompute, no evaluation, no new sidecar, nothing deleted.
#
# Prediction to check on arrival, from the 66 already measured: This_work spends
# far more directions than JEFF and JEFF far more than JENDL, and JENDL's 0.034
# on Cierjacks should come with a single-digit n_absorbed.
#
# Runtime: three 28631^2 Choleskys, ~10-20 min each => well inside the 12 h
# header. Memory is the binding constraint, not cpu: one float64 28631^2
# (6.6 GB) plus the mmap'd float32 block (3.3 GB); the 300G header is ample.

# --- DONE (job 8481178). Do NOT re-enable. ----------------------------------
# THE VETO PASSED: n_absorbed outside Kinney +3.73 % (1611.663 -> 1671.776 over
# N=4980), Kinney +13.6 %, corpus +8.61 %, and JEFF/JENDL at 0.000e+00 on 66
# experiments each. §6e is closed with a measurement, not an inference.
#
# BASETAG=91_rewrite
# python corpus_absorbed.py $BASETAG --probes 128 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_${BASETAG}.csv || exit 1
# python corpus_absorbed.py $BASETAG --only 10571002 --max-n 30000 --probes 256 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_${BASETAG}_kinney.csv || exit 1

# ===========================================================================
# === CURRENT JOB, 2026-08-14: SCORE RUN 95 — MF34's SPLIT-COMBINE ==========
# ===   post-run-92 roadmap §2.7-quater, step 12(e).
# ===========================================================================
#
# Run 95's own gate passed in all six parts, so the tape is admissible: MF33's
# six .npy, nominal_fits.parquet and mf34_std_perbin.npy all byte-identical to
# run 94, the saved corr_kw equal to the p16 input at 0.000e+00, the pipeline
# rebuilding the p16 product to 2.2e-16, and all three tapes differing as
# required. Config diff against run 94: 154 keys vs 153, single addition
# MF34_CORR_OVERRIDE. It is monovariable.
#
# ⚠⚠ THIS IS NOT THE ROADMAP HANDOFF'S COMMAND, and the difference is not
# cosmetic. The handoff wrote:
#     KIKA_THIS_WORK_DIR=$R95 KIKA_MF33_MF34_CROSS_DIR=$R95 KIKA_RUN_TAG=95
# which gets two things wrong at once:
#   1. KIKA_THIS_WORK_ENDF defaults to `26-Fe-56g_nominal_mg.endf`, so it would
#      score the NON-cross tape instead of the deliverable `_a0cross_mg`.
#   2. KIKA_MF33_MF34_CROSS_DIR is the SIDECAR route. Per precompute's own
#      §10.7-4 note the sidecar "declares is_relative=False against a relative
#      MF34 family, which the §L13 guard refuses outright, so it has never been
#      scorable at all" — and it is what killed runs 87-90.
# The base `predictive_91_cross` was scored through the a_0 blocks, and a
# candidate must be read against its base on the same route. Hence
# KIKA_MF33_MF34_CROSS_FROM_FILE=1 and the explicit tape below.
#
# ⚠ KIKA_MF34_NULL_MASK IS DELIBERATELY OMITTED, and it is the one open choice
# inside this job. `mf34_null_mask.npz` is on run 86's 703-group shape grid and
# its 1542 slots come from run 86's own `|a_nom_group| < tiny`; run 95's MF34
# sits on 896 groups, so it is not the same set. eval_covariance places the mask
# BY ENERGY and precompute only checks the npz against itself, so a stale mask
# would be applied silently rather than raise. The tape should already carry the
# removal (CROSS_NULL_FILL='zero' plus write_consistent_mf34's `live`
# complement, which §10.6-1 says remove the SAME slots) — but that is an
# INFERENCE, not a measurement, so run 95 vs 91_cross moves the null-slot
# treatment as well as MF34's correlation. To close it rather than declare it,
# rebuild the mask on run 95's grid FIRST and pass it to step 1:
#     python build_group_cross.py --run $R95 --write-null-mask \
#         --out /share_snc/snc/JuanMonleon/chi2/mf34_null_mask_95.npz
#
# ⚠⚠ AND THE ONE THIS JOB DOES NOT SETTLE, because it is Juan's call:
# RUN 95 DOES NOT CARRY MF33's REPAIR. Measured, not inferred — run 95's
# mf33_absolute_covariance.npy has the same md5 (c68ac0ac...) as run 92's, while
# the split MF33 is a different object (4511d2b7..., in
# myworkspace/chi2/p12_new_test_92_integrated/). So this prices MF34's repair
# ALONE. `predictive_92split` priced MF33's repair alone, with the cross NOT
# read. The tape carrying BOTH repairs plus the cross — which is what §8 step 12
# calls the deliverable and what the track's objective names — does not exist.
#
# ⚑ MEMORY IS THE OPEN RISK AND THE HEADER WAS LEFT ALONE ON PURPOSE. The
# header's own warning says the scoring reads an 11 GB sidecar and the MF34
# fine-grid work wants ~300G; run 95's MF34 is FINER than anything scored so far
# (896 groups, 485 MB tape against 91_cross's 703 groups and 347 MB), so the
# par_IB default may not be enough. `--mem=300G` was NOT put back because
# par_IB's node memory cannot be queried from WSL (no slurm client) and a --mem
# above the node size leaves the job PENDING instead of failing loudly. If it is
# OOM-killed, uncomment the line in the header — that is a one-word change and
# the diagnosis is already written.

# ===========================================================================
# === CURRENT JOB, 2026-08-14 (b): THE GATE, AND ONLY THE GATE ==============
# --- THE BLEND SWEEP: how much of the split survives next to the cross? -----
# One pass. The sweep's own endpoints are the controls, so PASS 1/PASS 2 of the
# previous job are not repeated:
#     s = 0  must reproduce the CONTROL's F'    sigma_max(K) ~ 1.020
#     s = 1  must reproduce the CANDIDATE's F'  sigma_max(K)   84.716
# If either endpoint misses, the blend is not between the two objects we think.
# --- DONE (job 8488838, 2026-08-14). OPTION D IS DEAD. Do NOT re-enable. ---
# Both endpoints reproduced exactly (s = 0 -> 1.020488, s = 1 -> 84.715790) and
# the pre-flight was clean (max|diag(ref)-diag(src)| = 0.000000e+00), so the
# blend was between the two objects we thought. But lam_min goes
# -1.27e-07 -> -0.175 AT s = 0.05 -- six orders in one step, no plateau. So
# s_max < 0.05 and there is no useful partial split next to the cross term.
# ⚠ sigma_max(K) ALONE MISLEADS HERE: it grows smoothly to 1.12 at s = 0.2
# because its whitening truncates rank, while rank33 climbs 1916 -> 1990.
# lam_min is the honest column on this table.
#
# echo "############ BLEND SWEEP  C33(s) = (1-s)*unsplit + s*split ############"
# python build_group_cross.py \
#     --run-dir $P17RUN \
#     --source-endf $P17SRC \
#     --cache $P17OUT/.parse_cache \
#     --mag-grid fine \
#     --null-fill zero \
#     --c33-from-file \
#     --c33-blend-ref $P17CTRL \
#     --c33-blend-s 0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
#     --check || exit 1

# --- THE PASS-1 SCALE SWEEP (roadmap Sec. 10) ------------------------------
# THE QUESTION. The blend asked "how much of the split survives next to the
# cross term as it is". This asks the other half: the cross block ships as
#     cx_post = cx_mc * outer(j33, j34),   j33 = sigma_declared / sigma_Pass1
# i.e. scaled UP to the declared MF33 diagonal -- and that inflation is exactly
# what the split-combine removes. `cx_mc` itself is honest (same np.cov call,
# same Pass-1 replicas). So the split and the cross term may not be two correct
# incompatible objects; they may be one corrected and one NOT corrected.
#
#     t = 1  -> what ships today            t = 0  -> the Pass-1 scale
#
# ANCHORS, and if either misses the table is void:
#     t = 1, split MF33    sigma_max(K) = 84.715790   (F', job 8488793)
#     t = 1, unsplit MF33  sigma_max(K) =  1.020488   (F', job 8488801)
# --c33-blend-s 0,1 re-runs the blend endpoints too, for two more free controls.
#
# CRITERION, FIXED BEFORE MEASURING, read at t = 0 vs the SPLIT MF33:
#     sigma_max(K) <= 1.05 and lam_min >= -1e-6*scale  -> COMPATIBLE. Write the
#         tape with cx(0) (steps 2-5 below, with --cx-j33-exp 0) and score ONCE.
#     1.05 < sigma_max(K) <= 2                         -> right mechanism, not
#         sufficient alone; roadmap Sec. 10.5.
#     sigma_max(K) > 2                                 -> inflation was not the
#         dominant cause; roadmap Sec. 10.6.
#
# WRITES NO TAPE AND SCORES NO CHI2. It is a PSD table. ~14 diagnose() calls,
# same order as job 8488838, so budget the same walltime.
# ==> IF IT IS OOM-KILLED, uncomment `##SBATCH --mem=300G` in the header. <==
# --- DONE (job 8488882, 2026-08-14). ROUTE 1 FAILS, AND THE PREMISE IS
# --- WITHDRAWN. Do NOT re-enable. ------------------------------------------
# All four anchors landed, so the table was readable. Criterion was
# sigma_max(0) <= 1.05; measured 7.613221. FAIL.
#
#     t      split            unsplit (CONTROL)
#   1.00     84.7158          1.0205
#   0.75     34.5237         96.3945
#   0.50     18.9415         99.7051
#   0.25     11.8065         80.0226
#   0.10      9.0466         67.1667
#   0.00      7.6132         59.2512
#
# ⚑ THE CONTROL IS THE FINDING. De-scaling blows up a pair that WORKED
# (1.0205 -> 96.39 in one step), and at t < 1 the unsplit is WORSE than the
# split -- physically absurd, so t < 1 measures a broken congruence, not the
# split. Only t = 1 is interpretable.
# WHY: j33 is not an inflation applied to the cross block, it is HALF OF A
# CONGRUENCE. sigma_max(K) is invariant under one, which the first table has
# been printing all along: CONTROL (unscaled) and CANDIDATE (J*joint*J) both
# read 0.999993.
# ⚠ AND j33's median 13.77 IS NOT AN INFLATION FACTOR: c0 is read absolute and
# c33_ship is relative, so j33 carries 1/c0_nom. Physical = 13.77 * 0.192845
# = 2.66. Never quote 13.8x.
#
# echo "############ PASS-1 SCALE SWEEP  cx(t) = cx_mc * outer(j33**t, j34) ###"
# python build_group_cross.py \
#     --run-dir $P17RUN \
#     --source-endf $P17SRC \
#     --cache $P17OUT/.parse_cache \
#     --mag-grid fine \
#     --null-fill zero \
#     --c33-from-file \
#     --c33-blend-ref $P17CTRL \
#     --c33-blend-s 0,1 \
#     --cx-j33-exp 1.0,0.75,0.5,0.25,0.1,0.0 \
#     --check || exit 1

# --- SPLIT FORENSICS: physical, or representation? (roadmap Sec. 10.6) ------
# ⛔ THIS IS THE LAST PROBE ON THIS LINE. Two hypotheses died here on 08-14.
# If it does not come back clean, ship A or B, report 84.72 as it stands, and
# do NOT chain a fourth.
#
# THE QUESTION IS BINARY. The sweep tested the FILE's representation. The
# algebra lives on the MC's OWN grid:
#     [[C33_split, cx_mc],[cx_mc^T, c34_mc]]
#       = [[c33_mc, cx_mc],[cx_mc^T, c34_mc]]   <- a SAMPLE covariance, PSD
#       + [[D_exc R_loc D_exc, 0],[0, 0]]       <- PSD
# PSD by construction IF C1' = c33_mc. If the MC-grid joint passes, the
# incompatibility is REPRESENTATIONAL (2317 file bins, the m_of injection,
# c34_ship for c34_mc) and there may be a route. If it fails, it is physical.
#
# THE SPLIT PRODUCT IS p12_split, NOT _splitdiag: `mf33_absolute_covariance.npy`
# is what built the p13_split tape; `..._splitdiag.npy` is the white-excess
# BOUND and is not what ships. Do not point --split-forensics at it.
#
# CRITERIA, FIXED BEFORE MEASURING:
#   UNIT GATE   max|ratio - 1| ~ 0                -> else the table is VOID
#   (1) min eig(C_split - c33_mc)/max|C| >= -1e-10 and #(j33*c0_nom < 1) = 0
#                                                 -> premise holds, read (2)
#       otherwise                                 -> LINE CLOSED
#   (2) SPLIT sigma_max(K) <= 1.001, ANCHOR on 0.999993
#                                                 -> REPRESENTATIONAL, read (3)
#       otherwise                                 -> PHYSICAL. LINE CLOSED.
#   (3) >= 80 % of the load in the directions only the split spans
#                                                 -> the repair is a PROJECTION
#                                                    with a physical basis
#       load spread                               -> LINE CLOSED
#
# --c33-blend-s 0,1 is kept: two free anchors (1.020488 and 84.715790).
# WRITES NO TAPE AND SCORES NO CHI2.

# ===========================================================================
# CURRENT JOB (2026-08-14): SCORE THE ANALYTIC (R2) EVALUATION, ONCE.
#
# docs/cross_term_two_pass_investigation.md Sec. 10-11. MF33, MF34 and the a_0
# cross blocks all come out of ONE object: the closed-form covariance of the
# estimator that made the central values, C = A A^T + sum_b B_b(model), with A
# read out of the pipeline's own solvers by identity probing. PSD by
# construction, so the cross term needs no rescaling and no repair.
#
# ALREADY DONE, ON THE WORKSTATION -- do not repeat here:
#   * nominal-only rerun, reproduces run 92's nominal_fits to 1e-11
#   * analytic joint; sigma(c0) matches the MC bin by bin (median ratio 1.000,
#     89.8 % within +-10 %, spearman 0.987)
#   * rebuild_mf33 on the analytic MF33   -> chi2/r2_analytic_tape/*_mg.endf
#   * cross + MF34 written                -> chi2/r2_analytic_tape/*_a0cross_mg.endf
#     gate at write time: sigma_max(K) = 0.99949, lam_min = -6.5e-19
#     (shipped: 1.0205 / -1.3e-07;  MF33-split + cross: 84.72 / -4.38)
#
# WHAT IS LEFT IS THE SCORE, AND IT IS FOR THE RECORD, NOT FOR CHOOSING.
# Sec. 10.8-6 applies: the object was built and gated on structural criteria
# (marginal reproduction, PSD, Cauchy-Schwarz), never on V4.
#
# READ IN THIS ORDER, and the first two are vetoes:
#   1. JEFF and JENDL must move 0.00 % against predictive_91_cross. They share
#      the corpus and nothing about them changed.
#   2. V2 must come back UNCHANGED (the analytic MF33 diagonal is 0.9994 of the
#      shipped one, median). V2 reads the diagonal only, so a V2 move means the
#      marginals moved and the object is not what this comment says it is.
#   3. THEN V4, on `no_Cierjacks` (91_cross base: 8.4437) and n_absorbed.
#      ⛔ n_absorbed must not FALL outside Kinney -- Sec. 6e, unchanged.
#
# ~1 h + an 11 GB sidecar for the precompute, ~30 min for the analysis.
# ⚑ DONE 2026-08-14 (the block above, now commented out). Both vetoes passed —
#   JEFF/JENDL moved 0.00000 % and V2 is identical to four decimals in all six
#   subsets — and V4 on `no_Cierjacks` went 8.4437 -> 6.4329 (-23.8 %), i.e.
#   below JEFF's 7.5194 and JENDL's 9.2743. Report in
#   CHI_Figures/chi2_predictive/run_r2analytic/.

# ===========================================================================
# CURRENT JOB (2026-08-14): Sec. 6e FOR THE ANALYTIC EVALUATION.
#
# The disqualifying criterion, unchanged: improving Kinney by SPENDING
# directions the rest of the corpus had is redistribution dressed as
# information. It is read AFTER the chi2, and it can still kill the object.
#
# BASE, already on disk from run 91_cross:
#   chi2/n_absorbed_91_cross.csv            (whole corpus bar Cierjacks)
#   chi2/n_absorbed_91_cross_kinney.csv     (Kinney: This_work n_absorbed 1447.216815)
#   chi2/n_absorbed_91_cross_cierjacks.csv
#
# READ: Kinney's n_absorbed may rise, but ⛔ the aggregate over the OTHER
# experiments must NOT fall. Compare This_work rows only; JEFF and JENDL are
# there as the invariance control and must reproduce the base exactly.
#
# Step 1 is the corpus (blocks <= 16000, so Cierjacks is reported as SKIPPED,
# never silently dropped). Step 2 is Kinney alone, matching the base's file
# names. Step 3 is Cierjacks, 28631^2 = 6.6 GB in float64 — if it is
# OOM-killed, uncomment the #SBATCH --mem=300G line at the top of this file
# and resubmit ONLY step 3.
# --- STEP 2: PRICE THE DEGENERATE MF34 SLOTS -------------------------------
# ⚑ THE QUESTION. `eval_covariance.build_mf34_block` scales the file's RELATIVE
# MF34 by `a_l_per_pt` -- MF4's a_l interpolated AT EACH DATA POINT -- not by
# the group mean the file's denominator actually is. That is a CORRECT reading
# of ENDF. But on 201 live slots our group mean cancels (sigma_rel up to 6337 %,
# max |a_l| inside the group ~200x the mean), and those 201 slots carry
# **44.6 %** of MF34's absolute variance. So the chi2 may be reconstructing an
# absolute sigma far larger than the one we built -- which would INFLATE
# Sigma_eval for This_work alone and FLATTER the headline V4 = 6.4329.
#
# Same mechanism §10.6-1 already priced for the NULL slots on run 86 (2.14 % of
# the Sigma_eval diagonal, 82.9 % of points). Nobody has priced it for the
# live-but-degenerate ones, and this tape declares up to 4016 there.
#
# ⛔ THIS IS AN EXPOSURE MEASUREMENT, NOT A CANDIDATE DELIVERABLE. Removing
# parameters without measured structure in exchange is what disqualified the
# MF34 split (§6e). If V4 RISES here, the headline was resting on the artefact
# and the tape needs the repair; if it barely moves, the headline is safe and
# the degenerate slots are a publication-readiness issue only.
#
# ~1.5 h + an 11 GB sidecar. ⚠ /share_snc sits near 91 % -- check `df` first and
# delete the sidecar after scoring; the parquet is the durable artefact.
# Masks built by myworkspace/chi2/audit_a7_degenerate_mask.py.
#
# STEP 2a -- DONE, job 8497757 (2026-08-17). The precompute SUCCEEDED: mask
# applied (201 slots, a_1 29 / a_2 0 / a_3 35 / a_4 49 / a_5 62 / a_6 26),
# parquet (140457, 28) and the 10.9 GB sidecar are on disk. ⛔ DO NOT RE-RUN IT:
# an hour and 11 GB to reproduce a file that exists.
#
# KIKA_MF34_NULL_MASK=/share_snc/snc/JuanMonleon/chi2/mf34_p1degenerate_mask_r2analytic.npz \
# KIKA_THIS_WORK_DIR=$R2TAPEDIR \
# KIKA_THIS_WORK_ENDF=26-Fe-56g_nominal_a0cross_mg.endf \
# KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
# KIKA_RUN_TAG=r2analytic_nodeg \
#     python precompute_chi2_predictive.py || exit 1

# STEP 2b -- what job 8497757 died on, and it was avoidable: the methodology
# registry in chi2_analysis_cluster.py is a fixed dict and
# `predictive_r2analytic_nodeg` was not in it. Runs 85 and 89 died the same way
# (§10.1.8-L1) and the file even says so above the 86_nonull entry. NOW
# REGISTERED. The .eval_cov.npz sidecar is inferred from the parquet path, so
# this reuses the one already written -- minutes, not an hour.
# STEP 2b -- DONE, job 8498153. BOTH VETOES EXACT (libraries 0.000000 %, V2
# identical in all six subsets) and V4 no_Cierjacks 6.4329 -> 6.6044 (+2.67 %),
# still below JEFF 7.5194 and JENDL 9.2743.
#
# KIKA_CHI2_METHODOLOGIES=predictive_r2analytic_nodeg \
# KIKA_CHI2_RUN_ID=r2analytic_nodeg \
#     python chi2_analysis_cluster.py || exit 1

# ===========================================================================

# --- previous job (2026-08-14): n_absorbed for the analytic tape, Sec. 6e ----
# Produced n_absorbed_r2analytic{,_kinney,_cierjacks}.csv. Passed: +10.8 %
# outside Kinney. Step 3 is Cierjacks, 28631^2 = 6.6 GB in float64 -- if it is
# OOM-killed, uncomment #SBATCH --mem=300G and resubmit ONLY step 3.
#
# R2TAG=r2analytic
# python corpus_absorbed.py $R2TAG || exit 1
# python corpus_absorbed.py $R2TAG --only 10571002 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_${R2TAG}_kinney.csv || exit 1
# python corpus_absorbed.py $R2TAG --only 20743002 --max-n 30000 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_${R2TAG}_cierjacks.csv || exit 1

# R2DIR=/share_snc/snc/JuanMonleon/chi2/r2_analytic_tape
# R2TAG=r2analytic
#
# KIKA_THIS_WORK_DIR=$R2DIR \
# KIKA_THIS_WORK_ENDF=26-Fe-56g_nominal_a0cross_mg.endf \
# KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
# KIKA_RUN_TAG=$R2TAG \
#     python precompute_chi2_predictive.py || exit 1
#
# KIKA_CHI2_METHODOLOGIES=predictive_r2analytic KIKA_CHI2_RUN_ID=r2analytic \
#     python chi2_analysis_cluster.py || exit 1
# ===========================================================================

# echo "############ SPLIT FORENSICS: physical, or representation? ############"
# python build_group_cross.py \
#     --run-dir $P17RUN \
#     --source-endf $P17SRC \
#     --cache $P17OUT/.parse_cache \
#     --mag-grid fine \
#     --null-fill zero \
#     --c33-from-file \
#     --c33-blend-ref $P17CTRL \
#     --c33-blend-s 0,1 \
#     --split-forensics /share_snc/snc/JuanMonleon/chi2/p12_split/mf33_absolute_covariance.npy \
#     --check || exit 1

# --- STEPS 2-5: UNCOMMENT ONLY AFTER READING THE GATE ABOVE -----------------
#
# STEP 2 — write the tape. MF3/MF4/MF33 are copied from --source-endf
# untouched, so the split MF33 goes through verbatim; MF34's shape blocks and
# the (L=0, L1) a_0 cross blocks are written from c34_post / cx_post.
#
# python build_group_cross.py \
#     --run-dir $P17RUN \
#     --source-endf $P17SRC \
#     --cache $P17OUT/.parse_cache \
#     --mag-grid fine \
#     --null-fill zero \
#     --write-endf $P17ENDF || exit 1
#
# STEP 3 — precompute (~1 h, 11 GB sidecar). The cross comes FROM THE FILE:
# KIKA_MF33_MF34_CROSS_DIR is the retired sidecar route (§L13) and must stay
# unset, or the term is folded twice.
#
# KIKA_THIS_WORK_DIR=$P17OUT \
# KIKA_THIS_WORK_ENDF=26-Fe-56g_nominal_a0cross_mg.endf \
# KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
# KIKA_RUN_TAG=$P17TAG \
#     python precompute_chi2_predictive.py || exit 1
#
# STEP 4 — the scoring (~30 min). BOTH variables, always (job 8480897).
# `predictive_92split_cross` is REGISTERED in chi2_analysis_cluster.py
# (deployed 2026-08-14, backup in _backup_pre_predictive95/).
#
# KIKA_CHI2_METHODOLOGIES=predictive_92split_cross KIKA_CHI2_RUN_ID=$P17TAG \
#     python chi2_analysis_cluster.py || exit 1
#
# STEP 5 — §6e, disqualifying. Same arguments as every other candidate so the
# CSVs subtract row by row. Both bases already have their CSVs on disk
# (n_absorbed_92split.csv and n_absorbed_91_cross.csv), so nothing else is
# needed for the comparison.
#
# python corpus_absorbed.py $P17TAG --probes 128 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_${P17TAG}.csv || exit 1
# python corpus_absorbed.py $P17TAG --only 10571002 --max-n 30000 --probes 256 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_${P17TAG}_kinney.csv || exit 1

# --- DONE (job 8488732, 2026-08-14). ⛔ DISQUALIFIED. Do NOT re-enable. -----
# The veto passed exactly (JEFF/JENDL 0.00000 % on all 24 subset x variant
# comparisons; max |delta n_absorbed| = 0.000e+00 on 62 experiments each), so
# the reading is sound. §6e then tripped:
#     Kinney        n_absorbed 1447.22 -> 1596.39   +10.31 %
#     THE OTHER 61  n_absorbed 1563.55 -> 1535.82    -1.77 %   <-- FALLS
# and the chi2 agrees: no_Cierjacks V4/N 8.4437 -> 8.5445 (+1.19 %), every
# subset worse except only_KS (-0.69 %). V2 moved +0.00 % everywhere, which is
# the free control that the declared sigma really did not change.
# Artefacts kept: chi2_data_predictive_95.parquet, run_095/, the four
# n_absorbed CSVs. The 11 GB sidecar can go — the candidate is rejected.
#
# R95=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_95_mf34split
# R95ENDF=26-Fe-56g_nominal_a0cross_mg.endf
#
# ls -la $R95/$R95ENDF || exit 1
#
# # --- STEP 1: Sigma_eval, cross term read from the a_0 blocks (~1 h, 11 GB) --
# KIKA_THIS_WORK_DIR=$R95 \
# KIKA_THIS_WORK_ENDF=$R95ENDF \
# KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
# KIKA_RUN_TAG=95 \
#     python precompute_chi2_predictive.py || exit 1
#
# # --- STEP 2: the scoring (~30 min) ------------------------------------------
# # ⚠ BOTH variables, always. Setting KIKA_CHI2_RUN_ID without
# # KIKA_CHI2_METHODOLOGIES is what killed job 8480897: the script fell through to
# # its default `predictive_82` and went looking for a deleted 11 GB sidecar.
# # `predictive_95` is REGISTERED in chi2_analysis_cluster.py (deployed
# # 2026-08-14, backup in _backup_pre_predictive95/).
# KIKA_CHI2_METHODOLOGIES=predictive_95 KIKA_CHI2_RUN_ID=095 \
#     python chi2_analysis_cluster.py || exit 1
#
# # --- STEP 3: §6e for the candidate, and it is DISQUALIFYING -----------------
# # Same arguments as every other candidate so the CSVs subtract row by row
# # (--max-n 16000 skips Cierjacks in both; seed 7 lives inside corpus_absorbed).
# python corpus_absorbed.py 95 --probes 128 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_95.csv || exit 1
# python corpus_absorbed.py 95 --only 10571002 --max-n 30000 --probes 256 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_95_kinney.csv || exit 1
#
# # --- STEP 4: the §6e BASELINE, which does not exist yet. READ-ONLY. ---------
# # `n_absorbed_91_cross.csv` has never been measured — only the Cierjacks-only
# # slice (`n_absorbed_91_cross_cierjacks.csv`) exists, and 91_rewrite is NOT the
# # control for a cross-carrying candidate. This reads 91_cross's existing 11 GB
# # sidecar, scores nothing and writes no sidecar, so disk does not move.
# python corpus_absorbed.py 91_cross --probes 128 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_91_cross.csv || exit 1
# python corpus_absorbed.py 91_cross --only 10571002 --max-n 30000 --probes 256 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_91_cross_kinney.csv || exit 1

# --- HOW TO READ IT: two numbers, and the second one is the veto -----------
# Compare n_absorbed_91_rewrite.csv (this job) against n_absorbed_92split.csv,
# joining on (library, experiment_id). Only the This_work rows move; JEFF and
# JENDL are read from the same tapes in both and are a free consistency check
# that must come out at 0.
#
#  1. Kinney (10571002), This_work ....... 92split gives 1792.2 of 13208.
#     If the baseline is LOWER, we bought Kinney directions.
#  2. THE VETO -- the 65 experiments that are NOT Kinney, This_work, summed.
#     92split gives 3465.0 - 1792.2 = 1672.8 over N = 18188 - 13208 = 4980.
#     If that sum FALLS relative to the baseline, the candidate financed
#     Kinney by taking declared directions away from everyone else, which is
#     exactly what this work accuses JENDL of. It is then DROPPED, whatever
#     the chi2 says.
#
# ⚠ The split-combine MOVES variance between coherent and energy-local without
#   changing the trace, so a first guess is that n_absorbed goes UP outside
#   Kinney too (finer structure = more directions). That is a guess written
#   before the measurement, not a requirement: the veto only forbids a FALL.
#
# ⚠ Cierjacks (20743002, N=28631) is skipped by --max-n 16000 in BOTH CSVs.
#   It is not in the veto sum. That is deliberate: the headline subset is
#   no_Cierjacks anyway. Do not raise --max-n on only one of the two.

# --- PREVIOUS JOB (8470703): Cierjacks' n_absorbed block. DONE --------------
# Returned clean, 112-116 s per library, results in
# n_absorbed_91_cross_cierjacks.csv and folded into §10.8-19's corpus table.
# python corpus_absorbed.py 91_cross --only 20743002 --max-n 30000 --probes 128 \
#     --out /share_snc/snc/JuanMonleon/chi2/n_absorbed_91_cross_cierjacks.csv || exit 1

# --- PREVIOUS JOB (8470630 part B): the cross-scale dose-response. DONE -----
# §10.8-19 Step 0: CONVEX on all six subsets, 5.6-19.1 % below linear, so the
# chain does not amplify and §10.8-1's certificate passes. no_Cierjacks V4 =
# 6.3075 / 6.3418 / 6.4637 / 8.4437 at s = 0 / .25 / .50 / 1. Do not re-run.
# ⚠ Both s-sidecars (~11 GB each) are still on disk and their parquets are read;
# they can be deleted whenever the space is wanted.
#
# KIKA_THIS_WORK_DIR=$A0DIR \
# KIKA_THIS_WORK_ENDF=$A0ENDF \
# KIKA_MF34_NULL_MASK=$NULL_MASK \
# KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
# KIKA_MF33_MF34_CROSS_SCALE=0.50 \
# KIKA_RUN_TAG=91_cross_s50 \
#     python precompute_chi2_predictive.py || exit 1
#
# KIKA_CHI2_METHODOLOGIES=predictive_91_cross_s50 KIKA_CHI2_RUN_ID=091_cross_s50 \
#     python chi2_analysis_cluster.py || exit 1
#
# KIKA_THIS_WORK_DIR=$A0DIR \
# KIKA_THIS_WORK_ENDF=$A0ENDF \
# KIKA_MF34_NULL_MASK=$NULL_MASK \
# KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
# KIKA_MF33_MF34_CROSS_SCALE=0.25 \
# KIKA_RUN_TAG=91_cross_s25 \
#     python precompute_chi2_predictive.py || exit 1
#
# KIKA_CHI2_METHODOLOGIES=predictive_91_cross_s25 KIKA_CHI2_RUN_ID=091_cross_s25 \
#     python chi2_analysis_cluster.py || exit 1

# --- PREVIOUS JOB (8470367): run 91, the a_0 cross term. DONE and read ------
# §10.8-1/2. Kept so it can be flipped back; do not re-run.
# mkdir -p $A0DIR

# python build_group_cross.py \
#     --run-dir $XDIR \
#     --source-endf $DIR86/$SHIPPED_ENDF \
#     --cache $A0DIR/.group_cross_cache \
#     --mag-grid fine \
#     --null-fill zero \
#     --write-endf $A0DIR/$A0ENDF || exit 1

# ls -la $A0DIR/$A0ENDF

# --- STEP 2: the marginal rewrite alone (cross term NOT read) ----------------
# KIKA_THIS_WORK_DIR=$A0DIR \
# KIKA_THIS_WORK_ENDF=$A0ENDF \
# KIKA_MF34_NULL_MASK=$NULL_MASK \
# KIKA_RUN_TAG=91_rewrite \
#     python precompute_chi2_predictive.py || exit 1

# KIKA_CHI2_METHODOLOGIES=predictive_91_rewrite KIKA_CHI2_RUN_ID=091_rewrite \
#     python chi2_analysis_cluster.py || exit 1

# --- STEP 3: the same file, cross term READ from the a_0 blocks --------------
# KIKA_THIS_WORK_DIR=$A0DIR \
# KIKA_THIS_WORK_ENDF=$A0ENDF \
# KIKA_MF34_NULL_MASK=$NULL_MASK \
# KIKA_MF33_MF34_CROSS_FROM_FILE=1 \
# KIKA_RUN_TAG=91_cross \
#     python precompute_chi2_predictive.py || exit 1

# KIKA_CHI2_METHODOLOGIES=predictive_91_cross KIKA_CHI2_RUN_ID=091_cross \
#     python chi2_analysis_cluster.py || exit 1

# --- PREVIOUS JOB (8456466): 86_fine, DONE and read (§10.7-9) ----------------
# Kept for the record; do not re-run.
# FINE_ENDF=26-Fe-56g_nominal.endf
#
# KIKA_THIS_WORK_DIR=$DIR86 \
# KIKA_THIS_WORK_ENDF=$FINE_ENDF \
# KIKA_RUN_TAG=86_fine \
#     python precompute_chi2_predictive.py || exit 1
#
# KIKA_CHI2_METHODOLOGIES=predictive_86_fine KIKA_CHI2_RUN_ID=086_fine \
#     python chi2_analysis_cluster.py || exit 1

# --- PREVIOUS JOB (8456309): A then B, DONE and read -------------------------
# Kept for the record; do not re-run.
# --- A: the mask alone -------------------------------------------------------
# KIKA_THIS_WORK_DIR=$DIR86 \
# KIKA_THIS_WORK_ENDF=$SHIPPED_ENDF \
# KIKA_RUN_TAG=86_maskfix \
#     python precompute_chi2_predictive.py || exit 1

# KIKA_CHI2_METHODOLOGIES=predictive_86_maskfix KIKA_CHI2_RUN_ID=086_maskfix \
#     python chi2_analysis_cluster.py || exit 1

# --- B: the MF33 regrouping on top ------------------------------------------
# KIKA_THIS_WORK_DIR=$DIR86 \
# KIKA_THIS_WORK_ENDF=$GROUPED_ENDF \
# KIKA_RUN_TAG=86_mf33grouped \
#     python precompute_chi2_predictive.py || exit 1

# KIKA_CHI2_METHODOLOGIES=predictive_86_mf33grouped KIKA_CHI2_RUN_ID=086_mf33grouped \
#     python chi2_analysis_cluster.py || exit 1

# --- PREVIOUS JOB (8456264): the L18 re-certification, DONE and read ---------
# A and A' went -0.062091 -> -0.000059 (1.28e-06 relative, the offline number to
# the digit); every row carrying Cx was invariant to 5e-06; every MF33 row was
# byte-identical. Kept below for the record; do not re-run it.
#
# --- 2026-08-07, THIRD PASS: SAME COMMAND, NEW REASON (roadmap §10.1.8-L18) --
#
# The invocation below is UNCHANGED. What changed is the code it runs:
# `shipped_c34_rel_on_base` no longer PINS out-of-range base centres to a
# block's first or last interval, it zeroes them (_backup_pre_L18clip).
#
# WHY THAT MATTERS HERE. The blocks come back on four grids, because
# `merge_mf34` builds each (L, L1) pair on that pair's own union of JEFF's grid
# and the pipeline overlay's. Block (2,6) starts at 0.8468 MeV and leaves 9 of
# the 703 base groups uncovered, so pinning credited the 0.1-0.7 MeV region --
# where JEFF publishes (2,2) but no (2,6) -- with (2,6)'s FIRST interval: 5120
# fabricated cross-seam cells at up to 2.07e-02. Measured offline against the
# `mf34_groups__*.npz` cache: -6.2091e-02 pinned, -5.9359e-05 masked, i.e. the
# ENDF round-trip floor. §L17's "the shipped MF34 is not PSD" was that one line.
#
# ⚑ SO EVERY PSD ROW THIS SCRIPT HAS EVER PRINTED WAS TAKEN AGAINST A PINNED
# MF34 -- §§J, K, L3, L12, L14.1 and L17 alike. This run re-takes all of them
# masked. It is a pure single-variable change: nothing else moved.
#
# HOW TO READ IT:
#   Row A (Cx = 0) -> should now sit at the round-trip floor, ~-5.9e-05, not
#     -0.0621. That is the marginal, and it is the number §L17.4's theorem
#     `lam_min(J) <= lam_min(C34)` reads. If it does, the theorem stops binding
#     and §10.6-3 is un-capped.
#   Rows B/C and B'/C' (Cx present) -> the real question. The cross block has
#     never been PSD next to these marginals, and NOTHING here predicts that it
#     now is. If they are still ~-26 the units mismatch of §L13 is untouched
#     and the a_0 route is still the plan, just no longer blocked by a theorem.
#   lam_min across --tol-sweep -> must still be flat. It is the control.
#
# ⚠ DO NOT read `sigma_max(K)`. Retired as a decision metric in §L17.1: it moves
# 4.8x and 236x over the same sweep on which lam_min is flat to nine figures.
#
# --source-endf stays explicit: 26-Fe-56g_nominal_consistent_mg.endf is still
# sitting in $XDIR from run 90 and sorts FIRST, so auto-discovery would read the
# grids and c34_ship off run 90's own output and self-compare.
# DONE — job 8456264, read 2026-08-07. Left here so it can be flipped back.
# python build_group_cross.py --run-dir "$XDIR" \
#     --source-endf "$XDIR/$SHIPPED_ENDF" \
#     --c33-from-file \
#     --tol-sweep \
#     --check \
#     || exit 1

# --- 2026-08-07, SECOND PASS: job 8454785's headline was CONFOUNDED ---------
#
# 8454785 ran and reported lam_min -0.4249 against -26.308, which was read as
# "the MF33 swap moves it by 62x". IT DOES NOT FOLLOW. `rows3` built the cross
# block from the RAW `cx_post`, whose shape axis is ABSOLUTE, and folded it
# against `rel_ship_K`, which is RELATIVE -- while `rows2` divides by `a_pt`
# first and its own comment calls that mandatory. So B' moved the MF33 grid AND
# the cross leg's units together. `a_pt` is small and crosses zero, so the
# missing division is most likely the whole effect: it is Sec. 10.1.8-L13's own
# mechanism. Fixed: `cx_rel_full = cx_post / a_pt`.
#
# --tol-sweep is the second half. `sigma_max(K)` went 1560 -> 18837 in the same
# table, the OPPOSITE direction to lam_min, and the file's MF33 explains it
# without physics: 237 negative eigenvalues at -1.27e-07 and a smallest
# RETAINED eigenvalue of 6.37e-10, i.e. 200x below its own noise floor, which
# is what ENDF-6's 6-significant-digit ASCII does to a rank-deficient matrix.
# `whiten` inverts that. The sweep prints the spectra and re-diagnoses B and B'
# from 1e-10 to 1e-4.
#
# HOW TO READ IT:
#   lam_min MUST be flat across the sweep -- it does not depend on the
#     tolerance. If it moves, something other than the tolerance is moving and
#     the run is not trustworthy.
#   B' vs B on lam_min, NOW SINGLE-VARIABLE -> the real cost of the MF33 swap,
#     and whether Sec. 10.1.8-L14.1's invariance survives.
#   sigma_max(K) -> read it at the tolerance where the 188-group and file rows
#     stop disagreeing, not at NULL_TOL.
# ---------------------------------------------------------------------------

# --- 2026-08-07: STEP 1 IS NOW THE WHOLE JOB --------------------------------
#
# Steps 2-3 ALREADY RAN (job 8452834) and are scored: run_086_nonull. This work
# +3.34 % on `all`, +4.36 % on `no_Cierjacks`, +8.74 % on `no_KS_no_Cierjacks`;
# JEFF and JENDL identical to 0.0 and This_work's V2 identical too, so the
# single-variable control held both ways. DO NOT re-run them -- it costs the
# long precompute and overwrites an 11 GB sidecar for a result we have.
#
# What did NOT run in 8452834 is step 1: it died on
#   ModuleNotFoundError: No module named 'scripts'   (build_group_cross.py:873)
# because that file lacked the sys.path insertion null_slot_exposure.py:58-60
# has. Fixed and deployed 2026-08-07 (_backup_pre_c33_import_fix).
#
# `|| exit 1` NOW, not `|| echo`. The comment above argued for `|| echo`
# because a long re-score hung off this step; nothing hangs off it any more, so
# a silent failure would just produce an empty job -- and `|| echo` on a
# non-zero exit is exactly what misdiagnosed run 89 (§10.1.8-L1).
#
# KIKA_THIS_WORK_DIR=$DIR86 \
# KIKA_MF34_NULL_MASK=$NULL_MASK \
# KIKA_RUN_TAG=86_nonull \
#     python precompute_chi2_predictive.py || exit 1
#
# KIKA_CHI2_METHODOLOGIES=predictive_86_nonull KIKA_CHI2_RUN_ID=086_nonull \
#     python chi2_analysis_cluster.py || exit 1
# ---------------------------------------------------------------------------

# --- WHAT TO DO WITH THE ANSWER --------------------------------------------
# Compare run_086_nonull/summary.json against run_086/summary.json. V2 and V4
# are the columns that matter.
#
# The chi2 gets BIGGER when the bias is removed (less evaluated variance), so
# expect This_work to look slightly worse and the JEFF/JENDL gap to narrow.
#   moves less than ~1 %  -> note it in the write-up and move on; the published
#     comparisons stand as they are.
#   moves ~1-5 %          -> it is the same order as §8.9's +0.95 %/+5.85 % and
#     §8.10's "1-2 % of V4", so every quoted chi2 delta has to be re-checked
#     against this baseline before publication.
#   moves more            -> stop and re-plan; the parameter cap (§10.6-2)
#     becomes the next pipeline run rather than a tidy-up.
#
# Either way the FIX is upstream (§10.6-2): cap the published order per energy
# region by frozen_degree so these parameters are never written. The mask is a
# measurement device, not a shipping decision -- it changes what the chi2 folds
# without changing the file, and the file is what we deliver.

# --- PREVIOUS JOB: the two Sec. 10.6 measurements (job 8451762) -----------
# ---------------------------------------------------------------------------
# CURRENT JOB: THE TWO MEASUREMENTS THAT GO BEFORE THE PLAN (roadmap §10.6-0
# and §10.6-1). No chi2, no 11 GB sidecar, no ENDF written. ~20 min total.
# Either one can invalidate what follows, which is why they are first.
#
# WHERE THE TRACK STANDS. Run 90 (job 8451628) reproduced run 89 -- worst
# normalised lam_min -37.94 against -38.15, 694 negative Sigma_eval diagonals
# against 683, 22.8 % of points bit-identical. Job 8451718 then diagnosed it in
# the space the chi2 actually folds, and §10.1.8-L13 derived the mechanism from
# `eval_covariance.py`: Sigma_eval is a congruence M J M^T only if the cross
# term's legs use the same maps as the self blocks, and they do not. MF34 is
# folded RELATIVE (its leg carries a_l(E_j)) while Cx is folded ABSOLUTE (its
# leg does not), so the chi2 folds Cx against r^2*Var34 where it was certified
# against Var34, r = a_l(E_j)/a_nom(g). The certification saturates
# Cauchy-Schwarz (sigma_max(K) = 0.99994, a singular sample covariance), so
# every point with |r| < 1 violates PSD. No cross block can fix that.
#
# THE PLAN IS NOW: put the cross term in MF34's a_0 blocks and read it FROM THE
# FILE, so both legs carry the same factor (§10.6). Juan's constraint: what the
# chi2 scores is the ENDF file we write, not a sidecar. But these two
# measurements come first.
#
# --- STEP 1 (§10.6-0): print the normaliser, and re-read the L12 table --------
#
# The D/E rows of the second table were read off `lam_min_norm`, which divides
# by max|diag(J)| -- and under --null-fill ship that maximum IS a null-slot
# relative variance, up to 7.6, while under zero it is not. -3.462 -> -26.45 is
# a factor 7.64 against a removed diagonal of 7.6, and the scale-free column
# moves 1 % (sigma_max 1544.7 -> 1561.5). `diagnose` now reports `lam_min` and
# `scale` as their own columns.
#
# WHAT THIS SETTLES: whether the "D and E both bad -> the cross block does not
# belong next to these marginals" branch ever fired. PRIOR, RECORDED BEFORE THE
# JOB LANDS: it did not. Expect scale(C) ~ 7.6 and scale(D) ~ 1, with lam_min
# itself nearly equal across the two rows.
#
# --- STEP 2 (§10.6-1): what do the unsupported parameters do to RUN 86? -------
#
# ⚠⚠ THIS IS THE ONE THAT CAN MOVE PUBLISHED NUMBERS, and it has nothing to do
# with the cross block. `row_aggregator` never populated 1542 of 4218 (group,
# order) slots, yet the shipped MF34 declares nonzero relative variance at ~95 %
# of them -- up to 7.6, sigma_rel ~ 276 %. §10.1.8-L11 established the chi2
# folds them at FULL weight. Run 86 folds the same file, and run 86 is the
# baseline every chi2 in the roadmap is quoted against. A bigger Sigma_eval
# gives a SMALLER chi2, so if this bites it has been flattering This_work
# against JEFF and JENDL.
#
# THE FILE IS THE RIGHT ONE. $XDIR is run 88's directory, but its _mg.endf is
# byte-identical to run 86's (205 838 091 bytes, §10.1.7-A), so this measures
# exactly what run 86 scored.
#
# ⚠ run 86's parquet predates `sigma_eval_var_diag` and carries only the CLIPPED
# `sigma_eval_diag`. The script says so and falls back to its square. Harmless
# here -- run 86 has no cross block, and the cross block is what manufactures
# negative diagonals -- but do not carry that fallback to a run that has one.
#
# WHAT IT DOES NOT SETTLE: a diagonal share is a SIZE, not a chi2. If it is
# material the escalation is a full re-score with the slots zeroed.
#
# Resources: 300G is far more than enough. Step 1 peaks ~5 GB; step 2 is one
# 205 MB ENDF parse plus ~47k interpolations.
# ---------------------------------------------------------------------------

#XDIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_88_fullcross
#SHIPPED_ENDF=26-Fe-56g_nominal_mg.endf
#NULL_MASK=/share_snc/snc/JuanMonleon/chi2/mf34_null_mask.npz

# --check writes NO covariance and NO ENDF. --write-null-mask is the one thing
# it does write, and it is a ~30 kB npz.
#
# --source-endf stays explicit: 26-Fe-56g_nominal_consistent_mg.endf is still
# sitting in $XDIR from run 90 and sorts FIRST, so auto-discovery would read the
# grids and c34_ship off run 90's own output and self-compare.
#python build_group_cross.py --run-dir "$XDIR" \
#    --source-endf "$XDIR/$SHIPPED_ENDF" \
#    --write-null-mask "$NULL_MASK" \
#    --check || exit 1

#python null_slot_exposure.py \
#    --parquet /share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_86.parquet \
#    --endf    "$XDIR/$SHIPPED_ENDF" \
#    --mask    "$NULL_MASK" \
#    --out-csv /share_snc/snc/JuanMonleon/CHI_Figures/null_slot_exposure_86.csv \
#    || exit 1

# --- WHAT TO DO WITH THE ANSWER --------------------------------------------
# Step 2 material (say, >few % of Sigma_eval, or concentrated in the experiments
# that drive the chi2)  -> STOP. Re-score run 86 with the unsupported slots
#   zeroed before anything else is published, and cap the published order per
#   energy region by frozen_degree in the pipeline (§10.6-2). That cap also
#   dissolves the --null-fill question rather than answering it: there is
#   nothing to fill if the parameter is not published.
# Step 2 negligible -> the baseline stands as published; go straight to §10.6-3,
#   the a_0 blocks. That work is in eval_covariance.py, not here: stop skipping
#   l < 1 in build_mf34_block, use y_eval as the a_0 sensitivity, RETIRE the
#   sidecar for This_work (or the term is counted twice), and make the cross
#   term's magnitude leg use the same W overlap average build_mf33_block uses.
#
# Either way `predictive_91` must be REGISTERED in chi2_analysis_cluster.py
# BEFORE any scoring run is launched -- that omission cost runs 85 and 89 their
# analyses.

# --- PREVIOUS JOB: the L12 diagnosis (job 8451718) ------------------------
# ---------------------------------------------------------------------------
# CURRENT JOB: DIAGNOSIS ONLY (roadmap §10.1.8-L11). No chi2, no 11 GB sidecar,
# no ENDF written. ~10 min. It answers ONE question: why run 90 failed.
#
# WHAT RUN 90 DID (job 8451628). Steps 1 and 2 succeeded, step 3 died on
# Cholesky (potrf info = 63) and there is no run_090/. The PSD numbers came out
# indistinguishable from run 89's:
#
#                            run 88     run 89     RUN 90
#   worst normalised lam_min  -1.767    -38.15     -37.94
#   negative Sigma_eval diag    1005       683        694
#   min diagonal            -0.01004  -0.007416  -0.007429
#
# and point by point, 10691 of 46827 (22.8 %) are BIT-IDENTICAL to run 89, with
# a median relative difference of 9.7e-05. Against run 88: 0 identical, median
# 0.32. So rebuilding MF34 moved almost nothing and did NOT fix the violation.
#
# THE PRIME SUSPECT, AND IT IS AN ERROR IN THE PREVIOUS ANALYSIS. Run 90 kept
# the SHIPPED relative values at the 1542 slots (37 %) where a_l is exactly
# zero, on the argument that it was free because "a consumer converting back
# multiplies by a_l = 0". That is true only for a consumer using OUR a_l. The
# chi2 does not: `eval_covariance._mf34_sigma` scales a relative block by
# `a_l_per_pt`, the MF4 coefficients interpolated onto each EXFOR energy, and
# those are NOT zero there -- what is empty at those slots is the MC validity
# mask, not the coefficient. So the preserved values (up to 7.6 relative
# variance) enter at full weight, into precisely the directions where c34_post
# has zero variance, i.e. the softest ones, which whitening amplifies by
# lambda^-1/2.
#
# ⚠ AND THE OLD PSD TABLE COULD NOT HAVE SEEN THIS. Every row of it is taken
# against a_nom, which is zero at those 1542 slots, so the congruence is
# SINGULAR and annihilates exactly the directions in question. Run 90 passed
# that table at lam_min/scale = -1.9e-17 and still died. Hence the second table.
#
# THE SECOND GAP, INDEPENDENT AND ALSO NEVER DIAGNOSED. The triple certified PSD
# was (c33_post, c34_post, cx_post). We rebuilt MF34 but MF33 is still shipped
# as-is, so what the chi2 folds is (c33_ship, c34_written, cx_post). Row E
# isolates it.
#
# HOW TO READ THE SECOND TABLE -- this is the whole point of the job:
#
#   C ~ B                     -> the null-slot fill is the problem, D is the fix.
#                                Re-run run 90 with --null-fill zero (default).
#   C and D both bad, E clean -> MF33 is the problem, not MF34. Rebuilding MF34
#                                alone can never work; MF33 must be rewritten too
#                                and that is a bigger change.
#   D and E both bad          -> neither. The cross block does not belong next to
#                                these marginals at all, and §10.2 (Level B, the
#                                properly joint MC) is the only route left.
#
# PRIOR, RECORDED BEFORE THE JOB LANDS: C ~ B and D is clean, i.e. the null-slot
# fill is the whole story and the fix is one flag.
#
# Resources: 300G is more than enough (peaks ~5 GB). Nine eigendecompositions of
# a 4218^2 / 4406^2 joint, ~30 s each, on top of a ~5 min builder.
# ---------------------------------------------------------------------------

#XDIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_88_fullcross
#SHIPPED_ENDF=26-Fe-56g_nominal_mg.endf

# --check writes NOTHING -- not the sidecars, not the ENDF. The sidecars from
# run 90 are already on disk and are unchanged by this.
#
# --source-endf stays explicit: 26-Fe-56g_nominal_consistent_mg.endf is still
# sitting in $XDIR from run 90 and sorts FIRST, so auto-discovery would read the
# grids and c34_ship off run 90's own output and self-compare.
#python build_group_cross.py --run-dir "$XDIR" \
#    --source-endf "$XDIR/$SHIPPED_ENDF" \
#    --check || exit 1

# --- WHAT TO DO WITH THE ANSWER --------------------------------------------
# If C ~ B and D is clean, the re-run of run 90 is exactly the previous job with
# --null-fill zero added (it is now the DEFAULT, so the flag is only for the
# record) and a fresh tag. Uncomment the block below and flip step 1 back to
# --write-endf. `predictive_91` must be REGISTERED in chi2_analysis_cluster.py
# BEFORE launching -- that omission cost runs 85 and 89 their analyses.
#
# CONSISTENT_ENDF=26-Fe-56g_nominal_consistent_mg.endf
# python build_group_cross.py --run-dir "$XDIR" \
#     --source-endf "$XDIR/$SHIPPED_ENDF" \
#     --null-fill zero \
#     --write-endf "$XDIR/$CONSISTENT_ENDF" || exit 1
# KIKA_THIS_WORK_DIR=$XDIR KIKA_THIS_WORK_ENDF=$CONSISTENT_ENDF \
# KIKA_MF33_MF34_CROSS_DIR=$XDIR KIKA_MF33_MF34_CROSS_SCALE=1.0 \
# KIKA_RUN_TAG=91 \
#     python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_91 KIKA_CHI2_RUN_ID=091 \
#     python chi2_analysis_cluster.py \
#     || echo "[FAIL] run 91 chi2 exited non-zero — read the [PSD] lines FIRST"
# KIKA_CROSS_BASE_RUN=086 KIKA_CROSS_BASE_TAG=86 \
# KIKA_CROSS_RUN=091 KIKA_CROSS_TAG=91 KIKA_CROSS_SCAN="091:1.0" \
#     python compare_cross_block_runs.py \
#     || echo "[WARN] comparison reported a failed gate — read it"

# --- PREVIOUS JOB: run 90 (job 8451628) — no chi2 produced ------------------
# Steps 1 and 2 fine; step 3 died on potrf 63. Kept as the record. Do NOT re-run
# as written: it ships --null-fill ship, which is what §L11 is diagnosing.
#
# CONSISTENT_ENDF=26-Fe-56g_nominal_consistent_mg.endf
# python build_group_cross.py --run-dir "$XDIR" \
#     --source-endf "$XDIR/$SHIPPED_ENDF" \
#     --write-endf  "$XDIR/$CONSISTENT_ENDF" || exit 1
# KIKA_THIS_WORK_DIR=$XDIR KIKA_THIS_WORK_ENDF=$CONSISTENT_ENDF \
# KIKA_MF33_MF34_CROSS_DIR=$XDIR KIKA_MF33_MF34_CROSS_SCALE=1.0 \
# KIKA_RUN_TAG=90 \
#     python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_90 KIKA_CHI2_RUN_ID=090 \
#     python chi2_analysis_cluster.py || echo "[FAIL] ..."

# --- PREVIOUS JOB: run 89 (job 8449493) — no chi2 produced -------------------
# Kept only as the record. Do not re-run as written: it ships cx_post as a
# sidecar next to the shipped MF34, which measures lam_min/scale = -0.447.
# The two failures are §10.1.8-L1 (missing methodology entry) and L2/L3 (the
# undiagnosed joint). Step 1 below is what run 90 replaces with --write-endf.
#
# python build_group_cross.py --run-dir "$XDIR" --write || exit 1
# KIKA_THIS_WORK_DIR=$XDIR KIKA_MF33_MF34_CROSS_DIR=$XDIR \
# KIKA_MF33_MF34_CROSS_SCALE=1.0 KIKA_RUN_TAG=89 \
#     python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_89 KIKA_CHI2_RUN_ID=089 \
#     python chi2_analysis_cluster.py || echo "[FAIL] ..."

# --- WHAT TO DO WITH THE ANSWER --------------------------------------------
# Step 3 prints s_max exactly. Three outcomes:
#
#   s_max >~ 0.5   the cross term is nearly usable as measured; adopt the
#                  largest PD scale, state the damping in the write-up, and
#                  report what it does to V4 and to coverage.
#   s_max ~ 0.1    a damped term is a token, not the measurement. That is the
#                  signal to go to option 3: rebuild Sigma^cross in the SAME
#                  representation as the diagonals — collapse it onto the
#                  multigroup grid and put it through the same near-zero guard —
#                  so the three pieces are consistent by construction instead of
#                  by damping.
#   s_max ~ 0      the inconsistency is structural, not marginal. Option 3 is
#                  the only route, and the roadmap needs to say so.
#
# Either way the deliverable is the same: ONE consistent evaluation shipping
# MF4 + MF34 + MF33 + the cross block, scored against JEFF and JENDL.

# --- previous job: run 87 at full strength (job 8445747) --------------------
# Parquet and sidecar are on disk; the analysis died on a non-PD Sigma_V4, so
# there is no run_087 report and there will not be one at scale 1.0.
#
# KIKA_THIS_WORK_DIR=$XDIR KIKA_MF33_MF34_CROSS_DIR=$XDIR KIKA_RUN_TAG=87 \
#     python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_87 KIKA_CHI2_RUN_ID=087 \
#     python chi2_analysis_cluster.py || exit 1

# --- previous job: run 86 — the multigroup mask fix, scored ----------------
# Landed 2026-08-03 -> run_086. The fine ENDF came out byte-identical to run
# 85's (cmp over 843,376,212 bytes), so run 86 differs from run 85 only in the
# _mg.endf. Dead parameters at l=6: 81.7 % -> 1.8 %. V4 better 1.0-2.2 % on
# every subset; calibration flat to 0.20 pp. 100G was enough for that one.
#
# KIKA_THIS_WORK_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_86_mgfix \
# KIKA_RUN_TAG=86 python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_86 KIKA_CHI2_RUN_ID=086 \
#     python chi2_analysis_cluster.py || exit 1

# --- previous job: score run 85's FINE ENDF (the Phase-3 verdict) -----------
# -> run_085_fine. Needed 300G: the fine MF34 is ~734 MB on disk, ~0.5 GB
# parsed. Kept because it is the fine-grid contrast for anything that follows.
#
# KIKA_THIS_WORK_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_85_mixture \
# KIKA_THIS_WORK_ENDF=26-Fe-56g_nominal.endf \
# KIKA_RUN_TAG=85_fine python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_85_fine KIKA_CHI2_RUN_ID=085_fine \
#     python chi2_analysis_cluster.py || exit 1

# --- previous job: score run 83, and repair the stale representation gate ----
# Produced run_082_enrsl/ and run_083/, and resolved the repr_mg gate failure
# (it reproduces run_082_enrsl at 0.000e+00). Restore -t 1-00:00:00 / 100G.
#
# KIKA_RUN_TAG=82_enrsl python precompute_chi2_predictive.py || exit 1
# KIKA_THIS_WORK_DIR=/share_snc/snc/JuanMonleon/ENDF_samples/new_test_83 \
# KIKA_RUN_TAG=83 python precompute_chi2_predictive.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_82_enrsl KIKA_CHI2_RUN_ID=082_enrsl \
#     python chi2_analysis_cluster.py || exit 1
# KIKA_CHI2_METHODOLOGIES=predictive_83 KIKA_CHI2_RUN_ID=083 \
#     python chi2_analysis_cluster.py || exit 1
# python compare_enrsl_runs.py || exit 1
# REPR_MODES="mg fine fine_cov3 fine_eval3" KIKA_GATE_REF_RUN=082_enrsl \
#     python compare_representation_modes.py || exit 1

# --- previous job: the MF34 representation sweep ----------------------------
# Filling in cov4/cov5/eval4/eval5 remains the natural follow-up: the extremes
# showed a real effect (cov3 +10.6 %, eval3 +185.6 % on V4/all). 300G.
#
# REPR_MODES="fine_cov4 fine_cov5 fine_eval4 fine_eval5"
# export REPR_MODES
# python precompute_chi2_representation.py || exit 1
# METHS=""
# for m in $REPR_MODES; do METHS="${METHS:+$METHS,}repr_${m}"; done
# KIKA_CHI2_METHODOLOGIES="$METHS" python chi2_analysis_cluster.py || exit 1
# KIKA_GATE_REF_RUN=082_enrsl python compare_representation_modes.py || exit 1
# unset REPR_MODES KIKA_CHI2_METHODOLOGIES
