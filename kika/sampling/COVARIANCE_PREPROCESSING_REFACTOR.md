# Covariance pre-processing refactor — TODO

Status: **not started**. This is a parking note from a 2026-05-09 audit so the
work can be picked up later.

## The philosophy

**Perturbation pipelines should perturb, nothing else.** If an input covariance
has 1000% uncertainty in a threshold bin, an evaluator placeholder, or a tiny
negative eigenvalue, that's a *data* problem and the fix belongs in a separate,
explicit pre-processing step — not silently inside `generate_samples`.

Today the σ-pipelines (`pendf_perturbation`, `ace_perturbation`,
`combined_perturbation` MF33 leg) violate that: they modify the covariance
mid-flight before the draw. The MF34/ENDF leg is already close to clean (only
PSD repair).

## What runs today (default-ON, in `generate_samples`)

In execution order — see `kika/sampling/generators.py` and
`kika/cov/decomposition.py`:

1. **Threshold-bin rescale**
   `generators.py:515-536` → `flag_threshold_bins` + `rescale_threshold_bins_congruence`.
   Toggle `PIN_THRESHOLD_BINS = True`.
   The bin straddling each reaction threshold has σ² pulled down (one-sided)
   to the per-MT median, via `Σ' = diag(s) Σ diag(s)`. PSD- and
   correlation-preserving but does change σ².

2. **Outlier rescale (1000× per-MT median)**
   `generators.py:547-568` → `flag_outlier_variance_bins`.
   **No toggle — always runs.** Targets evaluator placeholders such as
   JEFF-4.0 Mn-55 MT=856 in filler bins above ~22 MeV.

3. **Global variance cap**
   `generators.py:573-587` → `cap_variance_congruence`.
   `MAX_RELATIVE_STD = 3.0` (300% σ_rel). Same congruence transform.

4. **PSD repair**
   `psd_method="auto"` default. Inside `cholesky_decomposition` /
   `svd_decomposition` / `eigen_decomposition`. Clips negative eigenvalues if
   `|λ_min|/λ_max < 1%`, otherwise Higham (with fallback-to-clip if Higham
   doesn't converge). Always silently runs — sampling literally cannot proceed
   on a non-PSD matrix.

`AUTOFIX` defaults to `None` (off) in pendf/combined, so `cov.fix_covariance()`
isn't part of the default path.

## What runs today (legitimate, leave alone)

- **σ̄-floor in absolute→relative MF33 conversion** —
  `mf33_sampling.py:454-472`. Zeroes the relative cov where σ̄ ≈ 0 to avoid
  `1/σ̄ → ∞`. This is a guard, not a modification of meaningful data.
- **Inert-bin mask** — `generators.py:600-621`. Drops σ²<1e-12 / non-finite
  bins from the decomposition; restored as factor=1 after. Mathematically a
  no-op.
- **Negative-variance zeroing** in `legendre_covariance.py:1361-1363`. Only
  on a display path, not on the sampling cov.

## Target architecture

Two-stage flow, with the modifications moved upstream into an explicit,
file-level pre-processing pass that the user runs once and inspects.

```
   raw cov (MF33 / MF34 / SCALE) 
         │
         ▼
   ┌──────────────────────────┐
   │  cov pre-processing      │   ← new module, e.g. kika/cov/preprocess.py
   │  - threshold-bin rescale │      runs once per (ENDF, MT-set)
   │  - outlier rescale       │      writes a "cleaned cov" artifact
   │  - variance cap          │      logs every modification with
   │  - PSD repair            │        before/after diagnostics
   └─────────┬────────────────┘
             │  cleaned cov (PSD, capped, threshold-fixed)
             ▼
   ┌──────────────────────────┐
   │  generate_samples        │   ← strictly draws from the cov it's given
   │  no caps / no rescales   │      no autofix / no PSD repair
   │  PSD assumed             │      fails loudly if not PSD
   └──────────────────────────┘
```

The pre-processing pass is what the user runs as a one-shot, inspects, and
versions. The sampler trusts its input.

## Concrete tasks

In rough order of independence:

1. **Carve out a `kika/cov/preprocess.py` module** that exposes one function
   per modification (already exist as primitives in `decomposition.py`):
   - `rescale_threshold_bins(cov, mt_thresholds, ...)`
   - `rescale_outlier_bins(cov, factor=1000, ...)`
   - `cap_relative_uncertainty(cov, max_relative_std, ...)`
   - `make_psd(cov, method="auto", ...)`

   Each returns `(cleaned_cov, info_dict)`. Pure functions, no implicit logging
   side-effects, no implicit toggles.

2. **Make a single orchestration entry point** —
   `preprocess_cov(cov, recipe: dict) -> (cov_cleaned, full_info_dict)`.
   The recipe is explicit (which steps to run, with which parameters).
   Default recipe matches today's behavior so existing pipeline runs reproduce.

3. **Strip the modifications out of `generate_samples`.** Steps 2a, 2a-bis,
   2b in `generators.py` go away. PSD repair stays *only* if you decide the
   sampler should still self-protect; otherwise it goes too and the sampler
   raises on non-PSD input.

4. **Wire the pipelines** (`pendf_perturbation`, `ace_perturbation`,
   `combined_perturbation`) to call `preprocess_cov` between cov assembly
   and `generate_samples`. The pipelines surface the recipe as a single
   parameter (or accept a pre-cleaned cov).

5. **Add a CLI / utility script** to run pre-processing standalone, write the
   cleaned cov to disk (parquet / .npz), and print the modification report.
   Lets users inspect the cleaning before sampling.

6. **Update `PIPELINES.md`** to describe the two-stage flow and remove the
   mentions of `MAX_RELATIVE_STD`, `PIN_THRESHOLD_BINS`, etc., as in-pipeline
   knobs.

7. **Tests**:
   - Unit tests on each preprocess primitive (already partially covered for
     congruence transforms — see `tests/test_*` if present, otherwise add).
   - End-to-end: run pipeline on a known dirty cov twice — once with the
     default recipe, once with a `null` recipe — and assert the cov fed to
     `generate_samples` matches expectations.

## What to keep an eye on while refactoring

- **Threshold detection lives across two places today.** `pendf_perturbation`
  reads thresholds from PENDF MF3 first-positive σ; `ace_perturbation` reads
  them from ACE σ(E) plus a cov-diagonal fallback. The new preprocess module
  should accept thresholds as input (decoupled from PENDF/ACE) and let the
  pipeline supply them.
- **Outlier rescale has no toggle today.** Easy to miss — it always runs.
- **PSD repair is non-trivial to externalize** because the Higham→clip
  fallback decision uses post-hoc Frobenius error. Either replicate that
  logic in the preprocess pass, or keep a thin "guarantee PSD" call inside
  the sampler as the only remaining mod (and document it).
- **Backwards compatibility.** Existing pipelines and any saved parquet
  factors implicitly depend on the current modifications. New runs with the
  empty recipe will produce different samples; flag this clearly.

## Files to touch

- `kika/cov/decomposition.py` — primitives stay here or move
- `kika/cov/preprocess.py` — **new**, orchestration
- `kika/sampling/generators.py` — strip steps 2a, 2a-bis, 2b (and possibly PSD)
- `kika/sampling/pendf_perturbation.py` — drop `MAX_RELATIVE_STD`,
  `PIN_THRESHOLD_BINS`, `AUTOFIX` knobs; accept cleaned cov / recipe instead
- `kika/sampling/ace_perturbation.py` — same
- `kika/sampling/combined_perturbation.py` — same (MF33 leg)
- `kika/sampling/endf_perturbation.py` — only the PSD-repair question applies
- `kika/sampling/PIPELINES.md` — update prose
- New CLI under `kika/sampling/run_preprocess_cov.py` (or similar)

## Decisions deferred to whoever picks this up

- Should PSD repair stay inside the sampler (as the *only* remaining mod) or
  move out and have the sampler raise on non-PSD input? The latter is purer
  but means anyone using the API directly has to know about pre-processing.
- Should the pre-processing artifact be saved alongside the perturbed
  ENDF/PENDF/ACE outputs (audit trail) or kept ephemeral?
- Naming: `preprocess` vs `clean` vs `condition` vs `regularize`. Pick one
  and stick to it — today the same thing is called "autofix", "PSD repair",
  "rescale", "cap", "fix" depending on which file you look at.
