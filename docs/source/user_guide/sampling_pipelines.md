# Cross-section perturbation pipelines

Three pipelines live in `kika/sampling/` for sampling perturbed nuclear data:

| Pipeline                       | Perturbs                  | Driven by   | Output                |
|--------------------------------|---------------------------|-------------|-----------------------|
| `ace_perturbation.py`          | σ(E) on a multigroup grid | MF33 cov    | ACE                   |
| `pendf_perturbation.py`        | σ(E) on the MF33 native grid | MF33 cov | PENDF (+ optional ACE) |
| `endf_perturbation.py`         | MF4 Legendre coefficients | MF34 cov    | ENDF (+ optional ACE) |
| `combined_perturbation.py`     | both σ(E) **and** MF4     | MF33 + MF34 | ENDF + PENDF + ACE    |

This guide focuses on `pendf_perturbation` and `combined_perturbation` — the
two newer pipelines built on the MF33 native grid. `ace_perturbation` is the
older multigroup path; prefer the others.

---

## How the new pipelines are wired

Both new pipelines split into two stages so the combined orchestrator can
reuse the single-physics building blocks without re-running NJOY:

- **Stage A** — sample on the covariance native grid, write a perturbed tape
  (PENDF or ENDF) to disk. No NJOY downstream of RECONR.
- **Stage B** — given a `(perturbed ENDF, perturbed PENDF)` pair, run
  NJOY (skipping RECONR) → ACE + xsdir.

`pendf_perturbation` runs Stage A then optionally Stage B with the
*original* ENDF + the perturbed PENDF.
`combined_perturbation` runs Stage A for MF33 and Stage A for MF34
independently, then calls Stage B with the *perturbed* ENDF + perturbed
PENDF for sample i.

The decoupling matters because MF33↔MF34 cross-correlations are
structurally zero in our methodology (see
`manuscript/notes/mf3_mf4_cross_correlations_methodology.md`), so the two
samplings are genuinely independent and pairing-by-index is correct.

---

## Files at a glance

```
kika/sampling/
├── pendf_perturbation.py        # MF33 σ → PENDF (+ ACE)
│   └── perturb_PENDF_files()
├── mf33_sampling.py             # cov assembly + MF3 rewrite helpers (unit-testable)
│   ├── load_mf33_covariance()
│   ├── apply_factors_to_pendf_mf3()
│   └── perturb_pointwise_xs()
├── endf_perturbation.py         # MF34 → MF4 ENDF (pre-existing)
│   └── perturb_ENDF_files()
├── combined_perturbation.py     # MF33 + MF34 paired
│   └── perturb_ENDF_PENDF_combined()
├── run_pendf_perturbation.py    # CLI wrapper
├── run_combined_perturbation.py # CLI wrapper
└── generators.py, utils.py, diagnostics.py   (shared sampling math + parquet)

kika/processing/njoy_pendf_cache.py
    # SHA256-keyed PENDF cache — runs NJOY RECONR on first call,
    # returns cached tape on subsequent calls. Used by both new pipelines.

kika/njoy/
├── run_njoy.py                  # full chain (RECONR → ACER)
│   ├── run_njoy()               # original
│   └── run_njoy_with_pendf()    # injects an externally-generated PENDF
└── templates.py
    ├── NJOY_INPUT_TEMPLATE              # standard chain
    └── NJOY_INPUT_TEMPLATE_WITH_PENDF   # RECONR replaced by `moder 26 -21`

tests/
├── test_mf33_sampling.py        # unit tests (no NJOY)
├── test_pendf_perturbation.py   # integration (skips without NJOY+ENDF)
└── test_combined_perturbation.py
```

---

## Pipeline 1 — `perturb_PENDF_files` (MF33 → σ)

Samples cross-section perturbation factors directly on the MF33 native
energy grid (no multigroup), rewrites MF3 in the PENDF, and optionally
chains through NJOY broadr → acer to produce ACE.

### Sequential flow

```
perturb_PENDF_files()
  for each ENDF file:
    1. parse MF33 from ENDF
    2. cached PENDF via NJOY RECONR  (kika.processing.njoy_pendf_cache)
    3. assemble unified covariance on the MF33 native union grid
       (load_mf33_covariance — converts absolute LB-types to relative,
        resolves NC LTY=0 sum-rules using PENDF MF3 σ(E))
    4. extract per-MT thresholds from the PENDF (for threshold-bin pinning)
    5. generate_samples(cov, ...) → factors of shape (N, n_MTs × n_native_bins)
    6. write the per-isotope parquet of factors
    7. for each sample s:
         a. apply_factors_to_pendf_mf3(...)
            σ_new(E_k) = σ_orig(E_k) × factor[g(E_k)]   piecewise constant
         b. if "ace" in output_formats:
              _stage_b_run_njoy_for_pair(original ENDF, perturbed PENDF)
```

### Quick start

```python
from kika.sampling.pendf_perturbation import perturb_PENDF_files

summary = perturb_PENDF_files(
    endf_files="/path/to/Fe-56.endf",
    mt_list=[2, 102],          # MTs to perturb
    num_samples=100,
    output_formats=("pendf", "ace"),
    njoy_exe="/home/MONLEON-JUAN/NJOY2016/build/njoy",
    ace_temperatures=[293.6],
    ace_library_name="endfb81",
    pendf_cache_dir="/mnt/c/Users/MONLEON-DE-LA-JAN/AppData/Local/Temp/kika_pendf_cache",
    output_dir="/path/to/output",
    seed=42,
    nprocs=4,
)
```

For a PENDF-only run (no NJOY post-processing), drop ACE:

```python
output_formats=("pendf",)   # no NJOY post-PENDF; NJOY only used for RECONR
```

### Key knobs

All knobs live as ALL_CAPS constants at the top of `pendf_perturbation.py`
and as defaults in `perturb_PENDF_files`. The ones that actually move the
needle:

- `space="log"` — sampling space (preserves σ > 0; locked-in default).
- `pin_threshold_bins=True` — below-threshold bins forced to factor 1.
- `max_relative_std=3.0` — 300 % cap on per-bin σ relative std.
- `psd_method="auto"` + `autofix=None` — autofix is off by default; set to
  `"soft"` / `"medium"` / `"hard"` only if a covariance refuses to sample.
- `verbose_diagnostics` — single integer gating all extra logging
  (0 = off, 1 = summaries, 2 = per-MT).

---

## Pipeline 2 — `perturb_ENDF_PENDF_combined` (MF33 + MF34)

Pairs the σ-perturbed PENDF with the angular-perturbed ENDF, sample by
sample, and runs NJOY once per pair to produce one ACE per sample with
*both* perturbations applied.

### Sequential flow

```
perturb_ENDF_PENDF_combined()
  Stage A.MF34: perturb_ENDF_files(generate_ace=False)
                → mf4/endf/<zaid>/<NNNN>/<base>_<NNNN>.endf
  Stage A.MF33: perturb_PENDF_files(output_formats=("pendf",))
                → mf3/pendf/<zaid>/<NNNN>/<base>_<NNNN>.pendf
  Stage B    : for each sample s:
                _stage_b_run_njoy_for_pair(
                    endf_path = perturbed ENDF for sample s,
                    pendf_path = perturbed PENDF for sample s,
                    ...)
                → ace/<library>/<TK>/<zaid>/<NNNN>/...
```

### Seed handling

A single `seed=S` reproduces the same paired set: MF33 sampling uses `S`,
MF34 uses `S + 1`. The two draws stay independent (no MF33↔MF34 correlation
is implied) but are repeatable.

### Quick start

```python
from kika.sampling.combined_perturbation import perturb_ENDF_PENDF_combined

summary = perturb_ENDF_PENDF_combined(
    endf_files="/path/to/Fe-56.endf",
    mt_list_mf33=[2, 102, 103],   # σ MTs
    mt_list_mf34=[2],             # angular MTs (often only MT=2 has MF34)
    legendre_coeffs=[1, 2, 3],
    num_samples=200,
    generate_ace=True,
    njoy_exe="/home/MONLEON-JUAN/NJOY2016/build/njoy",
    ace_temperatures=[293.6],
    ace_library_name="endfb81",
    pendf_cache_dir="/mnt/c/Users/MONLEON-DE-LA-JAN/AppData/Local/Temp/kika_pendf_cache",
    output_dir="/path/to/output",
    seed=42,
    nprocs=8,
)
```

Set `generate_ace=False` to land just the perturbed ENDFs and PENDFs (no
NJOY ACER pass) — useful for inspecting the inputs before committing to a
long run.

---

## Output layout

ACE/xsdir are co-located per-sample. All isotopes and temperatures for
sample `NNNN` share the same `ace/<NNNN>/` directory; the filename
`<ZAID>.<ext>` disambiguates them. The extension is **caller-supplied**
via `extensions=['06c']` (paired with `temperatures=[600]`), not derived
from a hardcoded `K_TO_SUFFIX` lookup.

### `pendf_perturbation`

```
output_dir/
├── pendf_perturbation_<ts>.log
├── perturbation_matrix_<ts>_master.parquet           # (N, n_MTs × n_bins)
├── pendf/<zaid>/<NNNN>/<base>_<NNNN>.pendf
├── ace/<NNNN>/<ZAID>.<ext>                           # if "ace" in output_formats
├── ace/<NNNN>/xsdir/<ZAID>.<ext>.xsdir               # one per (ZAID, ext)
├── njoy_files/<NNNN>/                                # if keep_njoy_io
└── xsdir/...                                         # if xsdir_file given (master)
```

### `combined_perturbation`

Same flat layout as the single-physics pipelines. Both sub-pipelines write
their tapes, parquets, and logs directly into `output_dir/` — no
intermediate `mf3/` / `mf4/` subtree. Filenames don't collide
(`endf_perturbation_*` vs. `pendf_perturbation_*` vs.
`combined_perturbation_*`).

```
output_dir/
├── combined_perturbation_<ts>.log                  # orchestrator
├── endf_perturbation_<ts>.log                      # MF34 sub-step
├── pendf_perturbation_<ts>.log                     # MF33 sub-step
├── endf_perturbation_factors_<ts>.parquet          # Legendre factors
├── perturbation_matrix_<ts>_master.parquet         # σ factors
├── endf/<zaid>/<NNNN>/<base>_<NNNN>.endf           # MF34-perturbed ENDF
├── pendf/<zaid>/<NNNN>/<base>_<NNNN>.pendf         # MF33-perturbed PENDF
├── ace/<NNNN>/<ZAID>.<ext>                         # paired ACEs (Stage B)
├── ace/<NNNN>/xsdir/<ZAID>.<ext>.xsdir             # paired xsdir snippets
├── njoy_files/recon/<zaid>/<base>_recon.{input,output}
│                                                   # RECONR deck + stdout
│                                                   # (only if keep_njoy_io)
├── njoy_files/<NNNN>/<base>.{input,output}         # acer-chain I/O
│                                                   # (only if keep_njoy_io)
└── xsdir/...                                       # if xsdir_file given (master)
```

The two parquet matrices stay separate by design — one per perturbation
kind. Don't try to merge them; they have different column schemas (per-MT
× native bin for σ vs. per-MT × Legendre × energy bin for angular).

When `keep_njoy_io=True`, the pipeline preserves both NJOY runs:

- **First pass — RECONR**, run once per ENDF (cached) to produce the
  pointwise PENDF that MF33 sampling needs. Lands at
  `njoy_files/recon/<zaid>/<base>_recon.input` and `.output`.
- **Second pass — broadr → heatr → purr → gaspr → acer**, run once per
  sample with the perturbed PENDF on tape 26. Lands at
  `njoy_files/<temp_str>/<zaid>/<NNNN>/<base>_<NNNN>.{input,output}`.

Cache hits skip RECONR; the pipeline still copies the previously-saved
RECONR I/O from the cache directory into `njoy_files/recon/` so the
audit trail is complete.

### PENDF cache

Lives at `pendf_cache_dir` (default = `tempfile.gettempdir()/kika_pendf_cache`,
override on WSL to `/mnt/c/Users/MONLEON-DE-LA-JAN/AppData/Local/Temp/kika_pendf_cache`).
Files are keyed by `<sha256(ENDF bytes)>_<tolerance>.pendf`, so a byte
edit to the ENDF invalidates the cache automatically.

---

## MF4 (angular) perturbation defaults

`perturb_ENDF_files` (also routed via `combined_perturbation` Stage A.MF34)
samples the raw MF34 covariance without modifying it:

- `psd_method="none"` (default) — no PSD repair. SVD silently folds
  negative eigenvalues to their magnitude; one WARN line per (MT, block)
  is emitted when |λ_min|/λ_max > 1e-8.
- `higham_projection=False` (default) — opt-in Higham repair. Setting to
  True overrides `psd_method` to `"higham"`; the two are mutually exclusive.
- `decomposition_method="svd"` (default).
- `enforce_positivity=True` (default) — after each sample is assembled
  (factor × baseline → ENDF a_l), the per-energy angular probability density
  `f(μ) = (1/2) Σ (2l+1) a_l P_l(μ)` is evaluated on
  `positivity_check_points=101` evenly-spaced μ points in [-1, 1]. If
  `f(μ) < 0` anywhere on the grid, the coefficient vector is projected to
  the L²-nearest non-negative set via SLSQP (with `a_0=1` and the high-l
  tail outside the perturbed band pinned at their baselines).

Every projection is logged. In verbose mode each event prints a line with
the sample index, MT, incident energy, the worst pre-projection σ(μ), and
the largest coefficient shift. Both modes emit per-MT summary totals at
the end of the file's processing.

The positivity primitives live in `kika/sampling/mf4_positivity.py`
(`check_mf4_positivity`, `project_mf4_to_positive`). They are pure
functions and have a standalone test suite in `tests/test_mf4_positivity.py`.

## Verifying a run

After `perturb_PENDF_files` completes, the contract is that for each sample
`s`, MT `m`, and pointwise PENDF energy `E_k`:

```
σ_new(E_k) = σ_orig(E_k) × factor[g(E_k)]
```

with `g = searchsorted(union_grid, E_k, side="right") - 1` and
`factor = 1.0` whenever `E_k` falls outside the MF33 coverage.

Equality holds at the ENDF-6 numerical-format precision floor (~5e-7
relative). To check on a real run:

```python
import glob, numpy as np, pandas as pd
from kika.endf import read_endf
from kika.sampling.mf33_sampling import (
    load_mf33_covariance, extract_mt_param_blocks, perturb_pointwise_xs,
)

ENDF, PENDF_CACHE, OUT = ...  # paths
cov, mf3, union_grid, mts = load_mf33_covariance(ENDF, PENDF_CACHE, [2])
mt_blocks = extract_mt_param_blocks(cov)

master = pd.read_parquet(sorted(glob.glob(f"{OUT}/perturbation_matrix_*_master.parquet"))[0])
factors_flat = master.iloc[0][[c for c in master.columns if c != "Sample_ID"]].to_numpy()
fblock = factors_flat[mt_blocks[2]]

ref = np.asarray(mf3[2].cross_sections)
e   = np.asarray(mf3[2].energies)
expected, _, _ = perturb_pointwise_xs(e, ref, fblock, np.asarray(union_grid))

pert_xs = np.asarray(read_endf(
    sorted(glob.glob(f"{OUT}/pendf/<zaid>/0001/*.pendf"))[0],
    mf_numbers=[3]
).get_file(3).sections[2].cross_sections)

rel = np.abs(pert_xs - expected) / np.maximum(expected, 1e-30)
assert rel.max() < 2e-5
```

For a `combined_perturbation` run, the natural sanity check is that
distinct sample ACEs have *different* energy grids (NJOY's adaptive
linearizer responds to even small σ changes), which by itself proves the
σ perturbation propagated through broadr/heatr/purr/acer.

---

## Warnings you may see (and what they mean)

- **`UserWarning: Skipping MF sections without parsers: [6, 8, 10, 12, 13, 14]`**
  Emitted by `kika.endf.parsers.parse_endf` whenever an ENDF tape contains
  MF sections kika doesn't have a parser for (energy-angle distributions,
  decay/yield, photon production). Benign — the pipeline only needs
  MF1/2/3/4/33/34 and ignores the rest. MF32, the resonance-parameter
  covariance, used to appear in this list and no longer does: it is read as of
  2026-08-10, though nothing in the sampling pipeline consumes it yet.

- **`Unexpected MT152 in MF2 (only MT151 expected), skipping`**
  ENDF/B-VIII added MT152 (energy-dependent self-shielding) under MF2. The
  parser only reads MT151 (resonance parameters) and skips the rest cleanly.

- **`MTx: skipping MTy (circular NC reference)`**
  The MF33 NC LTY=0 sum-rule resolver detected an MT referring back to one
  already being resolved up the call stack. The recursion is bounded
  by `MF33MT.resolve_nc_lty0`'s `_resolving` guard — safe to ignore.

- **`RuntimeWarning: divide by zero encountered in log` at `mf33.py`**
  Was emitted by `MF33MT._bin_average_xs` when a covariance bin edge was
  exactly 0 eV. The current code guards `np.log(grid[g])` against
  non-positive edges and just skips the degenerate bin (output 0). If
  you still see it, the kika install lags behind the local source.

## Common gotchas

- **NJOY `tape21` vs `-21`**. NJOY/gfortran treats `tapeNN` and `tape-NN`
  as the same Fortran unit, so you can't open one as formatted (ASCII
  PENDF input) and another as unformatted (binary scratch) in the same
  run. `NJOY_INPUT_TEMPLATE_WITH_PENDF` routes the ASCII PENDF through
  `tape26` for this reason. If you copy the template into a new pipeline,
  keep the input unit out of the {21, 22, 23, 24, 25, 40, 41, 43, 44}
  range that the chain uses internally.

- **`update_directory=False` when rewriting MF3 in a PENDF**. NJOY MODER
  reads PENDF tapes via tape units, not via the MF1/MT451 directory; if
  you ask the writer to update the directory you'll waste cycles and risk
  inconsistencies. `apply_factors_to_pendf_mf3` already passes this.

- **MF33 absolute → relative conversion needs PENDF MF3**. If MF33 carries
  any LB-type ∈ {0, 1, 2} block, `load_mf33_covariance` converts it to
  relative using bin-averaged σ(E) from the PENDF. A missing PENDF means
  those blocks are silently dropped (with a warning). Keep the cache
  reachable.

- **MF33 native grid coverage gaps**. MF33 typically stops at 20 MeV;
  PENDF goes higher. Out-of-coverage points get factor = 1.0 (no
  perturbation). The diagnostics dict logs `frac_out_of_coverage` per
  sample × MT — non-zero is fine, but if you're seeing > 0.1 it means
  half your high-energy points aren't being touched.

- **Combined pipeline pair-mismatch**. The pair of `(perturbed ENDF,
  perturbed PENDF)` sample `i` is matched purely by directory naming
  (`<base>_NNNN.endf` ↔ `<base>_NNNN.pendf`). If you intercept either
  Stage A and rename or remove files, the Stage B loop will skip the
  affected samples with a `[WARN] [COMB] ... missing inputs` line.

- **PSD warnings are warnings**. `psd_method="auto"` will *clip* mildly
  negative eigenvalues before SVD and log a warning; this is the
  intended path. Don't enable `autofix` unless you've confirmed clipping
  is materially changing the spectrum (final-matrix PSD checks are
  warn-only by design).

---

## Reference: the test fixtures

Integration tests in `tests/test_pendf_perturbation.py` and
`tests/test_combined_perturbation.py` look for:

- `NJOY_EXECUTABLE` env var **or** any of the hard-coded fallbacks (e.g.
  `/home/MONLEON-JUAN/NJOY2016/build/njoy`).
- `KIKA_ENDF_FILES` env var pointing to a directory holding
  `Fe56_jeff4.0_n.endf`, **or** the default `<repo>/files/endf/`.

Both skip cleanly when either is absent. For a quick smoke on this WSL
host the JENDL-5 Fe-56 tape at
`/mnt/c/Users/MONLEON-DE-LA-JAN/Documents/endf_files/260560.jendl5` works
well — it carries MF34 only for MT=2 and exercises an NC LTY=0
sub-subsection in MF33, both of which are good corners to hit.
