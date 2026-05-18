"""
Runnable wrapper around perturb_PENDF_files.

Everything produced by the run lands under ``output_dir``:

    output_dir/
    ├── pendf_perturbation_<ts>.log              ← single run log
    ├── perturbation_matrix_<ts>_master.parquet  ← σ factors per sample
    ├── pendf/<zaid>/<NNNN>/<base>_<NNNN>.pendf  ← perturbed PENDF tapes
    ├── pendf_cache/<sha>_<tol>.pendf            ← RECONR cache (cross-run)
    ├── ace/<temp>/<zaid>/<NNNN>/...             ← ACE files (Stage B)
    ├── njoy_files/recon/<zaid>/...              ← RECONR I/O (if keep_njoy_io)
    ├── njoy_files/<temp>/<zaid>/<NNNN>/...      ← acer-chain I/O (if keep_njoy_io)
    └── xsdir/<temp>/...                         ← xsdir entries (if xsdir_file)

The PENDF cache lives inside ``output_dir`` and is keyed by SHA256 of the
ENDF bytes, so a byte edit to the ENDF auto-invalidates the cache.
"""
from kika.sampling.pendf_perturbation import perturb_PENDF_files

# 1) Prepare inputs
endf_files = [
    # "/soft_snc/lib/endf/JEFF40/endf/Fe56.endf",
]

mt_numbers  = [2, 4, 16, 102, 103]
num_samples = 16
output_dir  = "./pendf_perturbation_output"

# PENDF cache (NJOY RECONR; survives across runs) ----------------------------
pendf_cache_dir = f"{output_dir}/pendf_cache"
pendf_tolerance = 1.0e-3
keep_njoy_io    = True         # preserve NJOY input/output (both passes)

# NJOY (only used when 'ace' in output_formats) ------------------------------
njoy_exe         = "njoy"
ace_temperatures = [293.6]
ace_library_name = "endfb81"
xsdir_file       = None        # path to a master xsdir to update, or None

# Runtime --------------------------------------------------------------------
seed   = 42
nprocs = 4

# 2) Call perturb_PENDF_files
print(f"Generating {num_samples} perturbed PENDF (+ ACE) files in {output_dir} ...")
perturb_PENDF_files(
    endf_files       = endf_files,
    mt_list          = mt_numbers,
    num_samples      = num_samples,
    output_formats   = ("pendf", "ace"),
    keep_njoy_io     = keep_njoy_io,
    pendf_cache_dir  = pendf_cache_dir,
    pendf_tolerance  = pendf_tolerance,
    output_dir       = output_dir,
    njoy_exe         = njoy_exe,
    ace_temperatures = ace_temperatures,
    ace_library_name = ace_library_name,
    xsdir_file       = xsdir_file,
    seed             = seed,
    nprocs           = nprocs,
)
