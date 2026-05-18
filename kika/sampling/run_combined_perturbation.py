"""
Runnable wrapper around perturb_ENDF_PENDF_combined.

Everything produced by the run lands under ``output_dir`` (flat layout):

    output_dir/
    ├── combined_perturbation_<ts>.log       ← single run log (all stages)
    ├── endf_perturbation_factors_<ts>.parquet   ← Legendre factors per sample
    ├── perturbation_matrix_<ts>_master.parquet  ← σ factors per sample
    ├── endf/<zaid>/<NNNN>/<base>_<NNNN>.endf    ← MF34-perturbed ENDF
    ├── pendf/<zaid>/<NNNN>/<base>_<NNNN>.pendf  ← MF33-perturbed PENDF
    ├── pendf_cache/<sha>_<tol>.pendf            ← RECONR cache (cross-run)
    ├── ace/<temp>/<zaid>/<NNNN>/...             ← paired ACEs (Stage B)
    ├── njoy_files/recon/<zaid>/...              ← RECONR I/O (if keep_njoy_io)
    ├── njoy_files/<temp>/<zaid>/<NNNN>/...      ← acer-chain I/O (if keep_njoy_io)
    └── xsdir/<temp>/...                         ← xsdir entries (if xsdir_file)

The PENDF cache lives inside ``output_dir`` and is keyed by SHA256 of the
ENDF bytes, so a byte edit to the ENDF auto-invalidates the cache.
"""
from kika.sampling.combined_perturbation import perturb_ENDF_PENDF_combined

# 1) Prepare inputs
endf_files = [
    # "/path/to/Fe-56.endf",
]

mt_list_mf33    = [2, 16, 102, 103]   # MTs to perturb in MF33 (cross sections)
mt_list_mf34    = [2, 4]              # MTs to perturb in MF34 (angular)
legendre_coeffs = [1, 2, 3]
num_samples     = 16
output_dir      = "./combined_perturbation_output"

# PENDF cache (NJOY RECONR; survives across runs) ----------------------------
pendf_cache_dir = f"{output_dir}/pendf_cache"
pendf_tolerance = 1.0e-3
keep_njoy_io    = True         # preserve NJOY input/output (both passes)

# NJOY (Stage B → ACE) -------------------------------------------------------
generate_ace     = True
njoy_exe         = "njoy"
ace_temperatures = [293.6]
ace_extensions   = ['02c']    # caller-supplied ACE extension per temperature
ace_library_name = "endfb81"
xsdir_file       = None

# MF34 (angular) defaults ---------------------------------------------------
psd_method_mf34            = "none"   # do nothing to the matrix
higham_projection_mf34     = False    # set True to opt into Higham PSD repair
enforce_positivity_mf34    = True     # project unphysical samples to f(μ)≥0
positivity_check_points_mf34 = 101

# Runtime --------------------------------------------------------------------
seed   = 42
nprocs = 4

# 2) Call perturb_ENDF_PENDF_combined
print(f"Generating {num_samples} paired (ENDF, PENDF, ACE) samples in {output_dir} ...")
perturb_ENDF_PENDF_combined(
    endf_files                   = endf_files,
    mt_list_mf33                 = mt_list_mf33,
    mt_list_mf34                 = mt_list_mf34,
    legendre_coeffs              = legendre_coeffs,
    num_samples                  = num_samples,
    generate_ace                 = generate_ace,
    keep_njoy_io                 = keep_njoy_io,
    pendf_cache_dir              = pendf_cache_dir,
    pendf_tolerance              = pendf_tolerance,
    psd_method_mf34              = psd_method_mf34,
    higham_projection_mf34       = higham_projection_mf34,
    enforce_positivity_mf34      = enforce_positivity_mf34,
    positivity_check_points_mf34 = positivity_check_points_mf34,
    njoy_exe                     = njoy_exe,
    ace_temperatures             = ace_temperatures,
    ace_extensions               = ace_extensions,
    ace_library_name             = ace_library_name,
    xsdir_file                   = xsdir_file,
    output_dir                   = output_dir,
    seed                         = seed,
    nprocs                       = nprocs,
)
