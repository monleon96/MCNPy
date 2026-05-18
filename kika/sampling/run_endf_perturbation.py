from kika.sampling.endf_perturbation import perturb_ENDF_files

# Path to ENDF files to perturb
endf_files = [
  '/soft_snc/lib/endf/endfb81/neutrons/n-008_O_016.endf'
  ]

# Specify which MT reactions to perturb (empty list = all available)
mt_list = [2]

# Specify which Legendre coefficients to perturb (Empty list or [-1] = all available L coefficients)
legendre_coeffs = []  

# Number of perturbed samples to generate
num_samples = 512

# Output directory
output_dir = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/ENDFB81_O16"

# Path to njoy executable
njoy_exe = "/soft_snc/NJOY/2016.78/bin/njoy"

# List of temperatures in K, paired with ACE extensions (e.g. '06c' for 600 K)
temperatures = [600]
extensions = ['06c']

# Number of CPUs (in one node)
nprocs = 40

# Library name
library_name = "endfb81"					# 'jendl5', 'jeff40'

# Path to reference xsdir file
xsdir_file = "/share_snc/snc/JuanMonleon/xsdir/xsdir81-DPA"


perturb_ENDF_files(
  endf_files=endf_files,
  mt_list=mt_list,
  legendre_coeffs=legendre_coeffs,
  num_samples=num_samples,
  space="linear",
  decomposition_method="svd",
  psd_method="none",            # do nothing to the matrix by default
  higham_projection=False,      # set True to opt into Higham PSD repair
  enforce_positivity=True,      # project unphysical MF4 samples to f(μ)≥0
  positivity_check_points=101,
  sampling_method="sobol",
  output_dir=output_dir,
  seed=42,
  dry_run=True,
  verbose=True,
  nprocs=nprocs,
  # ACE generation parameters
  generate_ace=False,
  njoy_exe=njoy_exe,
  temperatures=temperatures,
  extensions=extensions,
  library_name=library_name,
  njoy_version="NJOY 2016.78",
  xsdir_file=xsdir_file
)
