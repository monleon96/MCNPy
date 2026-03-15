import os
import kika
from kika.sampling.ace_perturbation import perturb_ACE_files

# 1) Paths to ACE files
ace_H1   = '/soft_snc/lib/ace/80/600/10010_80.06c'
ace_B10  = '/soft_snc/lib/ace/80/600/50100_80.06c'
ace_B11  = '/soft_snc/lib/ace/80/600/50110_80.06c'
ace_C12  = '/soft_snc/lib/ace/80/600/60120_80.06c'
ace_C13  = '/soft_snc/lib/ace/80/600/60130_80.06c'
ace_O16  = '/soft_snc/lib/ace/80/600/80160_80.06c'
ace_Na23 = '/soft_snc/lib/ace/80/600/110230_80.06c'
ace_Si28 = '/soft_snc/lib/ace/80/600/140280_80.06c'
ace_Si29 = '/soft_snc/lib/ace/80/600/140290_80.06c'
ace_Si30 = '/soft_snc/lib/ace/80/600/140300_80.06c'
ace_P31  = '/soft_snc/lib/ace/80/600/150310_80.06c'
ace_S32  = '/soft_snc/lib/ace/80/600/160320_80.06c'
ace_S33  = '/soft_snc/lib/ace/80/600/160330_80.06c'
ace_S34  = '/soft_snc/lib/ace/80/600/160340_80.06c'
ace_S36  = '/soft_snc/lib/ace/80/600/160360_80.06c'
ace_K39  = '/soft_snc/lib/ace/80/600/190390_80.06c'
ace_K41  = '/soft_snc/lib/ace/80/600/190410_80.06c'
ace_Cr50 = '/soft_snc/lib/ace/80/600/240500_80.06c'
ace_Cr52 = '/soft_snc/lib/ace/80/600/240520_80.06c'
ace_Cr53 = '/soft_snc/lib/ace/80/600/240530_80.06c'
ace_Cr54 = '/soft_snc/lib/ace/80/600/240540_80.06c'
ace_Mn55 = '/soft_snc/lib/ace/80/600/250550_80.06c'
ace_Fe54 = '/soft_snc/lib/ace/80/600/260540_80.06c'
ace_Fe56 = '/soft_snc/lib/ace/80/600/260560_80.06c'
ace_Fe57 = '/soft_snc/lib/ace/80/600/260570_80.06c'
ace_Fe58 = '/soft_snc/lib/ace/80/600/260580_80.06c'
ace_Co59 = '/soft_snc/lib/ace/80/600/270590_80.06c'
ace_Ni58 = '/soft_snc/lib/ace/80/600/280580_80.06c'
ace_Ni60 = '/soft_snc/lib/ace/80/600/280600_80.06c'
ace_Ni61 = '/soft_snc/lib/ace/80/600/280610_80.06c'
ace_Ni62 = '/soft_snc/lib/ace/80/600/280620_80.06c'
ace_Ni64 = '/soft_snc/lib/ace/80/600/280640_80.06c'
ace_Mo92 = '/soft_snc/lib/ace/80/600/420920_80.06c'
ace_Mo94 = '/soft_snc/lib/ace/80/600/420940_80.06c'
ace_Mo95 = '/soft_snc/lib/ace/80/600/420950_80.06c'
ace_Mo96 = '/soft_snc/lib/ace/80/600/420960_80.06c'
ace_Mo97 = '/soft_snc/lib/ace/80/600/420970_80.06c'
ace_Mo98 = '/soft_snc/lib/ace/80/600/420980_80.06c'
ace_Mo100= '/soft_snc/lib/ace/80/600/421000_80.06c'
ace_Cd106= '/soft_snc/lib/ace/80/600/481060_80.06c'
ace_Cd108= '/soft_snc/lib/ace/80/600/481080_80.06c'
ace_Cd110= '/soft_snc/lib/ace/80/600/481100_80.06c'
ace_Cd111= '/soft_snc/lib/ace/80/600/481110_80.06c'
ace_Cd112= '/soft_snc/lib/ace/80/600/481120_80.06c'
ace_Cd113= '/soft_snc/lib/ace/80/600/481130_80.06c'
ace_Cd114= '/soft_snc/lib/ace/80/600/481140_80.06c'
ace_Cd116= '/soft_snc/lib/ace/80/600/481160_80.06c'

acelist = [
    ace_H1, ace_B10, ace_B11, ace_C12, ace_C13, ace_O16, 
    ace_Na23, ace_Si28, ace_Si29, ace_Si30, ace_P31, ace_S32,
    ace_S33, ace_S34, ace_S36, ace_K39, ace_K41, ace_Cr50,
    ace_Cr52, ace_Cr53, ace_Cr54, ace_Mn55, ace_Fe54, ace_Fe56,
    ace_Fe57, ace_Fe58, ace_Co59, ace_Ni58, ace_Ni60, ace_Ni61,
    ace_Ni62, ace_Ni64, ace_Mo92, ace_Mo94, ace_Mo95, ace_Mo96,
    ace_Mo97, ace_Mo98, ace_Mo100, ace_Cd106, ace_Cd108, ace_Cd110,
    ace_Cd111, ace_Cd112, ace_Cd113, ace_Cd114, ace_Cd116
]
           

# 2) Path to the covariance matrix
cov_H1    = '/soft_snc/lib/cov/80/600/10100_80.06.xs.gendf'    
cov_B10   = '/soft_snc/lib/cov/80/600/50100_80.06.xs.gendf'
cov_B11   = '/soft_snc/lib/cov/80/600/50110_80.06.xs.gendf'
cov_C12   = '/soft_snc/lib/cov/80/600/60120_80.06.xs.gendf'
cov_C13   = '/soft_snc/lib/cov/80/600/60130_80.06.xs.gendf'
cov_O16   = '/soft_snc/lib/cov/80/600/80160_80.06.xs.gendf'
cov_Na23  = '/soft_snc/lib/cov/80/600/110230_80.06.xs.gendf'
cov_Si28  = '/soft_snc/lib/cov/80/600/140280_80.06.xs.gendf'
cov_Si29  = '/soft_snc/lib/cov/80/600/140290_80.06.xs.gendf'
cov_Si30  = '/soft_snc/lib/cov/80/600/140300_80.06.xs.gendf'
cov_P31   = '/soft_snc/lib/cov/80/600/150310_80.06.xs.gendf'
cov_S32   = '/soft_snc/lib/cov/80/600/160320_80.06.xs.gendf'
cov_S33   = '/soft_snc/lib/cov/80/600/160330_80.06.xs.gendf'
cov_S34   = '/soft_snc/lib/cov/80/600/160340_80.06.xs.gendf'
cov_S36   = '/soft_snc/lib/cov/80/600/160360_80.06.xs.gendf'
cov_K39   = '/soft_snc/lib/cov/80/600/190390_80.06.xs.gendf'
cov_K41   = '/soft_snc/lib/cov/80/600/190410_80.06.xs.gendf'
cov_Cr50  = '/soft_snc/lib/cov/80/600/240500_80.06.xs.gendf'
cov_Cr52  = '/soft_snc/lib/cov/80/600/240520_80.06.xs.gendf'
cov_Cr53  = '/soft_snc/lib/cov/80/600/240530_80.06.xs.gendf'
cov_Cr54  = '/soft_snc/lib/cov/80/600/240540_80.06.xs.gendf'
cov_Mn55  = '/soft_snc/lib/cov/80/600/250550_80.06.xs.gendf'
cov_Fe54  = '/share_snc/snc/JuanMonleon/COV/cov/80/600/260540_80.06.xs.gendf'
cov_Fe56  = '/soft_snc/lib/cov/80/600/260560_80.06.xs.gendf'
cov_Fe57  = '/soft_snc/lib/cov/80/600/260570_80.06.xs.gendf'
cov_Fe58  = '/soft_snc/lib/cov/80/600/260580_80.06.xs.gendf'
cov_Co59  = '/soft_snc/lib/cov/80/600/270590_80.06.xs.gendf'
cov_Ni58  = '/soft_snc/lib/cov/80/600/280580_80.06.xs.gendf'
cov_Ni60  = '/soft_snc/lib/cov/80/600/280600_80.06.xs.gendf'
cov_Ni61  = '/soft_snc/lib/cov/80/600/280610_80.06.xs.gendf'
cov_Ni62  = '/soft_snc/lib/cov/80/600/280620_80.06.xs.gendf'
cov_Ni64  = '/soft_snc/lib/cov/80/600/280640_80.06.xs.gendf'
cov_Mo92  = '/soft_snc/lib/cov/80/600/420920_80.06.xs.gendf'
cov_Mo94  = '/soft_snc/lib/cov/80/600/420940_80.06.xs.gendf'
cov_Mo95  = '/soft_snc/lib/cov/80/600/420950_80.06.xs.gendf'
cov_Mo96  = '/soft_snc/lib/cov/80/600/420960_80.06.xs.gendf'
cov_Mo97  = '/soft_snc/lib/cov/80/600/420970_80.06.xs.gendf'
cov_Mo98  = '/soft_snc/lib/cov/80/600/420980_80.06.xs.gendf'
cov_Mo100 = '/soft_snc/lib/cov/80/600/421000_80.06.xs.gendf'
cov_Cd106 = '/soft_snc/lib/cov/80/600/481060_80.06.xs.gendf'
cov_Cd108 = '/soft_snc/lib/cov/80/600/481080_80.06.xs.gendf'
cov_Cd110 = '/soft_snc/lib/cov/80/600/481100_80.06.xs.gendf'
cov_Cd111 = '/soft_snc/lib/cov/80/600/481110_80.06.xs.gendf'
cov_Cd112 = '/soft_snc/lib/cov/80/600/481120_80.06.xs.gendf'
cov_Cd113 = '/soft_snc/lib/cov/80/600/481130_80.06.xs.gendf'
cov_Cd114 = '/soft_snc/lib/cov/80/600/481140_80.06.xs.gendf'
cov_Cd116 = '/soft_snc/lib/cov/80/600/481160_80.06.xs.gendf'


cov_paths = [
    cov_H1, cov_B10, cov_B11, cov_C12, cov_C13, cov_O16,
    cov_Na23, cov_Si28, cov_Si29, cov_Si30, cov_P31, cov_S32,
    cov_S33, cov_S34, cov_S36, cov_K39, cov_K41, cov_Cr50,
    cov_Cr52, cov_Cr53, cov_Cr54, cov_Mn55, cov_Fe54, cov_Fe56,
    cov_Fe57, cov_Fe58, cov_Co59, cov_Ni58, cov_Ni60, cov_Ni61,
    cov_Ni62, cov_Ni64, cov_Mo92, cov_Mo94, cov_Mo95, cov_Mo96,
    cov_Mo97, cov_Mo98, cov_Mo100, cov_Cd106,cov_Cd108,cov_Cd110,
    cov_Cd111,cov_Cd112,cov_Cd113,cov_Cd114,cov_Cd116
]

covmatlist = []
missing_paths = []
for cov_path in cov_paths:
    if os.path.exists(cov_path):
        cov = kika.read_njoy_covmat(cov_path)
    else:
        missing_paths.append(cov_path)
        cov = kika.cov.cross_section_covariance.CrossSectionCovariance()
    covmatlist.append(cov)

if missing_paths:
    print("Following covariance matrix files are missing:")
    for path in missing_paths:
        print(f" - {path}")
else:
    print("All covariance matrix files are present.")


mt_numbers   = []
num_samples  = 1024
output_dir   = "/SCRATCH/users/monleon-de-la-jan/KIKA_LIB/ESP"
xsdir_file   = "/soft_snc/lib/xsdir/xsdir80"
seed         = 42
nprocs       = 40

# 2) Call the new perturb_ace_files
print(f"Generating {num_samples} perturbed ACE files...")
perturb_ACE_files(
    ace_files            = acelist,
    covmat               = covmatlist,
    mt_list              = mt_numbers,
    num_samples          = num_samples,
    output_dir           = output_dir,
    xsdir_file           = xsdir_file,
    sampling_method      = 'sobol',
    decomposition_method = 'svd',
    seed                 = seed,
    nprocs               = nprocs,
)
