# KIKA

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/juanmonleon/kika)
[![Documentation Status](https://readthedocs.org/projects/kika/badge/?version=latest)](https://kika.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/kika-nd)](https://pypi.org/project/kika-nd/)
[![Python](https://img.shields.io/pypi/pyversions/kika-nd)](https://pypi.org/project/kika-nd/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](https://github.com/juanmonleon/kika/blob/main/LICENSE)
[![Website](https://img.shields.io/badge/website-kika--app.com-4db8eb)](https://kika-app.com/)

A comprehensive Python toolkit for nuclear data analysis, Monte Carlo simulation support, and uncertainty quantification. KIKA provides tools for working with MCNP, ENDF, ACE files, covariance matrices, and sensitivity analysis, and powers the KIKA desktop workspace.

> **Looking for the desktop application?** Visit [**kika-app.com**](https://kika-app.com/) to download KIKA for Windows, macOS, or Linux and explore the user guides. No Python installation is required.

## Features

### MCNP Processing
- Parse and manipulate MCNP input files (materials, PERT cards)
- Read and analyze MCTAL output files
- Tally data extraction and visualization

### Sensitivity Analysis
- Compute sensitivity data using PERT cards
- Generate and visualize sensitivity profiles
- Create Sensitivity Data Files (SDF) compatible with SCALE

### Nuclear Data
- **ACE**: Parse ACE format nuclear data files
- **ENDF**: Read Evaluated Nuclear Data Files
- **GNDS**: Read and write GNDS 2.0/2.1 — see *What "GNDS support" means here*
- **Covariance**: Handle covariance matrices from SCALE and NJOY

#### What "GNDS support" means here

kika reads and writes GNDS. It does **not** implement GNDS 2.1, and those are
different claims: it covers the parts the ENDF/B-VIII.1 neutron evaluations
use. Rather than leave you to find the edge, the library states it:

```python
>>> import kika.gnds
>>> print(kika.gnds.capabilities().summary())
300 of GNDS's nodes: 134 full, 7 partial, 159 unsupported (17 lost without a report line); and 12 nodes kika names that gnds.xsd does not declare
```

The left-hand column is every element `gnds.xsd` and `covariances.xsd`
declare, so a node kika does not touch is listed as unsupported rather than
being missing from the list. Every row says why, citing a section of the
specification or a line of the source. In short: the covariance chapter (§25)
is complete; the thermal scattering law and the double-differential cross
sections are not read at all.

```python
>>> print(kika.gnds.capabilities(coverage="partial").text())
>>> print(kika.gnds.capabilities(group="thermalScattering").text())
```

`capabilities()` says what the library can lose without opening a file; the
`report` on a suite you read says what your file lost.

### Additional Tools
- Energy group structure definitions
- Serpent Monte Carlo code support
- Uncertainty quantification utilities

## Installation

```bash
pip install kika-nd
```

For development features:

```bash
# Install with development dependencies
pip install kika-nd[dev]

# Install with documentation dependencies
pip install kika-nd[docs]
```

## Quick Start

```python
import kika

# Read an MCNP input file
input_data = kika.read_mcnp("path/to/input_file")

# Read a MCTAL file
mctal = kika.read_mctal("path/to/mctal_file")

# Access materials
materials = input_data.materials

# Compute sensitivity data
sens_data = kika.compute_sensitivity(
    inputfile="path/to/input_file",
    mctalfile="path/to/mctal_file", 
    tally=4, 
    nuclide=26056, 
    label='Sensitivity Fe-56'
)

# Read ACE data
ace_data = kika.read_ace("path/to/ace_file")

# Read covariance matrices
cov = kika.read_coverx("path/to/covmat_file")  # text or binary, auto-detected
```

### SDF uncertainty convention

KIKA follows the SCALE SDF convention: reaction error arrays and e0 are
absolute one-sigma standard deviations. Energy boundaries are represented
internally in MeV and written to SDF files in eV.

```python
# Standard SCALE/KIKA SDF (absolute uncertainties)
sdf = kika.read_sdf("profile.sdf")

# Historical KIKA SDF written with relative uncertainties
legacy = kika.read_sdf("old_profile.sdf", uncertainty_convention="relative")
```

### Sensitivity/covariance alignment and c-k

UQ calculations use a format-neutral `SensitivityProfile`. Alignment is exact
by default: energy grids and units must agree, and missing covariance raises an
actionable error instead of silently reducing the calculation.

```python
from kika.UQ import align_sensitivity_covariance, similarity_ck
import kika.benchmarks as benchmarks

application = kika.read_sdf("application.sdf").to_sensitivity_profile()
benchmark = benchmarks.get_sensitivity_profile(profile_id)

aligned = align_sensitivity_covariance(
    [application, benchmark], covariance,
    alias_policy="tsurfer",       # explicit SCALE/TSURFER aliases
    missing="drop",               # explicit opt-in; inspect aligned.report
)
ck = similarity_ck(application, benchmark, covariance)
ranking = benchmarks.rank_benchmarks_by_ck(
    application, covariance, benchmark_ids=candidate_ids
)
```

Ranking and propagation never condense grids implicitly. Until explicit SDF
condensation is implemented, candidates must use the same grid as the supplied
covariance.

## Documentation

- [Desktop application and workflow guides](https://kika-app.com/docs)
- [Python library documentation and API reference](https://kika.readthedocs.io/en/latest/)
- [Latest desktop installers](https://kika-app.com/#downloads)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
