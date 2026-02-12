# KIKA

[![Version](https://img.shields.io/badge/version-0.1.2-blue.svg)](https://github.com/monleon96/kika)
[![Documentation Status](https://readthedocs.org/projects/kika/badge/?version=latest)](https://kika.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/kika-nd)](https://pypi.org/project/kika-nd/)
[![Python](https://img.shields.io/pypi/pyversions/kika-nd)](https://pypi.org/project/kika-nd/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](https://github.com/monleon96/kika/blob/main/LICENSE)

A comprehensive Python toolkit for nuclear data analysis, Monte Carlo simulation support, and uncertainty quantification. KIKA provides tools for working with MCNP, ENDF, ACE files, covariance matrices, and sensitivity analysis.

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
- **Covariance**: Handle covariance matrices from SCALE and NJOY

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
cov = kika.read_scale_covmat("path/to/covmat_file")
```

## Documentation

For complete documentation, examples, and API reference, visit:
[KIKA Documentation](https://kika.readthedocs.io/en/latest/)

## GUI Application

A standalone GUI application for KIKA is available at [kika-app](https://github.com/monleon96/kika-app).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
