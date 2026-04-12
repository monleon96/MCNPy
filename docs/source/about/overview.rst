Overview
========

KIKA is a comprehensive Python toolkit for nuclear data analysis and Monte Carlo simulation support. Originally designed for working with MCNP (Monte Carlo N-Particle) input and output files, it has evolved into a full-featured library covering nuclear data formats, sensitivity analysis, covariance matrices, and more.

The library provides modern, lightweight alternatives to traditional tools, focusing on a clear, class-based design and seamless integration with the Python scientific ecosystem (NumPy, pandas, xarray, matplotlib).

Originally created for personal use, KIKA is now available publicly to support the nuclear engineering and radiation transport community. Development continues according to the author's work requirements. Community contributions are warmly welcomed via GitHub at `https://github.com/juanmonleon/KIKA.git <https://github.com/juanmonleon/KIKA.git>`_.

Current Features
----------------

MCNP Input Processing
~~~~~~~~~~~~~~~~~~~~~

- Parsing MCNP input files for materials and PERT cards
- Conversion between atomic and weight fractions for materials
- Generation and printing of materials in MCNP input format
- PERT card generation for sensitivity analysis

MCTAL Analysis
~~~~~~~~~~~~~~

- Comprehensive parsing of MCNP tally output (MCTAL) files
- Extraction of F4 tally results, including uncertainties
- Access to Tally Fluctuation Chart (TFC) data
- Conversion of tally results to pandas DataFrames
- Visualization of tally convergence

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

- Preparation of MCNP input files for sensitivity calculations using PERT cards
- Processing and analysis of MCNP perturbation calculations
- Generation and visualization of sensitivity profiles
- Creation of Sensitivity Data Files (SDF) compatible with SCALE
- Analysis of linearity and visualization of sensitivity coefficients

ACE File Processing
~~~~~~~~~~~~~~~~~~~

- Parsing of ACE (A Compact ENDF) format nuclear data files
- Reading and processing xsdir files
- Cross-section data extraction and analysis

Covariance Matrix Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Reading covariance matrices from SCALE and NJOY outputs
- Covariance matrix decomposition and analysis
- Heatmap visualization for correlation and covariance data
- Support for MF34 angular distribution covariances

ENDF Data Processing
~~~~~~~~~~~~~~~~~~~~

- Reading and parsing Evaluated Nuclear Data Files (ENDF)
- Support for various ENDF file sections (MF/MT combinations)
- Data extraction and conversion utilities

Energy Group Structures
~~~~~~~~~~~~~~~~~~~~~~~

- Predefined energy group structures commonly used in nuclear analysis
- Support for SCALE, VITAMIN, and other standard group structures

Additional Features
~~~~~~~~~~~~~~~~~~~

- **Serpent**: Support for Serpent Monte Carlo code outputs
- **NJOY**: Integration with NJOY nuclear data processing results
- **Sampling**: Tools for uncertainty quantification through sampling
- **Plotting**: Consistent visualization utilities for nuclear data

