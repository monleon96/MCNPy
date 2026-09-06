API Reference
=============

This section provides detailed documentation of the KIKA API.

Package Overview
----------------

KIKA is structured into several subpackages:

* **kika.input**: For working with MCNP input files (materials, PERT cards)
* **kika.mctal**: For processing MCNP tally output files
* **kika.sensitivities**: For sensitivity analysis and SDF file generation
* **kika.ace**: For parsing ACE format nuclear data files
* **kika.cov**: For covariance matrix handling and visualization
* **kika.UQ**: For strict sensitivity/covariance alignment, sandwich propagation, and c-k
* **kika.benchmarks**: For DICE/ICSBEP SQLite profiles and benchmark ranking
* **kika.endf**: For reading Evaluated Nuclear Data Files (ENDF)
* **kika.energy_grids**: Predefined energy group structures

Module Reference
----------------

.. toctree::
   :maxdepth: 2
   
   input
   mctal
   sensitivities
   ace
   cov
   uq
   benchmarks
   endf
