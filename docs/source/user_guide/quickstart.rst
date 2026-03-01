Quickstart Guide
================

This guide will help you get started with KIKA by walking through basic usage examples.

Installation
------------

Install KIKA using pip (requires Python 3.12 or later):

.. code-block:: bash

   pip install kika-nd


To upgrade to the latest version:

1. Clear the pip cache (optional but recommended):

   .. code-block:: bash

      pip cache purge

2. Upgrade KIKA:

   .. code-block:: bash

      pip install --upgrade kika-nd

3. Verify the installation:

   .. code-block:: bash

      pip show kika-nd

   This command will display the package information, including the currently installed version.



Basic Usage
-----------

Import the KIKA package:

.. code-block:: python

   import kika

Working with MCNP Input Files
-----------------------------

Parse an MCNP input file:

.. code-block:: python

   from kika import read_mcnp
   
   # Parse an input file
   input_data = read_mcnp('path/to/input_file.i')

   # Print a summary of the input file
   input_data
   
   # Access materials or PERT cards (if available)
   materials = input_data.materials
   perturbations = input_data.perturbation
   
   # Print a summary of materials
   materials

   # Print materials in MCNP format
   print(materials)

   # Print a summary of perturbations
   perturbations



Working with MCTAL Files
------------------------

Parse and analyze MCNP tally files:

.. code-block:: python

   from kika import read_mctal
   
   # Parse a mctal file
   mctal = read_mctal('path/to/mctal_file')
   
   # Print a summary of the mctal file
   mctal

   # Access tallies
   tally = mctal.tally[tally_number]
   
   # Print a summary of the tally
   tally


Working with Sensitivities
--------------------------

Process perturbation data for sensitivity analysis:

.. code-block:: python

   from kika.sensitivities import compute_sensitivity
   
   # Compute sensitivity data
   sensdata = compute_sensitivity(
       input_path='path/to/input.i',
       mctal_path='path/to/mctal',
       tally=4,
       zaid=26056,
       label='Fe-56'
   )
   
   # Print a summary of the sensitivity data
   sensdata



Working with SDF Files
----------------------


Create and process Sensitivity Data Files (SDF) compatible with SCALE:

.. code-block:: python

   from kika.sensitivities.sdf import compute_sensitivity, create_sdf_data
   
   # Read the sensitivity data for each set
   sensdata1 = compute_sensitivity('path/to/input_file1', 'path/to/mctal_file1', 4, [pertuberd_nuclide], 'label-1')
   sensdata2 = compute_sensitivity('path/to/input_file1', 'path/to/mctal_file2', 4, [pertuberd_nuclide], 'label-2')

   # Create a list with the sensitivity data
   sensdata_list = [sensdata1, sensdata2]

   # Print detector energy bins
   sensdata1.energies

   # Create an sdf data object with the list of sensitivity data
   sdf_data = create_sdf_data(sensdata_list, energy='1.00e+00_3.00e+00', title='Example SDF Data')

   # Print a summary of the sdf data
   sdf_data
   
   # Write the sdf data to a sdf format file
   sdf_data.write_file()



Next Steps
----------

For more detailed examples, see the :doc:`tutorials/index`.


Working with ACE Files
----------------------

Parse ACE format nuclear data files:

.. code-block:: python

   from kika import read_ace
   
   # Read an ACE file
   ace_data = read_ace('path/to/ace_file')


Working with Covariance Matrices
--------------------------------

Read and process covariance matrices from SCALE or NJOY:

.. code-block:: python

   from kika import read_coverx, read_covfil

   # Read a COVERX covariance matrix (text or binary, auto-detected)
   covmat = read_coverx('path/to/coverx_file')

   # Read an NJOY COVFIL/GENDF covariance matrix
   covmat = read_covfil('path/to/njoy_covmat')


Working with ENDF Files
-----------------------

Read Evaluated Nuclear Data Files:

.. code-block:: python

   from kika.endf import read_endf
   
   # Read an ENDF file
   endf_data = read_endf('path/to/endf_file')
