kika.sensitivities
===================

The sensitivities module provides functionality for sensitivity analysis and working with perturbation data.

.. automodule:: kika.sensitivities
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------

kika.sensitivities.sensitivity
-------------------------------

.. automodule:: kika.sensitivities.sensitivity
   :members:
   :undoc-members:
   :show-inheritance:

kika.sensitivities.sensitivity_processing
------------------------------------------

.. automodule:: kika.sensitivities.sensitivity_processing
   :members:
   :undoc-members:
   :show-inheritance:

kika.sensitivities.sdf
-----------------------

.. automodule:: kika.sensitivities.sdf
   :members:
   :undoc-members:
   :show-inheritance:


kika.sensitivities.profile
--------------------------

``SensitivityProfile`` and ``SensitivityReaction`` are the validated,
format-neutral inputs used by UQ calculations. Their energy unit is explicit
and reaction uncertainties are absolute one-sigma standard deviations.

.. automodule:: kika.sensitivities.profile
   :members:
   :undoc-members:
   :show-inheritance:

kika.sensitivities.condensation
-------------------------------

Sensitivity condensation is explicit and format-neutral. Exact condensation
requires every target boundary to be present in the source grid; non-nested
projection is deliberately rejected. The returned report records the boundary
mapping, uncertainty assumption and integral-conservation diagnostic.

.. automodule:: kika.sensitivities.condensation
   :members:
   :undoc-members:
   :show-inheritance:
