kika.UQ
========

The UQ package aligns format-neutral sensitivity profiles with covariance
data, propagates nuclear-data uncertainty, and calculates the covariance-
weighted similarity index :math:`c_k`.

Alignment is strict by default. Energy units are converted explicitly, but
energy grids must coincide after conversion. Missing covariance raises
``MissingCovarianceError``; ``missing="drop"`` is an explicit opt-in and the
resulting loss is recorded in ``AlignmentReport``. Grid condensation is not
performed implicitly.

Exact alignment and sandwich propagation
-----------------------------------------

.. code-block:: python

   import kika
   from kika.UQ import align_sensitivity_covariance
   from kika.UQ.sandwich import sandwich_uncertainty_propagation

   profile = kika.read_sdf("application.sdf").to_sensitivity_profile()
   aligned = align_sensitivity_covariance([profile], covariance)
   result = sandwich_uncertainty_propagation(profile, covariance)

   print(result.total_uncertainty)
   print(aligned.report)

TSURFER aliases and incomplete coverage
----------------------------------------

Aliases and exclusions are never automatic. To use the documented TSURFER
mapping and continue with the covered covariance subset:

.. code-block:: python

   aligned = align_sensitivity_covariance(
       [application, benchmark],
       covariance,
       alias_policy="tsurfer",
       missing="drop",
   )

   print(aligned.report.aliases)
   print(aligned.report.dropped)
   print(aligned.report.parameter_coverage)
   print(aligned.report.sensitivity_coverage)

Similarity
----------

.. code-block:: python

   from kika.UQ import similarity_ck

   result = similarity_ck(application, benchmark, covariance)
   print(result.value)
   print(result.reaction_similarity)  # local, non-additive diagnostics

API
---

.. automodule:: kika.UQ.alignment
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: kika.UQ.sandwich
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: kika.UQ.similarity
   :members:
   :undoc-members:
   :show-inheritance:
