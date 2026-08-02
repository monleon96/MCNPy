kika.benchmarks
===============

Benchmark profiles stored in the schema-v3 SQLite database expose the same
``SensitivityProfile`` representation as SDF inputs. This keeps database I/O
outside the numerical UQ layer.

SQLite sandwich and c-k ranking
-------------------------------

.. code-block:: python

   from kika.benchmarks import (
       benchmark_uncertainty,
       get_sensitivity_profile,
       rank_benchmarks_by_ck,
   )

   benchmark = get_sensitivity_profile(profile_id, db_path="benchmarks.db")
   uncertainty = benchmark_uncertainty(
       profile_id,
       covariance,
       db_path="benchmarks.db",
   )
   ranking = rank_benchmarks_by_ck(
       application,
       covariance,
       db_path="benchmarks.db",
       benchmark_ids=["HEU-COMP-INTER-003-001"],
       limit=20,
   )

Preferred variants are ranked by signed :math:`c_k` in descending order by
default, with benchmark id as the deterministic tie-breaker. Set
``rank_by_absolute=True`` to rank by :math:`|c_k|`. Failed candidates are not
hidden in strict mode.

.. automodule:: kika.benchmarks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: kika.benchmarks.uq
   :members:
   :undoc-members:
   :show-inheritance:
