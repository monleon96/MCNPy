"""The two autofix exceptions, in a module that imports nothing.

They live here rather than in the code that raises them because both
``generators`` (retiring) and ``multigroup_draw`` (replacing it) have to raise
the *same objects* while the two coexist — an ACE call site that catches one
module's ``SoftAutofixWarning`` must catch the other's, or a soft-autofix miss
stops being a skipped isotope and becomes a dead run.

Defining them in either module would close a cycle: ``multigroup_draw`` imports
``core``, ``core`` imports ``generators._uncorrelated``. A leaf with no imports
is the only place both can reach.
"""
from __future__ import annotations

__all__ = [
    "CovarianceFixError",
    "SoftAutofixWarning",
]


class CovarianceFixError(Exception):
    """Autofix could not bring the covariance above the eigenvalue threshold."""


class SoftAutofixWarning(Exception):
    """Soft autofix missed the threshold; the decomposition is tried anyway."""
