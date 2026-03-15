"""
Pointwise cross-section reconstruction from resonance parameters (RECONR).

Public API
----------
reconr(mf2_mt151, mf3_data=None, tolerance=1e-3)
    Reconstruct pointwise cross sections from MF2 resonance parameters.
"""

from .reconr import reconr

__all__ = ["reconr"]
