"""
Resonance reconstruction (ENDF adapter).

Thin wrapper around ``kika.processing.reconstruct`` that accepts and returns
ENDF-specific types (``MF2MT151``, ``MF3MT``).  The actual physics
lives in ``kika.processing.reconstruct``.

Supported formalisms
--------------------
- LRF=1  SLBW  (Single-Level Breit-Wigner)
- LRF=2  MLBW  (Multi-Level Breit-Wigner)
- LRF=3  RM    (Reich-Moore, eliminated capture channel)

Only resolved resonance ranges (LRU=1) are processed.
"""

from typing import Dict, Optional

from ..classes.mf2.mf2mt151 import MF2MT151
from ..classes.mf3.mf3mt import MF3MT


def reconstruct(mf2_mt151: MF2MT151,
           mf3_data=None,
           tolerance: float = 1e-3) -> Dict[int, MF3MT]:
    """Reconstruct pointwise cross sections from MF2 resonance parameters.

    Parameters
    ----------
    mf2_mt151 : MF2MT151
        Parsed MF2/MT151 section.
    mf3_data : MF (MF3 file object), optional
        MF3 background cross-section data.  If provided, the smooth
        background is added to the reconstructed resonance cross sections
        inside the resonance region and preserved outside it.
    tolerance : float
        Linearization tolerance (default 0.1%).

    Returns
    -------
    Dict[int, MF3MT]
        Reconstructed pointwise cross sections keyed by MT number.
        Typically contains MT 1 (total), 2 (elastic), 18 (fission), 102 (capture).
    """
    # Lazy imports to avoid circular dependency at module load time
    from kika.nuclear_data import ResonanceParameters, CrossSection
    from kika.processing.reconstruct import reconstruct as _reconstruct

    # Convert ENDF types → canonical types
    res_params = ResonanceParameters.from_endf(mf2_mt151)

    background = None
    if mf3_data is not None:
        background = {
            mt: CrossSection.from_endf(section)
            for mt, section in mf3_data.sections.items()
        }

    # Delegate to format-independent processing
    canonical_result = _reconstruct(res_params, background, tolerance=tolerance)

    # Convert canonical types → ENDF types
    return {mt: xs.to_endf() for mt, xs in canonical_result.items()}
