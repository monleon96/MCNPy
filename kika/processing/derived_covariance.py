"""Resolve ENDF MF33 NC-type (derived) covariances end-to-end.

The sum-rule resolution for a derived reaction (e.g. MT2 of Fe-56 expressed
as MT1 minus every partial) needs three ingredients:

1. The MF33 section of the derived MT, along with every contributor's
   MF33 section — already loaded by :func:`kika.endf.read_endf`.
2. A reaction-indexed map of pointwise cross sections σ(E), with the
   resolved-resonance region **reconstructed** (not just the smooth MF3
   background) so that the relative↔absolute covariance conversion sees
   the actual resonance-broadened σ.
3. The bilinear sum-rule contraction, which
   :meth:`kika.endf.classes.mf33.mf33.MF33MT.resolve_nc_lty0` implements.

This module wires them together. Resonance reconstruction needs a path to
the NJOY binary: pass ``njoy_executable=`` and ``njoy_reconstruct`` runs
RECONR. Without it, the raw MF3 background is used and a warning says so —
for a resolved-resonance MT that is a different cross section, and the
caller should know which one the sum rule was resolved against.

This used to fall back to an in-Python reconstructor, ``endf.reconstruct_xs()``,
whose own docstring said it produced incorrect cross sections on real
evaluations. It has been removed; ``kika.processing.reconstruct`` still exists
and is still under test, but nothing calls it behind a caller's back.

Callers that already have a preferred σ source (a pre-computed PENDF, an
ACE file) can bypass this module and pass the map straight to
``to_xs_covmat(..., mf3_sections=...)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from kika.endf.read_endf import read_endf

_log = logging.getLogger(__name__)


def _build_xs_map(
    endf,
    njoy_executable=None,
    tolerance: float = 1e-3,
    endf_path: Optional[Path] = None,
) -> Dict[int, object]:
    """Return an MT → σ(E) map with resonances reconstructed.

    Entries for MTs that the reconstructor produced (typically MT1, MT2,
    MT102 — i.e. the classical reactions with resonance contributions)
    override the raw MF3 background. Every remaining MF3 MT is included
    verbatim so the sum rule can see partial cross sections that live
    purely above the resonance region.
    """
    xs_map: Dict[int, object] = {}

    mf3 = endf.files.get(3)
    if mf3 is not None:
        xs_map.update(mf3.sections)

    has_resonances = 2 in endf.files
    if not has_resonances:
        _log.debug("No MF2 in file — skipping resonance reconstruction.")
        return xs_map

    if njoy_executable is not None:
        if endf_path is None:
            raise ValueError(
                "njoy_executable was supplied, but no source ENDF path is "
                "available — pass the file path to resolve_derived_covariance() "
                "instead of a pre-parsed ENDF object so NJOY can read it."
            )
        from kika.processing.njoy_reconstruct import njoy_reconstruct
        xs_map.update(
            njoy_reconstruct(
                endf_path=endf_path,
                njoy_executable=njoy_executable,
                tolerance=tolerance,
            )
        )
        return xs_map

    # No njoy_executable. Use whatever the caller already put on endf.pendf,
    # and otherwise the raw MF3 background — which for a resolved-resonance MT
    # is not the same cross section, hence the warning.
    #
    # This used to fall through to endf.reconstruct_xs(), an in-Python
    # reconstructor whose own docstring said it produced incorrect cross
    # sections on real evaluations. Silently resolving a sum rule against
    # numbers known to be wrong is worse than resolving it against a background
    # that is at least honestly labelled.
    pendf = getattr(endf, 'pendf', None)
    if pendf:
        xs_map.update(pendf)
        return xs_map

    _log.warning(
        "MF2 is present but no reconstructed cross sections are available, so "
        "the raw MF3 background is used for the sum rule. Pass "
        "njoy_executable=... to reconstruct with NJOY RECONR, or set "
        "endf.pendf yourself."
    )
    return xs_map


def resolve_derived_covariance(
    endf_or_path: Union[str, Path, object],
    mt: int,
    *,
    njoy_executable: Optional[Union[str, Path]] = None,
    tolerance: float = 1e-3,
    energy_unit: str = 'eV',
):
    """Resolve an NC-type LTY=0 derived covariance on the data's native grid.

    Parameters
    ----------
    endf_or_path : str, Path, or ENDF
        ENDF filesystem path or an already-parsed ENDF object.
    mt : int
        MT number of the derived reaction (e.g. 2 for elastic).
    njoy_executable : str, Path, optional
        Path to the NJOY binary. If provided, NJOY RECONR is used for
        resonance reconstruction; otherwise kika's in-Python reconstructor
        is used.
    tolerance : float
        Linearization tolerance for resonance reconstruction (0.1% default).
    energy_unit : str
        Energy unit of the output grid ('eV' or 'MeV').

    Returns
    -------
    CrossSectionCovariance
        Self-covariance of the derived MT on the union of its
        contributors' coarse MF33 grids. Full matrix — includes
        off-diagonal correlations from the bilinear sum-rule contraction.

    Raises
    ------
    KeyError
        If the ENDF has no MF33 or no section for the requested MT.
    ValueError
        If the requested MT has no NC LTY=0 record (i.e. is not derived).
    """
    if isinstance(endf_or_path, (str, Path)):
        endf_path: Optional[Path] = Path(endf_or_path)
        endf = read_endf(str(endf_path))
    else:
        endf_path = None
        endf = endf_or_path

    mf33 = endf.files.get(33)
    if mf33 is None:
        raise KeyError("ENDF file has no MF33 (covariance) section")

    section = mf33.sections.get(mt)
    if section is None:
        raise KeyError(f"MF33 has no MT{mt} section")

    nc_records = [
        nc for ss in section.subsections for nc in ss.nc_records if nc.lty == 0
    ]
    if not nc_records:
        raise ValueError(
            f"MT{mt} MF33 has no NC LTY=0 record — reaction is not a derived "
            f"covariance. Call section.to_xs_covmat(...) directly instead."
        )

    xs_map = _build_xs_map(
        endf,
        njoy_executable=njoy_executable,
        tolerance=tolerance,
        endf_path=endf_path,
    )

    return section.to_xs_covmat(
        energy_unit=energy_unit,
        sibling_sections=mf33.sections,
        mf3_sections=xs_map,
    )
