"""
Unresolved resonance region (URR) average cross sections.

Computes infinitely-dilute average cross sections from average resonance
parameters following ENDF-102, Section D.2.
"""

import numpy as np
from .penetration import (
    wave_number_squared,
    penetration_factor,
    hard_sphere_phase_shift,
    rho as compute_rho,
)
from .resonance_formulas import statistical_spin_factor


def urr_cross_sections(E, spinGroups, spi, ap, awr):
    """Compute average cross sections from URR parameters.

    Parameters
    ----------
    E : ndarray
        Energy grid in eV.
    spinGroups : list of tabulated_widths.UnresolvedSpinGroup
        The model's unresolved spin groups — one per ``(L, J)``, with the
        averages already on *E*. Widths are read from named channels
        (``neutron``, ``capture``, ``fission``, ``competitive``) rather than
        from fixed attributes, which is the same move the resolved formulas
        make; ENDF's degrees-of-freedom counts ride on the channels they
        belong to.
    spi : float
        Target spin.
    ap : float
        Scattering radius (ENDF units: 10^{-12} cm).
    awr : float
        Atomic weight ratio, used for a group that declares none.

    Returns
    -------
    sig_el, sig_cap, sig_fis : ndarray
        Average cross sections in barns.
    """
    E = np.asarray(E, dtype=float)
    nE = len(E)

    sig_el = np.zeros(nE)
    sig_cap = np.zeros(nE)
    sig_fis = np.zeros(nE)

    for l, groups in _byOrbitalMomentum(spinGroups):
        # The potential-scattering term belongs to the l-block, so it is added
        # once per l and not once per (L, J) — which is why the groups are
        # blocked back up here rather than iterated flat.
        awri = groups[0].atomicWeightRatio
        awr_l = awri if awri and awri > 0 else awr

        k2_l = wave_number_squared(E, awr_l)
        rho_E_l = compute_rho(E, awr_l, ap)
        P_l_E = penetration_factor(l, rho_E_l)
        phi_l = hard_sphere_phase_shift(l, rho_E_l)
        sin_phi = np.sin(phi_l)

        k2_l_safe = np.where(k2_l > 0, k2_l, 1.0)
        pi_over_k2_l = np.where(k2_l > 0, np.pi / k2_l_safe * 0.01, 0.0)

        # Potential scattering for this l
        sig_el += 4.0 * (2.0 * l + 1.0) * pi_over_k2_l * sin_phi ** 2

        for group in groups:
            g_J = statistical_spin_factor(group.J, spi)

            # Get average parameters at E (scalar or interpolated)
            channels = {channel.label: channel for channel in group.channels}
            D = _as_array(_levelSpacing(group), nE)
            GN0 = _as_array(_width(channels.get("neutron")), nE)
            GG = _as_array(_width(channels.get("capture")), nE)
            GF = _as_array(_width(channels.get("fission")), nE)
            GX = _as_array(_width(channels.get("competitive")), nE)

            # Energy-dependent average neutron width
            # <Gamma_n>(E) = GN0 * sqrt(E) * P_l(rho_E)
            Gn_avg = GN0 * np.sqrt(E) * P_l_E

            # Average total width
            Gtot = Gn_avg + GG + GF + GX

            # Avoid division by zero
            safe = (Gtot > 0) & (D > 0)
            factor = np.where(safe, 2.0 * np.pi * Gn_avg / (Gtot * D), 0.0)

            # Cross sections per channel
            sig_el += pi_over_k2_l * g_J * factor * Gn_avg
            sig_cap += pi_over_k2_l * g_J * factor * GG
            sig_fis += pi_over_k2_l * g_J * factor * GF

    return np.maximum(sig_el, 0.0), np.maximum(sig_cap, 0.0), np.maximum(sig_fis, 0.0)


def _byOrbitalMomentum(spinGroups):
    """``[(L, [groups])]`` in ascending L — ENDF's l-blocks, rebuilt.

    The model keeps one spin group per ``(L, J)``, which is what the data is;
    ENDF nests J inside an l-block. Sorting by L rather than keeping first
    appearance is not arbitrary: it is what ``model.interop`` already does when
    it projects the same node back to the flat classes, so the two paths visit
    the blocks in the same order and sum the same series in the same sequence.
    """
    byL: dict = {}
    for group in spinGroups:
        byL.setdefault(group.L, []).append(group)
    return [(L, byL[L]) for L in sorted(byL)]


def _levelSpacing(group):
    """A one-element array is ENDF case A's scalar; anything longer is a table."""
    values = group.levelSpacing
    if values is None:
        return 0.0
    array = np.asarray(values, dtype=float)
    return float(array[0]) if array.size == 1 else array


def _width(channel):
    """The channel's average width, or zero for a channel the group has not got."""
    if channel is None:
        return 0.0
    if channel.widths is not None:
        return channel.widths
    return channel.constantWidth if channel.constantWidth is not None else 0.0


def _as_array(val, n):
    """Convert scalar or array to ndarray of length n."""
    if isinstance(val, np.ndarray):
        return val
    return np.full(n, float(val))
