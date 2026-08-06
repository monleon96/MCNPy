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


def urr_cross_sections(E, urr_l_groups, spi, ap, awr):
    """Compute average cross sections from URR parameters.

    Parameters
    ----------
    E : ndarray
        Energy grid in eV.
    urr_l_groups : list of URR_LGroup
        Average resonance parameters grouped by l-value.
        Each LGroup contains j_groups, each with:
          j, d, gn0, gg, gf, gx (scalars or arrays matching E).
    spi : float
        Target spin.
    ap : float
        Scattering radius (ENDF units: 10^{-12} cm).
    awr : float
        Atomic weight ratio.

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

    for lg in urr_l_groups:
        l = lg.l
        awr_l = lg.awri if lg.awri > 0 else awr

        k2_l = wave_number_squared(E, awr_l)
        rho_E_l = compute_rho(E, awr_l, ap)
        P_l_E = penetration_factor(l, rho_E_l)
        phi_l = hard_sphere_phase_shift(l, rho_E_l)
        sin_phi = np.sin(phi_l)

        k2_l_safe = np.where(k2_l > 0, k2_l, 1.0)
        pi_over_k2_l = np.where(k2_l > 0, np.pi / k2_l_safe * 0.01, 0.0)

        # Potential scattering for this l
        sig_el += 4.0 * (2.0 * l + 1.0) * pi_over_k2_l * sin_phi ** 2

        for jg in lg.j_groups:
            g_J = statistical_spin_factor(jg.j, spi)

            # Get average parameters at E (scalar or interpolated)
            D = _as_array(jg.d, nE)
            GN0 = _as_array(jg.gn0, nE)
            GG = _as_array(jg.gg, nE)
            GF = _as_array(jg.gf, nE)
            GX = _as_array(jg.gx, nE)

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


def _as_array(val, n):
    """Convert scalar or array to ndarray of length n."""
    if isinstance(val, np.ndarray):
        return val
    return np.full(n, float(val))
