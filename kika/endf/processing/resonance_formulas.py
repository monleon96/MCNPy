"""
Resonance cross-section formulas: SLBW, MLBW, Reich-Moore.

Each function computes partial cross sections for a single l-value block
at an array of energies E. Returns (sigma_elastic, sigma_capture, sigma_fission)
in barns.

All formulas follow ENDF-102 manual, Sections D.1.1 (SLBW), D.1.2 (MLBW),
and D.1.3 (Reich-Moore).
"""

import numpy as np
from .penetration import (
    wave_number_squared,
    penetration_factor,
    shift_factor,
    hard_sphere_phase_shift,
    rho as compute_rho,
)


def statistical_spin_factor(J: float, spi: float) -> float:
    """Statistical spin factor g_J = (2J+1) / (2(2I+1))."""
    return (2.0 * abs(J) + 1.0) / (2.0 * (2.0 * spi + 1.0))


def _energy_dependent_neutron_width(Gn_r, E, Er, l, rho_E, rho_Er):
    """Energy-dependent neutron width Gamma_n(E).

    Gamma_n(E) = Gamma_n(E_r) * sqrt(E/|E_r|) * P_l(rho_E) / P_l(rho_Er)
    """
    abs_Er = abs(Er)
    if abs_Er < 1e-30:
        return np.zeros_like(E)

    P_l_E = penetration_factor(l, rho_E)
    P_l_Er = penetration_factor(l, rho_Er)

    P_l_Er_safe = np.where(P_l_Er > 0, P_l_Er, 1.0)
    ratio = np.where(P_l_Er > 0, P_l_E / P_l_Er_safe, 0.0)

    sqrt_ratio = np.sqrt(np.abs(E / abs_Er))
    return abs(Gn_r) * sqrt_ratio * ratio


def _shifted_resonance_energy(Er, l, Gn_r, S_l_E, S_l_Er, P_l_Er):
    """Shifted resonance energy accounting for shift factor."""
    if l == 0:
        return np.full_like(S_l_E, Er)

    P_safe = np.where(P_l_Er > 0, P_l_Er, 1.0)
    shift = np.where(
        P_l_Er > 0,
        (S_l_Er - S_l_E) * abs(Gn_r) / (2.0 * P_safe),
        0.0,
    )
    return Er + shift


# ======================================================================
# SLBW (Single-Level Breit-Wigner), LRF=1
# ======================================================================

def slbw_cross_sections(E, resonances, l, spi, ap, awr, ap_table=None):
    """Compute SLBW cross sections for one l-value block.

    Parameters
    ----------
    E : array-like
        Energies in eV (must be > 0).
    resonances : list of Resonance
        For SLBW: c3=GT, c4=GN, c5=GG, c6=GF.
    l : int
        Orbital angular momentum.
    spi : float
        Target spin.
    ap : float
        Scattering radius (ENDF units: 10^{-12} cm).
    awr : float
        Atomic weight ratio.
    ap_table : tuple, optional
        ``(energies, ap_values, interp_regions)`` for NRO=1.

    Returns
    -------
    sigma_elastic, sigma_capture, sigma_fission : ndarray
    """
    E = np.asarray(E, dtype=float)
    nE = len(E)
    k2 = wave_number_squared(E, awr)
    if ap_table is not None:
        from .penetration import rho_energy_dependent
        rho_E = rho_energy_dependent(E, awr, *ap_table)
    else:
        rho_E = compute_rho(E, awr, ap)
    phi_l = hard_sphere_phase_shift(l, rho_E)
    sin_phi = np.sin(phi_l)
    cos_phi = np.cos(phi_l)

    k2_safe = np.where(k2 > 0, k2, 1.0)
    pi_over_k2 = np.where(k2 > 0, np.pi / k2_safe * 0.01, 0.0)  # barns

    sigma_el = np.zeros(nE)
    sigma_cap = np.zeros(nE)
    sigma_fis = np.zeros(nE)

    pot_scat = 4.0 * (2.0 * l + 1.0) * pi_over_k2 * sin_phi ** 2

    for res in resonances:
        Er = res.energy
        J = res.spin
        GN_r = res.c4
        GG = res.c5
        GF = abs(res.c6)

        g_J = statistical_spin_factor(J, spi)
        abs_Er = abs(Er)
        if abs_Er < 1e-30:
            continue

        rho_Er = compute_rho(np.array([abs_Er]), awr, ap)
        P_l_Er = penetration_factor(l, rho_Er)
        S_l_Er = shift_factor(l, rho_Er)
        S_l_E = shift_factor(l, rho_E)

        GN = _energy_dependent_neutron_width(GN_r, E, Er, l, rho_E, rho_Er)
        GT_E = GN + GG + GF
        Er_shifted = _shifted_resonance_energy(Er, l, GN_r, S_l_E, S_l_Er, P_l_Er)

        denom = (E - Er_shifted) ** 2 + (GT_E / 2.0) ** 2
        denom_safe = np.where(denom > 0, denom, 1.0)

        psi = (GT_E / 2.0) / denom_safe
        chi = (E - Er_shifted) / denom_safe

        sigma_cap += pi_over_k2 * g_J * GN * GG * psi
        sigma_fis += pi_over_k2 * g_J * GN * GF * psi

        # Elastic: resonance + interference with potential
        sigma_el += pi_over_k2 * g_J * (
            GN ** 2 * psi / np.where(GT_E > 0, GT_E / 2.0, 1.0)
            + 2.0 * GN * sin_phi ** 2 * chi
        )

    sigma_el += pot_scat
    return np.maximum(sigma_el, 0.0), np.maximum(sigma_cap, 0.0), np.maximum(sigma_fis, 0.0)


# ======================================================================
# MLBW (Multi-Level Breit-Wigner), LRF=2
# ======================================================================

def mlbw_cross_sections(E, resonances, l, spi, ap, awr, ap_table=None):
    """Compute MLBW cross sections for one l-value block.

    For MLBW: c3=GT, c4=GN, c5=GG, c6=GF.
    Multi-level interference in elastic channel between resonances of same J.
    """
    E = np.asarray(E, dtype=float)
    nE = len(E)
    k2 = wave_number_squared(E, awr)
    if ap_table is not None:
        from .penetration import rho_energy_dependent
        rho_E = rho_energy_dependent(E, awr, *ap_table)
    else:
        rho_E = compute_rho(E, awr, ap)
    phi_l = hard_sphere_phase_shift(l, rho_E)
    sin_phi = np.sin(phi_l)
    cos_phi = np.cos(phi_l)

    k2_safe = np.where(k2 > 0, k2, 1.0)
    pi_over_k2 = np.where(k2 > 0, np.pi / k2_safe * 0.01, 0.0)

    sigma_el = np.zeros(nE)
    sigma_cap = np.zeros(nE)
    sigma_fis = np.zeros(nE)

    pot_scat = 4.0 * (2.0 * l + 1.0) * pi_over_k2 * sin_phi ** 2

    # Group resonances by J for interference
    j_groups = {}
    for res in resonances:
        j_groups.setdefault(res.spin, []).append(res)

    for J, res_group in j_groups.items():
        g_J = statistical_spin_factor(J, spi)
        nres = len(res_group)

        # Accumulate sum1 = Sigma_r GN_r * psi_r, sum2 = Sigma_r GN_r * chi_r
        sum1 = np.zeros(nE)
        sum2 = np.zeros(nE)

        for res in res_group:
            Er = res.energy
            GN_r = res.c4
            GG = res.c5
            GF = abs(res.c6)

            abs_Er = abs(Er)
            if abs_Er < 1e-30:
                continue

            rho_Er = compute_rho(np.array([abs_Er]), awr, ap)
            P_l_Er = penetration_factor(l, rho_Er)
            S_l_Er = shift_factor(l, rho_Er)
            S_l_E = shift_factor(l, rho_E)

            GN = _energy_dependent_neutron_width(GN_r, E, Er, l, rho_E, rho_Er)
            GT_E = GN + GG + GF
            Er_shifted = _shifted_resonance_energy(Er, l, GN_r, S_l_E, S_l_Er, P_l_Er)

            denom = (E - Er_shifted) ** 2 + (GT_E / 2.0) ** 2
            denom_safe = np.where(denom > 0, denom, 1.0)

            psi = (GT_E / 2.0) / denom_safe
            chi = (E - Er_shifted) / denom_safe

            # Capture and fission (no multi-level interference)
            sigma_cap += pi_over_k2 * g_J * GN * GG * psi
            sigma_fis += pi_over_k2 * g_J * GN * GF * psi

            sum1 += GN * psi
            sum2 += GN * chi

        # MLBW elastic with interference
        cos2phi = np.cos(2.0 * phi_l)
        sin2phi = np.sin(2.0 * phi_l)

        A = sum1 * cos2phi + sum2 * sin2phi
        B = sum2 * cos2phi - sum1 * sin2phi

        sigma_el += pi_over_k2 * g_J * (
            A ** 2 + B ** 2
            + 4.0 * A * sin_phi ** 2
            + 4.0 * B * sin_phi * cos_phi
        )

    sigma_el += pot_scat
    return np.maximum(sigma_el, 0.0), np.maximum(sigma_cap, 0.0), np.maximum(sigma_fis, 0.0)


# ======================================================================
# Reich-Moore (LRF=3) — vectorized over energies
# ======================================================================

def reich_moore_cross_sections(E, resonances, l, spi, ap, awr, ap_table=None):
    """Compute Reich-Moore cross sections for one l-value block.

    For Reich-Moore: c3=GN, c4=GG, c5=GFA, c6=GFB.
    The capture channel is eliminated from the R-matrix.

    Uses the collision matrix U approach, fully vectorized over energies.
    Channels: 0=elastic, 1=fission_a, 2=fission_b.

    The R-matrix uses reduced width amplitudes evaluated at E_r (constant).
    Energy dependence enters through P_l(E) and phi_l(E) in the collision
    matrix formula, following ENDF-102 Section D.1.3.
    """
    E = np.asarray(E, dtype=float)
    nE = len(E)
    k2 = wave_number_squared(E, awr)
    if ap_table is not None:
        from .penetration import rho_energy_dependent
        rho_E = rho_energy_dependent(E, awr, *ap_table)
    else:
        rho_E = compute_rho(E, awr, ap)
    phi_l = hard_sphere_phase_shift(l, rho_E)

    k2_safe = np.where(k2 > 0, k2, 1.0)
    pi_over_k2 = np.where(k2 > 0, np.pi / k2_safe * 0.01, 0.0)

    sigma_el = np.zeros(nE)
    sigma_cap = np.zeros(nE)
    sigma_fis = np.zeros(nE)

    P_l_E = penetration_factor(l, rho_E)
    S_l_E = shift_factor(l, rho_E)

    # Group resonances by J
    j_groups = {}
    for res in resonances:
        j_groups.setdefault(res.spin, []).append(res)

    for J, res_group in j_groups.items():
        g_J = statistical_spin_factor(J, spi)
        has_fission = any(abs(r.c5) > 0 or abs(r.c6) > 0 for r in res_group)
        nch = 3 if has_fission else 1

        # Build R-matrix: R[c,c'](E) arrays of shape (nE,)
        # R_{cc'} = sum_r gamma_{rc} * gamma_{rc'} / (E_r - E - i*GG_r/2)
        # gamma are reduced width amplitudes at E_r (CONSTANT, not energy-dependent)
        # gamma_n = sign(GN_r) * sqrt(|GN_r| / (2*P_l(|E_r|)))
        # gamma_f = sign(GF_r) * sqrt(|GF_r| / 2)   (P_f = 1 for fission)
        R = np.zeros((nch, nch, nE), dtype=complex)

        for res in res_group:
            Er = res.energy
            GN_r = res.c3   # RM: c3=GN (neutron width at E_r)
            GG = res.c4     # RM: c4=GG (gamma width)
            GFA = res.c5    # RM: c5=GFA
            GFB = res.c6    # RM: c6=GFB

            abs_Er = abs(Er)
            if abs_Er < 1e-30:
                continue

            rho_Er = compute_rho(np.array([abs_Er]), awr, ap)
            P_l_Er = penetration_factor(l, rho_Er)[0]

            # Reduced width amplitude for neutron channel (constant, at E_r)
            P_l_Er_safe = max(P_l_Er, 1e-30)
            sign_n = 1.0 if GN_r >= 0 else -1.0
            gamma_n = sign_n * np.sqrt(abs(GN_r) / (2.0 * P_l_Er_safe))

            # Shifted resonance energy (accounts for shift factor difference)
            S_l_Er = shift_factor(l, rho_Er)[0]
            Er_shift = Er + (S_l_Er - S_l_E) * gamma_n ** 2

            # Denominator: E_r' - E - i*GG/2
            inv_denom = 1.0 / (Er_shift - E - 1j * GG / 2.0)  # (nE,)

            # R_nn += gamma_n^2 / denom
            R[0, 0] += gamma_n ** 2 * inv_denom

            if has_fission:
                sign_fa = 1.0 if GFA >= 0 else -1.0
                sign_fb = 1.0 if GFB >= 0 else -1.0
                gamma_fa = sign_fa * np.sqrt(abs(GFA) / 2.0)
                gamma_fb = sign_fb * np.sqrt(abs(GFB) / 2.0)

                R[0, 1] += gamma_n * gamma_fa * inv_denom
                R[0, 2] += gamma_n * gamma_fb * inv_denom
                R[1, 0] += gamma_fa * gamma_n * inv_denom
                R[1, 1] += gamma_fa ** 2 * inv_denom
                R[1, 2] += gamma_fa * gamma_fb * inv_denom
                R[2, 0] += gamma_fb * gamma_n * inv_denom
                R[2, 1] += gamma_fb * gamma_fa * inv_denom
                R[2, 2] += gamma_fb ** 2 * inv_denom

        # Collision matrix (exact R-matrix formula):
        # U = Omega * (I + 2i * P^{1/2} * W * P^{1/2}) * Omega
        # where W = R * (I - L0*R)^{-1}
        # L0 = diag(S_c - B_c + i*P_c) for each channel
        # B_c = S_c(E_r) for ENDF convention, but since we already applied
        # the shift factor correction to E_r, B = 0 for l=0, and for general l
        # the shift is already absorbed into Er_shift above (B = S_l(E_r)).
        # So effectively L0_n = S_l(E) - S_l(E_r) + i*P_l(E).
        # Since Er_shift already accounts for S_l(E)-S_l(E_r) in the denominator,
        # we use L0_n = i*P_l(E) (the shift part is already in the R-matrix).
        #
        # For fission channels: L0_f = i*1 (P=1, S=0, B=0)
        # For neutron channel: L0_n = i*P_l(E) (with boundary condition absorbed)

        sqrt_P = np.zeros((nch, nE))
        sqrt_P[0] = np.sqrt(np.maximum(P_l_E, 0.0))
        if has_fission:
            sqrt_P[1] = 1.0
            sqrt_P[2] = 1.0

        # Build L0 diagonal (nch, nE)
        L0 = np.zeros((nch, nE), dtype=complex)
        L0[0] = 1j * P_l_E
        if has_fission:
            L0[1] = 1j
            L0[2] = 1j

        # Compute (I - L0*R)^{-1} for each energy point
        # L0*R: (L0)_{cc} * R_{cc'} — L0 is diagonal, so this scales rows
        L0R = np.zeros_like(R)
        for c in range(nch):
            L0R[c, :] = L0[c] * R[c, :]

        # I - L0*R
        IminusL0R = -L0R.copy()
        for c in range(nch):
            IminusL0R[c, c] += 1.0

        # W = R * (I - L0*R)^{-1}: invert nch×nch at each energy
        # For nch=1: W = R / (1 - L0*R), simple scalar division
        # For nch=3: use explicit 3x3 inverse
        if nch == 1:
            denom_inv = 1.0 / IminusL0R[0, 0]  # (nE,) complex
            W = np.zeros_like(R)
            W[0, 0] = R[0, 0] * denom_inv
        else:
            # 3x3 matrix inverse per energy point via Cramer's rule
            W = np.zeros_like(R)
            for ie in range(nE):
                M = IminusL0R[:, :, ie]
                try:
                    M_inv = np.linalg.inv(M)
                except np.linalg.LinAlgError:
                    continue
                W[:, :, ie] = R[:, :, ie] @ M_inv

        # U = I + 2i * P^{1/2} * W * P^{1/2}
        # PWP[c,c'] = sqrt_P[c] * W[c,c'] * sqrt_P[c']
        PWP = np.copy(W)
        for c in range(nch):
            PWP[c, :] *= sqrt_P[c]
            PWP[:, c] *= sqrt_P[c]

        U = 2j * PWP
        for c in range(nch):
            U[c, c] += 1.0

        # Apply phase shift: U -> Omega * U * Omega
        e_neg_iphi = np.exp(-1j * phi_l)  # (nE,)
        U[0, :] *= e_neg_iphi
        U[:, 0] *= e_neg_iphi

        # Cross sections from collision matrix
        # sigma_elastic = pi/k^2 * g_J * |1 - U_nn|^2
        sigma_el += pi_over_k2 * g_J * np.abs(1.0 - U[0, 0]) ** 2

        if has_fission:
            sigma_fis += pi_over_k2 * g_J * (
                np.abs(U[0, 1]) ** 2 + np.abs(U[0, 2]) ** 2
            )

        # sigma_total = 2*pi/k^2 * g_J * (1 - Re(U_nn))
        sigma_tot_J = pi_over_k2 * g_J * 2.0 * (1.0 - U[0, 0].real)

        # sigma_capture = sigma_total - sigma_elastic - sigma_fission (from unitarity)
        el_J = pi_over_k2 * g_J * np.abs(1.0 - U[0, 0]) ** 2
        fis_J = 0.0
        if has_fission:
            fis_J = pi_over_k2 * g_J * (
                np.abs(U[0, 1]) ** 2 + np.abs(U[0, 2]) ** 2
            )
        sigma_cap += np.maximum(sigma_tot_J - el_J - fis_J, 0.0)

    return np.maximum(sigma_el, 0.0), np.maximum(sigma_cap, 0.0), np.maximum(sigma_fis, 0.0)
