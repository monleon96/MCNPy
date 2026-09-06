"""
Penetration factors, shift factors, and phase shifts for neutron channels.

All functions are pure math operating on NumPy arrays.
No Coulomb barriers — neutrons only.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (ENDF conventions, neutron-only)
# ---------------------------------------------------------------------------

# Neutron mass in amu (unified atomic mass units)
from kika._constants import NEUTRON_MASS_AMU  # noqa: F401

# (ħc)² in eV²·barn  (1 barn = 1e-24 cm²)
# ħc = 197.3269804 MeV·fm = 197.3269804e6 eV · 1e-13 cm
# (ħc)² = (197.3269804)² MeV²·fm² = 38937.92957 ... eV² * 1e12 * 1e-24 cm² = ... barn·eV²
# More directly: ħ²/(2m_n) = 2.0723 e-3 eV·barn  (standard ENDF value)
# k² = E / (ħ²/(2μ))  where μ = reduced mass
# Using ENDF convention: k² = AWR/(1+AWR) * E / E_0
# where E_0 = (ħc)² / (2 * m_n * c²) expressed so that k is in 1/fm
#
# Standard ENDF formula (ENDF-102, Appendix H):
#   k² [1/fm²] = (2 * m_n * c²) / (ħc)² * (AWR/(AWR+1)) * E [eV]
#   m_n * c² = 939.56542052e6 eV
#   (ħc)² = (197.3269804)² MeV²·fm² = 38937.92957 MeV²·fm²
#   => 2*m_n*c² / (ħc)² = 2 * 939.56542052 / 38937.92957 = 0.048254942 1/(MeV·fm²)
#   = 4.8254942e-8 1/(eV·fm²)

_TWO_MN_OVER_HBAR2 = 2.0 * 939.56542052e6 / (197.3269804**2 * 1e12)
# = 4.8254942e-8  [1/(eV·fm²)]


def wave_number_squared(E: np.ndarray, awr: float) -> np.ndarray:
    """Compute k² in 1/fm² for neutron with lab energy E (eV).

    Uses the ENDF convention with reduced mass correction:
        k² = (2·m_n/ħ²) · (AWR/(AWR+1)) · E

    Parameters
    ----------
    E : array
        Lab-frame neutron energy in eV (must be > 0).
    awr : float
        Atomic weight ratio (target mass / neutron mass).

    Returns
    -------
    array
        k² in 1/fm².
    """
    return _TWO_MN_OVER_HBAR2 * (awr / (awr + 1.0)) * E


def wave_number(E: np.ndarray, awr: float) -> np.ndarray:
    """Compute k in 1/fm."""
    return np.sqrt(wave_number_squared(E, awr))


def rho(E: np.ndarray, awr: float, ap: float) -> np.ndarray:
    """Compute rho = k*a where a = scattering radius.

    Parameters
    ----------
    E : array
        Lab-frame neutron energy in eV.
    awr : float
        Atomic weight ratio.
    ap : float
        Scattering radius in ENDF units of 10^{-12} cm.
        Converted internally: a [fm] = ap * 10.
    """
    # ENDF stores AP in 10^{-12} cm.  1 fm = 10^{-13} cm, so a[fm] = AP * 10.
    a_fm = ap * 10.0
    return wave_number(E, awr) * a_fm


def rho_energy_dependent(E: np.ndarray, awr: float,
                         ap_energies: np.ndarray, ap_values: np.ndarray,
                         interp_regions: list) -> np.ndarray:
    """Compute rho = k(E) * a(E) with energy-dependent scattering radius.

    Parameters
    ----------
    E : array
        Lab-frame neutron energy in eV.
    awr : float
        Atomic weight ratio.
    ap_energies, ap_values : array
        TAB1 data for AP(E) in ENDF units (10^{-12} cm).
    interp_regions : list of (NBT, INT) tuples
        ENDF interpolation regions for AP(E).
    """
    from .interpolation import interpolate_1d

    ap_E = interpolate_1d(ap_energies, ap_values, interp_regions,
                                E, out_of_range="hold")
    a_fm = np.asarray(ap_E) * 10.0  # 10^{-12} cm -> fm
    k = wave_number(E, awr)
    return k * a_fm


# ---------------------------------------------------------------------------
# Penetration factor P_ℓ(ρ) and shift factor S_ℓ(ρ)
# ---------------------------------------------------------------------------
# Recursive formulae from ENDF-102 manual, Section D.1.3
# For neutrons (no Coulomb):
#   P_0 = ρ,  S_0 = 0
#   P_1 = ρ³/(1+ρ²),  S_1 = -1/(1+ρ²)
#   General recursion:
#     D_ℓ = (ℓ - S_{ℓ-1})² + P_{ℓ-1}²
#     P_ℓ = ρ² · P_{ℓ-1} / D_ℓ
#     S_ℓ = ρ² · (ℓ - S_{ℓ-1}) / D_ℓ - ℓ
# ---------------------------------------------------------------------------

def penetration_factor(l: int, rho_val: np.ndarray) -> np.ndarray:
    """Penetration factor P_ℓ(ρ) for neutrons.

    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number (0, 1, 2, ...).
    rho_val : array
        ρ = k·a (dimensionless).

    Returns
    -------
    array
        P_ℓ values, same shape as rho_val.
    """
    rho2 = rho_val ** 2
    if l == 0:
        return rho_val.copy()

    # Start recursion from l=0
    P_prev = rho_val.copy()  # P_0
    S_prev = np.zeros_like(rho_val)  # S_0

    for ll in range(1, l + 1):
        D = (ll - S_prev) ** 2 + P_prev ** 2
        # Guard against D=0 (happens at E→0)
        D_safe = np.where(D > 0, D, 1.0)
        P_curr = rho2 * P_prev / D_safe
        S_curr = rho2 * (ll - S_prev) / D_safe - ll
        # Where D was zero, P and S should be zero
        P_curr = np.where(D > 0, P_curr, 0.0)
        S_curr = np.where(D > 0, S_curr, -float(ll))
        P_prev = P_curr
        S_prev = S_curr

    return P_prev


def shift_factor(l: int, rho_val: np.ndarray) -> np.ndarray:
    """Shift factor S_ℓ(ρ) for neutrons.

    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number.
    rho_val : array
        ρ = k·a (dimensionless).

    Returns
    -------
    array
        S_ℓ values.
    """
    rho2 = rho_val ** 2
    if l == 0:
        return np.zeros_like(rho_val)

    P_prev = rho_val.copy()
    S_prev = np.zeros_like(rho_val)

    for ll in range(1, l + 1):
        D = (ll - S_prev) ** 2 + P_prev ** 2
        D_safe = np.where(D > 0, D, 1.0)
        P_curr = rho2 * P_prev / D_safe
        S_curr = rho2 * (ll - S_prev) / D_safe - ll
        P_curr = np.where(D > 0, P_curr, 0.0)
        S_curr = np.where(D > 0, S_curr, -float(ll))
        P_prev = P_curr
        S_prev = S_curr

    return S_prev


def hard_sphere_phase_shift(l: int, rho_val: np.ndarray) -> np.ndarray:
    """Hard-sphere phase shift φ_ℓ(ρ) for neutrons.

    φ_ℓ = atan(P_ℓ / (ℓ - S_ℓ))   with special case φ_0 = ρ

    Note: this is the *potential scattering* phase shift used to compute
    sin²φ_ℓ and cos(2φ_ℓ) for the interference terms.
    """
    if l == 0:
        return rho_val.copy()

    rho2 = rho_val ** 2
    P_prev = rho_val.copy()
    S_prev = np.zeros_like(rho_val)

    for ll in range(1, l + 1):
        D = (ll - S_prev) ** 2 + P_prev ** 2
        D_safe = np.where(D > 0, D, 1.0)
        P_curr = rho2 * P_prev / D_safe
        S_curr = rho2 * (ll - S_prev) / D_safe - ll
        P_curr = np.where(D > 0, P_curr, 0.0)
        S_curr = np.where(D > 0, S_curr, -float(ll))
        P_prev = P_curr
        S_prev = S_curr

    return np.arctan2(P_prev, l - S_prev)
