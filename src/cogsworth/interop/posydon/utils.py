import numpy as np
import astropy.constants as const
import astropy.units as u

__all__ = [
    "POSYDON_STATE_TO_KSTAR",
    "POSYDON_EVENT_TO_EVOL_TYPE",
    "SOLAR_METALLICITY_MASS_FRACTION",
    "posydon_metallicity_to_mass_fraction",
    "posydon_time_to_myr",
    "porb_to_sep",
    "kick_vector_from_angles",
    "posydon_state_to_kstar",
]

# POSYDON metallicity is relative to solar (1.0 = solar); COSMIC uses mass fraction.
SOLAR_METALLICITY_MASS_FRACTION = 0.0142

POSYDON_STATE_TO_KSTAR = {
    "H-rich_Core_H_burning": 1,
    "H-rich_Shell_H_burning": 1,
    "H-rich_non_burning": 1,
    "H-rich_Core_He_burning": 4,
    "H-rich_Core_He_depleted": 5,
    "H-rich_Core_C_burning": 5,
    "H-rich_Core_C_depleted": 6,
    "stripped_He_Core_He_burning": 7,
    "stripped_He_Core_He_depleted": 9,
    "stripped_He_non_burning": 7,
    "accreted_He_Core_He_burning": 7,
    "accreted_He_non_burning": 7,
    "WD": 10,
    "NS": 13,
    "BH": 14,
    "massless_remnant": 15,
}

POSYDON_EVENT_TO_EVOL_TYPE = {
    "ZAMS": 1,
    "oRLO1": 3,
    "oRLO2": 3,
    "oCE1": 7,
    "oCE2": 7,
    "oDoubleCE1": 7,
    "oDoubleCE2": 7,
    "oMerging1": 6,
    "oMerging2": 6,
    "CC1": 15,
    "CC2": 16,
    "END": 10,
    "maxtime": 10,
    "FAILED": 10,
}


def posydon_metallicity_to_mass_fraction(z_relative):
    """Convert POSYDON metallicity (1.0 = solar) to a mass fraction."""
    return np.asarray(z_relative) * SOLAR_METALLICITY_MASS_FRACTION


def posydon_time_to_myr(time_s):
    """Convert POSYDON simulation time [s] to COSMIC tphys [Myr]."""
    return np.asarray(time_s) / u.Myr.to(u.s)


def posydon_state_to_kstar(state):
    """Map a POSYDON stellar state string to a COSMIC kstar integer."""
    if state is None or (isinstance(state, float) and np.isnan(state)):
        return 1
    return POSYDON_STATE_TO_KSTAR.get(str(state), 1)


def porb_to_sep(porb_days, mass_1, mass_2):
    """Convert orbital period [days] and masses [Msun] to separation [Rsun]."""
    scalar_input = np.isscalar(porb_days) and np.isscalar(mass_1) and np.isscalar(mass_2)
    porb_days = np.asarray(porb_days, dtype=float)
    mass_1 = np.asarray(mass_1, dtype=float)
    mass_2 = np.asarray(mass_2, dtype=float)
    sep = np.full_like(porb_days, np.nan, dtype=float)
    valid = np.isfinite(porb_days) & (porb_days > 0) & np.isfinite(mass_1) & np.isfinite(mass_2)
    if valid.any():
        a = ((porb_days[valid] * u.day / (2 * np.pi)) ** 2
             * const.G * (mass_1[valid] + mass_2[valid]) * u.M_sun) ** (1 / 3)
        sep[valid] = a.to(u.Rsun).value
    if scalar_input:
        return float(sep)
    return sep


def kick_vector_from_angles(vkick_kms, phi, theta):
    """Return natal kick velocity components [km/s] from POSYDON angles [rad]."""
    vk = float(vkick_kms) if np.isfinite(vkick_kms) else 0.0
    phi = float(phi) if np.isfinite(phi) else np.nan
    theta = float(theta) if np.isfinite(theta) else np.nan
    if vk == 0.0 or not (np.isfinite(phi) and np.isfinite(theta)):
        return 0.0, 0.0, 0.0
    vx = vk * np.sin(theta) * np.cos(phi)
    vy = vk * np.sin(theta) * np.sin(phi)
    vz = vk * np.cos(theta)
    return vx, vy, vz
