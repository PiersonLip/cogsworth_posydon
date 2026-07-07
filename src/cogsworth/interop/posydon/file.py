import numpy as np
import pandas as pd

from .utils import (
    POSYDON_EVENT_TO_EVOL_TYPE,
    kick_vector_from_angles,
    porb_to_sep,
    posydon_metallicity_to_mass_fraction,
    posydon_state_to_kstar,
    posydon_time_to_myr,
)

__all__ = ["get_initial_binaries", "get_bpp", "get_kick_info", "load_posydon_tables"]

KICK_INFO_COLS = [
    "disrupted", "natal_kick", "phi", "theta", "mean_anomaly",
    "delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1", "vsys_1_total",
    "delta_vsysx_2", "delta_vsysy_2", "delta_vsysz_2", "vsys_2_total",
]


def load_posydon_tables(filename):
    """Load the oneline and history tables from a POSYDON population HDF5 file."""
    oneline = pd.read_hdf(filename, key="oneline")
    history = pd.read_hdf(filename, key="history")
    return oneline, history


def _normalize_posydon_selection(posydon_indices=None, bin_nums=None, n_systems=None):
    """Return aligned POSYDON index and cogsworth bin_num arrays."""
    if posydon_indices is None and n_systems is None:
        raise ValueError("Either posydon_indices or n_systems must be provided.")
    if posydon_indices is None:
        posydon_indices = np.arange(n_systems)
    posydon_indices = np.asarray(posydon_indices)
    if bin_nums is None:
        bin_nums = np.arange(1, len(posydon_indices) + 1, dtype=int)
    else:
        bin_nums = np.asarray(bin_nums, dtype=int)
    if len(bin_nums) != len(posydon_indices):
        raise ValueError("posydon_indices and bin_nums must have the same length.")
    return posydon_indices, bin_nums


def get_initial_binaries(filename, indices=None, posydon_indices=None, bin_nums=None, tphysf=None):
    """Create a COSMIC InitialBinaries table from a POSYDON population file.

    Parameters
    ----------
    filename : str
        Path to a POSYDON synthetic population HDF5 file.
    indices : array-like, optional
        Deprecated alias for ``posydon_indices``.
    posydon_indices : array-like, optional
        POSYDON ``binary_index`` values to load. If None, all systems are used.
    bin_nums : array-like, optional
        Unique cogsworth ``bin_num`` values for each selected system. If None,
        sequential integers starting at 1 are assigned.
    tphysf : array-like or float, optional
        Maximum evolution time [Myr] for each binary. If None, values are taken
        from the POSYDON final evolution time.

    Returns
    -------
    initial_binaries : cosmic.sample.InitialBinaryTable.InitialBinaries
    """
    from cosmic.sample import InitialBinaryTable

    if posydon_indices is None:
        posydon_indices = indices

    oneline, _ = load_posydon_tables(filename)
    if posydon_indices is None:
        posydon_indices = oneline.index.values
    posydon_indices, bin_nums = _normalize_posydon_selection(posydon_indices, bin_nums)

    selected = oneline.loc[posydon_indices]
    if tphysf is None:
        tphysf = posydon_time_to_myr(selected["time_f"].values)
    elif np.isscalar(tphysf):
        tphysf = np.full(len(posydon_indices), tphysf)

    initial_binaries = InitialBinaryTable.InitialBinaries(
        m1=selected["S1_mass_i"].values,
        m2=selected["S2_mass_i"].values,
        porb=selected["orbital_period_i"].values,
        ecc=selected["eccentricity_i"].values,
        tphysf=tphysf,
        kstar1=np.array([posydon_state_to_kstar(s) for s in selected["S1_state_i"]]),
        kstar2=np.array([posydon_state_to_kstar(s) for s in selected["S2_state_i"]]),
        metallicity=posydon_metallicity_to_mass_fraction(selected["metallicity"].values),
    )
    initial_binaries.index = bin_nums
    initial_binaries["bin_num"] = bin_nums
    return initial_binaries


def _binary_orbital_state(state, orbital_period, eccentricity, mass_1, mass_2):
    """Return COSMIC-style sep, ecc, porb for a POSYDON binary state."""
    if state in ("disrupted", "initially_single_star"):
        return -1.0, -1.0, -1.0
    if state == "merged":
        return 0.0, 0.0, 0.0
    sep = porb_to_sep(orbital_period, mass_1, mass_2)
    ecc = float(eccentricity) if np.isfinite(eccentricity) else np.nan
    porb = float(orbital_period) if np.isfinite(orbital_period) else np.nan
    if not np.isfinite(sep):
        return -1.0, -1.0, -1.0
    return sep, ecc, porb


def _history_row_to_evol_type(row, prev_row=None):
    """Infer a COSMIC evol_type for a POSYDON history row."""
    event = row["event"]
    if pd.notna(event) and event in POSYDON_EVENT_TO_EVOL_TYPE:
        return POSYDON_EVENT_TO_EVOL_TYPE[event]
    if prev_row is not None:
        if (posydon_state_to_kstar(row["S1_state"]) != posydon_state_to_kstar(prev_row["S1_state"])
                or posydon_state_to_kstar(row["S2_state"]) != posydon_state_to_kstar(prev_row["S2_state"])):
            return 2
    if row["state"] == "disrupted":
        return 11
    return 2


def _history_rows(history, posydon_index):
    """Return history rows for one POSYDON binary as a DataFrame."""
    hrows = history.loc[posydon_index]
    if isinstance(hrows, pd.Series):
        return hrows.to_frame().T
    return hrows


def _oneline_row(oneline, posydon_index):
    """Return a single oneline row for one POSYDON binary."""
    row = oneline.loc[posydon_index]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def get_bpp(filename, indices=None, posydon_indices=None, bin_nums=None):
    """Create a COSMIC-like bpp table from a POSYDON population file."""
    if posydon_indices is None:
        posydon_indices = indices

    oneline, history = load_posydon_tables(filename)
    if posydon_indices is None:
        posydon_indices = oneline.index.values
    posydon_indices, bin_nums = _normalize_posydon_selection(posydon_indices, bin_nums)

    bpp_rows = []
    for posydon_index, bin_num in zip(posydon_indices, bin_nums):
        hrows = _history_rows(history, posydon_index)
        final = _oneline_row(oneline, posydon_index)
        zams_time = hrows.iloc[0]["time"]
        prev_row = None
        for _, row in hrows.iterrows():
            sep, ecc, porb = _binary_orbital_state(
                row["state"], row["orbital_period"], row["eccentricity"],
                row["S1_mass"], row["S2_mass"],
            )
            evol_type = _history_row_to_evol_type(row, prev_row=prev_row)
            bpp_rows.append({
                "bin_num": int(bin_num),
                "tphys": posydon_time_to_myr(row["time"] - zams_time),
                "mass_1": row["S1_mass"],
                "mass_2": row["S2_mass"],
                "kstar_1": posydon_state_to_kstar(row["S1_state"]),
                "kstar_2": posydon_state_to_kstar(row["S2_state"]),
                "sep": sep,
                "ecc": ecc,
                "porb": porb,
                "rad_1": 10 ** row["S1_log_R"] if pd.notna(row["S1_log_R"]) else np.nan,
                "rad_2": 10 ** row["S2_log_R"] if pd.notna(row["S2_log_R"]) else np.nan,
                "evol_type": evol_type,
                "RRLO_1": 0.0,
                "RRLO_2": 0.0,
            })
            prev_row = row

        # ensure a final-state row exists
        sep, ecc, porb = _binary_orbital_state(
            final["state_f"], final["orbital_period_f"], final["eccentricity_f"],
            final["S1_mass_f"], final["S2_mass_f"],
        )
        bpp_rows.append({
            "bin_num": int(bin_num),
            "tphys": posydon_time_to_myr(final["time_f"] - zams_time),
            "mass_1": final["S1_mass_f"],
            "mass_2": final["S2_mass_f"],
            "kstar_1": posydon_state_to_kstar(final["S1_state_f"]),
            "kstar_2": posydon_state_to_kstar(final["S2_state_f"]),
            "sep": sep,
            "ecc": ecc,
            "porb": porb,
            "rad_1": 10 ** final["S1_log_R_f"] if pd.notna(final["S1_log_R_f"]) else np.nan,
            "rad_2": 10 ** final["S2_log_R_f"] if pd.notna(final["S2_log_R_f"]) else np.nan,
            "evol_type": 10,
            "RRLO_1": 0.0,
            "RRLO_2": 0.0,
        })

    bpp = pd.DataFrame(bpp_rows)
    bpp = bpp.drop_duplicates(subset=["bin_num", "tphys", "evol_type"], keep="last")
    bpp = bpp.sort_values(["bin_num", "tphys", "evol_type"])
    for col in ["tphys", "mass_1", "mass_2", "sep", "ecc", "porb", "rad_1", "rad_2"]:
        bpp[col] = pd.to_numeric(bpp[col], errors="coerce")
    bpp.index = bpp["bin_num"].values
    return bpp


def _kick_row_from_oneline(oneline_row, star):
    """Build kick parameters for one star from an oneline row."""
    prefix = f"S{star}"
    v_col = f"{prefix}_natal_kick_velocity"
    phi_col = f"{prefix}_natal_kick_azimuthal_angle"
    theta_col = f"{prefix}_natal_kick_polar_angle"
    mean_col = f"{prefix}_natal_kick_mean_anomaly"

    if v_col not in oneline_row.index or not pd.notna(oneline_row[v_col]) or oneline_row[v_col] <= 0:
        return None

    vx, vy, vz = kick_vector_from_angles(
        oneline_row[v_col], oneline_row[phi_col], oneline_row[theta_col]
    )
    mean_anomaly = oneline_row[mean_col] if mean_col in oneline_row.index and pd.notna(oneline_row[mean_col]) else 0.0
    disrupted = 1.0 if oneline_row["state_f"] in ("disrupted", "initially_single_star") else 0.0

    kick = {
        "star": star,
        "disrupted": disrupted,
        "natal_kick": float(oneline_row[v_col]),
        "phi": np.degrees(float(oneline_row[phi_col])) if pd.notna(oneline_row[phi_col]) else 0.0,
        "theta": np.degrees(float(oneline_row[theta_col])) if pd.notna(oneline_row[theta_col]) else 0.0,
        "mean_anomaly": np.degrees(float(mean_anomaly)),
        "vsys_1_total": 0.0,
        "vsys_2_total": 0.0,
    }
    if star == 1:
        kick.update({
            "delta_vsysx_1": vx, "delta_vsysy_1": vy, "delta_vsysz_1": vz,
            "vsys_1_total": float(oneline_row[v_col]),
            "delta_vsysx_2": 0.0, "delta_vsysy_2": 0.0, "delta_vsysz_2": 0.0,
        })
    else:
        kick.update({
            "delta_vsysx_1": 0.0, "delta_vsysy_1": 0.0, "delta_vsysz_1": 0.0,
            "delta_vsysx_2": vx, "delta_vsysy_2": vy, "delta_vsysz_2": vz,
            "vsys_2_total": float(oneline_row[v_col]),
        })
    return kick


def get_kick_info(filename, indices=None, posydon_indices=None, bin_nums=None):
    """Create a COSMIC-like kick_info table from a POSYDON population file."""
    if posydon_indices is None:
        posydon_indices = indices

    oneline, history = load_posydon_tables(filename)
    if posydon_indices is None:
        posydon_indices = oneline.index.values
    posydon_indices, bin_nums = _normalize_posydon_selection(posydon_indices, bin_nums)

    full_index = pd.MultiIndex.from_product([bin_nums, [1, 2]], names=["bin_num", "star"])
    kick_info = pd.DataFrame(
        np.zeros((len(bin_nums) * 2, len(KICK_INFO_COLS))),
        columns=KICK_INFO_COLS,
        index=full_index,
    )
    kick_info["WAS_KICKED"] = 0

    # fill kicks from history CC events, falling back to oneline summary columns
    for posydon_index, bin_num in zip(posydon_indices, bin_nums):
        oneline_row = _oneline_row(oneline, posydon_index)
        hrows = _history_rows(history, posydon_index)
        kicks_found = []

        zams_time = hrows.iloc[0]["time"]
        for _, row in hrows.iterrows():
            if row["event"] == "CC1":
                kick = _kick_row_from_oneline(oneline_row, star=1)
                if kick is not None:
                    kick["tphys"] = posydon_time_to_myr(row["time"] - zams_time)
                    kicks_found.append(kick)
            elif row["event"] == "CC2":
                kick = _kick_row_from_oneline(oneline_row, star=2)
                if kick is not None:
                    kick["tphys"] = posydon_time_to_myr(row["time"] - zams_time)
                    kicks_found.append(kick)

        if not kicks_found:
            for star in [1, 2]:
                kick = _kick_row_from_oneline(oneline_row, star=star)
                if kick is not None:
                    kick["tphys"] = posydon_time_to_myr(oneline_row["time_f"] - zams_time)
                    kicks_found.append(kick)

        for kick in kicks_found:
            star = kick.pop("star")
            for col in KICK_INFO_COLS:
                kick_info.loc[(int(bin_num), star), col] = kick[col]
            kick_info.loc[(int(bin_num), star), "WAS_KICKED"] = 1

    kick_info["WAS_KICKED"] = kick_info.get("WAS_KICKED", 0).fillna(0)
    kick_info.reset_index(inplace=True)
    kick_info.loc[kick_info["WAS_KICKED"] == 0, "star"] = 0
    kick_info.drop(columns=["WAS_KICKED"], inplace=True, errors="ignore")
    kick_info.index = kick_info["bin_num"].values
    return kick_info
