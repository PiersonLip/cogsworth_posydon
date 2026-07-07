import logging
import os

import numpy as np
import pandas as pd

from ...pop import Population
from ..compas.utils import add_kicks_to_initial_binaries
from .file import get_bpp, get_initial_binaries, get_kick_info, load_posydon_tables

__all__ = ["POSYDONPopulation"]


class POSYDONPopulation(Population):
    """A Population class that uses pre-computed POSYDON evolution instead of COSMIC.

    This class extends the generic Population class to load binary stellar evolution
    from a POSYDON synthetic population HDF5 file (e.g. ``10kSample.h5``). Galaxy
    sampling and orbit integration proceed as in the base :class:`~cogsworth.pop.Population`.

    Parameters
    ----------
    n_binaries : int
        The number of binaries to simulate
    posydon_file : str
        Path to a POSYDON synthetic population HDF5 file containing ``oneline`` and
        ``history`` tables.
    sample_with_replacement : bool, optional
        Whether to draw systems from the POSYDON file with replacement when
        ``n_binaries`` exceeds the number of systems in the file. Default is True.
    random_seed : int, optional
        Random seed used when subsampling systems from the POSYDON file.
    **kwargs : dict
        Additional keyword arguments to pass to the Population class constructor
    """

    def __init__(
        self,
        n_binaries,
        posydon_file,
        sample_with_replacement=True,
        random_seed=42,
        **kwargs,
    ):
        if not os.path.isfile(posydon_file):
            raise FileNotFoundError(f"POSYDON population file not found: {posydon_file}")

        self.posydon_file = posydon_file
        self.sample_with_replacement = sample_with_replacement
        self.random_seed = random_seed
        self._posydon_indices = None

        if "use_default_BSE_settings" not in kwargs:
            kwargs["use_default_BSE_settings"] = True

        super().__init__(n_binaries=n_binaries, **kwargs)
        self.__citations__.append("posydon")

    def _extra_population_init_kwargs(self):
        return {
            "posydon_file": self.posydon_file,
            "sample_with_replacement": self.sample_with_replacement,
            "random_seed": self.random_seed,
        }

    def __getitem__(self, ind):
        new_pop = super().__getitem__(ind)
        if self._posydon_indices is not None and self._initial_binaries is not None:
            bin_num_to_posydon = dict(zip(
                self._initial_binaries["bin_num"].values,
                self._posydon_indices,
            ))
            new_pop._posydon_indices = np.array([
                bin_num_to_posydon[bin_num] for bin_num in new_pop.bin_nums
            ])
        return new_pop

    @classmethod
    def from_POSYDON_output(cls, posydon_file, indices=None, **kwargs):
        """Create a POSYDONPopulation from an existing POSYDON population file.

        Parameters
        ----------
        posydon_file : str
            Path to a POSYDON synthetic population HDF5 file.
        indices : array-like, optional
            Subset of ``binary_index`` values to load. If None, all systems in the
            file are used and ``n_binaries`` is set from the file length.
        **kwargs : dict
            Additional keyword arguments to pass to the POSYDONPopulation constructor.

        Returns
        -------
        pop : POSYDONPopulation
        """
        oneline, _ = load_posydon_tables(posydon_file)
        if indices is None:
            indices = oneline.index.values
        indices = np.asarray(indices)

        pop = cls(
            n_binaries=len(indices),
            posydon_file=posydon_file,
            **kwargs,
        )
        pop._posydon_indices = indices
        bin_nums = np.arange(1, len(indices) + 1, dtype=int)
        pop._initial_binaries = get_initial_binaries(
            posydon_file, posydon_indices=indices, bin_nums=bin_nums
        )
        pop._bpp = get_bpp(posydon_file, posydon_indices=indices, bin_nums=bin_nums)
        pop._kick_info = get_kick_info(posydon_file, posydon_indices=indices, bin_nums=bin_nums)
        pop._append_kicks()
        pop.n_binaries_match = len(indices)
        return pop

    def _append_kicks(self):
        """Add kick information from POSYDON to the initial binaries dataframe."""
        if self._initial_binaries is None or self._kick_info is None:
            raise ValueError("Either initial_binaries or kick_info is None, cannot append kicks.")
        self._initial_binaries = add_kicks_to_initial_binaries(self._initial_binaries, self._kick_info)
        return self._initial_binaries

    def _choose_posydon_indices(self):
        """Choose which POSYDON systems to use for this population."""
        oneline, _ = load_posydon_tables(self.posydon_file)
        available = oneline.index.values

        if self.sampling_mask != "":
            masked = oneline.query(self.sampling_mask)
            if len(masked) < 1:
                raise ValueError(
                    "No POSYDON systems matched the current sampling_mask. "
                    "Note that POSYDONPopulation uses POSYDON oneline column names "
                    "(e.g. S1_mass_i, orbital_period_i)."
                )
            available = masked.index.values

        replace = self.sample_with_replacement or self.n_binaries > len(available)
        rng = np.random.default_rng(self.random_seed)
        return rng.choice(available, size=self.n_binaries, replace=replace)

    def sample_initial_binaries(self):
        """Sample initial binaries from a POSYDON population file."""
        self._bin_nums = None
        self._final_bpp = None
        self._initial_binaries = None
        self._bpp = None
        self._kick_info = None

        self._posydon_indices = self._choose_posydon_indices()
        self.n_binaries_match = len(self._posydon_indices)
        bin_nums = np.arange(1, self.n_binaries_match + 1, dtype=int)

        self._initial_binaries = get_initial_binaries(
            self.posydon_file, posydon_indices=self._posydon_indices, bin_nums=bin_nums
        )
        self.sample_initial_galaxy()

        # overwrite sampled metallicities and birth times with galaxy values
        self._initial_binaries["metallicity"] = self._initial_galaxy.Z
        self._initial_binaries["tphysf"] = self._initial_galaxy.tau.to("Myr").value

    def perform_stellar_evolution(self):
        """Load pre-computed POSYDON evolution tables instead of running COSMIC."""
        self._final_bpp = None
        self._observables = None
        self._bin_nums = None
        self._disrupted = None
        self._escaped = None

        if self._initial_binaries is None:
            logging.getLogger("cogsworth").warning(
                "cogsworth warning: Initial binaries not yet sampled, performing sampling now."
            )
            self.sample_initial_binaries()

        if self._posydon_indices is None:
            self._posydon_indices = self._initial_binaries.index.values

        bin_nums = self._initial_binaries["bin_num"].values
        self._bpp = get_bpp(
            self.posydon_file, posydon_indices=self._posydon_indices, bin_nums=bin_nums
        )
        self._kick_info = get_kick_info(
            self.posydon_file, posydon_indices=self._posydon_indices, bin_nums=bin_nums
        )
        self._append_kicks()

    def create_population(self, with_timing=True):
        """Create a population using POSYDON evolution tables and galactic integration.

        If initial binaries and evolution tables were already loaded (e.g. via
        :meth:`from_POSYDON_output`), those are reused instead of resampling.
        """
        if self._initial_binaries is None:
            self.sample_initial_binaries()
        elif self._initial_galaxy is None:
            self.sample_initial_galaxy()

        if self._bpp is None:
            self.perform_stellar_evolution()

        if self._orbits is None:
            self.perform_galactic_evolution(progress_bar=with_timing)

    def to_Population(self, **kwargs):
        """Convert this POSYDONPopulation to a generic Population object."""
        use_defaults = kwargs.pop("use_default_BSE_settings", True)
        pop = Population(self.n_binaries, use_default_BSE_settings=use_defaults, **kwargs)
        attrs_to_copy = [
            "n_binaries", "n_binaries_match", "processes", "final_kstar1", "final_kstar2",
            "sfh_model", "galactic_potential", "v_dispersion", "max_ev_time",
            "timestep_size", "pool", "store_entire_orbits", "bpp_columns", "bcm_columns",
            "_file", "_initial_binaries", "_initial_galaxy", "_mass_singles", "_mass_binaries",
            "_n_singles_req", "_n_bin_req", "_bpp", "_bcm", "_kick_info",
            "_orbits", "_classes", "_final_pos", "_final_vel", "_final_bpp", "_disrupted",
            "_escaped", "_observables", "_bin_nums", "BSE_settings",
            "sampling_params", "bcm_timestep_conditions",
        ]
        for attr in attrs_to_copy:
            if attr not in kwargs:
                setattr(pop, attr, getattr(self, attr))

        kick_cols = [
            "natal_kick_1", "natal_kick_2", "phi_1", "theta_1", "phi_2", "theta_2",
            "mean_anomaly_1", "mean_anomaly_2",
        ]
        any_were_present = any(col in self.initial_binaries.columns for col in kick_cols)

        if any_were_present and use_defaults and "natal_kick_array" in pop.BSE_settings:
            del pop.BSE_settings["natal_kick_array"]
        elif any_were_present and not use_defaults:
            logging.getLogger("cogsworth").warning(
                "cogsworth warning: Natal kick settings found in BSE_settings will overwrite "
                "the kicks from POSYDON."
            )

        return pop
