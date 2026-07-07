# POSYDON interoperability

Load pre-computed POSYDON synthetic populations into cogsworth via
:class:`~cogsworth.interop.posydon.pop.POSYDONPopulation`.

Example::

    from cogsworth import POSYDONPopulation

    pop = POSYDONPopulation.from_POSYDON_output("/path/to/10kSample.h5")
    pop.sample_initial_galaxy()  # if not already sampled
    pop.perform_galactic_evolution()

Or sample ``n`` systems from a POSYDON file and run the full pipeline::

    pop = POSYDONPopulation(
        n_binaries=1000,
        posydon_file="/path/to/10kSample.h5",
        processes=4,
        use_default_BSE_settings=True,
    )
    pop.create_population()

POSYDON population files must contain ``oneline`` and ``history`` HDF5 tables in the
standard POSYDON synthetic population format.
