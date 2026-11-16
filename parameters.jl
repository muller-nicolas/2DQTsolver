# All parameters of the example run, in one place.
#
# This file is read by run_gpe.jl (it must evaluate to a NamedTuple, so keep the outer
# parentheses). To keep several setups, copy it and pass the copy to the script:
#
#     julia --project examples/run_gpe.jl                  # uses this file
#     julia --project examples/run_gpe.jl my_setup.jl      # uses another file
#
# Units: lengths in units set by Lx (the box), time in the units implied by c.
# Quantities named *_xi are in units of the healing length ξ, *_c in units of c,
# and potentials in units of the chemical potential μ = g n0 = c² (m = 1).

(
    # ------------------------------------------------------------ backend
    backend = "cpu",            # "cpu" | "cuda" | "metal"   (metal runs in Float32)
    live    = true,             # show density and phase on the fly (needs GLMakie)

    # ------------------------------------------------------------ grid
    nx = 128,                   # even; powers of two are best for FFTs
    ny = 128,
    Lx = 2π,
    Ly = 2π,

    # ------------------------------------------------------------ physics
    c  = 1.0,                   # speed of sound
    xi_over_dx = 2.0,           # healing length ξ = xi_over_dx · dx
    n0 = 1.0,                   # mean density

    # ------------------------------------------------------------ case
    case = "waves",             # "dipole": vortex dipole in a uniform condensate
                                # "obstacle": moving Gaussian obstacle + a little noise

    dipole_separation = 8.0,    # [ξ]  distance between the two vortices

    noise_amplitude = 0.1,     # relative to √n0 (used by case = "obstacle")
    kicmin = 1,                 # [k]  minimum wavenumber of the initial waves
    kicmax = 8,                 # [k]  maximum wavenumber of the initial waves
    obstacle = (V0       = 3.0, # [μ]  height
                sigma    = 1.5, # [ξ]  width
                velocity = 0.5, # [c]  speed along x
                x0       = -0.25, # initial position, as a fraction of Lx from the box centre
                y0       = 0.0),  # initial position, as a fraction of Ly from the box centre

    # ------------------------------------------------------------ time stepping
    dt     = 0.01,              # time step (IFRK4)
    nsteps = 3000,

    # ------------------------------------------------------------ HDF5 output
    save_output = true,         # master switch
    output_dir  = "output",     # folder (relative to where you launch Julia)
    output_name = "run",        # file name without extension → output/run.h5 (overwritten)

    # Three independent streams; the values are in time STEPS and 0 disables a stream.
    # The initial state (iteration 0) and the final state are always written.
    save_fields_every      = 100,   # snapshots of Re ψ and Im ψ
    save_diagnostics_every = 20,    # scalar time series: N (particle number) and E (energy)
    save_spectra_every     = 0,     # spectral outputs. Reserved: the spectra themselves come
                                    # with the spectral analysis module; until then a value
                                    # > 0 only prints a warning and is ignored.

    # ------------------------------------------------------------ screen output / live view
    print_every = 10,          # steps between lines with the drift of N and E
    plot_every  = 10,           # steps between plot updates
    plot_pause  = 0.0,          # extra seconds to wait after each frame (slows the movie down)
    density_range = (0.0, 1.5), # colour range of |ψ|²/n0
)
