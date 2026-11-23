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
    backend = "metal",            # "cpu" | "cuda" | "metal"   (metal runs in Float32)
    live    = false,             # show density and phase on the fly (needs GLMakie)

    # ------------------------------------------------------------ grid
    nx = 256,                   # even; powers of two are best for FFTs
    ny = 256,
    Lx = 2π,
    Ly = 2π,

    # ------------------------------------------------------------ physics
    c  = 1.0,                   # speed of sound
    xi_over_dx = 2.0,           # healing length ξ = xi_over_dx · dx
    n0 = 1.0,                   # mean density

    # ------------------------------------------------------------ case
    case = "random",            # "dipole":   vortex dipole in a uniform condensate
                                # "obstacle": moving Gaussian obstacle + a little noise
                                # "waves":    uniform condensate + waves; good for the dispersion relation
                                # "random":   Random array of quantum vortices

    dipole_separation = 10.0,    # [ξ]  distance between the two vortices
    nvortices = 200,               # number of vortices (case "random")
    seed = 1234,                  # random seed (case "random")

    noise_amplitude = 0.0,     # relative to √n0 (cases "obstacle" and "noise")
    kicmin = 1,                 # [kξ]  minimum wavenumber of the noise (in units of ξ⁻¹)
    kicmax = 4,                 # [kξ]  maximum wavenumber of the noise (in units of ξ⁻¹)
    obstacle = (V0       = 10.0, # [μ]  height
                sigma    = 2, # [ξ]  width
                velocity = 0.6, # [c]  speed along x
                x0       = -0.25, # initial position, as a fraction of Lx from the box centre
                y0       = 0.0),  # initial position, as a fraction of Ly from the box centre

    # ------------------------------------------------------------ time stepping
    dt     = 0.01,              # time step (IFRK4)
    nsteps = 10000,

    # ------------------------------------------------------------ imaginary-time relaxation (RGLE)
    # Optionally relax the initial state in imaginary time (real Ginzburg-Landau equation)
    # before the real-time run. With case = "obstacle" the obstacle is held at its initial
    # position during the relaxation, so the run starts from the ground state around it.
    # Not for case = "noise" (it would remove the noise).
    # A vortex dipole is not a stationary state of the RGLE: the vortices attract and
    # annihilate. For a separation of 8ξ the pair is unchanged for τ < 2ξ/c, visibly closer
    # at 3-5 ξ/c and gone at ~15 ξ/c. For the dipole use a short relax_max_time (1-2 ξ/c).
    relax               = true,
    relax_dt            = 0.01,           # imaginary-time step
    relax_max_time      = 1.0,            # stop after this imaginary time ...
    relax_tol           = 1e-6,           # ... or when the residual ‖∂τψ‖/(ω0‖ψ‖) is below this
    relax_normalization = "fixed_mu",     # "fixed_mu": μ = g n0 fixed, N free
                                          # "fixed_number": N kept at its initial value
    relax_check_every   = 100,            # steps between residual checks (and printed lines)

    # ------------------------------------------------------------ HDF5 output
    save_output = true,         # master switch
    output_dir  = "output",     # folder (relative to where you launch Julia)
    output_name = "run",        # file name without extension → output/run.h5 (overwritten)

    # Three independent streams; the values are in time STEPS and 0 disables a stream.
    # The initial state (iteration 0) and the final state are always written.
    save_fields_every      = 100,   # snapshots of Re ψ and Im ψ
    save_diagnostics_every = 20,    # scalar time series: N (particle number) and E (energy)
    save_spectra_every     = 200,   # energy spectra: incompressible and compressible kinetic,
                                    # quantum pressure and internal (9 FFTs per output, so
                                    # keep it much larger than the other two)

    # Dispersion relation: save ψ̂(kx, ky=0) and ψ̂(kx=0, ky) as functions of time, to get
    # ω(k) in post-processing by Fourier transforming in time. Cheap (no extra FFT).
    # The excitations must be seeded: use case = "noise" (or "obstacle").
    # The sampling time Δt = save_dispersion_every·dt must satisfy π/Δt > ω_max, and the
    # frequency resolution is 2π/(nsteps·dt); the script prints both.
    save_dispersion       = false,   # master switch for this stream
    save_dispersion_every = 10,      # steps between samples (uniform sampling in time)

    # ------------------------------------------------------------ screen output / live view
    print_every = 200,          # steps between lines with the drift of N and E
    plot_every  = 10,           # steps between plot updates
    plot_pause  = 0.0,          # extra seconds to wait after each frame (slows the movie down)
    density_range = (0.0, 1.05), # colour range of |ψ|²/n0
)
