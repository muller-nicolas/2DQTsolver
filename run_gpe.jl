# Run the 2D GPE. ALL parameters are in `parameters.jl` (or in the file you pass).
#
#   julia --project examples/run_gpe.jl                 # uses examples/parameters.jl
#   julia --project examples/run_gpe.jl my_setup.jl     # uses your own parameter file
#
# It also works from the REPL:  include("examples/run_gpe.jl")


using Printf

# ------------------------------------------------------------------ parameters
params_file = isempty(ARGS) ? joinpath(@__DIR__, "parameters.jl") : abspath(ARGS[1])
cfg = include(params_file)          # a NamedTuple
println("Parameters read from ", params_file, ":")
for (k, v) in pairs(cfg)
    println("  ", rpad(string(k), 18), " = ", v)
end

# ------------------------------------------------------------------ load packages
# The GPU package and GLMakie must be loaded BEFORE they are used.
cfg.backend == "cuda"  && @eval using CUDA
cfg.backend == "metal" && @eval using Metal
using GPE2D
cfg.live && include(joinpath(@__DIR__, "LivePlot.jl"))

arch, T = cfg.backend == "cuda"  ? (CUDAGPU(),  Float64) :
          cfg.backend == "metal" ? (MetalGPU(), Float32) :
                                   (CPU(),      Float64)

# ------------------------------------------------------------------ setup
grid   = Grid(arch, T; nx = cfg.nx, ny = cfg.ny, Lx = cfg.Lx, Ly = cfg.Ly)
params = GPEParameters(grid; c = cfg.c, ξ = cfg.xi_over_dx * grid.dx, n0 = cfg.n0)
plans  = FFTPlans(grid)

obs = cfg.obstacle
potential = cfg.case == "obstacle" ?
    GaussianObstacle(obs.V0, obs.sigma * params.ξ;
                     x0 = obs.x0 * grid.Lx, y0 = obs.y0 * grid.Ly,
                     vx = obs.velocity * params.c) :
    NoPotential()

model = GPE(params, grid, plans; potential = potential)
state = gpe_state(grid)

ψ0 = if cfg.case == "obstacle" || cfg.case == "noise"
    add_noise!(uniform_density(grid, params), params; amplitude = cfg.noise_amplitude)
elseif cfg.case == "dipole"
    vortex_dipole(grid, params; separation = cfg.dipole_separation * params.ξ)
elseif cfg.case == "quadrupole"
    vortex_quadrupole(grid, params; separation = cfg.dipole_separation * params.ξ)
elseif cfg.case == "random"
    vortex_random(grid, params; nvortices = cfg.nvortices, seed = cfg.seed)
elseif cfg.case == "waves"
    waves!(state, model, cfg)
else
    error("unknown case \"$(cfg.case)\" (use \"dipole\", \"obstacle\" or \"noise\")")
end
set_wavefunction!(state, model, ψ0)

stepper = IFRK4(grid, model, cfg.dt)

println()
println(model)
@printf("backend: %s, %s | case: %s | dt = %g (ω0·dt = %.3f) | %d steps → t_end = %.2f\n",
        architecture_name(arch), T, cfg.case, cfg.dt, params.ω0 * cfg.dt,
        cfg.nsteps, cfg.nsteps * cfg.dt)

# ------------------------------------------------------------------ callbacks
callbacks = Any[every(cfg.print_every, ConservationLogger(state, model))]

# Dispersion output: check that the time sampling resolves the Bogoliubov frequencies
if cfg.save_output && cfg.save_dispersion
    dt_out = cfg.save_dispersion_every * cfg.dt
    ω_nyq  = π / dt_out
    ω_max  = bogoliubov_frequency(params, minimum(grid.kcut))
    @printf("dispersion output: Δt = %.4g, ω_Nyquist = %.4g, Δω = %.3g, ω_Bogoliubov(kmax) = %.4g\n",
            dt_out, ω_nyq, 2π / (cfg.nsteps * cfg.dt), ω_max)
    ω_max > ω_nyq && @warn "The dispersion output is undersampled in time: decrease save_dispersion_every (or dt)."
end

writer = nothing
if cfg.save_output
    output_path = joinpath(cfg.output_dir, cfg.output_name * ".h5")
    writer = OutputWriter(output_path, state, model;
                          fields_every      = cfg.save_fields_every,
                          diagnostics_every = cfg.save_diagnostics_every,
                          spectra_every     = cfg.save_spectra_every,
                          traces_every      = cfg.save_dispersion ? cfg.save_dispersion_every : 0,
                          nsteps = cfg.nsteps, metadata = cfg)
    writer(state)                               # iteration 0
    push!(callbacks, writer)
    println(writer)
end

if cfg.live
    lp = LivePlot(model, grid; density_range = cfg.density_range, pause = cfg.plot_pause)
    lp(state)                                   # draw the initial condition
    push!(callbacks, every(cfg.plot_every, lp))
end

# ------------------------------------------------------------------ run
step!(state, stepper, model)                    # warm-up (compilation) step
t_wall = @elapsed begin
    evolve!(state, stepper, model, cfg.nsteps - 1; callback = CallbackSet(callbacks...))
    device_synchronize(arch)
end
@printf("done: %d steps in %.2f s  (%.2f ms/step)\n", state.clock.iteration, t_wall,
        1e3 * t_wall / max(state.clock.iteration - 1, 1))

if writer !== nothing
    write_final!(writer, state)                 # final state, if not already written
    close(writer)
    println("Output written to ", output_path)
end

if cfg.live && isopen(lp.screen)
    println("Close the plot window to exit.")
    wait(lp.screen)
end
