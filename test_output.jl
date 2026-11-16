# Tests for the HDF5 output (Output.jl).
#
#   julia --project test_output.jl            # CPU, Float64
#   julia --project test_output.jl cuda       # CUDA, Float64
#   julia --project test_output.jl metal      # Metal, Float32
#
# Requires HDF5 in the GPE2D environment:  ] add HDF5

using Test, HDF5

backend = isempty(ARGS) ? "cpu" : lowercase(ARGS[1])
backend == "cuda"  && @eval using CUDA      # load the backend BEFORE GPE2D
backend == "metal" && @eval using Metal
using GPE2D

arch, T = backend == "cuda"  ? (CUDAGPU(),  Float64) :
          backend == "metal" ? (MetalGPU(), Float32) :
                               (CPU(),      Float64)
println("Testing on ", architecture_name(arch), " with ", T)

# a small dipole simulation
function make_setup()
    grid   = Grid(arch, T; nx = 32, Lx = 2π)
    p      = GPEParameters(grid; c = 1.0, ξ = 2 * grid.dx)
    model  = GPE(p, grid, FFTPlans(grid))
    state  = gpe_state(grid)
    set_wavefunction!(state, model, vortex_dipole(grid, p; separation = 8 * p.ξ))
    return grid, p, model, state, IFRK4(grid, model, 0.01)
end

cfg = (nx = 32, case = "dipole", live = false, density_range = (0.0, 1.5),
       obstacle = (V0 = 3.0, sigma = 1.5))
dt = 0.01

@testset "Output: fields, time series, spectra hook" begin
    grid, p, model, state, stepper = make_setup()
    path = joinpath(mktempdir(), "sub", "test.h5")           # the folder is created

    # a model-independent spectra function, to test the spectra stream
    base = default_outputs(model)
    spec = OutputSpec(; fields = base.fields, scalars = base.scalars, metadata = base.metadata,
                      spectra = (s, m) -> (k = [1.0, 2.0], Ek = [3.0, 4.0]))

    w = OutputWriter(path, state, model; fields_every = 10, diagnostics_every = 5,
                     spectra_every = 10, nsteps = 20, outputs = spec, metadata = cfg)
    w(state)                                                   # iteration 0
    evolve!(state, stepper, model, 20; callback = w)
    update_physical!(state, model)
    ψ_final = Array(state.ψ)
    N_final, E_final = particle_number(state, model), energy(state, model)
    close(w)

    @test isfile(path)

    # snapshots: iterations 0, 10, 20
    @test snapshot_names(path) == ["00000000", "00000010", "00000020"]
    s = read_snapshot(path, 3)
    @test s.iteration == 20 && s.t ≈ 0.2
    @test size(s.fields.psi_re) == (32, 32)
    @test s.fields.psi_re == real.(ψ_final)
    @test s.fields.psi_im == imag.(ψ_final)
    @test read_snapshot(path, "00000010").iteration == 10

    # time series: iterations 0, 5, 10, 15, 20
    ts = read_timeseries(path)
    @test ts.iteration == [0, 5, 10, 15, 20]
    @test ts.t ≈ dt .* [0, 5, 10, 15, 20]
    @test all(isfinite, ts.N) && all(isfinite, ts.E)
    @test ts.N[end] ≈ N_final
    @test ts.E[end] ≈ E_final

    # spectra stream
    @test snapshot_names(path; group = "spectra") == ["00000000", "00000010", "00000020"]
    sp = read_snapshot(path, 2; group = "spectra")
    @test sp.fields.k == [1.0, 2.0] && sp.fields.Ek == [3.0, 4.0]

    # metadata: nested NamedTuples are flattened as "outer.inner"; Bool stored as Int
    conf = read_attributes(path, "config")
    @test conf["nx"] == 32 && conf["case"] == "dipole"
    @test conf["obstacle.V0"] == 3.0
    @test conf["live"] == 0
    @test conf["density_range"] == [0.0, 1.5]
    par = read_attributes(path, "parameters")
    @test par["c"] ≈ 1.0 && par["xi"] ≈ p.ξ && par["omega0"] ≈ p.ω0
    gridattrs = read_attributes(path, "grid")
    @test gridattrs["nx"] == 32 && gridattrs["precision"] == string(T)
    h5open(path, "r") do f
        @test length(read(f["grid/x"])) == 32
        @test read(f["grid/kx"])[2] ≈ 1.0                      # Lx = 2π → k = 1, 2, ...
    end
end

@testset "Output: final state, no duplicates" begin
    grid, p, model, state, stepper = make_setup()
    path = joinpath(mktempdir(), "final.h5")
    w = OutputWriter(path, state, model; fields_every = 10, diagnostics_every = 5, nsteps = 23)
    w(state)
    evolve!(state, stepper, model, 23; callback = w)           # last due: iteration 20
    write_final!(w, state)                                     # iteration 23
    write_final!(w, state)                                     # again: nothing new
    close(w)

    @test snapshot_names(path) == ["00000000", "00000010", "00000020", "00000023"]
    @test read_timeseries(path).iteration == [0, 5, 10, 15, 20, 23]
end

@testset "Output: independent frequencies, disabled streams" begin
    grid, p, model, state, stepper = make_setup()
    path = joinpath(mktempdir(), "partial.h5")
    w = OutputWriter(path, state, model; fields_every = 0, diagnostics_every = 7, nsteps = 14)
    w(state)
    evolve!(state, stepper, model, 14; callback = w)
    write_final!(w, state)
    close(w)

    @test snapshot_names(path) == String[]                     # fields disabled
    @test read_timeseries(path).iteration == [0, 7, 14]

    # spectra requested but the GPE has no spectra output yet: warning, stream disabled
    path2 = joinpath(mktempdir(), "nospectra.h5")
    w2 = @test_logs (:warn, r"pectra") match_mode = :any OutputWriter(
        path2, state, model; fields_every = 5, spectra_every = 5, nsteps = 10)
    @test w2.spectra_every == 0
    close(w2)
    @test snapshot_names(path2; group = "spectra") == String[]
end

println("\nAll tests finished.")
