# Tests for the GPE model, potentials, initial conditions, diagnostics and callbacks.
#
#   julia --project test_gpe.jl            # CPU, Float64
#   julia --project test_gpe.jl cuda       # CUDA, Float64
#   julia --project test_gpe.jl metal      # Metal, Float32

using Test

backend = isempty(ARGS) ? "cpu" : lowercase(ARGS[1])
backend == "cuda"  && @eval using CUDA      # load the backend BEFORE GPE2D
backend == "metal" && @eval using Metal
using GPE2D

arch, T = backend == "cuda"  ? (CUDAGPU(),  Float64) :
          backend == "metal" ? (MetalGPU(), Float32) :
                               (CPU(),      Float64)
println("Testing on ", architecture_name(arch), " with ", T)

maxdiff(a, b) = maximum(abs.(Array(a) .- Array(b)))

# grid 2π × 2π with ξ = 2 dx, c = 1 (the test parameters of the project)
function make_setup(; nx = 64, n0 = 1.0, potential = NoPotential())
    grid   = Grid(arch, T; nx = nx, Lx = 2π)
    params = GPEParameters(grid; c = 1.0, ξ = 2 * grid.dx, n0 = n0)
    plans  = FFTPlans(grid)
    model  = GPE(params, grid, plans; potential = potential)
    return grid, params, model, gpe_state(grid)
end

# ====================================================================== parameters
@testset "GPEParameters" begin
    p = GPEParameters(Float64; c = 1.0, ξ = 0.1, n0 = 2.0)
    @test p.ω0 ≈ 1 / (√2 * 0.1)                  # c/(√2 ξ)
    @test p.hbar_over_m ≈ √2 * 0.1               # √2 c ξ
    @test p.μ ≈ 1.0                              # c² = g n0 / m
    @test p.g ≈ 1.0 / 2.0                        # c²/n0
    @test p.g * p.n0 ≈ p.μ
    @test p.hbar_over_m^2 / 2 ≈ p.ξ^2 * p.μ      # ħ²/2m = ξ² μ  (definition of ξ)
    @test_throws ArgumentError GPEParameters(Float64; c = -1.0, ξ = 0.1)
    show(stdout, p); println()
end

# ====================================================================== potentials
@testset "Potentials" begin
    grid = Grid(arch, T; nx = 64, Lx = 2π)
    V = allocate_field(grid, RealPhysical())
    ic = grid.nx ÷ 2 + 1                                   # index where x = 0

    g0 = GaussianObstacle(2.0, 0.3)
    @test is_static(g0) && is_static(NoPotential())
    evaluate!(V, g0, grid, 0.0)
    Vh = Array(V)
    @test Vh[ic, ic] ≈ 2.0
    @test argmax(Vh) == CartesianIndex(ic, ic)

    # moving obstacle: the peak follows x0 + vx t
    g1 = GaussianObstacle(2.0, 0.3; vx = 1.0)
    @test !is_static(g1)
    evaluate!(V, g1, grid, 1.0)
    ipeak = argmax(Array(V))[1]
    @test abs(xcoords(grid)[ipeak] - 1.0) <= grid.dx

    # periodic box: an obstacle just beyond the right edge acts on the left edge
    g2 = GaussianObstacle(2.0, 0.3; x0 = π - 0.02)
    evaluate!(V, g2, grid, 0.0)
    @test Array(V)[1, ic] > 0.99 * 2.0                     # x = -π is 0.02 away through the boundary

    # StaticPotential from a host function
    evaluate!(V, StaticPotential((x, y) -> 0.25), grid, 0.0)
    @test all(Array(V) .≈ 0.25)

    # allocation
    @test allocate_potential(grid, NoPotential()) === nothing
    @test size(allocate_potential(grid, g0)) == (64, 64)
end

# ====================================================================== GPE model
@testset "GPE: uniform state is stationary" begin
    grid, p, model, state = make_setup()
    set_wavefunction!(state, model, uniform_density(grid, p))
    ψ0 = copy(state.ψk)
    evolve!(state, IFRK4(grid, model, 0.02), model, 50)
    @test maxdiff(state.ψk, ψ0) < 1e-4 * sqrt(p.n0) * ncells(grid)
    @test state.clock.iteration == 50
    @test state.clock.t ≈ 1.0
end

@testset "GPE: constant potential gives a global phase" begin
    # V = 0.7 μ everywhere → ψ(t) = √n0 exp(-i ω0 0.7 t)
    grid, p, model, state = make_setup(potential = StaticPotential((x, y) -> 0.7))
    set_wavefunction!(state, model, uniform_density(grid, p))
    evolve!(state, IFRK4(grid, model, 0.01), model, 50)
    update_physical!(state, model)
    exact = Complex{T}(sqrt(p.n0) * exp(-im * p.ω0 * 0.7 * state.clock.t))
    @test maxdiff(state.ψ, fill(exact, size(grid))) < 1e-4
end

@testset "GPE: linear rate" begin
    grid, p, model, state = make_setup()
    λ = linear_rate(model)
    k = T(3.0)
    @test λ(k, T(0)) ≈ -im * p.ω0 * p.ξ^2 * k^2
end

@testset "GPE: Bogoliubov dispersion" begin
    # ψ = √n0 (1 + ε cos(kx)) → a(t) = ε cos(ω t),  ħω/μ... ω = ω0 √(ξ²k²(ξ²k²+2))
    grid, p, model, state = make_setup()
    m, ε = 2, 1e-3
    k = 2π * m / grid.Lx
    x, y = xcoords(grid), ycoords(grid)
    ψ0 = Complex{T}.(sqrt(p.n0) .* ((1 .+ ε .* cos.(k .* x)) .+ 0 .* y'))
    set_wavefunction!(state, model, ψ0)

    evolve!(state, IFRK4(grid, model, 0.01), model, 100)
    update_physical!(state, model)
    a = real.(Array(state.ψ)) ./ sqrt(p.n0) .- 1
    amp = 2 * sum(a .* cos.(k .* x)) / length(a)           # projection on cos(kx)

    ω = p.ω0 * sqrt((p.ξ * k)^2 * ((p.ξ * k)^2 + 2))
    expected = ε * cos(ω * state.clock.t)
    println("Bogoliubov: ω t = ", ω * state.clock.t, ", amplitude = ", amp, ", expected = ", expected)
    @test abs(amp - expected) < 2e-5
    @test abs(expected) > 0.2 * ε                           # the test is not trivially zero
end

# ====================================================================== initial conditions + diagnostics
@testset "Vortex dipole, conservation" begin
    grid, p, model, state = make_setup(nx = 128)
    ψ0 = vortex_dipole(grid, p; separation = 8 * p.ξ)
    @test size(ψ0) == size(grid)
    @test minimum(abs2.(ψ0)) < 0.5 * p.n0                   # there are cores
    @test maximum(abs2.(ψ0)) ≈ p.n0 rtol = 1e-2             # far field density is n0

    set_wavefunction!(state, model, ψ0)
    N0, E0 = particle_number(state, model), energy(state, model)
    @test N0 > 0 && E0 isa Float64

    logger = ConservationLogger(state, model; io = devnull)
    evolve!(state, IFRK4(grid, model, 0.01), model, 100; callback = every(50, logger))
    N, E = particle_number(state, model), energy(state, model)
    println("Dipole: ΔN/N0 = ", (N - N0) / N0, ",  ΔE/|E0| = ", (E - E0) / abs(E0))
    @test abs(N - N0) / N0 < 1e-3
    @test abs(E - E0) / abs(E0) < 1e-2
    @test state.clock.iteration == 100

    # noise helper keeps the shape and changes the field
    ψn = add_noise!(copy(ψ0), p; amplitude = 0.05)
    @test size(ψn) == size(ψ0) && ψn != ψ0
end

@testset "Diagnostics: uniform state" begin
    grid, p, model, state = make_setup(nx = 32, n0 = 2.0)
    set_wavefunction!(state, model, uniform_density(grid, p))
    area = Float64(grid.Lx * grid.Ly)
    @test particle_number(state, model) ≈ p.n0 * area rtol = 1e-5
    @test energy(state, model) ≈ -p.n0 * area / 2 rtol = 1e-5     # ∫(n0/2 - n0)
end

# ====================================================================== callbacks
@testset "Callbacks" begin
    grid, p, model, state = make_setup(nx = 32)
    set_wavefunction!(state, model, uniform_density(grid, p))
    stepper = IFRK4(grid, model, 0.01)

    calls = Int[]
    rec = s -> (push!(calls, s.clock.iteration); nothing)
    evolve!(state, stepper, model, 10; callback = every(5, rec))
    @test calls == [5, 10]

    # early stop
    stopper = s -> s.clock.iteration >= 13 ? :stop : nothing
    evolve!(state, stepper, model, 100; callback = CallbackSet(every(1, rec), stopper))
    @test state.clock.iteration == 13
end

println("\nAll tests finished.")
