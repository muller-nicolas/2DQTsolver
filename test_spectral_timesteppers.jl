# Tests for Spectral.jl and TimeSteppers.jl
#
#   julia --project test_spectral_timesteppers.jl            # CPU, Float64
#   julia --project test_spectral_timesteppers.jl cuda       # CUDA, Float64
#   julia --project test_spectral_timesteppers.jl metal      # Metal, Float32

using Test

backend = isempty(ARGS) ? "cpu" : lowercase(ARGS[1])
backend == "cuda"  && @eval using CUDA      # load the backend BEFORE GPE2D
backend == "metal" && @eval using Metal
using GPE2D
import GPE2D: linear_rate, nonlinear!, evolved_field   # to add methods for our test model

arch, T = backend == "cuda"  ? (CUDAGPU(),  Float64) :
          backend == "metal" ? (MetalGPU(), Float32) :
                               (CPU(),      Float64)
println("Testing on ", architecture_name(arch), " with ", T)

maxdiff(a, b) = maximum(abs.(Array(a) .- Array(b)))

# ====================================================================== Spectral
@testset "Spectral" begin
    nx, ny = 64, 64
    Lx, Ly = 2π, 2π
    g = Grid(arch, T; nx = nx, ny = ny, Lx = Lx, Ly = Ly)     # integer wavenumbers
    plans = FFTPlans(g)
    x, y = xcoords(g), ycoords(g)

    tospec(fh) = (fk = allocate_field(g, FullSpectral());
                  to_spectral!(fk, to_device(arch, fh), plans); fk)
    tophys(fk) = (f = allocate_field(g, ComplexPhysical());
                  Array(to_physical!(f, fk, plans)))

    # f = sin(3x) cos(2y)
    f   = Complex{T}.(sin.(3 .* x) .* cos.(2 .* y'))
    fx  = Complex{T}.(3 .* cos.(3 .* x) .* cos.(2 .* y'))
    fy  = Complex{T}.(-2 .* sin.(3 .* x) .* sin.(2 .* y'))
    lap = -13 .* f
    fk  = tospec(f)
    out = allocate_field(g, FullSpectral())

    ddx!(out, fk, g);        @test maxdiff(tophys(out), fx)  < 5e-4
    ddy!(out, fk, g);        @test maxdiff(tophys(out), fy)  < 5e-4
    laplacian!(out, fk, g);  @test maxdiff(tophys(out), lap) < 5e-3

    gx, gy = allocate_field(g, FullSpectral()), allocate_field(g, FullSpectral())
    gradient!(gx, gy, fk, g)
    @test maxdiff(tophys(gx), fx) < 5e-4 && maxdiff(tophys(gy), fy) < 5e-4

    # inverse Laplacian undoes the Laplacian (f has zero mean)
    inverse_laplacian!(out, tospec(lap), g)
    @test maxdiff(tophys(out), f) < 5e-4

    # Nyquist mode: odd derivative removes it, even derivative keeps it
    nyq = Complex{T}.(cos.(32 .* x) .+ 0 .* y')              # pure Nyquist mode in x
    nk = tospec(nyq)
    ddx!(out, nk, g);        @test maximum(abs, Array(out)) < 1e-3
    laplacian!(out, nk, g);  @test maxdiff(tophys(out), -1024 .* nyq) < 1e-2
    @test Array(g.kx1)[nx ÷ 2 + 1] == 0 && Array(g.kx)[nx ÷ 2 + 1] != 0

    # dealiasing: mode (30, 0) is outside the 2/3 mask, mode (5, 3) is inside
    A = zeros(Complex{T}, nx, ny)
    A[mod(30, nx) + 1, 1] = 1
    A[mod(5, nx) + 1, mod(3, ny) + 1] = 1
    Ad = to_device(arch, A)
    dealias!(Ad, g)
    Ah = Array(Ad)
    @test Ah[mod(30, nx) + 1, 1] == 0
    @test Ah[mod(5, nx) + 1, mod(3, ny) + 1] == 1

    # linear rate and integrating factor:  λ = -(i/2) k²,  E = exp(λ τ)
    λ = LaplacianRate(T, 0.5im)
    kx, ky = kxcoords(g), kycoords(g)
    @test λ(kx[3], ky[2]) ≈ -0.5im * (kx[3]^2 + ky[2]^2)
    τ = 0.3
    E = Array(integrating_factor(g, λ, τ))
    @test maxdiff(E, exp.(-0.5im .* (kx .^ 2 .+ ky' .^ 2) .* T(τ))) < 1e-4
    @test all(abs.(E) .≈ 1)

    apply_rate!(out, fk, λ, g)
    @test maxdiff(tophys(out), Complex{T}(0, -6.5) .* f) < 5e-3             # λ = -(i/2)·13 on f
end

# ====================================================================== test model
# Plane-wave test model:  ∂ₜψ = (i/2)∇²ψ − i g |ψ|²ψ.
# For ψ = A exp(i(k·x − ωt)) the solution is exact, with ω = k²/2 + g A².
struct TestNLS{R,G,P,C} <: AbstractModel
    g::R
    grid::G
    plans::P
    ψ::C                      # physical-space buffer owned by the model
end
linear_rate(m::TestNLS) = LaplacianRate(float_type(m.grid), 0.5im)
evolved_field(::TestNLS, state) = state.ψk
function nonlinear!(out, Uk, m::TestNLS, t)
    to_physical!(m.ψ, Uk, m.plans)
    ψ, gT = m.ψ, m.g
    z = zero(gT)
    @. ψ = complex(z, -gT) * abs2(ψ) * ψ
    to_spectral!(out, ψ, m.plans)
    out .*= m.grid.mask
    return out
end

# Evolve the plane wave (mode (2,1), amplitude 1) up to tfinal; return max error vs exact.
function run_nls(make_stepper, dt, tfinal; g = 4.0)
    grid  = Grid(arch, T; nx = 16, ny = 16, Lx = 2π, Ly = 2π)
    plans = FFTPlans(grid)
    model = TestNLS(T(g), grid, plans, allocate_field(grid, ComplexPhysical()))
    state = State(grid, (ψ = ComplexPhysical(), ψk = FullSpectral()))

    x, y = xcoords(grid), ycoords(grid)
    m, n = 2, 1
    ψ0 = Complex{T}.(exp.(im .* (m .* x .+ n .* y')))
    copyto!(state.ψ, to_device(arch, ψ0))
    to_spectral!(state.ψk, state.ψ, plans)

    stepper = make_stepper(grid, model, dt)
    nsteps = round(Int, tfinal / dt)
    evolve!(state, stepper, model, nsteps)

    to_physical!(state.ψ, state.ψk, plans)
    ω = (m^2 + n^2) / 2 + g
    exact = ψ0 .* Complex{T}(exp(-im * ω * state.clock.t))
    return maxdiff(state.ψ, exact), state
end

# ====================================================================== TimeSteppers
@testset "TimeSteppers" begin
    # With g = 0 only the linear term acts: IFRK4 is exact for any dt
    err, _ = run_nls(IFRK4, 1.0, 3.0; g = 0.0)
    println("IFRK4, linear problem, dt = 1: error = ", err)
    @test err < 1e-4

    # Both steppers are 4th order
    # for (name, make) in (("IFRK4", IFRK4), ("RK4", RK4))
    #     dts  = (0.2, 0.1, 0.05)
    #     errs = [run_nls(make, dt, 1.0)[1] for dt in dts]
    #     p = log2.(errs[1:2] ./ errs[2:3])
    #     println(rpad(name, 6), " errors = ", errs, "   observed orders = ", p)
    #     @test all(3.0 .< p .< 5.0)
    # end

    name, make = "IFRK4", IFRK4
    dts  = (0.2, 0.1, 0.05)
    errs = [run_nls(make, dt, 1.0)[1] for dt in dts]
    p = log2.(errs[1:2] ./ errs[2:3])
    println(rpad(name, 6), " errors = ", errs, "   observed orders = ", p)
    @test all(3.0 .< p .< 5.0)

    # State-level API: clock, norm conservation
    err, st = run_nls(IFRK4, 0.05, 1.0)
    @test st.clock.iteration == 20
    @test st.clock.t ≈ 1.0
    @test err < 1e-3

    # Stepper bookkeeping: buffers, set_dt!
    grid  = Grid(arch, T; nx = 16, ny = 16, Lx = 2π, Ly = 2π)
    plans = FFTPlans(grid)
    model = TestNLS(T(1), grid, plans, allocate_field(grid, ComplexPhysical()))
    s = IFRK4(grid, model, 0.1)
    @test timestep(s) == 0.1
    @test memory_bytes(s.ws) == 3 * 16 * 16 * sizeof(Complex{T})
    set_dt!(s, 0.05)
    @test timestep(s) == 0.05
    @test maxdiff(s.E,  integrating_factor(grid, linear_rate(model), 0.05))  < 1e-6
    @test maxdiff(s.E2, integrating_factor(grid, linear_rate(model), 0.025)) < 1e-6
end

println("\nAll tests finished.")
