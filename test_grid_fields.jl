# Smoke tests for Architectures, Grid, Fields and FFT plans.
#
# Usage:
#   julia --project test_grid_fields.jl            # CPU, Float64
#   julia --project test_grid_fields.jl cuda       # CUDA, Float64
#   julia --project test_grid_fields.jl metal      # Metal, Float32
#
# Required packages: AbstractFFTs, FFTW (+ CUDA or Metal for the GPU backends).
# Works both with the layout  GPE2D/src/*.jl + GPE2D/test/*.jl  and with all the
# files in a single folder.

using Test, AbstractFFTs

backend = isempty(ARGS) ? "cpu" : lowercase(ARGS[1])
backend == "cuda"  && @eval using CUDA      # the backend must be loaded BEFORE building grids
backend == "metal" && @eval using Metal

nx, ny = 2048, 2048
Lx, Ly = 2π, 2π
using GPE2D

arch, T = backend == "cuda"  ? (CUDAGPU(),  Float64) :
          backend == "metal" ? (MetalGPU(), Float32) :
                               (CPU(),      Float64)

println("Testing on ", architecture_name(arch), " with ", T)

@testset "Grid" begin
    g = Grid(arch, T; nx = nx, ny = ny, Lx = Lx, Ly = Ly)
    show(stdout, g); println()

    @test size(g) == (nx, ny)
    @test half_size(g) == (nx÷2+1, ny)
    @test g.dx ≈ T(Lx / nx)
    @test float_type(g) == T

    # coordinates
    @test xcoords(g)[1] ≈ -g.Lx / 2
    @test ycoords(g)[2] - ycoords(g)[1] ≈ g.dy

    # wavenumbers agree with AbstractFFTs ordering (k = 2π * cycles/length)
    @test kxcoords(g) ≈ T.(2π .* fftfreq(nx, 1 / g.dx))
    @test kycoords(g) ≈ T.(2π .* fftfreq(ny, 1 / g.dy))
    @test kxcoords(g)[nx ÷ 2 + 1] ≈ -π / g.dx           # Nyquist stored as -n/2

    # broadcast shapes on the device
    @test size(g.kx) == (nx, 1) && size(g.ky) == (1, ny)
    @test size(wavenumber_squared(g)) == (nx, ny)

    # dealiasing mask: 0/1 values, circular 2/3 rule keeps about π/4·(2/3)² ≈ 35 % of modes
    m = Array(g.mask)
    @test all(x -> x == 0 || x == 1, m)
    @test 0.30 < sum(m) / length(m) < 0.40
    @test m[1, 1] == 1                                    # k = 0 is always kept

    g_none = Grid(arch, T; nx = 16, Lx = 1.0, dealias = :none)
    @test all(Array(g_none.mask) .== 1)

    # input validation (these checks happen before any array is allocated)
    @test_throws ArgumentError Grid(arch, T; nx = 15, Lx = 1.0)
    @test_throws ArgumentError Grid(MetalGPU(), Float64; nx = 16, Lx = 1.0)
    @test_throws ArgumentError Grid(arch, T; nx = 16, Lx = 1.0, dealias = :bogus)
end

@testset "Fields" begin
    g = Grid(arch, T; nx = nx, ny = ny, Lx = Lx, Ly = Ly)

    st = State(g, (ψ = ComplexPhysical(), ψk = FullSpectral(), ρk = HalfSpectral(), ρ = RealPhysical()))
    @test size(st.ψ) == (nx, ny) && eltype(st.ψ) == Complex{T}
    @test size(st.ρk) == (nx÷2+1, ny) && eltype(st.ρk) == Complex{T}
    @test size(st.ρ) == (nx, ny) && eltype(st.ρ) == T
    @test st.clock.t == 0.0 && st.clock.iteration == 0
    @test all(Array(st.ψ) .== 0)
    @test :ψk in propertynames(st)
    show(stdout, MIME"text/plain"(), st)

    # copy / copyto! (no aliasing, no reallocation)
    st.ψ .= one(Complex{T})
    advance!(st.clock, 0.5)
    st2 = copy(st)
    st.ψ .= 2 * one(Complex{T})
    @test all(Array(st2.ψ) .== 1)
    @test st2.clock.t == 0.5 && st2.clock.iteration == 1
    copyto!(st2, st)
    @test all(Array(st2.ψ) .== 2)

    # GPE convenience state + workspace
    gs = gpe_state(g)
    @test size(gs.ψk) == (nx, ny)
    ws = Workspace(g, repeated_layout(FullSpectral(), :k1, :k2, :k3, :k4))
    @test propertynames(ws) == (:buffers, :k1, :k2, :k3, :k4)
    @test memory_bytes(ws) == 4 * nx * ny * sizeof(Complex{T})
    show(stdout, MIME"text/plain"(), ws)

    # host transfer
    h = host_fields(st)
    @test h.ψ isa Array{Complex{T},2}

    # in-place broadcasting works on the device
    ws.k1 .= one(Complex{T})
    ws.k2 .= ws.k1 .* 3
    @test all(Array(ws.k2) .== 3)
end

@testset "FFT plans" begin
    # nx, ny = 64, 32
    g = Grid(arch, T; nx = nx, ny = ny, Lx = Lx, Ly = Ly)
    plans = FFTPlans(g)
    println(plans)
    st = State(g, (ψ = ComplexPhysical(), ψk = FullSpectral()))
    back = allocate_field(g, ComplexPhysical())

    # A plane wave with integer mode numbers (m, n) must give a single spike at the
    # index where Grid says k = (kx0, ky0): this checks that the plans and the
    # wavenumbers in Grid use the same ordering.
    m, n = 3, -2
    kx0, ky0 = 2π * m / g.Lx, 2π * n / g.Ly
    xh, yh = Float64.(xcoords(g)), Float64.(ycoords(g))
    ψ0 = Complex{T}.(exp.(im .* (kx0 .* xh .+ ky0 .* yh')))
    copyto!(st.ψ, to_device(arch, ψ0))

    to_spectral!(st.ψk, st.ψ, plans)
    ψk = Array(st.ψk)
    i, j = mod(m, nx) + 1, mod(n, ny) + 1
    @test kxcoords(g)[i] ≈ T(kx0) && kycoords(g)[j] ≈ T(ky0)
    @test abs(ψk[i, j]) ≈ T(nx * ny) rtol = 1e-3
    ψk[i, j] = 0
    @test maximum(abs, ψk) < 0.05 * nx * ny * 1e-2          # everything else is ~0

    # source untouched, round trip recovers the input
    @test Array(st.ψ) == ψ0
    to_physical!(back, st.ψk, plans)
    @test maximum(abs.(Array(back) .- ψ0)) < 1e-4

    # Parseval:  Σ|ψ|² = Σ|ψk|² / (nx·ny)
    s_phys = sum(abs2, Array(st.ψ))
    s_spec = sum(abs2, Array(st.ψk)) / (nx * ny)
    @test s_phys ≈ s_spec rtol = 1e-3

    # in-place API and size check
    forward!(st.ψ, plans); backward!(st.ψ, plans)
    @test maximum(abs.(Array(st.ψ) .- ψ0)) < 1e-4
    @test_throws DimensionMismatch forward!(allocate_field(Grid(arch, T; nx = 16, Lx = 1.0), FullSpectral()), plans)
end

println("\nAll tests finished.")
