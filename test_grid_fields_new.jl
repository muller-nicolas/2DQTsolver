# Smoke tests for Architectures, Grid and Fields.
#
# Usage (from the package root, after `] dev .` in your environment):
#   julia --project test/test_grid_fields.jl            # CPU, Float64
#   julia --project test/test_grid_fields.jl cuda       # CUDA, Float64
#   julia --project test/test_grid_fields.jl metal      # Metal, Float32

using Test, AbstractFFTs

backend = isempty(ARGS) ? "cpu" : lowercase(ARGS[1])
backend == "cuda"  && @eval using CUDA
backend == "metal" && @eval using Metal

include("GPE2D.jl")
using .GPE2D

arch, T = backend == "cuda"  ? (CUDAGPU(),  Float64) :
          backend == "metal" ? (MetalGPU(), Float32) :
                               (CPU(),      Float64)

N = 128
L = 2π
println("Testing on ", architecture_name(arch), " with ", T)

@testset "Grid" begin
    g = Grid(arch, T; nx = 64, ny = 32, Lx = 16π, Ly = 8π)
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
    g = Grid(arch, T; nx = 64, ny = 32, Lx = 16π, Ly = 8π)

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

println("\nAll tests finished.")
