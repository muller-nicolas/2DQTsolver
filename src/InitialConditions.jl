"""
    InitialConditions

Initial wavefunctions for the GPE. All functions are evaluated on the HOST (they run
once) and return an `nx × ny` `Matrix{Complex{T}}`; pass the result to
`set_wavefunction!(state, model, ψ0)`, which moves it to the device and dealiases it.

* `uniform_density(grid, p)`          ψ = √n0
* `vortices(grid, p, list)`           vortices with an EXACTLY periodic phase; `list` is a
                                      collection of `(x, y, κ)` with integer circulation `κ`
                                      and zero net circulation
* `vortex_dipole(grid, p; ...)`       vortex–antivortex pair
* `add_noise!(ψ, p; amplitude, rng)`  complex white noise (relative to √n0)

Periodic phase (why a plain product of `atan2` is wrong)
--------------------------------------------------------
Taking ψ = ∏ᵢ (single vortex at rᵢ) with minimum-image distances gives a phase that is
discontinuous across the box boundary, even if the net circulation is zero. In a
periodic box the phase must wind by an integer multiple of 2π around every closed
loop, including the two loops that go around the box. For the phase of a neutral
set of vortices this forces a UNIFORM background flow, which a product of atan2 does
not contain.

The construction used here is exact. With z = x + iy, zᵢ = xᵢ + iyᵢ and the Jacobi
theta function θ₁,

    f(z) = ∏ᵢ θ₁((z − zᵢ)/Lx | τ)^κᵢ,      τ = i Ly/Lx,

has zeros of order κᵢ at the vortices and is periodic along x. Along y it is
periodic up to the constant factor exp(2πi Σκᵢzᵢ/Lx), whose phase is cancelled by the
linear term β y with β = −2π Σκᵢxᵢ/(Lx Ly). The phase is  φ = arg f + β y.

The remaining freedom is an integer winding (2π nx x/Lx + 2π ny y/Ly). By default
(`winding = :minimal`) it is chosen to give the smallest mean flow velocity. What is
left is the unavoidable part: for a dipole of separation d along y the mean velocity is
2π d/(Lx Ly) (in units of ħ/m), directed along x. Pass `winding = (nx, ny)` to add
other integer windings.

The density uses the Padé core of Berloff (2004), written for the unit length
`√2 ξ`, i.e. `r_B = r/(√2 ξ)`, with minimum-image distances.
"""
module InitialConditions

using Random
using FFTW: fft
using ..Architectures: to_host
using ..Grids
using ..GrossPitaevskii: GPEParameters

export uniform_density, vortices, vortex_dipole, add_noise!, vortex_quadrupole, waves!, vortex_random

uniform_density(grid::Grid{T}, p::GPEParameters) where {T} =
    fill(Complex{T}(sqrt(p.n0)), grid.nx, grid.ny)

# minimum-image displacement in a periodic box
_wrap(d, L) = d - L * round(d / L)

# Padé approximant of the vortex density profile, in units of √2 ξ  (Berloff 2004) (for a simply charged vortex)
function _core_amplitude(rB2)
    a1, a2, b1, b2 = 11/32, 11/384, 1/3, 11/384
    return sqrt(rB2 * (a1 + a2 * rB2) / (1 + b1 * rB2 + b2 * rB2^2))
end

# ---------------------------------------------------------------- theta function
"""
    _theta1(x, y, x0, y0, Lx, Ly)

`θ₁(u | τ)` with `u = ((x − x0) + i (y − y0))/Lx` and `τ = i Ly/Lx`, on the grid
spanned by the vectors `x` and `y` (an `nx × ny` complex matrix), from the series

    θ₁(u|τ) = 2 Σ_{n≥0} (−1)ⁿ q^{(n+1/2)²} sin((2n+1)π u),     q = exp(−π Ly/Lx),

using sin(a + ib) = sin a cosh b + i cos a sinh b so that everything is built from
outer products of 1D arrays. The overall positive scale does not matter (only the
argument is used), so overflow is not an issue for reasonable aspect ratios.
"""
function _theta1(x::AbstractVector, y::AbstractVector, x0, y0, Lx, Ly)
    r = Ly / Lx
    maxdy = maximum(abs, y .- y0)
    # size of the n-th term (log), to decide how many terms are needed
    lm = [log(2) - π * r * (n + 0.5)^2 + (2n + 1) * π * maxdy / Lx for n in 0:400]
    lmax = maximum(lm)
    nterms = findlast(l -> l >= lmax - 40, lm)           # terms n = 0, ..., nterms-1
    nterms >= length(lm) && error("box too anisotropic for the theta-function series (Ly/Lx = $r)")
    θre = zeros(length(x), length(y))
    θim = zeros(length(x), length(y))
    for n in 0:(nterms - 1)
        w  = 2 * (-1)^n * exp(-π * r * (n + 0.5)^2)
        a  = (2n + 1) * π .* (x .- x0) ./ Lx
        b  = (2n + 1) * π .* (y .- y0) ./ Lx
        θre .+= w .* (sin.(a) * cosh.(b)')
        θim .+= w .* (cos.(a) * sinh.(b)')
    end
    return complex.(θre, θim)
end

# Integer windings (nx, ny) that minimise the mass-weighted mean velocity of ψ.
# The mean velocity is  Σ k |ψ̂_k|² / Σ |ψ̂_k|²  (Nyquist mode excluded: odd derivative).
function _minimal_winding(ψ::AbstractMatrix, grid::Grid)
    w  = abs2.(fft(ψ))
    kx = Float64.(vec(to_host(grid.kx1)))
    ky = Float64.(vec(to_host(grid.ky1)))
    tot = sum(w)
    vx = sum(kx .* w) / tot
    vy = sum(w .* ky') / tot
    return round(Int, -vx * Float64(grid.Lx) / 2π), round(Int, -vy * Float64(grid.Ly) / 2π)
end

# ---------------------------------------------------------------- vortices
"""
    vortices(grid, p, list; winding = :minimal)

Wavefunction with vortices at the positions in `list = [(x, y, κ), ...]` (integer κ,
for example ±1). The net circulation must be zero, which is required for a periodic
phase. The phase is exactly periodic (see the module documentation), and includes the
uniform flow that periodicity forces on the state.

`winding = :minimal` (default) adds the integer winding that minimises the mean flow;
`winding = (nx, ny)` adds the given integer windings to the theta-function phase instead.
"""
function vortices(grid::Grid{T}, p::GPEParameters, list; winding = :minimal) where {T}
    isempty(list) && return uniform_density(grid, p)
    all(isinteger(κ) for (_, _, κ) in list) ||
        error("The circulations κ must be integers")
    sum(κ for (_, _, κ) in list) == 0 ||
        error("Periodic phase requires zero net circulation (the κ must add up to 0)")

    x, y = Float64.(xcoords(grid)), Float64.(ycoords(grid))
    Lx, Ly = Float64(grid.Lx), Float64(grid.Ly)
    s2 = 1 / (2 * Float64(p.ξ)^2)                      # r_B² = r² / (2ξ²)

    # --- amplitude: product of Padé cores (minimum-image distances)
    ρ = ones(Float64, grid.nx, grid.ny)
    for (x0, y0, _) in list
        for j in eachindex(y), i in eachindex(x)
            dx = _wrap(x[i] - x0, Lx)
            dy = _wrap(y[j] - y0, Ly)
            ρ[i, j] *= _core_amplitude((dx^2 + dy^2) * s2)
        end
    end

    # --- phase: φ = Σ κᵢ arg θ₁(zᵢ) + β y   (positions folded into the box first)
    pos = [(_wrap(Float64(x0), Lx), _wrap(Float64(y0), Ly), Int(κ)) for (x0, y0, κ) in list]
    φ = zeros(Float64, grid.nx, grid.ny)
    for (x0, y0, κ) in pos
        φ .+= κ .* angle.(_theta1(x, y, x0, y0, Lx, Ly))
    end
    β = -2π * sum(κ * x0 for (x0, _, κ) in pos) / (Lx * Ly)
    φ .+= β .* y'

    ψ = ρ .* cis.(φ)

    # --- integer winding
    nxw, nyw = winding === :minimal ? _minimal_winding(ψ, grid) : winding
    if nxw != 0 || nyw != 0
        ψ .*= cis.(2π .* (nxw .* x ./ Lx .+ nyw .* y' ./ Ly))
    end

    return Complex{T}.(sqrt(Float64(p.n0)) .* ψ)
end

"""
    vortex_dipole(grid, p; separation = 8p.ξ, center = (0, 0), winding = :minimal)

Vortex (+1) at `center + (0, d/2)` and antivortex (−1) at `center − (0, d/2)`, with an
exactly periodic phase. The pair moves along x with speed ≈ ħ/(m d) = √2 c ξ / d. The
state necessarily contains a uniform flow of magnitude 2π d/(Lx Ly) along x (units of
ħ/m), see the module documentation.
"""
function vortex_dipole(grid::Grid, p::GPEParameters; separation::Real = 8 * p.ξ,
                       center = (0.0, 0.0), winding = :minimal)
    cx, cy = center
    d = Float64(separation)
    return vortices(grid, p, [(cx, cy + d / 2, +1), (cx, cy - d / 2, -1)]; winding = winding)
end

function vortex_quadrupole(grid::Grid, p::GPEParameters; separation::Real = 8 * p.ξ,
                       center = (0.0, 0.0), winding = :minimal)
    cx, cy = center
    d = Float64(separation)
    return vortices(grid, p, [(cx, cy + d / 2, +1), (cx, cy - d / 2, -1), (cx + d / 2, cy, +1), (cx - d / 2, cy, -1)]; winding = winding)
end

function vortex_random(grid::Grid, p::GPEParameters; nvortices::Int, seed::Int = 1234, winding = :minimal)
    pxpos = rand(Float64, nvortices÷2) .* Float64(grid.Lx)
    pypos = rand(Float64, nvortices÷2) .* Float64(grid.Ly)
    pxneg = rand(Float64, nvortices÷2) .* Float64(grid.Lx)
    pyneg = rand(Float64, nvortices÷2) .* Float64(grid.Ly)
    # Positives first, then negatives
    list = vcat([(pxpos[i], pypos[i], +1) for i in 1:nvortices÷2],
                [(pxneg[i], pyneg[i], -1) for i in 1:nvortices÷2])
    return vortices(grid, p, list; winding = winding)
end

"""
    add_noise!(ψ, p; amplitude = 0.01, rng = Random.default_rng())

Add complex Gaussian white noise of relative amplitude `amplitude` (vs √n0), in place.
High-k components are removed later by `set_wavefunction!` (dealiasing).
"""
function add_noise!(ψ::AbstractMatrix{Complex{T}}, p::GPEParameters; amplitude::Real = 0.01,
                    rng::AbstractRNG = Random.default_rng()) where {T}
    s = T(amplitude * sqrt(Float64(p.n0) / 2))
    ψ .+= s .* complex.(randn(rng, T, size(ψ)...), randn(rng, T, size(ψ)...))
    return ψ
end

"""
    waves!(state, model, params)

Add random waves between kmin and kmax to the wavefunction in place. The amplitude is relative to √n0.
"""
function waves!(state, model, params)
    mask = (model.grid.k2 .> params.kicmin^2) .& (model.grid.k2 .< params.kicmax^2)
    state.ψk[mask] .= cis.(2π .* rand(count(mask)))
    to_physical!(state.ψ, state.ψk, model.plans)
    E = sum(abs2, state.ψ) / (model.grid.nx*model.grid.ny)
    # E = sum(abs2, state.ψ) * Float64(cell_area(model.grid)) #/ ncells(model.grid) / (model.grid.nx*model.grid.ny)
    state.ψ .= sqrt(params.n0)*( (1. - params.noise_amplitude) .+ params.noise_amplitude .* state.ψ ./ sqrt(E) )
    E = sum(abs2, state.ψ) / (model.grid.nx*model.grid.ny)
    # E = sum(abs2, state.ψ) * Float64(cell_area(model.grid)) #/ ncells(model.grid) / (model.grid.nx*model.grid.ny)
    state.ψ ./= sqrt(E)  # normalize to unit energy
    to_spectral!(state.ψk, state.ψ, model.plans)
    return state.ψ
end

end # module
