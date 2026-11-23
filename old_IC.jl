"""
    InitialConditions

Initial wavefunctions for the GPE. All functions are evaluated on the HOST (they run
once) and return an `nx × ny` `Matrix{Complex{T}}`; pass the result to
`set_wavefunction!(state, model, ψ0)`, which moves it to the device and dealiases it.

* `uniform_density(grid, p)`          ψ = √n0
* `vortices(grid, p, list)`           product of vortex profiles; `list` is a collection
                                      of `(x, y, κ)` with `κ = ±1` the circulation
* `vortex_dipole(grid, p; ...)`       vortex–antivortex pair
* `add_noise!(ψ, p; amplitude, rng)`  complex white noise (relative to √n0)

Vortex core: Padé approximation of Berloff (2004). It is written for the unit length
`√2 ξ`, i.e. `r_B = r/(√2 ξ)`.

Caveat: the product ansatz is not exactly periodic. The phase mismatch at the box
boundary is small if the vortices are far from the edges and not too far apart from
each other; it radiates a little sound. For clean states, relax in imaginary time
(planned) before using them.
"""
module InitialConditions

using Random
using ..Grids
using ..GrossPitaevskii: GPEParameters
using ..FFT

export uniform_density, vortices, vortex_dipole, add_noise!, waves!

uniform_density(grid::Grid{T}, p::GPEParameters) where {T} =
    fill(Complex{T}(sqrt(p.n0)), grid.nx, grid.ny)

# minimum-image displacement in a periodic box
_wrap(d, L) = d - L * round(d / L)
# _wrap2(d, L) = 2*sin( .5*(d-.5*L)/(L/(2*π)) ) * L/(2*π)
_wrap2(d, L) = (L / π) * sin(π * d / L)
# _wrap2(d, L) = 2*sin( .5*(d-L*π)/(L) ) * L

# Padé approximant of the vortex density profile, in units of √2 ξ  (Berloff 2004)
function _core_amplitude(rB2)
    a1, a2, b1 = 11/32, 11/384, 1/3
    b2 = a2
    return sqrt(rB2 * (a1 + a2 * rB2) / (1 + b1 * rB2 + b2 * rB2^2))
end

function window(u, L)
    s = abs(_wrap(u, L)) / (L / 2)
    s0 = 1 - 0.25
    return s <= s0 ? 1.0 : 0.5 * (1 + cos(π * (s - s0) / 0.25))
end

"""
    vortices(grid, p, list)

Wavefunction with vortices at the positions in `list = [(x, y, κ), ...]`, κ = ±1.
"""
function vortices(grid::Grid{T}, p::GPEParameters, list) where {T}
    x, y = Float64.(xcoords(grid)), Float64.(ycoords(grid))
    Lx, Ly = Float64(grid.Lx), Float64(grid.Ly)

    # Neutrality is required for a periodic phase
    sum(κ for (_, _, κ) in list) == 0 ||
        error("Periodic phase requires zero net circulation")

    s2 = 1 / (2 * Float64(p.ξ)^2)                      # r_B² = r² / (2ξ²)
    ψ = ones(ComplexF64, grid.nx, grid.ny)
    for (x0, y0, κ) in list
        for j in eachindex(y), i in eachindex(x)
            dx = _wrap2(x[i] - x0, Lx)
            dy = _wrap2(y[j] - y0, Ly)
            ψ[i, j] *= _core_amplitude((dx^2 + dy^2) * s2) * cis(κ * atan(dy, dx))
            # ψ[i, j] *= _core_amplitude((dx^2 + dy^2) * s2) * cis(κ * atan(dy, dx) * window(dx,Lx)*window(dy,Ly))
            # amp = _core_amplitude((dx^2 + dy^2) * s2)
            # ψ[i, j] *= amp * ((dx + i·dy)/r)^κ
        end
    end
    return Complex{T}.(sqrt(Float64(p.n0)) .* ψ)
end

"""
    vortex_dipole(grid, p; separation = 8p.ξ, center = (0, 0))

Vortex (+1) at `center + (0, d/2)` and antivortex (−1) at `center − (0, d/2)`.
The pair moves along x with speed ≈ ħ/(m d) = √2 c ξ / d.
"""
function vortex_dipole(grid::Grid, p::GPEParameters; separation::Real = 8 * p.ξ,
                       center = (0.0, 0.0))
    cx, cy = center
    d = Float64(separation)
    return vortices(grid, p, [(cx, cy + d / 2, +1), (cx, cy - d / 2, -1)])
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
