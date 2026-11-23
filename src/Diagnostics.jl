"""
    Diagnostics

Global quantities for the GPE, always returned as `Float64` on the host.

* `particle_number(state, model)`  N = ∫|ψ|² dA          (computed in Fourier space,
                                    no FFT needed)
* `energy(state, model)`           E/μ = ∫ [ ξ²|∇ψ|² + |ψ|⁴/(2 n0) - |ψ|² + V|ψ|² ] dA
                                    in units of μ (the same units as V). Refreshes
                                    `state.ψ` first.
* `ConservationLogger(state, model)` callable for `evolve!`; prints the relative
                                    drift of N and E with respect to their initial values.

They are meant to be called every few hundred steps, not every step: they use
temporaries.
"""
module Diagnostics

using Printf
using ..Grids
using ..Fields
using ..Models
using ..GrossPitaevskii

export particle_number, energy, ConservationLogger

function particle_number(state, m::GPE)
    # Parseval: Σ|ψ|² = Σ|ψk|² / (nx ny)
    g = m.grid
    ψk = evolved_field(m, state)
    return Float64(sum(abs2, ψk)) * Float64(cell_area(g)) / ncells(g)
    # ψ = state.ψ
    # return Float64(sum(abs2, ψ)) * Float64(cell_area(g))
end

function energy(state, m::GPE)
    g, p = m.grid, m.params
    ψk = evolved_field(m, state)
    update_physical!(state, m)
    ψ = state.ψ
    dA, N = Float64(cell_area(g)), ncells(g)
    kx, ky, n0 = g.kx, g.ky, p.n0

    kinetic = Float64(sum(@. (kx^2 + ky^2) * abs2(ψk))) * dA / N      # ∫|∇ψ|²
    # local_  = Float64(sum(x -> abs2(x)^2 / (2n0) - abs2(x), ψ)) * dA  # ∫(|ψ|⁴/2n0 - |ψ|²)
    local_  = Float64(sum(x -> (abs2(x)-n0)^2 / (2n0) - (abs2(x)-n0), ψ)) * dA  # ∫(|ψ|⁴/2n0 - |ψ|²)
    # local_  = Float64(sum(x -> (abs2(x)^2 - n0) / (2n0), ψ)) * dA  # ∫(|ψ|⁴/2n0)
    V = m.V
    pot = V === nothing ? 0.0 : Float64(sum(@. abs2(ψ) * V)) * dA     # ∫V|ψ|²

    Ekin = Float64(p.ξ)^2 * kinetic
    Eint = local_
    Epot = pot
    H = Ekin + Eint + Epot
    # return H, Ekin, Eint, Epot
    return H
end

"""
    ConservationLogger(state, model; io = stdout)

Callable `logger(state)`: prints time, iteration and the relative drifts
`ΔN/N0`, `ΔE/|E0|` (N0, E0 taken when the logger is created).
"""
mutable struct ConservationLogger{M, IOT<:IO}
    model::M
    N0::Float64
    E0::Float64
    # Ekin::Float64
    # Eint::Float64
    # Epot::Float64
    io::IOT
end

function ConservationLogger(state, model::GPE; io::IO = stdout)
    return ConservationLogger(model, particle_number(state, model), energy(state, model), io)
end

function (l::ConservationLogger)(state)
    N = particle_number(state, l.model)
    # E, Ekin, Eint, Epot = energy(state, l.model)
    E = energy(state, l.model)
    c = clock(state)
    @printf(l.io, "t = %9.3f   iter = %7d   ΔN/N0 = %+.2e   ΔE/|E0| = %+.2e\n",
            c.t, c.iteration, (N - l.N0) / l.N0, (E - l.E0) / abs(l.E0)) #, Ekin, Eint, Epot
    return nothing
end

end # module
