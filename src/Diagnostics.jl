"""
    Diagnostics

Global quantities for the GPE, always returned as `Float64` on the host.

* `particle_number(state, model)`  N = ∫|ψ|² dA          (computed in Fourier space,
                                    no FFT needed)
* `energy(state, model)`           E/μ = ∫ [ ξ²|∇ψ|² + |ψ|⁴/(2 n0) - |ψ|² + V|ψ|² ] dA
                                    in units of μ (the same units as V). Refreshes
                                    `state.ψ` first.
* `chemical_potential(state, model)` μ = ⟨ψ|(-ξ²∇² + |ψ|²/n0 + V)|ψ⟩/N in units of μ
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

export particle_number, energy, chemical_potential, ConservationLogger

function particle_number(state, m::GPE)
    # Parseval: Σ|ψ|² = Σ|ψk|² / (nx ny)
    g = m.grid
    ψk = evolved_field(m, state)
    return Float64(sum(abs2, ψk)) * Float64(cell_area(g)) / ncells(g)
    # ψ = state.ψ
    # return Float64(sum(abs2, ψ)) * Float64(cell_area(g))
end

# Integrals shared by `energy` and `chemical_potential` (all Float64):
#   kin = ξ² ∫|∇ψ|²,   int4 = ∫|ψ|⁴/n0,   pot = ∫V|ψ|²,   N = ∫|ψ|²
function _energy_terms(state, m::GPE)
    g, p = m.grid, m.params
    ψk = evolved_field(m, state)
    update_physical!(state, m)
    ψ = state.ψ
    dA, Nk = Float64(cell_area(g)), ncells(g)
    kx, ky, n0 = g.kx, g.ky, p.n0

    kin  = Float64(p.ξ)^2 * Float64(sum(@. (kx^2 + ky^2) * abs2(ψk))) * dA / Nk   # ξ²∫|∇ψ|²
    int4 = Float64(sum(x -> abs2(x)^2, ψ)) * dA / Float64(n0)                     # ∫|ψ|⁴/n0
    # int4 = Float64(sum(x -> (abs2(x)-Float64(n0))^2 / (2*Float64(n0)) - (abs2(x)-Float64(n0)), ψ)) * dA  # ∫(|ψ|⁴/2n0 - |ψ|²)
    V = m.V
    pot = V === nothing ? 0.0 : Float64(sum(@. abs2(ψ) * V)) * dA                 # ∫V|ψ|²
    N = Float64(sum(abs2, ψk)) * dA / Nk                                          # ∫|ψ|²
    return (kin = kin, int4 = int4, pot = pot, N = N)
end

function energy(state, m::GPE)
    t = _energy_terms(state, m)
    return t.kin + .5*t.int4 - t.N + t.pot
end


"""
    chemical_potential(state, model)

Chemical potential of the state in units of μ = g n0, `μ_ψ = ⟨ψ|H̃|ψ⟩/⟨ψ|ψ⟩` with
`H̃ = -ξ²∇² + |ψ|²/n0 + V`. For a stationary state of the GPE in the frame rotating at
μ (the equation contains the shift -1) it equals 1 at infinity of a uniform fluid; for
an RGLE run at fixed particle number it is the quantity that adapts.
"""
function chemical_potential(state, m::GPE)
    t = _energy_terms(state, m)
    return (t.kin + t.int4 + t.pot) / t.N
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
    io::IOT
end

function ConservationLogger(state, model::GPE; io::IO = stdout)
    return ConservationLogger(model, particle_number(state, model), energy(state, model), io)
end

function (l::ConservationLogger)(state)
    N = particle_number(state, l.model)
    E = energy(state, l.model)
    c = clock(state)
    @printf(l.io, "t = %9.3f   iter = %7d   ΔN/N0 = %+.2e   ΔE/|E0| = %+.2e\n",
            c.t, c.iteration, (N - l.N0) / l.N0, (E - l.E0) / abs(l.E0))
    return nothing
end

end # module
