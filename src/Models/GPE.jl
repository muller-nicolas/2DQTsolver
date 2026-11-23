"""
    GrossPitaevskii

The 2D Gross–Pitaevskii equation in terms of the speed of sound `c`, the healing
length `ξ` and the mean density `n0`.

With  ξ = ħ/√(2 m g n0),  c = √(g n0/m)  and  μ = g n0  (m = 1), the equation in the
frame rotating at μ/ħ is

    ∂ₜψ = -i ω0 [ -ξ² ∇²ψ + (|ψ|²/n0 - 1 + V) ψ ],        ω0 = μ/ħ = c/(√2 ξ),

where `V` is the external potential in units of μ.

Splitting for the time steppers (see `Models.jl`):

    linear part     λ(k)  = -i ω0 ξ² k²                     →  LaplacianRate(i ω0 ξ²)
    nonlinear part  N(ψ)  = γ F[ (|ψ|²/n0 - 1 + V) ψ ],      γ = -i ω0, dealiased

The uniform state ψ = √n0 is stationary. Derived units: length ξ, velocity c, time
ξ/c, density n0, energy μ = c² (m = 1), ħ/m = √2 c ξ.

Dynamics
--------
The same operator H = -ξ²∇² + |ψ|²/n0 - 1 + V drives two evolutions, selected by the
`dynamics` of the model:

* `RealTime()`       ∂ₜψ = -iω0 H ψ   (GPE, γ = -iω0)
* `ImaginaryTime()`  ∂τψ = -ω0 H ψ    (real Ginzburg–Landau equation, RGLE, γ = -ω0)

`RGLE(params, grid, plans; potential)` is the GPE model with `ImaginaryTime()`. The
RGLE is a gradient flow: it decreases `Diagnostics.energy` (= Ω/μ, with Ω = E - μN)
until a stationary state of the GPE is reached. See `Relaxation` for the driver.

Extension points kept in mind (not implemented yet)
---------------------------------------------------
* advection:        a constant velocity `v` adds a term `∝ v·k` to `linear_rate`
                    (advective RGLE: a new dynamics carrying `v`);
* dipolar / other nonlocal interactions: extra term in `nonlinear!` built from a
                    spectral kernel applied to F[|ψ|²];
* stochastic terms: added by a dedicated stochastic stepper.
"""
module GrossPitaevskii

using ..Architectures
using ..Grids
using ..Fields
using ..FFT
using ..Models
using ..Spectral
using ..Potentials
import ..Models: linear_rate, nonlinear!, evolved_field

export GPEParameters, GPE, RGLE, AbstractDynamics, RealTime, ImaginaryTime,
       GPE_STATE_LAYOUT, gpe_state, update_physical!, set_wavefunction!,
       bogoliubov_frequency

# ================================================================== parameters
"""
    GPEParameters(T = Float64; c, ξ, n0 = 1)
    GPEParameters(grid; c, ξ, n0 = 1)

Physical parameters (with `m = 1`). Derived quantities:

* `ω0 = c/(√2 ξ)`   frequency unit μ/ħ
* `hbar_over_m = √2 c ξ`
* `g = c²/n0`       contact coupling (times 1/m)
* `μ = c²`          chemical potential (times 1/m)
"""
struct GPEParameters{T<:AbstractFloat}
    c::T
    ξ::T
    n0::T
    ω0::T
    hbar_over_m::T
    g::T
    μ::T
end

function GPEParameters(::Type{T} = Float64; c::Real, ξ::Real, n0::Real = 1) where {T<:AbstractFloat}
    (c > 0 && ξ > 0 && n0 > 0) || throw(ArgumentError("c, ξ and n0 must be positive"))
    c64, ξ64, n64 = Float64(c), Float64(ξ), Float64(n0)
    return GPEParameters{T}(T(c64), T(ξ64), T(n64), T(c64 / (√2 * ξ64)),
                            T(√2 * c64 * ξ64), T(c64^2 / n64), T(c64^2))
end
GPEParameters(::Grid{T}; kwargs...) where {T} = GPEParameters(T; kwargs...)

function Base.show(io::IO, p::GPEParameters{T}) where {T}
    print(io, "GPEParameters{$T}: c = ", p.c, ", ξ = ", p.ξ, ", n0 = ", p.n0,
          "  (μ = ", p.μ, ", ω0 = ", p.ω0, ", ξ/c = ", p.ξ / p.c, ")")
end

"""
    bogoliubov_frequency(params, k)

Bogoliubov dispersion relation of the uniform condensate,
`ω(k) = ω0 √(ξ²k² (ξ²k² + 2))`, which tends to `c k` for `ξk ≪ 1`.
"""
bogoliubov_frequency(p::GPEParameters, k::Real) = p.ω0 * sqrt((p.ξ * k)^2 * ((p.ξ * k)^2 + 2))

# ================================================================== state
#   ψk : evolved variable (Fourier space)
#   ψ  : physical-space cache, refreshed from ψk with `update_physical!` (needed for
#        plots, diagnostics, output).
const GPE_STATE_LAYOUT = (ψ = ComplexPhysical(), ψk = FullSpectral())

"Allocate a state with the fields of the GPE (`ψ`, `ψk`)."
gpe_state(g::Grid) = State(g, GPE_STATE_LAYOUT)

# ================================================================== dynamics
abstract type AbstractDynamics end

"Real-time dynamics: the Gross–Pitaevskii equation, `∂ₜψ = -iω0 H ψ`."
struct RealTime <: AbstractDynamics end

"Imaginary-time dynamics: the real Ginzburg–Landau equation, `∂τψ = -ω0 H ψ`."
struct ImaginaryTime <: AbstractDynamics end

# prefactor γ of the nonlinear term (the linear part is  λ(k) = γ ξ² k²)
_prefactor(::RealTime, ω0::T) where {T}      = Complex{T}(0, -ω0)
_prefactor(::ImaginaryTime, ω0::T) where {T} = Complex{T}(-ω0, 0)
_name(::RealTime)      = "GPE"
_name(::ImaginaryTime) = "RGLE"

# ================================================================== model
"""
    GPE(params, grid, plans; potential = NoPotential(), dynamics = RealTime())
    RGLE(params, grid, plans; potential = NoPotential())

The GPE model (real time) and the RGLE (imaginary time, `RGLE(...)` is
`GPE(...; dynamics = ImaginaryTime())`). It owns one physical-space buffer (`ψ`),
the FFT plans and the external potential (and its array `V`, `nothing` if there is
no potential).
"""
struct GPE{P<:GPEParameters, G<:Grid, F<:FFTPlans, Pot<:AbstractPotential, VT,
           C<:AbstractMatrix, Γ<:Complex, D<:AbstractDynamics} <: AbstractModel
    params::P
    grid::G
    plans::F
    potential::Pot
    V::VT
    ψ::C
    γ::Γ
    dynamics::D
end

function GPE(params::GPEParameters{T}, grid::Grid{T}, plans::FFTPlans;
             potential::AbstractPotential = NoPotential(),
             dynamics::AbstractDynamics = RealTime()) where {T}
    V = allocate_potential(grid, potential)
    ψ = allocate_field(grid, ComplexPhysical())
    return GPE(params, grid, plans, potential, V, ψ, _prefactor(dynamics, params.ω0), dynamics)
end

RGLE(params::GPEParameters, grid::Grid, plans::FFTPlans; potential::AbstractPotential = NoPotential()) =
    GPE(params, grid, plans; potential = potential, dynamics = ImaginaryTime())

function Base.show(io::IO, m::GPE)
    print(io, _name(m.dynamics), " model\n  ", m.params, "\n  ", m.grid,
          "\n  potential: ", typeof(m.potential))
end

# ---------------------------------------------------------------- interface
evolved_field(::GPE, state) = state.ψk

# λ(k) = γ ξ² k² = -coef·k²  with  coef = -γ ξ²   (GPE: i ω0 ξ²;  RGLE: ω0 ξ², decaying)
linear_rate(m::GPE) = LaplacianRate(-m.γ * m.params.ξ^2)

# ψ ← γ (|ψ|²/n0 - 1 [+ V]) ψ    (fused, in place)
function _local_term!(ψ, γ, invn0, ::Nothing)
    o = one(real(eltype(ψ)))
    @. ψ = γ * (abs2(ψ) * invn0 - o) * ψ
    return ψ
end
function _local_term!(ψ, γ, invn0, V)
    o = one(real(eltype(ψ)))
    @. ψ = γ * (abs2(ψ) * invn0 - o + V) * ψ
    return ψ
end

function nonlinear!(out, Uk, m::GPE, t)
    to_physical!(m.ψ, Uk, m.plans)
    refresh!(m.V, m.potential, m.grid, t)
    _local_term!(m.ψ, m.γ, inv(m.params.n0), m.V)
    to_spectral!(out, m.ψ, m.plans)
    dealias!(out, m.grid)
    return out
end

# ---------------------------------------------------------------- helpers
"Refresh the physical-space cache `state.ψ` from `state.ψk`."
update_physical!(state, m::GPE) = to_physical!(state.ψ, state.ψk, m.plans)

"""
    set_wavefunction!(state, model, ψ0)

Set the wavefunction from a host array `ψ0` (size `nx × ny`): it is moved to the
device, transformed, dealiased, and the physical-space cache is refreshed.
"""
function set_wavefunction!(state, m::GPE, ψ0::AbstractMatrix)
    T = float_type(m.grid)
    size(ψ0) == size(m.grid) ||
        throw(DimensionMismatch("ψ0 has size $(size(ψ0)), expected $(size(m.grid))"))
    copyto!(state.ψ, to_device(m.grid.arch, Complex{T}.(ψ0)))
    to_spectral!(state.ψk, state.ψ, m.plans)
    dealias!(state.ψk, m.grid)
    update_physical!(state, m)
    return state
end

end # module
