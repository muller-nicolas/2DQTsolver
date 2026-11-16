"""
    Potentials

External potentials, independent of any particular equation.

A potential is an object `pot <: AbstractPotential` that knows how to fill a real
physical-space array `V` on the device:

    evaluate!(V, pot, grid, t)

Units: for the GPE, potentials are given in units of the chemical potential
`μ = g n0` (i.e. `V/μ`). Other models may use their own units.

Static potentials are evaluated ONCE when the model is built; time-dependent ones
are re-evaluated by the model at every stage time (`refresh!`), directly on the
device. `NoPotential` allocates nothing and costs nothing.

Available potentials
--------------------
* `NoPotential()`
* `StaticPotential(f)`           : `f(x, y)` evaluated once on the host (any shape)
* `GaussianObstacle(V0, σ; x0, y0, vx, vy)` : Gaussian bump, optionally moving at
                                   constant velocity, periodic in the box.

To add your own, define a struct, an `evaluate!` method and (if it depends on time)
`is_static(p) = false`.
"""
module Potentials

using ..Architectures
using ..Grids
using ..Fields

export AbstractPotential, NoPotential, StaticPotential, GaussianObstacle,
       is_static, allocate_potential, evaluate!, refresh!

abstract type AbstractPotential end

"""
    evaluate!(V, potential, grid, t)

Fill the real device array `V` (size `nx × ny`) with the potential at time `t`.
"""
function evaluate! end

"Whether the potential is independent of time (evaluated only once)."
is_static(::AbstractPotential) = true

"""
    allocate_potential(grid, potential)

Array holding the potential, already evaluated at `t = 0`; `nothing` for `NoPotential`
(so that nothing is stored or multiplied).
"""
function allocate_potential(grid::Grid, pot::AbstractPotential)
    V = allocate_field(grid, RealPhysical())
    evaluate!(V, pot, grid, 0.0)
    return V
end

"Update `V` to time `t` if the potential depends on time; otherwise do nothing."
refresh!(::Nothing, ::AbstractPotential, ::Grid, t) = nothing
function refresh!(V, pot::AbstractPotential, grid::Grid, t)
    is_static(pot) || evaluate!(V, pot, grid, t)
    return V
end

# ================================================================== NoPotential
struct NoPotential <: AbstractPotential end
allocate_potential(::Grid, ::NoPotential) = nothing

# ================================================================== StaticPotential
"""
    StaticPotential(f)

Potential `f(x, y)` (a plain Julia function of two numbers). It is evaluated once on
the host, so `f` can be anything; it is then copied to the device.

    StaticPotential((x, y) -> 0.5 * exp(-(x^2 + y^2) / 0.1))
"""
struct StaticPotential{F} <: AbstractPotential
    f::F
end

function evaluate!(V, p::StaticPotential, grid::Grid{T}, t) where {T}
    x, y = xcoords(grid), ycoords(grid)
    Vh = T.(p.f.(x, y'))
    copyto!(V, to_device(grid.arch, Vh))
    return V
end

# ================================================================== GaussianObstacle
"""
    GaussianObstacle(V0, σ; x0 = 0, y0 = 0, vx = 0, vy = 0)

`V(x, y, t) = V0 · exp(-r² / 2σ²)`, centred at `(x0 + vx t, y0 + vy t)`, with `r` the
minimum-image distance (periodic box). Static if `vx = vy = 0`.
"""
struct GaussianObstacle <: AbstractPotential
    V0::Float64
    σ::Float64
    x0::Float64
    y0::Float64
    vx::Float64
    vy::Float64
end
GaussianObstacle(V0::Real, σ::Real; x0::Real = 0.0, y0::Real = 0.0, vx::Real = 0.0, vy::Real = 0.0) =
    GaussianObstacle(Float64(V0), Float64(σ), Float64(x0), Float64(y0), Float64(vx), Float64(vy))

is_static(p::GaussianObstacle) = p.vx == 0 && p.vy == 0

# minimum-image displacement in a periodic box of length L
@inline _wrap(d, L) = d - L * round(d / L)

function evaluate!(V, p::GaussianObstacle, grid::Grid{T}, t) where {T}
    x, y = grid.x, grid.y
    Lx, Ly = grid.Lx, grid.Ly
    xc, yc = T(p.x0 + p.vx * t), T(p.y0 + p.vy * t)
    V0, c = T(p.V0), T(1 / (2 * p.σ^2))
    @. V = V0 * exp(-(_wrap(x - xc, Lx)^2 + _wrap(y - yc, Ly)^2) * c)
    return V
end

end # module
