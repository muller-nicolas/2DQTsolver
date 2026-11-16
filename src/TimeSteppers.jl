"""
    TimeSteppers

Time integrators for models of the form  ∂ₜU = λ(k)·U + N(U, t)  in Fourier space
(see `Models.jl`). They know nothing about the equation: they only call
`linear_rate(model)` and `nonlinear!(out, U, model, t)`.

Available steppers
------------------
* `RK4`   : classical explicit Runge–Kutta 4 on the full right-hand side. Needs
            `dt ≲ 2.8/|λ|max`, i.e. it is limited by the kinetic term at large k.
            Mainly a reference.
* `IFRK4` : integrating-factor RK4. The linear part is propagated exactly with
            `exp(λ dt)`, RK4 is applied to the rest. Still 4th order, but `dt` is
            set by the nonlinear term only. 4 evaluations of `N` per step
            (8 FFTs for the GPE). Recommended.

Memory: each stepper allocates its buffers ONCE (3 arrays, plus 2 integrating-factor
arrays for IFRK4). A step performs no allocations: only fused broadcasts and the
model's own `nonlinear!`.

Usage
-----
    stepper = IFRK4(grid, model, dt)
    step!(state, stepper, model)                 # one step (also advances the clock)
    evolve!(state, stepper, model, 1000; callback = s -> ...)
"""
module TimeSteppers

using ..Architectures
using ..Grids
using ..Fields
using ..Models
using ..Spectral

export AbstractTimeStepper, RK4, IFRK4, step!, evolve!, set_dt!, timestep,
       every, CallbackSet

abstract type AbstractTimeStepper end

"Time step of a stepper."
timestep(s::AbstractTimeStepper) = s.dt

# the 3 scratch arrays used by both steppers
_buffers(grid) = Workspace(grid, repeated_layout(FullSpectral(), :buf, :tmp, :acc))

# ================================================================== RK4
"""
    RK4(grid, model, dt)

Classical 4th-order Runge–Kutta on `λU + N(U, t)`.
"""
mutable struct RK4{G<:Grid, R, W<:Workspace} <: AbstractTimeStepper
    grid::G
    rate::R
    dt::Float64
    ws::W
end

function RK4(grid::Grid, model::AbstractModel, dt::Real)
    return RK4(grid, linear_rate(model), Float64(dt), _buffers(grid))
end

"Change the time step."
set_dt!(s::RK4, dt::Real) = (s.dt = Float64(dt); s)

# out = λ·U + N(U, t)
function _rhs!(out, U, t, s::RK4, model)
    nonlinear!(out, U, model, t)
    rate, kx, ky = s.rate, s.grid.kx, s.grid.ky
    @. out += rate(kx, ky) * U
    return out
end

"""
    step!(Uk, t, stepper, model)

Advance the array `Uk` in place from time `t` to `t + dt`.
"""
function step!(Uk, t::Real, s::RK4, model)
    T = float_type(s.grid)
    h, h2, h6, two = T(s.dt), T(s.dt / 2), T(s.dt / 6), T(2)
    t = Float64(t)
    buf, tmp, acc = s.ws.buf, s.ws.tmp, s.ws.acc

    _rhs!(buf, Uk, t, s, model)                       # k1
    @. acc = buf
    @. tmp = Uk + h2 * buf
    _rhs!(buf, tmp, t + s.dt / 2, s, model)           # k2
    @. acc += two * buf
    @. tmp = Uk + h2 * buf
    _rhs!(buf, tmp, t + s.dt / 2, s, model)           # k3
    @. acc += two * buf
    @. tmp = Uk + h * buf
    _rhs!(buf, tmp, t + s.dt, s, model)               # k4
    @. Uk += h6 * (acc + buf)
    return Uk
end

# ================================================================== IFRK4
"""
    IFRK4(grid, model, dt)

Integrating-factor RK4. With `E = exp(λ dt)` and `E2 = exp(λ dt/2)`:

    a = N(u, t)
    b = N(E2 (u + dt/2 a), t + dt/2)
    c = N(E2 u + dt/2 b,   t + dt/2)
    d = N(E u + dt E2 c,   t + dt)
    u ← E u + dt/6 (E a + 2 E2 (b + c) + d)

`E` and `E2` are computed once (and again by `set_dt!`).
"""
mutable struct IFRK4{G<:Grid, R, C<:AbstractMatrix, W<:Workspace} <: AbstractTimeStepper
    grid::G
    rate::R
    dt::Float64
    E::C           # exp(λ dt)
    E2::C          # exp(λ dt/2)
    ws::W
end

function IFRK4(grid::Grid, model::AbstractModel, dt::Real)
    rate = linear_rate(model)
    return IFRK4(grid, rate, Float64(dt),
                 integrating_factor(grid, rate, dt),
                 integrating_factor(grid, rate, dt / 2),
                 _buffers(grid))
end

"Change the time step (recomputes the integrating factors in place)."
function set_dt!(s::IFRK4, dt::Real)
    s.dt = Float64(dt)
    copyto!(s.E,  integrating_factor(s.grid, s.rate, dt))
    copyto!(s.E2, integrating_factor(s.grid, s.rate, dt / 2))
    return s
end

function step!(Uk, t::Real, s::IFRK4, model)
    T = float_type(s.grid)
    h, h2, h6, two = T(s.dt), T(s.dt / 2), T(s.dt / 6), T(2)
    t = Float64(t)
    E, E2 = s.E, s.E2
    buf, tmp, acc = s.ws.buf, s.ws.tmp, s.ws.acc

    nonlinear!(buf, Uk, model, t)                     # a
    @. tmp = E2 * (Uk + h2 * buf)
    @. acc = E * buf
    nonlinear!(buf, tmp, model, t + s.dt / 2)         # b
    @. tmp = E2 * Uk + h2 * buf
    @. acc += two * E2 * buf
    nonlinear!(buf, tmp, model, t + s.dt / 2)         # c
    @. tmp = E * Uk + h * E2 * buf
    @. acc += two * E2 * buf
    nonlinear!(buf, tmp, model, t + s.dt)             # d
    @. Uk = E * Uk + h6 * (acc + buf)
    return Uk
end

# ================================================================== state-level API
"""
    step!(state, stepper, model)

One time step of the evolved field of `state` (as given by `evolved_field`), using the
state's clock for `t`. Advances the clock.
"""
function step!(state::AbstractState, s::AbstractTimeStepper, model::AbstractModel)
    c = clock(state)
    step!(evolved_field(model, state), c.t, s, model)
    advance!(c, s.dt)
    return state
end

"""
    evolve!(state, stepper, model, nsteps; callback = nothing)

Take `nsteps` steps. If given, `callback(state)` is called after every step (use
`every(n, f)` to call something only every `n` steps, and `CallbackSet` to combine
several). If the callback returns `:stop`, the run ends early.
"""
function evolve!(state::AbstractState, s::AbstractTimeStepper, model::AbstractModel,
                 nsteps::Integer; callback = nothing)
    for _ in 1:nsteps
        step!(state, s, model)
        if callback !== nothing && callback(state) === :stop
            break
        end
    end
    return state
end

# ================================================================== callbacks
"Callable that runs `f(state)` only when the iteration counter is a multiple of `n`."
struct Every{F}
    n::Int
    f::F
end
(e::Every)(state) = (clock(state).iteration % e.n == 0) ? e.f(state) : nothing

"`every(n, f)`: wrap callback `f` so that it runs every `n` steps."
every(n::Integer, f) = Every(Int(n), f)

"Run several callbacks in order; stops as soon as one of them returns `:stop`."
struct CallbackSet{T<:Tuple}
    callbacks::T
end
CallbackSet(cbs...) = CallbackSet(cbs)
function (c::CallbackSet)(state)
    for cb in c.callbacks
        cb(state) === :stop && return :stop
    end
    return nothing
end

end # module
