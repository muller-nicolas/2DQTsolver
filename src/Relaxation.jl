"""
    Relaxation

Imaginary-time relaxation with the real Ginzburg–Landau equation (RGLE, see
`GrossPitaevskii`): it drives a state towards a stationary state of the GPE.

    model = RGLE(params, grid, plans; potential = ...)
    result = relax!(state, model; dt = 0.01, max_time = 5.0, tol = 1e-6)

The RGLE is a gradient flow: `Diagnostics.energy` (Ω/μ, with Ω = E − μN) decreases
monotonically, and it stops when the state is stationary.

Two ways of fixing the density scale
------------------------------------
* `FixedMu()` (default, grand canonical). The equation is used as it is: the chemical
  potential is fixed to μ = g n0 and the particle number is free. The uniform state
  √n0 is the ground state without potential; with an obstacle the density heals to n0
  far from it. This is the natural choice to prepare the state for a real-time run.
* `FixedNumber(N)`. After every step ψ is rescaled to N particles, and the chemical
  potential (`Diagnostics.chemical_potential`) adapts.

Convergence
-----------
The residual is `‖∂τψ‖/‖ψ‖` in units of μ (i.e. divided by ω0): for `FixedMu` the full
`‖Hψ‖/‖ψ‖`; for `FixedNumber` only the part of `∂τψ` orthogonal to ψ, since a
stationary state satisfies `Hψ = μ_ψ ψ`.

Residual floor: the stationary state of the DISCRETE scheme differs from that of the
continuous equation by O(dt⁴), so the residual saturates (≈ 1e-9 for dt = 0.01 in the
tested case with an obstacle). Do not ask for `tol` below ~1e-8.

Caveat: a vortex dipole is NOT a stationary state of the RGLE: the vortices attract
and annihilate, so it cannot be relaxed to convergence. For a separation d = 8ξ (tested
with a Python prototype of the scheme) the pair is unchanged for τ ≲ 2 ξ/c, visibly
closer at τ ≈ 3–5 ξ/c, and annihilates at τ ≈ 15 ξ/c. Use a short `max_time` (about
1–2 ξ/c) to smooth the cores, or wait for the advective version (ARGLE), which holds a
travelling dipole stationary.

After `relax!` the clock is reset (`reset_clock = true`), so that a real-time run
starts at t = 0, iteration 0.
"""
module Relaxation

using Printf
using ..Architectures
using ..Grids
using ..Fields
using ..Models
using ..TimeSteppers
using ..GrossPitaevskii
using ..Diagnostics

export Normalization, FixedMu, FixedNumber, Renormalizer, relax!, relaxation_residual

# ================================================================== normalisation
abstract type Normalization end

"Grand-canonical relaxation: μ = g n0 fixed, particle number free."
struct FixedMu <: Normalization end

"""
    FixedNumber(N)
    FixedNumber(state, model)

Canonical relaxation: ψ is rescaled to `N` particles (`∫|ψ|² dA = N`) after every step.
`FixedNumber(state, model)` takes the number of particles of the current state.
"""
struct FixedNumber <: Normalization
    N::Float64
end
FixedNumber(state, model::GPE) = FixedNumber(particle_number(state, model))

"Callable that rescales ψ to `N` particles; use as a callback of `evolve!`."
struct Renormalizer{M<:GPE}
    model::M
    N::Float64
end

function (r::Renormalizer)(state)
    ψk = evolved_field(r.model, state)
    s = sqrt(r.N / particle_number(state, r.model))
    ψk .*= float_type(r.model.grid)(s)
    return nothing
end

# ================================================================== residual
"""
    relaxation_residual(state, model, buf; fixed_number = false)

Convergence measure `‖∂τψ‖/(ω0 ‖ψ‖)` (units of μ); with `fixed_number = true` the
component of `∂τψ` along ψ is removed. `buf` is a spectral work array
(`allocate_field(grid, FullSpectral())`).
"""
function relaxation_residual(state, model::GPE, buf; fixed_number::Bool = false)
    g = model.grid
    ψk = evolved_field(model, state)
    nonlinear!(buf, ψk, model, clock(state).t)
    rate, kx, ky = linear_rate(model), g.kx, g.ky
    @. buf += rate(kx, ky) * ψk                       # buf = ∂τψ̂
    f2 = Float64(sum(abs2, buf))
    n2 = Float64(sum(abs2, ψk))
    if fixed_number
        d = Float64(real(sum(@. conj(ψk) * buf)))     # Re⟨ψ, ∂τψ⟩
        f2 -= d^2 / n2
    end
    return sqrt(max(f2, 0.0) / n2) / Float64(model.params.ω0)
end

# ================================================================== driver
"""
    relax!(state, model; dt, max_time = Inf, max_steps = typemax(Int), tol = 1e-6,
           normalization = FixedMu(), check_every = 100, verbose = true, io = stdout,
           callback = nothing, reset_clock = true)

Evolve `state` in imaginary time with the integrating-factor RK4 stepper until the
residual drops below `tol`, or `max_time` / `max_steps` is reached. `model` must be an
`RGLE`. At least one of `max_time`, `max_steps` must be finite. The residual is
checked (and printed, if `verbose`) every `check_every` steps; `callback(state)` is
called after every step (it may return `:stop`).

Returns a `NamedTuple` `(converged, steps, τ, residual, energy, N, μ)`.
"""
function relax!(state, model::GPE; dt::Real, max_time::Real = Inf,
                max_steps::Integer = typemax(Int), tol::Real = 1e-6,
                normalization::Normalization = FixedMu(), check_every::Integer = 100,
                verbose::Bool = true, io::IO = stdout, callback = nothing,
                reset_clock::Bool = true)
    model.dynamics isa ImaginaryTime ||
        throw(ArgumentError("relax! needs an imaginary-time model: build it with RGLE(...)"))
    (isfinite(max_time) || max_steps < typemax(Int)) ||
        throw(ArgumentError("give a finite max_time and/or max_steps"))

    grid = model.grid
    stepper = IFRK4(grid, model, dt)
    buf = allocate_field(grid, FullSpectral())
    fixed_n = normalization isa FixedNumber

    callbacks = Any[]
    if fixed_n
        renorm = Renormalizer(model, normalization.N)
        renorm(state)                                  # start from the target number
        push!(callbacks, renorm)
    end
    callback === nothing || push!(callbacks, callback)
    cb = CallbackSet(callbacks...)

    c = clock(state)
    t0, it0 = c.t, c.iteration
    steps, converged, res = 0, false, NaN
    verbose && @printf(io, "RGLE relaxation: dt = %g, tol = %g, %s\n", dt, tol,
                       fixed_n ? "fixed particle number" : "fixed chemical potential")
    while true
        time_left = isfinite(max_time) ? ceil(Int, (t0 + max_time - c.t) / dt - 1e-9) : typemax(Int)
        n = min(check_every, max_steps - steps, time_left)
        n <= 0 && break
        evolve!(state, stepper, model, n; callback = cb)
        done = c.iteration - it0 - steps            # steps actually taken (a callback may stop early)
        steps += done
        res = relaxation_residual(state, model, buf; fixed_number = fixed_n)
        if verbose
            @printf(io, "  τ = %9.4f  steps = %7d  E = %+.8e  N = %.8e  μ = %.6f  residual = %.3e\n",
                    c.t - t0, steps, energy(state, model), particle_number(state, model),
                    chemical_potential(state, model), res)
        end
        if res < tol
            converged = true
            break
        end
        done < n && break                           # stopped by a callback
    end
    verbose && !converged && @printf(io, "  not converged (residual %.3e > tol %g) after %d steps\n", res, tol, steps)

    result = (converged = converged, steps = steps, τ = steps * Float64(dt), residual = res,
              energy = energy(state, model), N = particle_number(state, model),
              μ = chemical_potential(state, model))
    if reset_clock
        c.t = 0.0
        c.iteration = 0
    end
    return result
end

end # module
