"""
    Models

The interface between an equation (the *model*) and the time steppers. It contains
no physics, only the definition of what a model must provide. The GPE will be one
implementation; any other pseudo-spectral solver (Navier–Stokes, Cahn–Hilliard,
two-component GPE, ...) only has to implement the same three functions to reuse
all the time steppers.

A model describes an evolution equation in Fourier space for ONE complex array `Uk`:

    ∂ₜ Uk = λ(kx, ky) · Uk  +  N(Uk, t)

* `λ` is the linear part, diagonal in Fourier space (e.g. `-iħk²/2m` for the GPE).
* `N` is everything else (nonlinear terms, potentials, forcing). It is evaluated
  by the model, which owns the FFT plans and physical-space buffers it needs.

Required methods
----------------
    linear_rate(model)                  -> λ
    nonlinear!(out, Uk, model, t)       -> fills `out` with N(Uk, t), returns `out`
    evolved_field(model, state)         -> the array of `state` that is evolved (Uk)

Rules
-----
* `λ` must be a callable `isbits` object, `λ(kx, ky)` returning a number. It is used
  inside broadcast kernels and to build integrating factors
  (see `Spectral.LaplacianRate` for the common case `coef·∇²`).
* `nonlinear!` must NOT modify `Uk` and must not alias `out` with `Uk`.
  It must return a dealiased result (multiply by `grid.mask`), since the time
  steppers do not dealias.
* `t` is a `Float64` (simulation time, for time-dependent potentials or forcing).
"""
module Models

export AbstractModel, linear_rate, nonlinear!, evolved_field

abstract type AbstractModel end

"""
    linear_rate(model)

Callable `λ(kx, ky)` giving the eigenvalue of the (diagonal) linear operator.
"""
function linear_rate end

"""
    nonlinear!(out, Uk, model, t)

Compute the non-linear part `N(Uk, t)` of the equation in Fourier space (dealiased)
and store it in `out`.
"""
function nonlinear! end

"""
    evolved_field(model, state)

Return the array of `state` that the time stepper integrates (e.g. `state.ψk`).
"""
function evolved_field end

end # module
