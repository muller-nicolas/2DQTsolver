"""
    Spectral

Operations on fields in Fourier space (full complex layout, FFT ordering), using the
wavenumbers stored in the `Grid`. Equation-independent.

All functions work in place on arrays of the grid's architecture, are written as
fused broadcasts (no temporaries) and return their output array. They do NOT do
any FFT: transforms are the job of `FFT.jl`.

Derivatives
-----------
* Odd derivatives (`ddx!`, `ddy!`, `gradient!`) use `grid.kx1`, `grid.ky1`, where the
  Nyquist mode is zeroed (its sign is ambiguous).
* Even derivatives (`laplacian!`) use the full `grid.kx`, `grid.ky`.

Linear operators
----------------
`LaplacianRate(coef)` is the eigenvalue of `coef·∇²`, i.e. `-coef·(kx²+ky²)`.
For the GPE, `∂ₜψ = (iħ/2m)∇²ψ + ...` gives `coef = iħ/2m`. Other operators
(hyperviscosity, damping, ...) are just other callable isbits structs `λ(kx, ky)`.
`integrating_factor(grid, λ, τ)` returns the array `exp(λ·τ)`.
"""
module Spectral

using ..Architectures
using ..Grids

export LaplacianRate, dealias!, ddx!, ddy!, gradient!, laplacian!, inverse_laplacian!,
       compressible_part!, incompressible_part!,
       apply_rate!, integrating_factor

# ---------------------------------------------------------------- linear rates
"""
    LaplacianRate(coef)
    LaplacianRate(T, coef)

Eigenvalue `λ(kx, ky) = -coef·(kx² + ky²)` of the operator `coef·∇²`.
`LaplacianRate(T, coef)` stores `coef` as `Complex{T}`, which is what GPU kernels need.
"""
struct LaplacianRate{C<:Number}
    coef::C
end
LaplacianRate(::Type{T}, coef::Number) where {T<:AbstractFloat} = LaplacianRate(Complex{T}(coef))

@inline (r::LaplacianRate)(kx, ky) = -r.coef * (kx * kx + ky * ky)

# ---------------------------------------------------------------- dealiasing
"Zero all modes outside the dealiasing mask of `grid`, in place."
function dealias!(Ak, grid::Grid)
    Ak .*= grid.mask
    return Ak
end

# ---------------------------------------------------------------- derivatives
"`out = ∂x A` (spectral). Nyquist mode is zeroed."
function ddx!(out, Ak, grid::Grid{T}) where {T}
    kx1, z = grid.kx1, zero(T)
    @. out = complex(z, kx1) * Ak
    return out
end

"`out = ∂y A` (spectral). Nyquist mode is zeroed."
function ddy!(out, Ak, grid::Grid{T}) where {T}
    ky1, z = grid.ky1, zero(T)
    @. out = complex(z, ky1) * Ak
    return out
end

"`(gx, gy) = ∇A` (spectral)."
function gradient!(gx, gy, Ak, grid::Grid)
    ddx!(gx, Ak, grid)
    ddy!(gy, Ak, grid)
    return gx, gy
end

"`out = ∇²A` (spectral)."
function laplacian!(out, Ak, grid::Grid)
    kx, ky = grid.kx, grid.ky
    @. out = -(kx^2 + ky^2) * Ak
    return out
end

"`out = ∇⁻²A` (spectral). The k = 0 mode is set to zero."
function inverse_laplacian!(out, Ak, grid::Grid)
    kx, ky = grid.kx, grid.ky
    z = zero(eltype(out))
    @. out = ifelse(kx^2 + ky^2 == 0, z, -Ak / (kx^2 + ky^2))
    return out
end

# ---------------------------------------------------------------- Helmholtz decomposition
"""
    compressible_part!(cx, cy, wx, wy, grid)

Helmholtz decomposition of a 2D vector field `w = (wx, wy)` given in Fourier space:
the compressible (irrotational) part is the projection on the wavevector,
`wc = k (k·w) / k²`. The k = 0 mode is assigned to the incompressible part (`wc = 0`).
The outputs `cx, cy` must NOT alias the inputs `wx, wy`.
"""
function compressible_part!(cx, cy, wx, wy, grid::Grid)
    kx, ky = grid.kx, grid.ky
    z = zero(eltype(cx))
    @. cx = ifelse(kx^2 + ky^2 == 0, z, kx * (kx * wx + ky * wy) / (kx^2 + ky^2))
    @. cy = ifelse(kx^2 + ky^2 == 0, z, ky * (kx * wx + ky * wy) / (kx^2 + ky^2))
    return cx, cy
end

"""
    incompressible_part!(ix, iy, wx, wy, grid)

Incompressible (solenoidal) part `wi = w − wc` of the vector field `w`, in Fourier
space. The outputs `ix, iy` must NOT alias the inputs `wx, wy`.
"""
function incompressible_part!(ix, iy, wx, wy, grid::Grid)
    compressible_part!(ix, iy, wx, wy, grid)
    @. ix = wx - ix
    @. iy = wy - iy
    return ix, iy
end

# ---------------------------------------------------------------- generic linear operators
"`out = λ(kx, ky) · A` for a callable rate `λ` (see `LaplacianRate`)."
function apply_rate!(out, Ak, rate, grid::Grid)
    kx, ky = grid.kx, grid.ky
    @. out = rate(kx, ky) * Ak
    return out
end

"""
    integrating_factor(grid, rate, τ)

Device array `exp(λ(kx, ky)·τ)` of the grid's field size, with eltype `Complex{T}`.
It is computed once on the host (so no complex `exp` is needed inside GPU kernels)
and then moved to the device. Used by the integrating-factor time steppers.
"""
function integrating_factor(grid::Grid{T}, rate, τ::Real) where {T}
    kx, ky = to_host(grid.kx), to_host(grid.ky)
    τT = T(τ)
    E = @. Complex{T}(exp(rate(kx, ky) * τT))
    return to_device(grid.arch, E)
end

end # module
