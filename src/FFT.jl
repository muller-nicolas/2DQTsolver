"""
    FFT

All Fourier transforms of the code go through an `FFTPlans` object, created ONCE
per grid and reused at every time step. Nothing else in the code calls `fft` /
`ifft` directly.

The plans are built with the AbstractFFTs interface, which dispatches on the array
type: FFTW on CPU, cuFFT when `CUDA` is loaded, MPSGraph when `Metal` is loaded.
So this file contains no backend-specific code apart from the planning flags.

Conventions
-----------
* Complex-to-complex, 2D, in place.
* Forward transform is NOT normalised; the backward one includes `1/(nx*ny)`
  (the AbstractFFTs convention), so `backward!(forward!(A)) == A`.
* Parseval:  Σ|ψ|² = (1/(nx·ny)) Σ|ψk|².

In-place plans are created on a private scratch array of the grid's field size.
They may be applied to any array of the same size and element type (all fields
from `Fields.jl` qualify). Note that FFTW plans also assume the same memory
alignment, which is the case for ordinary Julia arrays.
"""
module FFT

using AbstractFFTs
using FFTW
using ..Architectures
using ..Grids
using ..Fields

export FFTPlans, forward!, backward!, to_spectral!, to_physical!

struct FFTPlans{P,Q}
    forward::P      # in-place plan:  physical → spectral
    backward::Q     # in-place plan:  spectral → physical (normalised)
    nx::Int
    ny::Int
end

"""
    FFTPlans(grid; flags = FFTW.MEASURE, nthreads = nothing)

Create the forward/backward in-place plans for the full complex layout of `grid`.

* `flags`    : FFTW planning flags (CPU only; ignored on GPUs). `FFTW.MEASURE`
               is slower to plan but faster to run; use `FFTW.ESTIMATE` for quick tests.
* `nthreads` : number of FFTW threads on the CPU (`nothing` keeps FFTW's current setting).
"""
function FFTPlans(grid::Grid; flags = FFTW.MEASURE, nthreads = nothing)
    scratch = allocate_field(grid, FullSpectral())   # planning may overwrite it
    return _make_plans(grid.arch, scratch, flags, nthreads)
end

function _make_plans(::CPU, A, flags, nthreads)
    nthreads === nothing || FFTW.set_num_threads(nthreads)
    return FFTPlans(plan_fft!(A; flags = flags), plan_ifft!(A; flags = flags), size(A)...)
end

# CUDA, Metal, ...: whatever backend package is loaded provides the plans
_make_plans(::AbstractArchitecture, A, flags, nthreads) =
    FFTPlans(plan_fft!(A), plan_ifft!(A), size(A)...)

@inline function _check(A, p::FFTPlans)
    size(A) == (p.nx, p.ny) ||
        throw(DimensionMismatch("array of size $(size(A)) does not match the plans ($(p.nx), $(p.ny))"))
    return nothing
end

"In-place forward transform of `A` (physical → spectral). Returns `A`."
function forward!(A, p::FFTPlans)
    _check(A, p)
    p.forward * A
    return A
end

"In-place backward transform of `A` (spectral → physical, normalised). Returns `A`."
function backward!(A, p::FFTPlans)
    _check(A, p)
    p.backward * A
    return A
end

"`dst = FFT(src)`; `src` is left untouched. No allocation."
function to_spectral!(dst, src, p::FFTPlans)
    copyto!(dst, src)
    return forward!(dst, p)
end

"`dst = IFFT(src)`; `src` is left untouched. No allocation."
function to_physical!(dst, src, p::FFTPlans)
    copyto!(dst, src)
    return backward!(dst, p)
end

function Base.show(io::IO, p::FFTPlans)
    print(io, "FFTPlans(", p.nx, "×", p.ny, ", ", nameof(typeof(p.forward)), ")")
end

end # module
