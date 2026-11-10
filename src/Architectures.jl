"""
    Architectures

Minimal backend abstraction. This is the ONLY place that knows which array type
lives on which device. Everything else (Grid, Fields, Spectral, ...) asks the
architecture for arrays via `allocate` / `to_device`.

The architecture *types* are defined here, but the methods that need the GPU
packages (`array_type`, `device_synchronize` for `CUDAGPU` / `MetalGPU`) live in
the package extensions `ext/CUDAExt.jl` and `ext/MetalExt.jl`. They are only
loaded when the user does `using CUDA` / `using Metal`.

The FFT module will build on top of this (plans per architecture).
"""
module Architectures

export AbstractArchitecture, CPU, CUDAGPU, MetalGPU,
       array_type, allocate, to_device, to_host,
       device_synchronize, architecture_name, supports_precision

abstract type AbstractArchitecture end

struct CPU      <: AbstractArchitecture end
struct CUDAGPU  <: AbstractArchitecture end
struct MetalGPU <: AbstractArchitecture end

# ---------------------------------------------------------------- array types
# NOTE: the fallback is defined on the abstract type so that the extensions can
# add more specific methods without overwriting anything (which Julia forbids
# during precompilation).
array_type(arch::AbstractArchitecture) =
    error("No array type available for $(typeof(arch)). " *
          "Load the backend package first (`using CUDA` or `using Metal`).")
array_type(::CPU) = Array

# ---------------------------------------------------------------- data movement
"""
    allocate(arch, T, dims...)

Allocate a zero-initialised array of element type `T` on `arch`.
"""
function allocate(arch::AbstractArchitecture, ::Type{T}, dims::Integer...) where {T}
    A = array_type(arch){T}(undef, dims...)
    fill!(A, zero(T))
    return A
end

"""
    to_device(arch, A)

Copy a host array to `arch` (no copy if it is already a CPU `Array` and `arch` is `CPU`).
"""
to_device(::CPU, A::Array) = A
to_device(::CPU, A::AbstractArray) = Array(A)
to_device(arch::AbstractArchitecture, A::AbstractArray) = array_type(arch)(A)

"Copy any array (host or device) to a plain host `Array`."
to_host(A::AbstractArray) = Array(A)

# ---------------------------------------------------------------- misc
"Block until all queued device work is finished (no-op on CPU). Needed for timing."
device_synchronize(::AbstractArchitecture) = nothing

architecture_name(::CPU)      = "CPU"
architecture_name(::CUDAGPU)  = "CUDA GPU"
architecture_name(::MetalGPU) = "Metal GPU"

"Whether floating-point type `T` can be used on `arch` (Apple GPUs have no Float64)."
supports_precision(::AbstractArchitecture, ::Type{<:AbstractFloat}) = true
supports_precision(::MetalGPU, ::Type{Float64}) = false

end # module
