"""
    Architectures

Minimal backend abstraction. This is the ONLY place that knows which array type
lives on which device. Everything else (Grid, Fields, FFT, ...) asks the
architecture for arrays via `allocate` / `to_device`.

No package extensions and no hard dependency on CUDA.jl / Metal.jl: the GPU
package is looked up *at runtime* among the modules the user has already loaded.
So it works the same whether GPE2D is included with `include("GPE2D.jl")` or
installed as a package. The user only has to load the backend first:

    using Metal              # or: using CUDA
    include("GPE2D.jl"); using .GPE2D
    grid = Grid(MetalGPU(), Float32; nx = 512, Lx = 64.0)
"""
module Architectures

export AbstractArchitecture, CPU, CUDAGPU, MetalGPU,
       array_type, allocate, to_device, to_host,
       device_synchronize, architecture_name, supports_precision

abstract type AbstractArchitecture end

struct CPU      <: AbstractArchitecture end
struct CUDAGPU  <: AbstractArchitecture end
struct MetalGPU <: AbstractArchitecture end

# ---------------------------------------------------------------- runtime lookup
function _backend_module(name::String)
    for (id, mod) in Base.loaded_modules
        id.name == name && return mod
    end
    error("The $name package is not loaded. Run `using $name` before creating the grid.")
end

# ---------------------------------------------------------------- array types
array_type(::CPU)      = Array
array_type(::CUDAGPU)  = getfield(_backend_module("CUDA"),  :CuArray)
array_type(::MetalGPU) = getfield(_backend_module("Metal"), :MtlArray)

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
device_synchronize(::CPU)      = nothing
device_synchronize(::CUDAGPU)  = getfield(_backend_module("CUDA"),  :synchronize)()
device_synchronize(::MetalGPU) = getfield(_backend_module("Metal"), :synchronize)()

architecture_name(::CPU)      = "CPU"
architecture_name(::CUDAGPU)  = "CUDA GPU"
architecture_name(::MetalGPU) = "Metal GPU"

"Whether floating-point type `T` can be used on `arch` (Apple GPUs have no Float64)."
supports_precision(::AbstractArchitecture, ::Type{<:AbstractFloat}) = true
supports_precision(::MetalGPU, ::Type{Float64}) = false

end # module
