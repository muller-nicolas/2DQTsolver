"""
    Fields

Generic, solver-independent containers for the arrays of a simulation.

Design
------
* A **layout** says what an array *is* (real/complex, physical/spectral, full/half
  spectrum). It determines size and element type from the `Grid`.
* A **State** holds the evolving fields (a `NamedTuple` of arrays) plus a `Clock`.
* A **Workspace** holds scratch buffers (RK stages, temporaries, ...).

Both are built from a `NamedTuple` of layouts, so a new solver only has to declare
what it needs:

    layout = (ψ = ComplexPhysical(), ψk = FullSpectral())
    state  = State(grid, layout)
    ws     = Workspace(grid, repeated_layout(FullSpectral(), :k1, :k2, :k3, :k4))

Everything is allocated ONCE. The containers are immutable structs holding
concrete array types (type-stable); the arrays themselves are mutated in place
(`.=`, `.*=`, `@.`). Only the `Clock` is mutable (host-side `Float64`, even when
the fields are `Float32` on Metal, so that long runs do not accumulate time error).
Fields are accessed by name: `state.ψk`, `ws.k1`.

Nothing here knows about the GPE, FFTs or time stepping.
"""
module Fields

using ..Architectures
using ..Grids

export Layout, RealPhysical, ComplexPhysical, FullSpectral, HalfSpectral,
       field_size, field_eltype, allocate_field, allocate_fields, repeated_layout,
       Clock, advance!,
       AbstractState, State, Workspace,
       fields, clock, host_fields, memory_bytes

# ---------------------------------------------------------------- layouts
abstract type Layout end
"Real field in physical space, size `nx × ny`, eltype `T`."
struct RealPhysical    <: Layout end
"Complex field in physical space, size `nx × ny`, eltype `Complex{T}` (e.g. the GPE wavefunction)."
struct ComplexPhysical <: Layout end
"Complex field in Fourier space from a complex-to-complex FFT, size `nx × ny`."
struct FullSpectral    <: Layout end
"Complex field in Fourier space from `rfft` of a real field, size `(nx÷2+1) × ny`."
struct HalfSpectral    <: Layout end

field_size(g::Grid, ::Union{RealPhysical,ComplexPhysical,FullSpectral}) = (g.nx, g.ny)
field_size(g::Grid, ::HalfSpectral) = (g.nx ÷ 2 + 1, g.ny)

field_eltype(::Grid{T}, ::RealPhysical) where {T} = T
field_eltype(::Grid{T}, ::Union{ComplexPhysical,FullSpectral,HalfSpectral}) where {T} = Complex{T}

"Allocate one zero-initialised array with the given layout on the grid's architecture."
allocate_field(g::Grid, l::Layout) = allocate(g.arch, field_eltype(g, l), field_size(g, l)...)

"Allocate a `NamedTuple` of arrays from a `NamedTuple` of layouts."
allocate_fields(g::Grid, spec::NamedTuple) = map(l -> allocate_field(g, l), spec)

"""
    repeated_layout(layout, names...)

Shortcut for `(name1 = layout, name2 = layout, ...)`:
`repeated_layout(FullSpectral(), :k1, :k2)`.
"""
repeated_layout(l::Layout, names::Symbol...) = NamedTuple{names}(ntuple(_ -> l, length(names)))

# ---------------------------------------------------------------- clock
"Simulation time (always `Float64`, on the host) and iteration counter."
mutable struct Clock
    t::Float64
    iteration::Int
end
Clock() = Clock(0.0, 0)

"Advance the clock by one step of size `dt`."
function advance!(c::Clock, dt::Real)
    c.t += Float64(dt)
    c.iteration += 1
    return c
end

# ---------------------------------------------------------------- State
abstract type AbstractState end

"""
    State(grid, layout::NamedTuple)

Evolving fields (`NamedTuple` of arrays) + `Clock`. Fields are accessed as
properties: `state.ψk`, `state.clock.t`.
"""
struct State{NT<:NamedTuple} <: AbstractState
    fields::NT
    clock::Clock
end
State(g::Grid, layout::NamedTuple) = State(allocate_fields(g, layout), Clock())

fields(s::State) = getfield(s, :fields)
clock(s::State)  = getfield(s, :clock)

Base.getproperty(s::State, name::Symbol) =
    (name === :fields || name === :clock) ? getfield(s, name) : getfield(getfield(s, :fields), name)
Base.propertynames(s::State) = (:fields, :clock, keys(getfield(s, :fields))...)

"Independent copy of a state (arrays and clock)."
function Base.copy(s::State)
    c = clock(s)
    return State(map(copy, fields(s)), Clock(c.t, c.iteration))
end

"Copy `src` into `dst` in place (same layout required): no allocation."
function Base.copyto!(dst::State{NT}, src::State{NT}) where {NT}
    foreach(copyto!, values(fields(dst)), values(fields(src)))
    clock(dst).t = clock(src).t
    clock(dst).iteration = clock(src).iteration
    return dst
end

"All fields of a state as plain host `Array`s (for output / checkpoints)."
host_fields(s::State) = map(to_host, fields(s))

# ---------------------------------------------------------------- Workspace
"""
    Workspace(grid, layout::NamedTuple)

Scratch buffers (RK stages, temporaries, ...), allocated once and reused every
step. Buffers are accessed as properties: `ws.k1`.
"""
struct Workspace{NT<:NamedTuple}
    buffers::NT
end
Workspace(g::Grid, layout::NamedTuple) = Workspace(allocate_fields(g, layout))

Base.getproperty(w::Workspace, name::Symbol) =
    name === :buffers ? getfield(w, :buffers) : getfield(getfield(w, :buffers), name)
Base.propertynames(w::Workspace) = (:buffers, keys(getfield(w, :buffers))...)

# ---------------------------------------------------------------- memory & printing
"Memory used by an array / NamedTuple of arrays / State / Workspace, in bytes."
memory_bytes(A::AbstractArray) = sizeof(A)
memory_bytes(nt::NamedTuple)   = sum(memory_bytes, values(nt); init = 0)
memory_bytes(s::State)         = memory_bytes(fields(s))
memory_bytes(w::Workspace)     = memory_bytes(getfield(w, :buffers))

function _show_fields(io::IO, nt::NamedTuple)
    for (k, A) in pairs(nt)
        println(io, "  ", k, " :: ", eltype(A), " ", size(A))
    end
end

function Base.show(io::IO, ::MIME"text/plain", s::State)
    println(io, "State  (t = ", clock(s).t, ", iteration = ", clock(s).iteration,
            ", ", round(memory_bytes(s) / 2^20; digits = 1), " MiB)")
    _show_fields(io, fields(s))
end

function Base.show(io::IO, ::MIME"text/plain", w::Workspace)
    println(io, "Workspace  (", round(memory_bytes(w) / 2^20; digits = 1), " MiB)")
    _show_fields(io, getfield(w, :buffers))
end

end # module
