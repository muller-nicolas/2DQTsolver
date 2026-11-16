"""
    Output

HDF5 output. One file per run, with three independent streams, each with its own
frequency (in time steps; `0` disables the stream):

* **fields**       snapshots of the fields (for the GPE: `psi_re`, `psi_im`)
* **diagnostics**  scalar time series (for the GPE: `N`, `E`)
* **spectra**      1D/ND arrays computed from the state (hook ready; the GPE spectra
                   themselves will be added with the spectral analysis module)

The writer itself knows nothing about the equation. What to write is described by an
`OutputSpec`, and each model provides `default_outputs(model)`.

Use as a callback (it checks by itself which streams are due):

    w = OutputWriter("output/run.h5", state, model;
                     fields_every = 100, diagnostics_every = 20, spectra_every = 0,
                     nsteps = 3000, metadata = cfg)
    w(state)                                   # iteration 0
    evolve!(state, stepper, model, 3000; callback = w)
    write_final!(w, state); close(w)

File layout
-----------
    /grid            x, y, kx, ky              attrs: nx, ny, Lx, Ly, dx, dy, precision, ...
    /config          attrs: everything in `metadata` (e.g. the parameter file)
    /parameters      attrs: model parameters (GPE: c, xi, n0, mu, omega0, ...)
    /snapshots/<it>/psi_re, psi_im             attrs: t, iteration   (<it> = 8-digit iteration)
    /timeseries      t, iteration, N, E        (preallocated; unwritten entries have
                                                iteration = -1 and are dropped by `read_timeseries`)
    /spectra/<it>/...                          attrs: t, iteration

Orientation: a Julia array `ψ[i, j]` (i along x, j along y) is stored as is. From
Python (h5py) it appears with shape `(ny, nx)`, i.e. `psi[j, i]`, which is what
`imshow` expects.

Reading: `snapshot_names`, `read_snapshot`, `read_timeseries`, `read_attributes`.
"""
module Output

using HDF5
using Printf
using ..Grids
using ..Fields
using ..Models
using ..GrossPitaevskii
using ..Diagnostics

export OutputSpec, OutputWriter, default_outputs,
       write_fields!, write_diagnostics!, write_spectra!, write_final!,
       snapshot_names, read_snapshot, read_timeseries, read_attributes

# ================================================================== what to write
"""
    OutputSpec(; fields, scalars, spectra, metadata)

Description of the outputs of a model. Each entry is a function of `(state, model)`
(`metadata` of `model` only), or `nothing` if the model has no such output:

* `fields(state, model)  -> NamedTuple` of host arrays, e.g. `(psi_re = ..., psi_im = ...)`
* `scalars(state, model) -> NamedTuple` of numbers,     e.g. `(N = ..., E = ...)`
* `spectra(state, model) -> NamedTuple` of host arrays, e.g. `(k = ..., Ek = ...)`
* `metadata(model)       -> NamedTuple` of numbers/strings, stored as attributes
"""
struct OutputSpec{F, S, P, M}
    fields::F
    scalars::S
    spectra::P
    metadata::M
end
OutputSpec(; fields = nothing, scalars = nothing, spectra = nothing,
           metadata = m -> NamedTuple()) = OutputSpec(fields, scalars, spectra, metadata)

default_outputs(m::AbstractModel) =
    error("no default outputs defined for models of type $(typeof(m)); pass `outputs = OutputSpec(...)`")

function default_outputs(::GPE)
    fields = (s, m) -> begin
        update_physical!(s, m)
        ψ = Array(s.ψ)
        (psi_re = real.(ψ), psi_im = imag.(ψ))
    end
    scalars = (s, m) -> (N = particle_number(s, m), E = energy(s, m))
    metadata = m -> (c = m.params.c, xi = m.params.ξ, n0 = m.params.n0, mu = m.params.μ,
                     omega0 = m.params.ω0, hbar_over_m = m.params.hbar_over_m, g = m.params.g)
    # spectra: to be provided by the spectral analysis module
    return OutputSpec(; fields = fields, scalars = scalars, metadata = metadata)
end

# ================================================================== attributes helpers
# Write a NamedTuple as HDF5 attributes. Nested NamedTuples become "outer.inner".
function _write_attributes!(obj, nt::NamedTuple, prefix::AbstractString = "")
    attrs = attributes(obj)
    for (k, v) in pairs(nt)
        name = prefix * string(k)
        if v isa NamedTuple
            _write_attributes!(obj, v, name * ".")
        elseif v isa Bool
            attrs[name] = Int(v)
        elseif v isa Union{Number, AbstractString}
            attrs[name] = v
        elseif v isa Tuple && all(x -> x isa Number, v)
            attrs[name] = collect(v)
        else
            attrs[name] = string(v)
        end
    end
    return nothing
end

function _write_grid!(file, grid)
    g = create_group(file, "grid")
    g["x"]  = xcoords(grid)
    g["y"]  = ycoords(grid)
    g["kx"] = kxcoords(grid)
    g["ky"] = kycoords(grid)
    _write_attributes!(g, (nx = grid.nx, ny = grid.ny,
                           Lx = Float64(grid.Lx), Ly = Float64(grid.Ly),
                           dx = Float64(grid.dx), dy = Float64(grid.dy),
                           precision = string(float_type(grid)),
                           dealias_fraction = grid.dealias_frac,
                           dealias_shape = string(grid.mask_shape)))
    return nothing
end

# ================================================================== the writer
mutable struct OutputWriter{M, S<:OutputSpec, F}
    path::String
    file::F
    model::M
    outputs::S
    fields_every::Int
    diagnostics_every::Int
    spectra_every::Int
    capacity::Int                       # preallocated length of the time series
    nsamples::Int                       # samples written so far
    series::Dict{Symbol, Any}           # open datasets of /timeseries
    last_fields::Int                    # iteration of the last write (avoid duplicates)
    last_diagnostics::Int
    last_spectra::Int
    warned_capacity::Bool
end

"""
    OutputWriter(path, state, model; fields_every = 0, diagnostics_every = 0,
                 spectra_every = 0, nsteps, outputs = default_outputs(model),
                 metadata = NamedTuple())

Create the HDF5 file `path` (overwriting an existing one) and write the grid, the
configuration and the model parameters. `nsteps` is the planned number of steps; it
sizes the preallocated time series. `metadata` (e.g. the parameter `NamedTuple`) is
stored in `/config`.

The writer must be closed with `close(writer)`.
"""
function OutputWriter(path::AbstractString, state, model::AbstractModel;
                      fields_every::Integer = 0, diagnostics_every::Integer = 0,
                      spectra_every::Integer = 0, nsteps::Integer,
                      outputs::OutputSpec = default_outputs(model),
                      metadata::NamedTuple = NamedTuple())
    mkpath(dirname(abspath(path)))
    isfile(path) && @warn "Overwriting existing output file" path

    # a stream is active only if it is requested AND the model provides it
    fe = (fields_every > 0 && outputs.fields !== nothing) ? Int(fields_every) : 0
    de = (diagnostics_every > 0 && outputs.scalars !== nothing) ? Int(diagnostics_every) : 0
    se = (spectra_every > 0 && outputs.spectra !== nothing) ? Int(spectra_every) : 0
    if spectra_every > 0 && outputs.spectra === nothing
        @warn "Spectral output requested (spectra_every = $spectra_every) but this model has no spectra output yet; it is disabled."
    end

    file = h5open(path, "w")
    attributes(file)["format"] = "GPE2D output v1"
    _write_grid!(file, model.grid)
    _write_attributes!(create_group(file, "config"), metadata)
    _write_attributes!(create_group(file, "parameters"), outputs.metadata(model))

    fe > 0 && create_group(file, "snapshots")
    se > 0 && create_group(file, "spectra")

    series = Dict{Symbol, Any}()
    capacity = 0
    if de > 0
        capacity = nsteps ÷ de + 2                      # iteration 0, every `de`, and the final state
        g = create_group(file, "timeseries")
        g["t"] = fill(NaN, capacity)
        g["iteration"] = fill(-1, capacity)
        series[:t] = g["t"]
        series[:iteration] = g["iteration"]
        for k in keys(outputs.scalars(state, model))    # discover the names of the scalars
            g[string(k)] = fill(NaN, capacity)
            series[k] = g[string(k)]
        end
    end
    flush(file)

    return OutputWriter(String(path), file, model, outputs, fe, de, se, capacity, 0, series,
                        -1, -1, -1, false)
end

Base.close(w::OutputWriter) = close(w.file)

function Base.show(io::IO, w::OutputWriter)
    print(io, "OutputWriter(", w.path, "; fields every ", w.fields_every,
          ", diagnostics every ", w.diagnostics_every, ", spectra every ", w.spectra_every, ")")
end

# ---------------------------------------------------------------- writing
# one group per iteration: <parent>/<iteration>/<name> = array, with attributes t, iteration
function _write_group!(file, parent::AbstractString, c::Clock, arrays::NamedTuple)
    sg = create_group(file[parent], @sprintf("%08d", c.iteration))
    attrs = attributes(sg)
    attrs["t"] = c.t
    attrs["iteration"] = c.iteration
    for (k, v) in pairs(arrays)
        sg[string(k)] = v
    end
    return nothing
end

"Write a snapshot of the fields (once per iteration)."
function write_fields!(w::OutputWriter, state)
    c = clock(state)
    (w.fields_every == 0 || w.last_fields == c.iteration) && return nothing
    _write_group!(w.file, "snapshots", c, w.outputs.fields(state, w.model))
    w.last_fields = c.iteration
    flush(w.file)
    return nothing
end

"Append a sample to the scalar time series (once per iteration)."
function write_diagnostics!(w::OutputWriter, state)
    c = clock(state)
    (w.diagnostics_every == 0 || w.last_diagnostics == c.iteration) && return nothing
    if w.nsamples >= w.capacity
        if !w.warned_capacity
            @warn "Time series full ($(w.capacity) samples preallocated from `nsteps`); further samples are not stored."
            w.warned_capacity = true
        end
        return nothing
    end
    i = w.nsamples + 1
    vals = w.outputs.scalars(state, w.model)
    w.series[:t][i] = c.t
    w.series[:iteration][i] = c.iteration
    for (k, v) in pairs(vals)
        w.series[k][i] = Float64(v)
    end
    w.nsamples = i
    w.last_diagnostics = c.iteration
    flush(w.file)
    return nothing
end

"Write a spectral output (once per iteration)."
function write_spectra!(w::OutputWriter, state)
    c = clock(state)
    (w.spectra_every == 0 || w.last_spectra == c.iteration) && return nothing
    _write_group!(w.file, "spectra", c, w.outputs.spectra(state, w.model))
    w.last_spectra = c.iteration
    flush(w.file)
    return nothing
end

"Callback: write whichever streams are due at the current iteration (iteration 0 counts)."
function (w::OutputWriter)(state)
    it = clock(state).iteration
    due(every) = every > 0 && it % every == 0
    due(w.fields_every)      && write_fields!(w, state)
    due(w.diagnostics_every) && write_diagnostics!(w, state)
    due(w.spectra_every)     && write_spectra!(w, state)
    return nothing
end

"Write all active streams now (e.g. the final state), unless already written at this iteration."
function write_final!(w::OutputWriter, state)
    write_fields!(w, state)
    write_diagnostics!(w, state)
    write_spectra!(w, state)
    return nothing
end

# ================================================================== reading
"Names (8-digit iteration numbers) of the snapshots in `group` (`\"snapshots\"` or `\"spectra\"`), sorted."
function snapshot_names(path::AbstractString; group::AbstractString = "snapshots")
    h5open(path, "r") do f
        haskey(f, group) ? sort(collect(keys(f[group]))) : String[]
    end
end

"""
    read_snapshot(path, which; group = "snapshots")

Read one snapshot, given by its position in `snapshot_names` (an integer) or by its
name. Returns `(t, iteration, fields)` with `fields` a `NamedTuple` of arrays.
"""
function read_snapshot(path::AbstractString, which::Union{Integer, AbstractString};
                       group::AbstractString = "snapshots")
    h5open(path, "r") do f
        g = f[group]
        name = which isa Integer ? sort(collect(keys(g)))[which] : String(which)
        sg = g[name]
        arrays = (; (Symbol(k) => read(sg[k]) for k in sort(collect(keys(sg))))...)
        attrs = attributes(sg)
        return (t = read(attrs["t"]), iteration = Int(read(attrs["iteration"])), fields = arrays)
    end
end

"Read the scalar time series as a `NamedTuple` of vectors (only the samples actually written)."
function read_timeseries(path::AbstractString)
    h5open(path, "r") do f
        g = f["timeseries"]
        valid = read(g["iteration"]) .>= 0
        return (; (Symbol(n) => read(g[n])[valid] for n in sort(collect(keys(g))))...)
    end
end

"Read the attributes of a group (`\"config\"`, `\"parameters\"`, `\"grid\"`) as a `Dict`."
function read_attributes(path::AbstractString, group::AbstractString)
    h5open(path, "r") do f
        a = attributes(f[group])
        return Dict(String(k) => read(a[k]) for k in keys(a))
    end
end

end # module
