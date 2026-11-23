"""
    Output

HDF5 output. One file per run, with four independent streams, each with its own
frequency (in time steps; `0` disables the stream):

* **fields**       snapshots of the fields (for the GPE: `psi_re`, `psi_im`)
* **diagnostics**  scalar time series (for the GPE: `N`, `E`)
* **spectra**      arrays computed from the state (for the GPE: the energy spectra
                   `k`, `E_kin_inc`, `E_kin_comp`, `E_quantum`, `E_internal`,
                   see `SpectralAnalysis`)
* **traces**       vectors saved at every sample and stored as the columns of 2D
                   datasets (for the GPE: ψ̂(kx, ky=0) and ψ̂(kx=0, ky), used to get
                   the dispersion relation ω(k) in post-processing)

The writer itself knows nothing about the equation. What to write is described by an
`OutputSpec`, and each model provides `default_outputs(model)`.

Use as a callback (it checks by itself which streams are due):

    w = OutputWriter("output/run.h5", state, model;
                     fields_every = 100, diagnostics_every = 20, spectra_every = 200,
                     traces_every = 1, nsteps = 3000, metadata = cfg)
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
    /traces          t, iteration, psi_kx_re, psi_kx_im, psi_ky_re, psi_ky_im
                                               (2D: one column per sample, preallocated;
                                                unwritten columns have iteration = -1)

Orientation: a Julia array `ψ[i, j]` (i along x, j along y) is stored as is. From
Python (h5py) it appears with shape `(ny, nx)`, i.e. `psi[j, i]`, which is what
`imshow` expects. The trace datasets (Julia `(nk, nsamples)`) appear in h5py as
`(nsamples, nk)`: time first, so `np.fft.fft(psi, axis=0)` is the time transform.

Traces are written only at iterations that are multiples of `traces_every` (and at
iteration 0), never at an extra "final" iteration, so the time sampling is exactly
uniform. They are buffered in memory and written to disk in blocks (and on `close`).

Reading: `snapshot_names`, `read_snapshot`, `read_timeseries`, `read_traces`,
`read_dispersion`, `read_attributes`.
"""
module Output

using HDF5
using Printf
using ..Grids
using ..Fields
using ..Models
using ..GrossPitaevskii
using ..Diagnostics
using ..SpectralAnalysis

export OutputSpec, OutputWriter, default_outputs,
       write_fields!, write_diagnostics!, write_spectra!, write_traces!, write_final!,
       snapshot_names, read_snapshot, read_timeseries, read_traces, read_dispersion,
       read_attributes

# ================================================================== what to write
"""
    OutputSpec(; fields, scalars, spectra, traces, metadata)

Description of the outputs of a model. Each entry is a function of `(state, model)`
(`metadata` of `model` only), or `nothing` if the model has no such output:

* `fields(state, model)  -> NamedTuple` of host arrays, e.g. `(psi_re = ..., psi_im = ...)`
* `scalars(state, model) -> NamedTuple` of numbers,     e.g. `(N = ..., E = ...)`
* `spectra(state, model) -> NamedTuple` of host arrays, e.g. `(k = ..., Ek = ...)`
* `traces(state, model)  -> NamedTuple` of host VECTORS of fixed length
* `metadata(model)       -> NamedTuple` of numbers/strings, stored as attributes
"""
struct OutputSpec{F, S, P, TR, M}
    fields::F
    scalars::S
    spectra::P
    traces::TR
    metadata::M
end
OutputSpec(; fields = nothing, scalars = nothing, spectra = nothing, traces = nothing,
           metadata = m -> NamedTuple()) = OutputSpec(fields, scalars, spectra, traces, metadata)

default_outputs(m::AbstractModel) =
    error("no default outputs defined for models of type $(typeof(m)); pass `outputs = OutputSpec(...)`")

function default_outputs(::GPE)
    fields = (s, m) -> begin
        update_physical!(s, m)
        ψ = Array(s.ψ)
        (psi_re = real.(ψ), psi_im = imag.(ψ))
    end
    # scalars = (s, m) -> (N = particle_number(s, m), E,Ek,Eint,Epot = energy(s, m))
    scalars = (s, m) -> (N = particle_number(s, m), E = energy(s, m))
    metadata = m -> (c = m.params.c, xi = m.params.ξ, n0 = m.params.n0, mu = m.params.μ,
                     omega0 = m.params.ω0, hbar_over_m = m.params.hbar_over_m, g = m.params.g)
    # energy spectra; the analyzer (buffers + shells) is created on first use only
    analyzer = Ref{Any}(nothing)
    spectra = (s, m) -> begin
        analyzer[] === nothing && (analyzer[] = SpectraAnalyzer(m))
        energy_spectra(analyzer[], s)
    end
    # ψ̂ on the two axes of Fourier space, as Fourier coefficients c_k = ψ̂/(nx ny),
    # i.e. ψ(x) = Σ c_k e^{ik·x}. FFT ordering (index 1 is k = 0).
    traces = (s, m) -> begin
        N = ncells(m.grid)
        cx = Array(s.ψk[:, 1]) ./ N                  # ψ(kx, ky = 0)
        cy = Array(s.ψk[1, :]) ./ N                  # ψ(kx = 0, ky)
        (psi_kx_re = real.(cx), psi_kx_im = imag.(cx),
         psi_ky_re = real.(cy), psi_ky_im = imag.(cy))
    end
    return OutputSpec(; fields = fields, scalars = scalars, spectra = spectra,
                      traces = traces, metadata = metadata)
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
    traces_every::Int
    traces_block::Int                   # samples buffered in memory before writing
    traces_capacity::Int                # preallocated number of trace columns
    ntraces::Int                        # trace samples already in the file
    trace_series::Dict{Symbol, Any}     # open datasets of /traces
    pending_t::Vector{Float64}          # buffered, not yet written
    pending_it::Vector{Int}
    pending::Vector{Any}
    last_traces::Int
    warned_traces::Bool
end

"""
    OutputWriter(path, state, model; fields_every = 0, diagnostics_every = 0,
                 spectra_every = 0, traces_every = 0, nsteps,
                 outputs = default_outputs(model), metadata = NamedTuple(),
                 traces_block = 64)

Create the HDF5 file `path` (overwriting an existing one) and write the grid, the
configuration and the model parameters. `nsteps` is the planned number of steps; it
sizes the preallocated time series and traces. `metadata` (e.g. the parameter
`NamedTuple`) is stored in `/config`.

The writer must be closed with `close(writer)` (this also writes buffered traces).
"""
function OutputWriter(path::AbstractString, state, model::AbstractModel;
                      fields_every::Integer = 0, diagnostics_every::Integer = 0,
                      spectra_every::Integer = 0, traces_every::Integer = 0,
                      nsteps::Integer, traces_block::Integer = 64,
                      outputs::OutputSpec = default_outputs(model),
                      metadata::NamedTuple = NamedTuple())
    mkpath(dirname(abspath(path)))
    isfile(path) && @warn "Overwriting existing output file" path

    # a stream is active only if it is requested AND the model provides it
    fe = (fields_every > 0 && outputs.fields !== nothing) ? Int(fields_every) : 0
    de = (diagnostics_every > 0 && outputs.scalars !== nothing) ? Int(diagnostics_every) : 0
    se = (spectra_every > 0 && outputs.spectra !== nothing) ? Int(spectra_every) : 0
    te = (traces_every > 0 && outputs.traces !== nothing) ? Int(traces_every) : 0
    if spectra_every > 0 && outputs.spectra === nothing
        @warn "Spectral output requested (spectra_every = $spectra_every) but this model has no spectra output; it is disabled."
    end
    if traces_every > 0 && outputs.traces === nothing
        @warn "Trace (dispersion) output requested (traces_every = $traces_every) but this model has no traces output; it is disabled."
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

    trace_series = Dict{Symbol, Any}()
    traces_capacity = 0
    if te > 0
        traces_capacity = nsteps ÷ te + 2               # iteration 0 and every `te`
        g = create_group(file, "traces")
        g["t"] = fill(NaN, traces_capacity)
        g["iteration"] = fill(-1, traces_capacity)
        trace_series[:t] = g["t"]
        trace_series[:iteration] = g["iteration"]
        for (k, v) in pairs(outputs.traces(state, model))   # discover names and lengths
            g[string(k)] = fill(NaN, length(v), traces_capacity)
            trace_series[k] = g[string(k)]
        end
    end
    flush(file)

    return OutputWriter(String(path), file, model, outputs, fe, de, se, capacity, 0, series,
                        -1, -1, -1, false,
                        te, max(Int(traces_block), 1), traces_capacity, 0, trace_series,
                        Float64[], Int[], Any[], -1, false)
end

function Base.close(w::OutputWriter)
    _flush_traces!(w)                   # do not lose the buffered trace samples
    close(w.file)
end

function Base.show(io::IO, w::OutputWriter)
    print(io, "OutputWriter(", w.path, "; fields every ", w.fields_every,
          ", diagnostics every ", w.diagnostics_every, ", spectra every ", w.spectra_every,
          ", traces every ", w.traces_every, ")")
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

# write the buffered trace samples to the file (one block)
function _flush_traces!(w::OutputWriter)
    n = length(w.pending)
    n == 0 && return nothing
    i0, i1 = w.ntraces + 1, w.ntraces + n
    w.trace_series[:t][i0:i1] = w.pending_t
    w.trace_series[:iteration][i0:i1] = w.pending_it
    for k in keys(w.pending[1])
        w.trace_series[k][:, i0:i1] = hcat((Float64.(p[k]) for p in w.pending)...)
    end
    w.ntraces = i1
    empty!(w.pending); empty!(w.pending_t); empty!(w.pending_it)
    flush(w.file)
    return nothing
end

"""
Take a trace sample (once per iteration). Samples are buffered and written to disk in
blocks of `traces_block`, and by `write_final!` / `close`.
"""
function write_traces!(w::OutputWriter, state)
    c = clock(state)
    (w.traces_every == 0 || w.last_traces == c.iteration) && return nothing
    if w.ntraces + length(w.pending) >= w.traces_capacity
        if !w.warned_traces
            @warn "Trace output full ($(w.traces_capacity) samples preallocated from `nsteps`); further samples are not stored."
            w.warned_traces = true
        end
        return nothing
    end
    push!(w.pending, w.outputs.traces(state, w.model))
    push!(w.pending_t, c.t)
    push!(w.pending_it, c.iteration)
    w.last_traces = c.iteration
    length(w.pending) >= w.traces_block && _flush_traces!(w)
    return nothing
end

"Callback: write whichever streams are due at the current iteration (iteration 0 counts)."
function (w::OutputWriter)(state)
    it = clock(state).iteration
    due(every) = every > 0 && it % every == 0
    due(w.fields_every)      && write_fields!(w, state)
    due(w.diagnostics_every) && write_diagnostics!(w, state)
    due(w.spectra_every)     && write_spectra!(w, state)
    due(w.traces_every)      && write_traces!(w, state)
    return nothing
end

"""
Write the final state of the fields, diagnostics and spectra now (unless already written
at this iteration) and flush the buffered traces. No extra trace sample is taken: that
would break the uniform time sampling needed for the dispersion analysis.
"""
function write_final!(w::OutputWriter, state)
    write_fields!(w, state)
    write_diagnostics!(w, state)
    write_spectra!(w, state)
    _flush_traces!(w)
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

"""
Read the traces as a `NamedTuple` (only the samples actually written): `t`, `iteration`
and the 2D arrays with one COLUMN per sample (Julia shape `(nk, nsamples)`).
"""
function read_traces(path::AbstractString)
    h5open(path, "r") do f
        g = f["traces"]
        valid = read(g["iteration"]) .>= 0
        out = Pair{Symbol, Any}[]
        for n in sort(collect(keys(g)))
            d = read(g[n])
            push!(out, Symbol(n) => (ndims(d) == 1 ? d[valid] : d[:, valid]))
        end
        return (; out...)
    end
end

"""
    read_dispersion(path)

The GPE dispersion traces as complex Fourier coefficients (ψ = Σ c_k e^{ik·x}), as a
`NamedTuple`:

    t, iteration          sample times and iterations (uniformly spaced)
    kx, ky                wavenumber axes in FFT order (index 1 is k = 0)
    psi_kx                ψ̂(kx, ky = 0, t), complex matrix of size (nx, nsamples)
    psi_ky                ψ̂(kx = 0, ky, t), complex matrix of size (ny, nsamples)

For the dispersion relation, Fourier transform along time (dimension 2), e.g.
`fft(psi_kx, 2)`, and use `fftshift` / `fftfreq` to order k and ω.
"""
function read_dispersion(path::AbstractString)
    tr = read_traces(path)
    kx, ky = h5open(path, "r") do f
        read(f["grid/kx"]), read(f["grid/ky"])
    end
    return (t = tr.t, iteration = tr.iteration, kx = kx, ky = ky,
            psi_kx = complex.(tr.psi_kx_re, tr.psi_kx_im),
            psi_ky = complex.(tr.psi_ky_re, tr.psi_ky_im))
end

"Read the attributes of a group (`\"config\"`, `\"parameters\"`, `\"grid\"`) as a `Dict`."
function read_attributes(path::AbstractString, group::AbstractString)
    h5open(path, "r") do f
        a = attributes(f[group])
        return Dict(String(k) => read(a[k]) for k in keys(a))
    end
end

end # module
