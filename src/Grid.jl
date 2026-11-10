"""
    Grids

Periodic 2D Cartesian grid + the associated Fourier-space information
(wavenumbers, dealiasing mask). Independent of any particular equation, so it is
shared by the GPE solver and any future solver.

Conventions
-----------
* Domain `[-Lx/2, Lx/2) x [-Ly/2, Ly/2)` (or `[0, L)` with `origin = :corner`).
* Field arrays are indexed `[i, j]` with `i` along x and `j` along y.
* Wavenumbers follow the standard FFT ordering: `k = 2π/L * [0, 1, ..., n/2-1, -n/2, ..., -1]`.
  (The Nyquist mode is stored as `-n/2`. When taking ODD derivatives, the Nyquist
  mode should be zeroed: the Spectral module will take care of that.)
* `x`, `y`, `kx`, `ky` are stored on the device with *broadcast shapes*
  (`x`, `kx`: `nx × 1`, `y`, `ky`: `1 × ny`), so that kernels can be written as
  `@. out = f(x, y)` or `@. k² = kx^2 + ky^2` with no extra memory.
* `kx_r` is the same as `kx` but for the half-spectrum layout of `rfft`
  (`nx÷2+1` entries) for real fields.
* The dealiasing mask is stored as 0/1 values of the field precision `T`
  (not `Bool`) to stay friendly to all GPU backends.
"""
module Grids

using ..Architectures
using AbstractFFTs: fftfreq, rfftfreq

export Grid, float_type, ncells, cell_area, domain_area, nyquist, half_size,
       xcoords, ycoords, kxcoords, kycoords, wavenumber_squared, dealias_fraction

struct Grid{T<:AbstractFloat, A<:AbstractArchitecture,
            K<:AbstractMatrix{T}, M<:AbstractMatrix{T}}
    arch::A
    nx::Int
    ny::Int
    Lx::T
    Ly::T
    dx::T
    dy::T
    x::K                    # nx × 1   (device)
    y::K                    # 1  × ny  (device)
    kx::K                   # nx × 1   (device, FFT order)
    ky::K                   # 1  × ny  (device, FFT order)
    kx_r::K                 # (nx÷2+1) × 1  (device, rfft order)
    mask::M                 # nx × ny  (device), 1 = keep, 0 = remove
    kcut::NTuple{2,T}       # dealiasing cutoff in (kx, ky)
    dealias_frac::Float64   # cutoff as a fraction of the Nyquist wavenumber
    mask_shape::Symbol      # :circular | :square | :none
end

# ---------------------------------------------------------------- constructor
"""
    dealias_fraction(spec)

Cutoff as a fraction of the Nyquist wavenumber.
`:two_thirds` → 2/3 (exact for quadratic nonlinearities),
`:half` → 1/2 (exact for the cubic GPE nonlinearity), `:none` → no truncation,
or any number in `(0, 1]`.
"""
function dealias_fraction(s::Symbol)
    s === :two_thirds && return 2 / 3
    s === :half       && return 1 / 2
    s === :none       && return 1.0
    throw(ArgumentError("unknown dealias spec :$s (use :two_thirds, :half, :none or a number in (0,1])"))
end
function dealias_fraction(f::Real)
    0 < f <= 1 || throw(ArgumentError("dealias fraction must be in (0, 1], got $f"))
    return Float64(f)
end

"""
    Grid(arch = CPU(), T = Float64; nx, ny = nx, Lx, Ly = Lx,
         origin = :center, dealias = :two_thirds, mask_shape = :circular)

Build a periodic grid on architecture `arch` with floating-point type `T`.
`nx` and `ny` must be even (powers of two are best for FFT speed on GPUs).
Lengths are in whatever units you use for the physics (e.g. multiples of ξ).

    grid = Grid(MetalGPU(), Float32; nx = 1024, Lx = 128.0)
"""
function Grid(arch::AbstractArchitecture = CPU(), ::Type{T} = Float64;
              nx::Integer, ny::Integer = nx, Lx::Real, Ly::Real = Lx,
              origin::Symbol = :center, dealias = :two_thirds,
              mask_shape::Symbol = :circular) where {T<:AbstractFloat}

    supports_precision(arch, T) ||
        throw(ArgumentError("$(architecture_name(arch)) does not support $T; use Float32."))
    (iseven(nx) && iseven(ny) && nx >= 4 && ny >= 4) ||
        throw(ArgumentError("nx and ny must be even and >= 4 (got $nx × $ny)"))
    (Lx > 0 && Ly > 0) || throw(ArgumentError("Lx and Ly must be positive"))
    origin in (:center, :corner) || throw(ArgumentError("origin must be :center or :corner"))
    mask_shape in (:circular, :square) || throw(ArgumentError("mask_shape must be :circular or :square"))

    # Everything is first built on the host in Float64, then converted to T.
    Lx64, Ly64 = Float64(Lx), Float64(Ly)
    dx, dy = Lx64 / nx, Ly64 / ny
    x0 = origin === :center ? -Lx64 / 2 : 0.0
    y0 = origin === :center ? -Ly64 / 2 : 0.0
    x_h = [x0 + (i - 1) * dx for i in 1:nx]
    y_h = [y0 + (j - 1) * dy for j in 1:ny]

    # Wavenumbers come from AbstractFFTs, i.e. from the same library whose plans do
    # the transforms (see FFT.jl), so the ordering cannot get out of sync with the
    # data. fs = n/L plays the role of the sampling frequency 1/dx.
    kx_h  = collect(2π .* fftfreq(nx, nx / Lx64))
    ky_h  = collect(2π .* fftfreq(ny, ny / Ly64))
    kxr_h = collect(2π .* rfftfreq(nx, nx / Lx64))

    # Dealiasing mask (ellipse/circle or rectangle in k-space)
    frac = dealias_fraction(dealias)
    kcx, kcy = frac * π / dx, frac * π / dy
    tol = 1 + 1e-9                         # keep modes sitting exactly on the cutoff
    if dealias === :none
        mask_h = ones(T, nx, ny)
        shape = :none
    elseif mask_shape === :circular
        mask_h = [T((kx_h[i] / kcx)^2 + (ky_h[j] / kcy)^2 <= tol) for i in 1:nx, j in 1:ny]
        shape = :circular
    else
        mask_h = [T(abs(kx_h[i]) <= kcx * tol && abs(ky_h[j]) <= kcy * tol) for i in 1:nx, j in 1:ny]
        shape = :square
    end

    col(v) = to_device(arch, T.(reshape(v, :, 1)))
    row(v) = to_device(arch, T.(reshape(v, 1, :)))

    return Grid(arch, Int(nx), Int(ny), T(Lx64), T(Ly64), T(dx), T(dy),
                col(x_h), row(y_h), col(kx_h), row(ky_h), col(kxr_h),
                to_device(arch, mask_h), (T(kcx), T(kcy)), frac, shape)
end

# ---------------------------------------------------------------- accessors
float_type(::Grid{T}) where {T} = T
Base.size(g::Grid) = (g.nx, g.ny)

"Size of the half-spectrum (`rfft`) layout."
half_size(g::Grid) = (g.nx ÷ 2 + 1, g.ny)

ncells(g::Grid)      = g.nx * g.ny
cell_area(g::Grid)   = g.dx * g.dy
domain_area(g::Grid) = g.Lx * g.Ly

"Nyquist wavenumbers `(π/dx, π/dy)`."
nyquist(g::Grid{T}) where {T} = (T(π) / g.dx, T(π) / g.dy)

# Host copies of the axes (for output / plotting)
xcoords(g::Grid)  = vec(to_host(g.x))
ycoords(g::Grid)  = vec(to_host(g.y))
kxcoords(g::Grid) = vec(to_host(g.kx))
kycoords(g::Grid) = vec(to_host(g.ky))

"""
    wavenumber_squared(grid)

Allocate and return `kx² + ky²` as an `nx × ny` device array. Convenient for
setup and diagnostics; inside hot loops prefer `@. kx^2 + ky^2` with
`kx, ky = grid.kx, grid.ky` so that nothing is stored.
"""
function wavenumber_squared(g::Grid)
    kx, ky = g.kx, g.ky
    return @. kx^2 + ky^2
end

function Base.show(io::IO, g::Grid{T}) where {T}
    print(io, "Grid{$T} on ", architecture_name(g.arch), ": ", g.nx, "×", g.ny,
          ", L = (", g.Lx, ", ", g.Ly, "), d = (", g.dx, ", ", g.dy, ")",
          ", dealias = ", round(g.dealias_frac; digits = 3), " (", g.mask_shape, ")")
end

end # module
