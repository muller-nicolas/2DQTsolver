"""
    SpectralAnalysis

Energy spectra of the GPE, split into the four standard contributions (Nore, Abid &
Brachet 1997). With ψ = √ρ e^{iφ}, the energy functional of `Diagnostics.energy`
(units of μ, without the external potential) is

    E + N = ξ² ∫|∇ψ|² dA + ∫ ρ²/(2 n0) dA,      |∇ψ|² = |∇√ρ|² + |w|²,   w = √ρ ∇φ

so that

    E_kin  = ξ² ∫ |w|²        → separated with a Helmholtz decomposition of `w`:
        E_kin_inc   incompressible (solenoidal) part   — vortices
        E_kin_comp  compressible (irrotational) part   — sound
    E_quantum = ξ² ∫ |∇√ρ|²                            quantum pressure
    E_internal = ∫ ρ²/(2 n0)                           interaction (internal) energy

(`E + N` because `energy` includes the `−∫ρ` term, i.e. the −μ N shift.)

Computation
-----------
Everything is computed from ∇ψ in physical space, without ever dividing a gradient
of |ψ| (which has cusps at vortex cores):

    w = Im(ψ* ∇ψ)/|ψ|,      ∇√ρ = Re(ψ* ∇ψ)/|ψ|

(set to 0 where |ψ|² ≤ 10⁻¹² n0). The identity |∇ψ|² = |∇√ρ|² + |w|² then holds
point by point, so the spectra add up EXACTLY to the total energy:

    Σ_k (E_kin_inc + E_kin_comp + E_quantum + E_internal) = energy + particle_number

(a good test of the whole chain). Each Fourier-space energy density is summed over
isotropic shells.

Shells
------
`k_n = n · dk`, `n = 0, 1, ...`, with `dk = 2π / max(Lx, Ly)`. Shell `n` holds the modes
with `|k|/dk` rounded to `n`. Shells extend up to the corner of the Fourier box, so no
mode is lost. The returned spectra are shell SUMS (their total is the energy); the
spectral density is `E[n] / dk`.

Notes: the spectra of the nonlinear quantities (w, ∇√ρ, ρ) are computed on the grid
and are affected by aliasing near and above the dealiasing cutoff.
"""
module SpectralAnalysis

using ..Architectures
using ..Grids
using ..Fields
using ..FFT
using ..Spectral
using ..GrossPitaevskii

export ShellBins, shell_sum, SpectraAnalyzer, energy_spectra

# ================================================================== shells
"""
    ShellBins(grid)

Assignment of every Fourier mode to an isotropic shell (host arrays; built once).
`idx[i, j]` is the shell (1-based) of mode `(i, j)`, `k[n]` the wavenumber `(n-1) dk`.
"""
struct ShellBins
    idx::Matrix{Int32}
    nbins::Int
    dk::Float64
    k::Vector{Float64}
end

function ShellBins(grid::Grid)
    kx = Float64.(kxcoords(grid))
    ky = Float64.(kycoords(grid))
    dk = 2π / Float64(max(grid.Lx, grid.Ly))
    idx = Matrix{Int32}(undef, grid.nx, grid.ny)
    for j in 1:grid.ny, i in 1:grid.nx
        idx[i, j] = Int32(round(Int, sqrt(kx[i]^2 + ky[j]^2) / dk) + 1)
    end
    nbins = Int(maximum(idx))
    # nbins = Int(maximum(kx))
    return ShellBins(idx, nbins, dk, collect(dk .* (0:nbins-1)))
end

"""
    shell_sum(weights, bins)

Sum the real array `weights` (`nx × ny`, on the device or the host) over each shell.
The reduction is done on the host in `Float64`, so it is portable to every backend.
"""
function shell_sum(w::AbstractMatrix, bins::ShellBins)
    wh = to_host(w)
    out = zeros(Float64, bins.nbins)
    idx = bins.idx
    @inbounds for n in eachindex(wh)
        out[idx[n]] += wh[n]
    end
    return out
end

# ================================================================== analyzer
# ψ*∇ψ/|ψ| components: real part = ∇√ρ, imaginary part = w = √ρ ∇φ.
# Where the density vanishes the direction is undefined and the value is set to 0.
@inline function _q_comp(ψ, dψ, ρmin)
    r = abs2(ψ)
    return ifelse(r > ρmin, real(conj(ψ) * dψ) / sqrt(max(r, ρmin)), zero(r))
end
@inline function _w_comp(ψ, dψ, ρmin)
    r = abs2(ψ)
    return ifelse(r > ρmin, imag(conj(ψ) * dψ) / sqrt(max(r, ρmin)), zero(r))
end

"""
    SpectraAnalyzer(model::GPE)

Holds the buffers (two complex and one real array of the grid size) and the shells
needed to compute the spectra. Create it once and reuse it.
"""
struct SpectraAnalyzer{M<:GPE, C<:AbstractMatrix, R<:AbstractMatrix}
    model::M
    bins::ShellBins
    a::C
    b::C
    wt::R
end

function SpectraAnalyzer(model::GPE)
    g = model.grid
    return SpectraAnalyzer(model, ShellBins(g),
                           allocate_field(g, ComplexPhysical()),
                           allocate_field(g, ComplexPhysical()),
                           allocate_field(g, RealPhysical()))
end

# ∇ψ in physical space: ∂xψ → a, ∂yψ → b
function _gradient_psi!(a, b, ψk, model)
    ddx!(a, ψk, model.grid); backward!(a, model.plans)
    ddy!(b, ψk, model.grid); backward!(b, model.plans)
    return nothing
end

"""
    energy_spectra(analyzer, state)

Energy spectra of `state` as a `NamedTuple` of `Float64` vectors:

    k             shell wavenumbers (0, dk, 2dk, ...)
    E_kin_inc     incompressible kinetic energy
    E_kin_comp    compressible kinetic energy
    E_quantum     quantum pressure energy
    E_internal    internal (interaction) energy

Shell sums, in the units of `Diagnostics.energy`; spectral density = `E ./ dk`.
Refreshes `state.ψ`. It performs 9 FFTs, so call it at output cadence only.
"""
function energy_spectra(an::SpectraAnalyzer, state)
    m = an.model
    g, plans, p = m.grid, m.plans, m.params
    T = float_type(g)
    update_physical!(state, m)
    ψk, ψ = state.ψk, state.ψ
    a, b, wt = an.a, an.b, an.wt
    kx, ky = g.kx, g.ky
    z = zero(T)
    ρmin = T(1e-12) * p.n0
    area_over_N = Float64(cell_area(g)) / ncells(g)         # Parseval: ∫|f|² = (dA/N) Σ|f̂|²
    kin = Float64(p.ξ)^2 * area_over_N                       # ξ² dA/N

    # --- quantum pressure: q = ∇√ρ = Re(ψ* ∇ψ)/|ψ|
    _gradient_psi!(a, b, ψk, m)
    @. a = _q_comp(ψ, a, ρmin)
    @. b = _q_comp(ψ, b, ρmin)
    forward!(a, plans); forward!(b, plans)
    @. wt = abs2(a) + abs2(b)
    E_quantum = kin .* shell_sum(wt, an.bins)

    # --- kinetic: w = Im(ψ* ∇ψ)/|ψ|, Helmholtz-decomposed in Fourier space
    _gradient_psi!(a, b, ψk, m)
    @. a = _w_comp(ψ, a, ρmin)
    @. b = _w_comp(ψ, b, ρmin)
    forward!(a, plans); forward!(b, plans)                   # a, b ← ŵx, ŵy
    @. wt = abs2(a) + abs2(b)
    E_kin_total = kin .* shell_sum(wt, an.bins)
    # |ŵ_c|² = |k·ŵ|²/k²  (the k = 0 mode belongs to the incompressible part)
    @. wt = ifelse(kx^2 + ky^2 == 0, z, abs2(kx * a + ky * b) / (kx^2 + ky^2))
    E_kin_comp = kin .* shell_sum(wt, an.bins)
    E_kin_inc = E_kin_total .- E_kin_comp

    # --- internal energy: ∫ρ²/(2 n0)
    @. a = abs2(ψ)
    forward!(a, plans)
    @. wt = abs2(a)
    E_internal = (area_over_N / (2 * Float64(p.n0))) .* shell_sum(wt, an.bins)

    return (k = an.bins.k, E_kin_inc = E_kin_inc, E_kin_comp = E_kin_comp,
            E_quantum = E_quantum, E_internal = E_internal)
end

end # module
