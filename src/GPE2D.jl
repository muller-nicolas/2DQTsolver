module GPE2D

# Dependency order (each file only uses the ones above it):
#   Architectures → Grid → Fields → FFT → Models (interface) → Spectral → Potentials
#   → TimeSteppers
#   → Models/GPE → InitialConditions → Diagnostics → SpectralAnalysis → Output
#   (next: spectral analysis, imaginary time, ...)
include("Architectures.jl")
include("Grid.jl")
include("Fields.jl")
include("FFT.jl")
include("Models.jl")
include("Spectral.jl")
include("Potentials.jl")
include("TimeSteppers.jl")
include("Models/GPE.jl")
include("InitialConditions.jl")
include("Diagnostics.jl")
include("SpectralAnalysis.jl")
include("Output.jl")

using .Architectures
using .Grids
using .Fields
using .FFT
using .Models
using .Spectral
using .Potentials
using .TimeSteppers
using .GrossPitaevskii
using .InitialConditions
using .Diagnostics
using .SpectralAnalysis
using .Output

# Re-export the public API
export AbstractArchitecture, CPU, CUDAGPU, MetalGPU,
       array_type, allocate, to_device, to_host,
       device_synchronize, architecture_name, supports_precision
export Grid, float_type, ncells, cell_area, domain_area, nyquist, half_size,
       xcoords, ycoords, kxcoords, kycoords, wavenumber_squared, dealias_fraction
export Layout, RealPhysical, ComplexPhysical, FullSpectral, HalfSpectral,
       field_size, field_eltype, allocate_field, allocate_fields, repeated_layout,
       Clock, advance!, AbstractState, State, Workspace,
       fields, clock, host_fields, memory_bytes
export FFTPlans, forward!, backward!, to_spectral!, to_physical!
export AbstractModel, linear_rate, nonlinear!, evolved_field
export LaplacianRate, dealias!, ddx!, ddy!, gradient!, laplacian!, inverse_laplacian!,
       compressible_part!, incompressible_part!, apply_rate!, integrating_factor
export AbstractPotential, NoPotential, StaticPotential, GaussianObstacle,
       is_static, allocate_potential, evaluate!, refresh!
export AbstractTimeStepper, RK4, IFRK4, step!, evolve!, set_dt!, timestep,
       every, CallbackSet
export GPEParameters, GPE, GPE_STATE_LAYOUT, gpe_state, update_physical!, set_wavefunction!,
       bogoliubov_frequency
export uniform_density, vortices, vortex_dipole, add_noise!, waves!
export particle_number, energy, ConservationLogger
export ShellBins, shell_sum, SpectraAnalyzer, energy_spectra
export OutputSpec, OutputWriter, default_outputs,
       write_fields!, write_diagnostics!, write_spectra!, write_traces!, write_final!,
       snapshot_names, read_snapshot, read_timeseries, read_traces, read_dispersion,
       read_attributes

end # module
