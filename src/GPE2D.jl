module GPE2D

# Dependency order (each file only uses the ones above it):
#   Architectures → Grid → Fields → FFT → Models → Spectral → TimeSteppers
#   (next: Models/GPE, Potentials, InitialConditions, Diagnostics, Output, Simulation)
include("Architectures.jl")
include("Grid.jl")
include("Fields.jl")
include("FFT.jl")
include("Models.jl")
include("Spectral.jl")
include("TimeSteppers.jl")

using .Architectures
using .Grids
using .Fields
using .FFT
using .Models
using .Spectral
using .TimeSteppers

# Re-export the public API
export AbstractArchitecture, CPU, CUDAGPU, MetalGPU,
       array_type, allocate, to_device, to_host,
       device_synchronize, architecture_name, supports_precision
export Grid, float_type, ncells, cell_area, domain_area, nyquist, half_size,
       xcoords, ycoords, kxcoords, kycoords, wavenumber_squared, dealias_fraction
export Layout, RealPhysical, ComplexPhysical, FullSpectral, HalfSpectral,
       field_size, field_eltype, allocate_field, allocate_fields, repeated_layout,
       Clock, advance!, AbstractState, State, Workspace,
       fields, clock, host_fields, memory_bytes, GPE_STATE_LAYOUT, gpe_state
export FFTPlans, forward!, backward!, to_spectral!, to_physical!
export AbstractModel, linear_rate, nonlinear!, evolved_field
export LaplacianRate, dealias!, ddx!, ddy!, gradient!, laplacian!, inverse_laplacian!,
       apply_rate!, integrating_factor
export AbstractTimeStepper, RK4, IFRK4, step!, evolve!, set_dt!, timestep

end # module
