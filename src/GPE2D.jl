module GPE2D

# Dependency order (each file only uses the ones above it):
#   Architectures → Grid → Fields → (FFT → Spectral → Models/GPE → TimeSteppers → ...)
include("Architectures.jl")
include("Grid.jl")
include("Fields.jl")

using .Architectures
using .Grids
using .Fields

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

end # module
