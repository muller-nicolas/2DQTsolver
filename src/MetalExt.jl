module MetalExt

using Metal
import GPE2D.Architectures: array_type, device_synchronize, MetalGPU

array_type(::MetalGPU) = Metal.MtlArray
device_synchronize(::MetalGPU) = Metal.synchronize()

end # module
