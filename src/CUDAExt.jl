module CUDAExt

using CUDA
import GPE2D.Architectures: array_type, device_synchronize, CUDAGPU

array_type(::CUDAGPU) = CUDA.CuArray
device_synchronize(::CUDAGPU) = CUDA.synchronize()

end # module
