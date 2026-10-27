# A device array's package implements AbstractFFTs for it (CUFFT for a `CuArray`), so the Cartesian FFT
# solvers run on the device a `GPUBackend` names.
module HelmholtzDecompositionKernelAbstractionsAbstractFFTsExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using KernelAbstractions: KernelAbstractions
using AbstractFFTs: AbstractFFTs

HD._available(::HD._DeviceFFT) = true

end # module
