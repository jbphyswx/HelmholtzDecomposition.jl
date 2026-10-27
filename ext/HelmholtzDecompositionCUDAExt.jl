# CUFFT, part of CUDA.jl, implements AbstractFFTs for `CuArray`s of floating-point numbers, so the Cartesian
# FFT solvers run on them.
module HelmholtzDecompositionCUDAExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using CUDA: CUDA

HD._fft_available(::Type{<:CUDA.CuArray{<:Union{Float32,Float64}}}) = true

end # module
