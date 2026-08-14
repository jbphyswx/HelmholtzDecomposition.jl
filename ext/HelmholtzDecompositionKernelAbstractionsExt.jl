"""
    HelmholtzDecompositionKernelAbstractionsExt — the device path.

Every hot loop in this package is written once against `FlowGeometries.Execution.run_indices`, and
FlowGeometries' own KernelAbstractions extension turns that into a kernel launch. So the operators,
the solver and the multigrid smoother need nothing added here: they already run wherever the arrays
do. What is missing without this extension is the *memory* — buffers would come back as host arrays
a kernel cannot reach, and the plan's coefficients would stay on the host.

That is what this supplies: the two allocation hooks, `allocate_zeros` and `to_backend`. There is
deliberately no vendor-specific code and no per-vendor extension, because there is nothing for one
to do — a device array's own package provides the `Adapt` rules and the `AbstractFFTs` methods, and
a CUDA-specific FFT path would duplicate the generic one rather than add to it.
"""
module HelmholtzDecompositionKernelAbstractionsExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using KernelAbstractions: KernelAbstractions as KA
using Adapt: Adapt

HD.allocate_zeros(backend::KA.Backend, ::Type{T}, dims::Dims) where {T} =
    KA.zeros(backend, T, dims)

# `Adapt` is how an array's own package says what "on this device" means for it, so structs move by
# recursing to their fields rather than by anything vendor-specific here. `KA.CPU()` adapts to host
# arrays, which is what makes the device path checkable without a device.
HD.to_backend(backend::KA.Backend, x) = Adapt.adapt(backend, x)

function HD.to_backend(b::KA.Backend, c::HD.LaplacianCoefficients{N,T}) where {N,T}
    coef = map(a -> Adapt.adapt(b, a), c.coef)
    diag = Adapt.adapt(b, c.diag)
    meas = Adapt.adapt(b, c.measure)
    # `singular` rides along unchanged: it is a property of the operator, already decided.
    return HD.LaplacianCoefficients{N,T,typeof(coef),typeof(diag)}(coef, diag, meas, c.singular)
end

function HD.to_backend(b::KA.Backend, m::HD.FaceMetrics{N,T}) where {N,T}
    area = map(a -> Adapt.adapt(b, a), m.area)
    gap = map(a -> Adapt.adapt(b, a), m.gap)
    return HD.FaceMetrics{N,T,typeof(area)}(area, gap)
end

end # module
