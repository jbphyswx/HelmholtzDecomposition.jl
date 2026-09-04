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

# A kernel launch adapts what the body captures, and `Adapt` recurses into a tuple and into
# `FlowGeometries`' own wrappers; a struct it has no rule for it passes through whole. `LazyDiagonal`
# is captured by the smoother and by the Jacobi preconditioner, so it declares how it converts.
Adapt.adapt_structure(to, d::HD.LazyDiagonal{N,T}) where {N,T} =
    HD.LazyDiagonal{N,T}(Adapt.adapt(to, d.coef), Adapt.adapt(to, d.measure),
                         Adapt.adapt(to, d.grid))

# `LazyDiagonal` reads the coefficients and the measure, so it is rebuilt around the ones that just
# moved and the device holds one copy of each.
_adapt_diagonal(b, d::HD.LazyDiagonal{N,T}, coef, meas) where {N,T} =
    HD.LazyDiagonal{N,T}(coef, meas, Adapt.adapt(b, d.grid))
_adapt_diagonal(b, d, _, _) = Adapt.adapt(b, d)

function HD.to_backend(b::KA.Backend, c::HD.LaplacianCoefficients{N,T}) where {N,T}
    coef = map(a -> Adapt.adapt(b, a), c.coef)
    meas = Adapt.adapt(b, c.measure)
    diag = _adapt_diagonal(b, c.diag, coef, meas)
    # `singular` rides along unchanged: it is a property of the operator, already decided.
    return HD.LaplacianCoefficients{N,T,typeof(coef),typeof(diag),typeof(meas)}(
        coef, diag, meas, c.total, c.singular)
end

# The hierarchy is read during every cycle, so its grids and coefficients move with the plan. The
# per-level vectors are not here: they belong to the task and come from `multigrid_buffers`.
function HD.to_backend(b::KA.Backend, mg::HD.MultigridPreconditioner)
    levels = map(mg.levels) do lev
        HD.MultigridLevel(HD.to_backend(b, lev.grid), HD.to_backend(b, lev.coefficients))
    end
    return HD.MultigridPreconditioner(levels, mg.ω, mg.ν)
end

function HD.to_backend(b::KA.Backend, m::HD.FaceMetrics{N,T}) where {N,T}
    area = map(a -> Adapt.adapt(b, a), m.area)
    gap = map(a -> Adapt.adapt(b, a), m.gap)
    return HD.FaceMetrics{N,T,typeof(area)}(area, gap)
end

end # module
