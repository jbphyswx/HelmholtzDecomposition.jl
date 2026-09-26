"""
    HelmholtzDecompositionAdaptExt — `Adapt` rules for a plan's structs.

Every hot loop in this package is written once against `FlowGeometries.Execution.run_indices`, and
FlowGeometries' KernelAbstractions extension turns that into a kernel launch under a `GPUBackend`.
Buffers come from `FlowGeometries.Execution.allocate`, and a plan's coefficients, face metrics and
multigrid hierarchy move to the device through `FlowGeometries.Execution.on_backend`, which is
`Adapt.adapt`. These rules say how each of those structs moves.
"""
module HelmholtzDecompositionAdaptExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using Adapt: Adapt

# A kernel launch adapts what the body captures, and `Adapt` recurses into a tuple and into
# `FlowGeometries`' own wrappers; a struct with no rule passes through whole. `LazyDiagonal` is
# captured by the smoother and by the Jacobi preconditioner.
Adapt.adapt_structure(to, d::HD.LazyDiagonal{N,T}) where {N,T} =
    HD.LazyDiagonal{N,T}(Adapt.adapt(to, d.coef), Adapt.adapt(to, d.measure), Adapt.adapt(to, d.grid))

# `LazyDiagonal` reads the coefficients and the measure, so it is rebuilt around the ones that just
# moved and the device holds one copy of each.
_adapt_diagonal(to, d::HD.LazyDiagonal{N,T}, coef, meas) where {N,T} =
    HD.LazyDiagonal{N,T}(coef, meas, Adapt.adapt(to, d.grid))
_adapt_diagonal(to, d, _, _) = Adapt.adapt(to, d)

function Adapt.adapt_structure(to, c::HD.LaplacianCoefficients{N,T}) where {N,T}
    coef = map(a -> Adapt.adapt(to, a), c.coef)
    meas = Adapt.adapt(to, c.measure)
    diag = _adapt_diagonal(to, c.diag, coef, meas)
    # A kernel reads the component cells; the offsets and measures are tuples bounding the host
    # loop over components.
    ns = c.nullspace
    nsd = HD.ComponentNullspace(ns.full, Adapt.adapt(to, ns.perm), ns.offsets, ns.totals)
    return HD.LaplacianCoefficients{N,T,typeof(coef),typeof(diag),typeof(meas),typeof(nsd)}(
        coef, diag, meas, c.total, c.singular, nsd)
end

# The hierarchy is read during every cycle, so its grids and coefficients move with the plan. The
# per-level vectors are not here: they belong to the task and come from `multigrid_buffers`.
Adapt.adapt_structure(to, lev::HD.MultigridLevel) =
    HD.MultigridLevel(Adapt.adapt(to, lev.grid), Adapt.adapt(to, lev.coefficients))

Adapt.adapt_structure(to, mg::HD.MultigridPreconditioner) =
    HD.MultigridPreconditioner(map(lev -> Adapt.adapt(to, lev), mg.levels), mg.ω, mg.ν)

function Adapt.adapt_structure(to, m::HD.FaceMetrics{N,T}) where {N,T}
    area = map(a -> Adapt.adapt(to, a), m.area)
    gap = map(a -> Adapt.adapt(to, a), m.gap)
    return HD.FaceMetrics{N,T,typeof(area)}(area, gap)
end

end # module
