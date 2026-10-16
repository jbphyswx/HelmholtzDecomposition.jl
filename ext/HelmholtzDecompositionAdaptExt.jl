# `Adapt` rules for a plan's own structs. A plan moves to a device through
# `FlowGeometries.Execution.on_backend`, which is `Adapt.adapt`; FlowGeometries' rules move its face
# metrics and Laplacian coefficients, and this one moves the multigrid hierarchy built from them.
module HelmholtzDecompositionAdaptExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using Adapt: Adapt

# The hierarchy is read during every cycle, so its levels move with the plan. The per-level vectors
# belong to the task and come from `multigrid_buffers`.
Adapt.adapt_structure(to, mg::HD.MultigridPreconditioner) =
    HD.MultigridPreconditioner(map(c -> Adapt.adapt(to, c), mg.levels), mg.ω, mg.ν)

end # module
