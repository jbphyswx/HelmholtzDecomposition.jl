"""
    HelmholtzDecompositionOhMyThreadsExt — multithreaded batch decomposition.

Parallelizes [`helmholtz_decompose_batch`](@ref) over the batch, which is the axis that is
race-free by construction: the fields are independent and each writes only its own result.

One [`HelmholtzPlan`](@ref) is shared by every task — its Laplacian coefficients are identical
for all of them and are the expensive part, so rebuilding per task would defeat the batch. Each
task takes its own [`HelmholtzWorkspace`](@ref) through `@local`, allocated once per task rather
than once per field, since those buffers are written through and sharing them would race.

The inner decomposition is left serial: threading it as well would oversubscribe, with an outer
loop and an inner one each claiming every thread.
"""
module HelmholtzDecompositionOhMyThreadsExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends
using OhMyThreads: OhMyThreads

function HD._decompose_batch!(
    ::ComputationalBackends.AbstractThreadedBackend, batch::HD.HelmholtzBatch, fields,
    plan::HD.HelmholtzPlan; kwargs...,
)
    items = collect(fields)
    OhMyThreads.@tasks for i in eachindex(items)
        # Per task, not per field: `length(items)` workspaces would allocate the batch away.
        @local ws = HD.allocate_workspace(plan)
        # Serial inner backend, named rather than assumed: the plan's own backend drives the
        # operator and solver loops, so leaving it in place would have the outer and inner loops
        # each claim every thread. Slices are disjoint views, so the writes do not race.
        HD._decompose_slice!(batch, i, items[i], plan, ws;
                             backend = ComputationalBackends.SerialBackend(), kwargs...)
    end
    return batch
end

end # module
