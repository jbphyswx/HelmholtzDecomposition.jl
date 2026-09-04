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
    # The whole parallel section sits inside the pin: a host transform library's thread count is
    # process-global, so it is set once here for every task. See `HD.with_serial_transforms`.
    HD.with_serial_transforms(plan.solver) do
        OhMyThreads.@tasks for i in eachindex(items)
            # `@local` gives one workspace per task, and the fields of that task share it.
            @local ws = HD.allocate_workspace(plan)
            # The batch axis claims the threads, so the operator and solver loops within one field
            # run serially. Slices are disjoint views, so the writes do not race.
            HD._decompose_slice!(batch, i, items[i], plan, ws;
                                 backend = ComputationalBackends.SerialBackend(), kwargs...)
        end
    end
    return batch
end

end # module
