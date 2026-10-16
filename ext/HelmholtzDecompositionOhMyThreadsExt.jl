# Multithreaded batch decomposition: `helmholtz_decompose_batch` over the batch, whose fields are
# independent and each write only their own result. Every task shares one `HelmholtzPlan`, read-only
# during a solve, and takes one `HelmholtzWorkspace` through `@local`, which its fields reuse. The
# decomposition inside a field runs serially, FastTransforms on one OpenMP thread included
# (`FlowTransformBindings.FASTTRANSFORMS_THREADS`): the batch axis claims the threads.
module HelmholtzDecompositionOhMyThreadsExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends
using FlowTransformBindings: FlowTransformBindings as FTB
using OhMyThreads: OhMyThreads

function HD._decompose_batch!(
    ::ComputationalBackends.AbstractThreadedBackend, batch::HD.HelmholtzBatch, fields,
    plan::HD.HelmholtzPlan; kwargs...,
)
    items = collect(fields)
    Base.ScopedValues.with(FTB.FASTTRANSFORMS_THREADS => 1) do
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
