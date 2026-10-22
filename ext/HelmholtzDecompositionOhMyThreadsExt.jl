# Multithreaded batch decomposition: `helmholtz_decompose_batch` over the batch, whose fields are
# independent and each write only their own result. Every task shares one `HelmholtzPlan`, read-only
# during a solve, and works through one contiguous run of fields with one `HelmholtzWorkspace`, released
# when the run ends. On a host plan the decomposition inside a field runs serially, FastTransforms on one
# OpenMP thread included (`FlowTransformBindings.FASTTRANSFORMS_THREADS`): the batch axis claims the
# threads.
module HelmholtzDecompositionOhMyThreadsExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries
using FlowTransformBindings: FlowTransformBindings as FTB
using OhMyThreads: OhMyThreads

function HD._decompose_batch!(
    ::ComputationalBackends.AbstractThreadedBackend, batch::HD.HelmholtzBatch, fields,
    plan::HD.HelmholtzPlan,
)
    items = collect(fields)
    exec = HD._field_execution(plan, ComputationalBackends.SerialBackend())
    chunks = FlowGeometries.Execution.chunk_ranges(length(items), Threads.nthreads())
    Base.ScopedValues.with(FTB.FASTTRANSFORMS_THREADS => 1) do
        OhMyThreads.tforeach(chunks) do chunk
            ws = HD._allocate_workspace(plan, exec)
            try
                # Slices are disjoint views, so the writes do not race.
                for i in chunk
                    HD._decompose_slice!(batch, i, items[i], plan, ws, exec)
                end
            finally
                HD.close!(ws)
            end
        end
    end
    return batch
end

end # module
