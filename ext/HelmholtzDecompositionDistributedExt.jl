# Multiprocess batch decomposition: `helmholtz_decompose_batch` across worker processes, results in
# input order. The package must be loaded on the workers (`@everywhere using HelmholtzDecomposition`).
# Each worker receives the plan and its own contiguous chunk of the batch, and works through the chunk
# with one workspace on the backend's inner backend.
module HelmholtzDecompositionDistributedExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends
using Distributed: Distributed
using FlowTransformBindings: FlowTransformBindings as FTB

# Contiguous, near-equal blocks, so each chunk touches one span of the batch.
function _chunks(n::Int, k::Int)
    k = max(1, min(k, max(n, 1)))
    base, rem = divrem(n, k)
    out = Vector{UnitRange{Int}}(undef, k)
    lo = 1
    for c in 1:k
        len = base + (c <= rem ? 1 : 0)
        out[c] = lo:(lo + len - 1)
        lo += len
    end
    return out
end

function HD._decompose_batch!(
    b::ComputationalBackends.AbstractDistributedBackend, batch::HD.HelmholtzBatch, fields,
    plan::HD.HelmholtzPlan,
)
    items = collect(fields)
    isempty(items) && return batch
    exec = HD._field_execution(plan, ComputationalBackends.local_backend(b))
    # FastTransforms on one OpenMP thread where each field runs serially, the processes carrying the
    # parallelism; at its own count under a threaded inner backend.
    ft = exec isa ComputationalBackends.AbstractSerialBackend ? 1 : nothing
    ranges = _chunks(length(items), max(1, Distributed.nworkers()))
    # A worker cannot write into the caller's arrays, so each returns its chunk as its own batch, one
    # message per worker, and the slices are copied in here.
    parts = Distributed.pmap([items[rng] for rng in ranges]) do chunk
        Base.ScopedValues.with(FTB.FASTTRANSFORMS_THREADS => ft) do
            local_batch = HD.allocate_batch(plan, length(chunk))
            ws = HD._allocate_workspace(plan, exec)
            try
                for (k, field) in enumerate(chunk)
                    HD._decompose_slice!(local_batch, k, field, plan, ws, exec)
                end
            finally
                HD.close!(ws)
            end
            local_batch
        end
    end
    for (rng, part) in zip(ranges, parts)
        for (k, i) in enumerate(rng)
            HD.copy_slice!(batch, i, part, k)
        end
    end
    return batch
end

end # module
