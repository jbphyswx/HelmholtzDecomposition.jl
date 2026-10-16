# Multiprocess batch decomposition: `helmholtz_decompose_batch` across worker processes, results in
# input order. The package must be loaded on the workers (`@everywhere using HelmholtzDecomposition`).
# Each worker takes one contiguous chunk of the batch, receives the plan once, and reuses one
# workspace down its chunk.
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
    plan::HD.HelmholtzPlan; kwargs...,
)
    items = collect(fields)
    isempty(items) && return batch
    ranges = _chunks(length(items), max(1, Distributed.nworkers()))
    # A worker cannot write into the caller's arrays, so each returns its chunk as its own batch,
    # one message per worker, and the slices are copied in here.
    parts = Distributed.pmap(ranges) do rng
        # Serial within a worker, FastTransforms on one OpenMP thread included: each worker process
        # gets one chunk of the batch, and the processes carry the parallelism.
        Base.ScopedValues.with(FTB.FASTTRANSFORMS_THREADS => 1) do
            local_batch = HD.allocate_batch(plan, length(rng))
            ws = HD.allocate_workspace(plan)
            for (k, i) in enumerate(rng)
                HD._decompose_slice!(local_batch, k, items[i], plan, ws;
                                     backend = ComputationalBackends.SerialBackend(), kwargs...)
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
