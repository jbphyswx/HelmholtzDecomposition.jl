"""
    HelmholtzDecompositionDistributedExt — multiprocess batch decomposition.

Parallelizes [`helmholtz_decompose_batch`](@ref) across worker processes. The package must be
loaded on the workers (`@everywhere using HelmholtzDecomposition`). Results are returned in input
order.

Work is chunked **per worker**, not per field. A `pmap` over individual fields sends one message
and serializes the plan once per field; over `nworkers()` chunks it does so once per worker, and
each worker builds a single workspace and reuses it down its chunk — which is the same reason the
threaded extension takes a workspace per task rather than per field.
"""
module HelmholtzDecompositionDistributedExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends
using Distributed: Distributed

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
    # A worker cannot write into the caller's arrays, so each returns its chunk as its own batch
    # and the slices are copied in here — one message per worker rather than one per field.
    parts = Distributed.pmap(ranges) do rng
        local_batch = HD.allocate_batch(plan, length(rng))
        ws = HD.allocate_workspace(plan)
        # Serial within a worker: the worker process gets one chunk of the batch, and the plan's
        # own backend would otherwise have each of them thread across the whole machine.
        for (k, i) in enumerate(rng)
            HD._decompose_slice!(local_batch, k, items[i], plan, ws;
                                 backend = ComputationalBackends.SerialBackend(), kwargs...)
        end
        return local_batch
    end
    for (rng, part) in zip(ranges, parts)
        for (k, i) in enumerate(rng)
            HD.copy_slice!(batch, i, part, k)
        end
    end
    return batch
end

end # module
