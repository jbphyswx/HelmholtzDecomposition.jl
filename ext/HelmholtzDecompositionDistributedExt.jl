# Multiprocess batch decomposition: `helmholtz_decompose_batch` across worker processes, results in
# input order. The package must be loaded on the workers (`@everywhere using HelmholtzDecomposition`).
# A plan built under a `DistributedBackend` is built and held by every worker, and a batch then sends each
# worker its contiguous chunk of fields alone; a batch over a caller's plan sends the plan with each chunk.
# Each worker works through its chunk with one workspace on the backend's inner backend.
module HelmholtzDecompositionDistributedExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries as FG
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

# One chunk of fields through `plan`, each field's loops on `exec`, into a batch of the chunk's own.
function _decompose_chunk(plan::HD.HelmholtzPlan, chunk, exec)
    # FastTransforms on one OpenMP thread where the worker runs each field on one thread, the processes
    # carrying the parallelism; at its own count where the worker's fields run threaded.
    ft = HD._library_threads(exec) == 1 ? 1 : nothing
    return Base.ScopedValues.with(FTB.FASTTRANSFORMS_THREADS => ft) do
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

# A worker cannot write into the caller's arrays, so each returns its chunk as its own batch, one message
# per worker, and the slices are copied in here.
function _copy_parts!(batch::HD.HelmholtzBatch, ranges, parts)
    for (rng, part) in zip(ranges, parts), (k, i) in enumerate(rng)
        HD.copy_slice!(batch, i, part, k)
    end
    return batch
end

function HD._decompose_batch!(
    b::ComputationalBackends.AbstractDistributedBackend, batch::HD.HelmholtzBatch, fields,
    plan::HD.HelmholtzPlan,
)
    items = collect(fields)
    isempty(items) && return batch
    exec = HD._field_execution(plan, ComputationalBackends.local_backend(b))
    ranges = _chunks(length(items), max(1, Distributed.nworkers()))
    parts = Distributed.pmap(chunk -> _decompose_chunk(plan, chunk, exec), [items[rng] for rng in ranges])
    return _copy_parts!(batch, ranges, parts)
end

"""
    DistributedHelmholtzPlan

A [`HelmholtzPlan`](@ref HelmholtzDecomposition.HelmholtzPlan) built and held by each Distributed worker,
from `plan_helmholtz(grid; execution = DistributedBackend(inner))`: `plans[k]` is a `Future` whose value,
worker `workers[k]`'s plan built under `inner`, stays on that worker.
"""
struct DistributedHelmholtzPlan{F}
    workers::Vector{Int}
    plans::Vector{F}
end

Base.show(io::IO, p::DistributedHelmholtzPlan) =
    print(io, "DistributedHelmholtzPlan(", length(p.workers), " workers)")

function HD._plan_helmholtz(grid::FG.Grids.FaceIndexedGrid, boundary::FG.Operators.AbstractBoundaryCondition,
                            solver::HD.AbstractPoissonSolver, b::ComputationalBackends.AbstractDistributedBackend)
    inner = ComputationalBackends.local_backend(b)
    workers = Distributed.workers()
    plans = [Distributed.remotecall(() -> HD.plan_helmholtz(grid; boundary, solver, execution = inner), w)
             for w in workers]
    # A build that failed on a worker raises here; a successful one returns nothing and its plan stays.
    foreach((w, f) -> Distributed.remotecall_fetch(f -> (fetch(f); nothing), w, f), workers, plans)
    return DistributedHelmholtzPlan(workers, plans)
end

# On the worker holding `f`: its plan, local there, and the chunk through it on the plan's own backend.
function _decompose_held(f, chunk)
    plan = fetch(f)
    return _decompose_chunk(plan, chunk, HD._field_execution(plan, plan.execution))
end

function _held_parts(p::DistributedHelmholtzPlan, items)
    ranges = _chunks(length(items), length(p.workers))
    parts = asyncmap(eachindex(ranges)) do k
        Distributed.remotecall_fetch(_decompose_held, p.workers[k], p.plans[k], items[ranges[k]])
    end
    return ranges, parts
end

function HD.helmholtz_decompose_batch(p::DistributedHelmholtzPlan, fields)
    items = collect(fields)
    ranges, parts = _held_parts(p, items)
    return _copy_parts!(HD._batch_like(first(parts), length(items)), ranges, parts)
end

function HD.helmholtz_decompose_batch!(batch::HD.HelmholtzBatch, fields, p::DistributedHelmholtzPlan)
    length(batch) == length(fields) || throw(DimensionMismatch(
        "batch holds $(length(batch)) slots but $(length(fields)) fields were given"))
    ranges, parts = _held_parts(p, collect(fields))
    return _copy_parts!(batch, ranges, parts)
end

end # module
