"""
    HelmholtzDecompositionMPIExt — batch decomposition across MPI ranks.

Splits [`helmholtz_decompose_batch`](@ref)'s batch into contiguous per-rank blocks, decomposes each
block locally, and `Allgatherv`s so every rank ends with the whole batch in input order.

A [`HelmholtzBatch`](@ref) stores the batch axis last, so a rank's block is a contiguous run of
elements in every one of its arrays and is gathered in place — no packing buffer, and no
`Vector{NamedTuple}` with an abstract element type.

Call `MPI.Init()` first.
"""
module HelmholtzDecompositionMPIExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends
using MPI: MPI

# Contiguous block (1-based, inclusive) of `n` items for `rank` of `nranks`.
function _block(n::Int, nranks::Int, rank::Int)
    base, rem = divrem(n, nranks)
    len = base + (rank < rem ? 1 : 0)
    start = rank * base + min(rank, rem) + 1
    return start:(start + len - 1)
end

# Every array carrying the batch axis, in a fixed order shared by all ranks.
_arrays(b::HD.HelmholtzBatch) =
    (b.u_rot, b.u_div, b.u_harm, b.χ, b.divergence, b.rotation_potential...)

# Gather one array whose last axis is the batch: each rank owns a contiguous run of `stride`
# elements per field.
function _gather!(A, nf::Int, nranks::Int, rank::Int, comm)
    stride = length(A) ÷ nf
    mine = _block(nf, nranks, rank)
    counts = Cint[stride * length(_block(nf, nranks, r)) for r in 0:(nranks - 1)]
    send = view(vec(A), (stride * (first(mine) - 1) + 1):(stride * last(mine)))
    MPI.Allgatherv!(collect(send), MPI.VBuffer(vec(A), counts), comm)
    return A
end

# `N`, `P` and `T` come from the plan, not the batch: parameterising `HelmholtzBatch` here would
# make this method and the generic refusal mutually ambiguous, since the batch's later parameters
# are themselves constrained by `T`.
function HD._decompose_batch!(
    b::ComputationalBackends.AbstractMPIBackend, batch::HD.HelmholtzBatch, fields,
    plan::HD.HelmholtzPlan{N,P,T}; kwargs...,
) where {N,P,T}
    MPI.Initialized() || throw(ArgumentError(
        "MPI is not initialized — call `MPI.Init()` before `helmholtz_decompose_batch`."))
    comm = b.comm === nothing ? MPI.COMM_WORLD : b.comm
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    items = collect(fields)
    nf = length(items)
    nf == 0 && return batch

    ws = HD.allocate_workspace(plan)
    for i in _block(nf, nranks, rank)
        # The rank's own inner backend, as named on the `MPIBackend`: a rank owns a block of the
        # batch, so how it works through that block is a separate choice from how ranks divide it.
        HD._decompose_slice!(batch, i, items[i], plan, ws;
                             backend = HD.execution_backend(b.inner), kwargs...)
    end

    for A in _arrays(batch)
        _gather!(A, nf, nranks, rank, comm)
    end
    _gather!(batch.harmonic_fraction, nf, nranks, rank, comm)

    # The convergence diagnostics are structs, so they ride in one scalar buffer: `converged`,
    # `iterations` and `residual` for the primal solve and each of the `P` dual ones.
    w = 3 * (1 + P)
    flat = zeros(T, w * nf)
    for i in _block(nf, nranks, rank)
        o = w * (i - 1)
        for (k, s) in enumerate((batch.χ_solve[i], batch.rot_solve[i]...))
            flat[o + 3k - 2] = T(s.converged)
            flat[o + 3k - 1] = T(s.iterations)
            flat[o + 3k] = s.residual
        end
    end
    _gather!(flat, nf, nranks, rank, comm)
    for i in 1:nf
        o = w * (i - 1)
        batch.χ_solve[i] = HD.SolverResult{T}(!iszero(flat[o + 1]), Int(flat[o + 2]), flat[o + 3])
        batch.rot_solve[i] = ntuple(Val(P)) do p
            q = o + 3p
            HD.SolverResult{T}(!iszero(flat[q + 1]), Int(flat[q + 2]), flat[q + 3])
        end
    end
    return batch
end

end # module
