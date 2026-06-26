"""
    HelmholtzDecompositionMPIExt — distributed batch decomposition over MPI ranks.

Parallelizes [`helmholtz_decompose_batch`](@ref) across MPI ranks when called with an
`MPIBackend`. The batch is split into contiguous per-rank blocks; each rank decomposes its
block on its inner local backend, and the rotational/divergent/harmonic velocity
components are `Allgatherv`-ed so every rank ends with the full set in input order.

Returns a `Vector{NamedTuple}` `(; u_rot, u_div, u_harm)` — the primary scientific outputs.
(Potentials and per-solve diagnostics are not gathered to avoid serializing whole structs;
recompute them rank-locally if needed.) Call `MPI.Init()` before use. For a single field,
MPI offers no parallelism; use `helmholtz_decompose` directly.
"""
module HelmholtzDecompositionMPIExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using MPI: MPI

# Contiguous block [start, stop] (1-based, inclusive) of `n` items for `rank` of `nranks`.
function _block(n::Int, nranks::Int, rank::Int)
    base, rem = divrem(n, nranks)
    len = base + (rank < rem ? 1 : 0)
    start = rank * base + min(rank, rem) + 1
    return start:(start + len - 1)
end

function HD._decompose_batch(b::HD.MPIBackend, grid::HD.StructuredGrid, fields; kwargs...)
    MPI.Initialized() || throw(ArgumentError("MPI is not initialized — call `MPI.Init()` before use."))
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    inner = HD.local_backend(b)

    nf = length(fields)
    nf == 0 && return NamedTuple[]
    fields_vec = collect(fields)
    T = real(eltype(fields_vec[1]))
    fieldlen = length(fields_vec[1])          # prod(dims) * N
    per_field = 3 * fieldlen                  # u_rot, u_div, u_harm stacked

    myblock = _block(nf, nranks, rank)
    sendbuf = Vector{T}(undef, per_field * length(myblock))
    off = 0
    for i in myblock
        res = HD.helmholtz_decompose(fields_vec[i], grid; backend = inner, kwargs...)
        copyto!(sendbuf, off + 1, vec(res.u_rot), 1, fieldlen); off += fieldlen
        copyto!(sendbuf, off + 1, vec(res.u_div), 1, fieldlen); off += fieldlen
        copyto!(sendbuf, off + 1, vec(res.u_harm), 1, fieldlen); off += fieldlen
    end

    counts = Cint[per_field * length(_block(nf, nranks, r)) for r in 0:(nranks - 1)]
    recv = Vector{T}(undef, per_field * nf)
    MPI.Allgatherv!(sendbuf, MPI.VBuffer(recv, counts), comm)

    shape = (HD.size_tuple(grid)..., HD.ndims(grid))
    out = Vector{NamedTuple}(undef, nf)
    p = 0
    for i in 1:nf
        u_rot = reshape(recv[(p + 1):(p + fieldlen)], shape); p += fieldlen
        u_div = reshape(recv[(p + 1):(p + fieldlen)], shape); p += fieldlen
        u_harm = reshape(recv[(p + 1):(p + fieldlen)], shape); p += fieldlen
        out[i] = (; u_rot, u_div, u_harm)
    end
    return out
end

end # module
