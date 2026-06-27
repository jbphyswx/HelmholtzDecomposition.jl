"""
    HelmholtzDecompositionDistributedExt — multiprocess batch decomposition.

Parallelizes [`helmholtz_decompose_batch`](@ref) over the batch of fields across worker
processes using `Distributed.pmap`, when called with a `DistributedBackend`. The package
must be loaded on the workers (`@everywhere using HelmholtzDecomposition`). Results are
returned to the caller in input order.
"""
module HelmholtzDecompositionDistributedExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using Distributed: Distributed

function HD._decompose_batch(b::HD.DistributedBackend, grid::HD.StructuredGrid, fields; kwargs...)
    inner = HD.local_backend(b)
    return Distributed.pmap(f -> HD.helmholtz_decompose(f, grid; backend = inner, kwargs...), fields)
end

end # module
