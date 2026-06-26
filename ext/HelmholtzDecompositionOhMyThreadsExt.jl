"""
    HelmholtzDecompositionOhMyThreadsExt — multithreaded batch decomposition.

Parallelizes [`helmholtz_decompose_batch`](@ref) over the batch of fields using
OhMyThreads, when called with a `ThreadedBackend`. Each field is decomposed independently
on a serial local backend.
"""
module HelmholtzDecompositionOhMyThreadsExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using OhMyThreads: OhMyThreads

function HD._decompose_batch(::HD.ThreadedBackend, grid::HD.StructuredGrid, fields; kwargs...)
    return OhMyThreads.tmap(f -> HD.helmholtz_decompose(f, grid; backend = HD.SerialBackend(), kwargs...), fields)
end

end # module
