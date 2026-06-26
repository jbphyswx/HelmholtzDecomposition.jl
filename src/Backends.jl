"""
    Backends.jl — Execution-backend taxonomy.

The *execution backend* (where/how arrays compute) is an axis orthogonal to the
*spectral/solver backend* (SOR / FFT / NUFFT / SHT — the math). Names mirror the jbphyswx
ecosystem:

- **Local compute** — what one process/rank computes on: `SerialBackend`, `ThreadedBackend`
  (OhMyThreads ext), `GPUBackend{B}` (CUDA / KernelAbstractions ext).
- **Distribution wrapper** — how work splits across processes, **parametric over the inner
  local backend**: `DistributedBackend{Inner}` (Distributed ext), `MPIBackend{Inner}`
  (MPI ext). The wrapper owns only partition/gather; `inner` owns the compute, so
  `MPIBackend{GPUBackend{...}}` (multi-GPU) is expressible.

`AutoBackend` resolves to the best available local backend (inferred from the array type).
Heavy implementations live in extensions; this submodule defines only dispatch types and a
few helpers.
"""
module Backends

export AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, AutoBackend
export DistributedBackend, MPIBackend, local_backend, is_distributed

abstract type AbstractExecutionBackend end

"Serial single-threaded CPU compute (always available, no extension needed)."
struct SerialBackend <: AbstractExecutionBackend end

"Multithreaded CPU compute (requires `using OhMyThreads`)."
struct ThreadedBackend <: AbstractExecutionBackend end

"""
    GPUBackend{B}

GPU compute on backend object `B` (e.g. a KernelAbstractions `CUDABackend`). Requires the
corresponding extension (`using CUDA`).
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"Resolve to the best available local backend, inferred from the array type."
struct AutoBackend <: AbstractExecutionBackend end

"""
    DistributedBackend{Inner}

Distribute work across worker processes, each running `inner` locally. Requires
`using Distributed`.
"""
struct DistributedBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
DistributedBackend() = DistributedBackend(SerialBackend())

"""
    MPIBackend{Inner}

Distribute work across MPI ranks, each running `inner` locally. Requires `using MPI`.
Not CPU-only — `MPIBackend{GPUBackend{...}}` targets multi-GPU.
"""
struct MPIBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
MPIBackend() = MPIBackend(SerialBackend())

"""
    local_backend(backend) -> AbstractExecutionBackend

The per-process compute backend — `inner` for distribution wrappers, the backend itself otherwise.
"""
local_backend(b::AbstractExecutionBackend) = b
local_backend(b::DistributedBackend) = b.inner
local_backend(b::MPIBackend) = b.inner

"`true` if `backend` distributes work across processes."
is_distributed(::AbstractExecutionBackend) = false
is_distributed(::DistributedBackend) = true
is_distributed(::MPIBackend) = true

end # module Backends
