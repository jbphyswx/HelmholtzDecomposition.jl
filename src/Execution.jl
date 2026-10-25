# Where the loops of one decomposition run. The operators are FlowGeometries': `FaceMetrics`,
# `gradient!`, `divergence!`, `curl!`, `rotational_velocity!`, `to_faces!`, `to_centres!` and the
# assembled `LaplacianCoefficients` with its multigrid hierarchy, all under `FlowGeometries.Operators`.

"""
    resolve_execution_backend(backend) -> concrete backend

The backend a single decomposition's own loops run on, resolved once at plan time and passed to
`FlowGeometries.Execution` as it is.

`AutoBackend` is [`LoopSizedThreads`](@ref) when Julia runs more than one thread and
`SerialBackend()` otherwise. A `GPUBackend` needs FlowGeometries' KernelAbstractions extension,
checked here so the error names the package to load.

`DistributedBackend` and `MPIBackend` spread *fields across processes*, which is
[`helmholtz_decompose_batch`](@ref)'s axis, and are refused for the index space inside one field.
"""
resolve_execution_backend(b::ComputationalBackends.AbstractSerialBackend) = b
resolve_execution_backend(b::ComputationalBackends.AbstractThreadedBackend) = b
resolve_execution_backend(::ComputationalBackends.AbstractAutoBackend) =
    Threads.nthreads() > 1 ? LoopSizedThreads() : ComputationalBackends.SerialBackend()

"""
    LoopSizedThreads(min_per_thread = 16384)

The loops of one decomposition under `AutoBackend`: a loop over `n` indices runs threaded when each of
Julia's threads gets at least `min_per_thread` of them, serially otherwise, so the coarse levels of a
multigrid cycle and the small reductions run on one thread.
"""
struct LoopSizedThreads <: ComputationalBackends.AbstractThreadedBackend
    min_per_thread::Int
end

LoopSizedThreads() = LoopSizedThreads(16384)

"""
    loop_backend(backend, n) -> backend

The backend a loop over `n` cells runs on: [`LoopSizedThreads`](@ref) resolved by `n`, any other
backend as it is. An operator sweeping a grid of `n` cells is handed this, so every loop inside it,
whatever it counts, runs on the one choice.
"""
@inline loop_backend(b, ::Integer) = b
@inline loop_backend(b::LoopSizedThreads, n::Integer) =
    n >= b.min_per_thread * Threads.nthreads() ? ComputationalBackends.ThreadedBackend() :
                                                 ComputationalBackends.SerialBackend()

FlowGeometries.Execution.run_indices(f::F, n::Integer, b::LoopSizedThreads,
                                     outs::FlowGeometries.Execution.Written...) where {F} =
    FlowGeometries.Execution.run_indices(f, n, loop_backend(b, n), outs...)

FlowGeometries.Execution.run_chunks(f::F, n::Integer, b::LoopSizedThreads,
                                    outs::FlowGeometries.Execution.Written...) where {F} =
    FlowGeometries.Execution.run_chunks(f, n, loop_backend(b, n), outs...)

FlowGeometries.Execution.reduce_indices(f::F, op::O, init, n::Integer, b::LoopSizedThreads) where {F,O} =
    FlowGeometries.Execution.reduce_indices(f, op, init, n, loop_backend(b, n))

function resolve_execution_backend(b::ComputationalBackends.AbstractGPUBackend)
    Base.get_extension(FlowGeometries, :FlowGeometriesKernelAbstractionsExt) === nothing &&
        throw(ArgumentError(
            "$(typeof(b)) needs FlowGeometries' KernelAbstractions extension, which is not " *
            "loaded. Run `using KernelAbstractions` first (plus the vendor package owning the " *
            "device), or pass `execution = ComputationalBackends.SerialBackend()`."))
    return b
end

resolve_execution_backend(b::ComputationalBackends.AbstractExecutionBackend) = throw(ArgumentError(
    "$(typeof(b)) parallelises over fields, not over the index space within one field. Pass it " *
    "to `helmholtz_decompose_batch` instead; `plan_helmholtz`'s `execution` selects how the loops " *
    "inside a single decomposition run."))

# The backends whose loops run on the host, over host memory.
const _HostBackend = Union{ComputationalBackends.AbstractSerialBackend,
                           ComputationalBackends.AbstractThreadedBackend}

# Every buffer this package owns: zeroed, in the memory `backend`'s loops write.
@inline _zeros(backend, ::Type{T}, dims::Dims) where {T} =
    fill!(FlowGeometries.Execution.allocate(backend, T, dims...), zero(T))

# Component `c` of a component-last `(dims..., N)` array.
@inline _component(U::AbstractArray, c::Integer, ::Val{N}) where {N} =
    @view U[ntuple(_ -> Colon(), Val(N))..., c]

# The `N` components of a component-last array, as the tuple FlowGeometries' operators take.
@inline _components(U::AbstractArray, ::Val{N}) where {N} = ntuple(c -> _component(U, c, Val(N)), Val(N))
