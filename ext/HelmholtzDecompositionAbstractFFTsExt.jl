"""
    HelmholtzDecompositionAbstractFFTsExt — the Cartesian FFT solvers on any FFT library.

The periodic FFT (`CartesianSpectralSolver`) and the bounded and channel transforms
(`CartesianRealTransformSolver`) call the `AbstractFFTs` interface alone, so they run on whichever
library implements it for the array: FFTW on the host, CUFFT on a `CuArray`. A DCT-II/DST-II is an FFT
of the even/odd extension,

    even extension y = [x; reverse(x)]      ⇒  DCT-II[k] = real( rfft(y)[k+1] · e^{-iπk/2N} )
    odd  extension y = [x; −reverse(x)]     ⇒  DST-II[k] = −imag( rfft(y)[k+2] · e^{-iπ(k+1)/2N} )

and the inverse reads the first backwards: rebuild `Z[k+1] = C[k]·e^{+iπk/2N}`, pad one zero to the
`rfft` layout of a length-`2N` real signal, `brfft(Z, 2N)`, take the first `N` over `2N`.
"""
module HelmholtzDecompositionAbstractFFTsExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries: FlowGeometries as FG
using FlowTransformBindings: FlowTransformBindings as FTB
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra

const _FFTSolver = Union{HD.CartesianSpectralSolver,HD.CartesianRealTransformSolver}

# The periodic FFT expands in a periodic basis, so its grid has no boundary and every condition is
# accepted; the real transforms take a bounded direction under either condition. Both transform every
# cell and need constant spacing.
HD.supports_boundary(::HD.CartesianSpectralSolver, ::FG.Operators.AbstractBoundaryCondition) = true
HD.requires_periodic_domain(::HD.CartesianSpectralSolver) = true
HD.supports_boundary(::HD.CartesianRealTransformSolver, ::FG.Operators.Dirichlet) = true
HD.supports_boundary(::HD.CartesianRealTransformSolver, ::FG.Operators.Neumann) = true
HD.requires_full_domain(::_FFTSolver) = true
HD.requires_uniform_axes(::_FFTSolver) = true

HD.library_loaded(::_FFTSolver) = true

# Whether an FFT library transforms the arrays a backend allocates: host arrays or device arrays.
HD.supports_backend(::_FFTSolver, backend, grid) = HD._available(_fft_capability(backend))
_fft_capability(::CB.AbstractGPUBackend) = HD._DeviceFFT()
_fft_capability(::CB.AbstractExecutionBackend) = HD._HostFFT()

HD._backend_message(s::_FFTSolver, backend) =
    "$(nameof(typeof(s))) transforms through AbstractFFTs, and no FFT library implements it for the " *
    "arrays $(nameof(typeof(backend))) allocates. Load one (`using FFTW` on the host, CUDA for a " *
    "`CuArray`), or solve with `CGSolver()`."

# ---------------------------------------------------------------------------
# One bounded direction: the even or odd extension and its two plans
# ---------------------------------------------------------------------------

"""
    DirPlan

The even/odd extension buffer, its half-spectrum, the two plans and the twiddle for one bounded
direction.
"""
struct DirPlan{E,S,FP,BP,W}
    ext::E
    spec::S
    fplan::FP
    bplan::BP
    tw::W
end

function _dir_plan(::Type{T}, dims::Dims{N}, d::Int, off::Int, backend) where {T,N}
    ext = HD._zeros(backend, T, ntuple(e -> e == d ? 2dims[e] : dims[e], Val(N)))
    spec = HD._zeros(backend, complex(T), ntuple(e -> e == d ? dims[e] + 1 : dims[e], Val(N)))
    return DirPlan(ext, spec, AbstractFFTs.plan_rfft(ext, d:d; HD._plan_kwargs(ext, backend)...),
                   AbstractFFTs.plan_brfft(spec, 2dims[d], d:d; HD._plan_kwargs(spec, backend)...),
                   _twiddle(T, dims[d], off, N, d, backend))
end

# Symbol of the discrete `L` along one direction, so this inverts the operator the decomposition
# differentiates with. `πk/n` where the direction wraps, `π(k+off)/2n` where it ends.
@inline _axis_symbol(::Type{T}, k, n::Int, h::T, periodic::Bool, off::Int) where {T} =
    (s = periodic ? sin(T(π) * T(k) / T(n)) : sin(T(π) * T(k + off) / T(2n));
     -4 * s * s / (h * h))

"""
    _axis_symbols(T, dims, kdims, h, per, off, first_periodic, backend) -> NTuple{N,AbstractArray}

The symbol of `L` as one vector per direction, each reshaped to broadcast along its own axis.

`λ[I] = Σ_d λ_d(I[d])`, so `∑ n_d` numbers carry what an `N`-D array held, and the sum is formed at
the one divide that reads it. Filled by broadcast over an index range, so the vectors land wherever
the backend allocates.
"""
function _axis_symbols(::Type{T}, dims::NTuple{N,Int}, kdims::NTuple{N,Int}, h, per,
                       off::Int, first_periodic::Int, backend) where {T,N}
    return ntuple(Val(N)) do d
        n, nk = dims[d], kdims[d]
        v = HD._zeros(backend, T, (nk,))
        i = reshape(1:nk, nk)
        if !per[d] || d == first_periodic
            # A bounded direction indexes its own modes; the halved periodic one runs `0 : n÷2`.
            v .= _axis_symbol.(T, i .- 1, n, h[d], per[d], off)
        else
            # Full FFT order: the upper half stands for negative frequencies of equal magnitude.
            v .= _axis_symbol.(T, ifelse.(i .- 1 .< cld(n, 2), i .- 1, n .- i .+ 1), n, h[d],
                               true, off)
        end
        reshape(v, ntuple(e -> e == d ? nk : 1, Val(N)))
    end
end

# ---------------------------------------------------------------------------
# The solve states
# ---------------------------------------------------------------------------

"""
    RealBoundedState

Every direction bounded: the whole pipeline is real, and only the length-`2n` half-spectrum in
between is complex.
"""
struct RealBoundedState{T,N,D,L,A}
    dirs::D
    λ::L
    work::A
end

"""
    MixedTopologyState

At least one direction wraps. The bounded directions `bdims` transform first, in real arithmetic; the
periodic ones then go through one `rfft`, which halves the first of them. With every direction wrapping
this is the periodic FFT.
"""
struct MixedTopologyState{T,N,D,BD<:Tuple{Vararg{Int}},P,IP,L,A,S}
    dirs::D            # one per bounded direction, paired with `bdims`
    bdims::BD
    pplan::P
    iplan::IP
    λ::L
    work::A
    spec::S
    norm::T
end

# Plan-owning, so both get an explicit one-line `show`: the default prints transform internals.
Base.show(io::IO, ::RealBoundedState{T,N}) where {T,N} =
    print(io, "RealBoundedState{", T, ",", N, "}(…)")
Base.show(io::IO, ::MixedTopologyState{T,N}) where {T,N} =
    print(io, "MixedTopologyState{", T, ",", N, "}(…)")

function HD.prepare_solver(::HD.CartesianRealTransformSolver,
                           grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
                           boundary::FG.Operators.AbstractBoundaryCondition;
                           execution = CB.SerialBackend(), shared = nothing) where {T,N}
    dims = size(grid)
    off = boundary isa FG.Operators.Dirichlet ? 1 : 0
    per = ntuple(d -> FG.Grids.isperiodic(grid, d), Val(N))
    h = ntuple(d -> abs(FG.Grids.spacing(grid, d)), Val(N))
    work = HD._zeros(execution, T, dims)
    topo = FG.Grids.topology(grid)
    return _transform_state(_periodic_dims(topo, 1), _bounded_dims(topo, 1), T, dims, off, per, h, work,
                            execution)
end

# The directions that wrap and those that end, in order, as tuples the topology's types fix.
_periodic_dims(::Tuple{}, ::Int) = ()
_periodic_dims(t::Tuple{FG.Grids.Periodic,Vararg}, d::Int) = (d, _periodic_dims(Base.tail(t), d + 1)...)
_periodic_dims(t::Tuple, d::Int) = _periodic_dims(Base.tail(t), d + 1)
_bounded_dims(::Tuple{}, ::Int) = ()
_bounded_dims(t::Tuple{FG.Grids.Periodic,Vararg}, d::Int) = _bounded_dims(Base.tail(t), d + 1)
_bounded_dims(t::Tuple, d::Int) = (d, _bounded_dims(Base.tail(t), d + 1)...)

function _transform_state(::Tuple{}, bdims, ::Type{T}, dims::NTuple{N,Int}, off::Int, per, h, work,
                          execution) where {T,N}
    dirs = ntuple(d -> _dir_plan(T, dims, d, off, execution), Val(N))
    λ = _axis_symbols(T, dims, dims, h, per, off, 0, execution)
    return RealBoundedState{T,N,typeof(dirs),typeof(λ),typeof(work)}(dirs, λ, work)
end

function _transform_state(pdims::Tuple, bdims::Tuple, ::Type{T}, dims::NTuple{N,Int}, off::Int, per, h, work,
                          execution) where {T,N}
    dirs = map(d -> _dir_plan(T, dims, d, off, execution), bdims)
    spec = HD._zeros(execution, complex(T), ntuple(e -> e == pdims[1] ? dims[e] ÷ 2 + 1 : dims[e], Val(N)))
    pplan = AbstractFFTs.plan_rfft(work, pdims; HD._plan_kwargs(work, execution)...)
    iplan = AbstractFFTs.plan_brfft(spec, dims[pdims[1]], pdims; HD._plan_kwargs(spec, execution)...)
    λ = _axis_symbols(T, dims, size(spec), h, per, off, pdims[1], execution)
    nrm = T(prod(d -> dims[d], pdims))
    return MixedTopologyState{T,N,typeof(dirs),typeof(bdims),typeof(pplan),typeof(iplan),typeof(λ),
                              typeof(work),typeof(spec)}(dirs, bdims, pplan, iplan, λ, work, spec, nrm)
end

# The periodic FFT is the real transforms' path on a grid every direction of which wraps.
HD.prepare_solver(::HD.CartesianSpectralSolver,
                  grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
                  boundary::FG.Operators.AbstractBoundaryCondition;
                  execution = CB.SerialBackend(), shared = nothing) where {T,N} =
    HD.prepare_solver(HD.CartesianRealTransformSolver(), grid, boundary; execution = execution)

# x -> [x; ±reverse(x)] along `d`, into the preallocated buffer, reading `X` backwards through a
# view.
function _extend!(ext, X, d::Int, odd::Bool)
    n = size(X, d)
    copyto!(selectdim(ext, d, 1:n), X)
    tail = selectdim(ext, d, (n + 1):(2n))
    rev = selectdim(X, d, n:-1:1)
    odd ? (tail .= .-rev) : (tail .= rev)
    return ext
end

function _forward_real!(X, dp::DirPlan, d::Int, odd::Bool)
    n = size(X, d)
    _extend!(dp.ext, X, d, odd)
    LinearAlgebra.mul!(dp.spec, dp.fplan, dp.ext)
    if odd
        X .= .-imag.(selectdim(dp.spec, d, 2:(n + 1)) .* dp.tw)
    else
        X .= real.(selectdim(dp.spec, d, 1:n) .* dp.tw)
    end
    return X
end

function _inverse_real!(X::AbstractArray{T}, dp::DirPlan, d::Int, odd::Bool) where {T}
    n = size(X, d)
    Z = dp.spec
    fill!(Z, zero(eltype(Z)))
    if odd
        selectdim(Z, d, 2:(n + 1)) .= complex.(.-X) .* conj.(dp.tw) .* im
    else
        selectdim(Z, d, 1:n) .= complex.(X) .* conj.(dp.tw)
    end
    LinearAlgebra.mul!(dp.ext, dp.bplan, Z)
    X .= selectdim(dp.ext, d, 1:n) ./ T(2n)
    return X
end

function _solve!(Φ::AbstractArray{T,N}, RHS, st::RealBoundedState, boundary) where {T,N}
    odd = boundary isa FG.Operators.Dirichlet
    copyto!(st.work, RHS)
    for d in 1:N
        _forward_real!(st.work, st.dirs[d], d, odd)
    end
    λ = HD.lazy_axis_sum(st.λ)
    st.work .= ifelse.(iszero.(λ), zero(T), st.work ./ λ)
    for d in 1:N
        _inverse_real!(st.work, st.dirs[d], d, odd)
    end
    copyto!(Φ, st.work)
    return HD.SolverResult{T}(true, 1, zero(T))
end

# Bounded directions first, in real arithmetic; the periodic ones then take a single `rfft`, which
# halves the first of them. `RHS` and `Φ` pass through `work`, the array the plans were built for, since
# a plan applies to arrays of its own memory alignment and a batch hands views of its slices.
function _solve!(Φ::AbstractArray{T,N}, RHS, st::MixedTopologyState, boundary) where {T,N}
    odd = boundary isa FG.Operators.Dirichlet
    copyto!(st.work, RHS)
    for (i, d) in enumerate(st.bdims)
        _forward_real!(st.work, st.dirs[i], d, odd)
    end
    LinearAlgebra.mul!(st.spec, st.pplan, st.work)
    λ = HD.lazy_axis_sum(st.λ)
    st.spec .= ifelse.(iszero.(λ), zero(complex(T)), st.spec ./ λ)
    LinearAlgebra.mul!(st.work, st.iplan, st.spec)
    st.work ./= st.norm
    for (i, d) in enumerate(st.bdims)
        _inverse_real!(st.work, st.dirs[i], d, odd)
    end
    copyto!(Φ, st.work)
    return HD.SolverResult{T}(true, 1, zero(T))
end

function _twiddle(::Type{T}, n::Int, off::Int, nd::Int, d::Int, backend) where {T}
    w = HD._zeros(backend, complex(T), (n,))
    k = reshape(0:(n - 1), n)
    w .= cis.(.-T(π) .* T.(k .+ off) ./ T(2n))
    return reshape(w, ntuple(e -> e == d ? n : 1, nd))
end

"""
    solve_poisson!(Φ, RHS, grid, solver; boundary, state)

`state` comes from [`prepare_solver`](@ref) and selects the implementation through the positional
`_solve!`.
"""
function HD.solve_poisson!(
    Φ::AbstractArray{T,N},
    RHS::AbstractArray{T,N},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    solver::_FFTSolver;
    boundary::FG.Operators.AbstractBoundaryCondition = FG.Operators.Neumann(),
    state = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat,N}
    HD._require_boundary(solver, boundary)
    return _solve!(Φ, RHS, state, boundary)
end

# ---------------------------------------------------------------------------
# The spectral decomposition on a periodic grid
# ---------------------------------------------------------------------------

"""
    PeriodicSpectralState

What the periodic FFT decomposition reuses on one grid, in the memory of the backend it was built under:
the real forward and backward plans, the array each component passes through (a plan applies to arrays of
the alignment it was built for, which a component view need not have), one spectral scalar and `k·û`, the
velocity's spectrum, and the derivative symbol.
"""
struct PeriodicSpectralState{T,W,S,V,K,P,Q}
    work::W
    spec::S
    kdotu::S
    velocity_hat::V
    K::K
    fplan::P
    bplan::Q
    scale::T
end

Base.show(io::IO, ::PeriodicSpectralState{T}) where {T} = print(io, "PeriodicSpectralState{", T, "}(…)")

HD.close!(s::PeriodicSpectralState) = s

function HD.prepare_spectral(::HD.CartesianSpectralSolver,
                             grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N};
                             execution = CB.AutoBackend()) where {T,N}
    exec = HD.resolve_execution_backend(execution)
    dims = size(grid)
    kdims = FTB.packed_size(T, dims)
    work = HD._zeros(exec, T, dims)
    spec = HD._zeros(exec, Complex{T}, kdims)
    fplan = AbstractFFTs.plan_rfft(work; HD._plan_kwargs(work, exec)...)
    bplan = AbstractFFTs.plan_brfft(spec, dims[1]; HD._plan_kwargs(spec, exec)...)
    ks = FTB.wavenumber_axes(T, ntuple(d -> T(FG.Grids.domain_length(grid, d)), Val(N)), dims)
    return PeriodicSpectralState(work, spec, HD._zeros(exec, Complex{T}, kdims),
                                 HD._zeros(exec, Complex{T}, (kdims..., N)), HD._derivative_symbol(spec, T, ks),
                                 fplan, bplan, one(T) / T(prod(dims)))
end

function HD._decompose_spectral(
    ::HD.CartesianSpectralSolver,
    ::FG.Geometry.AbstractCartesianGeometry,
    U::AbstractArray{T,M},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    execution, st::PeriodicSpectralState,
) where {T,M,N}
    Ud = FG.Execution.on_backend(execution, U)
    for c in 1:N
        copyto!(st.work, HD._component(Ud, c, Val(N)))
        LinearAlgebra.mul!(st.spec, st.fplan, st.work)
        copyto!(HD._component(st.velocity_hat, c, Val(N)), st.spec)
    end
    # A multi-dimensional complex-to-real plan consumes its input, the scratch refilled for every inverse.
    invert! = function (out, x)
        LinearAlgebra.mul!(st.work, st.bplan, x)
        out .= st.work .* st.scale
        return out
    end
    return HD._cartesian_result(grid, Ud, st.velocity_hat, st.K, st.kdotu, st.spec, invert!)
end

end # module
