"""
    HelmholtzDecompositionAbstractFFTsExt — bounded and channel domains on any FFT backend.

A DCT-II/DST-II **is** an FFT — of the even/odd extension — so the cosine and sine transforms that
diagonalize the bounded Laplacian need no library beyond the `AbstractFFTs` interface. `FFTW.r2r`
is a faster *host implementation* of them, not the only one, which is why this exists: through
`AbstractFFTs` the same solver runs wherever the array's own backend provides a transform, with no
vendor-specific code.

The relations, each verified term by term against `FFTW.r2r` (see the plan's Stage 7a):

    even extension y = [x; reverse(x)]      ⇒  DCT-II[k] = real( rfft(y)[k+1] · e^{-iπk/2N} )
    odd  extension y = [x; −reverse(x)]     ⇒  DST-II[k] = −imag( rfft(y)[k+2] · e^{-iπ(k+1)/2N} )

and the inverse by reading the first backwards: rebuild `Z[k+1] = C[k]·e^{+iπk/2N}`, pad one zero
to the `rfft` layout of a length-`2N` real signal, `brfft(Z, 2N)`, take the first `N` over `2N`.

The `2N` extension is this form's cost, and it is not worth removing. The length-`N`
pre-permutation variant (Makhoul) was implemented and measured: it matches `r2r` to 2e-16 including
odd `N`, but runs at 1.03×/0.99× the extension form at 256²/512² — neutral. It halves the bytes and
gives them back by reading with stride 2, where the extension fill is two contiguous copies.

What remains, ~1.8× against `FFTW.r2r`, is inherent to building a cosine transform out of an FFT:
any such construction materialises an intermediate, and `r2r` does the transform natively with
none. That gap is only ever paid where `r2r` does not exist — a non-FFTW backend, or a device —
which is exactly when this solver runs at all.
"""
module HelmholtzDecompositionAbstractFFTsExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra

"""
    CartesianRealTransformSolver

Direct `O(N log N)` Poisson solve on a uniform Cartesian grid with any mix of bounded and periodic
directions, using only the `AbstractFFTs` interface — so it follows the array to whatever backend
owns it.
"""
struct CartesianRealTransformSolver <: HD.AbstractPoissonSolver end

HD.supports_boundary(::CartesianRealTransformSolver, ::HD.Dirichlet) = true
HD.supports_boundary(::CartesianRealTransformSolver, ::HD.Neumann) = true
HD.requires_full_domain(::CartesianRealTransformSolver) = true
HD.requires_uniform_axes(::CartesianRealTransformSolver) = true

# `AbstractFFTs` is an interface, not an implementation: loading it does not mean a transform can be
# planned, and packages pull it in without a backend. So capability is asked of the array type the
# grid's own storage uses — asking it of `Array` would answer for the host while the solver ran on
# a device.
@inline _has_backend(::Type{A}) where {A<:AbstractVector} =
    hasmethod(AbstractFFTs.plan_rfft, Tuple{A,UnitRange{Int}})

@inline _storage_vector(grid) = typeof(similar(FG.Grids.measure(grid), eltype(grid), 1))

HD._require_sampling(::CartesianRealTransformSolver, grid) =
    _has_backend(_storage_vector(grid)) ? nothing : throw(ArgumentError(
        "CartesianRealTransformSolver needs an FFT backend for $(_storage_vector(grid)); " *
        "`AbstractFFTs` alone is an interface and provides none. Load one (`using FFTW` on the " *
        "host, or the array's own package on a device)."))

# ---------------------------------------------------------------------------
# Real path: every direction bounded
# ---------------------------------------------------------------------------
#
# A cosine or sine transform of real data is real, so with no periodic direction nothing in the
# pipeline ever needs to be complex — only the length-`2n` half-spectrum in between. Doing it in
# the complex domain anyway, through the one-shot interface that plans and allocates per call,
# measured 15-22x slower than `FFTW.r2r` and **176 MiB per solve** at 512². Plans and buffers held
# here bring that to zero allocations, and this is the path a device takes, where `r2r` cannot go.

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

function _dir_plan(::Type{T}, dims::Dims{N}, d::Int, off::Int) where {T,N}
    ext = zeros(T, ntuple(e -> e == d ? 2dims[e] : dims[e], Val(N)))
    spec = zeros(complex(T), ntuple(e -> e == d ? dims[e] + 1 : dims[e], Val(N)))
    return DirPlan(ext, spec, AbstractFFTs.plan_rfft(ext, d:d),
                   AbstractFFTs.plan_brfft(spec, 2dims[d], d:d), _twiddle(T, dims[d], off, N, d))
end

# Symbol of the discrete `L` along one direction, so this inverts the operator the decomposition
# differentiates with. `πk/n` where the direction wraps, `π(k+off)/2n` where it ends.
@inline _axis_symbol(::Type{T}, k::Int, n::Int, h::T, periodic::Bool, off::Int) where {T} =
    (s = periodic ? sin(T(π) * T(k) / T(n)) : sin(T(π) * T(k + off) / T(2n));
     -4 * s * s / (h * h))

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

At least one direction wraps. The bounded directions transform first, in real arithmetic; the
periodic ones then go through one `rfft`, which halves the first of them.
"""
struct MixedTopologyState{T,N,D,P,IP,L,A,S}
    dirs::D            # bounded directions only, paired with `bdims`
    bdims::NTuple{<:Any,Int}
    pdims::NTuple{<:Any,Int}
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

function HD.prepare_solver(::CartesianRealTransformSolver,
                           grid::FG.Grids.StructuredGrid{<:FG.Geometry.AbstractCartesianGeometry,T,N},
                           boundary::HD.AbstractBoundaryCondition) where {T,N}
    dims = size(grid)
    off = boundary isa HD.Dirichlet ? 1 : 0
    per = ntuple(d -> FG.Grids.isperiodic(grid, d), Val(N))
    h = ntuple(d -> abs(FG.Grids.spacing(grid, d)), Val(N))
    work = HD.allocate_zeros(nothing, T, dims)

    if !any(per)
        dirs = ntuple(d -> _dir_plan(T, dims, d, off), Val(N))
        λ = zeros(T, dims)
        @inbounds for I in CartesianIndices(dims)
            λ[I] = sum(d -> _axis_symbol(T, I[d] - 1, dims[d], h[d], false, off), 1:N)
        end
        return RealBoundedState{T,N,typeof(dirs),typeof(λ),typeof(work)}(dirs, λ, work)
    end

    bdims = Tuple(d for d in 1:N if !per[d])
    pdims = Tuple(d for d in 1:N if per[d])
    dirs = map(d -> _dir_plan(T, dims, d, off), bdims)
    spec = zeros(complex(T), ntuple(e -> e == pdims[1] ? dims[e] ÷ 2 + 1 : dims[e], Val(N)))
    pplan = AbstractFFTs.plan_rfft(work, pdims)
    iplan = AbstractFFTs.plan_brfft(spec, dims[pdims[1]], pdims)
    kdims = size(spec)
    λ = zeros(T, kdims)
    @inbounds for I in CartesianIndices(kdims)
        λ[I] = sum(1:N) do d
            # A periodic direction after the halving carries `k = i − 1`; the others run in FFT
            # order, where the upper half stands for negative frequencies of the same magnitude.
            k = if !per[d]
                I[d] - 1
            elseif d == pdims[1]
                I[d] - 1
            else
                I[d] - 1 < cld(dims[d], 2) ? I[d] - 1 : dims[d] - I[d] + 1
            end
            _axis_symbol(T, k, dims[d], h[d], per[d], off)
        end
    end
    nrm = T(prod(d -> dims[d], pdims))
    return MixedTopologyState{T,N,typeof(dirs),typeof(pplan),typeof(iplan),typeof(λ),typeof(work),
                              typeof(spec)}(dirs, bdims, pdims, pplan, iplan, λ, work, spec, nrm)
end

# x -> [x; ±reverse(x)] along `d`, into the preallocated buffer. `selectdim` with a descending
# range is a view, where `reverse` would build the whole array.
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
    odd = boundary isa HD.Dirichlet
    copyto!(st.work, RHS)
    for d in 1:N
        _forward_real!(st.work, st.dirs[d], d, odd)
    end
    st.work .= ifelse.(iszero.(st.λ), zero(T), st.work ./ st.λ)
    for d in 1:N
        _inverse_real!(st.work, st.dirs[d], d, odd)
    end
    copyto!(Φ, st.work)
    return HD.SolverResult{T}(true, 1, zero(T))
end

# Bounded directions first, in real arithmetic; the periodic ones then take a single `rfft`, which
# halves the first of them. Promoting everything to complex would double the spectral storage and
# carry an imaginary part that is zero by construction.
function _solve!(Φ::AbstractArray{T,N}, RHS, st::MixedTopologyState, boundary) where {T,N}
    odd = boundary isa HD.Dirichlet
    copyto!(st.work, RHS)
    for (i, d) in enumerate(st.bdims)
        _forward_real!(st.work, st.dirs[i], d, odd)
    end
    LinearAlgebra.mul!(st.spec, st.pplan, st.work)
    st.spec .= ifelse.(iszero.(st.λ), zero(complex(T)), st.spec ./ st.λ)
    LinearAlgebra.mul!(st.work, st.iplan, st.spec)
    st.work ./= st.norm
    for (i, d) in enumerate(st.bdims)
        _inverse_real!(st.work, st.dirs[i], d, odd)
    end
    copyto!(Φ, st.work)
    return HD.SolverResult{T}(true, 1, zero(T))
end

@inline _twiddle(::Type{T}, n::Int, off::Int, nd::Int, d::Int) where {T} =
    reshape([cis(-T(π) * T(k + off) / T(2n)) for k in 0:(n - 1)],
            ntuple(e -> e == d ? n : 1, nd))

"""
    solve_poisson!(Φ, RHS, grid, solver; boundary, state)

`state` comes from [`prepare_solver`](@ref) and selects the implementation — positionally, on
`_solve!`, because Julia does not dispatch on keyword arguments: two methods differing only in a
keyword's type are the *same* method, and the second silently replaces the first.
"""
function HD.solve_poisson!(
    Φ::AbstractArray{T,N},
    RHS::AbstractArray{T,N},
    grid::FG.Grids.StructuredGrid{<:FG.Geometry.AbstractCartesianGeometry,T,N},
    solver::CartesianRealTransformSolver;
    boundary::HD.AbstractBoundaryCondition = HD.Neumann(),
    state = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat,N}
    HD._require_boundary(solver, boundary)
    return _solve!(Φ, RHS, state, boundary)
end

function __init__()
    HD.register_spectral_solver!(SB.FFTSpectralBackend, CartesianRealTransformSolver;
                                 priority = 20)
end

end # module
