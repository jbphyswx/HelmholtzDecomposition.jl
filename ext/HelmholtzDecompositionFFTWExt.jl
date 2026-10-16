# Cartesian Poisson solves and the spectral decomposition through FFTW, in any dimension: on a
# periodic grid forward `rfft`, divide by the discrete Laplacian's symbol, inverse `irfft`; on a
# bounded or channel grid the per-direction `r2r` cosine and sine transforms.
module HelmholtzDecompositionFFTWExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using FlowGeometries: FlowGeometries as FG
using FlowTransformBindings: FlowTransformBindings as FTB
using LinearAlgebra: LinearAlgebra
# The periodic path calls the `AbstractFFTs` interface, so one implementation serves whatever
# backend owns the array, CUFFT for a device array and FFTW for a host one. `r2r` is FFTW's own.
using AbstractFFTs: AbstractFFTs

# `rfft`/`irfft` expand in a periodic basis over every cell of the array: the grid must wrap in every
# direction, its axes must be evenly spaced, and no cell may be masked out, each checked separately.
# Such a grid has no boundary, so every boundary condition is accepted and none acts.
HD.supports_boundary(::HD.CartesianSpectralSolver, ::FG.Operators.AbstractBoundaryCondition) = true
HD.requires_full_domain(::HD.CartesianSpectralSolver) = true
HD.requires_uniform_axes(::HD.CartesianSpectralSolver) = true
HD.requires_periodic_domain(::HD.CartesianSpectralSolver) = true

# FFTW plans a host array with its process-wide thread count unless told otherwise; each plan here
# takes the thread count of the backend it runs under.
HD._plan_kwargs(::Array{<:Union{Float32,Float64,ComplexF32,ComplexF64}}, backend) =
    (; num_threads = HD._library_threads(backend))

"""
    PeriodicSpectralState

The reusable half of a periodic solve: the two transform plans, the spectral buffer they execute
through, the symbol as one vector per axis, and the inverse's normalisation.

None of it depends on the right-hand side, while a decomposition solves `P + 1` of those and a
batch multiplies that by the field count.

Plan-owning, so it gets an explicit one-line `show`: the default prints the plan's internals.
"""
struct PeriodicSpectralState{T,S,FP,BP,L}
    spec::S
    fplan::FP
    bplan::BP
    λ::L
    norm::T
end

Base.show(io::IO, ::PeriodicSpectralState{T}) where {T} =
    print(io, "PeriodicSpectralState{", T, "}(…)")

# The symbol of the discrete `L`, `−Σ_d (4/h_d²)·sin²(k_d h_d/2)`, one vector per axis, so the
# transform is an exact direct solve of the `L` that `D` and `G` compose.
#
# `k_d h_d / 2 = π·m/n_d` with `m` the signed mode index, and `sin²` is even, so the folded index
# `min(i−1, n−i+1)` serves both the halved first axis and the full ones.
function _periodic_symbols(::Type{T}, dims::NTuple{N,Int}, kdims::NTuple{N,Int}, h, backend) where {T,N}
    return ntuple(Val(N)) do d
        n, nk = dims[d], kdims[d]
        v = HD._zeros(backend, T, (nk,))
        i = reshape(1:nk, nk)
        s = sin.(T(π) .* min.(i .- 1, n .- i .+ 1) ./ T(n))
        v .= .-4 .* s .^ 2 ./ (h[d] * h[d])
        reshape(v, ntuple(e -> e == d ? nk : 1, Val(N)))
    end
end

function HD.prepare_solver(
    ::HD.CartesianSpectralSolver,
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    ::FG.Operators.AbstractBoundaryCondition;
    backend = CB.SerialBackend(), shared = nothing,
) where {T,N}
    dims = size(grid)
    h = ntuple(d -> abs(FG.Grids.spacing(grid, d)), Val(N))
    kdims = ntuple(d -> d == 1 ? dims[1] ÷ 2 + 1 : dims[d], Val(N))
    work = HD._zeros(backend, T, dims)
    spec = HD._zeros(backend, complex(T), kdims)
    return PeriodicSpectralState(spec, AbstractFFTs.plan_rfft(work; HD._plan_kwargs(work, backend)...),
                                 AbstractFFTs.plan_brfft(spec, dims[1]; HD._plan_kwargs(spec, backend)...),
                                 _periodic_symbols(T, dims, kdims, h, backend),
                                 one(T) / T(prod(dims)))
end

function HD.solve_poisson!(
    Φ::AbstractArray{T,N},
    RHS::AbstractArray{T,N},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    solver::HD.CartesianSpectralSolver;
    boundary::FG.Operators.AbstractBoundaryCondition = FG.Operators.Neumann(),
    state::PeriodicSpectralState = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat,N}
    HD._require_boundary(solver, boundary)
    LinearAlgebra.mul!(state.spec, state.fplan, RHS)
    λ = HD.lazy_axis_sum(state.λ)
    state.spec .= ifelse.(iszero.(λ), zero(complex(T)), state.spec ./ λ)
    # A multi-dimensional complex-to-real plan consumes its input; `spec` is the state's own and is
    # refilled by the forward transform on each solve.
    LinearAlgebra.mul!(Φ, state.bplan, state.spec)
    Φ .*= state.norm
    return HD.SolverResult{T}(true, 1, zero(T))
end

function HD._decompose_spectral(
    ::HD.CartesianSpectralSolver,
    ::FG.Geometry.AbstractCartesianGeometry,
    U::AbstractArray{T,M},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N}; kwargs...,
) where {T,M,N}
    dims = size(grid)
    ks = FTB.wavenumber_axes(ntuple(d -> T(FG.Grids.domain_length(grid, d)), Val(N)), dims, Val(true))

    # Forward transform each velocity component → component-last spectral array.
    ĉ1 = AbstractFFTs.rfft(HD._component(U, 1, Val(N)))
    # The spectral buffer is allocated with `similar`, so it lives where the transform put its output.
    velocity_hat = similar(ĉ1, (size(ĉ1)..., N))
    copyto!(HD._component(velocity_hat, 1, Val(N)), ĉ1)
    for c in 2:N
        copyto!(HD._component(velocity_hat, c, Val(N)),
                AbstractFFTs.rfft(HD._component(U, c, Val(N))))
    end

    inverse = x -> AbstractFFTs.irfft(x, dims[1])
    return HD.build_cartesian_result(grid, U, velocity_hat, ks, inverse)
end

# ---------------------------------------------------------------------------
# Bounded domains: the cosine and sine transforms
# ---------------------------------------------------------------------------

HD.supports_boundary(::HD.CartesianBoundedSolver, ::FG.Operators.Dirichlet) = true
HD.supports_boundary(::HD.CartesianBoundedSolver, ::FG.Operators.Neumann) = true
HD.supports_boundary(::HD.CartesianBoundedSolver, ::HD.CornerDirichlet) = true
HD.requires_full_domain(::HD.CartesianBoundedSolver) = true
HD.requires_uniform_axes(::HD.CartesianBoundedSolver) = true

# The forward transform kind along direction `d`, chosen per direction so one plan serves a mix: a
# channel is `R2HC` along its periodic direction and a cosine or sine transform across it.
@inline _forward_kind(::FG.Operators.Neumann, periodic::Bool, ::Int) = periodic ? FFTW.R2HC : FFTW.REDFT10
@inline _forward_kind(::FG.Operators.Dirichlet, periodic::Bool, ::Int) = periodic ? FFTW.R2HC : FFTW.RODFT10
@inline _forward_kind(bc::HD.CornerDirichlet, periodic::Bool, d::Int) =
    periodic ? FFTW.R2HC : bc.staggered[d] ? FFTW.RODFT00 : FFTW.RODFT10

@inline _inverse_kind(kind) = kind == FFTW.R2HC ? FFTW.HC2R : kind == FFTW.REDFT10 ? FFTW.REDFT01 :
                              kind == FFTW.RODFT10 ? FFTW.RODFT01 : kind

# A forward/inverse pair is unnormalized by `n` for the halfcomplex transform, `2n` for the
# cell-centred cosine and sine transforms and `2(n + 1)` for DST-I.
@inline _kind_norm(::Type{T}, n::Int, kind) where {T} =
    kind == FFTW.R2HC ? T(n) : kind == FFTW.RODFT00 ? T(2(n + 1)) : T(2n)

"""
    _axis_symbol(proto, T, n, h, kind) -> Vector{T}

The Laplacian's eigenvalue along one direction, per mode index, for that direction's transform.

- periodic (`R2HC`): `−(4/h²)·sin²(πk/n)`, where the halfcomplex layout stores index `i` at
  frequency `min(i, n−i)`, so both halves of a conjugate pair share an eigenvalue;
- Neumann (`REDFT10`): `−(4/h²)·sin²(πk/2n)`, `k = 0…n−1`, with `λ₀ = 0`, the constant;
- Dirichlet at the face (`RODFT10`): `−(4/h²)·sin²(π(k+1)/2n)`, with no null mode;
- Dirichlet at the node one cell beyond (`RODFT00`, DST-I): `−(4/h²)·sin²(π(k+1)/(2(n+1)))`.
"""
function _axis_symbol(proto, ::Type{T}, n::Int, h::T, kind) where {T}
    λ = similar(proto, T, n)
    @inbounds for i in 1:n
        s = if kind == FFTW.R2HC
            sin(T(π) * T(min(i - 1, n - i + 1)) / T(n))
        elseif kind == FFTW.RODFT10
            sin(T(π) * T(i) / T(2n))
        elseif kind == FFTW.RODFT00
            sin(T(π) * T(i) / T(2(n + 1)))
        else
            sin(T(π) * T(i - 1) / T(2n))
        end
        λ[i] = -4 * s * s / (h * h)
    end
    return λ
end

# One vector per axis, each reshaped to broadcast along its own. The whole-grid symbol is their
# sum, formed at the divide by `HD.lazy_axis_sum`.
function _bounded_symbol(proto, ::Type{T}, dims::NTuple{N,Int}, h::NTuple{N,T}, kinds) where {T,N}
    return ntuple(Val(N)) do d
        v = _axis_symbol(proto, T, dims[d], h[d], kinds[d])
        reshape(v, ntuple(e -> e == d ? dims[d] : 1, Val(N)))
    end
end

"""
    BoundedState{T,A,PF,PI,L}

The reusable half of a bounded solve: the two `r2r` plans, the symbol's per-axis factors, and the
scratch they transform through. All of it depends on the grid and the boundary condition only, so
it is built once per [`HelmholtzPlan`](@ref) across the `(P+1)·B` solves.

Plan-owning, so it gets an explicit one-line `show`: the default prints the FFTW plan's internals,
which can segfault.
"""
struct BoundedState{T,A<:AbstractArray{T},PF,PI,L}
    forward::PF
    inverse::PI
    λ::L
    scratch::A
    norm::T
end

Base.show(io::IO, s::BoundedState{T}) where {T} =
    print(io, "BoundedState{", T, "}(", join(size(s.scratch), "×"), ")")

function HD.prepare_solver(
    ::HD.CartesianBoundedSolver,
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    boundary::FG.Operators.AbstractBoundaryCondition;
    backend = CB.SerialBackend(), shared = nothing,
) where {T,N}
    dims = size(grid)
    h = ntuple(d -> abs(FG.Grids.spacing(grid, d)), Val(N))
    per = ntuple(d -> FG.Grids.isperiodic(grid, d), Val(N))
    # `FFTW.plan_r2r!` plans against host memory; a device array takes
    # `CartesianRealTransformSolver`, which goes through `AbstractFFTs`.
    scratch = HD._zeros(backend, T, dims)
    scratch isa Array || throw(ArgumentError(
        "CartesianBoundedSolver uses FFTW's real-to-real transform, which is host-only. Load " *
        "`AbstractFFTs` and let `CartesianRealTransformSolver` take this grid, or plan it on the " *
        "host."))
    kinds = ntuple(d -> _forward_kind(boundary, per[d], d), Val(N))
    fwd = FFTW.plan_r2r!(scratch, kinds; HD._plan_kwargs(scratch, backend)...)
    inv = FFTW.plan_r2r!(scratch, map(_inverse_kind, kinds); HD._plan_kwargs(scratch, backend)...)
    norm = one(T) / prod(ntuple(d -> _kind_norm(T, dims[d], kinds[d]), Val(N)))
    λ = _bounded_symbol(scratch, T, dims, h, kinds)
    return BoundedState{T,typeof(scratch),typeof(fwd),typeof(inv),typeof(λ)}(
        fwd, inv, λ, scratch, norm,
    )
end

function HD.solve_poisson!(
    Φ::AbstractArray{T,N},
    RHS::AbstractArray{T,N},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    solver::HD.CartesianBoundedSolver;
    boundary::FG.Operators.AbstractBoundaryCondition = FG.Operators.Neumann(),
    state::BoundedState = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat,N}
    HD._require_boundary(solver, boundary)
    copyto!(state.scratch, RHS)
    state.forward * state.scratch                    # in place
    λ = HD.lazy_axis_sum(state.λ)
    state.scratch .= ifelse.(iszero.(λ), zero(T), state.scratch ./ λ)
    state.inverse * state.scratch
    @. Φ = state.scratch * state.norm
    return HD.SolverResult{T}(true, 1, zero(T))
end

end # module
