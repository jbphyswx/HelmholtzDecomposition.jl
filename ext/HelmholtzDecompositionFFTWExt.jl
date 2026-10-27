# FFTW's own part of the Cartesian solvers: the thread count of every FFTW plan the AbstractFFTs solvers
# build on a host array, and the bounded and channel Poisson solve through FFTW's real-to-real cosine,
# sine and halfcomplex transforms, in any dimension.
module HelmholtzDecompositionFFTWExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using FlowGeometries: FlowGeometries as FG

# FFTW plans a host array with its process-wide thread count unless told otherwise; each plan here
# takes the thread count of the backend it runs under.
HD._plan_kwargs(::Array{<:Union{Float32,Float64,ComplexF32,ComplexF64}}, backend) =
    (; num_threads = HD._library_threads(backend))

HD._available(::HD._HostFFT) = true

HD.library_loaded(::HD.CartesianBoundedSolver) = true
HD.supports_boundary(::HD.CartesianBoundedSolver, ::FG.Operators.Dirichlet) = true
HD.supports_boundary(::HD.CartesianBoundedSolver, ::FG.Operators.Neumann) = true
HD.supports_boundary(::HD.CartesianBoundedSolver, ::HD.CornerDirichlet) = true
HD.requires_full_domain(::HD.CartesianBoundedSolver) = true
HD.requires_uniform_axes(::HD.CartesianBoundedSolver) = true

HD._backend_message(::HD.CartesianBoundedSolver, backend) =
    "CartesianBoundedSolver plans FFTW's real-to-real transforms, which run on the host, and cannot " *
    "run under $(nameof(typeof(backend))). `CartesianRealTransformSolver()` takes the same grids " *
    "through any FFT library for the device's arrays, and `CGSolver()` any grid."

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
    solver::HD.CartesianBoundedSolver,
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    boundary::FG.Operators.AbstractBoundaryCondition;
    execution = CB.SerialBackend(), shared = nothing,
) where {T,N}
    HD._require_backend(solver, execution, grid)
    dims = size(grid)
    h = ntuple(d -> abs(FG.Grids.spacing(grid, d)), Val(N))
    per = ntuple(d -> FG.Grids.isperiodic(grid, d), Val(N))
    scratch = HD._zeros(execution, T, dims)
    kinds = ntuple(d -> _forward_kind(boundary, per[d], d), Val(N))
    fwd = FFTW.plan_r2r!(scratch, kinds; HD._plan_kwargs(scratch, execution)...)
    inv = FFTW.plan_r2r!(scratch, map(_inverse_kind, kinds); HD._plan_kwargs(scratch, execution)...)
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
