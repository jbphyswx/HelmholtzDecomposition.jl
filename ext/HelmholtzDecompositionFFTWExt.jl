"""
    HelmholtzDecompositionFFTWExt — Cartesian spectral solver via FFTW (any dimension).

Provides `O(N log N)` spectral Poisson solves and full Helmholtz decomposition for regular
periodic Cartesian grids in 1D/2D/3D/ND using the real FFT: forward `rfft` → divide by
`-Σ k_d²` → inverse `irfft`. Returns physical-space fields via `build_cartesian_result`.
"""
module HelmholtzDecompositionFFTWExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using FlowGeometries: FlowGeometries as FG
using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB
# The periodic path goes through the `AbstractFFTs` interface rather than FFTW's own functions, so
# it dispatches to whatever backend owns the array — CUFFT for a device array, FFTW for a host one
# — from one implementation. The bounded path below has no such option: `r2r` is FFTW-specific.
using AbstractFFTs: AbstractFFTs

"""
    CartesianSpectralSolver <: AbstractPoissonSolver

Spectral Poisson solver for regular periodic Cartesian grids (any dimension) using FFTW.
"""
struct CartesianSpectralSolver <: HD.AbstractPoissonSolver end

# `rfft`/`irfft` expand in a periodic basis over every cell of the array. That makes three
# separate demands, each checked as itself rather than conflated into one flag: the grid must
# wrap in every direction, its axes must be evenly spaced, and no cell may be masked out. On a
# grid meeting those there is no boundary anywhere, so a boundary condition is vacuous — which
# is why this accepts any of them rather than naming one.
HD.supports_boundary(::CartesianSpectralSolver, ::HD.AbstractBoundaryCondition) = true
HD.requires_full_domain(::CartesianSpectralSolver) = true
HD.requires_uniform_axes(::CartesianSpectralSolver) = true
HD.requires_periodic_domain(::CartesianSpectralSolver) = true

# Per-axis angular wavenumbers matching an rfft layout (axis 1 reduced).
function _rfft_wavenumbers(::Type{T}, dims::NTuple{N,Int}, spacing::NTuple{N,T}) where {T,N}
    return ntuple(Val(N)) do d
        if d == 1
            T(2π) .* AbstractFFTs.rfftfreq(dims[1], one(T) / spacing[1])
        else
            T(2π) .* AbstractFFTs.fftfreq(dims[d], one(T) / spacing[d])
        end
    end
end

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

# The DISCRETE symbol, `−Σ_d (4/h_d²)·sin²(k_d h_d/2)`, one vector per axis. Dividing by the
# continuous `−k²` inverts a different operator from the one `D` and `G` compose; with this one the
# transform is an exact direct solve of the same `L` the iterative solver approaches.
#
# `k_d h_d / 2 = π·m/n_d` with `m` the signed mode index, and `sin²` is even, so the folded index
# `min(i−1, n−i+1)` serves both the halved first axis and the full ones.
function _periodic_symbols(::Type{T}, dims::NTuple{N,Int}, kdims::NTuple{N,Int}, h, backend) where {T,N}
    return ntuple(Val(N)) do d
        n, nk = dims[d], kdims[d]
        v = HD.allocate_zeros(backend, T, (nk,))
        i = reshape(1:nk, nk)
        s = sin.(T(π) .* min.(i .- 1, n .- i .+ 1) ./ T(n))
        v .= .-4 .* s .^ 2 ./ (h[d] * h[d])
        reshape(v, ntuple(e -> e == d ? nk : 1, Val(N)))
    end
end

function HD.prepare_solver(
    ::CartesianSpectralSolver,
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    ::HD.AbstractBoundaryCondition;
    backend = CB.SerialBackend(), shared = nothing,
) where {T,N}
    dims = size(grid)
    h = ntuple(d -> abs(FG.Grids.spacing(grid, d)), Val(N))
    kdims = ntuple(d -> d == 1 ? dims[1] ÷ 2 + 1 : dims[d], Val(N))
    work = HD.allocate_zeros(backend, T, dims)
    spec = HD.allocate_zeros(backend, complex(T), kdims)
    return PeriodicSpectralState(spec, AbstractFFTs.plan_rfft(work),
                                 AbstractFFTs.plan_brfft(spec, dims[1]),
                                 _periodic_symbols(T, dims, kdims, h, backend),
                                 one(T) / T(prod(dims)))
end

function HD.solve_poisson!(
    Φ::AbstractArray{T,N},
    RHS::AbstractArray{T,N},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    solver::CartesianSpectralSolver;
    boundary::HD.AbstractBoundaryCondition = HD.Neumann(),
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
    ::CartesianSpectralSolver,
    ::FG.Geometry.AbstractCartesianGeometry,
    U::AbstractArray{T,M},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N};
    backend = CB.SerialBackend(), kwargs...,
) where {T,M,N}
    dims = size(grid)
    ks = _rfft_wavenumbers(T, dims, ntuple(d -> abs(FG.Grids.spacing(grid, d)), Val(N)))

    # Forward transform each velocity component → component-last spectral array.
    ĉ1 = AbstractFFTs.rfft(HD._component(U, 1, Val(N)))
    # `similar`, not `Array{...}(undef, …)`: the spectral buffer must live wherever the transform
    # put its output, or a device array would be copied back to the host here.
    velocity_hat = similar(ĉ1, (size(ĉ1)..., N))
    copyto!(HD._component(velocity_hat, 1, Val(N)), ĉ1)
    for c in 2:N
        copyto!(HD._component(velocity_hat, c, Val(N)),
                AbstractFFTs.rfft(HD._component(U, c, Val(N))))
    end

    # Fully-spectral decomposition (exact derivatives) → physical HelmholtzResult.
    inverse = x -> AbstractFFTs.irfft(x, dims[1])
    return HD.build_cartesian_result(grid, U, velocity_hat, ks, inverse; backend = backend)
end

# ---------------------------------------------------------------------------
# Bounded domains: the cosine and sine transforms
# ---------------------------------------------------------------------------

"""
    CartesianBoundedSolver

Direct `O(N log N)` Poisson solve on a uniform Cartesian grid with **any** mix of bounded and
periodic directions — including a channel, which no single transform kind covers.

A complex-exponential basis diagonalizes the periodic Laplacian and nothing else, which is why
`CartesianSpectralSolver` refuses a bounded grid. The remedy is not to abandon the transform but
to use the one matched to the boundary: DCT-II/III (`REDFT10`/`REDFT01`) diagonalizes the
cell-centred **Neumann** Laplacian, DST-II/III (`RODFT10`/`RODFT01`) the **Dirichlet** one. Both
are FFTs — of the even and odd extension — so a bounded domain costs the same `O(N log N)` as a
periodic one instead of falling back to an iterative solve.

The eigenvalues are the *discrete* operator's, so this inverts exactly the `L = D G` the
decomposition differentiates with:

| condition | kinds | eigenvalue |
|---|---|---|
| Neumann   | `REDFT10`/`REDFT01` | `−(4/h²)·sin²(πk/2N)`, `k = 0…N−1`; `λ₀ = 0`, the constant |
| Dirichlet | `RODFT10`/`RODFT01` | `−(4/h²)·sin²(π(k+1)/2N)`, with no null mode |
"""
struct CartesianBoundedSolver <: HD.AbstractPoissonSolver end

HD.supports_boundary(::CartesianBoundedSolver, ::HD.Dirichlet) = true
HD.supports_boundary(::CartesianBoundedSolver, ::HD.Neumann) = true
HD.requires_full_domain(::CartesianBoundedSolver) = true
HD.requires_uniform_axes(::CartesianBoundedSolver) = true

# The transform kind is chosen per direction, so a grid may mix them: a channel — periodic in `x`,
# bounded in `y` — is one plan, `R2HC` along `x` and `REDFT10` along `y`. This is why the kind is
# selected from the grid's own topology rather than from a single whole-grid flag.
@inline _forward_kind(::HD.Neumann, periodic::Bool) = periodic ? FFTW.R2HC : FFTW.REDFT10
@inline _forward_kind(::HD.Dirichlet, periodic::Bool) = periodic ? FFTW.R2HC : FFTW.RODFT10
@inline _inverse_kind(::HD.Neumann, periodic::Bool) = periodic ? FFTW.HC2R : FFTW.REDFT01
@inline _inverse_kind(::HD.Dirichlet, periodic::Bool) = periodic ? FFTW.HC2R : FFTW.RODFT01

# A forward/inverse pair is unnormalized by `2n` for the even/odd transforms and by `n` for the
# halfcomplex one.
@inline _kind_norm(::Type{T}, n::Int, periodic::Bool) where {T} = periodic ? T(n) : T(2n)

"""
    _axis_symbol(T, n, h, bc, periodic) -> Vector{T}

The Laplacian's eigenvalue along one direction, per mode index, for that direction's transform.

- periodic (`R2HC`): `−(4/h²)·sin²(πk/n)`, where the halfcomplex layout stores index `i` at
  frequency `min(i, n−i)` — both halves of a conjugate pair share an eigenvalue, which is what
  makes the real transform valid here at all;
- Neumann (`REDFT10`): `−(4/h²)·sin²(πk/2n)`, with `λ₀ = 0`, the constant;
- Dirichlet (`RODFT10`): `−(4/h²)·sin²(π(k+1)/2n)` — the sine transform's lowest mode sits one
  step up, which is exactly why Dirichlet has no null mode and Neumann does.
"""
function _axis_symbol(proto, ::Type{T}, n::Int, h::T, bc, periodic::Bool) where {T}
    λ = similar(proto, T, n)
    @inbounds for i in 1:n
        s = if periodic
            sin(T(π) * T(min(i - 1, n - i + 1)) / T(n))
        elseif bc isa HD.Dirichlet
            sin(T(π) * T(i) / T(2n))
        else
            sin(T(π) * T(i - 1) / T(2n))
        end
        λ[i] = -4 * s * s / (h * h)
    end
    return λ
end

# One vector per axis, each reshaped to broadcast along its own. The whole-grid symbol is their
# sum, formed at the divide by `HD.lazy_axis_sum`.
function _bounded_symbol(proto, ::Type{T}, dims::NTuple{N,Int}, h::NTuple{N,T}, bc,
                         periodic::NTuple{N,Bool}) where {T,N}
    return ntuple(Val(N)) do d
        v = _axis_symbol(proto, T, dims[d], h[d], bc, periodic[d])
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
    ::CartesianBoundedSolver,
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    boundary::HD.AbstractBoundaryCondition;
    backend = CB.SerialBackend(), shared = nothing,
) where {T,N}
    dims = size(grid)
    h = ntuple(d -> abs(FG.Grids.spacing(grid, d)), Val(N))
    per = ntuple(d -> FG.Grids.isperiodic(grid, d), Val(N))
    # `FFTW.plan_r2r!` plans against host memory; a device array takes
    # `CartesianRealTransformSolver`, which goes through `AbstractFFTs`.
    scratch = HD.allocate_zeros(backend, T, dims)
    scratch isa Array || throw(ArgumentError(
        "CartesianBoundedSolver uses FFTW's real-to-real transform, which is host-only. Load " *
        "`AbstractFFTs` and let `CartesianRealTransformSolver` take this grid, or plan it on the " *
        "host."))
    fwd = FFTW.plan_r2r!(scratch, ntuple(d -> _forward_kind(boundary, per[d]), Val(N)))
    inv = FFTW.plan_r2r!(scratch, ntuple(d -> _inverse_kind(boundary, per[d]), Val(N)))
    norm = one(T) / prod(ntuple(d -> _kind_norm(T, dims[d], per[d]), Val(N)))
    λ = _bounded_symbol(scratch, T, dims, h, boundary, per)
    return BoundedState{T,typeof(scratch),typeof(fwd),typeof(inv),typeof(λ)}(
        fwd, inv, λ, scratch, norm,
    )
end

function HD.solve_poisson!(
    Φ::AbstractArray{T,N},
    RHS::AbstractArray{T,N},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,N},
    solver::CartesianBoundedSolver;
    boundary::HD.AbstractBoundaryCondition = HD.Neumann(),
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

function __init__()
    # Priority 10: FFTW's native real-to-real transform, against 20 for the generic even/odd
    # extension in the AbstractFFTs extension, which also works but builds a `2n` array.
    HD.register_spectral_solver!(SB.FFTSpectralBackend, CartesianSpectralSolver; priority = 5)
    HD.register_spectral_solver!(SB.FFTSpectralBackend, CartesianBoundedSolver; priority = 10)
end

end # module
