# Poisson solvers for `L Φ = f`, `L = D G` the operator FlowGeometries assembles as
# `Operators.LaplacianCoefficients`: symmetric and negative semidefinite under the cell measure, the
# boundary condition and the mask inside it through the face areas. The iterative solvers invert that
# `L`, so `D u_div = D u` holds to the solver tolerance. The spectral solvers (`SpectralSolvers.jl`,
# `Scattered.jl`) run on a library's transform and refuse the domains it does not diagonalize.

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

"""
    AbstractPoissonSolver

Supertype for Poisson solvers. Concrete subtypes implement

    solve_poisson!(Φ, RHS, grid, solver; boundary, kwargs...) -> SolverResult
"""
abstract type AbstractPoissonSolver end

"""
    solve_poisson!(Φ, RHS, grid, solver; boundary, coefficients, state, execution) -> SolverResult

Solve `L Φ = RHS` in place, `L` the operator `coefficients` holds.

`coefficients` and `state` are the reusable parts — the Hodge factors of every face, and whatever
the solver prepared for this `(grid, boundary)` pair. A decomposition solves `P + 1` right-hand
sides and a batch multiplies that by the field count, so both are built once and passed in.

Extensions add methods for their own solver types; the iterative [`CGSolver`](@ref) is the one that
works on any grid, mask and boundary condition.
"""
function solve_poisson! end

"""
    SolverResult{T}

Convergence diagnostics from a solve or fit: whether it met its tolerance, how many iterations it
took (`1` for a direct or spectral solve), and the final relative residual. A solve stopped short of
its tolerance returns what it reached with `converged == false`; the caller decides what to do with it.
"""
struct SolverResult{T<:AbstractFloat}
    converged::Bool
    iterations::Int
    residual::T
end

# The least converged of several fits: converged only if all are, the most iterations, the largest
# residual.
_worst(a::SolverResult{T}, b::SolverResult{T}) where {T} =
    SolverResult{T}(a.converged & b.converged, max(a.iterations, b.iterations),
                    max(a.residual, b.residual))

# ---------------------------------------------------------------------------
# Solver capability
# ---------------------------------------------------------------------------

"""
    supports_boundary(solver, boundary) -> Bool

Whether `solver` solves the problem `boundary` describes. A solver that does not is never
selected by [`AutoSolver`](@ref) and errors when named directly — solving a different problem
from the one asked for is indistinguishable from a correct answer at the call site.
"""
supports_boundary(::AbstractPoissonSolver, ::FlowGeometries.Operators.AbstractBoundaryCondition) = false

"""
    requires_full_domain(solver) -> Bool

Whether `solver` needs every cell active. The spectral solvers transform the whole array, masked
cells included, so they set this and a masked grid refuses them.
"""
requires_full_domain(::AbstractPoissonSolver) = false

"""
    requires_uniform_axes(solver) -> Bool

Whether `solver` needs constant spacing in every direction. An FFT-based solver does; FlowGeometries
reads the spacing from each axis's type, at no runtime cost.
"""
requires_uniform_axes(::AbstractPoissonSolver) = false

"""
    requires_periodic_domain(solver) -> Bool

Whether `solver` needs every direction to wrap. An FFT expands in a periodic basis, so it solves
the whole-torus problem and nothing else.

Whether a direction wraps is the grid's topology. A direction that wraps has no boundary to impose a
condition on, so on a grid this solver accepts `boundary` has no effect.
"""
requires_periodic_domain(::AbstractPoissonSolver) = false

"""
    supports_sampling(solver, grid) -> Bool

Whether `grid`'s node layout is one `solver` is defined on. A transform defined on a single node
set answers here.

[`AutoSolver`](@ref) selects on this, so it returns a `Bool` and never throws. A predicate that
throws makes selection raise on the first candidate it cannot use, which on a spherical grid
reaches every solver including the iterative one.
"""
supports_sampling(::AbstractPoissonSolver, grid) = true

"""
    _sampling_message(solver, grid) -> String

Why `solver` refuses `grid`'s node layout. Extensions override it to name the node set they need.
"""
_sampling_message(solver::AbstractPoissonSolver, grid) =
    "$(nameof(typeof(solver))) is not defined on this grid's node layout."

"""
    _require_sampling(solver, grid)

Throw unless [`supports_sampling`](@ref). A hook alongside [`_require_domain`](@ref), which checks
the mask, axis uniformity and topology.
"""
@inline _require_sampling(solver::AbstractPoissonSolver, grid) =
    supports_sampling(solver, grid) ? nothing :
    throw(ArgumentError(_sampling_message(solver, grid)))

"""
    inverts_discrete_laplacian(solver) -> Bool

Whether `solver` inverts the discrete `L = D G` the staggered decomposition differentiates with.
Inverting another operator leaves `D u_div ≠ D u` at truncation order, so
[`select_solver`](@ref) solves with such a solver only through [`CGSolver`](@ref) on `L`, preconditioned
by it, when it [`preconditions`](@ref) and refuses it otherwise. The spherical transforms divide by
`−ℓ(ℓ+1)/R²`, the continuous operator's eigenvalue.
"""
inverts_discrete_laplacian(::AbstractPoissonSolver) = false

"""
    preconditions(solver) -> Bool

Whether `solver`'s inverse of the continuous Laplacian can precondition conjugate gradients on the
discrete `L` of the grids it accepts. Extensions define [`spectral_inverse!`](@ref) for these.
"""
preconditions(::AbstractPoissonSolver) = false

"""
    spectral_inverse!(Φ, RHS, grid, solver, state) -> Φ

`Φ = Δ⁻¹ RHS` by `solver`'s transform, with the constant mode zero, through the state
[`prepare_solver`](@ref) built. An inexact inner fit is returned as it stands; a caller that needs it
converged checks.
"""
function spectral_inverse! end

"""
    supports_backend(solver, backend, grid) -> Bool

Whether `solver`'s state for a field on `grid` can live in the memory `backend` allocates, and its
transforms run there. [`AutoSolver`](@ref) passes over a solver that cannot, and one named directly
raises with [`_backend_message`](@ref).
"""
supports_backend(::AbstractPoissonSolver, backend, grid) = true

"""
    _backend_message(solver, backend) -> String

Why `solver` cannot run under `backend`. Solvers whose reason is not host-held state override it.
"""
_backend_message(solver::AbstractPoissonSolver, backend) =
    "$(nameof(typeof(solver))) holds its state in host memory and cannot run under " *
    "$(nameof(typeof(backend))). Plan it with a host `execution`, or solve with `CGSolver()`, whose " *
    "loops run on any backend."

@inline _require_backend(solver::AbstractPoissonSolver, backend, grid) =
    supports_backend(solver, backend, grid) ? nothing :
    throw(ArgumentError(_backend_message(solver, backend)))

"""
    library_loaded(solver) -> Bool

Whether the library `solver` transforms with is loaded. [`AutoSolver`](@ref) passes over a solver
whose library is not, and a solver named directly raises, naming the package to load.
"""
library_loaded(::AbstractPoissonSolver) = true

"""
    _unavailable_message(solver) -> String

Which package `solver` needs loaded. Solvers that run on a library override it.
"""
_unavailable_message(solver::AbstractPoissonSolver) =
    "$(nameof(typeof(solver))) needs a library that is not loaded."

@inline _require_library(solver::AbstractPoissonSolver) =
    library_loaded(solver) ? nothing : throw(ArgumentError(_unavailable_message(solver)))

"""
    prepare_solver(solver, grid, boundary; execution = SerialBackend(), shared) -> state

What `solver` builds once for a `(grid, boundary)` pair and reuses on every solve, handed back to
[`solve_poisson!`](@ref) as its `state`: transform plans and the buffers a solve writes, in the memory
`execution` allocates and at its thread count, with the read-only part `shared` from
[`prepare_shared`](@ref). `nothing` for a solver with nothing to reuse. A state holding a foreign plan
is released with [`close!`](@ref).
"""
prepare_solver(::AbstractPoissonSolver, grid, boundary;
               execution = ComputationalBackends.SerialBackend(), shared = nothing) = nothing

"""
    prepare_shared(solver, grid, boundary; coefficients) -> shared state

Whatever `solver` can build once for a `(grid, boundary)` pair and share across tasks: the part
that is read during a solve and never written, built for the operator `coefficients` describes.

`plan_helmholtz` holds this, so a batch builds it once in total. [`prepare_solver`](@ref) receives
it and allocates only what its task writes through. The multigrid hierarchy is the case that pays:
its grids and Galerkin coefficients are read-only, while its per-level vectors are not.

`nothing` by default.
"""
prepare_shared(::AbstractPoissonSolver, grid, boundary; coefficients = nothing) = nothing

@inline function _require_boundary(solver::AbstractPoissonSolver, boundary)
    supports_boundary(solver, boundary) && return nothing
    throw(ArgumentError(
        "$(nameof(typeof(solver))) cannot impose $(nameof(typeof(boundary))) boundary conditions. " *
        "Use an iterative solver for Dirichlet/Neumann problems on a bounded domain."))
end

"""
    _require_domain(solver, grid)

Throw unless `grid` is a domain `solver` can solve on. Reads the mask, so it is `O(N)` and is
called **once where a solver is chosen** — never from `solve_poisson!`, which a decomposition
enters `P + 1` times per field and once more per field of a batch.
"""
function _require_domain(solver::AbstractPoissonSolver, grid)
    if requires_full_domain(solver) && !all(FlowGeometries.Grids.mask(grid))
        msk = FlowGeometries.Grids.mask(grid)
        throw(ArgumentError(
            "$(nameof(typeof(solver))) transforms the whole domain and cannot honour a mask " *
            "($(count(!, msk)) of $(length(msk)) cells are inactive). Use an iterative solver, " *
            "which solves on the active cells only."))
    end
    if requires_uniform_axes(solver)
        for d in 1:ndims(grid)
            FlowGeometries.Grids.isuniform(grid, d) || throw(ArgumentError(
                "$(nameof(typeof(solver))) needs constant spacing, but direction $d of this grid " *
                "is stretched. Use a non-uniform transform or an iterative solver."))
        end
    end
    _require_sampling(solver, grid)
    if requires_periodic_domain(solver)
        for d in 1:ndims(grid)
            FlowGeometries.Grids.isperiodic(grid, d) || throw(ArgumentError(
                "$(nameof(typeof(solver))) divides by the Laplacian's symbol in a complex-exponential " *
                "basis, which diagonalizes the periodic Laplacian alone, and direction $d of this " *
                "grid is bounded. `CartesianBoundedSolver()` (FFTW) and " *
                "`CartesianRealTransformSolver()` (any FFT library) solve a bounded or channel grid " *
                "by cosine and sine transforms, and `CGSolver()` any grid."))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Automatic selection
# ---------------------------------------------------------------------------

"""
    AutoSolver()

Sentinel for automatic selection. It chooses on capability: geometry, node layout, mask, axis
uniformity, the boundary condition, the execution backend and which libraries are loaded. For a
staggered decomposition it takes the first of [`_auto_candidates`](@ref) that applies; failing that,
[`CGSolver`](@ref) preconditioned by the first of [`_preconditioner_candidates`](@ref) that applies;
failing that, `CGSolver()`.
"""
struct AutoSolver <: AbstractPoissonSolver end

function _applicable(solver::AbstractPoissonSolver, grid, boundary,
                     backend = ComputationalBackends.SerialBackend())
    library_loaded(solver) || return false
    supports_backend(solver, backend, grid) || return false
    supports_boundary(solver, boundary) || return false
    requires_full_domain(solver) && !all(FlowGeometries.Grids.mask(grid)) && return false
    if requires_uniform_axes(solver)
        all(d -> FlowGeometries.Grids.isuniform(grid, d), 1:ndims(grid)) || return false
    end
    supports_sampling(solver, grid) || return false
    if requires_periodic_domain(solver)
        all(d -> FlowGeometries.Grids.isperiodic(grid, d), 1:ndims(grid)) || return false
    end
    return true
end

function _resolve_auto_solver(grid::FlowGeometries.Grids.AbstractGrid, boundary;
                              execution = ComputationalBackends.SerialBackend())
    for candidate in _auto_candidates(grid)
        _applicable(candidate, grid, boundary, execution) && return candidate
    end
    # A device has a NUFFT library for its arrays or not, which shows when a plan is built, while
    # multigrid runs on every device.
    execution isa _HostBackend || return CGSolver()
    for candidate in _preconditioner_candidates(grid)
        _applicable(candidate, grid, boundary, execution) && return CGSolver(; spectral = candidate)
    end
    return CGSolver()
end

"""
    select_solver(solver, grid, boundary; execution = SerialBackend()) -> AbstractPoissonSolver

The concrete solver for one staggered decomposition's Poisson problems on `grid` under `execution`,
resolved and validated once. Resolution reads the mask, so a caller with several right-hand sides on one
grid calls this once and hands the result to each solve.

A solver that does not [`invert the discrete Laplacian`](@ref inverts_discrete_laplacian) and
[`preconditions`](@ref) is returned as `CGSolver(; spectral = solver)`, so the solve is of `L`; one
that does neither is refused, as is one whose state cannot live where `execution` runs.
"""
function select_solver(solver::AbstractPoissonSolver, grid, boundary;
                       execution = ComputationalBackends.SerialBackend())
    concrete = solver isa AutoSolver ? _resolve_auto_solver(grid, boundary; execution = execution) :
               solver
    _require_library(concrete)
    _require_boundary(concrete, boundary)
    _require_domain(concrete, grid)
    _require_backend(concrete, execution, grid)
    inverts_discrete_laplacian(concrete) && return _require_spectral(concrete, grid, boundary)
    preconditions(concrete) && return CGSolver(; spectral = concrete)
    throw(ArgumentError(
        "$(nameof(typeof(concrete))) has no discrete Laplacian of this grid to invert or " *
        "precondition. Decompose with it through `helmholtz_decompose_spectral`, or plan with " *
        "`CGSolver()` or `AutoSolver()`."))
end

# A conjugate-gradient solver's spectral preconditioner has to apply to the grid it solves on.
_require_spectral(s::AbstractPoissonSolver, grid, boundary) = s

# The solver a dual grid takes when the primal's was `solver`, the dual's condition `bc`. A transform
# that does not apply to the dual's nodes or condition gives way to conjugate gradients with the
# default preconditioner: either way the solve is of the dual `L`.
function _dual_request(solver::AbstractPoissonSolver, dual, bc, backend)
    solver isa AutoSolver && return solver
    s = solver isa CGSolver ? solver.spectral : solver
    s === nothing && return solver
    _applicable(s, dual, bc, backend) && return solver
    return solver isa CGSolver ? CGSolver(solver.max_iter, solver.rtol, solver.multigrid, nothing) :
           CGSolver()
end

# ---------------------------------------------------------------------------
# Iterative solvers on L
# ---------------------------------------------------------------------------

"""
    lazy_axis_sum(v::NTuple{N,AbstractArray}) -> array or Broadcasted

A quantity that is a sum of one term per axis, held as `N` vectors each reshaped along its own
axis, and added where it is read.

The Laplacian's Fourier symbol has this shape: `λ[I] = Σ_d λ_d(I[d])`. Held that way it costs
`∑ n_d` numbers, and the sum fuses into the broadcast that divides by it, so no spectral-sized
array of symbol values is stored or read.
"""
@inline lazy_axis_sum(v::NTuple{1,Any}) = @inbounds v[1]
@inline lazy_axis_sum(v::NTuple{N,Any}) where {N} = Broadcast.broadcasted(+, v...)

"""
    CGSolver(; max_iter = 1000, rtol = nothing, multigrid = true, spectral = nothing)

Conjugate gradients on `−L`, which is symmetric positive (semi)definite by construction (see
`FlowGeometries.Operators.LaplacianCoefficients`). The preconditioner is a multigrid V-cycle, Jacobi with `multigrid = false`, or with
`spectral` a transform solver that [`preconditions`](@ref), that transform's inverse of the continuous
Laplacian (see [`SpectralPreconditioner`](@ref)); `spectral` takes precedence over `multigrid`. On a
singular problem the iterate and residual are kept orthogonal to `L`'s null space.

Under a spectral preconditioner the step is the flexible form `β = −⟨z₊, A p⟩/⟨p, A p⟩` (Notay,
*SIAM J. Sci. Comput.* 22 (2000) 1444–1460), which converges when the preconditioner is not exactly
symmetric in the cell-measure inner product or is an inexact inner fit.

The solve stops when the residual falls to `rtol` of the right-hand side, both measured in the
cell-measure norm; `rtol = nothing` is `100·eps(T)` for the field's element type `T`. Works on any
grid, mask and boundary condition.
"""
struct CGSolver{R<:Union{Nothing,AbstractFloat},S<:Union{Nothing,AbstractPoissonSolver}} <:
       AbstractPoissonSolver
    max_iter::Int
    rtol::R
    multigrid::Bool
    spectral::S
end

CGSolver(; max_iter::Int = 1000, rtol::Union{Nothing,AbstractFloat} = nothing,
         multigrid::Bool = true, spectral::Union{Nothing,AbstractPoissonSolver} = nothing) =
    CGSolver(max_iter, rtol, multigrid, spectral)

@inline _rtol(s::CGSolver, ::Type{T}) where {T} = s.rtol === nothing ? 100 * eps(T) : T(s.rtol)

# Cell by cell, so every condition on a bounded direction is honoured, and a mask with it.
supports_boundary(::CGSolver, ::FlowGeometries.Operators.AbstractBoundaryCondition) = true
inverts_discrete_laplacian(::CGSolver) = true
supports_backend(s::CGSolver, backend, grid) =
    s.spectral === nothing || supports_backend(s.spectral, backend, grid)
_backend_message(s::CGSolver, backend) = _backend_message(s.spectral, backend)

function _require_spectral(s::CGSolver, grid, boundary)
    p = s.spectral
    p === nothing && return s
    preconditions(p) || throw(ArgumentError(
        "$(nameof(typeof(p))) cannot precondition CGSolver: it has no inverse of the continuous " *
        "Laplacian on this grid's nodes."))
    _require_library(p)
    _require_boundary(p, boundary)
    _require_domain(p, grid)
    return s
end

"""
    SpectralPreconditioner{S,ST,A,LV,T}

`z ≈ A⁻¹ r` for `A = −L` by a two-level cycle: `ν` sweeps of damped Jacobi (`ω`) from `z = 0`, the
spectral correction of the residual `t = r − A z`, then `ν` sweeps more. The correction is a
transform's inverse of the continuous Laplacian,

    M t = −Δ⁻¹(t − t̄) + t̄/λ₀,    t̄ = ⟨t, 1⟩/⟨1, 1⟩,    λ₀ = ⟨1, A 1⟩/⟨1, 1⟩

in the cell-measure inner product. The transform resolves the smooth modes, where `Δ` and `L`
agree; the sweeps damp the grid-scale ones, which a fit with fewer coefficients than nodes does not
carry. The transform zeroes the constant; `t̄/λ₀` restores it where a Dirichlet edge couples it, and
is absent (`λ₀ = 0`) where `L` is singular.
"""
struct SpectralPreconditioner{S<:AbstractPoissonSolver,ST,A<:AbstractArray,LV,T<:AbstractFloat}
    solver::S
    state::ST
    residual::A     # t
    correction::A   # M t
    scratch::A      # L z inside the sweeps
    level::LV       # the coefficients the sweeps relax with
    λ0::T
    ω::T
    ν::Int
end

# `λ₀ = ⟨1, −L 1⟩/⟨1, 1⟩`, zero where `L` annihilates the constants.
function _constant_rayleigh(c::FlowGeometries.Operators.LaplacianCoefficients, ::Type{T}) where {T}
    c.singular && return zero(T)
    dims = FlowGeometries.Grids.size_tuple(c.grid)
    L1 = FlowGeometries.Operators.apply_laplacian!(zeros(T, dims), fill(one(T), dims), c)
    return -T(sum(L1 .* c.measure)) / c.total
end

function _precondition!(z, r, grid, c::FlowGeometries.Operators.LaplacianCoefficients{N,T},
                        pre::SpectralPreconditioner, ::Any;
                        execution = ComputationalBackends.SerialBackend()) where {N,T}
    t, Mt, s = pre.residual, pre.correction, pre.scratch
    lb = loop_backend(execution, length(z))
    smooth_from_zero!(z, r, pre.level, s, pre.ω, pre.ν; execution = execution)
    FlowGeometries.Operators.apply_laplacian!(s, z, c; execution = lb)
    @. t = r + s                                        # r − A z with A = −L
    meas = c.measure
    tsum = FlowGeometries.Execution.reduce_indices(+, zero(T), length(t), lb) do lin
        @inbounds t[lin] * meas[lin]
    end
    t̄ = tsum / c.total
    @. t = t - t̄
    spectral_inverse!(Mt, t, grid, pre.solver, pre.state)
    shift = iszero(pre.λ0) ? zero(T) : t̄ / pre.λ0
    @. z += shift - Mt
    smooth!(z, r, pre.level, s, pre.ω, pre.ν; execution = execution)
    return z
end

# `β` of the next search direction from `⟨r, z⟩` and `⟨z, A p⟩`: Fletcher–Reeves under a fixed symmetric
# preconditioner, the flexible form under a spectral one, which alone needs `⟨z, A p⟩`.
@inline _cg_beta(::Any, zAp, pAp, rz_new, rz) = rz_new / rz
@inline _cg_beta(::SpectralPreconditioner, zAp, pAp, rz_new, rz) = -zAp / pAp
@inline _flexible(::Any) = Val(false)
@inline _flexible(::SpectralPreconditioner) = Val(true)

"""
    _dot_measure(a, b, grid, c) -> T

`⟨a, b⟩ = Σ a_I b_I V_I` over the active cells.

`L = D G` with `D = −G*` taken against the cell measure, so `L` is self-adjoint under this inner
product, the one conjugate gradients minimizes in.
"""
function _dot_measure(a, b, grid, c::FlowGeometries.Operators.LaplacianCoefficients{N,T};
                      execution = ComputationalBackends.SerialBackend()) where {N,T}
    # `measure` is zero on inactive cells, so no mask term is needed; each cell reads only its own
    # entries, so the index is linear.
    meas = c.measure
    return FlowGeometries.Execution.reduce_indices(+, zero(T), length(a), execution) do lin
        @inbounds a[lin] * b[lin] * meas[lin]
    end
end

"""
    CGWorkspace{A}

The four vectors a conjugate-gradient iteration needs, held so the solve allocates nothing.
"""
struct CGWorkspace{A}
    r::A
    p::A
    Ap::A
    z::A
end

"""
    CGState{W,P}

Everything `CGSolver` reuses across solves on one `(grid, boundary)`: its vectors and its
preconditioner.

`helmholtz_decompose!` solves `P + 1` right-hand sides per field, and every one of those solves reuses
the vectors built here.
"""
struct CGState{W<:CGWorkspace,P,B}
    workspace::W
    preconditioner::P    # the multigrid hierarchy (shared), a SpectralPreconditioner, or nothing
    mgbuffers::B         # per task: the vectors a cycle writes
end

# `L` on `grid` under `boundary`, from its face metrics.
_laplacian(grid, boundary) =
    FlowGeometries.Operators.laplacian_coefficients(FlowGeometries.Operators.face_metrics(grid, boundary))

# Read-only during a solve, so it goes on the plan and every task reads the same one: the multigrid
# hierarchy, or a spectral preconditioner's `λ₀`.
function prepare_shared(solver::CGSolver, grid, boundary::FlowGeometries.Operators.AbstractBoundaryCondition;
                        coefficients = _laplacian(grid, boundary))
    solver.spectral === nothing || return (_constant_rayleigh(coefficients, eltype(grid)), coefficients)
    return solver.multigrid ? multigrid(coefficients) : nothing
end

function prepare_solver(solver::CGSolver, grid, boundary::FlowGeometries.Operators.AbstractBoundaryCondition;
                        execution = ComputationalBackends.SerialBackend(),
                        shared = prepare_shared(solver, grid, boundary))
    T = eltype(grid)
    dims = size(grid)
    ws = CGWorkspace(_zeros(execution, T, dims), _zeros(execution, T, dims),
                     _zeros(execution, T, dims), _zeros(execution, T, dims))
    if solver.spectral !== nothing
        λ0, c = shared
        st = prepare_solver(solver.spectral, grid, boundary; execution = execution)
        # The sweep weight and count of the multigrid smoother (see `multigrid`).
        pre = SpectralPreconditioner(solver.spectral, st, _zeros(execution, T, dims),
                                     _zeros(execution, T, dims), _zeros(execution, T, dims), c, T(λ0),
                                     T(0.8), 2)
        return CGState(ws, pre, nothing)
    end
    bufs = shared === nothing ? nothing : multigrid_buffers(shared; execution = execution)
    return CGState(ws, shared, bufs)
end

function solve_poisson!(
    Φ::AbstractArray{T,N}, RHS::AbstractArray{T,N},
    grid::FlowGeometries.Grids.FaceIndexedGrid{T,G,N}, solver::CGSolver;
    boundary::FlowGeometries.Operators.AbstractBoundaryCondition,
    coefficients::FlowGeometries.Operators.LaplacianCoefficients = _laplacian(grid, boundary),
    state::CGState = prepare_solver(solver, grid, boundary),
    execution = ComputationalBackends.SerialBackend(),
    kwargs...,
) where {G,T,N}
    _require_boundary(solver, boundary)
    c = coefficients
    ws = state.workspace
    r, p, Ap, z = ws.r, ws.p, ws.Ap, ws.z
    rtol = _rtol(solver, T)
    # The fine-grid loops; the preconditioner sizes each of its levels' loops itself.
    lb = loop_backend(execution, length(Φ))

    pre = state.preconditioner
    flex = _flexible(pre)

    # Solve A Φ = b with A = −L, which is the positive-definite orientation.
    fill!(Φ, zero(T))
    @. r = -RHS
    FlowGeometries.Operators.project_out_constant!(r, c; execution = lb)
    bnorm = sqrt(_dot_measure(r, r, grid, c; execution = lb))
    bnorm == 0 && return SolverResult{T}(true, 0, zero(T))

    _precondition!(z, r, grid, c, pre, state.mgbuffers; execution = execution)
    rz, z̄, _ = _cg_dots!(r, z, Ap, zero(T), c, lb, Val(false))
    _cg_direction!(p, z, z̄, zero(T), c, lb)

    rnorm = bnorm
    for iter in 1:solver.max_iter
        FlowGeometries.Operators.apply_laplacian!(_negate, Ap, p, c; execution = lb)   # A p, A = −L
        pAp = _dot_measure(p, Ap, grid, c; execution = lb)
        # A zero curvature means `p` lies in the null space, and no step reduces the residual.
        if iszero(pAp)
            FlowGeometries.Operators.project_out_constant!(Φ, c; execution = lb)
            return SolverResult{T}(rnorm <= rtol * bnorm, iter, rnorm / bnorm)
        end
        α = rz / pAp
        rr, r̄ = _cg_step!(Φ, r, p, Ap, α, c, lb)
        rnorm = sqrt(rr)
        rnorm <= rtol * bnorm && begin
            FlowGeometries.Operators.project_out_constant!(Φ, c; execution = lb)
            return SolverResult{T}(true, iter, rnorm / bnorm)
        end
        _precondition!(z, r, grid, c, pre, state.mgbuffers; execution = execution)
        rz_new, z̄, zAp = _cg_dots!(r, z, Ap, r̄, c, lb, flex)
        _cg_direction!(p, z, z̄, _cg_beta(pre, zAp, pAp, rz_new, rz), c, lb)
        rz = rz_new
    end
    FlowGeometries.Operators.project_out_constant!(Φ, c; execution = lb)
    return SolverResult{T}(false, solver.max_iter, rnorm / bnorm)
end

@inline _negate(_I, L) = -L
@inline _tadd(a::Tuple, b::Tuple) = map(+, a, b)

# Whether `L`'s null space is one closed component holding every active cell. Its projection is then a
# mean, which the passes below fold into the updates; several components are projected on their own.
@inline _one_component(c) = c.singular && c.nullspace.full

# `Φ += α p` and `r −= α A p` in one pass, returning `‖r − r̄‖²` and `r̄`, the mean of `r` on the closed
# component. Under one component `r` keeps `r̄`, which `_cg_dots!` subtracts on its next pass, and
# `‖r − r̄‖² = ⟨r, r⟩ − r̄ ⟨r, 1⟩`; under several `r` is projected here.
function _cg_step!(Φ, r, p, Ap, α::T, c::FlowGeometries.Operators.LaplacianCoefficients, lb) where {T}
    meas = c.measure
    if _one_component(c)
        rr, rs = FlowGeometries.Execution.reduce_indices(_tadd, (zero(T), zero(T)), length(r), lb) do i
            @inbounds begin
                Φ[i] += α * p[i]
                ri = r[i] - α * Ap[i]
                r[i] = ri
                rm = ri * meas[i]
                return (ri * rm, rm)
            end
        end
        r̄ = rs / c.total
        return max(rr - r̄ * rs, zero(T)), r̄
    end
    rr = FlowGeometries.Execution.reduce_indices(+, zero(T), length(r), lb) do i
        @inbounds begin
            Φ[i] += α * p[i]
            ri = r[i] - α * Ap[i]
            r[i] = ri
            return ri * ri * meas[i]
        end
    end
    c.singular || return rr, zero(T)
    FlowGeometries.Operators.project_out_constant!(r, c; execution = lb)
    return _dot_measure(r, r, nothing, c; execution = lb), zero(T)
end

# `⟨r, z − z̄⟩`, `z̄` and, for the flexible step, `⟨z − z̄, A p⟩`, with `z̄` the mean of the preconditioned
# residual on a single closed component (`_cg_direction!` subtracts it). The same pass subtracts `r̄`
# from `r`. Under several components `z` is projected here.
function _cg_dots!(r, z, Ap, r̄::T, c::FlowGeometries.Operators.LaplacianCoefficients, lb,
                   ::Val{F}) where {T,F}
    meas = c.measure
    if _one_component(c)
        rz, zs, rs, zA, As = FlowGeometries.Execution.reduce_indices(_tadd, ntuple(_ -> zero(T), Val(5)),
                                                                     length(r), lb) do i
            @inbounds begin
                m = meas[i]
                ri = iszero(m) ? r[i] : r[i] - r̄
                r[i] = ri
                zm = z[i] * m
                a = F ? Ap[i] : zero(T)
                return (ri * zm, zm, ri * m, zm * a, a * m)
            end
        end
        z̄ = zs / c.total
        return rz - z̄ * rs, z̄, zA - z̄ * As
    end
    c.singular && FlowGeometries.Operators.project_out_constant!(z, c; execution = lb)
    rz, zA = FlowGeometries.Execution.reduce_indices(_tadd, (zero(T), zero(T)), length(r), lb) do i
        @inbounds begin
            zm = z[i] * meas[i]
            return (r[i] * zm, F ? zm * Ap[i] : zero(T))
        end
    end
    return rz, zero(T), zA
end

# `p = (z − z̄) + β p` on the active cells; `β = 0` starts a direction and does not read `p`.
function _cg_direction!(p, z, z̄::T, β::T, c::FlowGeometries.Operators.LaplacianCoefficients, lb) where {T}
    meas = c.measure
    if iszero(z̄)
        FlowGeometries.Execution.run_indices(length(p), lb) do i
            @inbounds p[i] = iszero(β) ? z[i] : z[i] + β * p[i]
            return nothing
        end
    else
        FlowGeometries.Execution.run_indices(length(p), lb) do i
            @inbounds begin
                zi = iszero(meas[i]) ? z[i] : z[i] - z̄
                p[i] = iszero(β) ? zi : zi + β * p[i]
            end
            return nothing
        end
    end
    return p
end

"""
    _precondition!(z, r, grid, c, preconditioner, buffers)

Apply `M⁻¹` to the residual: the Jacobi diagonal where there is no preconditioner, a V-cycle where
there is (`Multigrid.jl` adds that method, alongside the type).

`preconditioner` is shared across tasks and `buffers` are the task's own — see
[`prepare_shared`](@ref).
"""
_precondition!(z, r, grid, c, ::Nothing, ::Any;
               execution = ComputationalBackends.SerialBackend()) =
    _jacobi_precondition!(z, r, grid, c; execution = execution)

function _jacobi_precondition!(z, r, grid, c::FlowGeometries.Operators.LaplacianCoefficients{N,T};
                               execution = ComputationalBackends.SerialBackend()) where {N,T}
    d = c.diag
    # Linear, as in `smooth!`: a cell reads only its own diagonal entry.
    FlowGeometries.Execution.run_indices(length(z), execution) do lin
        @inbounds begin
            dI = d[lin]
            z[lin] = iszero(dI) ? zero(T) : r[lin] / -dI
        end
        return nothing
    end
    return z
end

