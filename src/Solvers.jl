"""
    Solvers.jl — Poisson solvers for `L Φ = f`.

`L` is the operator `Operators.jl` builds: `L = D G` with `D = −G*`, symmetric and negative
semidefinite under the cell-measure inner product, with the boundary condition and the mask
already inside it through the face areas. Every solver here inverts *that* `L` — the same one
the decomposition differentiates with — which is what makes `div(u_div) = div(u)` hold exactly
rather than to truncation order.

Written as a flux balance, `L`'s row at cell `I` is

    (L Φ)_I = (1/V_I) Σ_f c_f (Φ_nbr(f) − Φ_I),      c_f = A_f / g_f

over the faces of `I`, where a face with `A_f = 0` — masked, or a no-flux domain edge — simply
does not appear, and a Dirichlet edge appears with `Φ_nbr = 0`. Both the relaxation below and
the Krylov/multigrid solvers read the same coefficients.

Spectral solvers (FFTW, FINUFFT, FastSphericalHarmonics, NUFSHT) arrive through package
extensions and invert the continuous symbol instead; they are exact on the domains they apply
to and refuse the ones they do not.
"""

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
    solve_poisson!(Φ, RHS, grid, solver; boundary, coefficients, state, backend) -> SolverResult

Solve `L Φ = RHS` in place, with `L` the operator `Operators.jl` builds.

`coefficients` and `state` are the reusable parts — the Hodge factors of every face, and whatever
the solver prepared for this `(grid, boundary)` pair. A decomposition solves `P + 1` right-hand
sides and a batch multiplies that by the field count, so both are built once and passed in rather
than derived here.

Extensions add methods for their own solver types; the iterative [`CGSolver`](@ref) is the one that
works on any grid, mask and boundary condition.
"""
function solve_poisson! end

"""
    SolverResult{T}

Convergence diagnostics from a solve: whether it met its tolerance, how many iterations it took
(`1` for a direct or spectral solve), and the final residual.
"""
struct SolverResult{T<:AbstractFloat}
    converged::Bool
    iterations::Int
    residual::T
end

# ---------------------------------------------------------------------------
# Solver capability
# ---------------------------------------------------------------------------

"""
    supports_boundary(solver, boundary) -> Bool

Whether `solver` solves the problem `boundary` describes. A solver that does not is never
selected by [`AutoSolver`](@ref) and errors when named directly — solving a different problem
from the one asked for is indistinguishable from a correct answer at the call site.
"""
supports_boundary(::AbstractPoissonSolver, ::AbstractBoundaryCondition) = false

"""
    requires_full_domain(solver) -> Bool

Whether `solver` needs every cell active. The spectral solvers transform the whole array, so a
masked cell would be transformed as though it held data; they set this and are refused a masked
grid rather than returning a field that is wrong wherever the mask bites.
"""
requires_full_domain(::AbstractPoissonSolver) = false

"""
    requires_uniform_axes(solver) -> Bool

Whether `solver` needs constant spacing in every direction. An FFT-based solver does; the
question is answered from the axis TYPE by `FlowGeometries`, at no runtime cost.
"""
requires_uniform_axes(::AbstractPoissonSolver) = false

"""
    requires_periodic_domain(solver) -> Bool

Whether `solver` needs every direction to wrap. An FFT expands in a periodic basis, so it solves
the whole-torus problem and nothing else.

This is a question about the **grid's topology**, not about a boundary condition — which is why
there is no `Periodic` boundary condition to ask instead. A direction that wraps has no boundary
to impose a condition on, so on a grid this solver accepts, `boundary` is vacuous rather than
honoured or refused.
"""
requires_periodic_domain(::AbstractPoissonSolver) = false

"""
    _require_sampling(solver, grid)

Extra, solver-specific demand on the node layout — a transform defined on one node set and no
other says so here. A hook rather than an override of [`_require_domain`](@ref): overriding that
would silently drop the mask, uniformity and topology checks it also performs.
"""
_require_sampling(::AbstractPoissonSolver, grid) = nothing

"""
    prepare_solver(solver, grid, boundary) -> state

Whatever `solver` can compute once for a `(grid, boundary)` pair and reuse on every solve —
returned opaquely and handed back to [`solve_poisson!`](@ref) as its `state` keyword.

This exists because the transform plans are the dominant per-call cost and depend on nothing that
changes between calls. A decomposition solves `P + 1` right-hand sides on one grid and a batch
multiplies that by the number of fields, so a solver planning inside `solve_poisson!` rebuilds the
same plan `(P + 1) · B` times: in 3-D that is 17 FFTW plans per field, and for the non-uniform
transforms a plan *and* a node upload per conjugate-gradient iteration.

`nothing` is the default and means "nothing to reuse", which is correct for the iterative solvers —
their reusable part is the `LaplacianCoefficients`, which the plan already holds.
"""
prepare_solver(::AbstractPoissonSolver, grid, boundary) = nothing

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
                "$(nameof(typeof(solver))) inverts the Laplacian by dividing by its symbol in a " *
                "complex-exponential basis, and that basis diagonalizes the *periodic* Laplacian. " *
                "Direction $d of this grid is bounded, so the result would solve the periodic " *
                "problem instead — smoothly, plausibly, and to machine precision, which is what " *
                "makes it worth refusing. Nothing is wrong with transforming this data; only with " *
                "the divide. Bounded domains are solved here by the iterative solver, and by a " *
                "cosine (Neumann) or sine (Dirichlet) transform once those are in — those " *
                "diagonalize the bounded Laplacian and are equally O(N log N)."))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Automatic selection
# ---------------------------------------------------------------------------

"""
    AutoSolver()

Sentinel for automatic selection. The only thing permitted to choose a solver on the caller's
behalf, and it chooses on real capability — geometry, node layout, mask, axis uniformity, the
requested boundary condition, and which extensions are loaded.
"""
struct AutoSolver <: AbstractPoissonSolver end

"""
    _SPECTRAL_SOLVERS

Which solvers implement each `SpectralBackends` algorithm, populated by the extensions.

Keyed on the algorithm **type** rather than a `Symbol`: which transform a solver performs is
exactly what `SpectralBackends` names, so that vocabulary is shared with the rest of the ecosystem
instead of being re-invented as bare names only this file knows how to read.

The value is a list because more than one extension can implement the same algorithm — `FFTW.r2r`
and the `AbstractFFTs` even/odd extension are both the bounded FFT. They are ordered by an explicit
priority, not by load order, so which one runs does not depend on which package was imported first.
"""
# `Type`, not `Type{<:AbstractPoissonSolver}`: a solver parameterised on its tolerance type is a
# `UnionAll`, which that bound rejects.
const _SPECTRAL_SOLVERS =
    Dict{Type{<:SpectralBackends.AbstractSpectralBackend},Vector{Tuple{Int,Type}}}()

"""
    register_spectral_solver!(algorithm, solver_type; priority)

Declare that `solver_type` implements `algorithm`. Lower `priority` is tried first: a native
implementation takes a lower number than a generic one that would also work.
"""
function register_spectral_solver!(algorithm::Type{<:SpectralBackends.AbstractSpectralBackend},
                                   solver_type::Type{<:AbstractPoissonSolver}; priority::Int)
    entries = get!(() -> Tuple{Int,Type{<:AbstractPoissonSolver}}[], _SPECTRAL_SOLVERS, algorithm)
    filter!(e -> e[2] !== solver_type, entries)     # idempotent across reloads
    push!(entries, (priority, solver_type))
    sort!(entries; by = first)
    return nothing
end

"""
    _spectral_algorithms(geometry_type) -> Tuple

The algorithms worth trying on a geometry, in preference order: the uniform transform first, then
the non-uniform one, which subsumes it at greater cost. A candidate that does not apply refuses
itself through its own capability check, so this order only decides between several that do.
"""
_spectral_algorithms(::Type{<:FlowGeometries.Geometry.AbstractSphericalGeometry}) =
    (SpectralBackends.FSHTSpectralBackend, SpectralBackends.NUFSHTSpectralBackend)
_spectral_algorithms(::Type{<:FlowGeometries.Geometry.AbstractCartesianGeometry}) =
    (SpectralBackends.FFTSpectralBackend, SpectralBackends.NUFFTSpectralBackend)
_spectral_algorithms(::Type{<:FlowGeometries.Geometry.AbstractGeometry}) = ()

function _applicable(solver::AbstractPoissonSolver, grid, boundary)
    supports_boundary(solver, boundary) || return false
    requires_full_domain(solver) && !all(FlowGeometries.Grids.mask(grid)) && return false
    if requires_uniform_axes(solver)
        all(d -> FlowGeometries.Grids.isuniform(grid, d), 1:ndims(grid)) || return false
    end
    _require_sampling(solver, grid)
    if requires_periodic_domain(solver)
        all(d -> FlowGeometries.Grids.isperiodic(grid, d), 1:ndims(grid)) || return false
    end
    return true
end

function _resolve_auto_solver(grid::FlowGeometries.Grids.AbstractGrid{G}, boundary) where {G}
    for algorithm in _spectral_algorithms(G)
        entries = get(_SPECTRAL_SOLVERS, algorithm, nothing)
        entries === nothing && continue
        for (_, solver_type) in entries
            candidate = solver_type()
            _applicable(candidate, grid, boundary) && return candidate
        end
    end
    return CGSolver()
end

"""
    select_solver(solver, grid, boundary) -> AbstractPoissonSolver

The concrete solver for this problem, resolved and validated **once**. Resolution reads the
mask, so a caller with several right-hand sides on one grid — which a decomposition always has,
`P + 1` of them — calls this once and hands the result to each solve.
"""
function select_solver(solver::AbstractPoissonSolver, grid, boundary)
    concrete = solver isa AutoSolver ? _resolve_auto_solver(grid, boundary) : solver
    _require_boundary(concrete, boundary)
    _require_domain(concrete, grid)
    return concrete
end

# ---------------------------------------------------------------------------
# Iterative solvers on L
# ---------------------------------------------------------------------------

"""
    LaplacianCoefficients{N,T,A,B}

The per-face coefficients `c_f = A_f / g_f` of `L`, and the cell measures, evaluated once for a
(grid, boundary) pair.

Every iterative solve applies `L` repeatedly, and each application would otherwise re-evaluate
the geometry: face areas involve the scale factors at the face, which on a curved geometry means
trigonometry per cell per direction. That is invariant across iterations — and across the `P + 1`
potentials and the whole batch — so it is reduced once here.

`coef[I, d]` is the coefficient of the face **below** cell `I` along `d`; `diag[I]` is the
negative sum of a cell's own face coefficients, divided by its measure.
"""
struct LaplacianCoefficients{N,T,C<:NTuple{N,AbstractArray{T,N}},B<:AbstractArray{T,N}}
    coef::C          # coef[d][F] = A_F / g_F, the Hodge factor of face F
    diag::B          # −Σ over a cell's own faces, divided by its measure
    measure::B
    singular::Bool   # constants in the null space: no boundary removes them
end

"""
    laplacian_coefficients(grid, bc) -> LaplacianCoefficients

The Hodge factor `c_F = A_F / g_F` of every face, and the cell measures, reduced once for a
(grid, boundary) pair.

Every iterative solve applies `L` repeatedly, and each application would otherwise re-evaluate
the geometry — face areas involve the scale factors at the face, i.e. trigonometry per cell per
direction on a curved grid. That is invariant across iterations, across the `P + 1` potentials,
and across a whole batch, so it is computed once here and only read afterwards.
"""
function laplacian_coefficients(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, bc::AbstractBoundaryCondition,
) where {G,T,N}
    coef = ntuple(d -> zeros(T, face_dims(grid, d)), Val(N))
    diag = zeros(T, size(grid))
    meas = zeros(T, size(grid))
    @inbounds for d in 1:N
        cd = coef[d]
        for F in CartesianIndices(face_dims(grid, d))
            A = face_area(grid, F, d, bc, T)
            cd[F] = iszero(A) ? zero(T) : A / face_gap(grid, F, d, T)
        end
    end
    @inbounds for I in CartesianIndices(size(grid))
        FlowGeometries.Grids.isactive(grid, Tuple(I)...) || continue
        meas[I] = cell_measure(grid, I, T)
        acc = zero(T)
        for d in 1:N
            acc += coef[d][face_below(I, d)] + coef[d][face_above(grid, I, d)]
        end
        diag[I] = -acc / meas[I]
    end
    built = LaplacianCoefficients{N,T,typeof(coef),typeof(diag)}(coef, diag, meas, false)
    return LaplacianCoefficients{N,T,typeof(coef),typeof(diag)}(
        coef, diag, meas, _detect_singular(grid, built),
    )
end

"""
    _detect_singular(grid, c) -> Bool

Whether the constants lie in `L`'s null space, decided by applying `L` to the constant field and
seeing whether anything comes back.

Asking the boundary condition instead is not enough, and the dual grid is why: its boundary ring
of corners has no closed loop of cells around it, so those corners are masked out and every face
of the active region is closed — the operator there is singular **even under `Dirichlet`**, which
a condition-based test reports as nonsingular. Conjugate gradients on a singular system with no
null-space projection then drifts along the null space instead of converging, which showed up as a
harmonic fraction of `1.7e+32`.

`L·1 = 0` is exactly the property that matters and costs one operator application at plan time.
"""
function _detect_singular(grid, c::LaplacianCoefficients{N,T}) where {N,T}
    ones_ = fill(one(T), size(grid))
    out = similar(ones_)
    apply_laplacian!(out, ones_, grid, c)
    scale = zero(T)
    resid = zero(T)
    @inbounds for I in CartesianIndices(size(grid))
        FlowGeometries.Grids.isactive(grid, Tuple(I)...) || continue
        scale = max(scale, abs(c.diag[I]))
        resid = max(resid, abs(out[I]))
    end
    iszero(scale) && return true
    return resid <= sqrt(eps(T)) * scale
end

"""
    apply_laplacian!(out, Φ, grid, c) -> out

`out = L Φ`, from the prebuilt coefficients. A face at the outer edge of a bounded direction has
no cell beyond it and contributes `c·(0 − Φ_I)` — the zero ghost a Dirichlet condition places
there. Under Neumann that face's coefficient is zero and the term vanishes, so one expression
serves both.
"""
function apply_laplacian!(out, Φ, grid, c::LaplacianCoefficients{N,T}; backend = ComputationalBackends.SerialBackend()) where {N,T}
    msk = FlowGeometries.Grids.mask(grid)
    cart = CartesianIndices(size(grid))
    coef = c.coef
    meas = c.measure
    # One write per index. This is the innermost operation in the package — every conjugate-gradient
    # iteration and every multigrid smoothing sweep applies it — so collapsing `N + 2` passes into
    # one matters here more than anywhere else, and the same body is what a device launch needs.
    FlowGeometries.Execution.run_indices(length(out), backend) do lin
        @inbounds begin
            I = cart[lin]
            if msk[I]
                ΦI = Φ[I]
                acc = zero(T)
                for d in 1:N
                    cd = coef[d]
                    Fhi = face_above(grid, I, d)
                    Φlo = _cell_value(Φ, grid, I, d, false, T)
                    Φhi = _cell_value(Φ, grid, Fhi, d, true, T)
                    acc += cd[I] * (Φlo - ΦI) + cd[Fhi] * (Φhi - ΦI)
                end
                out[I] = acc / meas[I]
            end
        end
        return nothing
    end
    return out
end

"""
    CGSolver(; max_iter = 1000, rtol = 1e-10)

Conjugate gradients on `−L`, which is symmetric positive (semi)definite by construction — see
`Operators.jl`. Jacobi-preconditioned, and on a closed problem the iterate and residual are kept
orthogonal to the constants.

Works on any grid, any mask, any boundary condition. Stage 6 adds a multigrid preconditioner in
place of Jacobi; the outer iteration is unchanged by that.
"""
struct CGSolver{T<:AbstractFloat} <: AbstractPoissonSolver
    max_iter::Int
    rtol::T
    multigrid::Bool
end

CGSolver(; max_iter::Int = 1000, rtol::AbstractFloat = 1e-10, multigrid::Bool = true) =
    CGSolver(max_iter, rtol, multigrid)

# Cell by cell, so every condition on a bounded direction is honoured, and a mask with it.
supports_boundary(::CGSolver, ::AbstractBoundaryCondition) = true

"""
    _dot_measure(a, b, grid, c) -> T

`⟨a, b⟩ = Σ a_I b_I V_I` over the active cells.

`L` is self-adjoint under **this** inner product and not under the plain Euclidean one — it is
`D = −G*` with the adjoint taken against the cell measure. A Krylov method using the wrong inner
product is no longer minimising what it reports, so the weight is not optional.
"""
# `sum` of `f` applied elementwise, over the unmaterialised `Broadcasted`.
@inline function lazy_sum(f::F, args...) where {F}
    return sum(Broadcast.instantiate(Broadcast.broadcasted(f, args...)))
end

# `measure` is zero on inactive cells, so no mask term is needed.
_dot_measure(a, b, grid, c::LaplacianCoefficients) = lazy_sum(*, a, b, c.measure)

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

The vectors belong here and not in a default argument. `helmholtz_decompose!` solves `P + 1`
right-hand sides and a batch multiplies that by the field count, so defaulting them allocates four
grid arrays per solve for the lifetime of the program.
"""
struct CGState{W<:CGWorkspace,P}
    workspace::W
    preconditioner::P
end

function prepare_solver(solver::CGSolver, grid, boundary::AbstractBoundaryCondition)
    T = eltype(grid)
    ws = CGWorkspace(zeros(T, size(grid)), zeros(T, size(grid)),
                     zeros(T, size(grid)), zeros(T, size(grid)))
    pre = solver.multigrid ? multigrid(grid, boundary, T) : nothing
    return CGState(ws, pre)
end

function solve_poisson!(
    Φ::AbstractArray{T,N}, RHS::AbstractArray{T,N},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, solver::CGSolver;
    boundary::AbstractBoundaryCondition,
    coefficients::LaplacianCoefficients = laplacian_coefficients(grid, boundary),
    state::CGState = prepare_solver(solver, grid, boundary),
    backend = ComputationalBackends.SerialBackend(),
    kwargs...,
) where {G,T,N}
    _require_boundary(solver, boundary)
    c = coefficients
    ws = state.workspace
    r, p, Ap, z = ws.r, ws.p, ws.Ap, ws.z

    # Solve A Φ = b with A = −L, which is the positive-definite orientation.
    fill!(Φ, zero(T))
    @. r = -RHS
    project_out_constant!(r, grid, c)
    bnorm = sqrt(_dot_measure(r, r, grid, c))
    bnorm == 0 && return SolverResult{T}(true, 0, zero(T))

    _precondition!(z, r, grid, c, state.preconditioner; backend = backend)
    copyto!(p, z)
    rz = _dot_measure(r, z, grid, c)

    rnorm = bnorm
    for iter in 1:solver.max_iter
        apply_laplacian!(Ap, p, grid, c; backend = backend)
        @. Ap = -Ap
        pAp = _dot_measure(p, Ap, grid, c)
        # A zero curvature means `p` lies in the null space; with the projection above that only
        # happens once the residual is already exhausted.
        iszero(pAp) && return SolverResult{T}(true, iter, rnorm / bnorm)
        α = rz / pAp
        @. Φ += α * p
        @. r -= α * Ap
        c.singular && project_out_constant!(r, grid, c)
        rnorm = sqrt(_dot_measure(r, r, grid, c))
        rnorm <= solver.rtol * bnorm && begin
            c.singular && project_out_constant!(Φ, grid, c)
            return SolverResult{T}(true, iter, rnorm / bnorm)
        end
        _precondition!(z, r, grid, c, state.preconditioner; backend = backend)
        rz_new = _dot_measure(r, z, grid, c)
        @. p = z + (rz_new / rz) * p
        rz = rz_new
    end
    c.singular && project_out_constant!(Φ, grid, c)
    return SolverResult{T}(false, solver.max_iter, rnorm / bnorm)
end

"""
    _precondition!(z, r, grid, c, preconditioner)

Apply `M⁻¹` to the residual: the Jacobi diagonal when there is no preconditioner, a V-cycle when
there is (`Multigrid.jl` adds that method, where the type lives).
"""
_precondition!(z, r, grid, c, ::Nothing; backend = ComputationalBackends.SerialBackend()) = _jacobi_precondition!(z, r, grid, c)

function _jacobi_precondition!(z, r, grid, c::LaplacianCoefficients{N,T}) where {N,T}
    d = c.diag
    z .= ifelse.(iszero.(d), zero(T), r ./ .-d)
    return z
end

"""
    project_out_constant!(Φ, grid, c)

Remove the measure-weighted mean of `Φ` over the active cells.

A closed problem — every direction periodic, or a Neumann boundary — leaves the constants in
`L`'s null space. Krylov iterations must stay orthogonal to that null space or they drift along
it, so this is applied to the right-hand side once and to the iterate as it goes.
"""
function project_out_constant!(Φ, grid, c::LaplacianCoefficients{N,T}) where {N,T}
    c.singular || return Φ
    meas = c.measure
    weight = lazy_sum(identity, meas)
    iszero(weight) && return Φ
    m = lazy_sum(*, Φ, meas) / weight
    # Shifting an inactive cell would put `−m` where the operator keeps zero.
    Φ .= ifelse.(iszero.(meas), Φ, Φ .- m)
    return Φ
end
