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

`backend` is where the state's buffers are allocated. A solver that builds them with `zeros` hands
a kernel host memory it cannot reach, and a parity test run on `KernelAbstractions.CPU()` cannot
see the difference, since that backend's arrays are host arrays.
"""
prepare_solver(::AbstractPoissonSolver, grid, boundary;
               backend = ComputationalBackends.SerialBackend(), shared = nothing) = nothing

"""
    prepare_shared(solver, grid, boundary) -> shared state

Whatever `solver` can build once for a `(grid, boundary)` pair and **share across tasks**: the part
that is read during a solve and never written.

`plan_helmholtz` holds this, so a batch builds it once in total. [`prepare_solver`](@ref) receives
it and allocates only what its task writes through. The multigrid hierarchy is the case that pays:
its grids and Galerkin coefficients are read-only, while its per-level vectors are not, and
building the hierarchy dominates what a workspace costs.

`nothing` by default
"""
prepare_shared(::AbstractPoissonSolver, grid, boundary) = nothing

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
    supports_sampling(solver, grid) || return false
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
    LazyDiagonal{N,T,C,M,G} <: AbstractArray{T,N}

`L`'s diagonal read from the face coefficients: `diag[I] = −Σ_d (c_d[F_lo] + c_d[F_hi]) / V_I`,
zero where the measure is.
"""
struct LazyDiagonal{N,T,C,M,G} <: AbstractArray{T,N}
    coef::C
    measure::M
    grid::G
end

@inline LazyDiagonal{N,T}(coef::C, measure::M, grid::G) where {N,T,C,M,G} =
    LazyDiagonal{N,T,C,M,G}(coef, measure, grid)

Base.size(d::LazyDiagonal) = size(d.grid)
Base.IndexStyle(::Type{<:LazyDiagonal}) = IndexCartesian()

@inline function Base.getindex(d::LazyDiagonal{N,T}, I::Vararg{Int,N}) where {N,T}
    @boundscheck checkbounds(d, I...)
    @inbounds begin
        C = CartesianIndex(I)
        m = d.measure[C]
        iszero(m) && return zero(T)
        acc = zero(T)
        for e in 1:N
            cd = d.coef[e]
            acc += cd[face_below(C, e)] + cd[face_above(d.grid, C, e)]
        end
        return -acc / m
    end
end

"""
    LaplacianCoefficients{N,T,C,D,M}

The per-face coefficients `c_f = A_f / g_f` of `L`, and the cell measures, evaluated once for a
(grid, boundary) pair.

Every iterative solve applies `L` repeatedly, and each application would otherwise re-evaluate
the geometry: face areas involve the scale factors at the face, which on a curved geometry means
trigonometry per cell per direction. That is invariant across iterations — and across the `P + 1`
potentials and the whole batch — so it is reduced once here.

`coef[I, d]` is the coefficient of the face **below** cell `I` along `d`; `diag[I]` is the
negative sum of a cell's own face coefficients, divided by its measure.
"""
struct LaplacianCoefficients{N,T,C<:NTuple{N,AbstractArray{T,N}},D<:AbstractArray{T,N},
                             M<:AbstractArray{T,N}}
    coef::C          # coef[d][F] = A_F / g_F, the Hodge factor of face F
    diag::D          # −Σ over a cell's own faces, divided by its measure
    # `measure` and `diag` carry separate types: on an unmasked grid the measure is the grid's own
    # lazy per-axis factors, while the diagonal is a sum over faces and is held densely.
    measure::M
    # `Σ V_I` over the active cells. `project_out_constant!` divides by it on every conjugate-
    # gradient iteration, and it is a property of the grid.
    total::T
    singular::Bool   # constants in the null space: no boundary removes them
end

"""
    laplacian_coefficients(grid, bc, fm = face_metrics(grid, bc)) -> LaplacianCoefficients

The Hodge factor `c_F = A_F / g_F` of every face, and the cell measures, reduced once for a
(grid, boundary) pair.

Every iterative solve applies `L` repeatedly, and each application reading the geometry would
re-derive the scale factors at each face — trigonometry per cell per direction on a curved grid.
That is invariant across iterations, across the `P + 1` potentials, and across a whole batch, so it
is reduced here and only read afterwards.

`fm` carries the areas and gaps. Pass the same [`FaceMetrics`](@ref) the plan holds, and the
geometry is walked once for both.
"""
function laplacian_coefficients(
    grid::FaceIndexedGrid{T,G,N}, bc::AbstractBoundaryCondition,
    fm::FaceMetrics{N,T} = face_metrics(grid, bc),
) where {G,T,N}
    # A Hodge factor is an area over a gap, and `fm` holds both already. Evaluating the geometry
    # again here walked every face a second time, and on a curved grid that is trigonometry per
    # face. A gap is never zero, so a closed face's zero area carries straight through the ratio.
    #
    # A quotient of separable arrays is separable, so the coefficients take the storage the metrics
    # do; a quotient of dense ones is dense.
    coef = ntuple(d -> fm.area[d] ./ fm.gap[d], Val(N))
    meas = _measure_array(grid, T)
    diag = _diagonal(grid, coef, meas, T)
    # `sum` of a separable measure is the product of its per-axis sums, which `FlowGeometries`
    # specialises; on a masked grid the inactive entries are already zero.
    total = T(sum(meas))
    P = (N, T, typeof(coef), typeof(diag), typeof(meas))
    built = LaplacianCoefficients{P...}(coef, diag, meas, total, false)
    return LaplacianCoefficients{P...}(coef, diag, meas, total, _detect_singular(grid, built))
end

"""
    _diagonal(grid, coef, measure, T) -> AbstractArray{T,N}

`L`'s diagonal: a [`LazyDiagonal`](@ref) where the face coefficients are separable, a stored array
where they are dense.

Dense coefficients are `2N` grid-sized arrays already streamed by every operator application, and
reading the diagonal back out of them costs `2N` gathers against one from a stored copy. Separable
coefficients are per-axis factors that stay in cache, and the stored copy is the largest array the
plan holds.
"""
@inline _diagonal(grid, coef::NTuple{N,Array}, meas, ::Type{T}) where {N,T} =
    _dense_diagonal(grid, coef, meas, T)

@inline _diagonal(grid, coef::NTuple{N,Any}, meas, ::Type{T}) where {N,T} =
    LazyDiagonal{N,T}(coef, meas, grid)

function _dense_diagonal(grid, coef::NTuple{N,Any}, meas, ::Type{T}) where {N,T}
    diag = zeros(T, size(grid))
    @inbounds for I in CartesianIndices(size(grid))
        m = meas[I]
        iszero(m) && continue
        acc = zero(T)
        for d in 1:N
            acc += coef[d][face_below(I, d)] + coef[d][face_above(grid, I, d)]
        end
        diag[I] = -acc / m
    end
    return diag
end

"""
    _measure_array(grid, T) -> AbstractArray{T,N}

The cell measure `L` divides by: the grid's own per-axis factors where every cell is active, and a
copy zeroed on the inactive cells where a mask cuts some out.

`divergence!` already reads the grid's lazy measure while `apply_laplacian!` read a dense copy of
it. Both now read the same object on an unmasked grid.
"""
@inline _measure_array(grid, ::Type{T}) where {T} =
    _measure_array(grid, FlowGeometries.Grids.mask(grid), T)

@inline _measure_array(grid, ::FlowGeometries.Grids.AllActive, ::Type{T}) where {T} =
    FlowGeometries.Grids.measure(grid)

function _measure_array(grid, msk, ::Type{T}) where {T}
    m = zeros(T, size(grid))
    @inbounds for I in CartesianIndices(size(grid))
        FlowGeometries.Grids.isactive(grid, Tuple(I)...) && (m[I] = cell_measure(grid, I, T))
    end
    return m
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
    _apply_laplacian!(out, Φ, grid, FlowGeometries.Grids.mask(grid), c.coef, c.measure,
                      Val(N), T, backend)
    return out
end


"""
    _laplacian_at(Φ, grid, coef, meas, I, Val(N), T) -> T

`(L Φ)_I`: the flux balance over cell `I`'s faces, divided by its measure.

The one place the stencil is written. Both traversals below call it, so they cannot drift apart.
"""
@inline function _laplacian_at(Φ, grid, coef, meas, I::CartesianIndex{N},
                               v::Val{N}, ::Type{T}) where {N,T}
    @inbounds return _flux_sum(Φ, grid, coef, I, T(Φ[I]), v, T) / meas[I]
end

# The direction is a type parameter, so each term is compiled with `d` a literal: the grid's
# topology in that direction folds to a constant, its two branches with it, and `coef[d]` is a
# fixed slot of the tuple. Under a runtime `d` none of that is available, and the directions of a
# tuple of face arrays need not even share a type.
@inline _flux_sum(Φ, grid, coef, I::CartesianIndex{N}, ΦI, ::Val{0},
                  ::Type{T}) where {N,T} = zero(T)

@inline function _flux_sum(Φ, grid, coef, I::CartesianIndex{N}, ΦI,
                           ::Val{d}, ::Type{T}) where {N,d,T}
    @inbounds begin
        cd = coef[d]
        Fhi = face_above(grid, I, d)
        Φlo = _cell_value(Φ, grid, I, d, false, T)
        Φhi = _cell_value(Φ, grid, Fhi, d, true, T)
        here = cd[I] * (Φlo - ΦI) + cd[Fhi] * (Φhi - ΦI)
    end
    return here + _flux_sum(Φ, grid, coef, I, ΦI, Val(d - 1), T)
end

# One index per cell, which is the shape a device launch takes: adjacent threads read adjacent
# memory. Recovering `I` from a flat index costs an integer division per cell, and on a device that
# sits under the memory latency.
function _apply_laplacian!(out, Φ, grid, msk, coef, meas, v::Val{N}, ::Type{T},
                           backend) where {N,T}
    cart = CartesianIndices(size(grid))
    FlowGeometries.Execution.run_indices(length(out), backend) do lin
        @inbounds begin
            I = cart[lin]
            msk[I] && (out[I] = _laplacian_at(Φ, grid, coef, meas, I, v, T))
        end
        return nothing
    end
    return out
end

"""
    _laplacian_interior(Φ, coef, meas, I, Val(N), T) -> T

`(L Φ)_I` for a cell whose every coordinate is strictly inside the grid.

Such a cell has a neighbour on both sides in every direction, so no direction wraps and no face
carries the zero ghost: the topology and the edge tests drop out and each term is two reads at a
fixed offset. A mask does not change that — an inactive neighbour reaches this through a zero face
coefficient, never through the neighbour lookup — so this serves a masked grid as well.
"""
@inline function _laplacian_interior(Φ, coef, meas, I::CartesianIndex{N},
                                     v::Val{N}, ::Type{T}) where {N,T}
    @inbounds return _flux_interior(Φ, coef, I, T(Φ[I]), v, T) / meas[I]
end

@inline _flux_interior(Φ, coef, I::CartesianIndex{N}, ΦI, ::Val{0},
                       ::Type{T}) where {N,T} = zero(T)

@inline function _flux_interior(Φ, coef, I::CartesianIndex{N}, ΦI,
                                ::Val{d}, ::Type{T}) where {N,d,T}
    @inbounds begin
        e = _unit_axis(Val(N), Val(d))
        cd = coef[d]
        Ihi = I + e
        here = cd[I] * (Φ[I - e] - ΦI) + cd[Ihi] * (Φ[Ihi] - ΦI)
    end
    return here + _flux_interior(Φ, coef, I, ΦI, Val(d - 1), T)
end

# The direction is a type parameter, so every component of the offset is decided at compile time and
# the index is a constant. Handed the direction as a value, the comparison inside the tuple stays in
# the loop and the offset is rebuilt per cell, which stops the run from vectorising.
@inline _unit_axis(::Val{N}, ::Val{d}) where {N,d} =
    CartesianIndex(ntuple(i -> ifelse(i == d, 1, 0), Val(N)))

# Whether any of the coordinates of axes `2 … N` sits on its axis's first or last cell.
@inline function _rest_on_edge(rest::NTuple{M,Int}, dims::NTuple{N,Int}) where {M,N}
    @inbounds for e in 1:M
        (rest[e] == 1 || rest[e] == dims[e + 1]) && return true
    end
    return false
end

# Host arrays walk the index space itself, in slabs of the trailing axis. Stepping a
# `CartesianIndex` carries the coordinates along, so no cell pays the division the flat form does.
#
# A row whose other coordinates are all interior meets the boundary only at its two ends, so it
# splits into those two cells and a run that takes `_laplacian_interior`. That run is the whole grid
# apart from its surface.
function _apply_laplacian!(out::Array, Φ, grid, msk, coef, meas, v::Val{N}, ::Type{T},
                           backend) where {N,T}
    dims = size(grid)
    n1 = dims[1]
    mid = CartesianIndices(ntuple(e -> dims[e + 1], Val(N - 2)))
    FlowGeometries.Execution.run_chunks(dims[N], backend) do slab
        _laplacian_slab!(out, Φ, grid, msk, coef, meas, v, T, slab, dims, mid, n1)
    end
    return out
end

"""
    _laplacian_slab!(out, Φ, grid, msk, coef, meas, Val(N), T, slab, dims, mid, n1)

`L Φ` over one slab of the trailing axis.

It carries the loop so that `coef` arrives as an argument and the tuple rebuilt below belongs to the
same function. Where the coefficient arrays are reached through a closure capture, each cell
re-loads them and the interior run stays scalar.
"""
function _laplacian_slab!(out, Φ, grid, msk, coef, meas, v::Val{N}, ::Type{T},
                          slab, dims, mid, n1::Int) where {N,T}
    # Rebuilt here so the array pointers become values the loop carries for its whole length.
    cf = ntuple(d -> @inbounds(coef[d]), v)
    @inbounds for jN in slab, Im in mid
        rest = (Tuple(Im)..., jN)
        if n1 < 3 || _rest_on_edge(rest, dims)
            for i in 1:n1
                I = CartesianIndex(i, rest...)
                msk[I] && (out[I] = _laplacian_at(Φ, grid, cf, meas, I, v, T))
            end
        else
            for i in (1, n1)
                I = CartesianIndex(i, rest...)
                msk[I] && (out[I] = _laplacian_at(Φ, grid, cf, meas, I, v, T))
            end
            for i in 2:(n1 - 1)
                I = CartesianIndex(i, rest...)
                msk[I] && (out[I] = _laplacian_interior(Φ, cf, meas, I, v, T))
            end
        end
    end
    return nothing
end

# A one-dimensional grid has no leading axes to nest, so the trailing axis is the whole of it.
function _apply_laplacian!(out::Array, Φ, grid, msk, coef, meas, v::Val{1}, ::Type{T},
                           backend) where {T}
    FlowGeometries.Execution.run_chunks(size(grid, 1), backend) do slab
        @inbounds for i in slab
            I = CartesianIndex(i)
            msk[I] && (out[I] = _laplacian_at(Φ, grid, coef, meas, I, v, T))
        end
        return nothing
    end
    return out
end

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
# `measure` is zero on inactive cells, so no mask term is needed.
#
# Through `Execution.reduce_indices`, which has a method per backend, so the reduction runs where
# the rest of the iteration does. `sum` over a `Broadcasted` is serial whatever backend is asked
# for, and a conjugate-gradient step evaluates three of these.
function _dot_measure(a, b, grid, c::LaplacianCoefficients{N,T};
                      backend = ComputationalBackends.SerialBackend()) where {N,T}
    meas = c.measure
    # Linear: each cell contributes its own entry and reads no neighbour, so no `CartesianIndex`
    # has to be rebuilt. A lazy measure converts internally and is no worse than it was.
    return FlowGeometries.Execution.reduce_indices(+, zero(T), length(a), backend) do lin
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

The vectors belong here and not in a default argument. `helmholtz_decompose!` solves `P + 1`
right-hand sides and a batch multiplies that by the field count, so defaulting them allocates four
grid arrays per solve for the lifetime of the program.
"""
struct CGState{W<:CGWorkspace,P,B}
    workspace::W
    preconditioner::P    # shared: the hierarchy's grids and coefficients
    mgbuffers::B         # per task: the vectors a cycle writes
end

# The hierarchy is read-only during a solve, so it goes on the plan and every task reads the same
# one. Building it coarsens a grid and forms a Galerkin operator per level.
prepare_shared(solver::CGSolver, grid, boundary::AbstractBoundaryCondition) =
    solver.multigrid ? multigrid(grid, boundary, eltype(grid)) : nothing

function prepare_solver(solver::CGSolver, grid, boundary::AbstractBoundaryCondition;
                        backend = ComputationalBackends.SerialBackend(), shared = nothing)
    T = eltype(grid)
    dims = size(grid)
    ws = CGWorkspace(allocate_zeros(backend, T, dims), allocate_zeros(backend, T, dims),
                     allocate_zeros(backend, T, dims), allocate_zeros(backend, T, dims))
    # A plan built before `prepare_shared` existed, or a solver used without one, still works: the
    # hierarchy is built here instead.
    pre = shared === nothing && solver.multigrid ?
          prepare_shared(solver, grid, boundary) : shared
    bufs = pre === nothing ? nothing : multigrid_buffers(pre, T; backend = backend)
    return CGState(ws, pre, bufs)
end

function solve_poisson!(
    Φ::AbstractArray{T,N}, RHS::AbstractArray{T,N},
    grid::FaceIndexedGrid{T,G,N}, solver::CGSolver;
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
    project_out_constant!(r, grid, c; backend = backend)
    bnorm = sqrt(_dot_measure(r, r, grid, c; backend = backend))
    bnorm == 0 && return SolverResult{T}(true, 0, zero(T))

    _precondition!(z, r, grid, c, state.preconditioner, state.mgbuffers; backend = backend)
    copyto!(p, z)
    rz = _dot_measure(r, z, grid, c; backend = backend)

    rnorm = bnorm
    for iter in 1:solver.max_iter
        apply_laplacian!(Ap, p, grid, c; backend = backend)
        @. Ap = -Ap
        pAp = _dot_measure(p, Ap, grid, c; backend = backend)
        # A zero curvature means `p` lies in the null space; with the projection above that only
        # happens once the residual is already exhausted.
        iszero(pAp) && return SolverResult{T}(true, iter, rnorm / bnorm)
        α = rz / pAp
        @. Φ += α * p
        @. r -= α * Ap
        c.singular && project_out_constant!(r, grid, c; backend = backend)
        rnorm = sqrt(_dot_measure(r, r, grid, c; backend = backend))
        rnorm <= solver.rtol * bnorm && begin
            c.singular && project_out_constant!(Φ, grid, c; backend = backend)
            return SolverResult{T}(true, iter, rnorm / bnorm)
        end
        _precondition!(z, r, grid, c, state.preconditioner, state.mgbuffers; backend = backend)
        rz_new = _dot_measure(r, z, grid, c; backend = backend)
        @. p = z + (rz_new / rz) * p
        rz = rz_new
    end
    c.singular && project_out_constant!(Φ, grid, c; backend = backend)
    return SolverResult{T}(false, solver.max_iter, rnorm / bnorm)
end

"""
    _precondition!(z, r, grid, c, preconditioner, buffers)

Apply `M⁻¹` to the residual: the Jacobi diagonal where there is no preconditioner, a V-cycle where
there is (`Multigrid.jl` adds that method, alongside the type).

`preconditioner` is shared across tasks and `buffers` are the task's own — see
[`prepare_shared`](@ref).
"""
_precondition!(z, r, grid, c, ::Nothing, ::Any;
               backend = ComputationalBackends.SerialBackend()) =
    _jacobi_precondition!(z, r, grid, c; backend = backend)

function _jacobi_precondition!(z, r, grid, c::LaplacianCoefficients{N,T};
                               backend = ComputationalBackends.SerialBackend()) where {N,T}
    d = c.diag
    # Linear, as in `smooth!`: a cell reads only its own diagonal entry.
    FlowGeometries.Execution.run_indices(length(z), backend) do lin
        @inbounds begin
            dI = d[lin]
            z[lin] = iszero(dI) ? zero(T) : r[lin] / -dI
        end
        return nothing
    end
    return z
end

"""
    project_out_constant!(Φ, grid, c)

Remove the measure-weighted mean of `Φ` over the active cells.

A closed problem — every direction periodic, or a Neumann boundary — leaves the constants in
`L`'s null space. Krylov iterations must stay orthogonal to that null space or they drift along
it, so this is applied to the right-hand side once and to the iterate as it goes.
"""
function project_out_constant!(Φ, grid, c::LaplacianCoefficients{N,T};
                               backend = ComputationalBackends.SerialBackend()) where {N,T}
    c.singular || return Φ
    iszero(c.total) && return Φ
    meas = c.measure
    # Both passes are linear: a cell reads its own measure and its own value.
    acc = FlowGeometries.Execution.reduce_indices(+, zero(T), length(Φ), backend) do lin
        @inbounds Φ[lin] * meas[lin]
    end
    m = acc / c.total
    FlowGeometries.Execution.run_indices(length(Φ), backend) do lin
        # An inactive cell has zero measure and keeps the zero the operator puts there.
        @inbounds iszero(meas[lin]) || (Φ[lin] -= m)
        return nothing
    end
    return Φ
end
