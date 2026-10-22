# The physical-space Helmholtz–Hodge decomposition `u = u_div ⊕ u_rot ⊕ u_harm` on faces, with
# FlowGeometries' mimetic operators. `L χ = D u` and `u_div = G χ` give `D u_div = D u`, so
# `D(u − u_div) = 0` to the solver tolerance; `L R = curl u` on each dual grid splits that remainder
# into `u_rot = rot R` and the harmonic part, orthogonal to `u_div` because `curl ∘ G = 0`.

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

"""
    HelmholtzPlan

The **shareable** part of a decomposition on one `(grid, boundary, eltype)`: the Laplacian
coefficients on the primal grid and on each dual grid, and the dual grids themselves. Buffers live
in a [`HelmholtzWorkspace`](@ref), one per task, precisely so that a whole batch can share a single
plan without racing.

The geometry behind those coefficients — face areas need the scale factors *at the face*, which on
a curved grid is trigonometry per cell per direction — is invariant across solver iterations,
across the `P + 1` potentials, and across a whole batch. Rebuilding it per call is the single
largest avoidable cost in a decomposition, so it is built here and only read afterwards.
"""
struct HelmholtzPlan{N,P,T,G,BC,LC,FM,DG,DBC,DC,SV,DSV,EB,SH,DSH}
    grid::G
    boundary::BC
    coefficients::LC
    metrics::FM
    dual_grids::DG
    dual_boundaries::DBC
    dual_coefficients::DC
    solver::SV
    dual_solvers::DSV
    execution::EB
    # The read-only half of each solver's state — the multigrid hierarchy is the one that costs
    # anything to build. A batch builds it once here; a task adds only the vectors it writes.
    shared::SH
    dual_shared::DSH
end

"""
    plan_helmholtz(grid; boundary = FG.Operators.Neumann(), solver = AutoSolver(), execution = AutoBackend())

Resolve everything a decomposition on `grid` fixes once: the Laplacian coefficients on the primal
and dual grids, which solver runs on each (see [`select_solver`](@ref)), that solver's shared state,
and which execution backend the loops inside one decomposition use.

Every solve is of the grid's own `L`. A spherical transform, named or chosen by `AutoSolver`, is the
preconditioner of [`CGSolver`](@ref) on the grids whose nodes it accepts, and a dual grid whose nodes
it does not accept takes the default preconditioner.

`execution` is the *intra-field* backend: it drives the operator and solver loops of a single field.
A parallel batch runs each field with a serial inner backend.
"""
function plan_helmholtz(
    grid::FlowGeometries.Grids.FaceIndexedGrid{T,G,N};
    boundary::FlowGeometries.Operators.AbstractBoundaryCondition = FlowGeometries.Operators.Neumann(),
    solver::AbstractPoissonSolver = AutoSolver(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
) where {G,T,N}
    P = FlowGeometries.Operators.n_rotation_components(N)
    exec = resolve_execution_backend(execution)
    pairs = FlowGeometries.Operators.rotation_pairs(Val(N))

    # Everything geometric is assembled on the host, where the coordinates and scale factors are, and
    # the inner loops read its results. Choosing and preparing the solver reads the mask, so both
    # happen before anything moves.
    hmetrics = FlowGeometries.Operators.face_metrics(grid, boundary)
    hcoeff = FlowGeometries.Operators.laplacian_coefficients(hmetrics)
    hduals = ntuple(p -> dual_grid(grid, pairs[p][1], pairs[p][2], hmetrics), Val(P))
    # The rotation potential is zero on the corners the dual grid drops and on those it masks,
    # whatever the primal condition: each boundary component is a streamline of `u_rot`.
    dbcs = ntuple(p -> dual_boundary(grid, pairs[p]...), Val(P))
    hdcoef = ntuple(p -> dual_coefficients(grid, hmetrics, pairs[p]..., hduals[p]), Val(P))

    # The solver's choice and its read-only state are made here; the buffers each task writes
    # through are built by `allocate_workspace`. A dual grid has its own dimensions and mask, so it
    # chooses its own.
    concrete = select_solver(solver, grid, boundary; execution = exec)
    dsolvers = ntuple(Val(P)) do p
        select_solver(_dual_request(solver, hduals[p], dbcs[p], exec), hduals[p], dbcs[p];
                      execution = exec)
    end
    hshared = prepare_shared(concrete, grid, boundary; coefficients = hcoeff)
    hdshared = ntuple(p -> prepare_shared(dsolvers[p], hduals[p], dbcs[p]; coefficients = hdcoef[p]),
                      Val(P))

    move(x) = FlowGeometries.Execution.on_backend(exec, x)
    g = move(grid)
    coeff = move(hcoeff)
    metrics = move(hmetrics)
    duals = ntuple(p -> move(hduals[p]), Val(P))
    dcoef = ntuple(p -> move(hdcoef[p]), Val(P))
    # The shared solver state is assembled on the host alongside the coefficients and moves with
    # them; a kernel reads it during every solve.
    shared = move(hshared)
    dshared = ntuple(p -> move(hdshared[p]), Val(P))

    return HelmholtzPlan{N,P,T,typeof(g),typeof(boundary),typeof(coeff),typeof(metrics),
                         typeof(duals),typeof(dbcs),typeof(dcoef),typeof(concrete),typeof(dsolvers),
                         typeof(exec),typeof(shared),typeof(dshared)}(
        g, boundary, coeff, metrics, duals, dbcs, dcoef, concrete, dsolvers, exec, shared, dshared,
    )
end

"""
    HelmholtzWorkspace

The mutable buffers one decomposition writes through, held apart from the plan.

A [`HelmholtzPlan`](@ref) holds only what is the same for every field on a grid, so a batch shares
one. A workspace holds the face and corner buffers and the solver state every field writes through;
each concurrent task needs its own.
"""
struct HelmholtzWorkspace{FV,CV,ST,DST}
    v::FV               # face velocity
    gχ::FV              # face divergent velocity
    urot::FV            # face rotational velocity
    harm::FV            # face harmonic velocity
    W::CV               # corner vorticity, one array per pair
    state::ST           # solver state, primal grid
    dual_states::DST    # solver state, one per dual grid
end

"""
    allocate_workspace(plan) -> HelmholtzWorkspace

Buffers for one task working through `plan`, its transform plans included, which [`close!`](@ref)
releases. Concurrent tasks may build workspaces at once: every plan is built under FFTW.jl's planner
lock.
"""
allocate_workspace(plan::HelmholtzPlan) = _allocate_workspace(plan, plan.execution)

# A workspace for fields whose loops run on `b`, in the plan's memory: the plan's own backend, or the one
# a parallel batch runs each field on, which also sets the transform plans' thread count.
function _allocate_workspace(plan::HelmholtzPlan{N,P,T}, b) where {N,P,T}
    grid, bc = plan.grid, plan.boundary
    # Solver state holds the buffers a solve writes through — the conjugate-gradient vectors, the
    # multigrid level arrays, the transform scratch — so each task builds its own.
    state = prepare_solver(plan.solver, grid, bc; execution = b, shared = plan.shared)
    # Mapped across the tuples for the reason `_solve_potentials!` is — a workspace is built once
    # per task, so a dynamic dispatch here is paid by every member of a batch.
    dual_states = map(plan.dual_solvers, plan.dual_grids, plan.dual_boundaries,
                      plan.dual_shared) do ds, dg, dbc, sh
        prepare_solver(ds, dg, dbc; execution = b, shared = sh)
    end
    faces() = FlowGeometries.Operators.allocate_faces(T, grid; execution = b)
    return HelmholtzWorkspace(faces(), faces(), faces(), faces(),
                              FlowGeometries.Operators.allocate_corners(T, grid; execution = b),
                              state, dual_states)
end

"""
    close!(ws::HelmholtzWorkspace) -> ws

Release the foreign transform plans the workspace's solver states hold: NUFSHT's NUFFT plans under a
spherical preconditioner. Idempotent; host plans are also freed when collected. Every entry that builds
its own workspace releases it before returning.
"""
close!(ws::HelmholtzWorkspace) = (foreach(_close_state!, (ws.state, ws.dual_states...)); ws)

_close_state!(s) = s
_close_state!(s::CGState) = _close_state!(s.preconditioner)
_close_state!(p::SpectralPreconditioner) = _close_state!(p.state)

Base.show(io::IO, p::HelmholtzPlan{N,P,T}) where {N,P,T} =
    print(io, "HelmholtzPlan{", T, "}(", join(size(p.grid), "×"), ", ",
          nameof(typeof(p.boundary)), ", ", P, " rotation component", P == 1 ? "" : "s", ")")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

"""
    AbstractHelmholtzResult{N,P,T}

What a decomposition writes into. Every subtype carries `χ`, `rotation_potential`, `divergence`
and the two convergence diagnostics; [`HelmholtzResult`](@ref) adds the collocated velocity parts
and [`PotentialsResult`](@ref) stops at the potentials.

The accessors that read only the shared fields — [`velocity_potential`](@ref),
[`streamfunction`](@ref), [`vector_potential`](@ref), [`corner_rotation_potential`](@ref) — are
defined on this type once.
"""
abstract type AbstractHelmholtzResult{N,P,T<:AbstractFloat} end

"""
    HelmholtzResult

The result of every decomposition route: the iterative and transform solvers on a grid of cells,
the periodic FFT, the scattered NUFFT fit and the spherical spectral transforms.

- `u_rot`, `u_div`, `u_harm`: the rotational, divergent and harmonic velocity, component-last
  `(dims..., N)` (`(M, N)` on a node set).
- `χ`, `rotation_potential`, `divergence`: the velocity potential, the `P` rotation-potential
  components and `∇·u`, one value per cell or node. On the staggered path `rotation_potential` sits
  on the corner grid; on the spectral paths at the cells or nodes.
- `harmonic_fraction`: `‖u_harm‖/‖u‖`. On the staggered path both norms are taken on faces
  (weight `area_flux · gap`), where the split is exact.
- `residual_fraction`: `‖u − u_rot − u_div − u_harm‖/‖u‖` over the cells or nodes, the part of `u`
  the decomposition's representation does not carry: the collocated↔staggered round trip on the
  staggered path, the fit residual on a scattered or spherical fit, zero on the periodic FFT.
- `χ_solve`, `rot_solve`: convergence of the potential solves. On a scattered or spherical fit
  every entry reports the least converged of the fits the parts come from; on the periodic FFT
  they are exact.

The array fields are `const`; the diagnostics are written in place by `helmholtz_decompose!`.
"""
mutable struct HelmholtzResult{N,P,T<:AbstractFloat,AV<:AbstractArray{T},AS<:AbstractArray{T},
                               CV} <: AbstractHelmholtzResult{N,P,T}
    const u_rot::AV
    const u_div::AV
    const u_harm::AV
    const χ::AS
    const rotation_potential::CV
    const divergence::AS
    harmonic_fraction::T
    residual_fraction::T
    χ_solve::SolverResult{T}
    rot_solve::NTuple{P,SolverResult{T}}
end

"""
    PotentialsResult

The scalar potentials alone: `χ` at cell centres, the rotation potential `R` on the corner grid,
the divergence of the face velocity with the flux through the domain edge included, and the
convergence diagnostics.

Filtering works on these — on the sphere, coarse-graining commutes with the differential operators
when it is applied to `ψ` and `χ` as scalars (Aluie 2019). No collocated velocity array is
allocated: [`helmholtz_decompose!`](@ref) dispatches on this type, stops after the potentials, and
never interpolates back to cell centres or forms the norms.

Array fields are `const` and the diagnostics are not, as on [`HelmholtzResult`](@ref).
"""
mutable struct PotentialsResult{N,P,T<:AbstractFloat,AS<:AbstractArray{T,N},CV} <:
               AbstractHelmholtzResult{N,P,T}
    const χ::AS
    const rotation_potential::CV
    const divergence::AS
    χ_solve::SolverResult{T}
    rot_solve::NTuple{P,SolverResult{T}}
end

@inline Base.ndims(::AbstractHelmholtzResult{N}) where {N} = N

"""
    allocate_potentials(plan) -> PotentialsResult

Buffers for the potentials alone: `χ`, `R` and the divergence, against the three `(dims…, N)`
velocity arrays [`allocate_result`](@ref) adds.
"""
function allocate_potentials(plan::HelmholtzPlan{N,P,T}) where {N,P,T}
    dims = size(plan.grid)
    b = plan.execution
    dummy = SolverResult{T}(false, 0, zero(T))
    Rpot = FlowGeometries.Operators.allocate_corners(T, plan.grid; execution = b)
    χ = _zeros(b, T, dims)
    return PotentialsResult{N,P,T,typeof(χ),typeof(Rpot)}(
        χ, Rpot, _zeros(b, T, dims), dummy, ntuple(_ -> dummy, Val(P)),
    )
end

"""
    allocate_result(plan) -> HelmholtzResult

A zeroed result matching `plan`. Array types follow the plan's buffers, so a device-resident plan
gives a device-resident result — nothing here hardcodes `Array`.
"""
function allocate_result(plan::HelmholtzPlan{N,P,T}) where {N,P,T}
    dims = size(plan.grid)
    b = plan.execution
    dummy = SolverResult{T}(false, 0, zero(T))
    Rpot = FlowGeometries.Operators.allocate_corners(T, plan.grid; execution = b)
    u_rot = _zeros(b, T, (dims..., N))
    u_div = _zeros(b, T, (dims..., N))
    u_harm = _zeros(b, T, (dims..., N))
    χ = _zeros(b, T, dims)
    div = _zeros(b, T, dims)
    return HelmholtzResult{N,P,T,typeof(u_rot),typeof(χ),typeof(Rpot)}(
        u_rot, u_div, u_harm, χ, Rpot, div,
        zero(T), zero(T), dummy, ntuple(_ -> dummy, Val(P)),
    )
end

"""
    HelmholtzBatch

`B` decompositions on one grid, stored contiguously: each array field of a [`HelmholtzResult`](@ref)
gains a trailing batch axis, and `batch[b]` is a result of views onto slice `b`. The scalar
diagnostics are host vectors, one entry per field, whatever memory the arrays live in.
"""
struct HelmholtzBatch{N,P,T<:AbstractFloat,AV<:AbstractArray{T},AS<:AbstractArray{T},CV,
                      HV<:AbstractVector{T},SV<:AbstractVector,RV<:AbstractVector}
    u_rot::AV                       # (dims..., N, B)
    u_div::AV
    u_harm::AV
    χ::AS                           # (dims..., B)
    rotation_potential::CV          # per pair, (cdims..., B)
    divergence::AS
    harmonic_fraction::HV
    residual_fraction::HV
    χ_solve::SV
    rot_solve::RV
end

@inline Base.length(b::HelmholtzBatch) = length(b.harmonic_fraction)
@inline Base.size(b::HelmholtzBatch) = (length(b),)
@inline Base.eachindex(b::HelmholtzBatch) = Base.OneTo(length(b))
Base.show(io::IO, b::HelmholtzBatch{N,P,T}) where {N,P,T} =
    print(io, "HelmholtzBatch{", T, "}(", length(b), " fields)")

"""
    allocate_batch(plan, nfields) -> HelmholtzBatch

A zeroed [`HelmholtzBatch`](@ref) of `nfields` slots: its arrays in the plan's memory, its diagnostics
host vectors.
"""
function allocate_batch(plan::HelmholtzPlan{N,P,T}, nfields::Int) where {N,P,T}
    dims = size(plan.grid)
    bk = plan.execution
    pairs = FlowGeometries.Operators.rotation_pairs(Val(N))
    u_rot = _zeros(bk, T, (dims..., N, nfields))
    u_div = _zeros(bk, T, (dims..., N, nfields))
    u_harm = _zeros(bk, T, (dims..., N, nfields))
    χ = _zeros(bk, T, (dims..., nfields))
    div = _zeros(bk, T, (dims..., nfields))
    Rpot = ntuple(p -> _zeros(bk, T, (FlowGeometries.Grids.corner_dims(plan.grid, pairs[p]...)..., nfields)),
                  Val(P))
    dummy = SolverResult{T}(false, 0, zero(T))
    # The diagnostics are scalars written one per field from the host, so they live there.
    frac = zeros(T, nfields)
    resid = zeros(T, nfields)
    χs = fill(dummy, nfields)
    rots = fill(ntuple(_ -> dummy, Val(P)), nfields)
    return HelmholtzBatch{N,P,T,typeof(u_rot),typeof(χ),typeof(Rpot),typeof(frac),typeof(χs),
                          typeof(rots)}(u_rot, u_div, u_harm, χ, Rpot, div, frac, resid, χs, rots)
end

# A view onto one slice, with the batch axis dropped.
@inline _slice(A::AbstractArray{<:Any,M}, b::Int) where {M} =
    view(A, ntuple(_ -> Colon(), Val(M - 1))..., b)

function Base.getindex(batch::HelmholtzBatch{N,P,T}, b::Int) where {N,P,T}
    @boundscheck checkbounds(batch.harmonic_fraction, b)
    u_rot = _slice(batch.u_rot, b)
    χ = _slice(batch.χ, b)
    Rpot = ntuple(p -> _slice(batch.rotation_potential[p], b), Val(P))
    return HelmholtzResult{N,P,T,typeof(u_rot),typeof(χ),typeof(Rpot)}(
        u_rot, _slice(batch.u_div, b), _slice(batch.u_harm, b), χ, Rpot,
        _slice(batch.divergence, b),
        batch.harmonic_fraction[b], batch.residual_fraction[b], batch.χ_solve[b],
        batch.rot_solve[b],
    )
end

"""
    copy_slice!(dest, i, src, j) -> dest

Copy slice `j` of one batch into slot `i` of another, fields and diagnostics alike.
"""
function copy_slice!(dest::HelmholtzBatch{N,P}, i::Int, src::HelmholtzBatch{N,P},
                     j::Int) where {N,P}
    copyto!(_slice(dest.u_rot, i), _slice(src.u_rot, j))
    copyto!(_slice(dest.u_div, i), _slice(src.u_div, j))
    copyto!(_slice(dest.u_harm, i), _slice(src.u_harm, j))
    copyto!(_slice(dest.χ, i), _slice(src.χ, j))
    copyto!(_slice(dest.divergence, i), _slice(src.divergence, j))
    for p in 1:P
        copyto!(_slice(dest.rotation_potential[p], i), _slice(src.rotation_potential[p], j))
    end
    dest.harmonic_fraction[i] = src.harmonic_fraction[j]
    dest.residual_fraction[i] = src.residual_fraction[j]
    dest.χ_solve[i] = src.χ_solve[j]
    dest.rot_solve[i] = src.rot_solve[j]
    return dest
end

# The diagnostics live on the batch, so a decomposed slice writes them back.
@inline function _store_diagnostics!(batch::HelmholtzBatch, b::Int, r::HelmholtzResult)
    batch.harmonic_fraction[b] = r.harmonic_fraction
    batch.residual_fraction[b] = r.residual_fraction
    batch.χ_solve[b] = r.χ_solve
    batch.rot_solve[b] = r.rot_solve
    return batch
end

"""
    face_velocity(ws) -> NTuple{N,AbstractArray}

The face-normal velocity the last decomposition through `ws` split: one array per direction, shaped
by `FlowGeometries.Grids.face_dims`.

On faces the split is exact: [`face_velocity`](@ref) is the sum of [`face_divergent`](@ref),
[`face_rotational`](@ref) and [`face_harmonic`](@ref), the three are orthogonal under the weight
`area_flux · gap`, and the divergence and curl identities hold to the solver tolerance. The
collocated parts on the result are these averaged to cell centres.

Valid until the next decomposition through the same workspace, which overwrites them.
"""
face_velocity(ws::HelmholtzWorkspace) = ws.v

"""
    face_divergent(ws) -> NTuple{N,AbstractArray}

`u_div` on faces: `Gχ` wherever the operator couples, and the face velocity itself on an edge face of
a bounded direction under `Neumann`, the condition `∂χ/∂n = u·n̂`. Its divergence, flux through the
edge included, is the face velocity's. See [`face_velocity`](@ref).
"""
face_divergent(ws::HelmholtzWorkspace) = ws.gχ

"""
    face_rotational(ws) -> NTuple{N,AbstractArray}

`−δR` on faces: divergence-free, with no flux through the domain edge. See [`face_velocity`](@ref).
"""
face_rotational(ws::HelmholtzWorkspace) = ws.urot

"""
    face_harmonic(ws) -> NTuple{N,AbstractArray}

`u_harm` on faces, `face_velocity − face_divergent − face_rotational`: divergence-free, curl-free
on every corner of the dual grid, with no flux through the domain edge under `Neumann`. Its dimension
is the first Betti number of the active region; on a simply connected domain it is zero to the
solver tolerance. See [`face_velocity`](@ref).
"""
face_harmonic(ws::HelmholtzWorkspace) = ws.harm

"""
    corner_rotation_potential(result) -> NTuple{P,AbstractArray}

`R` on the corner (dual) grid, one array per rotation pair. The solve writes here, so this is the
staggered field itself — see [`face_velocity`](@ref).
"""
corner_rotation_potential(r::AbstractHelmholtzResult) = r.rotation_potential

"""
    streamfunction(result) -> Array

The 2-D streamfunction `ψ`, on the corner (dual) grid. Only defined in 2D.
"""
streamfunction(r::AbstractHelmholtzResult{2}) = r.rotation_potential[1]
streamfunction(::AbstractHelmholtzResult{N}) where {N} = throw(ArgumentError(
    "streamfunction is only defined in 2D (got N=$N); use `vector_potential` in 3D or " *
    "`rotation_potential` generally"))

"""
    velocity_potential(result) -> Array

The scalar velocity potential `χ`, at cell centres, in any dimension.
"""
velocity_potential(r::AbstractHelmholtzResult) = r.χ

"""
    vector_potential(result) -> (A1, A2, A3)

The 3-D vector potential with `u_rot = ∇ × A`: `A1 = −R_23`, `A2 = R_13`, `A3 = −R_12`, since
`u_rot_a = −Σ_b ∂_b R_ab`. Only defined in 3D.
"""
function vector_potential(r::AbstractHelmholtzResult{3})
    R12, R13, R23 = r.rotation_potential
    return (-1 .* R23, R13, -1 .* R12)
end
vector_potential(::AbstractHelmholtzResult{N}) where {N} =
    throw(ArgumentError("vector_potential is only defined in 3D (got N=$N)"))

# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------

"""
    helmholtz_decompose!(result, u, plan[, ws]) -> result

Decompose the cell-centred, component-last velocity `u` into `result`, in place, through the workspace
`ws`; without one, through a workspace built for the call and released on return.

How far the work goes is the `result`'s type. A [`PotentialsResult`](@ref) stops once `χ` and `R`
are solved; a [`HelmholtzResult`](@ref) goes on to the face-space parts, their collocated averages
and the two fractions.

Under `Neumann` the split is the natural Helmholtz–Hodge decomposition (Bhatia et al. 2013): the
divergent part carries the normal velocity through the edge of every bounded direction
(`∂χ/∂n = u·n̂`), and the rotational and harmonic parts carry none. Faces against masked cells carry
no flux in the data or in any part.

`plan` fixes the solver and the execution backend; select both with [`plan_helmholtz`](@ref). A
solve that stops short of its tolerance is returned as it stands, its `converged`, `iterations` and
`residual` in `result.χ_solve` and `result.rot_solve`.
"""
function helmholtz_decompose!(result::AbstractHelmholtzResult{N,P,T}, u::AbstractArray,
                              plan::HelmholtzPlan{N,P,T}) where {N,P,T}
    ws = allocate_workspace(plan)
    try
        return _decompose!(result, u, plan, ws, plan.execution)
    finally
        close!(ws)
    end
end

helmholtz_decompose!(result::AbstractHelmholtzResult{N,P,T}, u::AbstractArray, plan::HelmholtzPlan{N,P,T},
                     ws::HelmholtzWorkspace) where {N,P,T} =
    _decompose!(result, u, plan, ws, plan.execution)

@inline function _check_velocity(u, plan::HelmholtzPlan{N}, ::Val{N}) where {N}
    size(u) == (size(plan.grid)..., N) || throw(DimensionMismatch(
        "velocity array size $(size(u)) does not match (dims..., N) = $((size(plan.grid)..., N))"))
    return nothing
end

# One decomposition with its loops on `execution`: the plan's own backend, or the one a batch driver
# runs each field on.
function _decompose!(result::PotentialsResult{N,P,T}, u::AbstractArray, plan::HelmholtzPlan{N,P,T},
                     ws::HelmholtzWorkspace, execution) where {N,P,T}
    _check_velocity(u, plan, Val(N))
    return _solve_potentials!(result, u, plan, ws, execution)
end

function _decompose!(result::HelmholtzResult{N,P,T}, u::AbstractArray, plan::HelmholtzPlan{N,P,T},
                     ws::HelmholtzWorkspace, execution) where {N,P,T}
    _check_velocity(u, plan, Val(N))
    _solve_potentials!(result, u, plan, ws, execution)

    grid, fm = plan.grid, plan.metrics
    lb = loop_backend(execution, length(plan.coefficients.measure))
    FlowGeometries.Operators.rotational_velocity!(ws.urot, result.rotation_potential, fm; execution = lb)
    _face_remainder!(ws.harm, ws.v, ws.gχ, ws.urot)
    FlowGeometries.Operators.to_centres!(_components(result.u_div, Val(N)), ws.gχ, fm; execution = lb)
    FlowGeometries.Operators.to_centres!(_components(result.u_rot, Val(N)), ws.urot, fm; execution = lb)
    FlowGeometries.Operators.to_centres!(_components(result.u_harm, Val(N)), ws.harm, fm; execution = lb)

    vnorm = _face_norm(ws.v, fm, lb)
    result.harmonic_fraction = iszero(vnorm) ? zero(T) : _face_norm(ws.harm, fm, lb) / vnorm
    unorm = velocity_norm(u, grid; execution = lb)
    result.residual_fraction = iszero(unorm) ? zero(T) :
        _remainder_norm(u, result.u_div, result.u_rot, result.u_harm, grid, lb) / unorm
    return result
end

# `χ` and `R`: what both entry points share. Writes only fields the two result types have in
# common, so one body serves both.
function _solve_potentials!(result, u::AbstractArray, plan::HelmholtzPlan{N,P,T},
                            ws::HelmholtzWorkspace, backend) where {N,P,T}
    grid, bc = plan.grid, plan.boundary
    concrete = plan.solver
    state = ws.state
    fm = plan.metrics
    lb = loop_backend(backend, length(plan.coefficients.measure))

    # 1. To faces, where the projection is exact.
    FlowGeometries.Operators.to_faces!(ws.v, _components(u, Val(N)), fm; execution = lb)

    # 2. The right-hand side `D u` over the operator's faces. Under `Neumann` the flux through a
    #    bounded edge is the boundary datum `∂χ/∂n = u·n̂`, so it stays out of the source; every closed
    #    component then sums to zero, and the projection removes round-off.
    FlowGeometries.Operators.divergence!(result.divergence, ws.v, fm; execution = lb)
    FlowGeometries.Operators.project_out_constant!(result.divergence, plan.coefficients; execution = lb)

    # 3. L χ = D u, then u_div: `Gχ` on the operator's faces and the datum on the bounded edges, so
    #    `u_div` has the divergence of `u`, edge flux included, which `divergence` then reports.
    result.χ_solve = solve_poisson!(result.χ, result.divergence, grid, concrete;
                                    boundary = bc, coefficients = plan.coefficients, state = state,
                                    execution = backend)
    _divergent_faces!(ws.gχ, result.χ, ws.v, fm, lb)
    _add_edge_flux!(result.divergence, ws.v, fm, lb)

    # 4. W = curl u on the corners, and L R = W on each dual grid. The remainder u − u_div is
    #    divergence-free to the χ tolerance; `R` splits it into its rotational and harmonic parts.
    FlowGeometries.Operators.curl!(ws.W, ws.v, fm; execution = lb)
    # The `P` dual grids need not share a type; `map` compiles each solve against its own grid.
    result.rot_solve = map(result.rotation_potential, ws.W, plan.dual_grids, plan.dual_boundaries,
                           plan.dual_coefficients, plan.dual_solvers,
                           ws.dual_states) do Rp, Wp, dg, dbc, dc, ds, st
        FlowGeometries.Operators.project_out_constant!(Wp, dc; execution = loop_backend(backend, length(Wp)))
        solve_poisson!(Rp, Wp, dg, ds; boundary = dbc, coefficients = dc, state = st,
                       execution = backend)
    end
    return result
end

"""
    _divergent_faces!(g, χ, v, fm, backend) -> g

`u_div` on faces: `Gχ` on every face of nonzero `area_op`, and `v` elsewhere. `v` is zero on every face
of zero `area_flux`, as `to_faces!` writes it, so the faces that take `v` are the edges of bounded
directions under `Neumann`, where `v` is the boundary datum of `χ`.
"""
function _divergent_faces!(g::NTuple{N,AbstractArray}, χ, v::NTuple{N,AbstractArray},
                           fm::FlowGeometries.Operators.FaceMetrics{N}, backend) where {N}
    FlowGeometries.Operators.gradient!(g, χ, fm; execution = backend)
    map((gd, vd, ad) -> _edge_datum!(gd, vd, ad, backend), g, v, fm.area_op)
    return g
end

function _edge_datum!(gd, vd, ad, backend)
    fcart = CartesianIndices(size(gd))
    FlowGeometries.Execution.run_indices(length(gd), backend,
                                         FlowGeometries.Execution.Written(gd)) do lin
        @inbounds begin
            F = fcart[lin]
            iszero(ad[F]) && (gd[F] = vd[F])
        end
        return nothing
    end
    return nothing
end

"""
    _add_edge_flux!(δ, v, fm, backend) -> δ

Add to `δ` the flux of `v` through the faces where `area_flux` exceeds `area_op`, the edges of bounded
directions under `Neumann`, so `δ` holds the divergence of `v` over `area_flux`.
"""
function _add_edge_flux!(δ, v::NTuple{N,AbstractArray}, fm::FlowGeometries.Operators.FaceMetrics{N,T},
                         backend) where {N,T}
    fm.area_flux === fm.area_op && return δ
    ntuple(Val(N)) do d
        FlowGeometries.Grids.isperiodic(fm.grid, d) ||
            _edge_flux!(δ, v[d], fm.area_flux[d], fm.area_op[d], fm.orientation[d], fm.grid, d, backend)
        return nothing
    end
    return δ
end

function _edge_flux!(δ::AbstractArray{S}, vd, ad, od, σ, grid, d::Int, backend) where {S}
    msk = FlowGeometries.Grids.mask(grid)
    meas = FlowGeometries.Grids.measure(grid)
    dims = FlowGeometries.Grids.size_tuple(grid)
    plane = CartesianIndices(ntuple(e -> e == d ? 1 : dims[e], Val(length(dims))))
    # The face below cell 1 counts outward as `−σ`, the face above cell `n` as `+σ`. The two edges run
    # one after the other, since at `n = 1` they write the same cells.
    for (f, c, s) in ((1, 1, -S(σ)), (dims[d] + 1, dims[d], S(σ)))
        FlowGeometries.Execution.run_indices(length(plane), backend) do lin
            @inbounds begin
                P = plane[lin]
                F = CartesianIndex(ntuple(e -> e == d ? f : P[e], Val(length(dims))))
                C = CartesianIndex(ntuple(e -> e == d ? c : P[e], Val(length(dims))))
                msk[C] && (δ[C] += s * S(ad[F] - od[F]) * vd[F] / S(meas[C]))
            end
            return nothing
        end
    end
    return nothing
end

# `h = v − g − r` on every face.
function _face_remainder!(h::NTuple{N,AbstractArray}, v::NTuple{N,AbstractArray},
                          g::NTuple{N,AbstractArray}, r::NTuple{N,AbstractArray}) where {N}
    map((hd, vd, gd, rd) -> (hd .= vd .- gd .- rd), h, v, g, r)
    return h
end

"""
    _face_norm(f, fm, backend) -> T

`sqrt(Σ_d Σ_F area_flux · gap · f_d[F]²)`: the face-space norm of a staggered velocity, each face
standing for the volume between the two centres it separates.
"""
function _face_norm(f::NTuple{N,AbstractArray}, fm::FlowGeometries.Operators.FaceMetrics{N,T},
                    backend) where {N,T}
    return sqrt(sum(map((fd, ad, gd) -> _face_sum(fd, ad, gd, backend, T), f, fm.area_flux, fm.gap)))
end

function _face_sum(fd, ad, gd, backend, ::Type{T}) where {T}
    fcart = CartesianIndices(size(fd))
    return FlowGeometries.Execution.reduce_indices(+, zero(T), length(fd), backend) do lin
        @inbounds begin
            F = fcart[lin]
            T(ad[F] * gd[F] * abs2(fd[F]))
        end
    end
end

"""
    _remainder_norm(U, A, B, C, grid, backend) -> T

`‖U − A − B − C‖` in the measure-weighted norm of [`velocity_norm`](@ref), over the active cells.
"""
function _remainder_norm(U, A, B, C, grid::FlowGeometries.Grids.FaceIndexedGrid{T,G,N}, backend) where {G,T,N}
    msk = FlowGeometries.Grids.mask(grid)
    meas = FlowGeometries.Grids.measure(grid)
    cart = CartesianIndices(size(grid))
    u, a, b, c = map(X -> ntuple(k -> _component(X, k, Val(N)), Val(N)), (U, A, B, C))
    acc = FlowGeometries.Execution.reduce_indices(+, zero(T), length(cart), backend) do lin
        @inbounds begin
            I = cart[lin]
            if msk[I]
                s = zero(T)
                for k in 1:N
                    s += abs2(u[k][I] - a[k][I] - b[k][I] - c[k][I])
                end
                s * meas[I]
            else
                zero(T)
            end
        end
    end
    return sqrt(acc)
end

"""
    count_holes(grid) -> Int

Number of inactive regions fully enclosed by active cells. On a two-dimensional domain each is one
dimension of the harmonic space, and each direction along which the domain wraps adds one more: a
pure circulation about an island, or a uniform flow along a periodic channel, is harmonic.

The flood fill is `FlowGeometries.Connectivity.connected_components`, which walks the grid's own
wrapping, so a region running off one side of a periodic direction and back on the other is one
region, and touching the boundary is asked only of directions that have one.
"""
function count_holes(grid::FlowGeometries.Grids.AbstractGrid)
    labels, ncomp = FlowGeometries.Connectivity.connected_components(grid; active = false)
    ncomp == 0 && return 0
    N = ndims(grid)
    bounded = ntuple(d -> !FlowGeometries.Grids.isperiodic(grid, d), N)
    any(bounded) || return ncomp        # no boundary anywhere: every region is enclosed
    open = falses(ncomp)
    @inbounds for I in CartesianIndices(size(grid))
        lab = labels[I]
        lab == 0 && continue
        for d in 1:N
            bounded[d] || continue
            if I[d] == 1 || I[d] == size(grid, d)
                open[lab] = true
                break
            end
        end
    end
    return count(!, open)
end

"""
    velocity_norm(U, grid; execution = SerialBackend()) -> T

Measure-weighted `‖U‖ = sqrt(∫|U|² dV)` over the active cells, reduced on `execution`.
"""
function velocity_norm(U, grid::FlowGeometries.Grids.FaceIndexedGrid{T,G,N};
                       execution = ComputationalBackends.SerialBackend()) where {G,T,N}
    msk = FlowGeometries.Grids.mask(grid)
    # The grid's own measure is nonzero on inactive cells, so the mask is read separately.
    meas = FlowGeometries.Grids.measure(grid)
    cart = CartesianIndices(size(grid))
    comps = ntuple(c -> _component(U, c, Val(N)), Val(N))
    acc = FlowGeometries.Execution.reduce_indices(+, zero(T), length(cart), execution) do lin
        @inbounds begin
            I = cart[lin]
            if msk[I]
                s = zero(T)
                for c in 1:N
                    s += abs2(comps[c][I])
                end
                s * meas[I]
            else
                zero(T)
            end
        end
    end
    return sqrt(acc)
end

"""
    helmholtz_decompose(u, grid; boundary = FG.Operators.Neumann(), solver = AutoSolver(),
                        execution = AutoBackend()) -> HelmholtzResult
    helmholtz_decompose(u, v, grid; kwargs...)        # 2-D convenience
    helmholtz_decompose(u, v, w, grid; kwargs...)     # 3-D convenience

Decompose a cell-centred, component-last velocity field on a `StructuredGrid` or a
`CurvilinearGrid`. The component forms stack their arguments into that layout. `execution` is
[`plan_helmholtz`](@ref)'s: `AutoBackend()` runs on the host; a device array needs
`GPUBackend(…)` named.

On a `CurvilinearGrid` component `d` is the velocity along the grid's direction `d`, normal to the
direction-`d` faces.

Builds a [`HelmholtzPlan`](@ref) and discards it. A caller decomposing more than one field on the
same grid builds the plan once and calls [`helmholtz_decompose!`](@ref).
"""
function helmholtz_decompose(
    u::AbstractArray, grid::FlowGeometries.Grids.FaceIndexedGrid;
    boundary::FlowGeometries.Operators.AbstractBoundaryCondition = FlowGeometries.Operators.Neumann(),
    solver::AbstractPoissonSolver = AutoSolver(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    plan = plan_helmholtz(grid; boundary = boundary, solver = solver, execution = execution)
    return helmholtz_decompose!(allocate_result(plan), u, plan)
end

helmholtz_decompose(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N},
                    grid::FlowGeometries.Grids.FaceIndexedGrid; kwargs...) where {N} =
    helmholtz_decompose(_stack_components(grid, u, v), grid; kwargs...)

helmholtz_decompose(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N},
                    w::AbstractArray{<:Any,N}, grid::FlowGeometries.Grids.FaceIndexedGrid;
                    kwargs...) where {N} =
    helmholtz_decompose(_stack_components(grid, u, v, w), grid; kwargs...)

"""
    helmholtz_potentials(u, grid; boundary, solver, execution) -> PotentialsResult

Solve for `χ` and `R` and stop, for a caller who works on the potentials: filtering on the sphere
commutes with the differential operators on the scalars. Allocates neither the collocated velocity
parts nor the norms over them. Keywords and grids as in [`helmholtz_decompose`](@ref).
"""
function helmholtz_potentials(
    u::AbstractArray, grid::FlowGeometries.Grids.FaceIndexedGrid;
    boundary::FlowGeometries.Operators.AbstractBoundaryCondition = FlowGeometries.Operators.Neumann(),
    solver::AbstractPoissonSolver = AutoSolver(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    plan = plan_helmholtz(grid; boundary = boundary, solver = solver, execution = execution)
    return helmholtz_decompose!(allocate_potentials(plan), u, plan)
end

# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

"""
    _extension_loaded(name) -> Bool

Whether one of this package's extensions is loaded.
"""
@inline _extension_loaded(name::Symbol) = Base.get_extension(@__MODULE__, name) !== nothing

@inline _threading_available() = _extension_loaded(:HelmholtzDecompositionOhMyThreadsExt)

"""
    helmholtz_decompose_batch(plan, fields; execution = AutoBackend())

Decompose many fields sharing one grid. The fields are independent, so the batch is the parallel
axis: `ThreadedBackend` (OhMyThreads ext), `DistributedBackend` (Distributed ext) and `MPIBackend`
(MPI ext) each spread it across their workers; a serial backend maps sequentially. Results come
back in input order.

One `plan` serves the whole batch; each task takes its own [`HelmholtzWorkspace`](@ref), released when
its fields are done.

`execution` is the backend of the *batch* axis; the plan keeps its own for the loops inside one
field, set by [`plan_helmholtz`](@ref)'s `execution`. A serial or `GPUBackend` batch runs the fields one
after another on the plan's backend, so a short batch of large fields still parallelises within each
field. On a host plan a threaded batch runs each field serially, and a `DistributedBackend` or
`MPIBackend` batch on its inner backend; on a device plan every field runs on the plan's device.
"""
function helmholtz_decompose_batch(
    plan::HelmholtzPlan, fields;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    batch = allocate_batch(plan, length(fields))
    return helmholtz_decompose_batch!(batch, fields, plan; execution = execution)
end

"""
    helmholtz_decompose_batch!(batch, fields, plan; execution = AutoBackend()) -> batch

Decompose into a preallocated [`HelmholtzBatch`](@ref). `execution` is the batch axis's backend, as
in [`helmholtz_decompose_batch`](@ref).
"""
function helmholtz_decompose_batch!(
    batch::HelmholtzBatch, fields, plan::HelmholtzPlan;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    length(batch) == length(fields) || throw(DimensionMismatch(
        "batch holds $(length(batch)) slots but $(length(fields)) fields were given"))
    b = isempty(fields) ? ComputationalBackends.SerialBackend() :
                          _resolve_batch_backend(execution, fields)
    return _decompose_batch!(b, batch, fields, plan)
end

# One slice, into its views, with the field's loops on `execution`.
@inline function _decompose_slice!(batch::HelmholtzBatch, i::Int, field, plan, ws, execution)
    r = _decompose!(batch[i], field, plan, ws, execution)
    return _store_diagnostics!(batch, i, r)
end

"""
    _resolve_batch_backend(backend, fields) -> AbstractExecutionBackend

Resolve `Auto` over the batch axis against real capability: more than one field, more than one
thread, and the threading extension loaded. A concrete backend passes through unchanged.
"""
_resolve_batch_backend(b::ComputationalBackends.AbstractExecutionBackend, _) = b
function _resolve_batch_backend(::ComputationalBackends.AbstractAutoBackend, fields)
    parallel = length(fields) > 1 && Threads.nthreads() > 1 && _threading_available()
    return parallel ? ComputationalBackends.ThreadedBackend() : ComputationalBackends.SerialBackend()
end

# Sequential: one workspace, reused across fields, since nothing runs concurrently. Each field's
# loops run on the plan's own backend.
function _decompose_batch!(::ComputationalBackends.AbstractSerialBackend, batch::HelmholtzBatch,
                           fields, plan::HelmholtzPlan)
    ws = allocate_workspace(plan)
    try
        for (i, f) in enumerate(fields)
            _decompose_slice!(batch, i, f, plan, ws, plan.execution)
        end
    finally
        close!(ws)
    end
    return batch
end

# The backend each field of a parallel batch runs on, `inner` the backend of one worker: the plan's
# device on a device plan, and `inner` on a host plan, whose arrays only host loops read.
function _field_execution(plan::HelmholtzPlan, inner)
    plan.execution isa ComputationalBackends.AbstractGPUBackend && return plan.execution
    e = resolve_execution_backend(inner)
    e isa _HostBackend || throw(ArgumentError(
        "the plan holds host arrays, which $(nameof(typeof(e))) cannot run on. Plan with " *
        "`plan_helmholtz(grid; execution = $(nameof(typeof(e)))(…))`, or give the batch a host inner " *
        "backend."))
    return e
end

# On a device the fields run one after another, each on the device its plan was built for.
function _decompose_batch!(b::ComputationalBackends.AbstractGPUBackend, batch::HelmholtzBatch,
                           fields, plan::HelmholtzPlan)
    plan.execution == b || throw(ArgumentError(
        "a $(typeof(b)) batch runs each field on the device of its plan, which runs on " *
        "$(typeof(plan.execution)); build the plan with `plan_helmholtz(grid; execution = $(typeof(b))(…))`."))
    return _decompose_batch!(ComputationalBackends.SerialBackend(), batch, fields, plan)
end

_decompose_batch!(b::ComputationalBackends.AbstractExecutionBackend, ::HelmholtzBatch, _,
                  ::HelmholtzPlan) =
    throw(ArgumentError(_unsupported_backend_message(b, "helmholtz_decompose_batch")))

"""
    _unsupported_backend_message(backend, entry) -> String

Why `entry` cannot honour `backend`, and what to load so it can.
"""
function _unsupported_backend_message(backend, entry::AbstractString)
    pkg = _backend_package(backend)
    pkg === nothing && return "$entry has no implementation for $(typeof(backend))."
    return "$entry cannot run on $(typeof(backend)): the $pkg extension is not loaded. Run " *
           "`using $pkg` first, or pass an explicit `execution = ComputationalBackends.SerialBackend()`."
end

_backend_package(::ComputationalBackends.AbstractThreadedBackend) = "OhMyThreads"
_backend_package(::ComputationalBackends.AbstractDistributedBackend) = "Distributed"
_backend_package(::ComputationalBackends.AbstractMPIBackend) = "MPI"
_backend_package(::ComputationalBackends.AbstractExecutionBackend) = nothing
