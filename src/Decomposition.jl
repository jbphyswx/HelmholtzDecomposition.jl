"""
    Decomposition.jl — The physical-space Helmholtz–Hodge decomposition.

    u = u_div ⊕ u_rot ⊕ u_harm

`u_div = Gχ` is curl-free, `u_rot` is divergence-free, and `u_harm` is the harmonic remainder
carried by the domain's topology. With the operators of `Operators.jl` all three hold *discretely*:

- `L χ = D u` and `u_div = G χ`, so `D u_div = D G χ = L χ = D u` — the divergent part reproduces
  the divergence exactly;
- therefore `D(u − u_div) = 0` **by construction**, so the non-divergent remainder needs no solve
  of its own to be exactly divergence-free;
- `L R = curl u` on the dual grid then splits that remainder into `u_rot = −δR` and the harmonic
  residue, `⟨u_div, u_rot⟩ = 0` holding discretely because `curl ∘ G = 0` does.

The consequence worth stating: `harmonic_fraction` measures domain topology and boundary
circulation, not disagreement between three separately-chosen stencils.
"""

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
struct HelmholtzPlan{N,P,T,G,BC,LC,FM,DG,DC,SV,DSV,EB}
    grid::G
    boundary::BC
    coefficients::LC
    metrics::FM
    dual_grids::DG
    dual_coefficients::DC
    solver::SV
    dual_solvers::DSV
    backend::EB
end

"""
    plan_helmholtz(grid; boundary = Neumann(), solver = AutoSolver(), backend = AutoBackend())

Resolve everything a decomposition on `grid` fixes once: the Laplacian coefficients on the primal
and dual grids, **which solver will run**, that solver's reusable state, and **which execution
backend the loops inside one decomposition use**.

The solver is chosen here rather than per call deliberately. Choosing it reads the mask, which is
`O(N)`, and preparing it builds the transform plans; doing both inside `helmholtz_decompose!`
repeats them for each of the `P + 1` potentials and again for every field of a batch, which is
exactly the work a batch exists to amortize.

`backend` is the *intra-field* one — it drives the operator and solver loops, so a single large
field is parallel rather than only a batch of them. A batch overrides it with a serial inner
backend, because an outer and an inner loop each claiming every thread is slower than either.
"""
function plan_helmholtz(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N};
    boundary::AbstractBoundaryCondition = Neumann(),
    solver::AbstractPoissonSolver = AutoSolver(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
) where {G,T,N}
    P = n_rotation_components(N)
    exec = execution_backend(resolve_execution_backend(backend))
    pairs = rotation_pairs(Val(N))

    # Everything geometric is assembled on the host, where the coordinates and scale factors are:
    # this is one-off work, and the inner loops only ever read its results. Choosing and preparing
    # the solver also reads the mask, so both happen before anything moves.
    hcoeff = laplacian_coefficients(grid, boundary)
    hmetrics = face_metrics(grid, boundary)
    hduals = ntuple(p -> dual_grid(grid, pairs[p][1], pairs[p][2], boundary), Val(P))
    # `Dirichlet`, whatever the primal condition is. The rotation potential is pinned to zero on
    # the corners the dual grid drops — a closed boundary is a streamline, so `R` is constant along
    # it and that constant is taken as zero — and stating it as a condition rather than as a mask
    # is what leaves the dual grid unmasked and solvable by a sine transform.
    dbc = Dirichlet()
    hdcoef = ntuple(p -> laplacian_coefficients(hduals[p], dbc), Val(P))

    # Only the CHOICE is made here; the state each solver writes through is per task and is built
    # by `allocate_workspace`. A dual grid has its own dimensions and mask, so it chooses its own.
    concrete = select_solver(solver, grid, boundary)
    dsolvers = ntuple(p -> select_solver(solver, hduals[p], dbc), Val(P))

    g = to_backend(exec, grid)
    coeff = to_backend(exec, hcoeff)
    metrics = to_backend(exec, hmetrics)
    duals = ntuple(p -> to_backend(exec, hduals[p]), Val(P))
    dcoef = ntuple(p -> to_backend(exec, hdcoef[p]), Val(P))

    return HelmholtzPlan{N,P,T,typeof(g),typeof(boundary),typeof(coeff),typeof(metrics),
                         typeof(duals),typeof(dcoef),typeof(concrete),typeof(dsolvers),
                         typeof(exec)}(
        g, boundary, coeff, metrics, duals, dcoef, concrete, dsolvers, exec,
    )
end

"""
    HelmholtzWorkspace

The mutable buffers one decomposition writes through, held apart from the plan.

The split is what makes a batch correct rather than merely possible. A [`HelmholtzPlan`](@ref)
holds only things that are the same for every field on a grid — the Laplacian coefficients on the
primal and dual grids — so a batch shares exactly one, which is the whole point, those being the
expensive part. A workspace holds the face and corner buffers, which every field writes through,
so each task needs its own; sharing one across threads would race on them.
"""
struct HelmholtzWorkspace{FV,CV,SC,ST,DST}
    v::FV               # face velocity
    gχ::FV              # face gradient of χ
    urot::FV            # face rotational velocity
    W::CV               # corner vorticity, one array per pair
    R::CV               # corner rotation potential
    scratch::SC         # cell scratch
    state::ST           # solver state, primal grid
    dual_states::DST    # solver state, one per dual grid
end

"""
    allocate_workspace(plan) -> HelmholtzWorkspace

Buffers for one task working through `plan`.
"""
function allocate_workspace(plan::HelmholtzPlan{N,P,T}) where {N,P,T}
    grid, b, bc = plan.grid, plan.backend, plan.boundary
    # Solver state belongs here and not on the plan: it holds the buffers a solve writes through —
    # the conjugate-gradient vectors, the multigrid level arrays, the transform scratch — so a
    # threaded batch sharing one plan would have every task writing the same ones.
    state = prepare_solver(plan.solver, grid, bc)
    dual_states = ntuple(p -> prepare_solver(plan.dual_solvers[p], plan.dual_grids[p], Dirichlet()),
                         Val(P))
    return HelmholtzWorkspace(
        allocate_faces(T, grid; backend = b), allocate_faces(T, grid; backend = b),
        allocate_faces(T, grid; backend = b),
        allocate_corners(T, grid; backend = b), allocate_corners(T, grid; backend = b),
        allocate_zeros(b, T, size(grid)), state, dual_states,
    )
end

Base.show(io::IO, p::HelmholtzPlan{N,P,T}) where {N,P,T} =
    print(io, "HelmholtzPlan{", T, "}(", join(size(p.grid), "×"), ", ",
          nameof(typeof(p.boundary)), ", ", P, " rotation component", P == 1 ? "" : "s", ")")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

"""
    HelmholtzResult

Velocity-like fields use the component-last layout `(dims..., N)`; potentials and scalar
diagnostics are `(dims...)`.

The array fields are `const` and the diagnostics are not: an in-place decomposition writes its
convergence information into the result it was handed rather than allocating a fresh struct to
carry two numbers.
"""
mutable struct HelmholtzResult{N,P,T<:AbstractFloat,AV<:AbstractArray{T},AS<:AbstractArray{T,N},CV}
    const u_rot::AV
    const u_div::AV
    const u_harm::AV
    const χ::AS
    const rotation_potential::CV
    const divergence::AS
    harmonic_fraction::T
    χ_solve::SolverResult{T}
    # `P` is a type parameter so this field has a concrete type. Written as
    # `NTuple{<:Any,SolverResult{T}}` the length is unknown, the field is abstract, and every
    # assignment to it boxes the tuple.
    rot_solve::NTuple{P,SolverResult{T}}
end

@inline Base.ndims(::HelmholtzResult{N}) where {N} = N

"""
    allocate_result(plan) -> HelmholtzResult

A zeroed result matching `plan`. Array types follow the plan's buffers, so a device-resident plan
gives a device-resident result — nothing here hardcodes `Array`.
"""
function allocate_result(plan::HelmholtzPlan{N,P,T}) where {N,P,T}
    dims = size(plan.grid)
    b = plan.backend
    dummy = SolverResult{T}(false, 0, zero(T))
    Rpot = allocate_corners(T, plan.grid; backend = b)
    # The array types come from what the backend actually returned rather than being written as
    # `Array`: a device-resident result has to be expressible, and hardcoding the host type is
    # what made it not be.
    u_rot = allocate_zeros(b, T, (dims..., N))
    u_div = allocate_zeros(b, T, (dims..., N))
    u_harm = allocate_zeros(b, T, (dims..., N))
    χ = allocate_zeros(b, T, dims)
    div = allocate_zeros(b, T, dims)
    return HelmholtzResult{N,P,T,typeof(u_rot),typeof(χ),typeof(Rpot)}(
        u_rot, u_div, u_harm, χ, Rpot, div,
        zero(T), dummy, ntuple(_ -> dummy, Val(P)),
    )
end

"""
    HelmholtzBatch

`B` decompositions on one grid, stored contiguously: each field of a [`HelmholtzResult`](@ref)
gains a trailing batch axis, and `batch[b]` is a result of views onto slice `b`.

One allocation per output for the whole batch rather than one per field.
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
"""
function allocate_batch(plan::HelmholtzPlan{N,P,T}, nfields::Int) where {N,P,T}
    dims = size(plan.grid)
    bk = plan.backend
    pairs = rotation_pairs(Val(N))
    u_rot = allocate_zeros(bk, T, (dims..., N, nfields))
    u_div = allocate_zeros(bk, T, (dims..., N, nfields))
    u_harm = allocate_zeros(bk, T, (dims..., N, nfields))
    χ = allocate_zeros(bk, T, (dims..., nfields))
    div = allocate_zeros(bk, T, (dims..., nfields))
    Rpot = ntuple(p -> allocate_zeros(bk, T, (corner_dims(plan.grid, pairs[p]...)..., nfields)),
                  Val(P))
    dummy = SolverResult{T}(false, 0, zero(T))
    frac = allocate_zeros(bk, T, (nfields,))
    # Convergence diagnostics stay where they are read — on the host — while the fields follow the
    # backend; the types are parameters either way.
    χs = fill(dummy, nfields)
    rots = fill(ntuple(_ -> dummy, Val(P)), nfields)
    return HelmholtzBatch{N,P,T,typeof(u_rot),typeof(χ),typeof(Rpot),typeof(frac),typeof(χs),
                          typeof(rots)}(u_rot, u_div, u_harm, χ, Rpot, div, frac, χs, rots)
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
        batch.harmonic_fraction[b], batch.χ_solve[b], batch.rot_solve[b],
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
    dest.χ_solve[i] = src.χ_solve[j]
    dest.rot_solve[i] = src.rot_solve[j]
    return dest
end

# The diagnostics live on the batch, so a decomposed slice writes them back.
@inline function _store_diagnostics!(batch::HelmholtzBatch, b::Int, r::HelmholtzResult)
    batch.harmonic_fraction[b] = r.harmonic_fraction
    batch.χ_solve[b] = r.χ_solve
    batch.rot_solve[b] = r.rot_solve
    return batch
end

"""
    face_velocity(ws), face_divergent(ws), face_rotational(ws) -> NTuple{N,Array}

The staggered fields the last decomposition through `ws` computed: one face-normal array per
direction, `nfaces(grid, d)` long where the direction ends and `n` where it wraps.

These are where the projection is **exact**. `face_divergent` reproduces the input's divergence to
round-off and `face_rotational` is divergence-free to round-off, both discretely; the collocated
`u_div` and `u_rot` on the result are these interpolated back to cell centres, and that round trip
is the entire harmonic residue a smooth field shows — `9.6e-3` at n = 32, falling at second order.
A caller who works on the C-grid takes these and has no interpolation error at all.

Valid until the next decomposition through the same workspace, which overwrites them.
"""
face_velocity(ws::HelmholtzWorkspace) = ws.v

"""
    face_divergent(ws) -> NTuple{N,Array}

`Gχ` on faces — see [`face_velocity`](@ref).
"""
face_divergent(ws::HelmholtzWorkspace) = ws.gχ

"""
    face_rotational(ws) -> NTuple{N,Array}

`−δR` on faces, divergence-free to round-off — see [`face_velocity`](@ref).
"""
face_rotational(ws::HelmholtzWorkspace) = ws.urot

"""
    corner_rotation_potential(ws) -> NTuple{P,Array}

`R` on the corner (dual) grid, one array per rotation pair — see [`face_velocity`](@ref).
"""
corner_rotation_potential(ws::HelmholtzWorkspace) = ws.R

"""
    streamfunction(result) -> Array

The 2-D streamfunction `ψ`, on the corner (dual) grid. Only defined in 2D.
"""
streamfunction(r::HelmholtzResult{2}) = r.rotation_potential[1]
streamfunction(::HelmholtzResult{N}) where {N} = throw(ArgumentError(
    "streamfunction is only defined in 2D (got N=$N); use `vector_potential` in 3D or " *
    "`rotation_potential` generally"))

"""
    velocity_potential(result) -> Array

The scalar velocity potential `χ`, at cell centres, in any dimension.
"""
velocity_potential(r::HelmholtzResult) = r.χ

"""
    vector_potential(result) -> (A1, A2, A3)

The 3-D vector potential, the Hodge dual of the rotation potential: `A1 = R_23`, `A2 = −R_13`,
`A3 = R_12`. Only defined in 3D.
"""
function vector_potential(r::HelmholtzResult{3})
    R12, R13, R23 = r.rotation_potential
    return (R23, -1 .* R13, R12)
end
vector_potential(::HelmholtzResult{N}) where {N} =
    throw(ArgumentError("vector_potential is only defined in 3D (got N=$N)"))

# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------

"""
    helmholtz_decompose!(result, u, plan, ws = allocate_workspace(plan)) -> result

Decompose the cell-centred, component-last velocity `u` into `result`, in place.

The solver is not an argument here: it is fixed by `plan`. Accepting one would either be ignored —
a silent no-op — or force the plan's transform state to be rebuilt for a different solver, which
is the per-call cost the plan exists to remove. Pass `solver` to [`plan_helmholtz`](@ref).
"""
function helmholtz_decompose!(
    result::HelmholtzResult{N,P,T}, u::AbstractArray, plan::HelmholtzPlan{N,P,T},
    ws::HelmholtzWorkspace = allocate_workspace(plan);
    backend = plan.backend,
) where {N,P,T}
    grid, bc = plan.grid, plan.boundary
    size(u) == (size(grid)..., N) || throw(DimensionMismatch(
        "velocity array size $(size(u)) does not match (dims..., N) = $((size(grid)..., N))"))

    # Both the solver choice and its transform state come from the plan: neither depends on `u`,
    # so a batch resolves and builds them once in total rather than once per field.
    concrete = plan.solver
    state = ws.state

    # 1. To faces, where the projection is exact.
    to_faces!(ws.v, u, grid, bc, plan.metrics; backend = backend)

    # 2. δ = D u, and the solvability condition where the constants are in L's null space.
    divergence!(result.divergence, ws.v, grid, bc, plan.metrics; backend = backend)
    plan.coefficients.singular && project_out_constant!(result.divergence, grid, plan.coefficients)

    # 3. L χ = δ, then u_div = G χ. `D u_div = L χ = δ` now holds exactly.
    result.χ_solve = solve_poisson!(result.χ, result.divergence, grid, concrete;
                                    boundary = bc, coefficients = plan.coefficients, state = state,
                                    backend = backend)
    gradient!(ws.gχ, result.χ, grid, bc, plan.metrics; backend = backend)

    # 4. W = curl u on the corners, and L R = W on each dual grid. The remainder u − u_div is
    #    already exactly divergence-free; this is what separates its rotational part from the
    #    harmonic one.
    curl!(ws.W, ws.v, grid, bc, plan.metrics; backend = backend)
    rot = ntuple(Val(P)) do p
        dg, dc = plan.dual_grids[p], plan.dual_coefficients[p]
        dc.singular && project_out_constant!(ws.W[p], dg, dc)
        # `Dirichlet` here regardless of the primal condition — see `plan_helmholtz`.
        res = solve_poisson!(ws.R[p], ws.W[p], dg, plan.dual_solvers[p];
                             boundary = Dirichlet(), coefficients = dc,
                             state = ws.dual_states[p], backend = backend)
        copyto!(result.rotation_potential[p], ws.R[p])
        res
    end
    result.rot_solve = rot
    rotational_velocity!(ws.urot, ws.R, grid, bc, plan.metrics; backend = backend)

    # 5. Back to centres, and the harmonic remainder.
    to_centres!(result.u_div, ws.gχ, grid, bc, plan.metrics; backend = backend)
    to_centres!(result.u_rot, ws.urot, grid, bc, plan.metrics; backend = backend)
    @. result.u_harm = u - result.u_div - result.u_rot
    _zero_inactive!(result.u_harm, grid, Val(N); backend = backend)

    den = velocity_norm(u, grid, ws.scratch)
    result.harmonic_fraction = iszero(den) ? zero(T) :
                               velocity_norm(result.u_harm, grid, ws.scratch) / den
    return result
end

function _zero_inactive!(U, grid, ::Val{N}; backend = ComputationalBackends.SerialBackend()) where {N}
    T = eltype(U)
    msk = FlowGeometries.Grids.mask(grid)
    cart = CartesianIndices(size(grid))
    # Component views are built once, outside the loop: taking them per index would rebuild the
    # same `N` views for every cell, and a view is not something a device kernel should be forming.
    comps = ntuple(c -> _component(U, c, Val(N)), Val(N))
    FlowGeometries.Execution.run_indices(length(cart), backend) do lin
        @inbounds begin
            I = cart[lin]
            if !msk[I]
                for c in 1:N
                    comps[c][I] = zero(T)
                end
            end
        end
        return nothing
    end
    return U
end

"""
    count_holes(grid) -> Int

Number of inactive regions fully enclosed by active cells — an estimate of the first Betti
number `b₁` of the active region, i.e. the dimension of the harmonic subspace.

This is what makes [`harmonic_fraction`](@ref HelmholtzResult) readable: a large harmonic part on
a domain with `count_holes == 0` is a boundary-circulation effect, while on one with holes it is
the circulation around them, which no single-valued potential can represent.

The flood fill is `FlowGeometries.Connectivity.connected_components`, which walks the grid's own
wrapping — so on a periodic direction a region running off one side and back on the other is one
region, and "touching the boundary" is asked only of directions that actually have one.
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
    velocity_norm(U, grid, scratch) -> T

Measure-weighted `‖U‖ = sqrt(∫|U|² dV)` over the active cells.
"""
function velocity_norm(U, grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, scratch) where {G,T,N}
    msk = FlowGeometries.Grids.mask(grid)
    meas = FlowGeometries.Grids.measure(grid)
    fill!(scratch, zero(T))
    for c in 1:N
        scratch .+= abs2.(_component(U, c, Val(N)))
    end
    # The grid's own measure, unlike a `LaplacianCoefficients` one, is nonzero on inactive cells.
    return sqrt(lazy_sum((m, s, w) -> ifelse(m, s * w, zero(T)), msk, scratch, meas))
end

"""
    helmholtz_decompose(u, grid; boundary = Neumann(), solver = AutoSolver()) -> HelmholtzResult

Decompose a cell-centred, component-last velocity field on `grid`.

Builds a [`HelmholtzPlan`](@ref) and discards it. A caller decomposing more than one field on the
same grid should build the plan once and call [`helmholtz_decompose!`](@ref) — the plan holds the
grid's Laplacian coefficients, which are the expensive part.
"""
function helmholtz_decompose(
    u::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid;
    boundary::AbstractBoundaryCondition = Neumann(),
    solver::AbstractPoissonSolver = AutoSolver(),
)
    plan = plan_helmholtz(grid; boundary = boundary, solver = solver)
    return helmholtz_decompose!(allocate_result(plan), u, plan, allocate_workspace(plan))
end

# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

"""
    _extension_loaded(name) -> Bool

Whether one of this package's extensions is loaded. `hasmethod` cannot answer this — the dispatch
stubs exist unconditionally, so it is always `true`.
"""
@inline _extension_loaded(name::Symbol) = Base.get_extension(@__MODULE__, name) !== nothing

@inline _threading_available() = _extension_loaded(:HelmholtzDecompositionOhMyThreadsExt)

"""
    helmholtz_decompose_batch(plan, fields; backend = AutoBackend())

Decompose many fields sharing one grid. The fields are independent, so the batch is the parallel
axis: `ThreadedBackend` (OhMyThreads ext), `DistributedBackend` (Distributed ext) and `MPIBackend`
(MPI ext) each spread it across their workers; a serial backend maps sequentially. Results come
back in input order.

One `plan` serves the whole batch — its Laplacian coefficients are identical for every field and
are the expensive part — while each task takes its own [`HelmholtzWorkspace`](@ref), since those
buffers are written through.

`backend` here is the *batch* axis. A parallel one pins the decomposition inside each field to
serial; a serial one leaves the inner loops on the plan's own backend, so a short batch of large
fields still parallelises — just along the other axis.
"""
function helmholtz_decompose_batch(
    plan::HelmholtzPlan, fields;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    kwargs...,
)
    batch = allocate_batch(plan, length(fields))
    return helmholtz_decompose_batch!(batch, fields, plan; backend = backend, kwargs...)
end

"""
    helmholtz_decompose_batch!(batch, fields, plan; backend = AutoBackend()) -> batch

Decompose into a preallocated [`HelmholtzBatch`](@ref).
"""
function helmholtz_decompose_batch!(
    batch::HelmholtzBatch, fields, plan::HelmholtzPlan;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    kwargs...,
)
    length(batch) == length(fields) || throw(DimensionMismatch(
        "batch holds $(length(batch)) slots but $(length(fields)) fields were given"))
    b = isempty(fields) ? ComputationalBackends.SerialBackend() :
                          _resolve_batch_backend(backend, fields)
    return _decompose_batch!(b, batch, fields, plan; kwargs...)
end

# One slice, into its views. Kept in one place so every backend's loop body is identical.
@inline function _decompose_slice!(batch::HelmholtzBatch, i::Int, field, plan, ws; kwargs...)
    r = helmholtz_decompose!(batch[i], field, plan, ws; kwargs...)
    return _store_diagnostics!(batch, i, r)
end

"""
    _resolve_batch_backend(backend, fields) -> AbstractExecutionBackend

Resolve `Auto` over the batch axis against real capability: more than one field, more than one
thread, and the threading extension actually loaded. A concrete backend passes through and is then
honoured or refused — never quietly replaced.
"""
_resolve_batch_backend(b::ComputationalBackends.AbstractExecutionBackend, _) = b
function _resolve_batch_backend(::ComputationalBackends.AbstractAutoBackend, fields)
    parallel = length(fields) > 1 && Threads.nthreads() > 1 && _threading_available()
    return parallel ? ComputationalBackends.ThreadedBackend() : ComputationalBackends.SerialBackend()
end

# Sequential: one workspace, reused across fields, since nothing runs concurrently.
function _decompose_batch!(::ComputationalBackends.AbstractSerialBackend, batch::HelmholtzBatch,
                           fields, plan::HelmholtzPlan; kwargs...)
    ws = allocate_workspace(plan)
    for (i, f) in enumerate(fields)
        _decompose_slice!(batch, i, f, plan, ws; kwargs...)
    end
    return batch
end

_decompose_batch!(b::ComputationalBackends.AbstractExecutionBackend, ::HelmholtzBatch, fields,
                  ::HelmholtzPlan; kwargs...) =
    throw(ArgumentError(_unsupported_backend_message(b, "helmholtz_decompose_batch")))

"""
    _unsupported_backend_message(backend, entry) -> String

Why `entry` cannot honour `backend`, and what to load so it can. Reaching this is always an error:
a backend named explicitly is executed as named or refused, never downgraded.
"""
function _unsupported_backend_message(backend, entry::AbstractString)
    pkg = _backend_package(backend)
    pkg === nothing && return "$entry has no implementation for $(typeof(backend))."
    return "$entry cannot run on $(typeof(backend)): the $pkg extension is not loaded. Run " *
           "`using $pkg` first, or pass an explicit `backend = ComputationalBackends.SerialBackend()`."
end

_backend_package(::ComputationalBackends.AbstractThreadedBackend) = "OhMyThreads"
_backend_package(::ComputationalBackends.AbstractDistributedBackend) = "Distributed"
_backend_package(::ComputationalBackends.AbstractMPIBackend) = "MPI"
_backend_package(::ComputationalBackends.AbstractExecutionBackend) = nothing
