# Spherical Poisson solves and the spectral decomposition through NUFSHT, on any node set covering the
# sphere: analysis, divide by `−ℓ(ℓ+1)/R²`, synthesis. The analysis is the least-squares fit
# `NUFSHT.nusht_solve!`, each node weighted by its cell's measure, the inverse of synthesis at any node
# set; `nusht_type1!` is its adjoint, the inverse only on the Fejér first-rule grid. Plans are built under
# the `execution` each call is given, so a `GPUBackend` puts their nodes and buffers in its memory; every
# array here is allocated from the plans' memory and the coefficient-space steps are broadcasts.
module HelmholtzDecompositionNUFSHTExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries: FlowGeometries as FG
using FlowTransformBindings: FlowTransformBindings as FTB
using NUFSHT: NUFSHT

"""
    _resolve_lmax(solver, grid) -> Int

The expansion degree for `grid`, when the solver was built without one, within `(lmax+1)² ≤ M` for `M`
nodes: the band limit of a sampling that ties one to its latitude count (`has_bandlimit`), one below
the latitude count of another rectilinear grid, and on any other layout the largest degree leaving four
nodes per coefficient, `(lmax+1)² ≤ M/4`. For `M` random nodes and `n` coefficients the fit's matrix
has condition number near `(1 + √(n/M)) / (1 − √(n/M))` (Marchenko–Pastur), 3 at a quarter, and
unbounded as `n` approaches `M`.
"""
function _resolve_lmax(solver::HD.SphericalNUSHTSolver, grid)
    solver.lmax === nothing || return solver.lmax
    M = length(FG.Grids.mask(grid))
    return max(1, min(isqrt(M) - 1, _grid_degree(FG.Grids.sampling(grid), grid, M)))
end

_grid_degree(s::FG.SphericalSampling.AbstractSphericalSampling, grid, M::Int) =
    FG.SphericalSampling.has_bandlimit(s) ? FG.SphericalSampling.bandlimit(s, _latitude_count(grid)) :
    _grid_degree(nothing, grid, M)
_grid_degree(::Nothing, grid::FG.Grids.StructuredGrid, M::Int) = size(grid, 2) - 1
_grid_degree(::Nothing, grid, M::Int) = isqrt(M ÷ 4) - 1

_latitude_count(grid::FG.Grids.StructuredGrid) = size(grid, 2)
_latitude_count(grid::FG.Grids.RingGrid) = FG.Grids.nrings(grid)

# The expansion is in spherical harmonics over the whole sphere, so the samples must cover it and
# none may be masked out; a covering set has no boundary, making a boundary condition vacuous.
HD.supports_boundary(::HD.SphericalNUSHTSolver, ::FG.Operators.AbstractBoundaryCondition) = true
HD.requires_full_domain(::HD.SphericalNUSHTSolver) = true

# Named, the solver takes any spherical node set, the caller stating that it covers the sphere;
# `AutoSolver` takes it where the grid's type shows that it does.
HD.library_loaded(::HD.SphericalNUSHTSolver) = true
HD.supports_sampling(::HD.SphericalNUSHTSolver, grid) = false
HD.supports_sampling(::HD.SphericalNUSHTSolver, ::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry}) = true
HD.sampling_known(::HD.SphericalNUSHTSolver, grid) = false
HD.sampling_known(::HD.SphericalNUSHTSolver, grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry}) =
    FG.Grids.covers_sphere(grid)

HD._sampling_message(::HD.SphericalNUSHTSolver, grid) =
    "SphericalNUSHTSolver expands in spherical harmonics and takes a spherical grid; this grid is " *
    "$(nameof(typeof(grid)))."

"""
    _spherical_nodes(grid) -> (θ, λ)

Colatitude and longitude of every node, flattened in the grid's own cell order, as host vectors: a plan
takes its nodes from the host and places them where it runs.

A rectilinear grid is a tensor product and is expanded from its two axes. Any other layout stores
no coordinate vectors and supplies its cloud through `Grids.coords`.
"""
function _spherical_nodes(grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2}) where {T}
    nlon, nlat = size(grid)
    λ = Array(FG.Grids.coordinates(grid, 1))
    φ = Array(FG.Grids.coordinates(grid, 2))
    θ = Vector{T}(undef, nlon * nlat)
    Λ = Vector{T}(undef, nlon * nlat)
    k = 0
    @inbounds for j in 1:nlat, i in 1:nlon
        k += 1
        θ[k] = T(π) / 2 - T(φ[j])          # latitude → colatitude
        Λ[k] = T(λ[i])
    end
    return θ, Λ
end

function _spherical_nodes(grid::FG.Grids.UnstructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2}) where {T}
    λ = Array(FG.Grids.coordinates(grid, 1))
    φ = Array(FG.Grids.coordinates(grid, 2))
    return T(π) / 2 .- T.(φ), T.(λ)
end

# A pixelization names each cell by one integer and stores no coordinate vectors — `coordinates`
# raises on a HEALPix grid and directs the caller here. `coords` reads one cell and is defined for
# every layout, so this one method serves HEALPix, the ring grids, the cubed sphere, the
# icosahedral grid and Yin-Yang.
function _spherical_nodes(grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T}) where {T}
    n = length(FG.Grids.mask(grid))
    θ = Vector{T}(undef, n)
    Λ = Vector{T}(undef, n)
    @inbounds for i in 1:n
        p = FG.Grids.coords(Tuple, grid, i)
        Λ[i] = T(p[1])
        θ[i] = T(π) / 2 - T(p[2])
    end
    return θ, Λ
end

"""
    _warn_underdetermined(lmax, M, s = 0)

Warn once when `M` nodes leave the `(lmax+1)² − s²` coefficients of a spin-`s` expansion
undetermined.

Past this line the samples stop pinning down the coefficient vector, and the rotational/divergent
split reads that vector directly, so the split can be wrong while the fit reproduces every sample
and the residual sits at round-off. The split holds to round-off well past this line and degrades
further out, and the crossing moves with the field and the node layout, so this reports the counts
and leaves the choice with the caller.
"""
@inline function _warn_underdetermined(lmax::Int, M::Int, s::Int = 0)
    ncoef = (lmax + 1)^2 - s^2
    ncoef <= M && return nothing
    @warn("lmax=$lmax needs $ncoef coefficients and this point set has $M nodes, so the samples " *
          "do not determine the expansion. The fit reproduces them and its rotational/divergent " *
          "split may not. Lower `lmax`, or leave it unset to size it from the grid.", maxlog = 1)
    return nothing
end

# A per-degree factor `f(ℓ)`, `ℓ = 0:lmax`, as an `(lmax+1, 1)` column in `mem`'s memory, to broadcast
# along the degree rows of a spin coefficient array; a per-order one `f(m)`, `m = −lmax:lmax`, as a
# `(1, 2lmax+1)` row along its order columns.
_degree_column(f::F, mem, ::Type{T}, lmax::Int) where {F,T} =
    copyto!(similar(mem, T, lmax + 1, 1), reshape(T[f(ℓ) for ℓ in 0:lmax], :, 1))
_order_row(f::F, mem, ::Type{T}, lmax::Int) where {F,T} =
    copyto!(similar(mem, T, 1, 2lmax + 1), reshape(T[f(m) for m in (-lmax):lmax], 1, :))

# ---------------------------------------------------------------------------
# The Poisson solve
# ---------------------------------------------------------------------------

"""
    SphericalNUSHTState

The NUFSHT plan with its nodes uploaded, the plan with each node weighted by its cell's measure and the
LSMR workspace its fit runs in, the flattened right-hand side and output, the coefficient array —
`(lmax+1) × (2lmax+1)` in FastSphericalHarmonics' `sph_mode` layout — and the inverse eigenvalue
`−R²/(ℓ(ℓ+1))` of each of its slots, all in the plan's memory.

One per task: every buffer here is written during a solve. It owns the plan's NUFFT plans, which
[`close!`](@ref HelmholtzDecomposition.close!) releases.
"""
struct SphericalNUSHTState{T,P,O,W,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    plan::P
    op::O
    ws::W
    rhs::V
    out::V
    coeffs::M
    inv_eig::M
    lmax::Int
end

Base.show(io::IO, s::SphericalNUSHTState{T}) where {T} =
    print(io, "SphericalNUSHTState{", T, "}(", length(s.rhs), " nodes, lmax = ", s.lmax, ")")

HD.close!(s::SphericalNUSHTState) = (NUFSHT.close!(s.plan); s)
HD._close_state!(s::SphericalNUSHTState) = HD.close!(s)

# `−R²/(ℓ(ℓ+1))` at every slot of the `sph_mode` layout, slot `(i, j)` holding degree `ℓ = i − 1 + j ÷ 2`;
# zero at `ℓ = 0`, the constant the Laplacian annihilates, and on the slots past `lmax`.
function _inverse_eigenvalues(::Type{T}, lmax::Int, R::T) where {T}
    E = zeros(T, lmax + 1, 2lmax + 1)
    for j in axes(E, 2), i in axes(E, 1)
        ℓ = i - 1 + j ÷ 2
        1 <= ℓ <= lmax && (E[i, j] = -R^2 / T(ℓ * (ℓ + 1)))
    end
    return E
end

# `grid` is in the memory `execution` allocates, as a plan's own grid is, so its measure broadcasts into
# the plan's.
function HD.prepare_solver(solver::HD.SphericalNUSHTSolver,
                           grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T},
                           ::FG.Operators.AbstractBoundaryCondition;
                           execution = CB.SerialBackend(), shared = nothing) where {T}
    lmax = _resolve_lmax(solver, grid)
    θ, Λ = _spherical_nodes(grid)
    _warn_underdetermined(lmax, length(θ))
    R = T(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    # The field element type is positional, as it is for `zeros(T, …)`; a real `T` selects NUFSHT's real
    # specialization.
    plan = NUFSHT.make_plan(T, θ, Λ, lmax; tol = solver.tol, nufft = solver.nufft, execution = execution)
    mem = NUFSHT.allocate_coefficients(plan)
    w = similar(mem, T, size(grid)...)
    w .= FG.Grids.measure(grid)
    op = FTB.RangeWeighted(plan, vec(w))
    rhs = similar(mem, T, length(θ))
    coeffs = similar(mem, T, lmax + 1, 2lmax + 1)
    inv_eig = copyto!(similar(coeffs), _inverse_eigenvalues(T, lmax, R))
    return SphericalNUSHTState(plan, op, FTB.LSMRWorkspace(op), rhs, similar(rhs), coeffs, inv_eig, lmax)
end

function HD.solve_poisson!(
    Φ::AbstractArray{T},
    RHS::AbstractArray{T},
    grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T},
    solver::HD.SphericalNUSHTSolver;
    boundary::FG.Operators.AbstractBoundaryCondition = FG.Operators.Neumann(),
    state::SphericalNUSHTState = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat}
    iters, resid, converged = _inverse!(Φ, RHS, solver, state)
    return HD.SolverResult{T}(converged, iters, T(resid))
end

function HD.spectral_inverse!(
    Φ::AbstractArray{T}, RHS::AbstractArray{T},
    grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T},
    solver::HD.SphericalNUSHTSolver, state::SphericalNUSHTState,
) where {T<:AbstractFloat}
    _inverse!(Φ, RHS, solver, state)
    return Φ
end

# Fit, divide by the eigenvalues, synthesize; returns the fit's iterations, residual and convergence.
# The grid's cell order is the plan's node order, so the right-hand side and the result cross as flat
# copies.
function _inverse!(Φ, RHS, solver::HD.SphericalNUSHTSolver, state::SphericalNUSHTState{T}) where {T}
    copyto!(state.rhs, RHS)
    fill!(state.coeffs, zero(T))
    _, iters, resid, converged = NUFSHT.nusht_solve!(state.coeffs, state.rhs, state.op; ws = state.ws,
                                                     rtol = solver.rtol, maxiter = solver.maxiter)
    state.coeffs .*= state.inv_eig
    NUFSHT.nusht_type2!(state.out, state.coeffs, state.plan)
    copyto!(Φ, state.out)
    return iters, resid, converged
end

# ---------------------------------------------------------------------------
# Vector Hodge decomposition at arbitrary spherical nodes
# ---------------------------------------------------------------------------
#
# The tangent field `V = u_θ + i u_φ` has spin weight 1. Expanding it in spin(+1) and spin(−1)
# harmonics, the symmetric combination of the two coefficient sets is the E-type (rotational) part
# and the antisymmetric one the B-type (divergent) part. Synthesising each back gives the split at
# the nodes with no finite-difference stencil anywhere.
#
# `ₛY_{ℓm}` conjugates to spin `−s`, so no reality condition closes for `s ≠ 0` and the fit runs
# over the full complex coefficient array: the plans carry a complex element type.

"""
    NUSHTSpectralState

What the spin-weighted decomposition reuses on one spherical node set, in the plans' memory: the spin-1 and
spin-0 plans of its nodes, built under the backend of the call; each node's measure and the spin-1 plan
weighted by it, with the LSMR workspace its fit runs in; the velocity staged as two columns and as the
spin-1 field; the fit's coefficients and the four coefficient arrays formed from them; the four fields
synthesized from those; the per-order sign and the per-degree factors `R/L` and `L/R`; the expansion degree
and the radius. One per task; [`close!`](@ref HelmholtzDecomposition.close!) releases the plans.
"""
struct NUSHTSpectralState{T,PP,P0,W,O,WS,S,V,A,F}
    planp::PP
    plan0::P0
    w::W
    op::O
    ws::WS
    stage::S
    V::V
    coeffs::NTuple{5,A}
    fields::NTuple{4,V}
    sgn::F
    RoverL::F
    LoverR::F
    lmax::Int
    R::T
end

HD.close!(s::NUSHTSpectralState) = (NUFSHT.close!(s.planp); NUFSHT.close!(s.plan0); s)

function HD.prepare_spectral(solver::HD.SphericalNUSHTSolver,
                             grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T};
                             execution = CB.AutoBackend()) where {T}
    exec = HD.resolve_execution_backend(execution)
    θ, Λ = _spherical_nodes(grid)
    lmax = _resolve_lmax(solver, grid)
    _warn_underdetermined(lmax, length(θ), 1)
    R = T(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    plan(s) = NUFSHT.make_spin_plan(Complex{T}, θ, Λ, lmax, s; tol = solver.tol, nufft = solver.nufft,
                                    execution = exec)
    planp = plan(+1)
    plan0 = try
        plan(0)
    catch
        NUFSHT.close!(planp)
        rethrow()
    end
    mem = NUFSHT.allocate_coefficients(planp)
    CT = Complex{T}
    M = length(θ)
    w = copyto!(similar(mem, T, size(grid)...), Array{T}(FG.Grids.measure(grid)))
    op = FTB.RangeWeighted(planp, vec(w))
    coeffs = ntuple(_ -> fill!(similar(mem, CT, lmax + 1, 2lmax + 1), zero(CT)), Val(5))
    fields = ntuple(_ -> similar(mem, CT, M), Val(4))
    sgn = _order_row(m -> isodd(m) ? one(T) : -one(T), mem, T, lmax)
    RoverL = _degree_column(ℓ -> iszero(ℓ) ? zero(T) : R / sqrt(T(ℓ * (ℓ + 1))), mem, T, lmax)
    LoverR = _degree_column(ℓ -> sqrt(T(ℓ * (ℓ + 1))) / R, mem, T, lmax)
    return NUSHTSpectralState(planp, plan0, w, op, FTB.LSMRWorkspace(op), similar(mem, T, M, 2), similar(mem, CT, M),
                              coeffs, fields, sgn, RoverL, LoverR, lmax, R)
end

HD.allocate_result(st::NUSHTSpectralState{T}, grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T}) where {T} =
    HD._spectral_result(st.stage, T, size(grid), Val(2), Val(1))

"""
    _spin_hodge!(result, solver, state, U, grid) -> result

The decomposition of `U`, component-last `(u_east, u_north)` over `grid`'s nodes, through `state`'s spin-1
and spin-0 plans of those nodes, in the plans' memory: `U` is staged there whole and every part is written
into `result` there. The fit weights each node by its cell's measure.

With `ₛY_ℓm = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` (NUFSHT's convention) and the Wigner-d derivative relations,
`∇Y_ℓm` has `u_θ + i u_φ = L ₁Y_ℓm` and `u_θ − i u_φ = −L ₋₁Y_ℓm`, `L = √(ℓ(ℓ+1))`; `k̂ × ∇` multiplies
both by `i`. So for `u = (∇χ + k̂×∇ψ)/R` the fits of `V = u_θ + i u_φ` and `V̄` are `₊a = L(χ + iψ)/R`
and `₋a = −L(χ − iψ)/R`: their mean is the rotational part and their half difference the divergent
one, `χ + iψ = R ₊a/L`, and `∇·u = −L·(₊a − ₋a)/(2R)` per mode.

`d^ℓ_{m,−s} = (−1)^{m+s} d^ℓ_{−m,s}` gives `conj(ₛY_ℓm) = (−1)^{m+s} ₋ₛY_{ℓ,−m}`, so a real `u` has
`₋a_{ℓm} = (−1)^{m+1} conj(₊a_{ℓ,−m})`: one fit gives both. Each part's coefficients obey the same
relation, so its spin −1 field is the conjugate of its spin 1 field and `(u_θ, u_φ) = (Re V, Im V)`.
"""
function _spin_hodge!(result::HD.HelmholtzResult{2,1,T}, solver::HD.SphericalNUSHTSolver,
                      st::NUSHTSpectralState{T}, U::AbstractArray{T}, grid) where {T}
    planp, plan0, w = st.planp, st.plan0, st.w
    sz = size(grid)
    M = prod(sz)
    size(U) == (sz..., 2) || throw(DimensionMismatch(
        "velocity array size $(size(U)) does not match (dims..., 2) = $((sz..., 2))"))
    HD._check_spectral_result(result, sz, Val(2))

    # `U` in the plans' memory, `(u_east, u_north)` as two columns; with `θ̂` south and `φ̂` east,
    # `V = u_θ + i u_φ = −u_north + i u_east`.
    stage = copyto!(st.stage, U)
    V = st.V
    V .= complex.(.-view(stage, :, 2), view(stage, :, 1))

    ap, sym, anti, cΦ, cδ = st.coeffs
    fill!(ap, zero(eltype(ap)))
    _, iters, resid, ok = NUFSHT.nusht_solve_spin!(ap, V, st.op; ws = st.ws, rtol = solver.rtol,
                                                   maxiter = solver.maxiter)
    fit = HD.SolverResult{T}(ok, iters, T(resid))

    # Column `m` of `apr` holds `₊a_{ℓ,−m}`, so `₋a = sgn · conj(apr)`. The `|m| > ℓ` slots are zero in
    # `ap` and stay zero in every array formed from it.
    apr = view(ap, :, reverse(axes(ap, 2)))
    sym .= (ap .+ st.sgn .* conj.(apr)) ./ 2            # the rotational part
    anti .= (ap .- st.sgn .* conj.(apr)) ./ 2           # the divergent part
    cΦ .= st.RoverL .* ap                               # χ + iψ
    cδ .= .-st.LoverR .* anti                           # ∇·u

    rot, div, Φ, δ = st.fields
    NUFSHT.nusht_type2_spin!(rot, sym, planp)
    NUFSHT.nusht_type2_spin!(div, anti, planp)
    NUFSHT.nusht_type2_spin!(Φ, cΦ, plan0)
    NUFSHT.nusht_type2_spin!(δ, cδ, plan0)

    _store_part!(result.u_rot, rot, M)
    _store_part!(result.u_div, div, M)
    fill!(result.u_harm, zero(T))
    vec(result.χ) .= real.(Φ)
    vec(result.rotation_potential[1]) .= imag.(Φ)
    vec(result.divergence) .= real.(δ)
    Us = reshape(stage, sz..., 2)
    result.harmonic_fraction = zero(T)
    result.residual_fraction = HD._measure_ratio(Broadcast.broadcasted((u, a, b) -> u - a - b, Us, result.u_rot, result.u_div),
                                                 Us, w)
    result.χ_solve = fit
    result.rot_solve = (fit,)
    return result
end

# The part with spin-1 field `Vp = u_θ + i u_φ` into `u`, component-last `(u_east, u_north)` over `M`
# nodes: with `θ̂` south and `φ̂` east, `u_east = u_φ = Im Vp` and `u_north = −u_θ = −Re Vp`.
function _store_part!(u, Vp, M::Int)
    u2 = reshape(u, M, 2)
    view(u2, :, 1) .= imag.(Vp)
    view(u2, :, 2) .= .-real.(Vp)
    return u
end

"""
    _decompose_spectral!(solver, geometry, result, U, grid, execution, state) -> result

Helmholtz decomposition of a tangent velocity field on any spherical node set, through `state`'s plans,
into `result` in their memory. `U` is component-last carrying `(u_east, u_north)`, and the parts are
written in that layout. `S²` is simply connected, so `u_harm` is zero; the fit's residual at the nodes is
`residual_fraction`.
"""
HD._decompose_spectral!(solver::HD.SphericalNUSHTSolver, ::FG.Geometry.AbstractSphericalGeometry,
                        result::HD.HelmholtzResult, U::AbstractArray{T},
                        grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T}, execution,
                        st::NUSHTSpectralState) where {T} =
    _spin_hodge!(result, solver, st, U, grid)

end # module
