# Spherical Poisson solves and the spectral decomposition through NUFSHT, on any node set covering the
# sphere: analysis, divide by `−ℓ(ℓ+1)/R²`, synthesis. The analysis is the least-squares fit
# `NUFSHT.nusht_solve!`, each node weighted by its cell's measure, the inverse of synthesis at any node
# set; `nusht_type1!` is its adjoint, the inverse only on the Clenshaw–Curtis grid. Plans are built under
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

The expansion degree for `grid`, when the solver was built without one.

Two bounds, both from the grid. The node count caps it at `(lmax+1)² ≤ M`, which is conservative —
the fit and its split hold past that — and it needs no knowledge of the field. The sampling caps it
at the degree the node set represents, which `FlowGeometries` answers from the recipe where the grid
carries one and from the latitude count otherwise.
"""
function _resolve_lmax(solver::HD.SphericalNUSHTSolver, grid)
    solver.lmax === nothing || return solver.lmax
    npoints = length(FG.Grids.mask(grid))
    from_nodes = isqrt(npoints) - 1
    s = FG.Grids.sampling(grid)
    nlat = ndims(grid) == 2 ? size(grid, 2) : npoints
    from_sampling = s === nothing ? nlat - 1 : FG.SphericalSampling.bandlimit(s, nlat)
    return max(1, min(from_nodes, from_sampling))
end

# The expansion is in spherical harmonics over the whole sphere, so the samples must cover it and
# none may be masked out; a covering set has no boundary, making a boundary condition vacuous.
HD.supports_boundary(::HD.SphericalNUSHTSolver, ::FG.Operators.AbstractBoundaryCondition) = true
HD.requires_full_domain(::HD.SphericalNUSHTSolver) = true

"""
    _covers_sphere(grid) -> Bool

Whether `grid`'s nodes cover `S²`, which a fit in spherical harmonics needs: a field known on part of
the sphere does not determine `Δ⁻¹` of it even there, and the fit's coefficients are ill-conditioned off
the sampled region (the Slepian concentration problem).

A rectilinear grid covers it when longitude wraps and each pole lies within one and a half spacings of
its two outermost latitudes. A cell-centred grid sits about half a spacing from the pole and the dual
grid of one about a whole spacing, each half a spacing inside the bound, on a stretched axis too. Any
other layout tiles the sphere with its cells, and covers it when their total measure is `4πR²`.
"""
_covers_sphere(grid) = false

function _covers_sphere(
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
) where {T}
    FG.Grids.isperiodic(grid, 1) || return false
    φ = FG.Grids.coordinates(grid, 2)
    n = length(φ)
    n >= 2 || return false
    # The southmost and northmost latitudes with their inward neighbours, whichever way the axis runs.
    south, south2, north2, north = T(φ[1]), T(φ[2]), T(φ[n - 1]), T(φ[n])
    south > north && ((south, south2, north2, north) = (north, north2, south2, south))
    half = T(π) / 2
    return south + half <= 3 * (south2 - south) / 2 && half - north <= 3 * (north - north2) / 2
end

function _covers_sphere(
    grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T},
) where {T}
    R = T(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    return isapprox(T(sum(FG.Grids.measure(grid))), 4 * T(π) * R^2; rtol = sqrt(eps(T)))
end

HD.supports_sampling(::HD.SphericalNUSHTSolver, grid) = _covers_sphere(grid)

function HD._sampling_message(::HD.SphericalNUSHTSolver, grid)
    detail = if grid isa FG.Grids.StructuredGrid && ndims(grid) == 2
        lo, hi = extrema(FG.Grids.coordinates(grid, 2))
        "longitude wraps: $(FG.Grids.isperiodic(grid, 1)); latitude spans " *
        "$(round(lo; digits = 3))…$(round(hi; digits = 3)) of ±π/2"
    else
        string(nameof(typeof(grid)))
    end
    return "SphericalNUSHTSolver expands in spherical harmonics over the whole sphere, and these " *
           "nodes do not cover it ($detail). Analysis integrates over S² and the inverse Laplacian " *
           "there is nonlocal, so a regional patch does not determine the solution even on the " *
           "patch. Solve a patch as a boundary value problem with the iterative solver."
end

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
                           grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
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
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
    solver::HD.SphericalNUSHTSolver;
    boundary::FG.Operators.AbstractBoundaryCondition = FG.Operators.Neumann(),
    state::SphericalNUSHTState = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat}
    iters, resid, converged = _inverse!(Φ, RHS, solver, state)
    return HD.SolverResult{T}(converged, iters, T(resid))
end

function HD.spectral_inverse!(
    Φ::AbstractMatrix{T}, RHS::AbstractMatrix{T},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
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
    _spin_hodge(solver, planp, plan0, U, grid, lmax, R) -> HelmholtzResult

The decomposition of `U`, component-last `(u_east, u_north)` over `grid`'s nodes, through the spin-1 plan
`planp` and the spin-0 plan `plan0` of those nodes, in the plans' memory: `U` is staged there whole and
every part comes back there. The fit weights each node by its cell's measure.

With `ₛY_ℓm = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` (NUFSHT's convention) and the Wigner-d derivative relations,
`∇Y_ℓm` has `u_θ + i u_φ = L ₁Y_ℓm` and `u_θ − i u_φ = −L ₋₁Y_ℓm`, `L = √(ℓ(ℓ+1))`; `k̂ × ∇` multiplies
both by `i`. So for `u = (∇χ + k̂×∇ψ)/R` the fits of `V = u_θ + i u_φ` and `V̄` are `₊a = L(χ + iψ)/R`
and `₋a = −L(χ − iψ)/R`: their mean is the rotational part and their half difference the divergent
one, `χ + iψ = R ₊a/L`, and `∇·u = −L·(₊a − ₋a)/(2R)` per mode.

`d^ℓ_{m,−s} = (−1)^{m+s} d^ℓ_{−m,s}` gives `conj(ₛY_ℓm) = (−1)^{m+s} ₋ₛY_{ℓ,−m}`, so a real `u` has
`₋a_{ℓm} = (−1)^{m+1} conj(₊a_{ℓ,−m})`: one fit gives both. Each part's coefficients obey the same
relation, so its spin −1 field is the conjugate of its spin 1 field and `(u_θ, u_φ) = (Re V, Im V)`.
"""
function _spin_hodge(solver::HD.SphericalNUSHTSolver, planp, plan0, U::AbstractArray{T}, grid, lmax::Int,
                     R::T) where {T}
    CT = Complex{T}
    sz = size(grid)
    M = prod(sz)
    size(U) == (sz..., 2) || throw(DimensionMismatch(
        "velocity array size $(size(U)) does not match (dims..., 2) = $((sz..., 2))"))
    mem = NUFSHT.allocate_coefficients(planp)

    # `U` in the plans' memory, `(u_east, u_north)` as two columns; with `θ̂` south and `φ̂` east,
    # `V = u_θ + i u_φ = −u_north + i u_east`. `w` holds the cells' measures.
    stage = copyto!(similar(mem, T, M, 2), U)
    V = similar(mem, CT, M)
    V .= complex.(.-view(stage, :, 2), view(stage, :, 1))
    w = copyto!(similar(mem, T, sz...), Array{T}(FG.Grids.measure(grid)))

    ap = fill!(similar(mem, CT, lmax + 1, 2lmax + 1), zero(CT))
    _, iters, resid, ok = NUFSHT.nusht_solve_spin!(ap, V, planp; weights = vec(w), rtol = solver.rtol,
                                                   maxiter = solver.maxiter)
    fit = HD.SolverResult{T}(ok, iters, T(resid))

    # Column `m` of `apr` holds `₊a_{ℓ,−m}`, so `₋a = sgn · conj(apr)`. The `|m| > ℓ` slots are zero in
    # `ap` and stay zero in every array formed from it.
    apr = view(ap, :, reverse(axes(ap, 2)))
    sgn = _order_row(m -> isodd(m) ? one(T) : -one(T), mem, T, lmax)
    RoverL = _degree_column(ℓ -> iszero(ℓ) ? zero(T) : R / sqrt(T(ℓ * (ℓ + 1))), mem, T, lmax)
    LoverR = _degree_column(ℓ -> sqrt(T(ℓ * (ℓ + 1))) / R, mem, T, lmax)
    sym = similar(ap)           # the rotational part
    anti = similar(ap)          # the divergent part
    sym .= (ap .+ sgn .* conj.(apr)) ./ 2
    anti .= (ap .- sgn .* conj.(apr)) ./ 2
    cΦ = RoverL .* ap           # χ + iψ
    cδ = .-LoverR .* anti       # ∇·u

    synth(c, plan) = NUFSHT.nusht_type2_spin!(similar(mem, CT, M), c, plan)
    rot, div, Φ, δ = synth(sym, planp), synth(anti, planp), synth(cΦ, plan0), synth(cδ, plan0)

    u_rot = _store_part!(similar(mem, T, sz..., 2), rot, M)
    u_div = _store_part!(similar(mem, T, sz..., 2), div, M)
    u_harm = fill!(similar(u_rot), zero(T))
    χ = reshape(real.(Φ), sz)
    ψ = reshape(imag.(Φ), sz)
    divergence = reshape(real.(δ), sz)
    Us = reshape(stage, sz..., 2)
    rfrac = HD._measure_ratio(Us .- u_rot .- u_div, Us, w)
    return HD.HelmholtzResult{2,1,T,typeof(u_rot),typeof(χ),Tuple{typeof(ψ)}}(
        u_rot, u_div, u_harm, χ, (ψ,), divergence, zero(T), rfrac, fit, (fit,))
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
    _decompose_spectral(solver, geometry, U, grid, execution, state) -> HelmholtzResult

Helmholtz decomposition of a tangent velocity field on any spherical node set, on `execution`: the spin
plans are built under it and released on return, and the parts come back in their memory. `U` is
component-last carrying `(u_east, u_north)`, and the parts come back in that layout. `S²` is simply
connected, so `u_harm` is zero; the fit's residual at the nodes is `residual_fraction`.
"""
function HD._decompose_spectral(
    solver::HD.SphericalNUSHTSolver,
    ::FG.Geometry.AbstractSphericalGeometry,
    U::AbstractArray{T},
    grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T},
    execution, ::Nothing,
) where {T}
    θ, Λ = _spherical_nodes(grid)
    lmax = _resolve_lmax(solver, grid)
    _warn_underdetermined(lmax, length(θ), 1)
    R = T(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    plan(s) = NUFSHT.make_spin_plan(Complex{T}, θ, Λ, lmax, s; tol = solver.tol, nufft = solver.nufft,
                                    execution = execution)
    planp = plan(+1)
    try
        plan0 = plan(0)
        try
            return _spin_hodge(solver, planp, plan0, U, grid, lmax, R)
        finally
            NUFSHT.close!(plan0)
        end
    finally
        NUFSHT.close!(planp)
    end
end

end # module
