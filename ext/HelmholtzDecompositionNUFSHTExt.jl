# Spherical Poisson solves and the spectral decomposition through NUFSHT, on any node set covering the
# sphere: analysis, divide by `−ℓ(ℓ+1)/R²`, synthesis. The analysis is the least-squares fit
# `NUFSHT.nusht_solve!`, the inverse of synthesis at any node set; `nusht_type1!` is its adjoint, the
# inverse only on the Clenshaw–Curtis grid.
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

Whether `grid`'s nodes cover `S²`.

Spherical-harmonic analysis integrates over the whole sphere and the inverse Laplacian there is
nonlocal, so a field known on part of the surface does not determine `Φ` even on that part. For a
fitting-based transform the same fact appears as ill-conditioning: bandlimited functions
concentrated off the sampled region are nearly invisible to the fit (the Slepian spatiospectral
concentration problem). A regional patch is a boundary value problem, which the iterative solver
handles.

A rectilinear grid is judged geometrically: longitude wraps, and the outermost latitude lies within
one and a half of the largest latitude spacing `Δ` of each pole. A cell-centred grid sits `Δ/2` from
the pole and the dual grid of one sits `Δ`, so neither decision rests on a rounding; a ±74.5° band of
16 rows sits `1.56Δ` from the poles and is a regional patch.

Any other layout is a pixelization whose cells tile the sphere exactly, and there the total measure
against `4πR²` is the statement — `Grids.measure` is lazy with `sum` specialised per layout, so it
answers without visiting a cell.
"""
_covers_sphere(grid) = false

function _covers_sphere(
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
) where {T}
    FG.Grids.isperiodic(grid, 1) || return false
    φ = FG.Grids.coordinates(grid, 2)
    lo, hi = extrema(φ)
    dφ = length(φ) > 1 ? T(FG.Grids.maximum_spacing(grid, 2)) : zero(T)
    reach = 3 * dφ / 2
    return (lo + T(π) / 2) <= reach && (T(π) / 2 - hi) <= reach
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

# Colatitude/longitude of every node, flattened in the grid's own order.
function _nodes(grid::FG.Grids.StructuredGrid{T,G,2}) where {G,T}
    nlon, nlat = size(grid)
    λ = FG.Grids.coordinates(grid, 1)
    φ = FG.Grids.coordinates(grid, 2)
    θ = similar(φ, T, nlon * nlat)
    Λ = similar(λ, T, nlon * nlat)
    k = 0
    @inbounds for j in 1:nlat, i in 1:nlon
        k += 1
        θ[k] = T(π) / 2 - T(φ[j])          # latitude → colatitude
        Λ[k] = T(λ[i])
    end
    return θ, Λ
end

function _divide_eigenvalues!(C, lmax::Int, R::T) where {T}
    for ℓ in 1:lmax
        eig = -T(ℓ * (ℓ + 1)) / R^2
        for m in (-ℓ):ℓ
            C[NUFSHT.FastSphericalHarmonics.sph_mode(ℓ, m)] /= eig
        end
    end
    # ℓ = 0 is the constant the Laplacian annihilates.
    C[NUFSHT.FastSphericalHarmonics.sph_mode(0, 0)] = zero(T)
    return C
end

"""
    SphericalNUSHTState

The NUFSHT plan with its nodes uploaded, the LSMR workspace its fit runs in, the flattened
right-hand side and output, and the coefficient array — `(lmax+1) × (2lmax+1)` in
FastSphericalHarmonics' `sph_mode` layout, which `_divide_eigenvalues!` indexes.

One per task: every buffer here is written during a solve. It owns the plan's NUFFT plans, which
[`close!`](@ref HelmholtzDecomposition.close!) releases.
"""
struct SphericalNUSHTState{T,P,W,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    plan::P
    ws::W
    rhs::V
    out::V
    coeffs::M
    lmax::Int
    radius::T
end

Base.show(io::IO, s::SphericalNUSHTState{T}) where {T} =
    print(io, "SphericalNUSHTState{", T, "}(", length(s.rhs), " nodes, lmax = ", s.lmax, ")")

HD.close!(s::SphericalNUSHTState) = (NUFSHT.close!(s.plan); s)
HD._close_state!(s::SphericalNUSHTState) = HD.close!(s)

function HD.prepare_solver(solver::HD.SphericalNUSHTSolver,
                           grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
                           ::FG.Operators.AbstractBoundaryCondition;
                           backend = CB.SerialBackend(), shared = nothing) where {T}
    nlon, nlat = size(grid)
    lmax = _resolve_lmax(solver, grid)
    _warn_underdetermined(lmax, nlon * nlat)
    θ, Λ = _nodes(grid)
    R = T(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    # The field element type is positional, as it is for `zeros(T, …)`; a real `T` selects
    # NUFSHT's real specialization.
    plan = NUFSHT.make_plan(T, θ, Λ, lmax; tol = solver.tol, nufft = solver.nufft)
    rhs = Vector{T}(undef, nlon * nlat)
    return SphericalNUSHTState(plan, FTB.LSMRWorkspace(plan), rhs, similar(rhs),
                               zeros(T, lmax + 1, 2 * lmax + 1), lmax, R)
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
    iters, resid, converged = _inverse!(Φ, RHS, grid, solver, state)
    return HD.SolverResult{T}(converged, iters, T(resid))
end

function HD.spectral_inverse!(
    Φ::AbstractMatrix{T}, RHS::AbstractMatrix{T},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
    solver::HD.SphericalNUSHTSolver, state::SphericalNUSHTState,
) where {T<:AbstractFloat}
    _inverse!(Φ, RHS, grid, solver, state)
    return Φ
end

# Fit, divide by the eigenvalues, synthesize. Returns the fit's iterations, residual and convergence.
function _inverse!(Φ, RHS, grid, solver::HD.SphericalNUSHTSolver, state::SphericalNUSHTState{T}) where {T}
    nlon, nlat = size(grid)
    rhs, out, C = state.rhs, state.out, state.coeffs
    k = 0
    @inbounds for j in 1:nlat, i in 1:nlon
        k += 1
        rhs[k] = RHS[i, j]
    end
    fill!(C, zero(T))
    # `nusht_solve!` inverts synthesis at any node set; `nusht_type1!` is its adjoint.
    _, iters, resid, converged = NUFSHT.nusht_solve!(C, rhs, state.plan; ws = state.ws,
                                                     rtol = solver.rtol, maxiter = solver.maxiter)
    _divide_eigenvalues!(C, state.lmax, state.radius)
    NUFSHT.nusht_type2!(out, C, state.plan)
    k = 0
    @inbounds for j in 1:nlat, i in 1:nlon
        k += 1
        Φ[i, j] = out[k]
    end
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
    _spherical_nodes(grid) -> (θ, λ)

Colatitude and longitude of every node, flattened in the grid's own cell order.

A rectilinear grid is a tensor product and is expanded from its two axes. Any other layout stores
no coordinate vectors and supplies its cloud through `Grids.materialize`.
"""
function _spherical_nodes(grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2}) where {T}
    nlon, nlat = size(grid)
    λ = FG.Grids.coordinates(grid, 1)
    φ = FG.Grids.coordinates(grid, 2)
    θ = Vector{T}(undef, nlon * nlat)
    Λ = Vector{T}(undef, nlon * nlat)
    k = 0
    @inbounds for j in 1:nlat, i in 1:nlon
        k += 1
        θ[k] = T(π) / 2 - T(φ[j])
        Λ[k] = T(λ[i])
    end
    return θ, Λ
end

function _spherical_nodes(grid::FG.Grids.UnstructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2}) where {T}
    λ = FG.Grids.coordinates(grid, 1)
    φ = FG.Grids.coordinates(grid, 2)
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
    _require_resolvable(lmax, M, s = 0)

Throw unless `M` nodes determine the `(lmax+1)² − s²` coefficients of a spin-`s` expansion.

The residual cannot stand in for this. With more coefficients than nodes the fit still reproduces
every sample and the three parts still sum back to the input; what the samples stop pinning down is
the coefficient vector, and the rotational/divergent split reads that vector directly. The
least-norm vector the solver returns places energy in modes the nodes do not resolve, which cancels
in the total and survives in each part, so the split is wrong at a residual of round-off.

This is a necessary condition. Clustered nodes leave the system rank-deficient at any `M`.
"""
@inline function _require_resolvable(lmax::Int, M::Int, s::Int = 0)
    ncoef = (lmax + 1)^2 - s^2
    ncoef <= M || throw(ArgumentError(
        "lmax=$lmax needs $ncoef coefficients and this point set has $M nodes. The fit reproduces " *
        "the samples at that band limit and its rotational/divergent split does not follow from " *
        "them. Lower `lmax`, leave it unset to size it from the grid, or supply more nodes."))
    return nothing
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

"""
    _spin_hodge(solver, θ, Λ, uθ, uφ, lmax, R, T) -> (rot, div, Φ, δ, fit)

The spin±1 split at the nodes, each velocity part returned as `(u_θ, u_φ)`, with `Φ = χ + iψ` and
`δ = ∇·u` at the nodes and the least converged of the two fits.

With `ₛY_ℓm = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` (NUFSHT's convention) and the Wigner-d derivative relations,
`∇Y_ℓm` has `u_θ + i u_φ = L ₁Y_ℓm` and `u_θ − i u_φ = −L ₋₁Y_ℓm`, `L = √(ℓ(ℓ+1))`; `k̂ × ∇` multiplies
both by `i`. So for `u = (∇χ + k̂×∇ψ)/R` the fits are `₊a = L(χ + iψ)/R` and `₋a = −L(χ − iψ)/R`: their
mean is the rotational part and their half difference the divergent one, `χ + iψ = R ₊a/L`, and
`∇·u = −L·(₊a − ₋a)/(2R)` per mode.
"""
function _spin_hodge(solver::HD.SphericalNUSHTSolver, θ, Λ, uθ, uφ, lmax::Int, R, ::Type{T}) where {T}
    CT = Complex{T}
    _warn_underdetermined(lmax, length(θ), 1)
    Vp = uθ .+ im .* uφ
    Vm = uθ .- im .* uφ
    planp = NUFSHT.make_spin_plan(CT, θ, Λ, lmax, +1; tol = solver.tol, nufft = solver.nufft)
    planm = NUFSHT.make_spin_plan(CT, θ, Λ, lmax, -1; tol = solver.tol, nufft = solver.nufft)
    plan0 = NUFSHT.make_spin_plan(CT, θ, Λ, lmax, 0; tol = solver.tol, nufft = solver.nufft)
    try
        shp = (lmax + 1, 2lmax + 1)
        ap = zeros(CT, shp)
        am = zeros(CT, shp)
        _, ip, rp, okp = NUFSHT.nusht_solve_spin!(ap, Vp, planp; rtol = solver.rtol,
                                                  maxiter = solver.maxiter)
        _, iM, rm, okm = NUFSHT.nusht_solve_spin!(am, Vm, planm; rtol = solver.rtol,
                                                  maxiter = solver.maxiter)
        fit = HD._worst(HD.SolverResult{T}(okp, ip, T(rp)), HD.SolverResult{T}(okm, iM, T(rm)))

        sym = (ap .+ am) ./ 2       # the rotational part
        anti = (ap .- am) ./ 2      # the divergent part
        cΦ = similar(ap)
        cδ = similar(ap)
        @inbounds for ℓ in 0:lmax, m in (-ℓ):ℓ
            I = NUFSHT.spin_coeff_index(ℓ, m, lmax)
            L = sqrt(T(ℓ * (ℓ + 1)))
            cΦ[I] = iszero(ℓ) ? zero(CT) : R * ap[I] / L
            cδ[I] = -L * anti[I] / R
        end
        Φ = similar(Vp)
        δ = similar(Vp)
        NUFSHT.nusht_type2_spin!(Φ, cΦ, plan0)
        NUFSHT.nusht_type2_spin!(δ, cδ, plan0)
        return (_spin_synthesize(sym, sym, planp, planm, Vp, Vm),
                _spin_synthesize(anti, .-anti, planp, planm, Vp, Vm), Φ, real.(δ), fit)
    finally
        NUFSHT.close!(planp)
        NUFSHT.close!(planm)
        NUFSHT.close!(plan0)
    end
end

function _spin_synthesize(a_plus, a_minus, planp, planm, Vp, Vm)
    V1 = similar(Vp)
    V2 = similar(Vm)
    NUFSHT.nusht_type2_spin!(V1, a_plus, planp)
    NUFSHT.nusht_type2_spin!(V2, a_minus, planm)
    return (real.((V1 .+ V2) ./ 2), real.((V1 .- V2) ./ (2im)))
end

"""
    _decompose_spectral(solver, geometry, U, grid) -> HelmholtzResult

Helmholtz decomposition of a tangent velocity field on any spherical node set. `U` is
component-last carrying `(u_east, u_north)`, and the parts come back in that layout. `S²` is simply
connected, so `u_harm` is zero; the fit's residual at the nodes is `residual_fraction`.
"""
function HD._decompose_spectral(
    solver::HD.SphericalNUSHTSolver,
    ::FG.Geometry.AbstractSphericalGeometry,
    U::AbstractArray{T},
    grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T};
    kwargs...,
) where {T}
    θ, Λ = _spherical_nodes(grid)
    m = length(θ)
    nd = Val(ndims(grid))
    ue = reshape(HD._component(U, 1, nd), m)
    un = reshape(HD._component(U, 2, nd), m)
    R = T(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    # `θ̂` points south and `φ̂` east.
    lmax = _resolve_lmax(solver, grid)
    (rotθ, rotφ), (divθ, divφ), Φ, δ, fit = _spin_hodge(solver, θ, Λ, .-un, ue, lmax, R, T)

    sz = size(grid)
    u_rot = similar(U, T, (sz..., 2))
    u_div = similar(U, T, (sz..., 2))
    copyto!(HD._component(u_rot, 1, nd), reshape(rotφ, sz))
    copyto!(HD._component(u_rot, 2, nd), reshape(.-rotθ, sz))
    copyto!(HD._component(u_div, 1, nd), reshape(divφ, sz))
    copyto!(HD._component(u_div, 2, nd), reshape(.-divθ, sz))
    u_harm = zero(u_rot)
    χ = reshape(real.(Φ), sz)
    ψ = reshape(imag.(Φ), sz)
    divergence = reshape(δ, sz)
    rfrac = HD._measure_ratio(U .- u_rot .- u_div, U, grid)
    return HD.HelmholtzResult{2,1,T,typeof(u_rot),typeof(χ),Tuple{typeof(ψ)}}(
        u_rot, u_div, u_harm, χ, (ψ,), divergence, zero(T), rfrac, fit, (fit,))
end

end # module
