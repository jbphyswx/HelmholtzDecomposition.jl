"""
    HelmholtzDecompositionNUFSHTExt — spherical spectral Poisson solve via NUFSHT.

Analysis → divide by `−ℓ(ℓ+1)/R²` → synthesis, on an arbitrary spherical node set.

Analysis uses `NUFSHT.nusht_solve!`, not `nusht_type1!`. NUFSHT's own docstring is explicit that
type 1 is the exact inverse of type 2 *only on the Clenshaw–Curtis grid*, and "at general scattered
points it is only the adjoint — use `nusht_solve!` for exact inversion there". An arbitrary node
set is the entire reason this extension exists, so using the adjoint would apply `A†` where `A⁻¹`
is required — a wrong answer on precisely the grids it is meant to serve, and only on those, since
the two coincide where the old tests happened to look.
"""
module HelmholtzDecompositionNUFSHTExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using NUFSHT: NUFSHT

"""
    SphericalNUSHTSolver(; lmax = nothing, tol = 1e-8, rtol = 1e-10, maxiter = 500)

Spectral Poisson solver for an arbitrary spherical node set. `rtol`/`maxiter` govern the least
squares inside the exact inverse transform.

`lmax = nothing` sizes the expansion from the grid — see [`_resolve_lmax`](@ref). A fixed degree is
wrong on every grid but one: the fit recovers `(lmax+1)²` coefficients from `M` nodes, so a degree
chosen without reference to `M` truncates a fine grid and outruns a coarse one.

A degree that outruns the nodes by enough corrupts the **split** while leaving the fit exact: the
samples are reproduced, the three parts sum back to the input, and the rotational/divergent share
is wrong, because that share reads the coefficient vector and the samples stop pinning it down. The
residual is at round-off throughout, so it reports nothing about this. Measured on a lat-lon grid
the split holds to round-off well past `(lmax+1)² = M` and degrades beyond roughly twice it; the
crossing depends on the field and the node layout, so `lmax` is left to the caller and sized from
the grid when it is unset.
"""
struct SphericalNUSHTSolver{T<:AbstractFloat,L<:Union{Nothing,Int}} <: HD.AbstractPoissonSolver
    lmax::L
    tol::T
    rtol::T
    maxiter::Int
end

SphericalNUSHTSolver(; lmax::Union{Nothing,Int} = nothing, tol::AbstractFloat = 1e-8,
                     rtol::AbstractFloat = 1e-10, maxiter::Int = 500) =
    SphericalNUSHTSolver(lmax, promote(tol, rtol)..., maxiter)

"""
    _resolve_lmax(solver, grid) -> Int

The expansion degree for `grid`, when the solver was built without one.

Two bounds, both from the grid. The node count caps it at `(lmax+1)² ≤ M`, which is conservative —
the fit and its split hold past that — and it needs no knowledge of the field. The sampling caps it
at the degree the node set represents, which `FlowGeometries` answers from the recipe where the grid
carries one and from the latitude count otherwise.
"""
function _resolve_lmax(solver::SphericalNUSHTSolver, grid)
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
HD.supports_boundary(::SphericalNUSHTSolver, ::HD.AbstractBoundaryCondition) = true
HD.requires_full_domain(::SphericalNUSHTSolver) = true

"""
    _require_covering_sphere(grid)

Throw unless the nodes cover `S²`.

Spherical-harmonic analysis integrates over the whole sphere and the inverse Laplacian there is
nonlocal, so a field known on part of the surface does not determine `Φ` even on that part. For a
fitting-based transform the same fact appears as ill-conditioning: bandlimited functions
concentrated off the sampled region are nearly invisible to the fit (the Slepian spatiospectral
concentration problem). A regional patch is a boundary value problem — a different problem, and
the one the iterative solver handles.
"""

"""
    _covers_sphere(grid) -> Bool

Whether `grid`'s nodes cover `S²`.

A rectilinear grid is judged geometrically: longitude wraps, and latitude reaches both poles to
within one cell. The polar cap the samples leave is then smaller than the resolution and the metric
closes it. One cell and no more — at two, a ±74.5° band passes, which is a regional patch. The
largest cell sets the bar, so a stretched latitude axis is judged by its coarsest part.

A cell measure cannot answer this on a rectilinear grid: summing `R²cos φ ΔφΔλ` is the midpoint
rule for `∫cos φ dφ`, so it carries the discretisation error.

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
    return (lo + T(π) / 2) <= dφ && (T(π) / 2 - hi) <= dφ
end

function _covers_sphere(
    grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry,T},
) where {T}
    R = T(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    return isapprox(T(sum(FG.Grids.measure(grid))), 4 * T(π) * R^2; rtol = sqrt(eps(T)))
end

HD.supports_sampling(::SphericalNUSHTSolver, grid) = _covers_sphere(grid)

function HD._sampling_message(::SphericalNUSHTSolver, grid)
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

The NUFSHT plan with its nodes uploaded, the flattened right-hand side and output, and the
coefficient array — `(lmax+1) × (2lmax+1)` in FastSphericalHarmonics' `sph_mode` layout, which is
what `_divide_eigenvalues!` indexes.

**One per task.** Every buffer here is written during a solve and the plan carries internal scratch,
so concurrent solves need one each.

This is also where the plan is built, which keeps `NUFSHT.make_plan` off the parallel section: it
reaches FastTransforms, whose plans come from FFTW's process-global planner. Plan-owning, hence the
explicit `show`.
"""
struct SphericalNUSHTState{T,P,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    plan::P
    rhs::V
    out::V
    coeffs::M
    lmax::Int
    radius::T
end

Base.show(io::IO, s::SphericalNUSHTState{T}) where {T} =
    print(io, "SphericalNUSHTState{", T, "}(", length(s.rhs), " nodes, lmax = ", s.lmax, ")")

function HD.prepare_solver(solver::SphericalNUSHTSolver,
                           grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
                           ::HD.AbstractBoundaryCondition;
                           backend = CB.SerialBackend(), shared = nothing) where {T}
    nlon, nlat = size(grid)
    lmax = _resolve_lmax(solver, grid)
    _warn_underdetermined(lmax, nlon * nlat)
    θ, Λ = _nodes(grid)
    R = T(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    # The field element type is positional, as it is for `zeros(T, …)`; a real `T` selects
    # NUFSHT's real specialization.
    plan = NUFSHT.make_plan(T, θ, Λ, lmax; tol = solver.tol)
    rhs = Vector{T}(undef, nlon * nlat)
    return SphericalNUSHTState(plan, rhs, similar(rhs),
                               zeros(T, lmax + 1, 2 * lmax + 1), lmax, R)
end

function HD.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
    solver::SphericalNUSHTSolver;
    boundary::HD.AbstractBoundaryCondition = HD.Neumann(),
    state::SphericalNUSHTState = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat}
    nlon, nlat = size(grid)
    rhs, out, C = state.rhs, state.out, state.coeffs
    k = 0
    @inbounds for j in 1:nlat, i in 1:nlon
        k += 1
        rhs[k] = RHS[i, j]
    end

    fill!(C, zero(T))
    # `nusht_solve!` inverts; `nusht_type1!` is the adjoint — see the module docstring.
    _, iters, resid, converged =
        NUFSHT.nusht_solve!(C, rhs, state.plan; rtol = solver.rtol, maxiter = solver.maxiter)
    # An unconverged fit gives a smooth field that does not match the samples, with nothing in
    # the output to say so.
    converged || throw(ArgumentError(
        "the spherical fit did not reach rtol=$(solver.rtol) in $(solver.maxiter) iterations " *
        "(residual $(resid) after $(iters)). $(nlon * nlat) samples, $((state.lmax + 1)^2) " *
        "coefficients at lmax=$(state.lmax); lower `lmax`, supply more nodes, or raise `maxiter`."))
    _divide_eigenvalues!(C, state.lmax, state.radius)
    NUFSHT.nusht_type2!(out, C, state.plan)
    k = 0
    @inbounds for j in 1:nlat, i in 1:nlon
        k += 1
        Φ[i, j] = out[k]
    end
    return HD.SolverResult{T}(true, iters, T(resid))
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
    _spin_hodge(solver, θ, Λ, uθ, uφ, lmax, T) -> (rot, div)

The spin±1 split at the nodes, each part returned as `(u_θ, u_φ)`.
"""
function _spin_hodge(solver::SphericalNUSHTSolver, θ, Λ, uθ, uφ, lmax::Int, ::Type{T}) where {T}
    CT = Complex{T}
    _warn_underdetermined(lmax, length(θ), 1)
    Vp = uθ .+ im .* uφ
    Vm = uθ .- im .* uφ
    planp = NUFSHT.make_spin_plan(CT, θ, Λ, lmax, +1; tol = solver.tol)
    planm = NUFSHT.make_spin_plan(CT, θ, Λ, lmax, -1; tol = solver.tol)
    try
        shp = (lmax + 1, 2lmax + 1)
        ap = zeros(CT, shp)
        am = zeros(CT, shp)
        _, ip, rp, okp = NUFSHT.nusht_solve_spin!(ap, Vp, planp; rtol = solver.rtol,
                                                  maxiter = solver.maxiter)
        _, iM, rm, okm = NUFSHT.nusht_solve_spin!(am, Vm, planm; rtol = solver.rtol,
                                                  maxiter = solver.maxiter)
        (okp && okm) || throw(ArgumentError(
            "the spin±1 fit did not reach rtol=$(solver.rtol) (spin +1: residual $rp after $ip; " *
            "spin −1: residual $rm after $iM). $(length(θ)) nodes, $((lmax + 1)^2) coefficients " *
            "at lmax=$lmax; lower `lmax`, supply more nodes, or raise `maxiter`."))

        sym = (ap .+ am) ./ 2       # E-type: the rotational part
        anti = (ap .- am) ./ 2      # B-type: the divergent part
        return (_spin_synthesize(sym, sym, planp, planm, Vp, Vm),
                _spin_synthesize(anti, .-anti, planp, planm, Vp, Vm))
    finally
        NUFSHT.close!(planp)
        NUFSHT.close!(planm)
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
    _decompose_spectral(solver, geometry, U, grid)

Helmholtz decomposition of a tangent velocity field on any spherical node set. `U` is
component-last carrying `(u_east, u_north)`, and the three parts come back in that layout.
"""
function HD._decompose_spectral(
    solver::SphericalNUSHTSolver,
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
    # `θ̂` points south and `φ̂` east.
    lmax = _resolve_lmax(solver, grid)
    (rotθ, rotφ), (divθ, divφ) = _spin_hodge(solver, θ, Λ, .-un, ue, lmax, T)

    u_rot = similar(U, T, (size(grid)..., 2))
    u_div = similar(U, T, (size(grid)..., 2))
    copyto!(HD._component(u_rot, 1, nd), reshape(rotφ, size(grid)))
    copyto!(HD._component(u_rot, 2, nd), reshape(.-rotθ, size(grid)))
    copyto!(HD._component(u_div, 1, nd), reshape(divφ, size(grid)))
    copyto!(HD._component(u_div, 2, nd), reshape(.-divθ, size(grid)))
    u_harm = U .- u_rot .- u_div
    return (; u_rot, u_div, u_harm)
end

function __init__()
    HD.register_spectral_solver!(SB.NUFSHTSpectralBackend, SphericalNUSHTSolver; priority = 10)
end

end # module
