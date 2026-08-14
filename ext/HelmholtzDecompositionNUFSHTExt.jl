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
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using NUFSHT: NUFSHT

"""
    SphericalNUSHTSolver(; lmax = 128, tol = 1e-8, rtol = 1e-10, maxiter = 500)

Spectral Poisson solver for an arbitrary spherical node set. `rtol`/`maxiter` govern the conjugate
gradients inside the exact inverse transform.
"""
struct SphericalNUSHTSolver{T<:AbstractFloat} <: HD.AbstractPoissonSolver
    lmax::Int
    tol::T
    rtol::T
    maxiter::Int
end

SphericalNUSHTSolver(; lmax::Int = 128, tol::AbstractFloat = 1e-8, rtol::AbstractFloat = 1e-10,
                     maxiter::Int = 500) =
    SphericalNUSHTSolver(lmax, promote(tol, rtol)..., maxiter)

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
function _require_covering_sphere(grid::FG.Grids.StructuredGrid{G,T,2}) where {G,T}
    covered = FG.Grids.isperiodic(grid, 1)
    φ = FG.Grids.coordinates(grid, 2)
    lo, hi = extrema(φ)
    # Reaching the poles to within one cell is coverage — the polar cap the samples leave is then
    # smaller than the resolution, and the metric closes it. One cell, not two: at two, a ±74.5°
    # band passes, which is a regional patch and not a sphere. The largest cell is the measure, so
    # a stretched latitude axis is judged by its coarsest part.
    dφ = length(φ) > 1 ? T(FG.Grids.maximum_spacing(grid, 2)) : zero(T)
    covered &= (lo + T(π) / 2) <= dφ && (T(π) / 2 - hi) <= dφ
    covered && return nothing
    throw(ArgumentError(
        "SphericalNUSHTSolver expands in spherical harmonics over the whole sphere, but these " *
        "nodes do not cover it (longitude periodic: $(FG.Grids.isperiodic(grid, 1)); latitude " *
        "spans $(round(lo; digits = 3))…$(round(hi; digits = 3)) of ±π/2). Analysis integrates " *
        "over S² and the inverse Laplacian there is nonlocal, so a regional patch does not " *
        "determine the solution even on the patch. Solve a patch as a boundary value problem with " *
        "the iterative solver."))
end

HD._require_sampling(::SphericalNUSHTSolver, grid) = _require_covering_sphere(grid)

# Colatitude/longitude of every node, flattened in the grid's own order.
function _nodes(grid::FG.Grids.StructuredGrid{G,T,2}) where {G,T}
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

function HD.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::FG.Grids.StructuredGrid{<:FG.Geometry.AbstractSphericalGeometry,T,2},
    solver::SphericalNUSHTSolver;
    boundary::HD.AbstractBoundaryCondition = HD.Neumann(),
    kwargs...,
) where {T<:AbstractFloat}
    R = FG.Geometry.radius(FG.Grids.grid_geometry(grid))
    nlon, nlat = size(grid)
    θ, Λ = _nodes(grid)

    rhs = similar(RHS, T, nlon * nlat)
    k = 0
    @inbounds for j in 1:nlat, i in 1:nlon
        k += 1
        rhs[k] = RHS[i, j]
    end

    # The field element type is positional, as it is for `zeros(T, …)`; a real `T` selects
    # NUFSHT's real specialization.
    plan = NUFSHT.make_plan(T, θ, Λ, solver.lmax; tol = solver.tol)
    C = similar(plan.C)
    # Exact inversion, not the adjoint — see the module docstring.
    NUFSHT.nusht_solve!(C, rhs, plan; rtol = solver.rtol, maxiter = solver.maxiter)
    _divide_eigenvalues!(C, solver.lmax, R)
    out = similar(rhs)
    NUFSHT.nusht_type2!(out, C, plan)
    k = 0
    @inbounds for j in 1:nlat, i in 1:nlon
        k += 1
        Φ[i, j] = out[k]
    end
    return HD.SolverResult{T}(true, 1, zero(T))
end

function __init__()
    HD.register_spectral_solver!(SB.NUFSHTSpectralBackend, SphericalNUSHTSolver; priority = 10)
end

end # module
