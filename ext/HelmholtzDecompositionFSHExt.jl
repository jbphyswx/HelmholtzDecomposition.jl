"""
    HelmholtzDecompositionFSHExt — spherical spectral Poisson solve via FastSphericalHarmonics.

Forward SHT → divide by the eigenvalue `−ℓ(ℓ+1)/R²` → inverse SHT.

The transform is defined on one node set and no other: the Clenshaw–Curtis grid
`θᵢ = π(i−½)/N_θ`, `λⱼ = 2π(j−1)/(2N_θ−1)` (`FastSphericalHarmonics.sph_points`). The grid's
**node positions** are checked against it, not merely its shape — `N_λ = 2N_θ − 1` is necessary
and nowhere near sufficient, and feeding an evenly-spaced latitude grid to this transform returns
a smooth, plausible, wrong field.

That check subsumes the coverage question: a Clenshaw–Curtis set spans the whole sphere. It has to
be asked, because spherical-harmonic analysis integrates over `S²` and the inverse Laplacian there
is nonlocal, so a regional patch does not determine `Φ` even on the patch. A patch is a boundary
value problem — a different problem, and the one the iterative solver handles.
"""
module HelmholtzDecompositionFSHExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using FastSphericalHarmonics: FastSphericalHarmonics as FSH

struct SphericalSpectralSolver <: HD.AbstractPoissonSolver end

# A grid this solver accepts covers the closed sphere, so there is no boundary anywhere and a
# boundary condition is vacuous rather than honoured or refused. The transform reads every node,
# so no cell may be masked out.
HD.supports_boundary(::SphericalSpectralSolver, ::HD.AbstractBoundaryCondition) = true
HD.requires_full_domain(::SphericalSpectralSolver) = true

"""
    _require_clenshaw_curtis(grid)

Throw unless `grid`'s nodes are the Clenshaw–Curtis set this transform is defined on.

The node set is symmetric under `θ → π − θ`, so it may be supplied in either latitude order; the
reflection is an isometry of `S²` and commutes with `Δ`, so the transform is correct either way.
"""
function _require_clenshaw_curtis(grid::FG.Grids.StructuredGrid{G,T,2}) where {G,T}
    nlon, nlat = size(grid)
    want = FG.SphericalSampling.spherical_axes(T, FG.SphericalSampling.ClenshawCurtisSampling(), nlat)
    length(want.λ) == nlon || throw(ArgumentError(
        "FastSphericalHarmonics is defined on the Clenshaw–Curtis grid, which for $nlat latitudes " *
        "has $(length(want.λ)) longitudes; this grid has $nlon. Use the non-uniform transform " *
        "(`using NUFSHT`) for an arbitrary node set, or the iterative solver for a regional patch."))
    λ = FG.Grids.coordinates(grid, 1)
    φ = FG.Grids.coordinates(grid, 2)
    tol = sqrt(eps(T)) * T(2π)
    ok_λ = all(i -> abs(T(λ[i]) - want.λ[i]) <= tol, 1:nlon)
    # Either latitude order is acceptable; the node set is symmetric about the equator.
    ok_φ = all(j -> abs(T(φ[j]) - want.φ[j]) <= tol, 1:nlat) ||
           all(j -> abs(T(φ[j]) - want.φ[nlat - j + 1]) <= tol, 1:nlat)
    (ok_λ && ok_φ) || throw(ArgumentError(
        "this grid has the Clenshaw–Curtis shape ($(nlon)×$(nlat)) but not its node positions, " *
        "which are θᵢ = π(i−½)/$nlat and λⱼ = 2π(j−1)/$nlon. The transform is only defined on " *
        "those, and on any other set it returns a smooth but wrong field rather than failing. " *
        "Build the grid from `FlowGeometries.SphericalSampling.spherical_axes(T, " *
        "ClenshawCurtisSampling(), $nlat)`, or use `using NUFSHT` for an arbitrary node set."))
    return nothing
end

HD._require_sampling(::SphericalSpectralSolver, grid) = _require_clenshaw_curtis(grid)

"""
    SphericalSpectralState

The transposed coefficient array and the per-mode eigenvalues, built once.

`FastSphericalHarmonics` wants `(nlat, nlon)` where the grid is `(nlon, nlat)`, so a solve has to
transpose in and out. Allocating that array inside `solve_poisson!` allocates it once per potential
per field of a batch.
"""
struct SphericalSpectralState{T,A<:AbstractMatrix{T},E<:AbstractVector{T}}
    coeffs::A
    inv_eig::E      # 1/λ_ℓ per ℓ, with ℓ = 0 zeroed
end

function HD.prepare_solver(::SphericalSpectralSolver,
                           grid::FG.Grids.StructuredGrid{<:FG.Geometry.AbstractSphericalGeometry,T,2},
                           ::HD.AbstractBoundaryCondition) where {T}
    nlon, nlat = size(grid)
    lmax = nlat - 1
    R = FG.Geometry.radius(FG.Grids.grid_geometry(grid))
    coeffs = similar(FG.Grids.measure(grid), T, nlat, nlon)
    inv_eig = similar(FG.Grids.measure(grid), T, lmax + 1)
    inv_eig[1] = zero(T)          # ℓ = 0 is the constant the Laplacian annihilates
    @inbounds for ℓ in 1:lmax
        inv_eig[ℓ + 1] = -R^2 / T(ℓ * (ℓ + 1))
    end
    return SphericalSpectralState(coeffs, inv_eig)
end

function HD.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::FG.Grids.StructuredGrid{<:FG.Geometry.AbstractSphericalGeometry,T,2},
    solver::SphericalSpectralSolver;
    boundary::HD.AbstractBoundaryCondition = HD.Neumann(),
    state::SphericalSpectralState = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat}
    nlon, nlat = size(grid)
    lmax = nlat - 1

    C = state.coeffs
    @inbounds for j in 1:nlat, i in 1:nlon
        C[j, i] = RHS[i, j]
    end
    FSH.sph_transform!(C)
    @inbounds for ℓ in 1:lmax
        w = state.inv_eig[ℓ + 1]
        for m in (-ℓ):ℓ
            C[FSH.sph_mode(ℓ, m)] *= w
        end
    end
    # ℓ = 0 is the constant, which the Laplacian annihilates: the solve is defined up to it.
    C[FSH.sph_mode(0, 0)] = zero(T)
    FSH.sph_evaluate!(C)
    @inbounds for j in 1:nlat, i in 1:nlon
        Φ[i, j] = C[j, i]
    end
    return HD.SolverResult{T}(true, 1, zero(T))
end

function __init__()
    HD.register_spectral_solver!(SB.FSHTSpectralBackend, SphericalSpectralSolver; priority = 10)
end

end # module
