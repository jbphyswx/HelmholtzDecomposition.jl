"""
    HelmholtzDecompositionNUSHTExt — Spherical spectral solver via NUFSHT.

Provides `O(N log N)` spectral Poisson solves and Helmholtz decomposition for
irregular/scattered spherical grids using the non-uniform SHT (DFS + FINUFFT internally):
type 1 (analysis) → divide by `-ℓ(ℓ+1)/R²` → type 2 (synthesis). Works on any point
distribution.
"""
module HelmholtzDecompositionNUSHTExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using NUFSHT: NUFSHT

"""
    SphericalNUSHTSolver(; lmax=128, tol=1e-8)

Spectral Poisson solver for irregular/scattered spherical grids using NUFSHT.
"""
struct SphericalNUSHTSolver <: HD.AbstractPoissonSolver
    lmax::Int
    tol::Float64
end

SphericalNUSHTSolver(; lmax::Int = 128, tol::Float64 = 1e-8) = SphericalNUSHTSolver(lmax, tol)

function _flatten_nodes(grid::HD.StructuredGrid{2,<:HD.SphericalGeometry{T}}) where {T}
    Nlon, Nlat = HD.size_tuple(grid)
    lon, lat = grid.coords_axes
    M = Nlon * Nlat
    θ = Vector{T}(undef, M)
    φ = Vector{T}(undef, M)
    k = 0
    for j in 1:Nlat, i in 1:Nlon
        k += 1
        θ[k] = T(π / 2) - lat[j]   # latitude → colatitude
        φ[k] = lon[i]
    end
    return θ, φ
end

function _divide_eigenvalues!(C, lmax::Int, R::T) where {T}
    for ℓ in 1:lmax
        eig = -T(ℓ * (ℓ + 1)) / R^2
        for m in -ℓ:ℓ
            C[NUFSHT.FastSphericalHarmonics.sph_mode(ℓ, m)] /= eig
        end
    end
    C[NUFSHT.FastSphericalHarmonics.sph_mode(0, 0)] = zero(T)
    return C
end

function HD.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::HD.StructuredGrid{2,<:HD.SphericalGeometry{T}},
    solver::SphericalNUSHTSolver;
    kwargs...,
) where {T<:AbstractFloat}
    Nlon, Nlat = HD.size_tuple(grid)
    R = grid.geometry.R
    θ, φ = _flatten_nodes(grid)
    rhs_vec = Vector{T}(undef, Nlon * Nlat)
    k = 0
    for j in 1:Nlat, i in 1:Nlon
        k += 1
        rhs_vec[k] = RHS[i, j]
    end

    plan = NUFSHT.make_plan(θ, φ, solver.lmax; tol = solver.tol, T = T)
    C = similar(plan.C)
    NUFSHT.nusht_type1!(C, rhs_vec, plan)
    _divide_eigenvalues!(C, solver.lmax, R)
    phi_vec = similar(rhs_vec)
    NUFSHT.nusht_type2!(phi_vec, C, plan)
    k = 0
    for j in 1:Nlat, i in 1:Nlon
        k += 1
        Φ[i, j] = phi_vec[k]
    end
    return HD.SolverResult{T}(true, 1, zero(T))
end

function HD._decompose_spectral(
    solver::SphericalNUSHTSolver,
    ::HD.SphericalGeometry,
    U::AbstractArray{T},
    grid::HD.StructuredGrid{2,<:HD.SphericalGeometry{T}};
    lmax::Int = solver.lmax,
    tol::Real = solver.tol,
    kwargs...,
) where {T}
    Nlon, Nlat = HD.size_tuple(grid)
    R = grid.geometry.R

    div_f = zeros(T, Nlon, Nlat)
    vort = zeros(T, Nlon, Nlat, 1)
    HD._compute_div_rot!(div_f, vort, U, grid)
    ζ = HD._component(vort, 1, Val(2))

    θ, φ = _flatten_nodes(grid)
    M = Nlon * Nlat
    div_vec = Vector{T}(undef, M)
    vort_vec = Vector{T}(undef, M)
    k = 0
    for j in 1:Nlat, i in 1:Nlon
        k += 1
        div_vec[k] = div_f[i, j]
        vort_vec[k] = ζ[i, j]
    end

    plan = NUFSHT.make_plan(θ, φ, lmax; tol = tol, T = T)
    C_vort = similar(plan.C)
    C_div = similar(plan.C)
    NUFSHT.nusht_type1!(C_vort, vort_vec, plan)
    NUFSHT.nusht_type1!(C_div, div_vec, plan)
    _divide_eigenvalues!(C_vort, lmax, R)
    _divide_eigenvalues!(C_div, lmax, R)
    return HD.SpectralSphericalResult(C_vort, C_div, lmax)
end

function __init__()
    HD.register_spectral_solver!(:spherical_irregular, SphericalNUSHTSolver)
end

end # module
