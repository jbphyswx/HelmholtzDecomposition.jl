"""
    HelmholtzDecompositionNUSHTExt — SphericalSpectralSolver via NUFSHT.

Provides O(N log N) spectral Poisson solve for irregular/scattered spherical grids
using non-uniform SHT: forward nuSHT → divide by -l(l+1)/R² → inverse nuSHT.

NUFSHT.jl uses the DFS + FINUFFT algorithm internally to handle arbitrary point
distributions on the sphere.
"""
module HelmholtzDecompositionNUSHTExt

using HelmholtzDecomposition: HelmholtzDecomposition
using NUFSHT: NUFSHT

"""
    SphericalNUSHTSolver <: HelmholtzDecomposition.AbstractPoissonSolver

Spectral Poisson solver for irregular/scattered spherical grids using NUFSHT.

Solves ∇²Φ = RHS on the sphere by:
1. nuSHT type 1 (analysis): RHS at scattered points → R̂ₗₘ coefficients
2. Divide: Φ̂ₗₘ = R̂ₗₘ / [-l(l+1)/R²]  (skip l=0)
3. nuSHT type 2 (synthesis): Φ̂ₗₘ → Φ at scattered points

Works on ANY point distribution (regular, irregular, scattered).

# Fields
- `lmax::Int` — Maximum spherical harmonic degree for the solve
- `tol::Float64` — NUFSHT accuracy tolerance
"""
struct SphericalNUSHTSolver <: HelmholtzDecomposition.AbstractPoissonSolver
    lmax::Int
    tol::Float64
end

SphericalNUSHTSolver(; lmax::Int=128, tol::Float64=1e-8) = SphericalNUSHTSolver(lmax, tol)

function HelmholtzDecomposition.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::HelmholtzDecomposition.StructuredGrid{G,T},
    solver::SphericalNUSHTSolver;
    kwargs...
) where {T<:AbstractFloat, G<:HelmholtzDecomposition.SphericalGeometry{T}}
    R = grid.geometry.R
    Nlon, Nlat = HelmholtzDecomposition.size_tuple(grid)

    # Flatten grid points to vectors of (colatitude θ, longitude φ)
    # Our grid stores latitude φ ∈ [-π/2, π/2]; NUFSHT needs colatitude θ ∈ [0, π]
    M = Nlon * Nlat
    θ_nodes = Vector{T}(undef, M)
    φ_nodes = Vector{T}(undef, M)
    rhs_vec = Vector{T}(undef, M)

    k = 0
    for j in 1:Nlat
        lat = grid.lat[j]
        colat = T(π/2) - lat  # latitude → colatitude
        for i in 1:Nlon
            k += 1
            θ_nodes[k] = colat
            φ_nodes[k] = grid.lon[i]
            rhs_vec[k] = RHS[i, j]
        end
    end

    # Create NUFSHT plan
    plan = NUFSHT.make_plan(θ_nodes, φ_nodes, solver.lmax; tol=solver.tol, T=T)

    # Forward transform: scattered RHS → SH coefficients
    C = similar(plan.C)
    NUFSHT.nusht_type1!(C, rhs_vec, plan)

    # Divide by eigenvalue -l(l+1)/R²
    lmax = solver.lmax
    for ℓ in 1:lmax
        eigenval = -T(ℓ * (ℓ + 1)) / R^2
        for m in -ℓ:ℓ
            idx = NUFSHT.FastSphericalHarmonics.sph_mode(ℓ, m)
            C[idx] /= eigenval
        end
    end
    # l=0: set to zero
    idx0 = NUFSHT.FastSphericalHarmonics.sph_mode(0, 0)
    C[idx0] = zero(T)

    # Inverse transform: SH coefficients → scattered Φ values
    phi_vec = similar(rhs_vec)
    NUFSHT.nusht_type2!(phi_vec, C, plan)

    # Unflatten back to matrix
    k = 0
    for j in 1:Nlat
        for i in 1:Nlon
            k += 1
            Φ[i, j] = phi_vec[k]
        end
    end

    return HelmholtzDecomposition.SolverResult{T}(true, 1, zero(T))
end

function __init__()
    HelmholtzDecomposition.register_spectral_solver!(:spherical_irregular, SphericalNUSHTSolver)
end

end # module
