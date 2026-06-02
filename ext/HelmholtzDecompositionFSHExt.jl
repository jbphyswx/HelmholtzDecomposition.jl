"""
    HelmholtzDecompositionFSHExt — SphericalSpectralSolver via FastSphericalHarmonics.

Provides O(N log N) spectral Poisson solve for regular lat/lon spherical grids
using SHT: forward SHT → divide by -l(l+1)/R² → inverse SHT.
"""
module HelmholtzDecompositionFSHExt

using HelmholtzDecomposition: HelmholtzDecomposition
using FastSphericalHarmonics: FastSphericalHarmonics

"""
    SphericalSpectralSolver <: HelmholtzDecomposition.AbstractPoissonSolver

Spectral Poisson solver for regular spherical grids using FastSphericalHarmonics.

Solves ∇²Φ = RHS on the sphere by:
1. Forward SHT of RHS → R̂ₗₘ
2. Divide: Φ̂ₗₘ = R̂ₗₘ / [-l(l+1)/R²]  (skip l=0)
3. Inverse SHT → Φ

Requires RHS on a Clenshaw-Curtis compatible grid (regular lat/lon).
"""
struct SphericalSpectralSolver <: HelmholtzDecomposition.AbstractPoissonSolver end

function HelmholtzDecomposition.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::HelmholtzDecomposition.StructuredGrid{G,T},
    ::SphericalSpectralSolver;
    kwargs...
) where {T<:AbstractFloat, G<:HelmholtzDecomposition.SphericalGeometry{T}}
    R = grid.geometry.R
    Nlat, Nlon = size(RHS, 2), size(RHS, 1)

    # FastSphericalHarmonics expects (Nθ × Nφ) = (lmax+1 × 2lmax+1)
    # Our grid is (Nlon × Nlat). Need to transpose for FSH convention.
    lmax = Nlat - 1

    # Copy RHS into FSH-compatible layout (Nlat × Nlon) = (lmax+1 × 2lmax+1)
    C = Matrix{T}(undef, Nlat, Nlon)
    for j in 1:Nlat
        for i in 1:Nlon
            C[j, i] = RHS[i, j]
        end
    end

    # Forward SHT: spatial → coefficients
    FastSphericalHarmonics.sph_transform!(C)

    # Divide by eigenvalue -l(l+1)/R² (skip l=0)
    for ℓ in 1:lmax
        eigenval = -T(ℓ * (ℓ + 1)) / R^2
        for m in -ℓ:ℓ
            idx = FastSphericalHarmonics.sph_mode(ℓ, m)
            C[idx] /= eigenval
        end
    end
    # l=0 mode: set to zero (mean of solution is arbitrary for Poisson)
    idx0 = FastSphericalHarmonics.sph_mode(0, 0)
    C[idx0] = zero(T)

    # Inverse SHT: coefficients → spatial
    FastSphericalHarmonics.sph_evaluate!(C)

    # Transpose back to (Nlon × Nlat)
    for j in 1:Nlat
        for i in 1:Nlon
            Φ[i, j] = C[j, i]
        end
    end

    return HelmholtzDecomposition.SolverResult{T}(true, 1, zero(T))
end

function __init__()
    HelmholtzDecomposition.register_spectral_solver!(:spherical_regular, SphericalSpectralSolver)
end

end # module
