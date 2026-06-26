"""
    HelmholtzDecompositionFSHExt — Spherical spectral solver via FastSphericalHarmonics.

Provides `O(N log N)` spectral Poisson solves for regular lat/lon spherical grids using the
SHT: forward SHT → divide by `-ℓ(ℓ+1)/R²` → inverse SHT. Requires a Clenshaw–Curtis
compatible grid with `Nlon = 2·Nlat − 1` (validated); use the NUFSHT extension for
arbitrary or scattered grids.

The spectral decomposition returns the spherical-harmonic coefficients of the
streamfunction `ψ` and velocity potential `χ`; vorticity/divergence are computed with the
package's spherical finite-difference operators (periodic longitude, pole-guarded).
"""
module HelmholtzDecompositionFSHExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FastSphericalHarmonics: FastSphericalHarmonics as FSH

struct SphericalSpectralSolver <: HD.AbstractPoissonSolver end

function _validate_fsh_grid(Nlon::Int, Nlat::Int)
    Nlon == 2 * Nlat - 1 || throw(ArgumentError(
        "FastSphericalHarmonics requires a Clenshaw–Curtis grid with Nlon = 2·Nlat − 1 " *
        "(got Nlon=$Nlon, Nlat=$Nlat). Use the NUFSHT extension for arbitrary grids."))
    return nothing
end

function HD.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::HD.StructuredGrid{2,<:HD.SphericalGeometry{T}},
    ::SphericalSpectralSolver;
    kwargs...,
) where {T<:AbstractFloat}
    R = grid.geometry.R
    Nlon, Nlat = HD.size_tuple(grid)
    _validate_fsh_grid(Nlon, Nlat)
    lmax = Nlat - 1

    C = Matrix{T}(undef, Nlat, Nlon)
    @inbounds for j in 1:Nlat, i in 1:Nlon
        C[j, i] = RHS[i, j]
    end
    FSH.sph_transform!(C)
    for ℓ in 1:lmax
        eig = -T(ℓ * (ℓ + 1)) / R^2
        for m in -ℓ:ℓ
            C[FSH.sph_mode(ℓ, m)] /= eig
        end
    end
    C[FSH.sph_mode(0, 0)] = zero(T)
    FSH.sph_evaluate!(C)
    @inbounds for j in 1:Nlat, i in 1:Nlon
        Φ[i, j] = C[j, i]
    end
    return HD.SolverResult{T}(true, 1, zero(T))
end

# Solve the potentials with the SHT Poisson solver, then run the standard reconstruction
# pipeline (FD div/vort → SHT solve → spherical reconstruct) → physical HelmholtzResult.
function HD._decompose_spectral(
    solver::SphericalSpectralSolver,
    ::HD.SphericalGeometry,
    U::AbstractArray,
    grid::HD.StructuredGrid{2,<:HD.SphericalGeometry};
    kwargs...,
)
    _validate_fsh_grid(HD.size_tuple(grid)...)
    return HD.helmholtz_decompose(U, grid; solver = solver)
end

function __init__()
    HD.register_spectral_solver!(:spherical_regular, SphericalSpectralSolver)
end

end # module
