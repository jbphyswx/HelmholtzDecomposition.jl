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

function HelmholtzDecomposition._decompose_spectral(
    ::HelmholtzDecomposition.SphericalGeometry,
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::HelmholtzDecomposition.StructuredGrid;
    kwargs...
)
    Nlon, Nlat = HelmholtzDecomposition.size_tuple(grid)
    T = eltype(u)
    R = grid.geometry.R
    dλ = Nlon > 1 ? grid.lon[2] - grid.lon[1] : T(0)
    dφ = Nlat > 1 ? grid.lat[2] - grid.lat[1] : T(0)

    div_f = zeros(T, Nlon, Nlat)
    vort_f = zeros(T, Nlon, Nlat)

    for j in 1:Nlat
        for i in 1:Nlon
            ip = i < Nlon ? i+1 : i
            im = i > 1 ? i-1 : i
            jp = j < Nlat ? j+1 : j
            jm = j > 1 ? j-1 : j

            φ = grid.lat[j]
            cosφ = cos(φ)

            h_λ = (ip - im) * dλ
            h_φ = (jp - jm) * dφ

            dudλ = ip == im ? zero(T) : (u[ip, j] - u[im, j]) / h_λ
            v_cos_jp = v[i, jp] * cos(grid.lat[jp])
            v_cos_jm = v[i, jm] * cos(grid.lat[jm])
            d_vcos_dφ = jp == jm ? zero(T) : (v_cos_jp - v_cos_jm) / h_φ
            div_f[i, j] = (dudλ + d_vcos_dφ) / (R * cosφ)

            dvdλ = ip == im ? zero(T) : (v[ip, j] - v[im, j]) / h_λ
            u_cos_jp = u[i, jp] * cos(grid.lat[jp])
            u_cos_jm = u[i, jm] * cos(grid.lat[jm])
            d_ucos_dφ = jp == jm ? zero(T) : (u_cos_jp - u_cos_jm) / h_φ
            vort_f[i, j] = (dvdλ - d_ucos_dφ) / (R * cosφ)
        end
    end

    lmax = Nlat - 1
    C_vort = Matrix{T}(undef, Nlat, Nlon)
    C_div = Matrix{T}(undef, Nlat, Nlon)
    for j in 1:Nlat
        for i in 1:Nlon
            C_vort[j, i] = vort_f[i, j]
            C_div[j, i] = div_f[i, j]
        end
    end

    FastSphericalHarmonics.sph_transform!(C_vort)
    FastSphericalHarmonics.sph_transform!(C_div)

    for ℓ in 1:lmax
        eigenval = -T(ℓ * (ℓ + 1)) / R^2
        for m in -ℓ:ℓ
            idx = FastSphericalHarmonics.sph_mode(ℓ, m)
            C_vort[idx] /= eigenval
            C_div[idx] /= eigenval
        end
    end
    idx0 = FastSphericalHarmonics.sph_mode(0, 0)
    C_vort[idx0] = zero(T)
    C_div[idx0] = zero(T)

    return HelmholtzDecomposition.SpectralSphericalResult(C_vort, C_div, lmax)
end

function __init__()
    HelmholtzDecomposition.register_spectral_solver!(:spherical_regular, SphericalSpectralSolver)
end

end # module
