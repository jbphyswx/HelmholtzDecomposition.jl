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

function HelmholtzDecomposition._decompose_spectral(
    ::HelmholtzDecomposition.SphericalGeometry,
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::HelmholtzDecomposition.StructuredGrid;
    lmax::Int = 128,
    tol::Real = 1e-8,
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

    M = Nlon * Nlat
    θ_nodes = Vector{T}(undef, M)
    φ_nodes = Vector{T}(undef, M)
    div_vec = Vector{T}(undef, M)
    vort_vec = Vector{T}(undef, M)

    k = 0
    for j in 1:Nlat
        lat = grid.lat[j]
        colat = T(π/2) - lat
        for i in 1:Nlon
            k += 1
            θ_nodes[k] = colat
            φ_nodes[k] = grid.lon[i]
            div_vec[k] = div_f[i, j]
            vort_vec[k] = vort_f[i, j]
        end
    end

    plan = NUFSHT.make_plan(θ_nodes, φ_nodes, lmax; tol=tol, T=T)

    C_vort = similar(plan.C)
    C_div = similar(plan.C)

    NUFSHT.nusht_type1!(C_vort, vort_vec, plan)
    NUFSHT.nusht_type1!(C_div, div_vec, plan)

    for ℓ in 1:lmax
        eigenval = -T(ℓ * (ℓ + 1)) / R^2
        for m in -ℓ:ℓ
            idx = NUFSHT.FastSphericalHarmonics.sph_mode(ℓ, m)
            C_vort[idx] /= eigenval
            C_div[idx] /= eigenval
        end
    end
    idx0 = NUFSHT.FastSphericalHarmonics.sph_mode(0, 0)
    C_vort[idx0] = zero(T)
    C_div[idx0] = zero(T)

    return HelmholtzDecomposition.SpectralSphericalResult(C_vort, C_div, lmax)
end

function __init__()
    HelmholtzDecomposition.register_spectral_solver!(:spherical_irregular, SphericalNUSHTSolver)
end

end # module
