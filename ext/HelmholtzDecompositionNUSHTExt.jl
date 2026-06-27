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

# Solve via the nuSHT Poisson solver, then reconstruct → physical HelmholtzResult.
function HD._decompose_spectral(
    solver::SphericalNUSHTSolver,
    ::HD.SphericalGeometry,
    U::AbstractArray,
    grid::HD.StructuredGrid{2,<:HD.SphericalGeometry};
    kwargs...,
)
    return HD.helmholtz_decompose(U, grid; solver = solver)
end

# Genuinely scattered spherical points: vector Hodge/Helmholtz decomposition via NUFSHT's
# spin-weighted (spin±1) scattered transforms. The complex tangent field V = u_θ + i u_φ is
# spin-1; its E/B (gradient/curl) parts split the symmetric/antisymmetric combinations of the
# spin(+1) and spin(−1) coefficients. Requires NUFSHT with spin support (make_spin_plan etc.).
function HD._decompose_spectral(
    solver::SphericalNUSHTSolver,
    ::HD.SphericalGeometry,
    U::AbstractMatrix,
    pts::HD.ScatteredPoints{2,<:HD.SphericalGeometry};
    rtol::Real = 1e-9,
    maxiter::Int = 600,
    kwargs...,
)
    isdefined(NUFSHT, :make_spin_plan) || throw(ArgumentError(
        "scattered-spherical decomposition needs NUFSHT spin support (make_spin_plan/nusht_solve_spin!); update NUFSHT."))
    λ = collect(pts.coords[:, 1])
    lat = collect(pts.coords[:, 2])
    θ = (π / 2) .- lat                              # colatitude
    uθ = -U[:, 2]                                   # θ̂ points south ⇒ u_θ = −u_north
    uφ = U[:, 1]                                    # φ̂ points east  ⇒ u_φ = u_east
    Vp = uθ .+ im .* uφ                             # spin +1
    Vm = uθ .- im .* uφ                             # spin −1

    lmax = solver.lmax
    shp = (lmax + 1, 2lmax + 1)
    planp = NUFSHT.make_spin_plan(θ, λ, lmax, +1; tol = solver.tol)
    planm = NUFSHT.make_spin_plan(θ, λ, lmax, -1; tol = solver.tol)
    ap = zeros(ComplexF64, shp); NUFSHT.nusht_solve_spin!(ap, Vp, planp; rtol = rtol, maxiter = maxiter)
    am = zeros(ComplexF64, shp); NUFSHT.nusht_solve_spin!(am, Vm, planm; rtol = rtol, maxiter = maxiter)

    sym = (ap .+ am) ./ 2          # rotational coefficients
    anti = (ap .- am) ./ 2         # divergent coefficients

    function _recon(a_plus, a_minus)
        Vp1 = similar(Vp); NUFSHT.nusht_type2_spin!(Vp1, a_plus, planp)
        Vm1 = similar(Vm); NUFSHT.nusht_type2_spin!(Vm1, a_minus, planm)
        uθ1 = (Vp1 .+ Vm1) ./ 2
        uφ1 = (Vp1 .- Vm1) ./ (2im)
        return real.(uφ1), -real.(uθ1)             # (u_east, u_north)
    end

    T = real(eltype(U))
    uer, unr = _recon(sym, sym)
    ued, und = _recon(anti, .-anti)
    u_rot = Matrix{T}(undef, HD.npoints(pts), 2); u_rot[:, 1] .= uer; u_rot[:, 2] .= unr
    u_div = Matrix{T}(undef, HD.npoints(pts), 2); u_div[:, 1] .= ued; u_div[:, 2] .= und
    u_harm = U .- u_rot .- u_div
    return (; u_rot, u_div, u_harm)
end

function __init__()
    HD.register_spectral_solver!(:spherical_irregular, SphericalNUSHTSolver)
end

end # module
