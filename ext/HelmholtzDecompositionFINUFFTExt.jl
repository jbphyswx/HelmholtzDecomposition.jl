"""
    HelmholtzDecompositionFINUFFTExt — CartesianSpectralSolver via FINUFFT.

Provides O(N log N) spectral Poisson solve for irregular/non-uniform periodic
Cartesian grids using 2D NUFFT: type 1 (analysis) → divide by -(kx²+ky²) → type 2 (synthesis).
"""
module HelmholtzDecompositionFINUFFTExt

using HelmholtzDecomposition: HelmholtzDecomposition
using FINUFFT: FINUFFT

"""
    CartesianNUFFTSolver <: HelmholtzDecomposition.AbstractPoissonSolver

Spectral Poisson solver for irregular/non-uniform periodic Cartesian grids using FINUFFT.

Solves ∇²Φ = RHS by:
1. NUFFT type 1: scattered RHS → Fourier coefficients R̂(kx, ky)
2. Divide: Φ̂ = R̂ / (kx² + ky²)  (skip k=0)
3. NUFFT type 2: Fourier coefficients → Φ at scattered points

# Fields
- `Nk_x::Int` — Number of Fourier modes in x
- `Nk_y::Int` — Number of Fourier modes in y
- `tol::Float64` — FINUFFT accuracy tolerance
"""
struct CartesianNUFFTSolver <: HelmholtzDecomposition.AbstractPoissonSolver
    Nk_x::Int
    Nk_y::Int
    tol::Float64
end

CartesianNUFFTSolver(; Nk_x::Int=64, Nk_y::Int=64, tol::Float64=1e-8) =
    CartesianNUFFTSolver(Nk_x, Nk_y, tol)

function HelmholtzDecomposition.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::HelmholtzDecomposition.StructuredGrid{G,T},
    solver::CartesianNUFFTSolver;
    kwargs...
) where {T<:AbstractFloat, G<:HelmholtzDecomposition.CartesianGeometry{T}}
    Nx, Ny = HelmholtzDecomposition.size_tuple(grid)
    dx = grid.geometry.dx
    dy = grid.geometry.dy

    Lx = Nx * dx
    Ly = Ny * dy

    # Flatten grid to vectors, rescale coordinates to [-π, π) for FINUFFT
    M = Nx * Ny
    x_nodes = Vector{T}(undef, M)
    y_nodes = Vector{T}(undef, M)
    rhs_vec = Vector{Complex{T}}(undef, M)

    k = 0
    for j in 1:Ny
        for i in 1:Nx
            k += 1
            x_nodes[k] = T(2π) * grid.lon[i] / Lx - T(π)
            y_nodes[k] = T(2π) * grid.lat[j] / Ly - T(π)
            rhs_vec[k] = Complex{T}(RHS[i, j])
        end
    end

    Nk_x = solver.Nk_x
    Nk_y = solver.Nk_y

    # Type 1: scattered data → Fourier coefficients
    RHS_hat = FINUFFT.nufft2d1(x_nodes, y_nodes, rhs_vec, +1, solver.tol, Nk_x, Nk_y)
    RHS_hat_2d = dropdims(RHS_hat, dims=3)

    # Wavenumber arrays (FINUFFT convention: modes centered at 0)
    kx_range = -div(Nk_x, 2):div(Nk_x - 1, 2)
    ky_range = -div(Nk_y, 2):div(Nk_y - 1, 2)

    Φ_hat = similar(RHS_hat_2d)
    for (jj, kyv) in enumerate(ky_range)
        for (ii, kxv) in enumerate(kx_range)
            kx2 = (T(2π) * kxv / Lx)^2
            ky2 = (T(2π) * kyv / Ly)^2
            k2 = kx2 + ky2
            if k2 ≈ zero(T)
                Φ_hat[ii, jj] = zero(Complex{T})
            else
                Φ_hat[ii, jj] = RHS_hat_2d[ii, jj] / (-k2)
            end
        end
    end

    # Type 2: Fourier coefficients → scattered Φ values
    phi_vec = FINUFFT.nufft2d2(x_nodes, y_nodes, -1, solver.tol, Φ_hat)

    # Normalize and unflatten
    norm_factor = one(T) / (Nk_x * Nk_y)
    k = 0
    for j in 1:Ny
        for i in 1:Nx
            k += 1
            Φ[i, j] = real(phi_vec[k]) * norm_factor
        end
    end

    return HelmholtzDecomposition.SolverResult{T}(true, 1, zero(T))
end

function __init__()
    HelmholtzDecomposition.register_spectral_solver!(:cartesian_irregular, CartesianNUFFTSolver)
end

end # module
