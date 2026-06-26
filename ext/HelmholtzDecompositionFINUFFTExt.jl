"""
    HelmholtzDecompositionFINUFFTExt — Cartesian spectral solver via FINUFFT (2D).

Provides `O(N log N)` spectral Poisson solves and Helmholtz decomposition for
irregular/non-uniform periodic 2D Cartesian grids using the 2D NUFFT: type 1 (analysis)
→ divide by `-(kx²+ky²)` → type 2 (synthesis).
"""
module HelmholtzDecompositionFINUFFTExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FINUFFT: FINUFFT

"""
    CartesianNUFFTSolver(; Nk_x=64, Nk_y=64, tol=1e-8)

Spectral Poisson solver for irregular/non-uniform periodic 2D Cartesian grids using FINUFFT.
"""
struct CartesianNUFFTSolver <: HD.AbstractPoissonSolver
    Nk_x::Int
    Nk_y::Int
    tol::Float64
end

CartesianNUFFTSolver(; Nk_x::Int = 64, Nk_y::Int = 64, tol::Float64 = 1e-8) =
    CartesianNUFFTSolver(Nk_x, Nk_y, tol)

function HD.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::HD.StructuredGrid{2,<:HD.CartesianGeometry{2,T}},
    solver::CartesianNUFFTSolver;
    kwargs...,
) where {T<:AbstractFloat}
    Nx, Ny = HD.size_tuple(grid)
    dx, dy = grid.geometry.spacing
    lon, lat = grid.coords_axes
    Lx = Nx * dx
    Ly = Ny * dy

    M = Nx * Ny
    x_nodes = Vector{T}(undef, M)
    y_nodes = Vector{T}(undef, M)
    rhs_vec = Vector{Complex{T}}(undef, M)
    k = 0
    for j in 1:Ny, i in 1:Nx
        k += 1
        x_nodes[k] = T(2π) * lon[i] / Lx - T(π)
        y_nodes[k] = T(2π) * lat[j] / Ly - T(π)
        rhs_vec[k] = Complex{T}(RHS[i, j])
    end

    Nk_x, Nk_y = solver.Nk_x, solver.Nk_y
    RHS_hat = dropdims(FINUFFT.nufft2d1(x_nodes, y_nodes, rhs_vec, +1, solver.tol, Nk_x, Nk_y); dims = 3)

    kx_range = -div(Nk_x, 2):div(Nk_x - 1, 2)
    ky_range = -div(Nk_y, 2):div(Nk_y - 1, 2)
    Φ_hat = similar(RHS_hat)
    for (jj, kyv) in enumerate(ky_range), (ii, kxv) in enumerate(kx_range)
        k2 = (T(2π) * kxv / Lx)^2 + (T(2π) * kyv / Ly)^2
        Φ_hat[ii, jj] = k2 ≈ zero(T) ? zero(Complex{T}) : RHS_hat[ii, jj] / (-k2)
    end

    phi_vec = FINUFFT.nufft2d2(x_nodes, y_nodes, -1, solver.tol, Φ_hat)
    norm_factor = one(T) / (Nk_x * Nk_y)
    k = 0
    for j in 1:Ny, i in 1:Nx
        k += 1
        Φ[i, j] = real(phi_vec[k]) * norm_factor
    end
    return HD.SolverResult{T}(true, 1, zero(T))
end

function HD._decompose_spectral(
    ::HD.CartesianGeometry,
    U::AbstractArray{T},
    grid::HD.StructuredGrid{2,<:HD.CartesianGeometry{2,T}};
    Nk_x::Int = HD.size_tuple(grid)[1],
    Nk_y::Int = HD.size_tuple(grid)[2],
    tol::Real = 1e-8,
    kwargs...,
) where {T}
    Nx, Ny = HD.size_tuple(grid)
    dx, dy = grid.geometry.spacing
    lon, lat = grid.coords_axes
    Lx = Nx * dx
    Ly = Ny * dy
    u = HD._component(U, 1, Val(2))
    v = HD._component(U, 2, Val(2))

    M = Nx * Ny
    x_nodes = Vector{T}(undef, M)
    y_nodes = Vector{T}(undef, M)
    u_complex = Vector{Complex{T}}(undef, M)
    v_complex = Vector{Complex{T}}(undef, M)
    k = 0
    for j in 1:Ny, i in 1:Nx
        k += 1
        x_nodes[k] = T(2π) * lon[i] / Lx - T(π)
        y_nodes[k] = T(2π) * lat[j] / Ly - T(π)
        u_complex[k] = Complex{T}(u[i, j])
        v_complex[k] = Complex{T}(v[i, j])
    end

    u_hat = dropdims(FINUFFT.nufft2d1(x_nodes, y_nodes, u_complex, +1, tol, Nk_x, Nk_y); dims = 3)
    v_hat = dropdims(FINUFFT.nufft2d1(x_nodes, y_nodes, v_complex, +1, tol, Nk_x, Nk_y); dims = 3)

    kx_range = -div(Nk_x, 2):div(Nk_x - 1, 2)
    ky_range = -div(Nk_y, 2):div(Nk_y - 1, 2)
    kx = T[T(2π) * kxv / Lx for kxv in kx_range]
    ky = T[T(2π) * kyv / Ly for kyv in ky_range]

    return HD.helmholtz_project_spectral(u_hat, v_hat, kx, ky)
end

function __init__()
    HD.register_spectral_solver!(:cartesian_irregular, CartesianNUFFTSolver)
end

end # module
