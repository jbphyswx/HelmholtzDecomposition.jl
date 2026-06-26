"""
    HelmholtzDecompositionFINUFFTExt — Cartesian spectral solver via FINUFFT (2D).

Provides `O(N log N)` spectral Poisson solves and Helmholtz decomposition for
irregular/non-uniform periodic 2D Cartesian grids using the 2D NUFFT: type 1 (analysis)
→ divide by `-(kx²+ky²)` → type 2 (synthesis).
"""
module HelmholtzDecompositionFINUFFTExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra

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
    solver::CartesianNUFFTSolver,
    ::HD.CartesianGeometry,
    U::AbstractArray{T},
    grid::HD.StructuredGrid{2,<:HD.CartesianGeometry{2,T}};
    Nk_x::Int = solver.Nk_x,
    Nk_y::Int = solver.Nk_y,
    tol::Real = solver.tol,
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

# ---------------------------------------------------------------------------
# Scattered (unstructured) 2-D Cartesian decomposition
# ---------------------------------------------------------------------------

_drop3(A) = ndims(A) == 3 ? dropdims(A; dims = 3) : A

"""
    _inufft2(s1, s2, vals, Nk1, Nk2, tol; maxiter, rtol) -> F

Accurate inverse 2-D NUFFT: least-squares Fourier coefficients `F` (size `Nk1 × Nk2`) of the
band-limited field matching scattered samples `vals` at scaled nodes `(s1, s2) ∈ [0,2π)`.
Solves the normal equations `(AᴴA) F = Aᴴ vals` by conjugate gradients, where `A` is the
type-2 NUFFT (synthesis) and `Aᴴ` the type-1 NUFFT (its exact adjoint, opposite `iflag`).
This is the genuine non-uniform inverse — not the naive single adjoint, which is only correct
for quadrature-weighted/uniform nodes.
"""
function _inufft2(s1, s2, vals::AbstractVector{Complex{T}}, Nk1, Nk2, tol; maxiter, rtol) where {T}
    Aᴴ(c) = _drop3(FINUFFT.nufft2d1(s1, s2, c, -1, tol, Nk1, Nk2))   # type-1 (adjoint of A)
    A(F) = FINUFFT.nufft2d2(s1, s2, +1, tol, complex(F))             # type-2 (synthesis)
    AHA(F) = Aᴴ(A(F))

    b = Aᴴ(vals)
    F = zero(b)
    r = copy(b)
    p = copy(r)
    rs = real(LinearAlgebra.dot(r, r))
    bnorm = sqrt(real(LinearAlgebra.dot(b, b)))
    bnorm == 0 && return F
    for _ in 1:maxiter
        Ap = AHA(p)
        α = rs / real(LinearAlgebra.dot(p, Ap))
        @. F += α * p
        @. r -= α * Ap
        rs_new = real(LinearAlgebra.dot(r, r))
        sqrt(rs_new) <= rtol * bnorm && break
        @. p = r + (rs_new / rs) * p
        rs = rs_new
    end
    return F
end

"""
    _decompose_spectral(solver::CartesianNUFFTSolver, geometry, U, pts::ScatteredPoints)

Helmholtz decomposition of a velocity field sampled at scattered Cartesian points: inverse
NUFFT to recover the velocity Fourier coefficients, exact Leray projection in mode space,
then type-2 NUFFT back to the points. Returns a `(; u_rot, u_div, u_harm)` NamedTuple of
`(M, 2)` physical arrays.
"""
function HD._decompose_spectral(
    solver::CartesianNUFFTSolver,
    ::HD.CartesianGeometry,
    U::AbstractMatrix{T},
    pts::HD.ScatteredPoints{2,<:HD.CartesianGeometry{2,T}};
    maxiter::Int = 200,
    rtol::Real = 1e-10,
    kwargs...,
) where {T}
    Lx, Ly = pts.box
    Nk_x, Nk_y = solver.Nk_x, solver.Nk_y
    s1 = T(2π) .* view(pts.coords, :, 1) ./ Lx          # scale to FINUFFT's 2π period
    s2 = T(2π) .* view(pts.coords, :, 2) ./ Ly

    # Velocity Fourier coefficients via the accurate inverse NUFFT (per component).
    û = _inufft2(s1, s2, complex(U[:, 1]), Nk_x, Nk_y, solver.tol; maxiter, rtol)
    v̂ = _inufft2(s1, s2, complex(U[:, 2]), Nk_x, Nk_y, solver.tol; maxiter, rtol)

    # Exact Leray projection in mode space.
    velocity_hat = similar(û, Nk_x, Nk_y, 2)
    velocity_hat[:, :, 1] .= û
    velocity_hat[:, :, 2] .= v̂
    kx = T[T(2π) * m / Lx for m in (-div(Nk_x, 2)):div(Nk_x - 1, 2)]
    ky = T[T(2π) * m / Ly for m in (-div(Nk_y, 2)):div(Nk_y - 1, 2)]
    rot̂ = similar(velocity_hat)
    div̂ = similar(velocity_hat)
    HD.helmholtz_project_spectral!(rot̂, div̂, velocity_hat, (kx, ky))

    # Synthesize back to the scattered points (type-2).
    M = HD.npoints(pts)
    u_rot = Matrix{T}(undef, M, 2)
    u_div = Matrix{T}(undef, M, 2)
    for c in 1:2
        u_rot[:, c] .= real.(FINUFFT.nufft2d2(s1, s2, +1, solver.tol, rot̂[:, :, c]))
        u_div[:, c] .= real.(FINUFFT.nufft2d2(s1, s2, +1, solver.tol, div̂[:, :, c]))
    end
    u_harm = U .- u_rot .- u_div
    return (; u_rot, u_div, u_harm)
end

function __init__()
    HD.register_spectral_solver!(:cartesian_irregular, CartesianNUFFTSolver)
end

end # module
