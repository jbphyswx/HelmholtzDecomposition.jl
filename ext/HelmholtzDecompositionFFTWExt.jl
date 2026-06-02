"""
    HelmholtzDecompositionFFTWExt — CartesianSpectralSolver via FFTW.

Provides O(N log N) spectral Poisson solve for regular periodic Cartesian grids
using 2D FFT: forward FFT → divide by -(kx²+ky²) → inverse FFT.
"""
module HelmholtzDecompositionFFTWExt

using HelmholtzDecomposition: HelmholtzDecomposition
using FFTW: FFTW

"""
    CartesianSpectralSolver <: HelmholtzDecomposition.AbstractPoissonSolver

Spectral Poisson solver for regular periodic Cartesian grids using FFTW.

Solves ∇²Φ = RHS by:
1. Forward 2D FFT of RHS → R̂(kx, ky)
2. Divide: Φ̂ = R̂ / (kx² + ky²)  (skip k=0 mode)
3. Inverse FFT → Φ

O(N log N) — vastly faster than SOR for large grids.
"""
struct CartesianSpectralSolver <: HelmholtzDecomposition.AbstractPoissonSolver end

function HelmholtzDecomposition.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::HelmholtzDecomposition.StructuredGrid{G,T},
    ::CartesianSpectralSolver;
    kwargs...
) where {T<:AbstractFloat, G<:HelmholtzDecomposition.CartesianGeometry{T}}
    Nx, Ny = HelmholtzDecomposition.size_tuple(grid)
    dx = grid.geometry.dx
    dy = grid.geometry.dy

    # Forward FFT of RHS
    RHS_hat = FFTW.rfft(RHS)

    # Wavenumber arrays
    kx = T(2π) .* FFTW.rfftfreq(Nx, one(T) / dx)
    ky = T(2π) .* FFTW.fftfreq(Ny, one(T) / dy)

    # Divide by -(kx² + ky²), skip k=0
    Φ_hat = similar(RHS_hat)
    for j in 1:length(ky)
        for i in 1:length(kx)
            k2 = kx[i]^2 + ky[j]^2
            if k2 ≈ zero(T)
                Φ_hat[i, j] = zero(eltype(RHS_hat))
            else
                Φ_hat[i, j] = RHS_hat[i, j] / (-k2)
            end
        end
    end

    # Inverse FFT
    Φ .= FFTW.irfft(Φ_hat, Nx)

    return HelmholtzDecomposition.SolverResult{T}(true, 1, zero(T))
end

function __init__()
    HelmholtzDecomposition.register_spectral_solver!(:cartesian_regular, CartesianSpectralSolver)
end

end # module
