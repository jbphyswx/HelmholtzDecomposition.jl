"""
    Spectral.jl — Spectral-space Helmholtz decomposition.

Provides types and methods for decomposing velocity fields directly in Fourier
or Spherical Harmonic coefficient space.
"""

export AbstractSpectralHelmholtzResult, SpectralCartesianResult, SpectralSphericalResult
export helmholtz_decompose_spectral, helmholtz_project_spectral, helmholtz_project_spectral!

"""
    AbstractSpectralHelmholtzResult{T}

Abstract supertype for spectral decomposition results.
"""
abstract type AbstractSpectralHelmholtzResult{T} end

"""
    SpectralCartesianResult{T, A}

Result of a Cartesian spectral Helmholtz decomposition, containing the complex
Fourier coefficients of the rotational and divergent velocity components.
"""
struct SpectralCartesianResult{T, A} <: AbstractSpectralHelmholtzResult{T}
    u_rot::A
    v_rot::A
    u_div::A
    v_div::A
end

function SpectralCartesianResult(u_rot::A, v_rot::A, u_div::A, v_div::A) where {A}
    T = real(eltype(A))
    return SpectralCartesianResult{T, A}(u_rot, v_rot, u_div, v_div)
end

"""
    SpectralSphericalResult{T, V}

Result of a Spherical spectral Helmholtz decomposition, containing the Spherical
Harmonic coefficients of the streamfunction ψ and velocity potential χ.
"""
struct SpectralSphericalResult{T, V} <: AbstractSpectralHelmholtzResult{T}
    ψ::V
    χ::V
    lmax::Int
end

function SpectralSphericalResult(ψ::V, χ::V, lmax::Int) where {V}
    T = real(eltype(V))
    return SpectralSphericalResult{T, V}(ψ, χ, lmax)
end

# ---------------------------------------------------------------------------
# Direct Fourier algebraic projections (Cartesian)
# ---------------------------------------------------------------------------

"""
    helmholtz_project_spectral(u_hat, v_hat, kx, ky; kwargs...)

Perform Cartesian Helmholtz decomposition directly in Fourier space given the
velocity coefficients `u_hat`, `v_hat` and the wavenumber vectors `kx`, `ky`.

This function uses hardware-agnostic fused broadcasting, making it fully GPU compatible.
"""
function helmholtz_project_spectral(
    u_hat::AbstractMatrix,
    v_hat::AbstractMatrix,
    kx::AbstractVector,
    ky::AbstractVector;
    kwargs...
)
    T = real(eltype(u_hat))
    kx_2d = reshape(T.(kx), :, 1)
    ky_2d = reshape(T.(ky), 1, :)

    k2 = @. kx_2d^2 + ky_2d^2
    inv_k2 = @. ifelse(k2 == zero(T), zero(T), one(T) / k2)

    u_rot = @. (-kx_2d * ky_2d * v_hat + ky_2d^2 * u_hat) * inv_k2
    v_rot = @. (kx_2d^2 * v_hat - kx_2d * ky_2d * u_hat) * inv_k2
    u_div = @. (kx_2d^2 * u_hat + kx_2d * ky_2d * v_hat) * inv_k2
    v_div = @. (kx_2d * ky_2d * u_hat + ky_2d^2 * v_hat) * inv_k2

    return SpectralCartesianResult(u_rot, v_rot, u_div, v_div)
end

"""
    helmholtz_project_spectral(u_hat, v_hat, grid::StructuredGrid{<:CartesianGeometry}; kwargs...)

Perform Cartesian spectral projection directly in Fourier space given the
velocity coefficients `u_hat`, `v_hat` and a Cartesian structured grid.
Automatically reconstructs the appropriate wavenumber vectors `kx`, `ky` based on the
dimensions and spacing of the grid.
"""
function helmholtz_project_spectral(
    u_hat::AbstractMatrix,
    v_hat::AbstractMatrix,
    grid::StructuredGrid{<:CartesianGeometry};
    kwargs...
)
    T = real(eltype(u_hat))
    Nx, Ny = size_tuple(grid)
    dx = grid.geometry.dx
    dy = grid.geometry.dy

    Nk_x = size(u_hat, 1)
    Nk_y = size(u_hat, 2)

    if Nk_x == Nx ÷ 2 + 1
        kx = T[T(2π) * (i - 1) / (Nx * dx) for i in 1:Nk_x]
    else
        Lx = Nx * dx
        kx = T[T(2π) * (i - 1 <= Nx ÷ 2 ? i - 1 : i - 1 - Nx) / Lx for i in 1:Nk_x]
    end

    Ly = Ny * dy
    ky = T[T(2π) * (j - 1 <= Ny ÷ 2 ? j - 1 : j - 1 - Ny) / Ly for j in 1:Nk_y]

    return helmholtz_project_spectral(u_hat, v_hat, kx, ky; kwargs...)
end

# ---------------------------------------------------------------------------
# Unified physical-input spectral entry points and geometry dispatch
# ---------------------------------------------------------------------------

"""
    helmholtz_decompose_spectral(u, v, grid; kwargs...)

Decompose a physical velocity field `(u, v)` on the given `grid` spectrally,
returning the spectral coefficients (either Fourier coefficients `SpectralCartesianResult`
or Spherical Harmonic coefficients `SpectralSphericalResult`).

Requires the appropriate spectral solver extension to be loaded (e.g. `using FFTW`,
`using FastSphericalHarmonics`, etc.).
"""
function helmholtz_decompose_spectral(u, v, grid::AbstractGrid; kwargs...)
    return _decompose_spectral(grid.geometry, u, v, grid; kwargs...)
end

# Hook implemented by geometry/solver extensions
function _decompose_spectral end

"""
    helmholtz_project_spectral!(û_rot, û_div, velocity_hat, ks::Tuple)

In-place Cartesian spectral projection. Decomposes the spectral velocity field
`velocity_hat` along the wavenumbers `ks`, writing the rotational component into `û_rot`
and the divergent component into `û_div`.
"""
function helmholtz_project_spectral!(û_rot, û_div, velocity_hat, ks::Tuple)
    T = real(eltype(velocity_hat))
    kx_2d = reshape(T.(ks[1]), :, 1)
    ky_2d = reshape(T.(ks[2]), 1, :)

    k2 = @. kx_2d^2 + ky_2d^2
    inv_k2 = @. ifelse(k2 == zero(T), zero(T), one(T) / k2)

    u_hat = @view velocity_hat[:, :, 1]
    v_hat = @view velocity_hat[:, :, 2]

    u_rot = @view û_rot[:, :, 1]
    v_rot = @view û_rot[:, :, 2]

    u_div = @view û_div[:, :, 1]
    v_div = @view û_div[:, :, 2]

    @. u_rot = (-kx_2d * ky_2d * v_hat + ky_2d^2 * u_hat) * inv_k2
    @. v_rot = (kx_2d^2 * v_hat - kx_2d * ky_2d * u_hat) * inv_k2
    @. u_div = (kx_2d^2 * u_hat + kx_2d * ky_2d * v_hat) * inv_k2
    @. v_div = (kx_2d * ky_2d * u_hat + ky_2d^2 * v_hat) * inv_k2

    return nothing
end
