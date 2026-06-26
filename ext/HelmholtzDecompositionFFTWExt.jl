"""
    HelmholtzDecompositionFFTWExt — Cartesian spectral solver via FFTW (any dimension).

Provides `O(N log N)` spectral Poisson solves and full Helmholtz decomposition for regular
periodic Cartesian grids in 1D/2D/3D/ND using the real FFT: forward `rfft` → divide by
`-Σ k_d²` → inverse `irfft`. Returns physical-space fields via `build_cartesian_result`.
"""
module HelmholtzDecompositionFFTWExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FFTW: FFTW

"""
    CartesianSpectralSolver <: AbstractPoissonSolver

Spectral Poisson solver for regular periodic Cartesian grids (any dimension) using FFTW.
"""
struct CartesianSpectralSolver <: HD.AbstractPoissonSolver end

# Per-axis angular wavenumbers matching an rfft layout (axis 1 reduced).
function _rfft_wavenumbers(::Type{T}, dims::NTuple{N,Int}, spacing::NTuple{N,T}) where {T,N}
    return ntuple(Val(N)) do d
        if d == 1
            T(2π) .* FFTW.rfftfreq(dims[1], one(T) / spacing[1])
        else
            T(2π) .* FFTW.fftfreq(dims[d], one(T) / spacing[d])
        end
    end
end

function HD.solve_poisson!(
    Φ::AbstractArray{T,N},
    RHS::AbstractArray{T,N},
    grid::HD.StructuredGrid{N,<:HD.CartesianGeometry{N,T}},
    ::CartesianSpectralSolver;
    kwargs...,
) where {T<:AbstractFloat,N}
    dims = HD.size_tuple(grid)
    spacing = grid.geometry.spacing
    RHS_hat = FFTW.rfft(RHS)
    ks = _rfft_wavenumbers(T, dims, spacing)
    K = ntuple(d -> reshape(ks[d], ntuple(i -> i == d ? length(ks[d]) : 1, Val(N))), Val(N))
    k2 = K[1] .^ 2
    for d in 2:N
        k2 = k2 .+ K[d] .^ 2
    end
    @. RHS_hat = ifelse(k2 == zero(T), zero(eltype(RHS_hat)), RHS_hat / (-k2))
    Φ .= FFTW.irfft(RHS_hat, dims[1])
    return HD.SolverResult{T}(true, 1, zero(T))
end

function HD._decompose_spectral(
    ::CartesianSpectralSolver,
    ::HD.CartesianGeometry,
    U::AbstractArray{T,M},
    grid::HD.StructuredGrid{N,<:HD.CartesianGeometry{N,T}};
    output::Symbol = :physical,
    kwargs...,
) where {T,M,N}
    dims = HD.size_tuple(grid)
    spacing = grid.geometry.spacing
    ks = _rfft_wavenumbers(T, dims, spacing)

    # Forward transform each velocity component → component-last spectral array.
    velocity_hat = nothing
    for c in 1:N
        ĉ = FFTW.rfft(HD._component(U, c, Val(N)))
        if velocity_hat === nothing
            velocity_hat = Array{eltype(ĉ),N + 1}(undef, size(ĉ)..., N)
        end
        copyto!(HD._component(velocity_hat, c, Val(N)), ĉ)
    end

    if output === :coefficients
        return HD.helmholtz_project_spectral(velocity_hat, ks)
    end
    inverse = x -> FFTW.irfft(x, dims[1])
    result = HD.build_cartesian_result(grid, U, velocity_hat, ks, inverse)
    output === :both && return (result, HD.helmholtz_project_spectral(velocity_hat, ks))
    return result
end

function __init__()
    HD.register_spectral_solver!(:cartesian_regular, CartesianSpectralSolver)
end

end # module
