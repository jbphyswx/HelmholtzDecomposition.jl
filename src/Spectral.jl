"""
    Spectral.jl — Spectral-space Helmholtz decomposition (dimension-generic).

The Helmholtz (Leray) projection in Fourier space is dimension-trivial: for every
wavevector `k`,

    û_div(k) = (k̂ ⊗ k̂) û(k)        (curl-free / divergent)
    û_rot(k) = (I − k̂ ⊗ k̂) û(k)    (divergence-free / rotational)

with the `k = 0` mode (the mean) left untouched. This is implemented with fused
broadcasts over the component-last spectral array, so the same code runs on CPU and GPU
arrays and in any number of dimensions.
"""

export AbstractSpectralHelmholtzResult, SpectralCartesianResult, SpectralSphericalResult
export helmholtz_decompose_spectral, helmholtz_project_spectral, helmholtz_project_spectral!

"""
    AbstractSpectralHelmholtzResult{T}

Abstract supertype for spectral decomposition results.
"""
abstract type AbstractSpectralHelmholtzResult{T} end

"""
    SpectralCartesianResult{T,A}

Cartesian spectral decomposition result holding the component-last complex Fourier
coefficients of the rotational and divergent velocity (`(kdims..., N)`).
"""
struct SpectralCartesianResult{T,A} <: AbstractSpectralHelmholtzResult{T}
    u_rot::A
    u_div::A
end

function SpectralCartesianResult(u_rot::A, u_div::A) where {A}
    T = real(eltype(A))
    return SpectralCartesianResult{T,A}(u_rot, u_div)
end

"""
    SpectralSphericalResult{T,V}

Spherical spectral decomposition result holding the spherical-harmonic coefficients of
the streamfunction `ψ` and velocity potential `χ`, up to degree `lmax`.
"""
struct SpectralSphericalResult{T,V} <: AbstractSpectralHelmholtzResult{T}
    ψ::V
    χ::V
    lmax::Int
end

function SpectralSphericalResult(ψ::V, χ::V, lmax::Int) where {V}
    T = real(eltype(V))
    return SpectralSphericalResult{T,V}(ψ, χ, lmax)
end

# ---------------------------------------------------------------------------
# Generic Fourier-space Leray projection
# ---------------------------------------------------------------------------

# Reshape an axis wavenumber vector to broadcast along spectral dimension `d` of an
# `N`-dimensional spectral grid.
@inline function _reshape_k(k::AbstractVector, d::Integer, ::Val{N}) where {N}
    return reshape(k, ntuple(i -> i == d ? length(k) : 1, Val(N)))
end

"""
    helmholtz_project_spectral!(û_rot, û_div, velocity_hat, ks::NTuple{N})

In-place Leray projection. `velocity_hat`, `û_rot`, `û_div` are component-last spectral
arrays of size `(kdims..., N)`; `ks` holds the per-axis wavenumber vectors. Writes the
rotational (divergence-free) part into `û_rot` and the divergent (curl-free) part into
`û_div`. GPU-compatible (pure broadcast).
"""
function helmholtz_project_spectral!(û_rot, û_div, velocity_hat, ks::NTuple{N,Any}) where {N}
    T = real(eltype(velocity_hat))
    K = ntuple(d -> _reshape_k(T.(ks[d]), d, Val(N)), Val(N))

    k2 = K[1] .^ 2
    for d in 2:N
        k2 = k2 .+ K[d] .^ 2
    end
    inv_k2 = @. ifelse(k2 == zero(T), zero(T), one(T) / k2)

    comp(A, c) = _component(A, c, Val(N))

    # k·û  (sum over components, broadcast over the spectral grid).
    kdotu = K[1] .* comp(velocity_hat, 1)
    for b in 2:N
        kdotu = kdotu .+ K[b] .* comp(velocity_hat, b)
    end

    for a in 1:N
        ûa = comp(velocity_hat, a)
        diva = comp(û_div, a)
        rota = comp(û_rot, a)
        @. diva = K[a] * kdotu * inv_k2
        @. rota = ûa - diva
    end
    return nothing
end

"""
    helmholtz_potentials_spectral(velocity_hat, ks::NTuple{N}) -> (χ_hat, R_hat)

Compute the spectral scalar velocity potential `χ_hat` (size `(kdims...)`) and the
rotation-potential components `R_hat` (component-last, size `(kdims..., P)`, `P = N(N-1)/2`)
from a component-last spectral velocity array. Uses the spectral Poisson inverses
`χ̂ = −i (k·û)/k²` and `R̂_ab = −i (k_a û_b − k_b û_a)/k²`, with the `k = 0` mode set to zero.
"""
function helmholtz_potentials_spectral(velocity_hat::AbstractArray, ks::NTuple{N,Any}) where {N}
    T = real(eltype(velocity_hat))
    CT = Complex{T}
    K = ntuple(d -> _reshape_k(T.(ks[d]), d, Val(N)), Val(N))
    k2 = K[1] .^ 2
    for d in 2:N
        k2 = k2 .+ K[d] .^ 2
    end
    inv_k2 = @. ifelse(k2 == zero(T), zero(T), one(T) / k2)
    comp(A, c) = _component(A, c, Val(N))

    kdotu = K[1] .* comp(velocity_hat, 1)
    for b in 2:N
        kdotu = kdotu .+ K[b] .* comp(velocity_hat, b)
    end
    χ_hat = @. -im * kdotu * inv_k2

    kdims = size(velocity_hat)[1:N]
    P = n_rotation_components(N)
    R_hat = similar(velocity_hat, CT, (kdims..., P))
    pairs = rotation_pairs(Val(N))
    for (p, (a, b)) in enumerate(pairs)
        ûa = comp(velocity_hat, a)
        ûb = comp(velocity_hat, b)
        Rp = comp(R_hat, p)
        @. Rp = -im * (K[a] * ûb - K[b] * ûa) * inv_k2
    end
    return χ_hat, R_hat
end

"""
    helmholtz_project_spectral(velocity_hat, ks::NTuple) -> SpectralCartesianResult

Allocating Leray projection from a component-last spectral array.
"""
function helmholtz_project_spectral(velocity_hat::AbstractArray, ks::NTuple{N,Any}) where {N}
    û_rot = similar(velocity_hat)
    û_div = similar(velocity_hat)
    helmholtz_project_spectral!(û_rot, û_div, velocity_hat, ks)
    return SpectralCartesianResult(û_rot, û_div)
end

# 2-D convenience: separate (u_hat, v_hat) with explicit wavenumber vectors.
function helmholtz_project_spectral(u_hat::AbstractMatrix, v_hat::AbstractMatrix, kx::AbstractVector, ky::AbstractVector; kwargs...)
    velocity_hat = _stack_spectral(u_hat, v_hat)
    return helmholtz_project_spectral(velocity_hat, (kx, ky))
end

# Convenience: separate component arrays + grid (builds wavenumbers from the grid).
function helmholtz_project_spectral(velocity_hat::AbstractArray, grid::StructuredGrid{N,<:CartesianGeometry}; kwargs...) where {N}
    ks = _grid_wavenumbers(velocity_hat, grid)
    return helmholtz_project_spectral(velocity_hat, ks)
end

function helmholtz_project_spectral(u_hat::AbstractMatrix, v_hat::AbstractMatrix, grid::StructuredGrid{2,<:CartesianGeometry}; kwargs...)
    velocity_hat = _stack_spectral(u_hat, v_hat)
    return helmholtz_project_spectral(velocity_hat, grid)
end

function _stack_spectral(comps::Vararg{AbstractArray{<:Complex},M}) where {M}
    sz = size(comps[1])
    out = Array{eltype(comps[1])}(undef, sz..., M)
    N = length(sz)
    for c in 1:M
        copyto!(_component(out, c, Val(N)), comps[c])
    end
    return out
end

"""
    _grid_wavenumbers(velocity_hat, grid) -> NTuple{N}

Reconstruct the per-axis angular wavenumber vectors for a component-last spectral array
on a Cartesian grid. Axis 1 is treated as an `rfft` axis when its spectral length equals
`N₁÷2 + 1`, otherwise as a full `fft` axis.
"""
function _grid_wavenumbers(velocity_hat::AbstractArray{<:Complex}, grid::StructuredGrid{N,<:CartesianGeometry{N,T}}) where {N,T}
    dims = size_tuple(grid)
    spacing = grid.geometry.spacing
    kdims = size(velocity_hat)[1:N]
    return ntuple(Val(N)) do d
        Nd = dims[d]
        L = Nd * spacing[d]
        if d == 1 && kdims[1] == Nd ÷ 2 + 1
            T[T(2π) * (i - 1) / L for i in 1:kdims[1]]
        else
            T[T(2π) * (i - 1 <= Nd ÷ 2 ? i - 1 : i - 1 - Nd) / L for i in 1:kdims[d]]
        end
    end
end

# ---------------------------------------------------------------------------
# Unified physical-input entry points and geometry dispatch
# ---------------------------------------------------------------------------

"""
    helmholtz_decompose_spectral(u, grid; kwargs...)
    helmholtz_decompose_spectral(u, v, grid; kwargs...)        # 2D convenience
    helmholtz_decompose_spectral(u, v, w, grid; kwargs...)     # 3D convenience

Decompose a physical velocity field on `grid` using a spectral transform, returning a
physical [`HelmholtzResult`](@ref) (CPU) — or, on the GPU path, a `(; u_rot, u_div, u_harm)`
NamedTuple of `CuArray`s. Requires the appropriate extension (`using FFTW`,
`using FastSphericalHarmonics`, …). Pass `solver=` to select among loaded spectral backends.

For raw spectral coefficients, use the lower-level [`helmholtz_project_spectral`](@ref).
"""
function helmholtz_decompose_spectral(u::AbstractArray, grid::AbstractGrid; kwargs...)
    return _spectral_dispatch(u, grid; kwargs...)
end

function helmholtz_decompose_spectral(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}, grid::AbstractGrid; kwargs...) where {N}
    return _spectral_dispatch(_stack_components(grid, u, v), grid; kwargs...)
end

function helmholtz_decompose_spectral(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}, w::AbstractArray{<:Any,N}, grid::AbstractGrid; kwargs...) where {N}
    return _spectral_dispatch(_stack_components(grid, u, v, w), grid; kwargs...)
end

"""
    _spectral_dispatch(u, grid; solver=AutoSolver(), kwargs...)

Resolve a spectral solver (extensions register them; `AutoSolver` picks the best available)
and dispatch to the extension's `_decompose_spectral(solver, geometry, u, grid; …)`.
Dispatching on the solver *type* lets several spectral backends (FFTW + FINUFFT, FSH +
NUFSHT) coexist for the same geometry without method clashes. The CUDA extension overrides
this for `CuArray` inputs to take the CUFFT path directly.
"""
function _spectral_dispatch(u::AbstractArray, grid::AbstractGrid; solver::AbstractPoissonSolver = AutoSolver(), kwargs...)
    s = _resolve_spectral_solver(grid, solver)
    return _decompose_spectral(s, grid.geometry, u, grid; kwargs...)
end

function _resolve_spectral_solver(grid::AbstractGrid, solver::AbstractPoissonSolver)
    solver isa AutoSolver || return solver
    s = _resolve_auto_solver(grid)
    s isa SORSolver && throw(ArgumentError(
        "helmholtz_decompose_spectral requires a spectral extension for this geometry " *
        "(`using FFTW`/`FINUFFT` for Cartesian, `FastSphericalHarmonics`/`NUFSHT` for spherical)."))
    return s
end

# Hook implemented by spectral extensions: dispatch on the solver type.
function _decompose_spectral end

"""
    build_cartesian_result(grid, U, velocity_hat, ks, inverse) -> HelmholtzResult

Assemble a complete physical [`HelmholtzResult`](@ref) from a component-last spectral
velocity array. `inverse(spectral_scalar)` must map a single spectral scalar field
(`(kdims...)`) back to a physical scalar field (`(dims...)`). All decomposition fields
(rotational/divergent/harmonic velocity, potentials, vorticity, divergence) are computed
spectrally and inverse-transformed. Used by the regular-grid spectral extensions (FFTW).
"""
function build_cartesian_result(grid::StructuredGrid{N,<:CartesianGeometry,T}, U, velocity_hat, ks::NTuple{N,Any}, inverse) where {N,T}
    proj = helmholtz_project_spectral(velocity_hat, ks)
    χ_hat, R_hat = helmholtz_potentials_spectral(velocity_hat, ks)

    # δ_hat = i k·û ; W_hat_ab = i(k_a û_b − k_b û_a)
    K = ntuple(d -> _reshape_k(T.(ks[d]), d, Val(N)), Val(N))
    comp(A, c) = _component(A, c, Val(N))
    δ_hat = K[1] .* comp(velocity_hat, 1)
    for b in 2:N
        δ_hat = δ_hat .+ K[b] .* comp(velocity_hat, b)
    end
    δ_hat = δ_hat .* im

    _inv_components(spec, M) = begin
        out = nothing
        for c in 1:M
            phys = inverse(comp(spec, c))
            if out === nothing
                out = similar(phys, (size(phys)..., M))
            end
            copyto!(_component(out, c, Val(N)), phys)
        end
        out
    end

    P = n_rotation_components(N)
    u_rot = _inv_components(proj.u_rot, N)
    u_div = _inv_components(proj.u_div, N)
    χ = inverse(χ_hat)
    Rpot = P == 0 ? Array{T,N + 1}(undef, size_tuple(grid)..., 0) : _inv_components(R_hat, P)
    divergence = inverse(δ_hat)

    # Rotation tensor components in spectral space then inverse.
    vorticity = if P == 0
        Array{T,N + 1}(undef, size_tuple(grid)..., 0)
    else
        W_hat = similar(R_hat)
        for (p, (a, b)) in enumerate(rotation_pairs(Val(N)))
            Wp = comp(W_hat, p)
            ûa = comp(velocity_hat, a)
            ûb = comp(velocity_hat, b)
            @. Wp = im * (K[a] * ûb - K[b] * ûa)
        end
        _inv_components(W_hat, P)
    end

    u_harm = similar(u_rot)
    @. u_harm = U - u_rot - u_div
    hfrac = _velocity_norm(u_harm, grid) / max(_velocity_norm(U, grid), eps(T))
    ok = SolverResult{T}(true, 1, zero(T))
    return HelmholtzResult{N,T,typeof(u_rot),typeof(χ)}(
        u_rot, u_div, u_harm, χ, Rpot, vorticity, divergence, hfrac, ok, [ok for _ in 1:P],
    )
end
