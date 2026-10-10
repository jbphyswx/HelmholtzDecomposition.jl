# Spectral-space Helmholtz decomposition in any dimension. The Helmholtz (Leray) projection in
# Fourier space is, for every wavevector `k`,
#
#     û_div(k) = (k̂ ⊗ k̂) û(k)        (curl-free / divergent)
#     û_rot(k) = (I − k̂ ⊗ k̂) û(k)    (divergence-free / rotational)
#
# with the `k = 0` mode separated out as the harmonic part: a constant field is both curl-free and
# divergence-free. `k` is the symbol of the first derivative, which on a grid is zero at an even
# axis's Nyquist slot, so the projection is Hermitian there and the parts of a real field are exact.
# Fused broadcasts over the component-last spectral array, on host and device arrays alike.


"""
    AbstractSpectralHelmholtzResult{T}

Abstract supertype for spectral decomposition results.
"""
abstract type AbstractSpectralHelmholtzResult{T} end

"""
    SpectralCartesianResult{T,A}

Cartesian spectral decomposition result holding the component-last complex Fourier
coefficients of the rotational, divergent and harmonic velocity (`(kdims..., N)`).

`u_harm` carries the `k = 0` mode. On a periodic domain the constant fields are exactly the
harmonic subspace — `H¹(Tᴺ)` has dimension `N`, and a constant field is both curl-free and
divergence-free — so the mean flow belongs to neither of the other two parts.
"""
struct SpectralCartesianResult{T,A} <: AbstractSpectralHelmholtzResult{T}
    u_rot::A
    u_div::A
    u_harm::A
end

function SpectralCartesianResult(u_rot::A, u_div::A, u_harm::A) where {A}
    T = real(eltype(A))
    return SpectralCartesianResult{T,A}(u_rot, u_div, u_harm)
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

# The derivative symbol along each axis, shaped to broadcast along it and allocated like `proto`. On
# FlowTransformBindings axes it is the derivative wavenumber, zero at an even axis's Nyquist slot; a
# plain vector is the symbol as given.
_derivative_symbol(proto, ::Type{T}, ks::Tuple{Vararg{FlowTransformBindings.AbstractWavenumberAxis}}) where {T} =
    FlowTransformBindings.wavenumber_arrays(proto, T, ks; derivative = true)
_derivative_symbol(proto, ::Type{T}, ks::Tuple) where {T} =
    FlowTransformBindings.wavenumber_arrays(proto, T, ks)

# The mean, the harmonic part on a periodic box, from `ûa` into `harma`: the first slot of every
# FlowTransformBindings axis. A mode at an even axis's Nyquist slot has a zero derivative symbol but is
# not the mean, and stays with the rotational part. On plain vectors, where `k2` vanishes.
function _harmonic!(harma, ûa, ::Tuple{Vararg{FlowTransformBindings.AbstractWavenumberAxis}}, k2)
    fill!(harma, zero(eltype(harma)))
    first_slot = ntuple(_ -> 1:1, ndims(harma))
    view(harma, first_slot...) .= view(ûa, first_slot...)
    return harma
end
_harmonic!(harma, ûa, ::Tuple, k2) = (harma .= ifelse.(iszero.(k2), ûa, zero(eltype(harma))); harma)

"""
    helmholtz_project_spectral!(û_rot, û_div, û_harm, velocity_hat, ks::NTuple{N})

In-place Leray projection. `velocity_hat` and the three outputs are component-last spectral
arrays of size `(kdims..., N)`. `ks` holds the per-axis wavenumbers: FlowTransformBindings axes
(`FlowTransformBindings.wavenumber_axes`), on which the projection takes the derivative wavenumber, or
plain vectors, used as given. Writes the rotational (divergence-free) part into `û_rot`, the divergent
(curl-free) part into `û_div`, and the `k = 0` mode, the harmonic part, into `û_harm`, so that the
three sum to `velocity_hat`.
"""
function helmholtz_project_spectral!(û_rot, û_div, û_harm, velocity_hat, ks::NTuple{N,Any}) where {N}
    T = real(eltype(velocity_hat))
    K = _derivative_symbol(velocity_hat, T, ks)

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
        harma = comp(û_harm, a)
        @. diva = K[a] * kdotu * inv_k2
        _harmonic!(harma, ûa, ks, k2)
        @. rota = ûa - diva - harma
    end
    return nothing
end

"""
    helmholtz_potentials_spectral(velocity_hat, ks::NTuple{N}) -> (χ_hat, R_hat)

Compute the spectral scalar velocity potential `χ_hat` (size `(kdims...)`) and the
rotation-potential components `R_hat` (component-last, size `(kdims..., P)`, `P = N(N-1)/2`)
from a component-last spectral velocity array. Uses the spectral Poisson inverses
`χ̂ = −i (k·û)/k²` and `R̂_ab = −i (k_a û_b − k_b û_a)/k²`, with the `k = 0` mode set to zero; `ks`
as in [`helmholtz_project_spectral!`](@ref).
"""
function helmholtz_potentials_spectral(velocity_hat::AbstractArray, ks::NTuple{N,Any}) where {N}
    T = real(eltype(velocity_hat))
    CT = Complex{T}
    K = _derivative_symbol(velocity_hat, T, ks)
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
    û_harm = similar(velocity_hat)
    helmholtz_project_spectral!(û_rot, û_div, û_harm, velocity_hat, ks)
    return SpectralCartesianResult(û_rot, û_div, û_harm)
end

# 2-D convenience: separate (u_hat, v_hat) with explicit wavenumber vectors.
function helmholtz_project_spectral(u_hat::AbstractMatrix, v_hat::AbstractMatrix, kx::AbstractVector, ky::AbstractVector; kwargs...)
    velocity_hat = _stack_spectral(u_hat, v_hat)
    return helmholtz_project_spectral(velocity_hat, (kx, ky))
end

# Convenience: separate component arrays + grid (builds wavenumbers from the grid).
function helmholtz_project_spectral(velocity_hat::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractCartesianGeometry,N}; kwargs...) where {T,N}
    ks = _grid_wavenumbers(velocity_hat, grid)
    return helmholtz_project_spectral(velocity_hat, ks)
end

function helmholtz_project_spectral(u_hat::AbstractMatrix, v_hat::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractCartesianGeometry,2}; kwargs...) where {T}
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

The FlowTransformBindings wavenumber axes of a component-last spectral array on a Cartesian grid:
axis 1 halved (`rfft`) when its spectral length is `N₁÷2 + 1`, every axis full otherwise.
"""
function _grid_wavenumbers(velocity_hat::AbstractArray{<:Complex}, grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractCartesianGeometry,N}) where {N,T}
    dims = size(grid)
    Ls = ntuple(d -> T(FlowGeometries.Grids.domain_length(grid, d)), Val(N))
    return size(velocity_hat, 1) == dims[1] ÷ 2 + 1 ?
           FlowTransformBindings.wavenumber_axes(Ls, dims, Val(true)) :
           FlowTransformBindings.wavenumber_axes(Ls, dims, Val(false))
end

# ---------------------------------------------------------------------------
# Unified physical-input entry points and geometry dispatch
# ---------------------------------------------------------------------------

"""
    helmholtz_decompose_spectral(u, grid; kwargs...)
    helmholtz_decompose_spectral(u, v, grid; kwargs...)        # 2D convenience
    helmholtz_decompose_spectral(u, v, w, grid; kwargs...)     # 3D convenience

Decompose a physical velocity field on `grid` using a spectral transform: a physical
[`HelmholtzResult`](@ref) on a periodic Cartesian grid, the named tuple `(; u_rot, u_div, u_harm)` from
the scattered and spherical solvers. Requires the appropriate extension (`using FFTW`,
`using FastSphericalHarmonics`, …). Pass `solver=` to select among loaded spectral backends.

For raw spectral coefficients, use the lower-level [`helmholtz_project_spectral`](@ref).
"""
function helmholtz_decompose_spectral(u::AbstractArray, grid::FlowGeometries.Grids.AbstractGrid; kwargs...)
    return _spectral_dispatch(u, grid; kwargs...)
end

function helmholtz_decompose_spectral(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}, grid::FlowGeometries.Grids.AbstractGrid; kwargs...) where {N}
    return _spectral_dispatch(_stack_components(grid, u, v), grid; kwargs...)
end

function helmholtz_decompose_spectral(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}, w::AbstractArray{<:Any,N}, grid::FlowGeometries.Grids.AbstractGrid; kwargs...) where {N}
    return _spectral_dispatch(_stack_components(grid, u, v, w), grid; kwargs...)
end

"""
    _spectral_dispatch(u, grid; solver=AutoSolver(), kwargs...)

Resolve a spectral solver (extensions register them; `AutoSolver` picks the best available)
and dispatch to the extension's `_decompose_spectral(solver, geometry, u, grid; …)`.
Dispatching on the solver *type* lets several spectral backends (FFTW + FINUFFT, FSH +
NUFSHT) coexist for the same geometry without method clashes.
"""
function _spectral_dispatch(u::AbstractArray, grid::FlowGeometries.Grids.AbstractGrid; solver::AbstractPoissonSolver = AutoSolver(), kwargs...)
    s = _resolve_spectral_solver(grid, solver)
    return _decompose_spectral(s, FlowGeometries.Grids.grid_geometry(grid), u, grid; kwargs...)
end

function _resolve_spectral_solver(grid::FlowGeometries.Grids.AbstractGrid, solver::AbstractPoissonSolver)
    # The spectral solvers accept any condition, because a grid they accept has no boundary to
    # impose one on; the argument is nominal here and only the grid decides.
    s = solver isa AutoSolver ? _resolve_auto_solver(grid, Neumann()) : solver
    # Falling through to the iterative solver means no spectral extension applied — which on this
    # entry point is an error, not a fallback: the caller asked for the spectral path by name.
    s isa CGSolver && throw(ArgumentError(
        "helmholtz_decompose_spectral found no spectral solver for this grid. Load one that " *
        "applies (`using FFTW` for a periodic uniform Cartesian grid, `NonuniformFFTs` or `FINUFFT` " *
        "for scattered Cartesian samples, `FastSphericalHarmonics` for a Clenshaw–Curtis sphere, " *
        "`NUFSHT` for an arbitrary covering sphere), or call `helmholtz_decompose` to solve it " *
        "iteratively."))
    _require_library(s)
    _require_domain(s, grid)
    return s
end

# Hook implemented by spectral extensions: dispatch on the solver type.
function _decompose_spectral end

"""
    build_cartesian_result(grid, U, velocity_hat, ks, inverse) -> HelmholtzResult

Assemble a physical [`HelmholtzResult`](@ref) from a component-last spectral velocity array.
`inverse(spectral_scalar)` maps one spectral scalar field (`(kdims...)`) back to a physical one
(`(dims...)`). Used by the regular-grid spectral extensions (FFTW).

`N + P + 2` inverse transforms are taken, not `2N + 2P + 2`:

- `u_rot` is `U − u_div − u_harm` in physical space, so it costs nothing;
- `u_harm` is the `k = 0` mode, i.e. a constant field, and is the component mean — no transform;
- the rotation tensor `W = ΔR` is not materialised at all. It was `P` inverse transforms for a
  quantity recomputable from `R`, and no longer has a field on the result.

On this path `rotation_potential` holds **cell-centred** arrays, where the finite-difference path
holds corner-staggered ones. The field is type-parameterised for exactly this reason, but the
staggering does depend on which path produced the result.
"""
function build_cartesian_result(grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractCartesianGeometry,N}, U, velocity_hat, ks::NTuple{N,Any}, inverse;
                                backend = ComputationalBackends.SerialBackend()) where {N,T}
    proj = helmholtz_project_spectral(velocity_hat, ks)
    χ_hat, R_hat = helmholtz_potentials_spectral(velocity_hat, ks)

    K = _derivative_symbol(velocity_hat, T, ks)
    comp(A, c) = _component(A, c, Val(N))

    # δ_hat = i k·û, accumulated in place rather than by rebinding — the rebinding form
    # allocated a fresh full-size spectral array per component.
    # The component views are taken outside the `@.`: it dots every call in the expression, so
    # `comp(velocity_hat, b)` inside one becomes `comp.(velocity_hat, b)` — the accessor mapped
    # over each element rather than called once.
    û1 = comp(velocity_hat, 1)
    δ_hat = similar(û1)
    K1 = K[1]
    @. δ_hat = K1 * û1
    for b in 2:N
        ûb = comp(velocity_hat, b)
        Kb = K[b]
        @. δ_hat += Kb * ûb
    end
    @. δ_hat *= im

    P = n_rotation_components(N)
    dims = size(grid)

    # The output array is allocated from the known shape, so nothing here infers as
    # `Union{Nothing,Array}` and the closure is not captured into a box.
    # `similar(U, …)` rather than `Array{T,N+1}(undef, …)`: the result then lives wherever
    # the input does, which is what lets a device array take this path unchanged.
    u_div = similar(U, T, (dims..., N))
    for c in 1:N
        copyto!(_component(u_div, c, Val(N)), inverse(comp(proj.u_div, c)))
    end

    # One array per component, matching the finite-difference path's container, so that
    # `streamfunction` and `vector_potential` index the same way whichever path built the result.
    Rpot = ntuple(p -> inverse(comp(R_hat, p)), Val(P))

    χ = inverse(χ_hat)
    divergence = inverse(δ_hat)

    # The harmonic part is the k = 0 mode: a constant field, equal to each component's mean.
    u_harm = similar(U, T, (dims..., N))
    for c in 1:N
        fill!(_component(u_harm, c, Val(N)), sum(_component(U, c, Val(N))) / length(_component(U, c, Val(N))))
    end

    u_rot = similar(U, T, (dims..., N))
    @. u_rot = U - u_div - u_harm

    den = velocity_norm(U, grid; backend = backend)
    hfrac = iszero(den) ? zero(T) :
            velocity_norm(u_harm, grid; backend = backend) / den
    ok = SolverResult{T}(true, 1, zero(T))
    return HelmholtzResult{N,P,T,typeof(u_div),typeof(χ),typeof(Rpot)}(
        u_rot, u_div, u_harm, χ, Rpot, divergence, hfrac, ok, ntuple(_ -> ok, Val(P)),
    )
end

"""
    _stack_components(grid, comps...) -> Array

Stack scalar component arrays into the component-last layout `(dims..., M)` the transforms take.
"""
function _stack_components(grid::FlowGeometries.Grids.AbstractGrid, comps::Vararg{AbstractArray,M}) where {M}
    T = eltype(grid)
    N = ndims(grid)
    U = similar(first(comps), T, (size(grid)..., M))
    for c in 1:M
        copyto!(_component(U, c, Val(N)), comps[c])
    end
    return U
end
