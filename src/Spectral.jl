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
    k2 = lazy_axis_sum(map(k -> k .^ 2, K))
    comp(A, c) = _component(A, c, Val(N))

    # `k·û` in the last harmonic component, which is written after every divergent one is.
    kdotu = comp(û_harm, N)
    kdotu .= _k_dot(K, velocity_hat, Val(N))
    for a in 1:N
        comp(û_div, a) .= K[a] .* kdotu .* _inverse_k2.(k2)
    end
    for a in 1:N
        _harmonic!(comp(û_harm, a), comp(velocity_hat, a), ks, k2)
        comp(û_rot, a) .= comp(velocity_hat, a) .- comp(û_div, a) .- comp(û_harm, a)
    end
    return nothing
end

# `1/k²`, zero where `k² = 0`.
@inline _inverse_k2(x::T) where {T} = iszero(x) ? zero(T) : one(T) / x

# `k·û` over the components of a component-last spectral array, unmaterialized.
_k_dot(K::NTuple{N,Any}, vh, ::Val{N}) where {N} =
    lazy_axis_sum(ntuple(b -> Broadcast.broadcasted(*, K[b], _component(vh, b, Val(N))), Val(N)))

# `χ̂ = −i (k·û)/k²`, the `P` rotation-potential spectra `−i (k_a û_b − k_b û_a)/k²` and the divergence
# `i k·û` of the component-last spectral velocity `vh`, into the `P + 2` slices of `S` along its last axis.
function _scalar_spectra!(S, vh, K::NTuple{N,Any}, ::Val{N}) where {N}
    k2 = lazy_axis_sum(map(k -> k .^ 2, K))
    kdotu = _k_dot(K, vh, Val(N))
    comp(A, c) = _component(A, c, Val(N))
    selectdim(S, N + 1, 1) .= -im .* kdotu .* _inverse_k2.(k2)
    pairs = FlowGeometries.Operators.rotation_pairs(Val(N))
    for (p, (a, b)) in enumerate(pairs)
        selectdim(S, N + 1, 1 + p) .= -im .* (K[a] .* comp(vh, b) .- K[b] .* comp(vh, a)) .* _inverse_k2.(k2)
    end
    selectdim(S, N + 1, length(pairs) + 2) .= im .* kdotu
    return S
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
    k2 = lazy_axis_sum(map(k -> k .^ 2, K))
    comp(A, c) = _component(A, c, Val(N))

    kdims = ntuple(d -> size(velocity_hat, d), Val(N))
    χ_hat = similar(velocity_hat, CT, kdims)
    χ_hat .= -im .* _k_dot(K, velocity_hat, Val(N)) .* _inverse_k2.(k2)

    P = FlowGeometries.Operators.n_rotation_components(N)
    R_hat = similar(velocity_hat, CT, (kdims..., P))
    pairs = FlowGeometries.Operators.rotation_pairs(Val(N))
    for (p, (a, b)) in enumerate(pairs)
        comp(R_hat, p) .= -im .* (K[a] .* comp(velocity_hat, b) .- K[b] .* comp(velocity_hat, a)) .*
                          _inverse_k2.(k2)
    end
    return χ_hat, R_hat
end

"""
    helmholtz_project_spectral(velocity_hat, ks::NTuple) -> SpectralCartesianResult
    helmholtz_project_spectral(E, velocity_hat, grid) -> SpectralCartesianResult

Allocating Leray projection from a component-last spectral array. The grid form takes the wavenumbers
of `grid` for a field of element type `E`: a real field's coefficients are the `rfft` half along axis 1.
"""
function helmholtz_project_spectral(velocity_hat::AbstractArray, ks::NTuple{N,Any}) where {N}
    û_rot = similar(velocity_hat)
    û_div = similar(velocity_hat)
    û_harm = similar(velocity_hat)
    helmholtz_project_spectral!(û_rot, û_div, û_harm, velocity_hat, ks)
    return SpectralCartesianResult(û_rot, û_div, û_harm)
end

# 2-D convenience: separate (u_hat, v_hat) with explicit wavenumber vectors.
helmholtz_project_spectral(u_hat::AbstractMatrix, v_hat::AbstractMatrix, kx::AbstractVector, ky::AbstractVector) =
    helmholtz_project_spectral(_stack_spectral(u_hat, v_hat), (kx, ky))

function helmholtz_project_spectral(::Type{E}, velocity_hat::AbstractArray,
        grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractCartesianGeometry,N}) where {E<:Number,T,N}
    ks = _grid_wavenumbers(E, grid)
    size(velocity_hat) == (map(length, ks)..., N) || throw(DimensionMismatch(
        "velocity_hat is $(size(velocity_hat)); a $E field on this grid has coefficients $((map(length, ks)..., N))"))
    return helmholtz_project_spectral(velocity_hat, ks)
end

helmholtz_project_spectral(::Type{E}, u_hat::AbstractMatrix, v_hat::AbstractMatrix,
        grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractCartesianGeometry,2}) where {E<:Number,T} =
    helmholtz_project_spectral(E, _stack_spectral(u_hat, v_hat), grid)

function _stack_spectral(comps::Vararg{AbstractArray{<:Complex},M}) where {M}
    sz = size(comps[1])
    out = similar(comps[1], (sz..., M))
    N = length(sz)
    for c in 1:M
        copyto!(_component(out, c, Val(N)), comps[c])
    end
    return out
end

"""
    _grid_wavenumbers(E, grid) -> NTuple{N}

The FlowTransformBindings wavenumber axes of a field of element type `E` on a Cartesian grid: axis 1
halved (`rfft`) for a real field, every axis full for a complex one.
"""
_grid_wavenumbers(::Type{E}, grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractCartesianGeometry,N}) where {E,T,N} =
    FlowTransformBindings.wavenumber_axes(E, ntuple(d -> T(FlowGeometries.Grids.domain_length(grid, d)), Val(N)),
                                          size(grid))

# ---------------------------------------------------------------------------
# Unified physical-input entry points and geometry dispatch
# ---------------------------------------------------------------------------

"""
    helmholtz_decompose_spectral(u, grid; solver = AutoSolver(), execution = AutoBackend(), state = nothing)
    helmholtz_decompose_spectral(u, v, grid; kwargs...)        # 2D convenience
    helmholtz_decompose_spectral(u, v, w, grid; kwargs...)     # 3D convenience

Decompose a velocity field on `grid` with a spectral transform into a [`HelmholtzResult`](@ref): the
FFT on a periodic uniform Cartesian grid, the NUFFT fit on a Cartesian point cloud,
FastSphericalHarmonics on its Fejér first-rule sphere, NUFSHT on any node set covering the sphere.

`solver` names the transform, its library loaded by the caller; `AutoSolver()` takes the first of
[`_spectral_candidates`](@ref) that applies under `execution`. A host `execution` sets the libraries'
thread count. A `GPUBackend` moves `u` to its device and returns the parts there, which the periodic
FFT and NUFSHT routes support; the scattered fit and FastSphericalHarmonics hold host state. `state` is
the solver's [`prepare_spectral`](@ref) state for `grid`, reused across calls and released with
[`close!`](@ref); without one, a state is built for the call and released on return. A fit stopped
short of its tolerance returns with its convergence in `χ_solve`; [`helmholtz_project_spectral`](@ref)
projects raw coefficients.
"""
function helmholtz_decompose_spectral(
    u::AbstractArray, grid::FlowGeometries.Grids.AbstractGrid;
    solver::AbstractPoissonSolver = AutoSolver(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    state = nothing,
)
    return _spectral_dispatch(u, grid, solver, execution, state)
end

helmholtz_decompose_spectral(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N},
                             grid::FlowGeometries.Grids.AbstractGrid; kwargs...) where {N} =
    helmholtz_decompose_spectral(_stack_components(grid, u, v), grid; kwargs...)

helmholtz_decompose_spectral(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}, w::AbstractArray{<:Any,N},
                             grid::FlowGeometries.Grids.AbstractGrid; kwargs...) where {N} =
    helmholtz_decompose_spectral(_stack_components(grid, u, v, w), grid; kwargs...)

"""
    prepare_spectral(solver, grid; execution = AutoBackend()) -> state

What [`helmholtz_decompose_spectral`](@ref) with `solver` reuses across fields on `grid`: the transform
plans and the buffers one decomposition writes, built under `execution`. One per task; [`close!`](@ref)
releases the foreign plans it holds.
"""
function prepare_spectral end

"""
    _spectral_dispatch(u, grid, solver, execution, state)

Resolve a solver with a spectral route ([`_spectral_candidates`](@ref) under `AutoSolver`) that runs
under `execution`, and dispatch to its extension's `_decompose_spectral(solver, geometry, u, grid,
execution, state)`, which places `u` in the memory its transforms run in. Without a `state`, one is built
for the call and released on return.
"""
function _spectral_dispatch(u::AbstractArray, grid::FlowGeometries.Grids.AbstractGrid,
                            solver::AbstractPoissonSolver, execution, state)
    exec = resolve_execution_backend(execution)
    s = _resolve_spectral_solver(grid, solver, exec)
    geom = FlowGeometries.Grids.grid_geometry(grid)
    state === nothing || return _decompose_spectral(s, geom, u, grid, exec, state)
    st = prepare_spectral(s, grid; execution = exec)
    try
        return _decompose_spectral(s, geom, u, grid, exec, st)
    finally
        close!(st)
    end
end

function _resolve_spectral_solver(grid::FlowGeometries.Grids.AbstractGrid, solver::AbstractPoissonSolver,
                                  execution = ComputationalBackends.SerialBackend())
    # A grid a spectral solver accepts has no boundary, so the condition is nominal here.
    s = solver isa AutoSolver ? _auto_spectral_solver(grid, execution) : solver
    s === nothing && throw(ArgumentError(
        "helmholtz_decompose_spectral found no spectral solver for this grid under " *
        "$(nameof(typeof(execution))). Load one that applies (an FFT library for the array for a " *
        "periodic uniform Cartesian grid, `NonuniformFFTs` or `FINUFFT` for scattered Cartesian " *
        "samples, `FastSphericalHarmonics` for a Fejér first-rule sphere, `NUFSHT` for an arbitrary " *
        "covering sphere, the last three on a host backend), or call `helmholtz_decompose`."))
    has_spectral_route(s) || throw(ArgumentError(
        "$(nameof(typeof(s))) has no spectral decomposition route; pass it to `helmholtz_decompose`, " *
        "or name one of CartesianSpectralSolver, CartesianNUFFTSolver, SphericalSpectralSolver, " *
        "SphericalNUSHTSolver."))
    _require_library(s)
    _require_domain(s, grid)
    _require_backend(s, execution, grid)
    return s
end

_auto_spectral_solver(grid::FlowGeometries.Grids.AbstractGrid, execution) =
    _first_applicable(_spectral_candidates(grid), grid, FlowGeometries.Operators.Neumann(), execution, nothing)

# Hook implemented by spectral extensions: dispatch on the solver type.
function _decompose_spectral end

# The `HelmholtzResult` of a component-last spectral velocity `velocity_hat` on a uniform, unmasked,
# periodic Cartesian grid. `K` is the derivative symbol, one array per axis shaped to broadcast along it;
# `kdotu` takes `k·û` and `scratch` each spectral scalar in turn, and `invert!(out, scratch)` writes its
# physical field into `out`, free to overwrite `scratch`. `N + P + 2` inverse transforms: `u_div` (`N`),
# the rotation potential (`P`), `χ` and the divergence. `u_harm` is each component's mean and
# `u_rot = U − u_div − u_harm`. Every cell has the same measure, so the fractions are ratios of plain sums
# of squares, which run on the array's own device.
function _cartesian_result(grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractCartesianGeometry,N},
                           U, velocity_hat, K, kdotu, scratch, invert!::F) where {N,T,F}
    comp(A, c) = _component(A, c, Val(N))
    # `1/k²`, zero where `k² = 0`, formed where it is read.
    inv_k2 = Broadcast.broadcasted(x -> iszero(x) ? zero(T) : one(T) / x,
                                   lazy_axis_sum(ntuple(d -> K[d] .^ 2, Val(N))))
    fill!(kdotu, zero(eltype(kdotu)))
    for b in 1:N
        Kb, ûb = K[b], comp(velocity_hat, b)
        kdotu .+= Kb .* ûb
    end

    P = FlowGeometries.Operators.n_rotation_components(N)
    dims = size(grid)

    # Allocated like `U`, so the result lives where the input does.
    u_div = similar(U, T, (dims..., N))
    for a in 1:N
        Ka = K[a]
        scratch .= Ka .* kdotu .* inv_k2
        invert!(comp(u_div, a), scratch)
    end

    # One array per component, as on the staggered path, so `streamfunction` and `vector_potential`
    # index either the same way.
    pairs = FlowGeometries.Operators.rotation_pairs(Val(N))
    Rpot = ntuple(Val(P)) do p
        a, b = pairs[p]
        Ka, Kb, ûa, ûb = K[a], K[b], comp(velocity_hat, a), comp(velocity_hat, b)
        scratch .= -im .* (Ka .* ûb .- Kb .* ûa) .* inv_k2
        invert!(similar(U, T, dims), scratch)
    end

    scratch .= -im .* kdotu .* inv_k2
    χ = invert!(similar(U, T, dims), scratch)
    scratch .= im .* kdotu
    divergence = invert!(similar(U, T, dims), scratch)

    u_harm = similar(U, T, (dims..., N))
    for c in 1:N
        Uc = _component(U, c, Val(N))
        fill!(_component(u_harm, c, Val(N)), sum(Uc) / length(Uc))
    end

    u_rot = similar(U, T, (dims..., N))
    @. u_rot = U - u_div - u_harm

    den = sum(abs2, U)
    hfrac = iszero(den) ? zero(T) : sqrt(sum(abs2, u_harm) / den)
    ok = SolverResult{T}(true, 1, zero(T))
    return HelmholtzResult{N,P,T,typeof(u_div),typeof(χ),typeof(Rpot)}(
        u_rot, u_div, u_harm, χ, Rpot, divergence, hfrac, zero(T), ok, ntuple(_ -> ok, Val(P)),
    )
end

"""
    _measure_ratio(A, B, grid) -> T
    _measure_ratio(A, B, w) -> T

`‖A‖/‖B‖` for component-last arrays over the cells or nodes of `grid`, weighted by its measure, or by
the cell weights `w` held in the arrays' own memory; zero where `B` is zero. The reductions run where
the arrays live.
"""
_measure_ratio(A::AbstractArray, B::AbstractArray, grid::FlowGeometries.Grids.AbstractGrid) =
    _measure_ratio(A, B, FlowGeometries.Grids.measure(grid))

function _measure_ratio(A::AbstractArray{T}, B::AbstractArray{T}, w::AbstractArray) where {T}
    sq(X) = sum(c -> sum(w .* abs2.(selectdim(X, ndims(X), c))), 1:size(X, ndims(X)))
    den = sq(B)
    return iszero(den) ? zero(T) : T(sqrt(sq(A) / den))
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
