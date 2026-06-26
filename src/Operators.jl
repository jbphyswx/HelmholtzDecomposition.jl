"""
    Operators.jl — Dimension-generic finite-difference operators and rotation-tensor
    bookkeeping for Cartesian grids.

The rotational part of an `N`-dimensional Helmholtz decomposition is encoded by an
antisymmetric rotation-potential matrix `R` with `N(N-1)/2` independent components
(Glötzl & Richters, 2023). In 2D this single component is the streamfunction `ψ`; in
3D the three components are the Hodge dual of the vector potential `A`. This file
provides the dimension-generic machinery used by the physical-space decomposition:

- pair bookkeeping `(a, b)` with `a < b` indexing the independent components,
- masked centered finite differences along an arbitrary axis,
- divergence `δ = Σ_d ∂u_d/∂x_d`,
- the antisymmetric rotation tensor `W_ab = ∂_a u_b − ∂_b u_a`,
- reconstruction of `u_div = ∇χ` and `u_rot` from the potentials.

Conventions (verified against the 2D streamfunction and 3D vector-potential cases):
`Δχ = δ`, `ΔR_ab = W_ab`, `u_div_k = ∂_k χ`, and `u_rot_k = −Σ_m ∂_m R_km`
(i.e. each pair `(a,b)` contributes `u_rot_a -= ∂_b R_ab`, `u_rot_b += ∂_a R_ab`).
"""

# ---------------------------------------------------------------------------
# Rotation-component bookkeeping
# ---------------------------------------------------------------------------

"""
    n_rotation_components(N) -> Int

Number of independent rotation-potential components `N(N-1)/2` for `N` spatial
dimensions (0 in 1D, 1 in 2D, 3 in 3D, 6 in 4D, …).
"""
@inline n_rotation_components(N::Integer) = (N * (N - 1)) ÷ 2

"""
    rotation_pairs(Val(N)) -> NTuple{P, Tuple{Int,Int}}

Compile-time tuple of the `P = N(N-1)/2` index pairs `(a, b)` with `a < b`, in
lexicographic order: `(1,2), (1,3), …, (1,N), (2,3), …`.
"""
@generated function rotation_pairs(::Val{N}) where {N}
    pairs = Tuple{Int,Int}[]
    for a in 1:(N - 1), b in (a + 1):N
        push!(pairs, (a, b))
    end
    return Expr(:tuple, (Expr(:tuple, p[1], p[2]) for p in pairs)...)
end

# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

"""
    _unit(Val(N), d) -> CartesianIndex{N}

Unit offset along axis `d`.
"""
@inline _unit(::Val{N}, d::Integer) where {N} = CartesianIndex(ntuple(i -> i == d ? 1 : 0, Val(N)))

"""
    _wet_neighbor(grid, I, e, dir) -> CartesianIndex

Neighbor of `I` offset by `dir * e`, clamped back to `I` when out of bounds or masked
out (one-sided/no-flux behavior at boundaries and land).
"""
@inline function _wet_neighbor(grid, I::CartesianIndex{N}, e::CartesianIndex{N}, dir::Integer) where {N}
    J = I + dir * e
    return (checkbounds(Bool, grid.mask, J) && @inbounds grid.mask[J]) ? J : I
end

"""
    _deriv(field, grid, I, d, spacing_d) -> T

Masked centered finite difference `∂field/∂x_d` at index `I` on a Cartesian grid.
Falls back to a one-sided difference at boundaries/land and returns zero when no
valid stencil exists.
"""
@inline function _deriv(field, grid, I::CartesianIndex{N}, d::Integer, spacing_d::T) where {N,T}
    e = _unit(Val(N), d)
    Ip = _wet_neighbor(grid, I, e, +1)
    Im = _wet_neighbor(grid, I, e, -1)
    step = (Ip[d] - Im[d]) * spacing_d
    step == 0 && return zero(T)
    return (field[Ip] - field[Im]) / step
end

# ---------------------------------------------------------------------------
# Component-array views (component-last layout: U has size (dims..., N))
# ---------------------------------------------------------------------------

@inline function _component(U::AbstractArray{<:Any,M}, c::Integer, ::Val{N}) where {M,N}
    return @view U[ntuple(_ -> Colon(), Val(N))..., c]
end

# ---------------------------------------------------------------------------
# Divergence and rotation tensor (Cartesian, dimension-generic)
# ---------------------------------------------------------------------------

"""
    cartesian_divergence!(div, U, grid)

Compute `δ = Σ_d ∂u_d/∂x_d` into the `N`-d array `div` from the component-last
velocity array `U` (size `(dims..., N)`). Masked-out cells are set to zero.
"""
function cartesian_divergence!(div, U, grid::StructuredGrid{N,<:CartesianGeometry{N,T}}) where {N,T}
    spacing = grid.geometry.spacing
    comps = ntuple(c -> _component(U, c, Val(N)), Val(N))
    fill!(div, zero(T))
    @inbounds for I in CartesianIndices(grid.mask)
        grid.mask[I] || continue
        acc = zero(T)
        for d in 1:N
            acc += _deriv(comps[d], grid, I, d, spacing[d])
        end
        div[I] = acc
    end
    return div
end

"""
    cartesian_rotation_tensor!(W, U, grid)

Compute the independent components of the antisymmetric rotation tensor
`W_ab = ∂_a u_b − ∂_b u_a` (for pairs `a < b`) into the component-last array `W`
(size `(dims..., P)`, `P = N(N-1)/2`). Masked-out cells are set to zero.
"""
function cartesian_rotation_tensor!(W, U, grid::StructuredGrid{N,<:CartesianGeometry{N,T}}) where {N,T}
    spacing = grid.geometry.spacing
    pairs = rotation_pairs(Val(N))
    comps = ntuple(c -> _component(U, c, Val(N)), Val(N))
    fill!(W, zero(T))
    @inbounds for (p, (a, b)) in enumerate(pairs)
        ua = comps[a]
        ub = comps[b]
        Wp = _component(W, p, Val(N))
        for I in CartesianIndices(grid.mask)
            grid.mask[I] || continue
            Wp[I] = _deriv(ub, grid, I, a, spacing[a]) - _deriv(ua, grid, I, b, spacing[b])
        end
    end
    return W
end

# ---------------------------------------------------------------------------
# Reconstruction of velocity components from potentials
# ---------------------------------------------------------------------------

"""
    cartesian_reconstruct_div!(u_div, χ, grid)

Reconstruct the divergent (curl-free) velocity `u_div_k = ∂_k χ` into the component-last
array `u_div` (size `(dims..., N)`). Masked-out cells are zeroed.
"""
function cartesian_reconstruct_div!(u_div, χ, grid::StructuredGrid{N,<:CartesianGeometry{N,T}}) where {N,T}
    spacing = grid.geometry.spacing
    fill!(u_div, zero(T))
    @inbounds for k in 1:N
        out = _component(u_div, k, Val(N))
        for I in CartesianIndices(grid.mask)
            grid.mask[I] || continue
            out[I] = _deriv(χ, grid, I, k, spacing[k])
        end
    end
    return u_div
end

"""
    cartesian_reconstruct_rot!(u_rot, R, grid)

Reconstruct the rotational (divergence-free) velocity from the rotation potential `R`
(component-last, size `(dims..., P)`). Each pair `(a,b)` contributes
`u_rot_a -= ∂_b R_ab` and `u_rot_b += ∂_a R_ab`. Masked-out cells are zeroed.
"""
function cartesian_reconstruct_rot!(u_rot, R, grid::StructuredGrid{N,<:CartesianGeometry{N,T}}) where {N,T}
    spacing = grid.geometry.spacing
    pairs = rotation_pairs(Val(N))
    fill!(u_rot, zero(T))
    out = ntuple(c -> _component(u_rot, c, Val(N)), Val(N))
    @inbounds for (p, (a, b)) in enumerate(pairs)
        Rp = _component(R, p, Val(N))
        ua = out[a]
        ub = out[b]
        for I in CartesianIndices(grid.mask)
            grid.mask[I] || continue
            ua[I] -= _deriv(Rp, grid, I, b, spacing[b])
            ub[I] += _deriv(Rp, grid, I, a, spacing[a])
        end
    end
    return u_rot
end
