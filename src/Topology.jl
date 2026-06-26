"""
    Topology.jl — Domain topology diagnostics for the harmonic Helmholtz component.

On a bounded, multiply-connected domain the Helmholtz-Hodge decomposition acquires a
nonzero harmonic component `h` whose dimension equals the first Betti number `b₁` of the
wet region (the number of holes/islands). These helpers detect that topology so a large
[`harmonic_fraction`](@ref HelmholtzResult) can be attributed to genuine topology rather
than a boundary-condition mismatch.

Detection is grid-based: the number of "islands/holes" is the number of connected
components of the masked-out (dry) region that are fully enclosed by wet cells — i.e. that
do not touch the outer boundary of the array (using 1-connectivity along each axis,
dimension-generically).
"""

export count_islands, betti1_estimate

"""
    count_islands(grid) -> Int

Number of connected dry (masked-out) components fully enclosed by wet cells — an estimate
of the first Betti number `b₁` of the wet region, i.e. the dimension of the harmonic
subspace. Returns `0` for a fully wet or simply-connected domain.
"""
count_islands(grid::StructuredGrid) = count_enclosed_components(grid.mask)

"""
    betti1_estimate(grid) -> Int

Alias for [`count_islands`](@ref): an estimate of `b₁` of the wet region.
"""
betti1_estimate(grid::StructuredGrid) = count_islands(grid)

"""
    count_enclosed_components(mask::AbstractArray{Bool,N}) -> Int

Number of connected components of `.!mask` (the dry region) that do not touch any face of
the array bounding box. Uses face-connectivity (±1 along each axis).
"""
function count_enclosed_components(mask::AbstractArray{Bool,N}) where {N}
    dry = .!mask
    any(dry) || return 0
    visited = falses(size(mask))
    dims = size(mask)
    enclosed = 0
    stack = CartesianIndex{N}[]
    @inbounds for seed in CartesianIndices(mask)
        (dry[seed] && !visited[seed]) || continue
        # Flood-fill this dry component, tracking whether it touches the array boundary.
        empty!(stack)
        push!(stack, seed)
        visited[seed] = true
        touches_boundary = false
        while !isempty(stack)
            I = pop!(stack)
            t = Tuple(I)
            for d in 1:N
                if t[d] == 1 || t[d] == dims[d]
                    touches_boundary = true
                end
            end
            for d in 1:N
                e = _unit(Val(N), d)
                for J in (I + e, I - e)
                    (checkbounds(Bool, mask, J) && dry[J] && !visited[J]) || continue
                    visited[J] = true
                    push!(stack, J)
                end
            end
        end
        touches_boundary || (enclosed += 1)
    end
    return enclosed
end
