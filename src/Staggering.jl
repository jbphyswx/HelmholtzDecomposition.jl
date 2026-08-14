"""
    Staggering.jl — Where each quantity lives, and how many of it there are.

`χ` sits at cell centres, normal velocity on faces, and `R_ab` on the `(a,b)` corner/edge. The
counts differ per direction and that difference is the point:

- a **periodic** direction of `n` cells has `n` faces — the face below cell 1 *is* the face above
  cell `n`;
- a **bounded** direction of `n` cells has `n + 1`.

Storing faces as one `(dims..., N)` block gives `n` slots either way, so a bounded direction's
outermost face has nowhere to live. Under a no-flux condition it carries nothing and the loss is
invisible; under Dirichlet it is open, and an operator that silently drops it is imposing
Dirichlet on one side of the domain and Neumann on the other. Hence one array per direction,
each sized for its own direction.

Face array `d` is indexed so that face `F` along `d` separates cell `F − 1` from cell `F`.
"""

"""
    nfaces(grid, d) -> Int

Number of faces normal to direction `d`: `n` if the direction wraps, `n + 1` if it ends.
"""
@inline nfaces(grid, d::Integer) =
    FlowGeometries.Grids.isperiodic(grid, d) ? size(grid, d) : size(grid, d) + 1

"""
    face_dims(grid, d) -> NTuple{N,Int}

Shape of the face array normal to direction `d`.
"""
@inline face_dims(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, d::Integer) where {G,T,N} =
    ntuple(e -> e == d ? nfaces(grid, e) : size(grid, e), Val(N))

"""
    allocate_faces(T, grid) -> NTuple{N,Array{T,N}}

One zeroed face array per direction. This is the internal staggered layout; the public API stays
collocated, so nothing outside this package sees it.
"""
allocate_faces(::Type{T}, grid::FlowGeometries.Grids.StructuredGrid{G,TT,N};
               backend = ComputationalBackends.SerialBackend()) where {T,G,TT,N} =
    ntuple(d -> allocate_zeros(backend, T, face_dims(grid, d)), Val(N))

"""
    ncorners(grid, d) -> Int

Corners along a staggered direction: `n` where it wraps, `n - 1` where it ends.

A bounded direction has `n + 1` faces but only `n - 1` corners that carry an unknown. The two
outermost have no closed loop of cells around them, so the circulation there is not defined and
`R` is not solved for — it is zero. Leaving them in the array and masking them off says the same
thing, but says it as *data*, and a masked grid is refused by every direct transform. Dropping
them says it as a **Dirichlet condition on a smaller domain**, which a sine transform inverts
exactly. Measured at 512²: 98 iterations and 4.6 s became one transform and 19 ms.
"""
@inline ncorners(grid::FlowGeometries.Grids.StructuredGrid, d::Integer) =
    FlowGeometries.Grids.isperiodic(grid, d) ? nfaces(grid, d) : nfaces(grid, d) - 2

"""
    corner_offset(grid, d) -> Int

What to add to a corner index in direction `d` to get the face index it sits on: `1` where the
outer pair was dropped, `0` where the direction wraps and nothing was.
"""
@inline corner_offset(grid::FlowGeometries.Grids.StructuredGrid, d::Integer) =
    FlowGeometries.Grids.isperiodic(grid, d) ? 0 : 1

"""
    corner_dims(grid, a, b) -> NTuple{N,Int}

Shape of the array holding a rotation-potential component staggered in both `a` and `b`.
"""
@inline corner_dims(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, a::Integer, b::Integer) where {G,T,N} =
    ntuple(e -> (e == a || e == b) ? ncorners(grid, e) : size(grid, e), Val(N))

# Corner index -> face index, and back. `to_faces` is exact; `to_corner` can land outside the
# array, which is the dropped ring and means `R = 0` there.
@inline corner_to_face(grid, Q::CartesianIndex{N}, a::Integer, b::Integer) where {N} =
    CartesianIndex(ntuple(e -> Q[e] + (e == a ? corner_offset(grid, a) :
                                       e == b ? corner_offset(grid, b) : 0), Val(N)))

@inline face_to_corner(grid, F::CartesianIndex{N}, a::Integer, b::Integer) where {N} =
    CartesianIndex(ntuple(e -> F[e] - (e == a ? corner_offset(grid, a) :
                                       e == b ? corner_offset(grid, b) : 0), Val(N)))

"""
    allocate_corners(T, grid; backend) -> NTuple{P,AbstractArray{T,N}}

One zeroed corner array per rotation pair, shaped by [`corner_dims`](@ref).
"""
allocate_corners(::Type{T}, grid::FlowGeometries.Grids.StructuredGrid{G,TT,N};
                 backend = ComputationalBackends.SerialBackend()) where {T,G,TT,N} =
    ntuple(p -> allocate_zeros(backend, T, corner_dims(grid, rotation_pairs(Val(N))[p]...)),
           Val(n_rotation_components(N)))

# ---------------------------------------------------------------------------
# Walking between cells and faces
# ---------------------------------------------------------------------------

@inline _unit(::Val{N}, d::Integer) where {N} = CartesianIndex(ntuple(i -> i == d ? 1 : 0, Val(N)))

"""
    cell_below(grid, F, d) -> CartesianIndex or nothing

The cell on the low side of face `F` along `d`; `nothing` at the outer edge of a bounded
direction.
"""
@inline function cell_below(grid, F::CartesianIndex{N}, d::Integer) where {N}
    F[d] > 1 && return F - _unit(Val(N), d)
    FlowGeometries.Grids.isperiodic(grid, d) || return nothing
    return CartesianIndex(ntuple(j -> j == d ? size(grid, d) : F[j], Val(N)))
end

"""
    cell_above(grid, F, d) -> CartesianIndex or nothing

The cell on the high side of face `F` along `d`; `nothing` at the outer edge of a bounded
direction. On a periodic direction every face has both.
"""
@inline function cell_above(grid, F::CartesianIndex{N}, d::Integer) where {N}
    F[d] <= size(grid, d) && return F
    return nothing        # only reachable when bounded, where nfaces = n + 1
end

"""
    face_below(I, d) / face_above(grid, I, d) -> CartesianIndex

The two faces of cell `I` normal to `d`, as indices into face array `d`. On a periodic direction
the face above the last cell wraps to face 1.
"""
@inline face_below(I::CartesianIndex, ::Integer) = I

@inline function face_above(grid, I::CartesianIndex{N}, d::Integer) where {N}
    if FlowGeometries.Grids.isperiodic(grid, d) && I[d] == size(grid, d)
        return CartesianIndex(ntuple(j -> j == d ? 1 : I[j], Val(N)))
    end
    return I + _unit(Val(N), d)
end
