"""
    DualGrid.jl — The grid the rotation potential lives on.

`χ` sits at cell centres and solves `L χ = D u` on the primal grid. `R` sits on `(a,b)` corners
and solves the same equation there, so rather than writing a second operator family for corners,
the corner locations are assembled into a `FlowGeometries` grid of their own and the *existing*
operators are applied to it.

That is not a convenience: the corner locations of a rectilinear grid are themselves rectilinear
— the faces of direction `d` sit at one coordinate each — so the dual really is a
`StructuredGrid`, with the same geometry, the same wrap in every periodic direction, and a mask
saying which corners have a closed loop of active cells around them. Everything that was
established once for the primal (adjointness, the metric entering only through `A/g`, the
boundary condition living in the face areas) then holds for the dual with no restatement.
"""

"""
    face_coordinates(grid, d) -> axis

Coordinates of the faces normal to direction `d` — the midpoints between consecutive cell
centres, plus the two outer edges where the direction is bounded.

A periodic direction gets `n` of them and a bounded one `n + 1`, matching [`nfaces`](@ref).

**A uniform axis returns a range.** Uniformly spaced cells have uniformly spaced faces, but
FlowGeometries proves uniformity from the axis TYPE, so returning a `Vector` here would make every
dual grid report itself as stretched — and `requires_uniform_axes` would then refuse the direct
transform on a grid that qualifies for it. That is not a small loss: it left the rotation-potential
solve iterative at every size, and it was the whole cost of a 512² decomposition.
"""
function face_coordinates(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, d::Integer) where {G,T,N}
    return _face_axis(grid, FlowGeometries.Grids.coordinates(grid, d), d, nfaces(grid, d))
end

function _face_axis(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, x::AbstractRange,
                    d::Integer, nf::Int) where {G,T,N}
    h = step(x)
    if FlowGeometries.Grids.isperiodic(grid, d)
        # The seam gap is whatever the period leaves over after the samples. It equals the interior
        # spacing only when the samples span the period evenly; otherwise the faces really are
        # unevenly spaced and the general path is the correct answer. One `O(1)` comparison.
        seam = FlowGeometries.Grids.period(grid, d) - (last(x) - first(x))
        isapprox(seam, h; rtol = sqrt(eps(float(typeof(h))))) ||
            return _face_axis(grid, collect(x), d, nf)
    end
    # `UniformAxis`, not `range`: it is FlowGeometries' own uniform axis and is `isbits`, so moving
    # the dual grid to a device costs nothing.
    return FlowGeometries.Axes.UniformAxis(first(x) - h / 2, h, nf)
end

function _face_axis(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, x, d::Integer,
                    nf::Int) where {G,T,N}
    n = length(x)
    out = similar(x, T, nf)
    if FlowGeometries.Grids.isperiodic(grid, d)
        # Face 1 is the seam, between the last cell and the first. The gap there is whatever the
        # period leaves over after the samples, which is the one gap not visible in `x`.
        p = T(FlowGeometries.Grids.period(grid, d))
        @inbounds seam = p - abs(T(x[n]) - T(x[1]))
        @inbounds out[1] = T(x[1]) - seam / 2
        @inbounds for i in 2:nf
            out[i] = (T(x[i - 1]) + T(x[i])) / 2
        end
    else
        @inbounds out[1] = T(x[1]) - (T(x[min(2, n)]) - T(x[1])) / 2
        @inbounds for i in 2:n
            out[i] = (T(x[i - 1]) + T(x[i])) / 2
        end
        @inbounds out[nf] = T(x[n]) + (T(x[n]) - T(x[max(n - 1, 1)])) / 2
    end
    return out
end

"""
    corner_mask(grid, a, b) -> Array{Bool}

Which `(a,b)` corners have a closed loop of active cells around them — the same four-face test
[`curl!`](@ref) applies, hoisted so the dual grid carries it.

A corner without a closed loop is not in the complex: the circulation around it is not defined,
so `R` there is not solved for and contributes nothing back.
"""
function corner_mask(grid::FaceIndexedGrid{T,G,N}, a::Integer, b::Integer, bc) where {G,T,N}
    dims = corner_dims(grid, a, b)
    m = falses(dims)
    every = true
    # Openness is the mask and the boundary condition, with no geometry in it, so this reads the
    # same on a rectilinear and a curvilinear grid.
    @inbounds for Q in CartesianIndices(dims)
        F = corner_to_face(grid, Q, a, b)
        Cla = cell_below(grid, F, a)
        Cua = cell_above(grid, F, a)
        Clb = cell_below(grid, F, b)
        Cub = cell_above(grid, F, b)
        v = !(Cla === nothing || Cua === nothing || Clb === nothing || Cub === nothing) &&
            !iszero(_face_openness(grid, Cla, b, bc, T)) &&
            !iszero(_face_openness(grid, Cua, b, bc, T)) &&
            !iszero(_face_openness(grid, Clb, a, bc, T)) &&
            !iszero(_face_openness(grid, Cub, a, bc, T))
        m[Q] = v
        every &= v
    end
    # An unmasked primal grid leaves every interior corner in the complex, and `AllActive` says so
    # in the type: the mask read folds to `true` in each dual operator, `all(mask)` becomes a
    # constant, and the dual metrics take the separable form.
    return every ? FlowGeometries.Grids.AllActive(dims) : m
end

"""
    dual_grid(grid, a, b, bc) -> grid

The grid the `(a,b)` rotation-potential component lives on: corner coordinates in directions `a`
and `b`, cell coordinates elsewhere, the primal's geometry and per-direction topology, and
[`corner_mask`](@ref) as its mask.

A bounded direction contributes its *interior* corners only — see [`ncorners`](@ref). The pair it
drops is where `R` is pinned to zero, and the dual solve states that as a `Dirichlet` condition,
which leaves an unmasked domain unmasked here and keeps its direct transform.

On a `CurvilinearGrid` the positions are already there: **its cell vertices are the dual's cell
centres**, and its cell centres are the dual's vertices, so the dual is built from the two arrays
the primal holds and `FlowGeometries` derives the dual measure from them. A periodic direction
takes its first vertex column across the seam, shifted by the period.
"""
function dual_grid(grid::FlowGeometries.Grids.CurvilinearGrid{T,G,2}, a::Integer, b::Integer,
                   bc) where {G,T}
    (a, b) == (1, 2) || throw(ArgumentError(
        "a 2-D grid has the single rotation pair (1, 2); got ($a, $b)"))
    FlowGeometries.Grids.has_corners(grid) || throw(ArgumentError(
        "the dual grid is built from this CurvilinearGrid's cell-vertex arrays, which it did not " *
        "retain. Rebuild it with `keep_corners = true`, or pass `corners`."))
    k = FlowGeometries.Grids.corners(grid)
    c = FlowGeometries.Grids.coordinates(grid)
    n = size(grid)
    off = ntuple(d -> corner_offset(grid, d), Val(2))
    per = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(2))
    cdims = corner_dims(grid, a, b)

    # Dual centre `p` along `d` is primal vertex `p + off[d]`.
    centres = ntuple(Val(2)) do e
        out = Matrix{T}(undef, cdims)
        @inbounds for j in 1:cdims[2], i in 1:cdims[1]
            out[i, j] = T(k[e][i + off[1], j + off[2]])
        end
        out
    end

    # Dual vertices are the primal centres, one more than the dual has cells in each direction.
    vdims = cdims .+ 1
    L = ntuple(d -> per[d] ? T(FlowGeometries.Grids.period(grid, d)) : zero(T), Val(2))
    verts = ntuple(Val(2)) do e
        out = Matrix{T}(undef, vdims)
        @inbounds for j in 1:vdims[2], i in 1:vdims[1]
            i1, w1 = _dual_vertex_source(i, n[1], per[1])
            i2, w2 = _dual_vertex_source(j, n[2], per[2])
            shift = (e == 1 ? w1 * L[1] : zero(T)) + (e == 2 ? w2 * L[2] : zero(T))
            out[i, j] = T(c[e][i1, i2]) - shift
        end
        out
    end

    return FlowGeometries.Grids.CurvilinearGrid(
        FlowGeometries.Grids.grid_geometry(grid), centres..., corner_mask(grid, a, b, bc);
        corners = verts,
        topology = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(2)),
        period = ntuple(d -> per[d] ? FlowGeometries.Grids.period(grid, d) : nothing, Val(2)),
    )
end

# Which primal centre a dual vertex sits on, and how many periods to subtract.
#
# Bounded: dual cell `p ∈ 1:n−1` spans primal centres `p` and `p+1`, so vertex `q` is centre `q`.
# Periodic: dual cell `p ∈ 1:n` spans centres `p−1` and `p`, so vertex 1 is centre `n` taken one
# period back across the seam and vertex `q > 1` is centre `q−1`.
@inline function _dual_vertex_source(q::Int, n::Int, periodic::Bool)
    periodic || return (q, 0)
    return q == 1 ? (n, 1) : (q - 1, 0)
end

"""
    _cell_axis(x) -> axis

The primal axis `x` in the form [`face_coordinates`](@ref) gives a uniform direction.

A direction's dual axis then has the same type whichever pair asks for it, so the `P` dual grids of
one primal grid share a type. Where they do not, `plan.dual_grids[p]` under a loop variable is a
`P`-way union and every rotation solve dispatches dynamically.
"""
@inline _cell_axis(x::AbstractRange) =
    FlowGeometries.Axes.UniformAxis(first(x), step(x), length(x))
@inline _cell_axis(x) = x

function dual_grid(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, a::Integer, b::Integer,
                   bc) where {G,T,N}
    axes_dual = ntuple(Val(N)) do d
        if d == a || d == b
            o = corner_offset(grid, d)
            x = face_coordinates(grid, d)
            # Slicing keeps a range a range, so a uniform grid stays provably uniform.
            iszero(o) ? x : x[(1 + o):(end - o)]
        else
            _cell_axis(FlowGeometries.Grids.coordinates(grid, d))
        end
    end
    topo = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(N))
    per = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d) ?
                      FlowGeometries.Grids.period(grid, d) : nothing, Val(N))
    return FlowGeometries.Grids.StructuredGrid(
        FlowGeometries.Grids.grid_geometry(grid), axes_dual...;
        mask = corner_mask(grid, a, b, bc), topology = topo, period = per,
    )
end
