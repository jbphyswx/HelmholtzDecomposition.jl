# The grid the rotation potential lives on. `χ` sits at cell centres and solves `L χ = D u` on the
# primal grid; each component `R_ab` sits on the `(a, b)` corners, assembled into a FlowGeometries grid
# of its own: the corner coordinates along `a` and `b`, the cell coordinates elsewhere, the primal's
# geometry and wrap, and `FlowGeometries.Operators.corner_mask` as its mask.

"""
    dual_grid(grid, a, b, fm) -> grid

The grid the `(a, b)` rotation-potential component lives on, under the primal face metrics `fm`: one
cell per corner of `FlowGeometries.Grids.corner_dims`, active where
`FlowGeometries.Operators.corner_mask` puts the corner in the complex.

A bounded direction contributes its interior corners; `R = 0` on the pair it drops, the condition
[`dual_boundary`](@ref) states, so an unmasked primal grid gives an unmasked dual grid.

On a `CurvilinearGrid` the primal's cell vertices are the dual's cell centres and its cell centres are
the dual's vertices; a periodic direction takes its first vertex column across the seam, one period
back.
"""
function dual_grid(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, a::Integer, b::Integer,
                   fm::FlowGeometries.Operators.FaceMetrics) where {G,T,N}
    axes_dual = ntuple(Val(N)) do d
        d == a || d == b ? FlowGeometries.Grids.corner_coordinates(grid, d) :
                           _cell_axis(FlowGeometries.Grids.coordinates(grid, d))
    end
    topo = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(N))
    per = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d) ?
                      FlowGeometries.Grids.period(grid, d) : nothing, Val(N))
    return FlowGeometries.Grids.StructuredGrid(
        FlowGeometries.Grids.grid_geometry(grid), axes_dual...;
        mask = FlowGeometries.Operators.corner_mask(fm, a, b), topology = topo, period = per,
    )
end

function dual_grid(grid::FlowGeometries.Grids.CurvilinearGrid{T,G,2}, a::Integer, b::Integer,
                   fm::FlowGeometries.Operators.FaceMetrics) where {G,T}
    (a, b) == (1, 2) || throw(ArgumentError(
        "a 2-D grid has the single rotation pair (1, 2); got ($a, $b)"))
    FlowGeometries.Grids.has_corners(grid) || throw(ArgumentError(
        "the dual grid is built from this CurvilinearGrid's cell vertices, which it does not hold; " *
        "build it with `corners` given"))
    k = FlowGeometries.Grids.corners(grid)
    c = FlowGeometries.Grids.coordinates(grid)
    n = FlowGeometries.Grids.size_tuple(grid)
    off = ntuple(d -> FlowGeometries.Grids.corner_offset(grid, d), Val(2))
    per = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(2))
    cdims = FlowGeometries.Grids.corner_dims(grid, a, b)

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
        FlowGeometries.Grids.grid_geometry(grid), centres..., FlowGeometries.Operators.corner_mask(fm, a, b);
        corners = verts, topology = per,
        period = ntuple(d -> per[d] ? FlowGeometries.Grids.period(grid, d) : nothing, Val(2)),
    )
end

# Which primal centre a dual vertex sits on, and how many periods to subtract. Bounded: dual cell
# `p ∈ 1:n−1` spans primal centres `p` and `p+1`, so vertex `q` is centre `q`. Periodic: dual cell
# `p ∈ 1:n` spans centres `p−1` and `p`, so vertex 1 is centre `n` one period back and vertex `q > 1`
# is centre `q−1`.
@inline function _dual_vertex_source(q::Int, n::Int, periodic::Bool)
    periodic || return (q, 0)
    return q == 1 ? (n, 1) : (q - 1, 0)
end

# A primal direction's axis in the form a corner axis of a uniform direction takes, so the `P` dual
# grids of one primal grid share a type.
@inline _cell_axis(x::AbstractRange) = FlowGeometries.Axes.UniformAxis(first(x), step(x), length(x))
@inline _cell_axis(x) = x

"""
    dual_boundary(grid, a, b) -> AbstractBoundaryCondition

The condition the `(a, b)` rotation potential is solved under: on a two-dimensional grid
[`CornerDirichlet`](@ref), `R = 0` at the dropped corners one cell beyond the outermost unknowns, the
values `rotational_velocity!` reads there; in three or more dimensions `Dirichlet()`.
"""
dual_boundary(::FlowGeometries.Grids.FaceIndexedGrid{T,G,2}, a::Integer, b::Integer) where {G,T} =
    CornerDirichlet((true, true))
dual_boundary(::FlowGeometries.Grids.FaceIndexedGrid, a::Integer, b::Integer) =
    FlowGeometries.Operators.Dirichlet()

"""
    dual_coefficients(grid, fm, a, b, dual) -> LaplacianCoefficients

The operator the `(a, b)` rotation potential is solved with: on a two-dimensional grid exactly
`curl ∘ rot`, FlowGeometries' `curl!` of `rotational_velocity!` under the primal metrics `fm`. Two
corners joined by an open primal face `f` couple with conductance `g_f / A_f`, gap over area, under
the corner area `μ` of `fm.corner`; a corner the dual grid drops or masks holds `R = 0`. In three or
more dimensions the dual grid's own `Dirichlet` operator.
"""
function dual_coefficients(grid::FlowGeometries.Grids.FaceIndexedGrid{T,G,2},
                           fm::FlowGeometries.Operators.FaceMetrics, a::Integer, b::Integer,
                           dual) where {G,T}
    layout = FlowGeometries.Operators.metric_layout(dual)
    coef = ntuple(Val(2)) do d
        FlowGeometries.Operators.metric_array(F -> _dual_conductance(grid, fm, a, b, d, F, T),
                                              FlowGeometries.Grids.face_dims(dual, d), layout, T)
    end
    μ = fm.corner[1]
    meas = FlowGeometries.Operators.metric_array(
        Q -> FlowGeometries.Grids.isactive(dual, Tuple(Q)...) ? T(μ[Q]) : zero(T),
        FlowGeometries.Grids.size_tuple(dual), layout, T)
    return FlowGeometries.Operators.LaplacianCoefficients(dual, coef, meas)
end

dual_coefficients(grid::FlowGeometries.Grids.FaceIndexedGrid, fm::FlowGeometries.Operators.FaceMetrics,
                  a::Integer, b::Integer, dual) = _laplacian(dual, FlowGeometries.Operators.Dirichlet())

# The conductance of dual face `F` along `d`: gap over area of the open primal face normal to the other
# staggered direction that joins the two corners `F` separates.
function _dual_conductance(grid, fm, a, b, d, F::CartesianIndex{2}, ::Type{T}) where {T}
    e = d == a ? b : a
    n = FlowGeometries.Grids.size_tuple(grid)[d]
    cell = FlowGeometries.Grids.isperiodic(grid, d) ? (F[d] == 1 ? n : F[d] - 1) : F[d]
    f = CartesianIndex(ntuple(k -> k == d ? cell : F[k] + FlowGeometries.Grids.corner_offset(grid, e), Val(2)))
    A = T(fm.area_op[e][f])
    return iszero(A) ? zero(T) : T(fm.gap[e][f]) / A
end
