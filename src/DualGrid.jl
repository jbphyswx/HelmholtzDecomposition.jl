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
function face_coordinates(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, d::Integer) where {G,T,N}
    return _face_axis(grid, FlowGeometries.Grids.coordinates(grid, d), d, nfaces(grid, d))
end

function _face_axis(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, x::AbstractRange,
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

function _face_axis(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, x, d::Integer,
                    nf::Int) where {G,T,N}
    n = length(x)
    out = Vector{T}(undef, nf)
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
function corner_mask(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, a::Integer, b::Integer,
                     bc) where {G,T,N}
    dims = corner_dims(grid, a, b)
    m = falses(dims)
    @inbounds for Q in CartesianIndices(dims)
        F = corner_to_face(grid, Q, a, b)
        Cla = cell_below(grid, F, a)
        Cua = cell_above(grid, F, a)
        Clb = cell_below(grid, F, b)
        Cub = cell_above(grid, F, b)
        (Cla === nothing || Cua === nothing || Clb === nothing || Cub === nothing) && continue
        m[Q] = !iszero(face_area(grid, Cla, b, bc, T)) && !iszero(face_area(grid, Cua, b, bc, T)) &&
               !iszero(face_area(grid, Clb, a, bc, T)) && !iszero(face_area(grid, Cub, a, bc, T))
    end
    return m
end

"""
    dual_grid(grid, a, b, bc) -> StructuredGrid

The grid the `(a,b)` rotation-potential component lives on: corner coordinates in directions `a`
and `b`, cell coordinates elsewhere, the primal's geometry and per-direction topology, and
[`corner_mask`](@ref) as its mask.

A bounded direction contributes its *interior* corners only — see [`ncorners`](@ref). The pair it
drops is where `R` is pinned to zero, which the dual solve states as `Dirichlet` rather than as a
mask, so an unmasked domain stays unmasked here and keeps its direct transform.
"""
function dual_grid(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, a::Integer, b::Integer,
                   bc) where {G,T,N}
    axes_dual = ntuple(Val(N)) do d
        if d == a || d == b
            o = corner_offset(grid, d)
            x = face_coordinates(grid, d)
            # Slicing keeps a range a range, so a uniform grid stays provably uniform.
            iszero(o) ? x : x[(1 + o):(end - o)]
        else
            FlowGeometries.Grids.coordinates(grid, d)
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
