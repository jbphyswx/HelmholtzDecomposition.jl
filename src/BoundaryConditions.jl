# A condition closes a bounded direction; a direction that wraps has none. The primal grid takes
# `FlowGeometries.Operators.Dirichlet()` or `Neumann()`, which select the face areas its operators
# couple through.

"""
    CornerDirichlet(staggered::NTuple{N,Bool}, ghost = nothing)

The condition the rotation potential is solved under on a two-dimensional dual grid: `R = 0` at the
corner one cell beyond the outermost unknown along each `staggered` direction, the corners the dual
grid drops. A sine transform inverts it as DST-I (`RODFT00`) along those directions. `ghost[d]` holds
the coordinates of the two dropped corners along a bounded direction `d` (`nothing` along one that
wraps), where the multigrid levels keep the zero.
"""
struct CornerDirichlet{N,G} <: FlowGeometries.Operators.AbstractBoundaryCondition
    staggered::NTuple{N,Bool}
    ghost::G
end

CornerDirichlet(staggered::NTuple{N,Bool}) where {N} = CornerDirichlet(staggered, nothing)
