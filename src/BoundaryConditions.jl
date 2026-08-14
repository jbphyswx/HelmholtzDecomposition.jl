"""
    BoundaryConditions.jl — What closes a bounded direction.

A condition is only meaningful on a **bounded** direction. Whether a direction wraps is the
grid's own topology (`FlowGeometries.Grids.isperiodic`), carried in its type, so there is no
`Periodic` boundary condition here: a periodic direction has no boundary to impose one on, and
a second, unchecked copy of that fact could only contradict the grid.

These are not flags handed to a solver. They select which discrete operator is built — see
`face_area`, where a `Neumann` domain edge carries no flux and a `Dirichlet` one carries its
geometric area against a zero ghost value. That is what a boundary condition *is* in flux form,
and it is why the solver cannot silently ignore one.
"""

"""
    AbstractBoundaryCondition

Supertype for conditions on a bounded direction. Concrete: [`Dirichlet`](@ref), [`Neumann`](@ref).
"""
abstract type AbstractBoundaryCondition end

"Homogeneous Dirichlet condition (`Φ = 0` on the boundary)."
struct Dirichlet <: AbstractBoundaryCondition end

"Homogeneous Neumann condition (`∂Φ/∂n = 0` on the boundary): no flux through the domain edge."
struct Neumann <: AbstractBoundaryCondition end

# Whether the resulting Laplacian is singular is deliberately NOT asked of the condition: a
# grid whose mask closes off every boundary face is singular under `Dirichlet` too. It is
# decided by applying the operator to the constant field — see `_detect_singular`.
