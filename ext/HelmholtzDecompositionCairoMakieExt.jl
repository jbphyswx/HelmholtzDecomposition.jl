"""
    HelmholtzDecompositionCairoMakieExt — Visualization helpers for Helmholtz decomposition.

Provides plotting functions for velocity fields, potentials, and decomposed components.
"""
module HelmholtzDecompositionCairoMakieExt

using HelmholtzDecomposition: HelmholtzDecomposition
using CairoMakie: CairoMakie

"""
    plot_decomposition(result, grid; kwargs...) → Figure

Plot the original field, rotational component, and divergent component side-by-side
as heatmaps. Optionally overlay quiver arrows.

# Arguments
- `result::HelmholtzDecomposition.HelmholtzResult` — Decomposition result
- `grid::HelmholtzDecomposition.StructuredGrid` — Grid for coordinates
- `u_orig::AbstractMatrix` — Original u velocity (for top row)
- `v_orig::AbstractMatrix` — Original v velocity (for top row)
"""
function HelmholtzDecomposition.plot_decomposition(
    result::HelmholtzDecomposition.HelmholtzResult,
    grid::HelmholtzDecomposition.StructuredGrid,
    u_orig::AbstractMatrix,
    v_orig::AbstractMatrix;
    kwargs...
)
    # TODO: Implement full visualization with CairoMakie
    @info "plot_decomposition: CairoMakie extension loaded but full implementation pending"
    return nothing
end

end # module
