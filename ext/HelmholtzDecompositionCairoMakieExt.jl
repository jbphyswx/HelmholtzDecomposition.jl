"""
    HelmholtzDecompositionCairoMakieExt — visualization for 2-D Helmholtz decompositions.

Provides `plot_decomposition`, which lays out heatmaps of the speed `|u|` for the original
field and its rotational, divergent, and harmonic components.
"""
module HelmholtzDecompositionCairoMakieExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using CairoMakie: CairoMakie

_speed(U) = sqrt.(view(U, :, :, 1) .^ 2 .+ view(U, :, :, 2) .^ 2)

"""
    plot_decomposition(result::HelmholtzResult{2}, grid; orig=nothing) -> Figure

Heatmaps of the speed `|u|` for the rotational, divergent, and harmonic components of a 2-D
decomposition (and the original field, if `orig` — a component-last `(Nx, Ny, 2)` array — is
supplied). Returns the `CairoMakie.Figure`.
"""
function HD.plot_decomposition(
    result::HD.HelmholtzResult{2},
    grid::HD.StructuredGrid{2};
    orig::Union{Nothing,AbstractArray} = nothing,
)
    xs, ys = grid.coords_axes
    panels = Tuple{String,Any}[]
    orig === nothing || push!(panels, ("original |u|", _speed(orig)))
    push!(panels, ("rotational |u_rot|", _speed(result.u_rot)))
    push!(panels, ("divergent |u_div|", _speed(result.u_div)))
    push!(panels, ("harmonic |u_harm|", _speed(result.u_harm)))

    fig = CairoMakie.Figure(; size = (320 * length(panels), 300))
    for (k, (title, field)) in enumerate(panels)
        ax = CairoMakie.Axis(fig[1, k]; title = title, xlabel = "x", ylabel = "y", aspect = 1)
        hm = CairoMakie.heatmap!(ax, xs, ys, field)
        CairoMakie.Colorbar(fig[2, k], hm; vertical = false)
    end
    return fig
end

# Backwards-friendly 4-arg form (original passed as separate components).
HD.plot_decomposition(result::HD.HelmholtzResult{2}, grid::HD.StructuredGrid{2}, u_orig::AbstractMatrix, v_orig::AbstractMatrix; kwargs...) =
    HD.plot_decomposition(result, grid; orig = cat(u_orig, v_orig; dims = 3), kwargs...)

end # module
