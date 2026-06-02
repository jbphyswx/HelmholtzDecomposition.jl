"""
Generate static figure assets for HelmholtzDecomposition.jl docs.

Run from this directory:
    julia --project=. generate_assets.jl

Outputs PNG files to ../assets/ which are checked into the repo
and referenced from README.md and docs/ markdown.
"""

using CairoMakie: CairoMakie
using FFTW: FFTW
using HelmholtzDecomposition: HelmholtzDecomposition
using Statistics: Statistics

const ASSETS_DIR = joinpath(@__DIR__, "..", "assets")
mkpath(ASSETS_DIR)

# ─── Helper: heatmap panel ────────────────────────────────────────────────

function add_heatmap_panel!(fig, pos, data, xs, ys, title; colormap=:RdBu, symmetric=true)
    ax = CairoMakie.Axis(fig[pos...]; title=title, xlabel="x", ylabel="y",
                          aspect=CairoMakie.DataAspect())
    clim = symmetric ? maximum(abs.(data)) : nothing
    if symmetric && clim > 0
        hm = CairoMakie.heatmap!(ax, xs, ys, data; colormap=colormap,
                                  colorrange=(-clim, clim))
    else
        hm = CairoMakie.heatmap!(ax, xs, ys, data; colormap=colormap)
    end
    CairoMakie.Colorbar(fig[pos[1], pos[2]+1], hm; width=15)
    return ax
end

# ─── Figure 1: Taylor-Green vortex decomposition ─────────────────────────

function figure_taylor_green()
    N = 64; L = 1.0; dx = L / N
    geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
    xs = collect(range(0.0, L - dx, length=N))
    ys = collect(range(0.0, L - dx, length=N))
    grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

    u, v, _, _, _, _ = HelmholtzDecomposition.taylor_green_vortex(grid)
    result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

    fig = CairoMakie.Figure(; size=(1200, 800), fontsize=14)
    CairoMakie.Label(fig[0, 1:6], "Taylor-Green Vortex — Helmholtz Decomposition";
                      fontsize=18, font=:bold)

    # Row 1: u component
    add_heatmap_panel!(fig, (1, 1), u, xs, ys, "u (original)")
    add_heatmap_panel!(fig, (1, 3), result.u_rot, xs, ys, "u_rot (rotational)")
    add_heatmap_panel!(fig, (1, 5), result.u_div, xs, ys, "u_div (divergent)")

    # Row 2: v component
    add_heatmap_panel!(fig, (2, 1), v, xs, ys, "v (original)")
    add_heatmap_panel!(fig, (2, 3), result.v_rot, xs, ys, "v_rot (rotational)")
    add_heatmap_panel!(fig, (2, 5), result.v_div, xs, ys, "v_div (divergent)")

    # Row 3: potentials
    add_heatmap_panel!(fig, (3, 1), result.vorticity, xs, ys, "ζ (vorticity)")
    add_heatmap_panel!(fig, (3, 3), result.ψ, xs, ys, "ψ (stream function)")
    add_heatmap_panel!(fig, (3, 5), result.χ, xs, ys, "χ (velocity potential)")

    outpath = joinpath(ASSETS_DIR, "taylor_green_decomposition.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
    return fig
end

# ─── Figure 2: Mixed field (vortex + source) decomposition ───────────────

function figure_mixed_field()
    N = 64; L = 1.0; dx = L / N
    geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
    xs = collect(range(0.0, L - dx, length=N))
    ys = collect(range(0.0, L - dx, length=N))
    grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

    u, v, _, _, _, _ = HelmholtzDecomposition.rankine_vortex_with_source(grid)
    result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

    fig = CairoMakie.Figure(; size=(1200, 800), fontsize=14)
    CairoMakie.Label(fig[0, 1:6], "Vortex + Source — Helmholtz Decomposition";
                      fontsize=18, font=:bold)

    add_heatmap_panel!(fig, (1, 1), u, xs, ys, "u (original)")
    add_heatmap_panel!(fig, (1, 3), result.u_rot, xs, ys, "u_rot (rotational)")
    add_heatmap_panel!(fig, (1, 5), result.u_div, xs, ys, "u_div (divergent)")

    add_heatmap_panel!(fig, (2, 1), v, xs, ys, "v (original)")
    add_heatmap_panel!(fig, (2, 3), result.v_rot, xs, ys, "v_rot (rotational)")
    add_heatmap_panel!(fig, (2, 5), result.v_div, xs, ys, "v_div (divergent)")

    add_heatmap_panel!(fig, (3, 1), result.vorticity, xs, ys, "ζ (vorticity)")
    add_heatmap_panel!(fig, (3, 3), result.ψ, xs, ys, "ψ (stream function)")
    add_heatmap_panel!(fig, (3, 5), result.divergence, xs, ys, "δ (divergence)")

    outpath = joinpath(ASSETS_DIR, "mixed_field_decomposition.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
    return fig
end

# ─── Figure 3: Point source (purely divergent) ───────────────────────────

function figure_point_source()
    N = 64; L = 1.0; dx = L / N
    geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
    xs = collect(range(0.0, L - dx, length=N))
    ys = collect(range(0.0, L - dx, length=N))
    grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

    u, v, _, _, _, _ = HelmholtzDecomposition.point_source_sink(grid)
    result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

    fig = CairoMakie.Figure(; size=(1200, 350), fontsize=14)
    CairoMakie.Label(fig[0, 1:6], "Point Source — Helmholtz Decomposition";
                      fontsize=18, font=:bold)

    speed_orig = sqrt.(u.^2 .+ v.^2)
    speed_rot = sqrt.(result.u_rot.^2 .+ result.v_rot.^2)
    speed_div = sqrt.(result.u_div.^2 .+ result.v_div.^2)

    add_heatmap_panel!(fig, (1, 1), speed_orig, xs, ys, "|u| (original)";
                        colormap=:viridis, symmetric=false)
    add_heatmap_panel!(fig, (1, 3), speed_rot, xs, ys, "|u_rot| (should be ≈ 0)";
                        colormap=:viridis, symmetric=false)
    add_heatmap_panel!(fig, (1, 5), speed_div, xs, ys, "|u_div| (divergent)";
                        colormap=:viridis, symmetric=false)

    outpath = joinpath(ASSETS_DIR, "point_source_decomposition.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
    return fig
end

# ─── Generate all ─────────────────────────────────────────────────────────

println("Generating documentation assets...")
println()
figure_taylor_green()
figure_mixed_field()
figure_point_source()
println()
println("Done! Assets saved to: $ASSETS_DIR")
