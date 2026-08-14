"""
Generate static figure assets for HelmholtzDecomposition.jl docs.

Run from this directory:
    julia --project=. generate_assets.jl

Outputs PNG files to ../src/assets/ which are checked into the repo and referenced from
README.md and docs/ markdown. A single, consistent colorscheme is used throughout:
signed fields use the diverging `:balance` map (symmetric range); magnitudes use the
sequential `:dense` map.
"""

using CairoMakie: CairoMakie
using FFTW: FFTW
using NUFSHT: NUFSHT
using FlowGeometries: FlowGeometries as FG
using HelmholtzDecomposition: HelmholtzDecomposition as HD

include(joinpath(@__DIR__, "..", "..", "reference_flows", "reference_flows.jl"))

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS_DIR)

const DIVERGING = :balance     # signed fields (velocity components, potentials)
const SEQUENTIAL = :dense      # magnitudes / speeds

comp(A, c) = A[:, :, c]
speed(A) = sqrt.(A[:, :, 1] .^ 2 .+ A[:, :, 2] .^ 2)

# Signed-field heatmap panel with a symmetric diverging colorrange.
function signed_panel!(fig, pos, data, xs, ys, title)
    ax = CairoMakie.Axis(fig[pos...]; title = title, xlabel = "x", ylabel = "y", aspect = CairoMakie.DataAspect())
    clim = maximum(abs.(filter(isfinite, data)))
    clim = clim > 0 ? clim : 1.0
    hm = CairoMakie.heatmap!(ax, xs, ys, data; colormap = DIVERGING, colorrange = (-clim, clim))
    CairoMakie.Colorbar(fig[pos[1], pos[2] + 1], hm; width = 12)
    return ax
end

# Magnitude heatmap panel (sequential), optional shared colorrange.
function mag_panel!(fig, pos, data, xs, ys, title; colorrange = nothing)
    ax = CairoMakie.Axis(fig[pos...]; title = title, xlabel = "x", ylabel = "y", aspect = CairoMakie.DataAspect())
    hm = colorrange === nothing ?
         CairoMakie.heatmap!(ax, xs, ys, data; colormap = SEQUENTIAL) :
         CairoMakie.heatmap!(ax, xs, ys, data; colormap = SEQUENTIAL, colorrange = colorrange)
    CairoMakie.Colorbar(fig[pos[1], pos[2] + 1], hm; width = 12)
    return ax
end

function cartgrid(N, L)
    axis = FG.Axes.UniformAxis(0.0, L / N, N)
    return FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), axis, axis;
                                   topology = (true, true), period = (L, L))
end

decompose(u, v, grid) = HD.helmholtz_decompose(cat(u, v; dims = 3), grid)

# ─── Classic decompositions (signed components + potentials) ──────────────

function decomposition_figure(title, u, v, result, grid, fname)
    xs, ys = FG.Grids.coordinates(grid)
    # `ψ` lives on the corner (dual) grid, so it has its own axes.
    ψ = HD.streamfunction(result)
    ψx, ψy = range(first(xs), last(xs); length = size(ψ, 1)),
             range(first(ys), last(ys); length = size(ψ, 2))
    fig = CairoMakie.Figure(; size = (1200, 820), fontsize = 14)
    CairoMakie.Label(fig[0, 1:6], title; fontsize = 18, font = :bold)
    signed_panel!(fig, (1, 1), u, xs, ys, "u (original)")
    signed_panel!(fig, (1, 3), comp(result.u_rot, 1), xs, ys, "u_rot (rotational)")
    signed_panel!(fig, (1, 5), comp(result.u_div, 1), xs, ys, "u_div (divergent)")
    signed_panel!(fig, (2, 1), v, xs, ys, "v (original)")
    signed_panel!(fig, (2, 3), comp(result.u_rot, 2), xs, ys, "v_rot (rotational)")
    signed_panel!(fig, (2, 5), comp(result.u_div, 2), xs, ys, "v_div (divergent)")
    signed_panel!(fig, (3, 1), result.divergence, xs, ys, "δ = ∇·u")
    signed_panel!(fig, (3, 3), ψ, ψx, ψy, "ψ (stream function, corners)")
    signed_panel!(fig, (3, 5), result.χ, xs, ys, "χ (velocity potential)")
    out = joinpath(ASSETS_DIR, fname)
    CairoMakie.save(out, fig; px_per_unit = 2)
    println("Saved: $out")
end

function figure_taylor_green()
    grid = cartgrid(64, 1.0)
    u, v, = ReferenceFlows.taylor_green_vortex(grid)
    decomposition_figure("Taylor–Green Vortex — Helmholtz Decomposition", u, v,
        decompose(u, v, grid), grid, "taylor_green_decomposition.png")
end

function figure_mixed_field()
    grid = cartgrid(64, 1.0)
    u, v, = ReferenceFlows.rankine_vortex_with_source(grid)
    decomposition_figure("Vortex + Source — Helmholtz Decomposition", u, v,
        decompose(u, v, grid), grid, "mixed_field_decomposition.png")
end

function figure_point_source()
    grid = cartgrid(64, 1.0)
    u, v, = ReferenceFlows.point_source_sink(grid)
    decomposition_figure("Point Source — Helmholtz Decomposition", u, v,
        decompose(u, v, grid), grid, "point_source_decomposition.png")
end

# ─── NEW: harmonic component on a multiply-connected domain (annulus) ─────

function figure_harmonic_annulus()
    n = 81
    h = 2.0 / (n - 1)
    xs = FG.Axes.UniformAxis(-1.0, h, n); ys = xs
    cart = FG.Geometry.CartesianGeometry{Float64}()
    base = FG.Grids.StructuredGrid(cart, xs, ys)
    mask = ReferenceFlows.disk_mask(base; center = (0.0, 0.0), radius = 0.3)
    grid = FG.Grids.StructuredGrid(cart, xs, ys; mask = mask)
    u, v = ReferenceFlows.harmonic_vortex(grid; Γ = 1.0)
    res = decompose(u, v, grid)

    blank(A) = (B = copy(A); B[.!mask] .= NaN; B)   # blank the masked hole
    U = cat(u, v; dims = 3)
    so, sr, sd, sh = blank(speed(U)), blank(speed(res.u_rot)), blank(speed(res.u_div)), blank(speed(res.u_harm))
    cr = (0.0, maximum(filter(isfinite, so)))

    fig = CairoMakie.Figure(; size = (1280, 360), fontsize = 14)
    CairoMakie.Label(fig[0, 1:8],
        "Harmonic circulation on an annulus  (b₁=$(HD.count_holes(grid)), harmonic_fraction=$(round(res.harmonic_fraction, digits=3)))";
        fontsize = 18, font = :bold)
    mag_panel!(fig, (1, 1), so, xs, ys, "|u| (original)"; colorrange = cr)
    mag_panel!(fig, (1, 3), sr, xs, ys, "|u_rot| ≈ 0"; colorrange = cr)
    mag_panel!(fig, (1, 5), sd, xs, ys, "|u_div| ≈ 0"; colorrange = cr)
    mag_panel!(fig, (1, 7), sh, xs, ys, "|u_harm| ≈ |u|"; colorrange = cr)
    out = joinpath(ASSETS_DIR, "harmonic_annulus_decomposition.png")
    CairoMakie.save(out, fig; px_per_unit = 2)
    println("Saved: $out")
end

# ─── NEW: 3-D decomposition (z mid-slice of an ABC/Beltrami flow) ──────────

function figure_3d_abc()
    n = 48; L = 2π; h = L / n
    ax = FG.Axes.UniformAxis(0.0, h, n)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), ax, ax, ax;
                                   topology = (true, true, true), period = (L, L, L))
    U = zeros(n, n, n, 3)
    for k in 1:n, j in 1:n, i in 1:n
        x, y, z = ax[i], ax[j], ax[k]
        U[i, j, k, 1] = sin(z) + cos(y)
        U[i, j, k, 2] = sin(x) + cos(z)
        U[i, j, k, 3] = sin(y) + cos(x)
    end
    res = HD.helmholtz_decompose(U, grid)
    kz = n ÷ 2
    fig = CairoMakie.Figure(; size = (1200, 360), fontsize = 14)
    CairoMakie.Label(fig[0, 1:6], "3-D ABC (Beltrami) flow — z mid-slice of uₓ  (solenoidal: u_div ≈ 0)";
        fontsize = 18, font = :bold)
    signed_panel!(fig, (1, 1), U[:, :, kz, 1], ax, ax, "uₓ (original)")
    signed_panel!(fig, (1, 3), res.u_rot[:, :, kz, 1], ax, ax, "u_rot,ₓ (rotational)")
    signed_panel!(fig, (1, 5), res.u_div[:, :, kz, 1], ax, ax, "u_div,ₓ ≈ 0")
    out = joinpath(ASSETS_DIR, "three_dimensional_decomposition.png")
    CairoMakie.save(out, fig; px_per_unit = 2)
    println("Saved: $out")
end

# ─── NEW: 3-D MIXED field (rotational ABC + divergent gradient) ───────────

function figure_3d_mixed()
    n = 48; L = 2π; h = L / n
    ax = FG.Axes.UniformAxis(0.0, h, n)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), ax, ax, ax;
                                   topology = (true, true, true), period = (L, L, L))
    U = zeros(n, n, n, 3)
    for k in 1:n, j in 1:n, i in 1:n
        x, y, z = ax[i], ax[j], ax[k]
        # ABC (solenoidal) + ∇φ with φ = cos(x)cos(y)cos(z) (curl-free): a genuinely mixed field.
        U[i, j, k, 1] = (sin(z) + cos(y)) - sin(x) * cos(y) * cos(z)
        U[i, j, k, 2] = (sin(x) + cos(z)) - cos(x) * sin(y) * cos(z)
        U[i, j, k, 3] = (sin(y) + cos(x)) - cos(x) * cos(y) * sin(z)
    end
    res = HD.helmholtz_decompose(U, grid)
    kz = n ÷ 2
    fig = CairoMakie.Figure(; size = (1200, 360), fontsize = 14)
    CairoMakie.Label(fig[0, 1:6], "3-D mixed field (ABC + gradient) — z mid-slice of uₓ  (both parts nonzero)";
        fontsize = 18, font = :bold)
    signed_panel!(fig, (1, 1), U[:, :, kz, 1], ax, ax, "uₓ (original)")
    signed_panel!(fig, (1, 3), res.u_rot[:, :, kz, 1], ax, ax, "u_rot,ₓ (rotational)")
    signed_panel!(fig, (1, 5), res.u_div[:, :, kz, 1], ax, ax, "u_div,ₓ (divergent)")
    out = joinpath(ASSETS_DIR, "three_dimensional_mixed_decomposition.png")
    CairoMakie.save(out, fig; px_per_unit = 2)
    println("Saved: $out")
end

# ─── NEW: spherical decomposition (mixed Kelvin–Ekman flow, NUFSHT solver) ─

function figure_spherical()
    Nlon, Nlat = 96, 48
    # A COVERING node set. The previous ±1.3 rad band is a regional patch, and a spherical-harmonic
    # transform does not determine the solution on one — analysis integrates over all of S².
    lons = FG.Axes.UniformAxis(0.0, 2π / Nlon, Nlon)
    lats = FG.Axes.UniformAxis(-π / 2 + π / (2Nlat), π / Nlat, Nlat)
    grid = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), lons, lats;
                                   topology = (true, false), period = (2π, nothing))
    u, v, = ReferenceFlows.kelvin_ekman_flow(grid)
    NUExt = Base.get_extension(HD, :HelmholtzDecompositionNUFSHTExt)
    nusht = NUExt.SphericalNUSHTSolver(; lmax = Nlat - 1, tol = 1e-8)
    res = HD.helmholtz_decompose(cat(u, v; dims = 3), grid; solver = nusht)

    fig = CairoMakie.Figure(; size = (1200, 760), fontsize = 14)
    CairoMakie.Label(fig[0, 1:6], "Spherical mixed flow (Kelvin–Ekman) — NUFSHT decomposition"; fontsize = 18, font = :bold)
    signed_panel!(fig, (1, 1), u, lons, lats, "u (zonal, original)")
    signed_panel!(fig, (1, 3), comp(res.u_rot, 1), lons, lats, "u_rot (rotational)")
    signed_panel!(fig, (1, 5), comp(res.u_div, 1), lons, lats, "u_div (divergent)")
    signed_panel!(fig, (2, 1), v, lons, lats, "v (meridional, original)")
    signed_panel!(fig, (2, 3), comp(res.u_rot, 2), lons, lats, "v_rot (rotational)")
    signed_panel!(fig, (2, 5), comp(res.u_div, 2), lons, lats, "v_div (divergent)")
    out = joinpath(ASSETS_DIR, "spherical_decomposition.png")
    CairoMakie.save(out, fig; px_per_unit = 2)
    println("Saved: $out")
end

println("Generating documentation assets...\n")
figure_taylor_green()
figure_mixed_field()
figure_point_source()
figure_harmonic_annulus()
figure_3d_abc()
figure_3d_mixed()
figure_spherical()
println("\nDone! Assets saved to: $ASSETS_DIR")
