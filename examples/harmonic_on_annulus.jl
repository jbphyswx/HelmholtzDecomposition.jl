"""
Harmonic component on a multiply-connected domain (annulus).

On a domain with a hole, a pure circulation around the hole is *harmonic* — both divergence-free
and curl-free — and no single-valued streamfunction or velocity potential represents it; it lands
in `u_harm`, and `count_holes` reports the topology.

A vortex centred in a square box also flows through the outer edge. The natural decomposition
gives that normal flux to the divergent part (`∂χ/∂n = u·n̂`), so `u_div` is a harmonic gradient
that does not vanish as the grid is refined.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG

include(joinpath(@__DIR__, "..", "reference_flows", "reference_flows.jl"))

speed(U) = sqrt.(U[:, :, 1] .^ 2 .+ U[:, :, 2] .^ 2)

println("=== Harmonic circulation on an annulus ===")
for N in (41, 81)
    xs = range(-1.0, 1.0; length = N)
    cart = FG.Geometry.CartesianGeometry{Float64}()
    base = FG.Grids.StructuredGrid(cart, xs, xs)
    mask = ReferenceFlows.disk_mask(base; center = (0.0, 0.0), radius = 0.3)
    grid = FG.Grids.StructuredGrid(cart, xs, xs; mask = mask)

    u, v = ReferenceFlows.harmonic_vortex(grid; Γ = 1.0)
    U = cat(u, v; dims = 3)
    result = HD.helmholtz_decompose(U, grid)

    println("N=$N  holes = $(HD.count_holes(grid))  " *
            "harmonic fraction = $(round(result.harmonic_fraction, sigdigits = 4))  " *
            "|u_rot|/|u| = $(round(maximum(speed(result.u_rot)) / maximum(speed(U)), sigdigits = 3))  " *
            "|u_div|/|u| = $(round(maximum(speed(result.u_div)) / maximum(speed(U)), sigdigits = 3))")
end
