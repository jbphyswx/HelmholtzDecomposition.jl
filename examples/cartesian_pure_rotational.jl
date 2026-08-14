"""
Cartesian pure rotational field (Taylor–Green vortex).

A purely rotational field decomposes with a near-zero divergent component. The residue that
appears in `u_harm` is the two collocated↔staggered averaging hops, and it converges at second
order — it is discretisation, not topology.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FFTW: FFTW

include(joinpath(@__DIR__, "..", "reference_flows", "reference_flows.jl"))

speed(U) = sqrt.(U[:, :, 1] .^ 2 .+ U[:, :, 2] .^ 2)

const L = 1.0

println("=== Cartesian pure rotational (Taylor–Green) ===")
for N in (32, 64, 128)
    axis = range(0.0, L - L / N; length = N)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), axis, axis;
                                   topology = (true, true), period = (L, L))
    u, v, = ReferenceFlows.taylor_green_vortex(grid)
    U = cat(u, v; dims = 3)
    result = HD.helmholtz_decompose(U, grid)
    ratio = maximum(speed(result.u_div)) / maximum(speed(U))
    println("N=$N  |u_div|/|u| = $(round(ratio, sigdigits = 3))   " *
            "harmonic fraction = $(round(result.harmonic_fraction, sigdigits = 3))")
end
