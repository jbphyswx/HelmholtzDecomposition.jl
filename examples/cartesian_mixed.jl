"""
Cartesian mixed rotational + divergent field.

A Taylor–Green vortex plus a source/sink on a doubly periodic box, verifying that
`u_rot + u_div + u_harm == u` and that each part is recovered.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FFTW: FFTW
using Statistics: Statistics

include(joinpath(@__DIR__, "..", "reference_flows", "reference_flows.jl"))

speed(U) = sqrt.(U[:, :, 1] .^ 2 .+ U[:, :, 2] .^ 2)

const N = 64
const L = 1.0

# Ranges, not `collect`: FlowGeometries proves uniformity from the axis TYPE, and a `Vector` is not
# provably uniform, so collecting one here would cost the direct transform.
axis = range(0.0, L - L / N; length = N)
grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), axis, axis;
                               topology = (true, true), period = (L, L))

u, v, = ReferenceFlows.rankine_vortex_with_source(grid)
U = cat(u, v; dims = 3)

result = HD.helmholtz_decompose(U, grid)

recon_err = maximum(abs.(result.u_rot .+ result.u_div .+ result.u_harm .- U))

println("=== Cartesian mixed field ===")
println("solver:                $(nameof(typeof(HD.plan_helmholtz(grid).solver)))")
println("Mean |u_rot|:          $(round(Statistics.mean(speed(result.u_rot)), sigdigits = 4))")
println("Mean |u_div|:          $(round(Statistics.mean(speed(result.u_div)), sigdigits = 4))")
println("Harmonic fraction:     $(round(result.harmonic_fraction, sigdigits = 4))")
println("Reconstruction error:  $(round(recon_err, sigdigits = 4))")
