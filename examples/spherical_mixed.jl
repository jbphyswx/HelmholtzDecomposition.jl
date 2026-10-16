"""
Spherical mixed flow (Kelvin–Ekman).

A spherical flow with both rotational and divergent parts, on a lat–lon grid, solved by
multigrid-preconditioned conjugate gradients on the grid's own Laplacian. Loading `NUFSHT` makes its
transform the preconditioner instead.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using Statistics: Statistics

include(joinpath(@__DIR__, "..", "reference_flows", "reference_flows.jl"))

speed(U) = sqrt.(U[:, :, 1] .^ 2 .+ U[:, :, 2] .^ 2)

const NLON, NLAT = 64, 32
λ = range(0, 2π - 2π / NLON; length = NLON)
φ = range(-π / 2 + π / (2NLAT), π / 2 - π / (2NLAT); length = NLAT)
grid = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), λ, φ;
                               topology = (true, false), period = (2π, nothing))

u, v, = ReferenceFlows.kelvin_ekman_flow(grid)
U = cat(u, v; dims = 3)
result = HD.helmholtz_decompose(U, grid)

println("=== Spherical mixed flow (Kelvin–Ekman) ===")
println("Mean |u_rot|:          $(round(Statistics.mean(speed(result.u_rot)), sigdigits = 4))")
println("Mean |u_div|:          $(round(Statistics.mean(speed(result.u_div)), sigdigits = 4))")
println("Harmonic fraction:     $(round(result.harmonic_fraction, sigdigits = 4))")
println("χ solve:               $(result.χ_solve.iterations) iterations, converged = $(result.χ_solve.converged)")
println("Residual fraction:     $(round(result.residual_fraction, sigdigits = 4))")
