"""
Spherical non-divergent field (Rossby wave).

A purely rotational flow on a lat–lon sphere: the divergent component should vanish. The curved
metric is what makes this a real test — the face areas carry the `cos φ` that closes the surface,
and the operators are built so `curl ∘ grad = 0` holds on it exactly.

`FastSphericalHarmonics` or `NUFSHT` give an `O(N log N)` spectral solve on a covering node set;
this lat–lon grid is not one of theirs, so it takes the multigrid-preconditioned iterative path.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG

include(joinpath(@__DIR__, "..", "reference_flows", "reference_flows.jl"))

speed(U) = sqrt.(U[:, :, 1] .^ 2 .+ U[:, :, 2] .^ 2)

const NLON, NLAT = 64, 32
λ = range(0, 2π - 2π / NLON; length = NLON)
φ = range(-π / 2 + π / (2NLAT), π / 2 - π / (2NLAT); length = NLAT)
grid = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), λ, φ;
                               topology = (true, false), period = (2π, nothing))

u, v, = ReferenceFlows.rossby_wave(grid)
U = cat(u, v; dims = 3)
result = HD.helmholtz_decompose(U, grid)

println("=== Spherical non-divergent (Rossby wave) ===")
println("solver:                $(nameof(typeof(HD.plan_helmholtz(grid).solver)))")
println("|u_div| / |u|:         $(round(maximum(speed(result.u_div)) / maximum(speed(U)), sigdigits = 3))")
println("Harmonic fraction:     $(round(result.harmonic_fraction, sigdigits = 4))")
println("χ solve:               $(result.χ_solve.iterations) iterations, converged = $(result.χ_solve.converged)")
println("Reconstruction error:  $(round(maximum(abs.(result.u_rot .+ result.u_div .+ result.u_harm .- U)), sigdigits = 4))")
