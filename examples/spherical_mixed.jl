"""
Spherical Mixed Flow (Kelvin–Ekman)

A spherical flow with both rotational and divergent parts. Uses the SOR solver (load
`FastSphericalHarmonics` or `NUFSHT` for an O(N log N) spectral solve — see the
spherical figure in the docs, generated with NUFSHT).
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using Statistics: Statistics

speed(U) = sqrt.(U[:, :, 1] .^ 2 .+ U[:, :, 2] .^ 2)

Nlon, Nlat = 96, 48
grid = HD.StructuredGrid(HD.SphericalGeometry(1.0),
    collect(range(0.0, 2π - 2π / Nlon, length = Nlon)), collect(range(-1.3, 1.3, length = Nlat)))

u, v, = HD.kelvin_ekman_flow(grid)
U = cat(u, v; dims = 3)

solver = HD.SORSolver(; max_iter = 30_000, tol = 1e-8, boundary = HD.Dirichlet())
result = HD.helmholtz_decompose(u, v, grid; solver = solver, boundary_χ = HD.Neumann(), boundary_ψ = HD.Dirichlet())

println("=== Spherical Mixed (Kelvin–Ekman), SOR ===")
println("Mean |u_rot|:          $(round(Statistics.mean(speed(result.u_rot)), sigdigits = 4))")
println("Mean |u_div|:          $(round(Statistics.mean(speed(result.u_div)), sigdigits = 4))")
println("Reconstruction error:  $(round(maximum(abs.(result.u_rot .+ result.u_div .+ result.u_harm .- U)), sigdigits = 4))")
println("Harmonic fraction:     $(round(result.harmonic_fraction, sigdigits = 4))")
