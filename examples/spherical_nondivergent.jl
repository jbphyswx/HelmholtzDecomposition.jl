"""
Spherical Non-divergent Field (Rossby Wave)

Helmholtz decomposition on a spherical grid with a purely rotational field; the divergent
component should be ≈ 0. Uses the SOR solver (load `FastSphericalHarmonics` or `NUFSHT`
for an O(N log N) spectral solve).
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using Statistics: Statistics

speed(U) = sqrt.(U[:, :, 1] .^ 2 .+ U[:, :, 2] .^ 2)

Nlon = 72
Nlat = 36
grid = HD.StructuredGrid(HD.SphericalGeometry(1.0),
    collect(range(0.0, 2π - 2π / Nlon, length = Nlon)), collect(range(-π / 3, π / 3, length = Nlat)))

u, v, = HD.rossby_wave(grid)
U = cat(u, v; dims = 3)

solver = HD.SORSolver(; max_iter = 20_000, tol = 1e-6, boundary = :dirichlet)
result = HD.helmholtz_decompose(u, v, grid; solver = solver, boundary_χ = :neumann, boundary_ψ = :dirichlet)

rot_mag = Statistics.mean(speed(result.u_rot))
div_mag = Statistics.mean(speed(result.u_div))

println("=== Spherical Non-divergent (Rossby Wave), SOR ===")
println("Mean |u_rot|:        $(round(rot_mag, sigdigits = 4))")
println("Mean |u_div|:        $(round(div_mag, sigdigits = 4))  (should be ≈ 0)")
println("Div/Rot ratio:       $(round(div_mag / rot_mag, sigdigits = 4))")
println("Harmonic fraction:   $(round(result.harmonic_fraction, sigdigits = 4))")
println("χ solve: converged=$(result.χ_solve.converged), $(result.χ_solve.iterations) iters")
println("ψ solve: converged=$(result.rot_solve[1].converged), $(result.rot_solve[1].iterations) iters")
