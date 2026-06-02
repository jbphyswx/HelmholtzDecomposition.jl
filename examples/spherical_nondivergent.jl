"""
Spherical Non-divergent Field (Rossby Wave)

Demonstrates Helmholtz decomposition on a spherical grid with a purely
rotational field. The divergent component should be ≈ 0.

Note: Without a spectral spherical extension (FastSphericalHarmonics or NUFSHT),
this will fall back to SOR which may be slow.
"""

using HelmholtzDecomposition: HelmholtzDecomposition
using Statistics: Statistics

# Setup spherical grid (unit sphere for simplicity)
geom = HelmholtzDecomposition.SphericalGeometry(1.0)
Nlon = 72
Nlat = 36
lons = collect(range(0.0, 2π - 2π/Nlon, length=Nlon))
lats = collect(range(-π/3, π/3, length=Nlat))
grid = HelmholtzDecomposition.StructuredGrid(geom, lons, lats)

# Generate Rossby wave (purely rotational)
u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact =
    HelmholtzDecomposition.rossby_wave(grid)

# Decompose (will use SOR fallback without spherical spectral extension)
solver = HelmholtzDecomposition.SORSolver(; max_iter=20_000, tol=1e-6)
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid; solver=solver)

# Verify
div_mag = Statistics.mean(sqrt.(result.u_div.^2 .+ result.v_div.^2))
rot_mag = Statistics.mean(sqrt.(result.u_rot.^2 .+ result.v_rot.^2))

println("=== Spherical Non-divergent (Rossby Wave) ===")
println("Mean |u_rot|: $(round(rot_mag, sigdigits=4))")
println("Mean |u_div|: $(round(div_mag, sigdigits=4))  (should be ≈ 0)")
println("Div/Rot ratio: $(round(div_mag/rot_mag, sigdigits=4))")
println("ψ solve: $(result.ψ_solve.converged), $(result.ψ_solve.iterations) iters, res=$(round(result.ψ_solve.residual, sigdigits=3))")
println("χ solve: $(result.χ_solve.converged), $(result.χ_solve.iterations) iters, res=$(round(result.χ_solve.residual, sigdigits=3))")
