"""
Cartesian Mixed Rotational + Divergent Field

Demonstrates decomposition of a field with both rotational and divergent components.
Verifies that reconstruction u_rot + u_div ≈ u.
"""

using HelmholtzDecomposition: HelmholtzDecomposition
using FFTW: FFTW
using Statistics: Statistics

# Setup grid
N = 64
L = 1.0
dx = L / N
geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
xs = collect(range(0.0, L - dx, length=N))
ys = collect(range(0.0, L - dx, length=N))
grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

# Generate mixed field
u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact =
    HelmholtzDecomposition.rankine_vortex_with_source(grid)

# Decompose
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

# Verify
rot_mag = Statistics.mean(sqrt.(result.u_rot.^2 .+ result.v_rot.^2))
div_mag = Statistics.mean(sqrt.(result.u_div.^2 .+ result.v_div.^2))
recon_err_u = maximum(abs.(result.u_rot .+ result.u_div .- u))
recon_err_v = maximum(abs.(result.v_rot .+ result.v_div .- v))

println("=== Cartesian Mixed Field ===")
println("Mean |u_rot|: $(round(rot_mag, sigdigits=4))")
println("Mean |u_div|: $(round(div_mag, sigdigits=4))")
println("Reconstruction error (u): $(round(recon_err_u, sigdigits=4))")
println("Reconstruction error (v): $(round(recon_err_v, sigdigits=4))")
println("ψ solve: $(result.ψ_solve.converged), $(result.ψ_solve.iterations) iters")
println("χ solve: $(result.χ_solve.converged), $(result.χ_solve.iterations) iters")
