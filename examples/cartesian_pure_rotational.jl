"""
Cartesian Pure Rotational Field (Taylor-Green Vortex)

Demonstrates that a purely rotational field is correctly identified:
the divergent component should be ≈ 0.
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

# Generate Taylor-Green vortex (purely rotational)
u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact =
    HelmholtzDecomposition.taylor_green_vortex(grid)

# Decompose
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

# Verify
div_mag = Statistics.mean(sqrt.(result.u_div.^2 .+ result.v_div.^2))
rot_mag = Statistics.mean(sqrt.(result.u_rot.^2 .+ result.v_rot.^2))
recon_err = maximum(abs.(result.u_rot .+ result.u_div .- u))

println("=== Cartesian Pure Rotational (Taylor-Green) ===")
println("Mean |u_rot|: $(round(rot_mag, sigdigits=4))")
println("Mean |u_div|: $(round(div_mag, sigdigits=4))  (should be ≈ 0)")
println("Div/Rot ratio: $(round(div_mag/rot_mag, sigdigits=4))")
println("Reconstruction error: $(round(recon_err, sigdigits=4))")
println("ψ solve converged: $(result.ψ_solve.converged) ($(result.ψ_solve.iterations) iters)")
println("χ solve converged: $(result.χ_solve.converged) ($(result.χ_solve.iterations) iters)")
