"""
Cartesian Mixed Rotational + Divergent Field

Decomposition of a field with both rotational and divergent components, verifying that
`u_rot + u_div + u_harm ≈ u`. Uses the FFTW spectral path.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FFTW: FFTW
using Statistics: Statistics

speed(U) = sqrt.(U[:, :, 1] .^ 2 .+ U[:, :, 2] .^ 2)

N = 64
L = 1.0
dx = L / N
grid = HD.StructuredGrid(HD.CartesianGeometry(dx, dx),
    collect(range(0.0, L - dx, length = N)), collect(range(0.0, L - dx, length = N)))

u, v, = HD.rankine_vortex_with_source(grid)
U = cat(u, v; dims = 3)

result = HD.helmholtz_decompose_spectral(u, v, grid)

rot_mag = Statistics.mean(speed(result.u_rot))
div_mag = Statistics.mean(speed(result.u_div))
recon_err = maximum(abs.(result.u_rot .+ result.u_div .+ result.u_harm .- U))

println("=== Cartesian Mixed Field, FFTW spectral ===")
println("Mean |u_rot|:          $(round(rot_mag, sigdigits = 4))")
println("Mean |u_div|:          $(round(div_mag, sigdigits = 4))")
println("Reconstruction error:  $(round(recon_err, sigdigits = 4))")
println("Both components non-negligible: $(rot_mag > 0.01 && div_mag > 0.01)")
