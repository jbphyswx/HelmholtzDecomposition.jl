"""
Cartesian Pure Rotational Field (Taylor-Green Vortex)

A purely rotational field should decompose with a near-zero divergent component.
Uses the FFTW spectral path (periodic Cartesian grid).
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

u, v, = HD.taylor_green_vortex(grid)
U = cat(u, v; dims = 3)

# Spectral decomposition → physical HelmholtzResult.
result = HD.helmholtz_decompose_spectral(u, v, grid)

rot_mag = Statistics.mean(speed(result.u_rot))
div_mag = Statistics.mean(speed(result.u_div))
recon_err = maximum(abs.(result.u_rot .+ result.u_div .+ result.u_harm .- U))

println("=== Cartesian Pure Rotational (Taylor-Green), FFTW spectral ===")
println("Mean |u_rot|:          $(round(rot_mag, sigdigits = 4))")
println("Mean |u_div|:          $(round(div_mag, sigdigits = 4))  (should be ≈ 0)")
println("Div/Rot ratio:         $(round(div_mag / rot_mag, sigdigits = 4))")
println("Reconstruction error:  $(round(recon_err, sigdigits = 4))")
println("Harmonic fraction:     $(round(result.harmonic_fraction, sigdigits = 4))")
