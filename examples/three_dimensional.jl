"""
3-D Helmholtz decomposition (ABC / Beltrami flow).

The Arnold–Beltrami–Childress flow is fully solenoidal, so its divergent component vanishes. In
3-D the rotation potential has three components — the Hodge dual of the vector potential.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FFTW: FFTW

const N = 32
const L = 2π
axis = range(0.0, L - L / N; length = N)
grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), axis, axis, axis;
                               topology = (true, true, true), period = (L, L, L))

A, B, C = 1.0, 0.7, 0.5
U = zeros(N, N, N, 3)
for k in 1:N, j in 1:N, i in 1:N
    x, y, z = axis[i], axis[j], axis[k]
    U[i, j, k, 1] = A * sin(z) + C * cos(y)
    U[i, j, k, 2] = B * sin(x) + A * cos(z)
    U[i, j, k, 3] = C * sin(y) + B * cos(x)
end

result = HD.helmholtz_decompose(U, grid)
mag(V) = sqrt.(sum(abs2, V; dims = 4))

println("=== 3-D ABC flow ===")
println("|u_div| / |u|:         $(round(maximum(mag(result.u_div)) / maximum(mag(U)), sigdigits = 3))")
println("Harmonic fraction:     $(round(result.harmonic_fraction, sigdigits = 4))")
println("Reconstruction error:  $(round(maximum(abs.(result.u_rot .+ result.u_div .+ result.u_harm .- U)), sigdigits = 4))")
println("Rotation components:   $(length(result.rotation_potential))")
