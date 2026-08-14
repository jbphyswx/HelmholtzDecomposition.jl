"""
3-D mixed decomposition (ABC + gradient).

A field with both a solenoidal (ABC) and a divergent (gradient) part. Both come out nonzero and
`u_rot + u_div + u_harm` reconstructs the field.
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
    # ABC (solenoidal) plus ∇(sin x sin y sin z) (irrotational).
    U[i, j, k, 1] = A * sin(z) + C * cos(y) + cos(x) * sin(y) * sin(z)
    U[i, j, k, 2] = B * sin(x) + A * cos(z) + sin(x) * cos(y) * sin(z)
    U[i, j, k, 3] = C * sin(y) + B * cos(x) + sin(x) * sin(y) * cos(z)
end

result = HD.helmholtz_decompose(U, grid)
mag(V) = sqrt.(sum(abs2, V; dims = 4))

println("=== 3-D mixed (ABC + gradient) ===")
println("Max |u_rot|:           $(round(maximum(mag(result.u_rot)), sigdigits = 4))")
println("Max |u_div|:           $(round(maximum(mag(result.u_div)), sigdigits = 4))")
println("Harmonic fraction:     $(round(result.harmonic_fraction, sigdigits = 4))")
println("Reconstruction error:  $(round(maximum(abs.(result.u_rot .+ result.u_div .+ result.u_harm .- U)), sigdigits = 4))")
