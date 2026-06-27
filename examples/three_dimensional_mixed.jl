"""
3-D Mixed Helmholtz Decomposition (ABC + gradient)

A field with both a rotational (solenoidal ABC) and a divergent (gradient) part. Both
components come out nonzero, and `u_rot + u_div + u_harm` reconstructs the field.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FFTW: FFTW

relnorm(x) = sqrt(sum(abs2, x))

n = 32
L = 2π
h = L / n
ax = collect(range(0, L - h, length = n))
grid = HD.StructuredGrid(HD.CartesianGeometry(h, h, h), ax, ax, ax)

U = zeros(n, n, n, 3)
for k in 1:n, j in 1:n, i in 1:n
    x, y, z = ax[i], ax[j], ax[k]
    U[i, j, k, 1] = (sin(z) + cos(y)) - sin(x) * cos(y) * cos(z)   # ABC + ∂φ/∂x
    U[i, j, k, 2] = (sin(x) + cos(z)) - cos(x) * sin(y) * cos(z)
    U[i, j, k, 3] = (sin(y) + cos(x)) - cos(x) * cos(y) * sin(z)
end

result = HD.helmholtz_decompose_spectral(U, grid)

println("=== 3-D mixed (ABC + gradient), FFTW spectral ===")
println("|u_rot| / |u|:        $(round(relnorm(result.u_rot) / relnorm(U), sigdigits = 4))")
println("|u_div| / |u|:        $(round(relnorm(result.u_div) / relnorm(U), sigdigits = 4))")
println("Reconstruction error: $(round(maximum(abs.(result.u_rot .+ result.u_div .+ result.u_harm .- U)), sigdigits = 4))")
println("Both components non-negligible: $(relnorm(result.u_rot) > 1 && relnorm(result.u_div) > 1)")
