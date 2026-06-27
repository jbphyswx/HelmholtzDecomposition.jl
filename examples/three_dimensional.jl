"""
3-D Helmholtz Decomposition (ABC / Beltrami flow)

The Arnold–Beltrami–Childress flow is fully solenoidal (∇·u = 0), so its divergent
component vanishes. Demonstrates the dimension-generic spectral path in 3-D via FFTW.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FFTW: FFTW

relnorm(x) = sqrt(sum(abs2, x))

n = 32
L = 2π
h = L / n
ax = collect(range(0, L - h, length = n))
grid = HD.StructuredGrid(HD.CartesianGeometry(h, h, h), ax, ax, ax)

A, B, C = 1.0, 1.0, 1.0
U = zeros(n, n, n, 3)
for k in 1:n, j in 1:n, i in 1:n
    x, y, z = ax[i], ax[j], ax[k]
    U[i, j, k, 1] = A * sin(z) + C * cos(y)
    U[i, j, k, 2] = B * sin(x) + A * cos(z)
    U[i, j, k, 3] = C * sin(y) + B * cos(x)
end

result = HD.helmholtz_decompose_spectral(U, grid)
A1, A2, A3 = HD.vector_potential(result)   # 3-D vector potential (Hodge dual of R)

println("=== 3-D ABC flow, FFTW spectral ===")
println("|u_div| / |u|:        $(round(relnorm(result.u_div) / relnorm(U), sigdigits = 4))  (≈ 0, solenoidal)")
println("|u_rot| / |u|:        $(round(relnorm(result.u_rot) / relnorm(U), sigdigits = 4))  (≈ 1)")
println("Harmonic fraction:    $(round(result.harmonic_fraction, sigdigits = 4))")
println("Vector potential size: $(size(A1))")
