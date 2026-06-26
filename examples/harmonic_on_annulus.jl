"""
Harmonic Component on a Multiply-Connected Domain (annulus)

On a domain with a hole, a pure circulation around the hole is *harmonic* — both
divergence-free and curl-free — and cannot be represented by a single-valued streamfunction
or velocity potential. The decomposition places (essentially) all of it in `u_harm`, and
`count_holes` detects the topology. This is the generalized Helmholtz-Hodge case (issue #1).
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD

relnorm(x) = sqrt(sum(abs2, x))

n = 61
xs = collect(range(-1.0, 1.0, length = n))
h = xs[2] - xs[1]
base = HD.StructuredGrid(HD.CartesianGeometry(h, h), xs, xs)
mask = HD.disk_mask(base; center = (0.0, 0.0), radius = 0.3)   # mask out the center → annulus
grid = HD.StructuredGrid(HD.CartesianGeometry(h, h), xs, xs; mask = mask)

u, v = HD.harmonic_vortex(grid; Γ = 1.0)   # pure circulation about the hole
U = cat(u, v; dims = 3)

solver = HD.SORSolver(; max_iter = 20_000, tol = 1e-9, boundary = :dirichlet)
result = HD.helmholtz_decompose(u, v, grid; solver = solver, boundary_χ = :dirichlet, boundary_ψ = :dirichlet)

println("=== Harmonic circulation on an annulus ===")
println("Holes (b₁ estimate):  $(HD.count_holes(grid))")
println("Harmonic fraction:    $(round(result.harmonic_fraction, sigdigits = 4))  (≈ 1: the field is harmonic)")
println("|u_rot| / |u|:        $(round(relnorm(result.u_rot) / relnorm(U), sigdigits = 4))  (≈ 0)")
println("|u_div| / |u|:        $(round(relnorm(result.u_div) / relnorm(U), sigdigits = 4))  (≈ 0)")
