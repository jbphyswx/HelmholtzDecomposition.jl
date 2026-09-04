"""
    HelmholtzDecomposition.jl — Helmholtz–Hodge decomposition of velocity fields.

Decomposes a velocity field into rotational (divergence-free), divergent (curl-free) and
harmonic parts, in any number of dimensions, on Cartesian and spherical grids.

# Why this package exists
On the sphere, filtering velocity Cartesian components does **not** commute with differential
operators (Aluie 2019, Proposition 2). The correct approach — mathematically equivalent to
the generalized convolution that **does** commute — is to filter the scalar Helmholtz
potentials (ψ, χ) separately. This package provides the decomposition step.

# Solver extensions
The base package solves any grid, mask and boundary condition with multigrid-preconditioned
conjugate gradients. A transform is `O(N log N)` where one applies, and each arrives with its
package:

| Geometry   | Regular grid                      | Irregular grid                         |
|------------|-----------------------------------|----------------------------------------|
| Cartesian  | `using FFTW`, `using AbstractFFTs`| `using FINUFFT`, `using NonuniformFFTs`|
| Spherical  | `using FastSphericalHarmonics`    | `using NUFSHT`                         |

`AutoSolver` picks among the loaded ones on the grid's own properties — mask, axis uniformity,
topology and node layout — and refuses a solver named directly that cannot solve the problem given.

# Quick start
```julia
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FFTW: FFTW

xs = range(0, 1; length = 128)          # a range, so uniformity is provable from the axis type
grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, xs)
u = randn(128, 128, 2)                  # component-last: (dims..., N)

result = HD.helmholtz_decompose(u, grid)
result.u_rot              # rotational (divergence-free) velocity, (128, 128, 2)
result.u_div              # divergent (curl-free) velocity
result.u_harm             # harmonic remainder
result.χ                  # scalar velocity potential, at cell centres
HD.streamfunction(result) # ψ in 2-D; `HD.vector_potential` in 3-D
result.harmonic_fraction  # ‖u_harm‖ / ‖u‖
```

Decomposing more than one field on a grid builds the plan once:

```julia
plan = HD.plan_helmholtz(grid)
ws, res = HD.allocate_workspace(plan), HD.allocate_result(plan)
HD.helmholtz_decompose!(res, u, plan, ws)      # allocation-free
```

Nothing is exported: every name is reached as `HelmholtzDecomposition.name`.

# References
- Aluie (2019): doi:10.1007/s13137-019-0123-9 — Convolutions on the sphere
- Glötzl & Richters (2023): doi:10.1016/j.jmaa.2023.127138 — n-dimensional Helmholtz potentials
- Buzzicotti et al. (2023): doi:10.1126/sciadv.adi7420 — Global cascade of kinetic energy
- Storer et al. (2022): doi:10.1038/s41467-022-33031-3 — Global energy spectrum
"""
module HelmholtzDecomposition

using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries
using SpectralBackends: SpectralBackends
using LinearAlgebra: LinearAlgebra

include("BoundaryConditions.jl")
include("Staggering.jl")
include("Operators.jl")
include("Solvers.jl")
include("LeastSquares.jl")
include("Multigrid.jl")
include("DualGrid.jl")
include("Decomposition.jl")
include("Spectral.jl")

# Implemented in ext/HelmholtzDecompositionCairoMakieExt.jl.
function plot_decomposition end

end # module HelmholtzDecomposition
