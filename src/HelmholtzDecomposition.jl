"""
    HelmholtzDecomposition.jl — Helmholtz–Hodge decomposition of velocity fields.

Decomposes a velocity field into rotational (divergence-free), divergent (curl-free) and
harmonic parts, in any number of dimensions, on Cartesian and spherical grids.

# Why this package exists
On the sphere, filtering velocity Cartesian components does **not** commute with differential
operators (Aluie 2019, Proposition 2). The correct approach — mathematically equivalent to
the generalized convolution that **does** commute — is to filter the scalar Helmholtz
potentials (ψ, χ) separately. This package provides the decomposition step.

# Solver Extensions (important for performance!)
The base package includes only the SOR iterative solver, which works on any grid but may be
**orders of magnitude slower** than spectral solvers. Load an appropriate extension:

| Geometry   | Regular Grid                      | Irregular Grid          |
|------------|-----------------------------------|-------------------------|
| Cartesian  | `using FFTW`                      | `using FINUFFT`         |
| Spherical  | `using FastSphericalHarmonics`    | `using NUFSHT`          |

# Quick Start
```julia
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FFTW: FFTW  # load spectral extension for Cartesian grids

grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, ys)
result = HD.helmholtz_decompose(u, v, grid)
# result.u_rot, result.v_rot  — rotational velocity
# result.u_div, result.v_div  — divergent velocity
# result.ψ, result.χ          — scalar potentials
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
include("Multigrid.jl")
include("DualGrid.jl")
include("Decomposition.jl")
include("Spectral.jl")

# Implemented in ext/HelmholtzDecompositionCairoMakieExt.jl.
function plot_decomposition end

end # module HelmholtzDecomposition
