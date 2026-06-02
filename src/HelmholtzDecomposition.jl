"""
    HelmholtzDecomposition.jl — Helmholtz decomposition of 2D velocity fields.

Decomposes a horizontal velocity field into rotational (non-divergent, stream function ψ)
and divergent (irrotational, velocity potential χ) components by solving Poisson equations.

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
using HelmholtzDecomposition: HelmholtzDecomposition
using FFTW: FFTW  # load spectral extension for Cartesian grids

geom = HelmholtzDecomposition.CartesianGeometry(1000.0, 1000.0)
grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)
# result.u_rot, result.v_rot  — rotational velocity
# result.u_div, result.v_div  — divergent velocity
# result.ψ, result.χ          — scalar potentials (filter these for coarse-graining!)
```

# References
- Aluie (2019): doi:10.1007/s13137-019-0123-9 — Convolutions on the sphere
- Buzzicotti et al. (2023): doi:10.1126/sciadv.adi7420 — Global cascade of kinetic energy
- Storer et al. (2022): doi:10.1038/s41467-022-33031-3 — Global energy spectrum
"""
module HelmholtzDecomposition

include("Geometry.jl")
include("Grids.jl")
include("Solvers.jl")
include("Decomposition.jl")
include("TestFields.jl")

# Extension point for CairoMakie visualization (implemented in ext/HelmholtzDecompositionCairoMakieExt.jl)
function plot_decomposition end
export plot_decomposition

end # module HelmholtzDecomposition
