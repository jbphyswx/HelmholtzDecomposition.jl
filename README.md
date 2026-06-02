# HelmholtzDecomposition.jl

Helmholtz decomposition of 2D velocity fields into rotational (non-divergent) and divergent (irrotational) components, with support for both Cartesian and spherical geometries.

## Why This Package Exists

On the sphere, filtering velocity Cartesian components **does not commute** with differential operators (Aluie 2019, eq. 38). The correct approach — mathematically equivalent to the generalized convolution that **does** commute (Proposition 2) — is to filter the scalar Helmholtz potentials (ψ, χ) separately.

This package provides the decomposition step:

```
u → (ψ, χ)     via solving ∇²ψ = ζ, ∇²χ = δ
```

Then for coarse-graining: filter ψ̄, χ̄ as scalars → reconstruct velocity from filtered potentials.

### Taylor-Green Vortex (purely rotational → divergent component ≈ 0)

![Taylor-Green Decomposition](docs/assets/taylor_green_decomposition.png)

### Vortex + Source (mixed rotational and divergent)

![Mixed Field Decomposition](docs/assets/mixed_field_decomposition.png)

### Point Source (purely divergent → rotational component ≈ 0)

![Point Source Decomposition](docs/assets/point_source_decomposition.png)

## Solver Extensions ⚠️

The base package includes only an iterative SOR solver (Red-Black Successive Over-Relaxation). While correct, it is **orders of magnitude slower** than spectral solvers for large grids.

**Load the appropriate extension for your geometry:**

| Geometry | Regular Grid | Irregular Grid |
|----------|-------------|----------------|
| **Cartesian (periodic)** | `using FFTW` | `using FINUFFT` |
| **Spherical** | `using FastSphericalHarmonics` | `using NUFSHT` |

The `AutoSolver()` (default) automatically picks the best loaded solver. If no spectral extension is loaded, it falls back to SOR with a `@debug` message.

## Quick Start

```julia
using HelmholtzDecomposition: HelmholtzDecomposition
using FFTW: FFTW  # load spectral extension

# Create grid
geom = HelmholtzDecomposition.CartesianGeometry(1000.0, 1000.0)
xs = collect(0.0:1000.0:99000.0)
ys = collect(0.0:1000.0:99000.0)
grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

# Decompose
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

# Access results
result.u_rot, result.v_rot   # rotational (non-divergent) velocity
result.u_div, result.v_div   # divergent (irrotational) velocity
result.ψ                     # stream function
result.χ                     # velocity potential
```

## Mathematical Formulation

The Helmholtz decomposition expresses any 2D vector field as:

**u** = **u**_rot + **u**_div

where:
- **u**_rot = ∇ × (ψ ẑ) is non-divergent (∇ · **u**_rot = 0)
- **u**_div = ∇χ is irrotational (∇ × **u**_div = 0)

The scalar potentials are found by solving:
- ∇²ψ = ζ (vorticity = ∂v/∂x − ∂u/∂y)
- ∇²χ = δ (divergence = ∂u/∂x + ∂v/∂y)

On the sphere (radius R):
- ∇²ψ = ζ → eigenvalue −ℓ(ℓ+1)/R²
- u_rot = −(1/R) ∂ψ/∂φ, v_rot = 1/(R cos φ) ∂ψ/∂λ

## When is Helmholtz Required?

- **NOT needed:** Non-divergent velocity (e.g., SSH-derived geostrophic flow) — Storer et al. (2022)
- **REQUIRED:** Full model velocity with both rotational AND divergent components
- **ALSO needed:** Separating energy flux Π into toroidal/potential contributions — Buzzicotti et al. (2023)

## Solver Backends

| Solver | Complexity | When to use |
|--------|-----------|-------------|
| `SORSolver` | O(N²) | Masked domains, complex BCs, small grids |
| `CartesianSpectralSolver` (FFTW) | O(N log N) | Regular periodic Cartesian |
| `CartesianNUFFTSolver` (FINUFFT) | O(N log N) | Irregular Cartesian |
| `SphericalSpectralSolver` (FSH) | O(N log N) | Regular lat/lon |
| `SphericalNUSHTSolver` (NUFSHT) | O(N log N) | Irregular/scattered spherical |

## Relationship to Structure Function "Helmholtz Decomposition"

This package performs **spatial** Helmholtz decomposition (Poisson solver). This is distinct from:
- **Lindborg (2015) integral relations** used in `StructureFunctions.jl` for decomposing D_LL, D_TT → D_rot, D_div via cumulative integrals (no Poisson solver needed).

These are two completely different operations that happen to share the name "Helmholtz decomposition."

## References

- **Aluie (2019)**: doi:10.1007/s13137-019-0123-9 — Convolutions on the sphere; Proposition 2 proves Helmholtz filtering commutes with ∇
- **Buzzicotti, Storer, Khatri, Griffies, Aluie (2023)**: doi:10.1126/sciadv.adi7420 — Global kinetic energy cascade using Helmholtz filtering
- **Storer et al. (2022)**: doi:10.1038/s41467-022-33031-3 — When Helmholtz is not needed (non-divergent fields)
- **Lindborg (2015)**: doi:10.1017/jfm.2014.685 — SF integral relations (different "Helmholtz")
- **Berlinghieri et al. (2023)**: doi:10.1029/2022GL097713 — GP Helmholtz for ocean currents

## See Also

- [ImmersedLayers.jl](https://juliaibpm.github.io/ImmersedLayers.jl/stable/manual/helmholtz/) — Alternative Helmholtz implementation using lattice Green's functions
- [FlowSieve](https://flowsieve.readthedocs.io/) — C++ coarse-graining toolkit with Helmholtz mode
