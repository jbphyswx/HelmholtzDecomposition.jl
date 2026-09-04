# HelmholtzDecomposition.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://jbphyswx.github.io/HelmholtzDecomposition.jl/dev/)
[![Build Status](https://github.com/jbphyswx/HelmholtzDecomposition.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jbphyswx/HelmholtzDecomposition.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jbphyswx/HelmholtzDecomposition.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jbphyswx/HelmholtzDecomposition.jl)

Helmholtz–Hodge decomposition of velocity fields into **rotational** (divergence-free),
**divergent** (curl-free), and **harmonic** components — in **1D/2D/3D and generically N-D**,
on Cartesian and spherical grids, on **CPU and GPU**, with serial / threaded / distributed /
MPI execution backends.

```
u  =  u_div (∇χ)  ⊕  u_rot (rot R)  ⊕  u_harm
```

- **u_div** is curl-free, from the scalar velocity potential `χ` (`Δχ = ∇·u`).
- **u_rot** is divergence-free, from the rotation potential `R` (`ΔR_ab = ∂_a u_b − ∂_b u_a`);
  in 2D the single component is the streamfunction `ψ`, in 3D the Hodge dual of the vector
  potential `A`, and in N-D an antisymmetric matrix with `N(N−1)/2` components
  (Glötzl & Richters 2023).
- **u_harm** is the harmonic remainder (both div- and curl-free), nonzero on bounded /
  multiply-connected domains (islands, holes), where it carries the circulation/flux that no
  single-valued potential can represent.

## Why This Package Exists

On the sphere, filtering velocity Cartesian components **does not commute** with differential operators (Aluie 2019, eq. 38). The correct approach — mathematically equivalent to the generalized convolution that **does** commute (Proposition 2) — is to filter the scalar Helmholtz potentials (ψ, χ) separately.

Then for coarse-graining: filter ψ̄, χ̄ as scalars → reconstruct velocity from filtered potentials.

### Taylor-Green Vortex (purely rotational → divergent component ≈ 0)

![Taylor-Green Decomposition](docs/src/assets/taylor_green_decomposition.png)

### Vortex + Source (mixed rotational and divergent)

![Mixed Field Decomposition](docs/src/assets/mixed_field_decomposition.png)

### Point Source (purely divergent → rotational component ≈ 0)

![Point Source Decomposition](docs/src/assets/point_source_decomposition.png)

### Harmonic circulation on an annulus (multiply-connected domain)

A pure circulation around a masked hole is **harmonic** — `u_rot ≈ 0`, `u_div ≈ 0`, and the
whole field lands in `u_harm` (`harmonic_fraction ≈ 1`, `count_holes = 1`).

![Harmonic Annulus Decomposition](docs/src/assets/harmonic_annulus_decomposition.png)

### 3-D ABC (Beltrami) flow (`z` mid-slice)

A fully solenoidal 3-D field: the rotational part recovers the original and the divergent
part vanishes.

![3-D Decomposition](docs/src/assets/three_dimensional_decomposition.png)

### 3-D mixed field (`z` mid-slice)

A 3-D field with **both** components (solenoidal ABC + a gradient): the decomposition
splits `uₓ` into a nonzero rotational and a nonzero divergent part.

![3-D Mixed Decomposition](docs/src/assets/three_dimensional_mixed_decomposition.png)

### Spherical mixed flow (Kelvin–Ekman, NUFSHT)

Decomposition on a longitude–latitude grid via the non-uniform spherical-harmonic
transform: a rotational core plus a smaller divergent (Ekman-like) part.

![Spherical Decomposition](docs/src/assets/spherical_decomposition.png)

## Solver extensions

The base package solves any grid, mask and boundary condition with multigrid-preconditioned
conjugate gradients. Where a transform applies it is `O(N log N)`, and each arrives with its
package:

| Geometry | Regular grid | Irregular grid |
|----------|-------------|----------------|
| **Cartesian** | `using FFTW` (periodic, bounded, channel) or `using AbstractFFTs` | `using FINUFFT` or `using NonuniformFFTs` |
| **Spherical** | `using FastSphericalHarmonics` (Clenshaw–Curtis) | `using NUFSHT` (any covering node set) |

`AutoSolver()` (the default) picks among the loaded ones on the grid's own properties — mask, axis
uniformity, topology and node layout.

**Build axes from ranges.** FlowGeometries proves uniformity from the axis *type*, so a
`collect`ed range is a `Vector`, carries no such proof, and quietly costs every transform fast
path.

## Quick Start

```julia
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FFTW: FFTW  # load a spectral extension

xs = range(0.0, 99_000.0; length = 100)      # a range: uniformity provable from the type
grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, xs)

# Physical-space decomposition (mask- and boundary-aware; a transform where one applies)
result = HD.helmholtz_decompose(u, v, grid)

# Or the fully spectral path (exact derivatives, physical fields back)
result = HD.helmholtz_decompose_spectral(u, v, grid)

# Velocity-like fields use a component-last layout (dims..., N):
result.u_rot          # rotational (divergence-free) velocity, (Nx, Ny, 2)
result.u_div          # divergent (curl-free) velocity
result.u_harm         # harmonic remainder
result.χ              # scalar velocity potential
HD.streamfunction(result)   # ψ (2D); HD.vector_potential(result) in 3D
result.harmonic_fraction    # ‖u_harm‖ / ‖u‖ — how much lives in the harmonic subspace
```

### N-dimensional, GPU, and batch

```julia
# 3-D: pass a single component-last array (Nx, Ny, Nz, 3)
res3 = HD.helmholtz_decompose_spectral(U3, grid3)
A1, A2, A3 = HD.vector_potential(res3)

# GPU: pass a CuArray (requires `using CUDA`); AutoBackend routes to the CUFFT path
res_gpu = HD.helmholtz_decompose_spectral(CUDA.cu(U), grid)

# Batch many snapshots in parallel (ThreadedBackend / DistributedBackend / MPIBackend)
results = HD.helmholtz_decompose_batch(grid, fields; backend = HD.ThreadedBackend())
```

### Scattered / unstructured data

For data sampled at arbitrary point locations (not a grid — observation networks, floats,
tracks), use `ScatteredPoints`. It routes through the non-uniform transforms: an accurate
inverse NUFFT (conjugate-gradient least-squares, not the naive adjoint) → exact Leray
projection → synthesis back to the points.

```julia
using FINUFFT: FINUFFT
# One coordinate vector per direction, plus a control volume per node.
pts = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), (x, y), areas;
                                periodic = (true, true), period = (Lx, Ly))
res = HD.helmholtz_decompose_spectral(U, pts)   # U is (M, 2) → (; u_rot, u_div, u_harm)
```

On the sphere the same entry point takes **any node set covering `S²`**, through NUFSHT's
spin-weighted transforms: the tangent velocity `V = u_θ + i u_φ` is a spin-1 field, and the
symmetric and antisymmetric parts of its spin(±1) coefficients are its rotational and divergent
components — an exact Hodge split with no finite-difference stencil.

Because it asks only for node positions, one implementation covers every spherical layout
FlowGeometries builds: lat–lon and the spectral quadrature grids, `UnstructuredGrid` point clouds,
and the pixelizations (`HEALPixGrid`, `RingGrid`, `CubedSphereGrid`, `IcosahedralGrid`,
`YinYangGrid`).

```julia
using NUFSHT: NUFSHT
sph  = FG.Geometry.SphericalGeometry(1.0)
grid = FG.Grids.StructuredGrid(sph, lons, lats; topology = (true, false), period = (2π, 0.0))
res  = HD.helmholtz_decompose_spectral(U, grid)   # U is (nlon, nlat, 2) = (u_east, u_north)
```

The expansion degree is sized from the grid unless `lmax` is given, and a fit that does not
converge raises.

### Multiply-connected domains (the harmonic part)

```julia
# An island: any Bool array of the grid's shape, false where a cell is out of the domain.
mask = [ (x - 0.5)^2 + (y - 0.5)^2 > 0.09 for x in xs, y in ys ]
grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, ys; mask = mask)
HD.count_holes(grid)                                  # 1  (b₁ of the active region)
res = HD.helmholtz_decompose(u, v, grid)
res.harmonic_fraction                                 # ≈ 1 for a pure circulation about the hole
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

## Two backend axes

The package keeps two orthogonal axes separate:

**Spectral / Poisson solver** (the math), selected by `AutoSolver()` or passed explicitly:

| Solver | When to use |
|--------|-------------|
| `CGSolver` (base, dimension-generic) | any grid: masked domains, any boundary condition, any geometry. Multigrid-preconditioned by default |
| `CartesianSpectralSolver` (FFTW) | uniform periodic Cartesian, any dimension |
| `CartesianBoundedSolver` (FFTW) | uniform Cartesian with any mix of bounded and periodic directions, including a channel |
| `CartesianRealTransformSolver` (AbstractFFTs) | the same, through whichever backend owns the array |
| `CartesianNUFFTSolver` (FINUFFT) / NonuniformFFTs | scattered 2-D Cartesian |
| `SphericalSpectralSolver` (FastSphericalHarmonics) | Clenshaw–Curtis lat/lon (`Nlon = 2·Nlat−1`) |
| `SphericalNUSHTSolver` (NUFSHT) | any spherical node set covering `S²` |

`AutoSolver` is mask-aware (never picks a periodic spectral solver on a masked domain) and
prefers the regular FFT/SHT on structured grids.

**Execution backend** (where/how arrays compute): `SerialBackend`, `ThreadedBackend`
(`using OhMyThreads`), `GPUBackend` (`using CUDA`), `DistributedBackend` (`using Distributed`),
`MPIBackend` (`using MPI`). Passed via `backend=` to `helmholtz_decompose` /
`helmholtz_decompose_batch`; `AutoBackend()` infers it from the array type.

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
