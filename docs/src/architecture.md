# Architecture

## Package Structure

```
src/
  HelmholtzDecomposition.jl   # Main module, includes all subfiles
  Geometry.jl                 # CartesianGeometry, SphericalGeometry
  Grids.jl                    # StructuredGrid with mask
  Solvers.jl                  # AbstractPoissonSolver, SORSolver, AutoSolver
  Decomposition.jl            # helmholtz_decompose!, HelmholtzResult
  TestFields.jl               # Synthetic velocity field generators
ext/
  HelmholtzDecompositionFFTWExt.jl      # CartesianSpectralSolver
  HelmholtzDecompositionFINUFFTExt.jl   # CartesianNUFFTSolver
  HelmholtzDecompositionFSHExt.jl       # SphericalSpectralSolver
  HelmholtzDecompositionNUSHTExt.jl     # SphericalNUSHTSolver
  HelmholtzDecompositionCairoMakieExt.jl # Visualization
```

## Solver Extension Mechanism

Extensions register themselves at `__init__()` time via:
```julia
HelmholtzDecomposition.register_spectral_solver!(:cartesian_regular, CartesianSpectralSolver)
```

The `AutoSolver()` then queries this registry to pick the best available solver.

## Type Hierarchy

```
AbstractGeometry{T}
├── CartesianGeometry{T}
└── SphericalGeometry{T}

AbstractGrid{G, T}
└── StructuredGrid{G, T, V, M, B}

AbstractPoissonSolver
├── AutoSolver
├── SORSolver
├── CartesianSpectralSolver      (ext: FFTW)
├── CartesianNUFFTSolver         (ext: FINUFFT)
├── SphericalSpectralSolver      (ext: FastSphericalHarmonics)
└── SphericalNUSHTSolver         (ext: NUFSHT)
```

## Import Style

All Julia imports follow `using X: X` then `X.method()`. Enforced by Aqua.jl tests.
