# Architecture

## Package Structure

```
src/
  HelmholtzDecomposition.jl   # Main module, includes all subfiles
  Backends.jl                 # Execution-backend taxonomy (Serial/Threaded/GPU/MPI/...)
  Geometry.jl                 # CartesianGeometry{N}, SphericalGeometry
  Grids.jl                    # StructuredGrid{N} with mask (1D/2D/3D/ND)
  Operators.jl                # Dimension-generic FD: divergence, rotation tensor, reconstruction
  Topology.jl                 # count_holes / betti1_estimate (harmonic-subspace dimension)
  Solvers.jl                  # AbstractPoissonSolver, SORSolver (ND), AutoSolver (mask-aware)
  Decomposition.jl            # helmholtz_decompose[!], helmholtz_decompose_batch, HelmholtzResult{N}
  Spectral.jl                 # Dimension-generic Leray projector + physical-result assembler
  TestFields.jl               # Synthetic fields (incl. harmonic vortex/source, disk_mask)
ext/
  HelmholtzDecompositionFFTWExt.jl         # CartesianSpectralSolver (ND, physical output)
  HelmholtzDecompositionFINUFFTExt.jl      # CartesianNUFFTSolver (2D scattered)
  HelmholtzDecompositionFSHExt.jl          # SphericalSpectralSolver (Clenshaw–Curtis)
  HelmholtzDecompositionNUSHTExt.jl        # SphericalNUSHTSolver (arbitrary spherical)
  HelmholtzDecompositionCUDAExt.jl         # GPU spectral path via CUFFT
  HelmholtzDecompositionOhMyThreadsExt.jl  # threaded batch
  HelmholtzDecompositionDistributedExt.jl  # multiprocess batch
  HelmholtzDecompositionMPIExt.jl          # MPI batch
  HelmholtzDecompositionCairoMakieExt.jl   # Visualization
```

## Two orthogonal backend axes

- **Spectral/Poisson solver** (the math): `AbstractPoissonSolver` + a registry; `AutoSolver`
  is mask-aware and picks the regular FFT/SHT on structured grids. `helmholtz_decompose_spectral`
  dispatches `_decompose_spectral` on the *solver type*, so multiple spectral backends coexist.
- **Execution backend** (where/how arrays compute): `AbstractExecutionBackend`
  (`SerialBackend`, `ThreadedBackend`, `GPUBackend`, `DistributedBackend{Inner}`,
  `MPIBackend{Inner}`), passed via `backend=`; `AutoBackend` infers it from the array type.

## Solver Extension Mechanism

Extensions register themselves at `__init__()` time via:
```julia
HelmholtzDecomposition.register_spectral_solver!(:cartesian_regular, CartesianSpectralSolver)
```

The `AutoSolver()` then queries this registry to pick the best available solver.

## Type Hierarchy

```
AbstractGeometry{T}
├── CartesianGeometry{N, T}
└── SphericalGeometry{T}        # a 2-surface (N == 2)

AbstractGrid{G, T}
└── StructuredGrid{N, G, T, C, A, B}

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
