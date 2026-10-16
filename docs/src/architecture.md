# Architecture

## Package structure

```
src/
  HelmholtzDecomposition.jl   # module; qualified imports only, no exports
  BoundaryConditions.jl       # CornerDirichlet, the dual grid's condition
  Execution.jl                # the execution backend of one decomposition
  Solvers.jl                  # AbstractPoissonSolver, CGSolver, AutoSolver
  Multigrid.jl                # geometric V-cycle preconditioner over FlowGeometries' Galerkin levels
  DualGrid.jl                 # the corner grid the rotation potential lives on
  Decomposition.jl            # plan/workspace/result/batch and helmholtz_decompose[!]
  Spectral.jl                 # Leray projector in mode space
  Scattered.jl                # scattered Cartesian: CartesianNUFFTSolver over FlowTransformBindings
  SpectralSolvers.jl          # the transform solver types and the order AutoSolver tries them in
reference_flows/
  reference_flows.jl          # analytic fields with known decompositions; NOT part of the library
ext/
  HelmholtzDecompositionAbstractFFTsExt.jl        # bounded + channel + periodic, any FFT backend
  HelmholtzDecompositionFFTWExt.jl                # r2r fast path on the host
  HelmholtzDecompositionFSHExt.jl                 # Clenshaw–Curtis sphere
  HelmholtzDecompositionNUFSHTExt.jl              # arbitrary covering sphere
  HelmholtzDecompositionAdaptExt.jl               # how a plan's structs move to a device
  HelmholtzDecompositionOhMyThreadsExt.jl         # threaded batch
  HelmholtzDecompositionDistributedExt.jl         # multiprocess batch
  HelmholtzDecompositionMPIExt.jl                 # MPI batch
  HelmholtzDecompositionCairoMakieExt.jl          # visualization
```

Grids, geometries, masks, topology, the execution primitives and the staggered operators with their
boundary conditions come from
[FlowGeometries](https://github.com/jbphyswx/FlowGeometries.jl); the backend taxonomy from
[ComputationalBackends](https://github.com/jbphyswx/ComputationalBackends.jl); the names for
transform algorithms from
[SpectralBackends](https://github.com/jbphyswx/SpectralBackends.jl); the NUFFT library plans from
[FlowTransformBindings](https://github.com/jbphyswx/FlowTransformBindings.jl). This package defines
none of them.

## One consistent operator family

One gradient `G` is chosen — compact, cell centres to faces — and everything else is *defined* from
it:

```
D = −G*      (adjoint under the cell-measure inner product)
L = D G      (so L = −G*G)
```

`L` is then symmetric negative-semidefinite by construction, so conjugate gradients and multigrid
apply; `div(u_div) = L χ = D u` holds to the solver tolerance; and `u_div ⊥ u_rot` holds discretely
on every grid, the face flux of `u_rot` being the metric-free difference of `R ℓ`, `ℓ` the edge
length at each corner. In two dimensions and on a periodic Cartesian grid the rotation solve is
`curl ∘ rot`, so `harmonic_fraction` measures the domain's loops. On a bounded or curved
three-dimensional grid each rotation component is solved with its own dual Laplacian, and the
harmonic part also carries that solve's departure from `curl ∘ rot`.

A masked face carries zero area. No-flux, the mask treatment and the preservation of adjointness
are then the same statement, and no stencil reaches across a hole.

## Plan, workspace, result

| | holds | shared? |
|---|---|---|
| `HelmholtzPlan` | grid, Laplacian coefficients, face metrics, dual grids, solver *choice* | one per grid, shared by a whole batch |
| `HelmholtzWorkspace` | face/corner buffers **and solver state** | one per task |
| `HelmholtzResult` / `HelmholtzBatch` | outputs; the batch stores `(dims…, N, B)` contiguously | per field / per batch |

Solver state holds buffers a solve writes through — the conjugate-gradient vectors, the multigrid
level arrays, the transform scratch — so it lives in the workspace, one per task.

## Two orthogonal backend axes

- **Which transform** — `AbstractPoissonSolver`. `AutoSolver` chooses on capability: geometry,
  node layout, mask, axis uniformity, boundary condition, and which libraries are loaded. A solver
  named directly that cannot solve the problem raises, because a solver that silently solves a
  *different* boundary value problem returns a smooth, plausible, wrong field.
- **Where it runs** — `ComputationalBackends.AbstractExecutionBackend`, passed to
  `plan_helmholtz`. Every hot loop is written once against `FlowGeometries.Execution.run_indices`,
  which resolves to a serial loop, threaded chunks, or a KernelAbstractions launch with no
  branching here.

## Solver selection

Every solver type is declared in the package; an extension adds the methods that run it on its
library, and `library_loaded(solver)` says whether that library is loaded. Every solve of the
staggered decomposition is of the grid's own discrete `L`. `AutoSolver` takes, in order, the first
that applies:

- a Cartesian transform that inverts `L`: `CartesianSpectralSolver`, `CartesianBoundedSolver`,
  `CartesianRealTransformSolver` (`_auto_candidates`);
- on a sphere, `CGSolver` preconditioned by `SphericalSpectralSolver` or `SphericalNUSHTSolver`
  (`_preconditioner_candidates`), which invert the continuous Laplacian;
- `CGSolver` with multigrid.

`helmholtz_decompose_spectral` keeps its own list (`_spectral_candidates`): the periodic FFT and
the scattered NUFFT on a Cartesian grid, FastSphericalHarmonics and NUFSHT on a sphere. Each list is
written down once, so load order decides nothing.

## Type hierarchy

```
AbstractPoissonSolver
├── AutoSolver
├── CGSolver{R,S}                     # multigrid or spectral (S) preconditioning; R the tolerance type
├── CartesianSpectralSolver           (FFTW)            periodic rfft
├── CartesianBoundedSolver            (FFTW)            r2r, per-direction kind
├── CartesianRealTransformSolver      (AbstractFFTs)    bounded/channel/periodic, any backend
├── CartesianNUFFTSolver{L,D,T}       (NonuniformFFTs or FINUFFT, through FlowTransformBindings)
├── SphericalSpectralSolver           (FastSphericalHarmonics)
└── SphericalNUSHTSolver{T}           (NUFSHT)
```

## Import style

All imports are `using X: X`, called as `X.method()`. No top-level `export`, no
`const X = Submodule.X` flattening.
