# Architecture

## Package structure

```
src/
  HelmholtzDecomposition.jl   # module; qualified imports only, no exports
  BoundaryConditions.jl       # CornerDirichlet, the dual grid's condition
  Execution.jl                # the execution backend of one decomposition
  Solvers.jl                  # AbstractPoissonSolver, CGSolver, AutoSolver
  Multigrid.jl                # geometric V-cycle preconditioner over FlowGeometries' coarse levels
  DualGrid.jl                 # the corner grid the rotation potential lives on
  Decomposition.jl            # plan/workspace/result/batch and helmholtz_decompose[!]
  Spectral.jl                 # Leray projector in mode space
  Scattered.jl                # scattered Cartesian: CartesianNUFFTSolver over FlowTransformBindings
  SpectralSolvers.jl          # the transform solver types and the order AutoSolver tries them in
reference_flows/
  reference_flows.jl          # analytic fields with known decompositions; NOT part of the library
ext/
  HelmholtzDecompositionAbstractFFTsExt.jl        # periodic FFT, bounded and channel transforms, any FFT library
  HelmholtzDecompositionFFTWExt.jl                # FFTW's r2r transforms and plan thread count, on the host
  HelmholtzDecompositionFSHExt.jl                 # Clenshaw–Curtis sphere
  HelmholtzDecompositionNUFSHTExt.jl              # arbitrary covering sphere
  HelmholtzDecompositionAdaptExt.jl               # how a plan's structs move to a device
  HelmholtzDecompositionCUDAExt.jl                # CUFFT serves the FFT solvers on a CuArray
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
level arrays, the transform scratch — so it lives in the workspace, one per task. `close!(ws)` releases
the foreign plans a workspace holds, NUFSHT's under a spherical preconditioner; every entry that builds
a workspace of its own releases it before returning.

## Two orthogonal backend axes

- **Which transform** — `AbstractPoissonSolver`. `AutoSolver` chooses from types: the grid's
  geometry, node layout, mask, axis spacing and topology, the boundary condition, the execution backend,
  and which libraries are loaded, so a plan's type follows from the types of its inputs. A solver named
  directly that cannot solve the problem on that backend raises.
- **Where it runs** — `ComputationalBackends.AbstractExecutionBackend`, passed to
  `plan_helmholtz` as `execution`. Every hot loop is written once against `FlowGeometries.Execution.run_indices`,
  which resolves to a serial loop, threaded chunks, or a KernelAbstractions launch with no
  branching here. On a device backend the solver state lives on the device: `AutoSolver` takes the
  Cartesian FFT solvers where the package owning the device's arrays declares an FFT library for them
  (`HelmholtzDecompositionCUDAExt` for a `CuArray`) and multigrid-preconditioned
  conjugate gradients otherwise, and NUFSHT's plans are built on the device when it is named or is the
  spectral route of a covering sphere. FastSphericalHarmonics, FFTW's real-to-real transforms and the
  scattered fit hold host state.

## Solver selection

Every solver type is declared in the package; an extension adds the methods that run it on its
library, and `library_loaded(solver)` says whether that library is loaded. Every solve of the
staggered decomposition is of the grid's own discrete `L`. `AutoSolver` takes, in order, the first
that applies:

- a Cartesian transform that inverts `L`: `CartesianSpectralSolver`, `CartesianBoundedSolver`,
  `CartesianRealTransformSolver` (`_auto_candidates`);
- `CGSolver` with multigrid, smoothed along longitude on a sphere.

`helmholtz_decompose_spectral` keeps its own list (`_spectral_candidates`): the periodic FFT and
the scattered NUFFT on a Cartesian grid, FastSphericalHarmonics and NUFSHT on a sphere. Each list is
written down once, so load order decides nothing.

## Type hierarchy

```
AbstractPoissonSolver
├── AutoSolver
├── CGSolver{R,P}                     # preconditioner P: MultigridCycle, DiagonalScaling or a transform; R the tolerance type
├── CartesianSpectralSolver           (AbstractFFTs)    periodic rfft, any FFT library
├── CartesianBoundedSolver            (FFTW)            r2r, per-direction kind, host
├── CartesianRealTransformSolver      (AbstractFFTs)    bounded/channel/periodic, any FFT library
├── CartesianNUFFTSolver{L,D,T}       (NonuniformFFTs or FINUFFT, through FlowTransformBindings)
├── SphericalSpectralSolver           (FastSphericalHarmonics)
└── SphericalNUSHTSolver{T}           (NUFSHT)
```

## Import style

All imports are `using X: X`, called as `X.method()`. No top-level `export`, no
`const X = Submodule.X` flattening.
