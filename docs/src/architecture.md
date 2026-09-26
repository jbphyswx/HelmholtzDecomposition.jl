# Architecture

## Package structure

```
src/
  HelmholtzDecomposition.jl   # module; qualified imports only, no exports
  BoundaryConditions.jl       # Dirichlet, Neumann
  Staggering.jl               # face/corner counts and indexing (Arakawa C-grid / MAC)
  Operators.jl                # G, D = −G*, L = D G, curl, δ; face metrics; execution backend
  Solvers.jl                  # AbstractPoissonSolver, CGSolver, AutoSolver
  Multigrid.jl                # geometric V-cycle preconditioner (Galerkin coarsening)
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

Grids, geometries, masks, topology and the execution primitives come from
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

`L` is then symmetric negative-semidefinite by construction, which is what makes conjugate
gradients and multigrid applicable rather than merely plausible; `div(u_div) = L χ = D u` holds
exactly; and `u_div ⊥ u_rot` discretely, so `harmonic_fraction` reports topology rather than
disagreement between separately-chosen stencils.

A masked face carries zero area. No-flux, the mask treatment and the preservation of adjointness
are then the same statement, and no stencil reaches across a hole.

## Plan, workspace, result

| | holds | shared? |
|---|---|---|
| `HelmholtzPlan` | grid, Laplacian coefficients, face metrics, dual grids, solver *choice* | one per grid, shared by a whole batch |
| `HelmholtzWorkspace` | face/corner buffers **and solver state** | one per task |
| `HelmholtzResult` / `HelmholtzBatch` | outputs; the batch stores `(dims…, N, B)` contiguously | per field / per batch |

Solver state is in the workspace, not the plan, because it holds buffers a solve writes through —
the conjugate-gradient vectors, the multigrid level arrays, the transform scratch. On the plan,
a threaded batch would have every task writing the same ones.

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
library, and `library_loaded(solver)` says whether that library is loaded. `AutoSolver` walks one
fixed list per geometry (`_auto_candidates`) and takes the first solver whose library is loaded
and whose requirements the grid meets:

- Cartesian: `CartesianSpectralSolver`, `CartesianBoundedSolver`, `CartesianRealTransformSolver`,
  `CartesianNUFFTSolver`;
- spherical: `SphericalSpectralSolver`, `SphericalNUSHTSolver`;
- otherwise, and when none applies, `CGSolver`.

The order is written down once, so load order decides nothing.

## Type hierarchy

```
AbstractPoissonSolver
├── AutoSolver
├── CGSolver{R}                       # multigrid-preconditioned by default; R the tolerance type
├── CartesianSpectralSolver           (FFTW)            periodic rfft
├── CartesianBoundedSolver            (FFTW)            r2r, per-direction kind
├── CartesianRealTransformSolver      (AbstractFFTs)    bounded/channel/periodic, any backend
├── CartesianNUFFTSolver{L,D,T}       (NonuniformFFTs or FINUFFT, through FlowTransformBindings)
├── SphericalSpectralSolver           (FastSphericalHarmonics)
└── SphericalNUSHTSolver{T}           (NUFSHT)
```

## Import style

All imports are `using X: X`, called as `X.method()`. No top-level `export`, no
`const X = Submodule.X` flattening. Enforced by Aqua.
