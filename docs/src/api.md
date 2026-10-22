# API

Everything is reached through the module: this package exports nothing, so calls are written
`HelmholtzDecomposition.plan_helmholtz(...)` or via an alias, `HD.plan_helmholtz(...)`.

```@docs
HelmholtzDecomposition.HelmholtzDecomposition
```

## Entry points

```@docs
HelmholtzDecomposition.helmholtz_decompose
HelmholtzDecomposition.helmholtz_decompose!
HelmholtzDecomposition.helmholtz_potentials
HelmholtzDecomposition.helmholtz_decompose_batch
HelmholtzDecomposition.helmholtz_decompose_batch!
```

How much work `helmholtz_decompose!` does is the result's type: a
[`PotentialsResult`](@ref HelmholtzDecomposition.PotentialsResult) stops once `χ` and `R` are
solved and allocates no collocated velocity array, while a
[`HelmholtzResult`](@ref HelmholtzDecomposition.HelmholtzResult) reconstructs the three parts.

## Plan, workspace and results

```@docs
HelmholtzDecomposition.plan_helmholtz
HelmholtzDecomposition.HelmholtzPlan
HelmholtzDecomposition.HelmholtzWorkspace
HelmholtzDecomposition.allocate_workspace
HelmholtzDecomposition.close!
HelmholtzDecomposition.AbstractHelmholtzResult
HelmholtzDecomposition.HelmholtzResult
HelmholtzDecomposition.allocate_result
HelmholtzDecomposition.PotentialsResult
HelmholtzDecomposition.allocate_potentials
HelmholtzDecomposition.HelmholtzBatch
HelmholtzDecomposition.allocate_batch
```

## Reading a result

```@docs
HelmholtzDecomposition.streamfunction
HelmholtzDecomposition.velocity_potential
HelmholtzDecomposition.vector_potential
HelmholtzDecomposition.count_holes
```

The staggered fields are where the projection is exact; the collocated ones on the result are
these interpolated back to cell centres.

```@docs
HelmholtzDecomposition.face_velocity
HelmholtzDecomposition.face_divergent
HelmholtzDecomposition.face_rotational
HelmholtzDecomposition.face_harmonic
HelmholtzDecomposition.corner_rotation_potential
```

## Boundary conditions

A bounded direction takes FlowGeometries' `FG.Operators.Neumann()` (the default) or
`FG.Operators.Dirichlet()`. Whether a direction wraps is the grid's own topology, so there is no
periodic condition. The rotation potential's dual grid is solved under:

```@docs
HelmholtzDecomposition.CornerDirichlet
```

## Solvers

```@docs
HelmholtzDecomposition.AbstractPoissonSolver
HelmholtzDecomposition.AutoSolver
HelmholtzDecomposition.CGSolver
HelmholtzDecomposition.solve_poisson!
HelmholtzDecomposition.prepare_solver
HelmholtzDecomposition.prepare_shared
HelmholtzDecomposition.select_solver
HelmholtzDecomposition.SolverResult
```

### Transform solvers

Each runs once its library is loaded; see [`library_loaded`](@ref HelmholtzDecomposition.library_loaded).

```@docs
HelmholtzDecomposition.CartesianSpectralSolver
HelmholtzDecomposition.CartesianBoundedSolver
HelmholtzDecomposition.CartesianRealTransformSolver
HelmholtzDecomposition.CartesianNUFFTSolver
HelmholtzDecomposition.SphericalSpectralSolver
HelmholtzDecomposition.SphericalNUSHTSolver
```

### Capability

`AutoSolver` chooses on these. A named solver whose state cannot live on the execution backend is
refused, as is one that cannot invert the grid's `L`, except a spherical transform, which becomes the
preconditioner of [`CGSolver`](@ref HelmholtzDecomposition.CGSolver).

```@docs
HelmholtzDecomposition.supports_boundary
HelmholtzDecomposition.requires_full_domain
HelmholtzDecomposition.requires_uniform_axes
HelmholtzDecomposition.requires_periodic_domain
HelmholtzDecomposition.supports_sampling
HelmholtzDecomposition.supports_backend
HelmholtzDecomposition.library_loaded
HelmholtzDecomposition.inverts_discrete_laplacian
HelmholtzDecomposition.preconditions
HelmholtzDecomposition.spectral_inverse!
```

## Operators and the dual grid

The staggered operators are FlowGeometries': `FG.Operators.FaceMetrics` with `gradient!`,
`divergence!`, `curl!`, `rotational_velocity!`, `to_faces!` and `to_centres!`, and the assembled
`FG.Operators.LaplacianCoefficients` with its null space and Galerkin hierarchy. A plan holds the
primal grid's metrics and coefficients and, per rotation pair, a dual grid with its own operator:

```@docs
HelmholtzDecomposition.dual_grid
HelmholtzDecomposition.dual_boundary
HelmholtzDecomposition.dual_coefficients
```

## Execution

```@docs
HelmholtzDecomposition.resolve_execution_backend
HelmholtzDecomposition.LoopSizedThreads
HelmholtzDecomposition.loop_backend
```

## Multigrid

```@docs
HelmholtzDecomposition.multigrid
HelmholtzDecomposition.MultigridPreconditioner
HelmholtzDecomposition.LevelBuffers
HelmholtzDecomposition.MultigridBuffers
HelmholtzDecomposition.multigrid_buffers
HelmholtzDecomposition.apply_preconditioner!
HelmholtzDecomposition.smooth!
HelmholtzDecomposition.smooth_from_zero!
HelmholtzDecomposition.vcycle!
```

## Spectral

```@docs
HelmholtzDecomposition.helmholtz_project_spectral!
HelmholtzDecomposition.helmholtz_project_spectral
HelmholtzDecomposition.helmholtz_potentials_spectral
HelmholtzDecomposition.helmholtz_decompose_spectral
HelmholtzDecomposition.velocity_norm
```
