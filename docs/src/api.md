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
HelmholtzDecomposition.helmholtz_decompose_batch
HelmholtzDecomposition.helmholtz_decompose_batch!
```

## Plan, workspace and results

```@docs
HelmholtzDecomposition.plan_helmholtz
HelmholtzDecomposition.HelmholtzPlan
HelmholtzDecomposition.HelmholtzWorkspace
HelmholtzDecomposition.allocate_workspace
HelmholtzDecomposition.HelmholtzResult
HelmholtzDecomposition.allocate_result
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
HelmholtzDecomposition.corner_rotation_potential
```

## Boundary conditions

```@docs
HelmholtzDecomposition.AbstractBoundaryCondition
HelmholtzDecomposition.Dirichlet
HelmholtzDecomposition.Neumann
```

There is no `Periodic` boundary condition: whether a direction wraps is a property of the grid's
topology, and a direction that wraps has no boundary for a condition to act on.

## Solvers

```@docs
HelmholtzDecomposition.AbstractPoissonSolver
HelmholtzDecomposition.AutoSolver
HelmholtzDecomposition.CGSolver
HelmholtzDecomposition.solve_poisson!
HelmholtzDecomposition.prepare_solver
HelmholtzDecomposition.select_solver
HelmholtzDecomposition.SolverResult
```

### Capability

`AutoSolver` chooses on these, and a solver named directly is refused rather than allowed to solve
a different problem.

```@docs
HelmholtzDecomposition.supports_boundary
HelmholtzDecomposition.requires_full_domain
HelmholtzDecomposition.requires_uniform_axes
HelmholtzDecomposition.requires_periodic_domain
HelmholtzDecomposition.register_spectral_solver!
```

## Operators

```@docs
HelmholtzDecomposition.gradient!
HelmholtzDecomposition.divergence!
HelmholtzDecomposition.laplacian!
HelmholtzDecomposition.apply_laplacian!
HelmholtzDecomposition.curl!
HelmholtzDecomposition.rotational_velocity!
HelmholtzDecomposition.to_faces!
HelmholtzDecomposition.to_centres!
HelmholtzDecomposition.LaplacianCoefficients
HelmholtzDecomposition.laplacian_coefficients
HelmholtzDecomposition.FaceMetrics
HelmholtzDecomposition.face_area
HelmholtzDecomposition.face_gap
```

## Staggering and the dual grid

```@docs
HelmholtzDecomposition.nfaces
HelmholtzDecomposition.ncorners
HelmholtzDecomposition.corner_offset
HelmholtzDecomposition.face_dims
HelmholtzDecomposition.corner_dims
HelmholtzDecomposition.allocate_faces
HelmholtzDecomposition.allocate_corners
HelmholtzDecomposition.rotation_pairs
HelmholtzDecomposition.rotation_terms
HelmholtzDecomposition.n_rotation_components
HelmholtzDecomposition.face_coordinates
HelmholtzDecomposition.corner_mask
HelmholtzDecomposition.dual_grid
```

## Execution

```@docs
HelmholtzDecomposition.execution_backend
HelmholtzDecomposition.resolve_execution_backend
HelmholtzDecomposition.allocate_zeros
HelmholtzDecomposition.to_backend
```

## Multigrid

```@docs
HelmholtzDecomposition.multigrid
HelmholtzDecomposition.MultigridPreconditioner
HelmholtzDecomposition.MultigridLevel
HelmholtzDecomposition.coarsen
HelmholtzDecomposition.galerkin_coefficients
HelmholtzDecomposition.restrict!
HelmholtzDecomposition.prolong_add!
HelmholtzDecomposition.smooth!
HelmholtzDecomposition.vcycle!
```

## Spectral

```@docs
HelmholtzDecomposition.helmholtz_project_spectral!
HelmholtzDecomposition.helmholtz_project_spectral
HelmholtzDecomposition.helmholtz_potentials_spectral
HelmholtzDecomposition.helmholtz_decompose_spectral
HelmholtzDecomposition.velocity_norm
HelmholtzDecomposition.project_out_constant!
```
