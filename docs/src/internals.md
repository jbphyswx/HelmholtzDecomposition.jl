# Internals

Not part of the interface and not subject to its stability. Documented because a reader following
the operator construction or the solver dispatch will meet them, and because a manual that omits
half of what a module defines is harder to trust than one that marks the boundary explicitly.

## Spectral result types

The physical fields come back in a [`HelmholtzResult`](@ref HelmholtzDecomposition.HelmholtzResult);
these carry the spectral path's own intermediate form.

```@docs
HelmholtzDecomposition.AbstractSpectralHelmholtzResult
HelmholtzDecomposition.SpectralCartesianResult
HelmholtzDecomposition.SpectralSphericalResult
HelmholtzDecomposition.build_cartesian_result
```

## Solver plumbing

```@docs
HelmholtzDecomposition.CGWorkspace
HelmholtzDecomposition.CGState
HelmholtzDecomposition._precondition!
HelmholtzDecomposition.ComponentNullspace
HelmholtzDecomposition._nullspace
HelmholtzDecomposition._require_converged
HelmholtzDecomposition._require_domain
HelmholtzDecomposition._require_sampling
HelmholtzDecomposition._unavailable_message
HelmholtzDecomposition._auto_candidates
HelmholtzDecomposition._spectral_dispatch
```

## Scattered Cartesian fit

```@docs
HelmholtzDecomposition._nufft_library
HelmholtzDecomposition._auto_nufft_solver
HelmholtzDecomposition.ScatteredState
HelmholtzDecomposition.close!
HelmholtzDecomposition.CGBuffers
HelmholtzDecomposition.inverse_nufft!
HelmholtzDecomposition._coldot
HelmholtzDecomposition._reflect
HelmholtzDecomposition._unpack!
```

## Grid walking

```@docs
HelmholtzDecomposition.cell_above
HelmholtzDecomposition.cell_below
HelmholtzDecomposition.face_below
HelmholtzDecomposition.geometric_face_area
```

## Batch and backend plumbing

```@docs
HelmholtzDecomposition.copy_slice!
HelmholtzDecomposition._resolve_batch_backend
HelmholtzDecomposition._extension_loaded
HelmholtzDecomposition._unsupported_backend_message
```

## Spectral plumbing

```@docs
HelmholtzDecomposition._grid_wavenumbers
HelmholtzDecomposition._stack_components
```
