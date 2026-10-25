# Internals

Names internal to the package, outside its stability guarantee.

## Spectral result types

The physical fields come back in a [`HelmholtzResult`](@ref HelmholtzDecomposition.HelmholtzResult);
these carry the spectral path's own intermediate form.

```@docs
HelmholtzDecomposition.AbstractSpectralHelmholtzResult
HelmholtzDecomposition.SpectralCartesianResult
```

## Solver plumbing

```@docs
HelmholtzDecomposition.CGWorkspace
HelmholtzDecomposition.CGState
HelmholtzDecomposition.SpectralPreconditioner
HelmholtzDecomposition._precondition!
HelmholtzDecomposition._require_domain
HelmholtzDecomposition._require_sampling
HelmholtzDecomposition._unavailable_message
HelmholtzDecomposition._backend_message
HelmholtzDecomposition._auto_candidates
HelmholtzDecomposition._spectral_candidates
HelmholtzDecomposition._spectral_dispatch
```

## Solver kernels

```@docs
HelmholtzDecomposition._dot_measure
HelmholtzDecomposition.lazy_axis_sum
HelmholtzDecomposition._sampling_message
```

## Conditioning of the scattered fit

```@docs
HelmholtzDecomposition.CGLanczos
HelmholtzDecomposition.condition_estimate
HelmholtzDecomposition.condition_limit
HelmholtzDecomposition.warn_conditioning
```

## Face-space parts and fractions

```@docs
HelmholtzDecomposition._divergent_faces!
HelmholtzDecomposition._add_edge_flux!
HelmholtzDecomposition._face_norm
HelmholtzDecomposition._remainder_norm
HelmholtzDecomposition._measure_ratio
```

## Scattered Cartesian fit

```@docs
HelmholtzDecomposition._nufft_library
HelmholtzDecomposition._auto_nufft_solver
HelmholtzDecomposition.ScatteredState
HelmholtzDecomposition.CGBuffers
HelmholtzDecomposition.inverse_nufft!
HelmholtzDecomposition._coldot
HelmholtzDecomposition._reflect
HelmholtzDecomposition._unpack!
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
