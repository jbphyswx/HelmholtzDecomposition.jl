# HelmholtzDecomposition.jl

Helmholtz decomposition of 2D velocity fields into rotational and divergent components.

## Overview

This package decomposes a horizontal velocity field **(u, v)** into:

- **Rotational** (non-divergent): u_rot, v_rot derived from stream function ψ
- **Divergent** (irrotational): u_div, v_div derived from velocity potential χ

by solving the Poisson equations ∇²ψ = ζ and ∇²χ = δ.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jbphyswx/HelmholtzDecomposition.jl")
```

For spectral solver performance, also install one of:
```julia
Pkg.add("FFTW")                    # Cartesian periodic grids
Pkg.add("FastSphericalHarmonics")  # Regular spherical grids
Pkg.add(url="https://github.com/jbphyswx/NUFSHT.jl")  # Irregular spherical
```

## Usage

See [Examples](@ref) for full worked examples, and [Theory](@ref) for mathematical background.
