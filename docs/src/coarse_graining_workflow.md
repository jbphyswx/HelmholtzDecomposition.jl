# Coarse-Graining Workflow

## The Problem

When coarse-graining (filtering) velocity fields on the sphere, the naive approach of:
1. Convert (u_east, u_north) → (u_X, u_Y, u_Z) planetary Cartesian
2. Filter each Cartesian component as a scalar
3. Convert back to spherical

does **NOT** commute with differential operators (Aluie 2019). This means:
- Filtering a divergence-free field does NOT produce a divergence-free filtered field
- Physical properties (incompressibility, etc.) are violated

## The Correct Approach

The theoretically correct workflow (equivalent to the Edmonds transformation):

```
Original velocity (u, v)
    │
    ▼
┌─────────────────────────────┐
│  Helmholtz Decomposition    │  ← This package
│  Solve ∇²ψ = ζ, ∇²χ = δ    │
└─────────────────────────────┘
    │
    ▼
Scalar potentials (ψ, χ)
    │
    ▼
┌─────────────────────────────┐
│  Filter ψ and χ separately  │  ← CoarseGrainingEnergyFluxes.jl
│  as scalars on the sphere   │     (or any filtering package)
└─────────────────────────────┘
    │
    ▼
Filtered potentials (ψ̄, χ̄)
    │
    ▼
┌─────────────────────────────┐
│  Reconstruct velocity from  │
│  filtered potentials        │
└─────────────────────────────┘
    │
    ▼
Correctly filtered velocity (ū, v̄)
    ✓ Commutes with ∇
    ✓ Preserves incompressibility
    ✓ Enables Π_tor / Π_pot decomposition
```

## Integration with CoarseGrainingEnergyFluxes.jl

```julia
using HelmholtzDecomposition: HelmholtzDecomposition
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes
using FFTW: FFTW  # or using NUFSHT: NUFSHT for spherical

# Step 1: Helmholtz decomposition
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

# Step 2: Filter the scalar potentials
# (CoarseGrainingEnergyFluxes provides the filtering machinery)
ψ_bar = filter_scalar(HelmholtzDecomposition.streamfunction(result), grid, ℓ)
χ_bar = filter_scalar(result.χ, grid, ℓ)

# Step 3: Compute energy flux from filtered Helmholtz scalars
# This gives Π_tor, Π_pot and cross-terms (Buzzicotti 2023)
```

## When to Skip This

Per Storer et al. (2022): if your velocity is **already non-divergent** (e.g., geostrophic velocity derived from SSH), the commutativity issue does not apply. The planetary Cartesian approach is correct in that case.

Only use the full Helmholtz workflow when:
- Velocity has both rotational AND divergent components
- You need to decompose Π into toroidal/potential contributions
- You need guaranteed commutativity for general velocity fields
