# Examples

## Visual Results

### Taylor-Green Vortex (purely rotational)

![Taylor-Green Decomposition](../assets/taylor_green_decomposition.png)

### Vortex + Source (mixed rotational and divergent)

![Mixed Field Decomposition](../assets/mixed_field_decomposition.png)

### Point Source (purely divergent)

![Point Source Decomposition](../assets/point_source_decomposition.png)

## Cartesian: Pure Rotational Field (Taylor-Green Vortex)

```julia
using HelmholtzDecomposition: HelmholtzDecomposition
using FFTW: FFTW

N = 64; L = 1.0; dx = L / N
geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
xs = collect(range(0.0, L - dx, length=N))
ys = collect(range(0.0, L - dx, length=N))
grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

u, v, u_rot_exact, v_rot_exact, _, _ = HelmholtzDecomposition.taylor_green_vortex(grid)
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

# Verify: divergent component is near zero
@assert maximum(abs.(result.u_div)) < 1e-10
```

## Cartesian: Mixed Rotational + Divergent

```julia
u, v, _, _, _, _ = HelmholtzDecomposition.rankine_vortex_with_source(grid)
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

# Both components are non-trivial
@assert maximum(abs.(result.u_rot)) > 0.01
@assert maximum(abs.(result.u_div)) > 0.01

# Reconstruction is accurate
@assert maximum(abs.(result.u_rot .+ result.u_div .- u)) < 1e-10
```

## Spherical: Rossby Wave (Non-divergent)

```julia
geom = HelmholtzDecomposition.SphericalGeometry(6.371e6)
lons = collect(range(0.0, 2π, length=128))
lats = collect(range(-π/3, π/3, length=64))
grid = HelmholtzDecomposition.StructuredGrid(geom, lons, lats)

u, v, _, _, _, _ = HelmholtzDecomposition.rossby_wave(grid)
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)
# result.u_div should be small relative to result.u_rot
```

## Coarse-Graining Workflow

The full workflow for correct spherical filtering:

```julia
# 1. Decompose velocity into potentials
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

# 2. Filter the scalar potentials (use your filtering package)
ψ_filtered = your_filter(result.ψ, grid, filter_scale)
χ_filtered = your_filter(result.χ, grid, filter_scale)

# 3. Reconstruct velocity from filtered potentials
# (this is the step that guarantees commutativity)
u_bar, v_bar = reconstruct_from_potentials(ψ_filtered, χ_filtered, grid)
```
