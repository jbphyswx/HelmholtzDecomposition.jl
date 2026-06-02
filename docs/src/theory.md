# Theory

## The Commutativity Problem on the Sphere

Aluie (2019, doi:10.1007/s13137-019-0123-9) proves that on the sphere S², filtering vector fields by converting to Cartesian components and filtering each as a scalar does **NOT** commute with differential operators:

```
G * (∇·u) ≠ ∇·(G * u)    on S²
G * (∇×u) ≠ ∇×(G * u)    on S²
```

This means that filtering a divergence-free velocity field via Cartesian components does NOT produce a divergence-free filtered field.

## The Helmholtz Solution

Aluie (2019, Section 7, Proposition 2) proves that the generalized convolution that DOES commute with derivatives is mathematically equivalent to:

1. Decomposing u into scalar potentials: u → (ψ, χ, u_r)
2. Filtering each scalar separately: ψ̄, χ̄, ū_r
3. Reconstructing velocity from filtered potentials

This is the **Edmonds transformation** (Edmonds 1960).

## The Helmholtz Decomposition

Any 2D vector field on a surface can be decomposed as:

```
u = u_rot + u_div
```

where u_rot = ∇×(ψ ẑ) is non-divergent and u_div = ∇χ is irrotational.

### Cartesian Geometry

Given vorticity ζ = ∂v/∂x − ∂u/∂y and divergence δ = ∂u/∂x + ∂v/∂y:

- Solve ∇²ψ = ζ
- Solve ∇²χ = δ
- u_rot = −∂ψ/∂y, v_rot = ∂ψ/∂x
- u_div = ∂χ/∂x, v_div = ∂χ/∂y

### Spherical Geometry

On a sphere of radius R with coordinates (λ, φ):

- Solve ∇²ψ = ζ where ∇² is the spherical Laplacian
- u_rot = −(1/R) ∂ψ/∂φ
- v_rot = 1/(R cos φ) ∂ψ/∂λ

The spherical Laplacian has eigenvalues −ℓ(ℓ+1)/R² for spherical harmonic degree ℓ.

## Spectral Poisson Solvers

The Poisson equation can be solved in O(N log N) time using spectral methods:

1. **Transform** RHS to spectral space (FFT or SHT)
2. **Divide** each mode by its eigenvalue (−k² for FFT, −ℓ(ℓ+1)/R² for SHT)
3. **Transform back** to physical space

The k=0 (or ℓ=0) mode is set to zero (the mean of the solution is arbitrary).

## References

- Aluie, H. (2019). Convolutions on the sphere: commutation with differential operators. *GEM - International Journal on Geomathematics*, 10(1), 9. doi:10.1007/s13137-019-0123-9
- Buzzicotti, M., Storer, B. A., Khatri, H., Griffies, S. M., & Aluie, H. (2023). Spatio-temporal coarse-graining decomposition of the global ocean geostrophic kinetic energy. *Science Advances*, 9(45). doi:10.1126/sciadv.adi7420
- Storer, B. A., Buzzicotti, M., Khatri, H., Griffies, S. M., & Aluie, H. (2022). Global energy spectrum of the general oceanic circulation. *Nature Communications*, 13, 5314. doi:10.1038/s41467-022-33031-3
