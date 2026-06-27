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

## The full Helmholtz–Hodge decomposition

On a domain Ω ⊂ ℝⁿ, a vector field admits the L²-orthogonal decomposition

```
u  =  ∇χ  ⊕  rot R  ⊕  h
```

into a **curl-free** part `∇χ` (divergent), a **divergence-free** part `rot R` (rotational),
and a **harmonic** part `h` that is *both* divergence- and curl-free. On a periodic torus or
an unbounded domain with decay, `dim` of the harmonic space is 0 (only the mean, which the
solvers remove). On a **bounded or multiply-connected** domain the harmonic space has
dimension equal to the first Betti number `b₁` of the domain (the number of holes/islands):
`h` then carries the net circulation around each hole and the net flux through it, which no
single-valued `ψ`/`χ` can represent.

The decomposition is **not unique on a bounded domain** without boundary conditions. The
"natural HHD" (Bhatia et al. 2013) fixes uniqueness and orthogonality by assigning the
normal trace to `∇χ` (Neumann `∂χ/∂n = u·n̂`) and the tangential trace to `rot R`
(Dirichlet `ψ = const` per boundary component). This package computes `h` as the residual
`h = u − ∇χ − rot R` after the potential solves (`harmonic_fraction = ‖h‖/‖u‖` is reported),
and detects topology with `count_holes` / `betti1_estimate`.

### N-dimensional potentials

The 3-D vector potential generalizes to an **antisymmetric rotation-potential matrix** `R`
with `N(N−1)/2` independent components (Glötzl & Richters 2023). The governing equations are
component-wise Poisson problems,

```
Δχ      = ∇·u                              (gradient/scalar potential)
ΔR_ab   = ∂_a u_b − ∂_b u_a   (a < b)      (rotation potential)
```

with the identity `grad div u + ROT ROT̄ u = Δu` generalizing `grad div − curl curl = Δ`.
Per dimension: **1-D** → pure gradient (no rotation); **2-D** → one component `R₁₂ = ψ`
(streamfunction); **3-D** → three components, the Hodge dual of the vector potential `A`
(`u_rot = ∇×A`); **N ≥ 4** → only `χ` and the matrix `R` are well-defined (no `ψ`, no `A`).
In Fourier space the velocity split is dimension-trivial — the **Leray projection**
`û_div = (k̂⊗k̂)û`, `û_rot = (I − k̂⊗k̂)û` — which is the fast path used by the spectral solvers.

## The 2D special case

Any 2D vector field can be written `u = u_rot + u_div` (+ harmonic), with
`u_rot = ∇×(ψ ẑ)` non-divergent and `u_div = ∇χ` irrotational.

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
- Glötzl, E., & Richters, O. (2023). Helmholtz decomposition and potential functions for n-dimensional analytic vector fields. *Journal of Mathematical Analysis and Applications*, 525(2), 127138. doi:10.1016/j.jmaa.2023.127138
- Bhatia, H., Norgard, G., Pascucci, V., & Bremer, P.-T. (2013). The Helmholtz-Hodge Decomposition — A Survey. *IEEE TVCG*, 19(8), 1386–1404. doi:10.1109/TVCG.2012.316
- Buzzicotti, M., Storer, B. A., Khatri, H., Griffies, S. M., & Aluie, H. (2023). Spatio-temporal coarse-graining decomposition of the global ocean geostrophic kinetic energy. *Science Advances*, 9(45). doi:10.1126/sciadv.adi7420
- Storer, B. A., Buzzicotti, M., Khatri, H., Griffies, S. M., & Aluie, H. (2022). Global energy spectrum of the general oceanic circulation. *Nature Communications*, 13, 5314. doi:10.1038/s41467-022-33031-3
