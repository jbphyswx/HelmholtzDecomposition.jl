# The transform solvers and the order `AutoSolver` tries them in. Each solver type is declared here and
# implemented by the extension of the library it transforms with, which adds its `prepare_solver`,
# `solve_poisson!`, `_decompose_spectral` and capability methods.

"""
    CartesianSpectralSolver()

Spectral Poisson solver for uniform periodic Cartesian grids in any dimension, through the real FFT of
the `AbstractFFTs` interface (`using FFTW` on the host, CUDA's CUFFT for a `CuArray`): forward `rfft`,
divide by the discrete Laplacian's symbol, inverse `irfft`.
"""
struct CartesianSpectralSolver <: AbstractPoissonSolver end

"""
    CartesianBoundedSolver()

Direct `O(N log N)` Poisson solve on a uniform Cartesian grid with any mix of bounded and periodic
directions, a channel included, through FFTW's real-to-real transforms (`using FFTW`).

Each direction takes the transform that diagonalizes its Laplacian: DCT-II/III
(`REDFT10`/`REDFT01`) the cell-centred Neumann one, DST-II/III (`RODFT10`/`RODFT01`) the Dirichlet
one, and the halfcomplex FFT a periodic direction. The cosine and sine transforms are FFTs of the
even and odd extensions, so a bounded domain costs the `O(N log N)` of a periodic one.

The eigenvalues are the discrete operator's, so this inverts exactly the `L = D G` the
decomposition differentiates with:

| condition | kinds | eigenvalue |
|---|---|---|
| Neumann   | `REDFT10`/`REDFT01` | `−(4/h²)·sin²(πk/2N)`, `k = 0…N−1`; `λ₀ = 0` |
| Dirichlet | `RODFT10`/`RODFT01` | `−(4/h²)·sin²(π(k+1)/2N)`, with no null mode |
"""
struct CartesianBoundedSolver <: AbstractPoissonSolver end

"""
    CartesianRealTransformSolver()

Direct `O(N log N)` Poisson solve on a uniform Cartesian grid with any mix of bounded and periodic
directions, through the `AbstractFFTs` interface alone (`using AbstractFFTs` and an FFT backend
for the array), so it follows the array to whichever backend owns it.
"""
struct CartesianRealTransformSolver <: AbstractPoissonSolver end

"""
    SphericalSpectralSolver()

Spectral Poisson solver on FastSphericalHarmonics' Clenshaw–Curtis grid
(`using FastSphericalHarmonics`): forward SHT, divide by `−ℓ(ℓ+1)/R²`, inverse SHT.
"""
struct SphericalSpectralSolver <: AbstractPoissonSolver end

"""
    SphericalNUSHTSolver(; lmax = nothing, tol = 1e-8, rtol = 1e-10, maxiter = 500,
                         nufft = SpectralBackends.AutoSpectralBackend())

Spectral Poisson solver for an arbitrary spherical node set covering `S²` (`using NUFSHT`).
`rtol`/`maxiter` govern the least squares inside the exact inverse transform, which weights each node
by its cell's measure; `tol` is the NUFFT accuracy, and `nufft` the NUFFT library NUFSHT runs: a
FlowTransformBindings tag, or `AutoSpectralBackend()` for NUFSHT's own choice. NUFSHT's plans are built
under the execution backend of the call, a `GPUBackend`'s device included, given a NUFFT library for
that device's arrays.

`lmax = nothing` sizes the expansion from the grid: the degree a grid of latitude rings or with a
sampling recipe represents, and on scattered nodes the largest degree leaving four nodes per
coefficient, where the fit is well conditioned.

A degree that outruns the nodes by enough corrupts the split while leaving the fit exact: the
samples are reproduced, the three parts sum back to the input, and the rotational/divergent share
is wrong, because that share reads the coefficient vector and the samples stop pinning it down.
The residual is at round-off throughout, so it reports nothing about this. Where that happens
depends on the field and the node layout.
"""
struct SphericalNUSHTSolver{T<:AbstractFloat,L<:Union{Nothing,Int},N<:SpectralBackends.AbstractSpectralBackend} <:
       AbstractPoissonSolver
    lmax::L
    tol::T
    rtol::T
    maxiter::Int
    nufft::N
end

SphericalNUSHTSolver(; lmax::Union{Nothing,Int} = nothing, tol::AbstractFloat = 1e-8,
                     rtol::AbstractFloat = 1e-10, maxiter::Int = 500,
                     nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend()) =
    SphericalNUSHTSolver(lmax, promote(tol, rtol)..., maxiter, nufft)

library_loaded(::CartesianBoundedSolver) = _extension_loaded(:HelmholtzDecompositionFFTWExt)
library_loaded(::Union{CartesianSpectralSolver,CartesianRealTransformSolver}) =
    _extension_loaded(:HelmholtzDecompositionAbstractFFTsExt)
library_loaded(::SphericalSpectralSolver) = _extension_loaded(:HelmholtzDecompositionFSHExt)
library_loaded(::SphericalNUSHTSolver) = _extension_loaded(:HelmholtzDecompositionNUFSHTExt)

_unavailable_message(::CartesianBoundedSolver) =
    "CartesianBoundedSolver transforms with FFTW; run `using FFTW`."
_unavailable_message(s::Union{CartesianSpectralSolver,CartesianRealTransformSolver}) =
    "$(nameof(typeof(s))) transforms through AbstractFFTs; load an FFT library for the array " *
    "(`using FFTW` on the host, CUDA for a `CuArray`)."
_unavailable_message(::SphericalSpectralSolver) =
    "SphericalSpectralSolver transforms with FastSphericalHarmonics; run " *
    "`using FastSphericalHarmonics`."
_unavailable_message(::SphericalNUSHTSolver) =
    "SphericalNUSHTSolver transforms with NUFSHT; run `using NUFSHT`."

const _CartesianGrid =
    FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}
const _SphericalGrid =
    FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}

# The Cartesian transforms divide by the discrete operator's symbol. The spherical ones divide by
# `−ℓ(ℓ+1)/R²`, the continuous Laplacian's eigenvalue, and the scattered fit has no `L` at all.
inverts_discrete_laplacian(::Union{CartesianSpectralSolver,CartesianBoundedSolver,
                                   CartesianRealTransformSolver}) = true

# The solvers with a `helmholtz_decompose_spectral` route.
has_spectral_route(::AbstractPoissonSolver) = false
has_spectral_route(::Union{CartesianSpectralSolver,CartesianNUFFTSolver,SphericalSpectralSolver,
                           SphericalNUSHTSolver}) = true

const _SphericalTransform = Union{SphericalSpectralSolver,SphericalNUSHTSolver}
preconditions(::_SphericalTransform) = true
# FastSphericalHarmonics and FFTW's real-to-real transforms hold their state in host memory.
supports_backend(::Union{SphericalSpectralSolver,CartesianBoundedSolver}, backend, grid) =
    backend isa _HostBackend

"""
    _auto_candidates(grid) -> Tuple

The transform solvers [`AutoSolver`](@ref) tries for the staggered decomposition on `grid`, in
order: on a Cartesian grid the periodic FFT, FFTW's bounded transform, then the bounded transform
through any `AbstractFFTs` backend. Each inverts the discrete `L`. The first whose library is loaded
and whose requirements the grid and the execution backend meet is taken, and the iterative
[`CGSolver`](@ref) otherwise.
"""
_auto_candidates(::_CartesianGrid) =
    (CartesianSpectralSolver(), CartesianBoundedSolver(), CartesianRealTransformSolver())
_auto_candidates(::FlowGeometries.Grids.AbstractGrid) = ()

"""
    _spectral_candidates(grid) -> Tuple

The solvers [`helmholtz_decompose_spectral`](@ref) tries under `AutoSolver`, in order. Cartesian:
the periodic FFT, then the scattered NUFFT. Spherical: FastSphericalHarmonics, then NUFSHT.
"""
_spectral_candidates(grid::_CartesianGrid) = (CartesianSpectralSolver(), _auto_nufft_solver(grid))
_spectral_candidates(::_SphericalGrid) = (SphericalSpectralSolver(), SphericalNUSHTSolver())
_spectral_candidates(::FlowGeometries.Grids.AbstractGrid) = ()
