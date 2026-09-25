"""
    SpectralSolvers.jl — the transform solvers and the order `AutoSolver` tries them in.

Each solver type is declared here and implemented by the extension of the library it transforms
with, which adds its `prepare_solver`, `solve_poisson!`, `_decompose_spectral` and capability
methods. A solver whose library is not loaded reports so through [`library_loaded`](@ref).
"""

"""
    CartesianSpectralSolver()

Spectral Poisson solver for uniform periodic Cartesian grids in any dimension, through the real
FFT (`using FFTW`): forward `rfft`, divide by the discrete Laplacian's symbol, inverse `irfft`.
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
`rtol`/`maxiter` govern the least squares inside the exact inverse transform, `tol` is the NUFFT
accuracy, and `nufft` the NUFFT library NUFSHT runs: a FlowTransformBindings tag, or
`AutoSpectralBackend()` for NUFSHT's own choice.

`lmax = nothing` sizes the expansion from the grid. The fit recovers `(lmax+1)²` coefficients from
`M` nodes, so a degree chosen without reference to `M` truncates a fine grid and outruns a coarse
one.

A degree that outruns the nodes by enough corrupts the split while leaving the fit exact: the
samples are reproduced, the three parts sum back to the input, and the rotational/divergent share
is wrong, because that share reads the coefficient vector and the samples stop pinning it down.
The residual is at round-off throughout, so it reports nothing about this. On a lat-lon grid the
split holds to round-off well past `(lmax+1)² = M` and degrades beyond roughly twice it; the
crossing depends on the field and the node layout, so `lmax` is left to the caller and sized from
the grid when it is unset.
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

library_loaded(::Union{CartesianSpectralSolver,CartesianBoundedSolver}) =
    _extension_loaded(:HelmholtzDecompositionFFTWExt)
library_loaded(::CartesianRealTransformSolver) =
    _extension_loaded(:HelmholtzDecompositionAbstractFFTsExt)
library_loaded(::SphericalSpectralSolver) = _extension_loaded(:HelmholtzDecompositionFSHExt)
library_loaded(::SphericalNUSHTSolver) = _extension_loaded(:HelmholtzDecompositionNUFSHTExt)

_unavailable_message(s::Union{CartesianSpectralSolver,CartesianBoundedSolver}) =
    "$(nameof(typeof(s))) transforms with FFTW; run `using FFTW`."
_unavailable_message(::CartesianRealTransformSolver) =
    "CartesianRealTransformSolver transforms through AbstractFFTs; run `using AbstractFFTs` and " *
    "load an FFT backend for the array (`using FFTW` on the host)."
_unavailable_message(::SphericalSpectralSolver) =
    "SphericalSpectralSolver transforms with FastSphericalHarmonics; run " *
    "`using FastSphericalHarmonics`."
_unavailable_message(::SphericalNUSHTSolver) =
    "SphericalNUSHTSolver transforms with NUFSHT; run `using NUFSHT`."

const _CartesianGrid =
    FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}
const _SphericalGrid =
    FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}

"""
    _auto_candidates(grid) -> Tuple

The solvers [`AutoSolver`](@ref) tries on `grid`, in order. On a Cartesian grid: the periodic
FFT, FFTW's bounded transform, the bounded transform through any `AbstractFFTs` backend, then the
scattered NUFFT. On a spherical grid: FastSphericalHarmonics, then NUFSHT. The first whose library
is loaded and whose requirements the grid meets is taken, and the iterative [`CGSolver`](@ref)
otherwise.
"""
_auto_candidates(grid::_CartesianGrid) =
    (CartesianSpectralSolver(), CartesianBoundedSolver(), CartesianRealTransformSolver(),
     _auto_nufft_solver(grid))
_auto_candidates(::_SphericalGrid) = (SphericalSpectralSolver(), SphericalNUSHTSolver())
_auto_candidates(::FlowGeometries.Grids.AbstractGrid) = ()
