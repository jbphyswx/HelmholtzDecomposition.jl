"""
    HelmholtzDecompositionFSHExt — spherical spectral Poisson solve via FastSphericalHarmonics.

Forward SHT → divide by the eigenvalue `−ℓ(ℓ+1)/R²` → inverse SHT.

The transform is defined on one node set and no other: the Clenshaw–Curtis grid
`θᵢ = π(i−½)/N_θ`, `λⱼ = 2π(j−1)/(2N_θ−1)` (`FastSphericalHarmonics.sph_points`). The grid's
**node positions** are checked against it, not merely its shape — `N_λ = 2N_θ − 1` is necessary
and nowhere near sufficient, and feeding an evenly-spaced latitude grid to this transform returns
a smooth, plausible, wrong field.

That check subsumes the coverage question: a Clenshaw–Curtis set spans the whole sphere. It has to
be asked, because spherical-harmonic analysis integrates over `S²` and the inverse Laplacian there
is nonlocal, so a regional patch does not determine `Φ` even on the patch. A patch is a boundary
value problem — a different problem, and the one the iterative solver handles.
"""
module HelmholtzDecompositionFSHExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using FastSphericalHarmonics: FastSphericalHarmonics as FSH

struct SphericalSpectralSolver <: HD.AbstractPoissonSolver end

# A grid this solver accepts covers the closed sphere, so there is no boundary anywhere and a
# boundary condition is vacuous rather than honoured or refused. The transform reads every node,
# so no cell may be masked out.
HD.supports_boundary(::SphericalSpectralSolver, ::HD.AbstractBoundaryCondition) = true
HD.requires_full_domain(::SphericalSpectralSolver) = true

"""
    _is_clenshaw_curtis(grid) -> Bool

Whether `grid`'s nodes are the Clenshaw–Curtis set this transform is defined on.

A grid built from a sampling recipe carries it as a zero-size type parameter, so the first test is
answered from the type. A grid built from bare axes carries `nothing`, and its nodes are compared
against the reconstructed set.

The node set is symmetric under `θ → π − θ`, so either latitude order is accepted: the reflection
is an isometry of `S²` and commutes with `Δ`.
"""
_is_clenshaw_curtis(grid) = false

function _is_clenshaw_curtis(
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
) where {T}
    FG.Grids.sampling(grid) isa FG.SphericalSampling.AbstractClenshawCurtisSampling && return true
    nlon, nlat = size(grid)
    want = FG.SphericalSampling.spherical_axes(T, FG.SphericalSampling.ClenshawCurtisSampling(), nlat)
    length(want.λ) == nlon || return false
    λ = FG.Grids.coordinates(grid, 1)
    φ = FG.Grids.coordinates(grid, 2)
    tol = sqrt(eps(T)) * T(2π)
    all(i -> abs(T(λ[i]) - want.λ[i]) <= tol, 1:nlon) || return false
    return all(j -> abs(T(φ[j]) - want.φ[j]) <= tol, 1:nlat) ||
           all(j -> abs(T(φ[j]) - want.φ[nlat - j + 1]) <= tol, 1:nlat)
end

HD.supports_sampling(::SphericalSpectralSolver, grid) = _is_clenshaw_curtis(grid)

function HD._sampling_message(::SphericalSpectralSolver, grid)
    sz = size(grid)
    nlat = length(sz) == 2 ? sz[2] : 0
    return "FastSphericalHarmonics is defined on the Clenshaw–Curtis node set, " *
           "θᵢ = π(i−½)/N_θ and λⱼ = 2π(j−1)/(2N_θ−1); this grid ($(join(sz, "×"))) is not it. " *
           "On another node set the transform returns a smooth, plausible, wrong field. Build the " *
           "grid from `FlowGeometries.SphericalSampling.spherical_axes(T, " *
           "ClenshawCurtisSampling(), $nlat)`, load `NUFSHT` for an arbitrary node set, or use " *
           "the iterative solver for a regional patch."
end

"""
    SphericalSpectralState

The transposed coefficient array and the per-mode eigenvalues, built once.

`FastSphericalHarmonics` wants `(nlat, nlon)` where the grid is `(nlon, nlat)`, so a solve has to
transpose in and out. Allocating that array inside `solve_poisson!` allocates it once per potential
per field of a batch.
"""
struct SphericalSpectralState{T,A<:AbstractMatrix{T},E<:AbstractVector{T},C}
    coeffs::A
    inv_eig::E      # 1/λ_ℓ per ℓ, with ℓ = 0 zeroed
    cache::C        # this task's own FastTransforms plans
end

"""
    _PLANNER_LOCK

Held while FastTransforms builds a plan.

Its sphere plans are built by `fftw_plan_many_r2r` inside the bundled FFTW, whose planner keeps one
process-global table and is not thread safe; two tasks planning at once abort in `malloc`. FFTW.jl
serialises its own planning, and a C library calling the planner directly sits outside that.

Execution needs no lock **when each task owns its plans**. Two tasks executing one shared plan
corrupt each other through its internal scratch, so the cache lives on the per-task state.
"""
const _PLANNER_LOCK = ReentrantLock()

function HD.prepare_solver(::SphericalSpectralSolver,
                           grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
                           ::HD.AbstractBoundaryCondition;
                           backend = CB.SerialBackend(), shared = nothing) where {T}
    # FastSphericalHarmonics wraps FastTransforms, which is a host library, so this solver's state
    # is host memory. A backend that allocates elsewhere is refused here, where the message can
    # name an alternative, ahead of a `MethodError` raised from inside the transform.
    HD.allocate_zeros(backend, T, (1,)) isa Array || throw(ArgumentError(
        "SphericalSpectralSolver runs on the host: FastSphericalHarmonics has no device " *
        "implementation. Use `SphericalNUSHTSolver`, whose non-uniform backend follows the array, " *
        "or plan this grid with a host backend."))
    nlon, nlat = size(grid)
    lmax = nlat - 1
    R = FG.Geometry.radius(FG.Grids.grid_geometry(grid))
    coeffs = zeros(T, nlat, nlon)
    inv_eig = zeros(T, lmax + 1)
    # ℓ = 0 is the constant the Laplacian annihilates, and stays zero.
    @inbounds for ℓ in 1:lmax
        inv_eig[ℓ + 1] = -R^2 / T(ℓ * (ℓ + 1))
    end
    # Both plans are built here, on this task, so a solve executes and plans nothing. The warm-up
    # runs one transform and one evaluation over `coeffs`, which is left zeroed for the first solve.
    cache = FSH.SphPlanCache{T}()
    Base.lock(_PLANNER_LOCK) do
        FSH.sph_transform!(coeffs; cache = cache)
        FSH.sph_evaluate!(coeffs; cache = cache)
    end
    fill!(coeffs, zero(T))
    return SphericalSpectralState(coeffs, inv_eig, cache)
end

function HD.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
    solver::SphericalSpectralSolver;
    boundary::HD.AbstractBoundaryCondition = HD.Neumann(),
    state::SphericalSpectralState = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat}
    nlon, nlat = size(grid)
    lmax = nlat - 1

    C = state.coeffs
    @inbounds for j in 1:nlat, i in 1:nlon
        C[j, i] = RHS[i, j]
    end
    FSH.sph_transform!(C; cache = state.cache)
    @inbounds for ℓ in 1:lmax
        w = state.inv_eig[ℓ + 1]
        for m in (-ℓ):ℓ
            C[FSH.sph_mode(ℓ, m)] *= w
        end
    end
    # ℓ = 0 is the constant, which the Laplacian annihilates: the solve is defined up to it.
    C[FSH.sph_mode(0, 0)] = zero(T)
    FSH.sph_evaluate!(C; cache = state.cache)
    @inbounds for j in 1:nlat, i in 1:nlon
        Φ[i, j] = C[j, i]
    end
    return HD.SolverResult{T}(true, 1, zero(T))
end

# ---------------------------------------------------------------------------
# Vector Hodge decomposition on the Clenshaw–Curtis grid
# ---------------------------------------------------------------------------
#
# `ð̄` lowers spin weight and, on a spin-1 tangent field, is the divergence; `ð` raises it and, on a
# real spin-0 field, is the gradient. So `u_div = ∇Δ⁻¹(∇·u)` is three coefficient-space steps with
# no stencil, and `u_rot = u − u_div` is exact: `S²` is simply connected, so a tangent field has no
# harmonic part.
#
# The radius cancels. Both `∇·` and `∇` carry one factor of `1/R` and `Δ` carries `1/R²`, so the
# split of the field is the same on a sphere of any radius; only the potentials carry `R`.

# FSH's grid runs colatitude ascending, so its latitudes descend. A grid built with ascending
# latitude is the same node set read the other way round, and its rows are reversed here.
@inline _fsh_row(j::Int, nlat::Int, flip::Bool) = flip ? nlat - j + 1 : j

# Column `2|m| + (m ≥ 0)` of a coefficient array holds one `m`, so an even column is `m < 0`.
# `spinsph_eth` and `spinsph_ethbar` negate those columns when they convert between the raw complex
# packing and the real one, and the projection below is stated in the real packing.
@inline _spin_phase(col::Int) = iseven(col) ? -1.0 : 1.0

"""
    _invert_and_project!(S) -> S

Turn the spin-0 coefficients of `∇·u` into those of the velocity potential `χ`, in place.

`Δ⁻¹` on spin 0 is `−1/(ℓ(ℓ+1))`, and `ℓ = 0` is the constant `Δ` annihilates. A column holds one
`|m|` and a row steps `ℓ` up from it, so `ℓ = row + |m| − 1` reaches every stored mode, including
those above `nlat − 1` the transform resolves at high `|m|`.

`V = ð(χ + iψ)` for real `χ`, `ψ`, so `Δ⁻¹ð̄V` is the complex potential and the velocity potential
is its real part. Realness of a spin-0 field is `c₋ₘ = conj(c₊ₘ)` in this storage, so averaging each
`±m` pair against its partner's conjugate is that projection — the one step here that is not
diagonal in `(ℓ, m)`, and the whole of the Hodge split.
"""
function _invert_and_project!(S::AbstractMatrix{ComplexF64})
    nrow = size(S, 1)
    @inbounds for row in 1:nrow
        l = row - 1                             # column 1 is m = 0
        S[row, 1] = iszero(l) ? zero(ComplexF64) : complex(-real(S[row, 1]) / (l * (l + 1)))
    end
    @inbounds for col in 2:2:size(S, 2)
        am = col ÷ 2                            # columns `2|m|` and `2|m|+1` hold ∓|m|
        αlo, αhi = _spin_phase(col), _spin_phase(col + 1)
        for row in 1:nrow
            l = row + am - 1                    # `am ≥ 1` here, so `ℓ ≥ 1`
            w = -1.0 / (l * (l + 1))
            a = αlo * w * S[row, col]
            b = αhi * w * S[row, col + 1]
            p = (a + conj(b)) / 2
            S[row, col] = αlo * p
            S[row, col + 1] = αhi * conj(p)
        end
    end
    return S
end

"""
    _decompose_spectral(solver, geometry, U, grid)

Helmholtz decomposition of a tangent velocity field on the Clenshaw–Curtis grid, by spin-weighted
quadrature. `U` is component-last carrying `(u_east, u_north)`.
"""
function HD._decompose_spectral(
    ::SphericalSpectralSolver,
    ::FG.Geometry.AbstractSphericalGeometry,
    U::AbstractArray{Float64,3},
    grid::FG.Grids.StructuredGrid{Float64,<:FG.Geometry.AbstractSphericalGeometry,2};
    kwargs...,
)
    nlon, nlat = size(grid)
    φ = FG.Grids.coordinates(grid, 2)
    flip = nlat > 1 && φ[1] < φ[end]

    # `θ̂` points south and `φ̂` east, and FSH indexes `(nlat, nlon)` where this package uses
    # `(nlon, nlat)`. The spin-1 field is carried as the complex combination `u_θ + i u_φ`: FSH's
    # `SVector{2,Float64}` spin-1 interface requires the `m = 0` coefficient to be real, which a
    # tangent field with a zonal mean — solid-body rotation — does not satisfy.
    V = Array{ComplexF64}(undef, nlat, nlon)
    @inbounds for i in 1:nlon, j in 1:nlat
        V[_fsh_row(j, nlat, flip), i] = complex(-U[i, j, 2], U[i, j, 1])
    end

    # One cache for both transforms: `spinsph_transform!` and `spinsph_evaluate!` key on `(s, N)`
    # and share the `spinsph2fourier` plan. This entry point carries no per-task state, so the whole
    # section is held under the planner lock — the batch path is the one that runs concurrently, and
    # it owns a cache per task. See `_PLANNER_LOCK`.
    cache = FSH.SpinSphPlanCache{ComplexF64}()
    D = Base.lock(_PLANNER_LOCK) do
        C = FSH.spinsph_transform!(V, 1; cache = cache)   # spin-1 coefficients, in place over `V`
        S = FSH.spinsph_ethbar(C, 1)                      # spin-0 coefficients of ∇·u
        _invert_and_project!(S)
        return FSH.spinsph_evaluate!(FSH.spinsph_eth(S, 0), 1; cache = cache)   # ∇χ at the nodes
    end

    u_div = similar(U)
    u_rot = similar(U)
    u_harm = similar(U)
    @inbounds for i in 1:nlon, j in 1:nlat
        d = D[_fsh_row(j, nlat, flip), i]
        u_div[i, j, 1] = imag(d)     # u_east = u_φ
        u_div[i, j, 2] = -real(d)    # u_north = −u_θ
        u_rot[i, j, 1] = U[i, j, 1] - u_div[i, j, 1]
        u_rot[i, j, 2] = U[i, j, 2] - u_div[i, j, 2]
        u_harm[i, j, 1] = 0.0
        u_harm[i, j, 2] = 0.0
    end
    return (; u_rot, u_div, u_harm)
end

# FastSphericalHarmonics is built on `Float64`/`ComplexF64`; another element type is refused here
# with the alternative named, ahead of an assertion from inside the transform.
HD._decompose_spectral(
    ::SphericalSpectralSolver, ::FG.Geometry.AbstractSphericalGeometry, U::AbstractArray{T},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2}; kwargs...,
) where {T} = throw(ArgumentError(
    "FastSphericalHarmonics works in Float64; this grid carries $T. Load `NUFSHT`, whose spin " *
    "transforms follow the element type, or build the grid in Float64."))

# `FastTransforms`' thread count and its FFTW planner count are process-global C state, reached
# through the library `FastSphericalHarmonics` is built on.
const _FT = FSH.FastTransforms

# `FastTransforms.__init__` installs this count, and it is restored after a pinned section.
_ft_default_threads() = max(1, ceil(Int, Sys.CPU_THREADS / 2))

@inline function _pin_fasttransforms(n::Int)
    _FT.ft_set_num_threads(n)
    _FT.ft_fftw_plan_with_nthreads(n)
    return nothing
end

"""
    with_serial_transforms(f, ::SphericalSpectralSolver)

Run `f` with `FastTransforms` pinned to one thread, restoring the count afterwards.

Entering its OpenMP parallel region from a non-root Julia task returns a different result, at any
Julia thread count, and a batch reaches `sph_transform!` from a worker task. Pinning leaves a
root-task result bit-identical, so the count is the whole of it.
"""
function HD.with_serial_transforms(f, ::SphericalSpectralSolver)
    _pin_fasttransforms(1)
    try
        return f()
    finally
        _pin_fasttransforms(_ft_default_threads())
    end
end

HD.pin_serial_transforms(::SphericalSpectralSolver) = _pin_fasttransforms(1)

function __init__()
    HD.register_spectral_solver!(SB.FSHTSpectralBackend, SphericalSpectralSolver; priority = 10)
end

end # module
