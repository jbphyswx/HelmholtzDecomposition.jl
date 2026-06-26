"""
    Solvers.jl — Poisson solver interface and base SOR implementation.

Provides the abstract `AbstractPoissonSolver` interface and a pure-Julia Red-Black
Successive Over-Relaxation (SOR) solver as the base fallback. The Cartesian SOR is
dimension-generic (1D/2D/3D/ND); the spherical SOR is the 2-surface `N == 2` case with
the spherical metric, periodic longitude, and pole guarding.

Spectral solvers (FFTW, FINUFFT, FastSphericalHarmonics, NUFSHT) are provided via package
extensions and are **strongly recommended** for performance. They assume *periodic* (whole
torus / whole sphere) boundary conditions and an unmasked domain; `AutoSolver` therefore
falls back to SOR whenever a land mask is present.

See the README "Solver Extensions" section for setup instructions.
"""

export AbstractPoissonSolver, SORSolver, AutoSolver
export solve_poisson!, register_spectral_solver!

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

"""
    AbstractPoissonSolver

Abstract supertype for all Poisson solvers. Concrete subtypes implement

    solve_poisson!(Φ, RHS, grid, solver; boundary=nothing, kwargs...) -> SolverResult

# Available solvers
- `SORSolver` — Red-Black SOR (base package, no external deps, dimension-generic).
- `CartesianSpectralSolver` / `CartesianNUFFTSolver` — via FFTW / FINUFFT extensions.
- `SphericalSpectralSolver` / `SphericalNUSHTSolver` — via FastSphericalHarmonics / NUFSHT.
"""
abstract type AbstractPoissonSolver end

"""
    SolverResult{T}

Convergence diagnostics from a Poisson solve.

# Fields
- `converged::Bool` — whether the solve met its tolerance.
- `iterations::Int` — iterations performed (1 for direct/spectral solves).
- `residual::T` — final residual (max absolute update for SOR).
"""
struct SolverResult{T<:AbstractFloat}
    converged::Bool
    iterations::Int
    residual::T
end

# ---------------------------------------------------------------------------
# Auto solver selection
# ---------------------------------------------------------------------------

"""
    AutoSolver()

Sentinel for automatic solver selection. `solve_poisson!` chooses the best available
solver based on grid geometry, the presence of a land mask, and which extension packages
are loaded. Falls back to `SORSolver()` when no spectral extension applies (or a mask is
present), emitting a `@debug` recommendation.
"""
struct AutoSolver <: AbstractPoissonSolver end

# Registry of available spectral solvers (populated by extensions).
const _SPECTRAL_SOLVERS = Dict{Symbol,Type{<:AbstractPoissonSolver}}()

"""
    register_spectral_solver!(key::Symbol, solver_type)

Called by extensions to register their spectral solver. Keys:
`:cartesian_regular`, `:cartesian_irregular`, `:spherical_regular`, `:spherical_irregular`.
"""
function register_spectral_solver!(key::Symbol, solver_type::Type{<:AbstractPoissonSolver})
    _SPECTRAL_SOLVERS[key] = solver_type
    return nothing
end

"""
    _resolve_auto_solver(grid) -> AbstractPoissonSolver

Pick a concrete solver. Spectral solvers are selected only when the domain is fully
active (`all(mask)`), because they assume periodic/whole-domain boundary conditions and
cannot honor a land mask. With a mask present, fall back to SOR.
"""
function _resolve_auto_solver(grid::StructuredGrid{N,G}) where {N,G<:AbstractGeometry}
    if all(grid.mask)
        if G <: SphericalGeometry
            for key in (:spherical_irregular, :spherical_regular)
                haskey(_SPECTRAL_SOLVERS, key) && return _SPECTRAL_SOLVERS[key]()
            end
        elseif G <: CartesianGeometry
            for key in (:cartesian_irregular, :cartesian_regular)
                haskey(_SPECTRAL_SOLVERS, key) && return _SPECTRAL_SOLVERS[key]()
            end
        end
    end

    if !all(grid.mask)
        @debug "Land mask present — using SORSolver (spectral solvers assume an unmasked, periodic domain)."
    else
        @debug """No spectral solver extension loaded — falling back to SORSolver which may be \
        orders of magnitude slower. Load a spectral extension for your geometry: \
        `using FFTW` (Cartesian regular), `using FINUFFT` (Cartesian irregular), \
        `using FastSphericalHarmonics` (spherical regular), or `using NUFSHT` (spherical irregular)."""
    end
    return SORSolver()
end

function solve_poisson!(
    Φ::AbstractArray{T,N},
    RHS::AbstractArray{T,N},
    grid::StructuredGrid{N},
    ::AutoSolver;
    kwargs...,
) where {T<:AbstractFloat,N}
    return solve_poisson!(Φ, RHS, grid, _resolve_auto_solver(grid); kwargs...)
end

# ---------------------------------------------------------------------------
# SOR solver
# ---------------------------------------------------------------------------

"""
    SORSolver(; max_iter=10_000, tol=1e-6, ω=1.85, boundary=:neumann)

Red-Black Successive Over-Relaxation solver for the Poisson equation `∇²Φ = RHS`.
Dimension-generic for Cartesian grids; `N == 2` for spherical grids. Works on any masked
grid but has `O(N)` convergence per sweep and may be **orders of magnitude slower** than
the spectral extensions for large grids.

# Keyword Arguments
- `max_iter::Int` — maximum sweeps (default `10_000`).
- `tol::Float64` — tolerance on the max absolute update (default `1e-6`).
- `ω::Float64` — relaxation factor, `1 < ω < 2` (default `1.85`).
- `boundary::Symbol` — `:neumann` (zero normal gradient) or `:dirichlet` (zero value).
"""
struct SORSolver <: AbstractPoissonSolver
    max_iter::Int
    tol::Float64
    ω::Float64
    boundary::Symbol
end

SORSolver(; max_iter::Int = 10_000, tol::Float64 = 1e-6, ω::Float64 = 1.85, boundary::Symbol = :neumann) =
    SORSolver(max_iter, tol, ω, boundary)

@inline _effective_boundary(solver::SORSolver, boundary) = boundary === nothing ? solver.boundary : boundary

# --- Cartesian (dimension-generic) -----------------------------------------

"""
    solve_poisson!(Φ, RHS, grid::StructuredGrid{N,<:CartesianGeometry}, solver::SORSolver; boundary=nothing)

Solve `∇²Φ = RHS` on the wet cells of an `N`-dimensional Cartesian grid via Red-Black SOR.
"""
function solve_poisson!(
    Φ::AbstractArray{T,N},
    RHS::AbstractArray{T,N},
    grid::StructuredGrid{N,<:CartesianGeometry{N,T}},
    solver::SORSolver;
    boundary::Union{Nothing,Symbol} = nothing,
    kwargs...,
) where {T<:AbstractFloat,N}
    spacing = grid.geometry.spacing
    inv_h2 = ntuple(d -> one(T) / spacing[d]^2, Val(N))
    denom = T(2) * sum(inv_h2)
    ω = T(solver.ω)
    tol = T(solver.tol)
    bc = _effective_boundary(solver, boundary)
    dirichlet = bc === :dirichlet

    fill!(Φ, zero(T))
    final_iter = 0
    final_diff = zero(T)

    for iter in 1:solver.max_iter
        max_diff = zero(T)
        final_iter = iter
        for color in 0:1
            @inbounds for I in CartesianIndices(grid.mask)
                (sum(Tuple(I)) % 2) == color || continue
                grid.mask[I] || continue
                acc = zero(T)
                for d in 1:N
                    e = _unit(Val(N), d)
                    Jp = _wet_neighbor(grid, I, e, +1)
                    Jm = _wet_neighbor(grid, I, e, -1)
                    Φp = Jp == I ? (dirichlet ? zero(T) : Φ[I]) : Φ[Jp]
                    Φm = Jm == I ? (dirichlet ? zero(T) : Φ[I]) : Φ[Jm]
                    acc += inv_h2[d] * (Φp + Φm)
                end
                Φ_new = (acc - RHS[I]) / denom
                diff = Φ_new - Φ[I]
                Φ[I] += ω * diff
                max_diff = max(max_diff, abs(diff))
            end
        end
        final_diff = max_diff
        max_diff < tol && return SolverResult{T}(true, iter, max_diff)
    end
    return SolverResult{T}(false, final_iter, final_diff)
end

# --- Spherical (N == 2, periodic longitude, guarded poles) ------------------

"""
    solve_poisson!(Φ, RHS, grid::StructuredGrid{2,<:SphericalGeometry}, solver::SORSolver; boundary=nothing)

Solve the spherical Poisson equation `∇²Φ = RHS` via Red-Black SOR with the spherical
metric. Longitude is treated as periodic; latitude uses the configured boundary condition.
Cells within `pole_tol` of the poles (where `cosφ → 0`) are skipped to avoid the metric
singularity — for pole-accurate work use a spectral spherical extension.
"""
function solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::StructuredGrid{2,<:SphericalGeometry{T}},
    solver::SORSolver;
    boundary::Union{Nothing,Symbol} = nothing,
    pole_tol::Real = sqrt(eps(T)),
    kwargs...,
) where {T<:AbstractFloat}
    Nlon, Nlat = size_tuple(grid)
    lon, lat = grid.coords_axes
    R = grid.geometry.R
    dλ = Nlon > 1 ? lon[2] - lon[1] : one(T)
    dφ = Nlat > 1 ? lat[2] - lat[1] : one(T)
    inv_dλ2 = one(T) / dλ^2
    inv_dφ2 = one(T) / dφ^2
    ω = T(solver.ω)
    tol = T(solver.tol)
    bc = _effective_boundary(solver, boundary)
    dirichlet = bc === :dirichlet
    periodic_lon = _is_periodic_longitude(lon, dλ)

    fill!(Φ, zero(T))
    final_iter = 0
    final_diff = zero(T)

    for iter in 1:solver.max_iter
        max_diff = zero(T)
        final_iter = iter
        for color in 0:1
            @inbounds for j in 1:Nlat
                cosφ = cos(lat[j])
                abs(cosφ) < pole_tol && continue
                sinφ = sin(lat[j])
                for i in 1:Nlon
                    ((i + j) % 2) == color || continue
                    iswet(grid, i, j) || continue

                    ip, im = _lon_neighbors(i, Nlon, grid, j, periodic_lon)
                    jp = j < Nlat && iswet(grid, i, j + 1) ? j + 1 : j
                    jm = j > 1 && iswet(grid, i, j - 1) ? j - 1 : j

                    Φ_ip = ip == i && !periodic_lon ? (dirichlet ? zero(T) : Φ[i, j]) : Φ[ip, j]
                    Φ_im = im == i && !periodic_lon ? (dirichlet ? zero(T) : Φ[i, j]) : Φ[im, j]
                    Φ_jp = jp == j ? (dirichlet ? zero(T) : Φ[i, j]) : Φ[i, jp]
                    Φ_jm = jm == j ? (dirichlet ? zero(T) : Φ[i, j]) : Φ[i, jm]

                    term_λ = (Φ_ip + Φ_im) * inv_dλ2 / (R^2 * cosφ^2)
                    term_φ = ((Φ_jp + Φ_jm) * inv_dφ2 - sinφ * (Φ_jp - Φ_jm) / (T(2) * dφ * cosφ)) / R^2
                    denom_sph = T(2) * inv_dλ2 / (R^2 * cosφ^2) + T(2) * inv_dφ2 / R^2

                    Φ_new = (term_λ + term_φ - RHS[i, j]) / denom_sph
                    diff = Φ_new - Φ[i, j]
                    Φ[i, j] += ω * diff
                    max_diff = max(max_diff, abs(diff))
                end
            end
        end
        final_diff = max_diff
        max_diff < tol && return SolverResult{T}(true, iter, max_diff)
    end
    return SolverResult{T}(false, final_iter, final_diff)
end

# Longitude is periodic when the axis spans ~2π (global grid).
@inline function _is_periodic_longitude(lon::AbstractVector{T}, dλ::T) where {T}
    length(lon) > 2 || return false
    span = lon[end] - lon[1] + dλ
    return isapprox(span, T(2π); rtol = T(1e-3))
end

@inline function _lon_neighbors(i::Int, Nlon::Int, grid, j::Int, periodic::Bool)
    if periodic
        ip = i == Nlon ? 1 : i + 1
        im = i == 1 ? Nlon : i - 1
        ip = iswet(grid, ip, j) ? ip : i
        im = iswet(grid, im, j) ? im : i
        return ip, im
    else
        ip = i < Nlon && iswet(grid, i + 1, j) ? i + 1 : i
        im = i > 1 && iswet(grid, i - 1, j) ? i - 1 : i
        return ip, im
    end
end
