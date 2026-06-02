"""
    Solvers.jl — Poisson solver interface and base SOR implementation.

Provides the abstract `AbstractPoissonSolver` interface and a pure-Julia
Red-Black Successive Over-Relaxation (SOR) solver as the base fallback.

Spectral solvers (FFTW, NUFSHT, FastSphericalHarmonics, FINUFFT) are provided
via package extensions and are **strongly recommended** for performance. The SOR
solver, while correct, may be orders of magnitude slower for large grids.

See the README "Solver Extensions" section for setup instructions.
"""

export AbstractPoissonSolver, SORSolver, AutoSolver
export solve_poisson!

# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------

"""
    AbstractPoissonSolver

Abstract supertype for all Poisson equation solvers.

Concrete subtypes must implement:
    solve_poisson!(Φ, RHS, grid, solver; kwargs...) → Φ

# Available Solvers
- `SORSolver` — Red-Black SOR (base package, no external deps, O(N²) convergence)
- `CartesianSpectralSolver` — via FFTW extension (regular) or FINUFFT extension (irregular)
- `SphericalSpectralSolver` — via FastSphericalHarmonics extension (regular) or NUFSHT extension (irregular)

Load the appropriate extension for your geometry to get O(N log N) performance.
"""
abstract type AbstractPoissonSolver end

"""
    SolverResult{T}

Result from a Poisson solve, containing convergence diagnostics.

# Fields
- `converged::Bool` — Whether the solver converged within tolerance
- `iterations::Int` — Number of iterations performed
- `residual::T` — Final residual (max absolute change for SOR)
"""
struct SolverResult{T<:AbstractFloat}
    converged::Bool
    iterations::Int
    residual::T
end

# ---------------------------------------------------------------------------
# Auto Solver Selection
# ---------------------------------------------------------------------------

"""
    AutoSolver()

Sentinel type for automatic solver selection. The `solve_poisson!` dispatch
will choose the best available solver based on:
1. Grid geometry (Cartesian vs Spherical)
2. Grid regularity (regular vs irregular)
3. Which extension packages are loaded

Falls back to `SORSolver()` if no spectral extension is loaded (with a `@debug`
message recommending the appropriate extension).
"""
struct AutoSolver <: AbstractPoissonSolver end

# Registry of available spectral solvers (populated by extensions)
const _SPECTRAL_SOLVERS = Dict{Symbol, Type{<:AbstractPoissonSolver}}()

"""
    register_spectral_solver!(key::Symbol, solver_type::Type{<:AbstractPoissonSolver})

Called by extensions to register their spectral solver as available.
Keys: `:cartesian_regular`, `:cartesian_irregular`, `:spherical_regular`, `:spherical_irregular`
"""
function register_spectral_solver!(key::Symbol, solver_type::Type{<:AbstractPoissonSolver})
    _SPECTRAL_SOLVERS[key] = solver_type
    return nothing
end

function _resolve_auto_solver(grid::StructuredGrid{G,T}) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    if G <: SphericalGeometry
        # Try spherical solvers
        for key in (:spherical_irregular, :spherical_regular)
            if haskey(_SPECTRAL_SOLVERS, key)
                return _SPECTRAL_SOLVERS[key]()
            end
        end
    elseif G <: CartesianGeometry
        # Try Cartesian solvers
        for key in (:cartesian_irregular, :cartesian_regular)
            if haskey(_SPECTRAL_SOLVERS, key)
                return _SPECTRAL_SOLVERS[key]()
            end
        end
    end

    # Fallback to SOR
    @debug """No spectral solver extension loaded — falling back to SORSolver which may be \
    orders of magnitude slower. Load a spectral extension for your geometry: \
    `using FFTW` (Cartesian regular), `using FINUFFT` (Cartesian irregular), \
    `using FastSphericalHarmonics` (spherical regular), or `using NUFSHT` (spherical irregular). \
    See the HelmholtzDecomposition.jl README for details."""
    return SORSolver()
end

# ---------------------------------------------------------------------------
# SOR Solver
# ---------------------------------------------------------------------------

"""
    SORSolver(; max_iter=10_000, tol=1e-6, ω=1.85, boundary=:neumann)

Red-Black Successive Over-Relaxation solver for the 2D Poisson equation ∇²Φ = RHS.

This is the base fallback solver included in the package without any external dependencies.
It works on any grid (regular, irregular, masked) but is O(N²) convergence and may be
**orders of magnitude slower** than the spectral solvers provided by extensions.

# Keyword Arguments
- `max_iter::Int` — Maximum SOR iterations (default: 10_000)
- `tol::Float64` — Convergence tolerance on max absolute update (default: 1e-6)
- `ω::Float64` — SOR relaxation factor, 1 < ω < 2 (default: 1.85)
- `boundary::Symbol` — Boundary condition: `:neumann` (zero normal gradient) or `:dirichlet` (zero value)

# Performance Warning
For grids larger than ~100×100, strongly consider loading a spectral extension:
- `using FFTW` for Cartesian periodic domains
- `using FastSphericalHarmonics` for regular spherical grids
- `using NUFSHT` for irregular/scattered spherical grids

See also: [`solve_poisson!`](@ref)
"""
struct SORSolver <: AbstractPoissonSolver
    max_iter::Int
    tol::Float64
    ω::Float64
    boundary::Symbol
end

SORSolver(; max_iter::Int=10_000, tol::Float64=1e-6, ω::Float64=1.85, boundary::Symbol=:neumann) =
    SORSolver(max_iter, tol, ω, boundary)

# ---------------------------------------------------------------------------
# solve_poisson! dispatch on AutoSolver
# ---------------------------------------------------------------------------

function solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::StructuredGrid{G,T},
    solver::AutoSolver;
    kwargs...
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    resolved = _resolve_auto_solver(grid)
    return solve_poisson!(Φ, RHS, grid, resolved; kwargs...)
end

# ---------------------------------------------------------------------------
# solve_poisson! for SORSolver
# ---------------------------------------------------------------------------

"""
    solve_poisson!(Φ, RHS, grid, solver::SORSolver; kwargs...) → SolverResult

Solve ∇²Φ = RHS on wet points of `grid` using Red-Black SOR.

# Arguments
- `Φ::AbstractMatrix{T}` — Solution array (modified in-place, initially zeroed)
- `RHS::AbstractMatrix{T}` — Right-hand side (vorticity or divergence field)
- `grid::StructuredGrid` — Grid with geometry and mask
- `solver::SORSolver` — Solver configuration

Returns a [`SolverResult`](@ref) with convergence information.
"""
function solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::StructuredGrid{G,T},
    solver::SORSolver;
    kwargs...
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    max_iter = solver.max_iter
    tol = T(solver.tol)
    ω = T(solver.ω)
    boundary = solver.boundary

    # Pre-calculate spatial factors
    if G <: CartesianGeometry{T}
        inv_dx2 = one(T) / (grid.geometry.dx^2)
        inv_dy2 = one(T) / (grid.geometry.dy^2)
        denom = T(2) * (inv_dx2 + inv_dy2)
    else
        R = grid.geometry.R
        dλ = Nlon > 1 ? grid.lon[2] - grid.lon[1] : T(0)
        dφ = Nlat > 1 ? grid.lat[2] - grid.lat[1] : T(0)
        inv_dλ2 = one(T) / (dλ^2)
        inv_dφ2 = one(T) / (dφ^2)
    end

    fill!(Φ, zero(T))

    final_iter = 0
    final_diff = zero(T)

    for iter in 1:max_iter
        max_diff = zero(T)
        final_iter = iter

        # Red-Black SOR sweeps
        for color in 0:1
            for j in 1:Nlat
                for i in 1:Nlon
                    if ((i + j) % 2) == color
                        iswet(grid, i, j) || continue

                        # Fetch neighbors with boundary conditions
                        if boundary === :neumann
                            Φ_ip = i < Nlon && iswet(grid, i+1, j) ? Φ[i+1, j] : Φ[i, j]
                            Φ_im = i > 1    && iswet(grid, i-1, j) ? Φ[i-1, j] : Φ[i, j]
                            Φ_jp = j < Nlat && iswet(grid, i, j+1) ? Φ[i, j+1] : Φ[i, j]
                            Φ_jm = j > 1    && iswet(grid, i, j-1) ? Φ[i, j-1] : Φ[i, j]
                        else # :dirichlet
                            Φ_ip = i < Nlon && iswet(grid, i+1, j) ? Φ[i+1, j] : zero(T)
                            Φ_im = i > 1    && iswet(grid, i-1, j) ? Φ[i-1, j] : zero(T)
                            Φ_jp = j < Nlat && iswet(grid, i, j+1) ? Φ[i, j+1] : zero(T)
                            Φ_jm = j > 1    && iswet(grid, i, j-1) ? Φ[i, j-1] : zero(T)
                        end

                        # Discretized Laplace operator
                        if G <: CartesianGeometry{T}
                            Φ_new = (inv_dx2 * (Φ_ip + Φ_im) + inv_dy2 * (Φ_jp + Φ_jm) - RHS[i, j]) / denom
                        else
                            φ = grid.lat[j]
                            cosφ = cos(φ)
                            sinφ = sin(φ)

                            term_λ = (Φ_ip + Φ_im) * inv_dλ2 / (R^2 * cosφ^2)
                            term_φ = ((Φ_jp + Φ_jm) * inv_dφ2 - sinφ * (Φ_jp - Φ_jm) / (T(2) * dφ * cosφ)) / R^2

                            denom_sph = T(2) * inv_dλ2 / (R^2 * cosφ^2) + T(2) * inv_dφ2 / R^2
                            Φ_new = (term_λ + term_φ - RHS[i, j]) / denom_sph
                        end

                        diff = Φ_new - Φ[i, j]
                        Φ[i, j] += ω * diff
                        max_diff = max(max_diff, abs(diff))
                    end
                end
            end
        end

        final_diff = max_diff
        if max_diff < tol
            return SolverResult{T}(true, iter, max_diff)
        end
    end

    return SolverResult{T}(false, final_iter, final_diff)
end
