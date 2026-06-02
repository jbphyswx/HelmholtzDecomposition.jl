"""
    Decomposition.jl — Helmholtz decomposition of 2D velocity fields.

Decomposes a horizontal velocity field (u, v) into rotational (non-divergent) and
divergent (irrotational) components by solving Poisson equations for the stream
function ψ and velocity potential χ:

    ∇²ψ = ζ   (vorticity)
    ∇²χ = δ   (divergence)

    u_rot = -∂ψ/∂y,   v_rot = ∂ψ/∂x       (Cartesian)
    u_div =  ∂χ/∂x,   v_div = ∂χ/∂y       (Cartesian)

On the sphere:
    u_rot = -(1/R) ∂ψ/∂φ,       v_rot = 1/(R cosφ) ∂ψ/∂λ
    u_div = 1/(R cosφ) ∂χ/∂λ,   v_div = (1/R) ∂χ/∂φ

# Why this matters for coarse-graining on the sphere
Aluie (2019, doi:10.1007/s13137-019-0123-9) proves that filtering the scalar
potentials (ψ, χ) separately and reconstructing the velocity is equivalent to
the generalized convolution that commutes with differential operators on S².
Simply filtering Cartesian velocity components does NOT commute with ∇ on S².

# References
- Aluie (2019): Convolutions on the sphere. Section 7, Proposition 2.
- Buzzicotti et al. (2023): doi:10.1126/sciadv.adi7420
"""

export HelmholtzResult, helmholtz_decompose!, helmholtz_decompose

"""
    HelmholtzResult{T}

Result of a Helmholtz decomposition, containing both the decomposed velocity
components and the scalar potentials.

# Fields
- `u_rot::Matrix{T}` — Rotational (non-divergent) u-component
- `v_rot::Matrix{T}` — Rotational (non-divergent) v-component
- `u_div::Matrix{T}` — Divergent (irrotational) u-component
- `v_div::Matrix{T}` — Divergent (irrotational) v-component
- `ψ::Matrix{T}` — Stream function (vorticity potential)
- `χ::Matrix{T}` — Velocity potential (divergence potential)
- `vorticity::Matrix{T}` — Computed vorticity field ζ = ∂v/∂x - ∂u/∂y
- `divergence::Matrix{T}` — Computed divergence field δ = ∂u/∂x + ∂v/∂y
- `ψ_solve::SolverResult{T}` — Convergence info for ψ Poisson solve
- `χ_solve::SolverResult{T}` — Convergence info for χ Poisson solve
"""
struct HelmholtzResult{T<:AbstractFloat}
    u_rot::Matrix{T}
    v_rot::Matrix{T}
    u_div::Matrix{T}
    v_div::Matrix{T}
    ψ::Matrix{T}
    χ::Matrix{T}
    vorticity::Matrix{T}
    divergence::Matrix{T}
    ψ_solve::SolverResult{T}
    χ_solve::SolverResult{T}
end

"""
    HelmholtzResult(grid::StructuredGrid{G,T}) where {G,T}

Pre-allocate a `HelmholtzResult` for the given grid dimensions.
"""
function HelmholtzResult(grid::StructuredGrid{G,T}) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    dummy_solve = SolverResult{T}(false, 0, zero(T))
    return HelmholtzResult{T}(
        zeros(T, Nlon, Nlat), zeros(T, Nlon, Nlat),
        zeros(T, Nlon, Nlat), zeros(T, Nlon, Nlat),
        zeros(T, Nlon, Nlat), zeros(T, Nlon, Nlat),
        zeros(T, Nlon, Nlat), zeros(T, Nlon, Nlat),
        dummy_solve, dummy_solve
    )
end

"""
    helmholtz_decompose!(result, u, v, grid; solver=AutoSolver(), boundary_χ=:neumann, boundary_ψ=:dirichlet)

In-place Helmholtz decomposition: decompose velocity (u, v) into rotational and
divergent components, writing results into `result::HelmholtzResult`.

# Arguments
- `result::HelmholtzResult{T}` — Pre-allocated result (modified in-place)
- `u::AbstractMatrix` — Zonal velocity (u_east or u_x)
- `v::AbstractMatrix` — Meridional velocity (v_north or v_y)
- `grid::StructuredGrid` — Grid with geometry and mask

# Keyword Arguments
- `solver::AbstractPoissonSolver` — Poisson solver (default: `AutoSolver()`)
- `boundary_χ::Symbol` — BC for velocity potential (default: `:neumann`)
- `boundary_ψ::Symbol` — BC for stream function (default: `:dirichlet`)

# Returns
The modified `result` struct.
"""
function helmholtz_decompose!(
    result::HelmholtzResult{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::StructuredGrid{G,T};
    solver::AbstractPoissonSolver = AutoSolver(),
    boundary_χ::Symbol = :neumann,
    boundary_ψ::Symbol = :dirichlet
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    Nlon, Nlat = size_tuple(grid)

    div_f = result.divergence
    vort_f = result.vorticity
    χ = result.χ
    ψ = result.ψ

    fill!(div_f, zero(T))
    fill!(vort_f, zero(T))

    # Precompute grid spacings
    if G <: CartesianGeometry{T}
        dx = grid.geometry.dx
        dy = grid.geometry.dy
    else
        R = grid.geometry.R
        dλ = Nlon > 1 ? grid.lon[2] - grid.lon[1] : T(0)
        dφ = Nlat > 1 ? grid.lat[2] - grid.lat[1] : T(0)
    end

    # 1. Compute divergence and vorticity
    for j in 1:Nlat
        for i in 1:Nlon
            iswet(grid, i, j) || continue

            ip = i < Nlon && iswet(grid, i+1, j) ? i+1 : i
            im = i > 1    && iswet(grid, i-1, j) ? i-1 : i
            jp = j < Nlat && iswet(grid, i, j+1) ? j+1 : j
            jm = j > 1    && iswet(grid, i, j-1) ? j-1 : j

            if G <: CartesianGeometry{T}
                h_x = ip == im ? dx : (ip - im) * dx
                h_y = jp == jm ? dy : (jp - jm) * dy

                dudx = (u[ip, j] - u[im, j]) / h_x
                dvdy = (v[i, jp] - v[i, jm]) / h_y
                div_f[i, j] = dudx + dvdy

                dvdx = (v[ip, j] - v[im, j]) / h_x
                dudy = (u[i, jp] - u[i, jm]) / h_y
                vort_f[i, j] = dvdx - dudy
            else
                φ = grid.lat[j]
                cosφ = cos(φ)

                h_λ = (ip - im) * dλ
                h_φ = (jp - jm) * dφ

                dudλ = ip == im ? zero(T) : (u[ip, j] - u[im, j]) / h_λ
                v_cos_jp = v[i, jp] * cos(grid.lat[jp])
                v_cos_jm = v[i, jm] * cos(grid.lat[jm])
                d_vcos_dφ = jp == jm ? zero(T) : (v_cos_jp - v_cos_jm) / h_φ
                div_f[i, j] = (dudλ + d_vcos_dφ) / (R * cosφ)

                dvdλ = ip == im ? zero(T) : (v[ip, j] - v[im, j]) / h_λ
                u_cos_jp = u[i, jp] * cos(grid.lat[jp])
                u_cos_jm = u[i, jm] * cos(grid.lat[jm])
                d_ucos_dφ = jp == jm ? zero(T) : (u_cos_jp - u_cos_jm) / h_φ
                vort_f[i, j] = (dvdλ - d_ucos_dφ) / (R * cosφ)
            end
        end
    end

    # 2. Divergence balancing (Fredholm solvability condition for Neumann BCs)
    total_div = zero(T)
    total_area = zero(T)
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                total_div += div_f[i, j] * area(grid, i, j)
                total_area += area(grid, i, j)
            end
        end
    end

    mean_div = total_div / total_area
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                div_f[i, j] -= mean_div
            end
        end
    end

    # 3. Solve Poisson equations
    χ_solver = solver isa AutoSolver ? solver : (
        solver isa SORSolver ? SORSolver(; max_iter=solver.max_iter, tol=solver.tol, ω=solver.ω, boundary=boundary_χ) : solver
    )
    ψ_solver = solver isa AutoSolver ? solver : (
        solver isa SORSolver ? SORSolver(; max_iter=solver.max_iter, tol=solver.tol, ω=solver.ω, boundary=boundary_ψ) : solver
    )

    χ_result = solve_poisson!(χ, div_f, grid, χ_solver)
    ψ_result = solve_poisson!(ψ, vort_f, grid, ψ_solver)

    # 4. Reconstruct velocities from potentials
    u_rot = result.u_rot
    v_rot = result.v_rot
    u_div = result.u_div
    v_div = result.v_div

    for j in 1:Nlat
        for i in 1:Nlon
            if !iswet(grid, i, j)
                u_div[i, j] = zero(T)
                v_div[i, j] = zero(T)
                u_rot[i, j] = zero(T)
                v_rot[i, j] = zero(T)
                continue
            end

            ip = i < Nlon && iswet(grid, i+1, j) ? i+1 : i
            im = i > 1    && iswet(grid, i-1, j) ? i-1 : i
            jp = j < Nlat && iswet(grid, i, j+1) ? j+1 : j
            jm = j > 1    && iswet(grid, i, j-1) ? j-1 : j

            if G <: CartesianGeometry{T}
                h_x = ip == im ? dx : (ip - im) * dx
                h_y = jp == jm ? dy : (jp - jm) * dy

                # u_div = ∇χ
                u_div[i, j] = (χ[ip, j] - χ[im, j]) / h_x
                v_div[i, j] = (χ[i, jp] - χ[i, jm]) / h_y

                # u_rot = ∇ × (ψ ẑ) = [-∂ψ/∂y, ∂ψ/∂x]
                u_rot[i, j] = -(ψ[i, jp] - ψ[i, jm]) / h_y
                v_rot[i, j] = (ψ[ip, j] - ψ[im, j]) / h_x
            else
                φ = grid.lat[j]
                cosφ = cos(φ)
                h_λ = (ip - im) * dλ
                h_φ = (jp - jm) * dφ

                # u_div_λ = 1/(R cosφ) ∂χ/∂λ,  v_div_φ = 1/R ∂χ/∂φ
                u_div[i, j] = ip == im ? zero(T) : (χ[ip, j] - χ[im, j]) / (h_λ * R * cosφ)
                v_div[i, j] = jp == jm ? zero(T) : (χ[i, jp] - χ[i, jm]) / (h_φ * R)

                # u_rot_λ = -1/R ∂ψ/∂φ,  v_rot_φ = 1/(R cosφ) ∂ψ/∂λ
                u_rot[i, j] = jp == jm ? zero(T) : -(ψ[i, jp] - ψ[i, jm]) / (h_φ * R)
                v_rot[i, j] = ip == im ? zero(T) : (ψ[ip, j] - ψ[im, j]) / (h_λ * R * cosφ)
            end
        end
    end

    # Build final result with solver diagnostics
    return HelmholtzResult{T}(
        u_rot, v_rot, u_div, v_div, ψ, χ,
        vort_f, div_f, ψ_result, χ_result
    )
end

"""
    helmholtz_decompose(u, v, grid; solver=AutoSolver(), kwargs...)

Allocating version of [`helmholtz_decompose!`](@ref). Returns a new `HelmholtzResult`.

# Example
```julia
using HelmholtzDecomposition: HelmholtzDecomposition

geom = HelmholtzDecomposition.CartesianGeometry(1000.0, 1000.0)
grid = HelmholtzDecomposition.StructuredGrid(geom, collect(0.0:1000.0:99000.0), collect(0.0:1000.0:99000.0))
result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)
# result.u_rot, result.v_rot, result.u_div, result.v_div
# result.ψ (stream function), result.χ (velocity potential)
```
"""
function helmholtz_decompose(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::StructuredGrid{G,T};
    kwargs...
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    result = HelmholtzResult(grid)
    return helmholtz_decompose!(result, u, v, grid; kwargs...)
end
