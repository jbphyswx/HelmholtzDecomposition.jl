"""
    Decomposition.jl — Physical-space Helmholtz(-Hodge) decomposition.

Decomposes a velocity field `u` (component-last array of size `(dims..., N)`) into

    u = u_div  ⊕  u_rot  ⊕  u_harm

where `u_div = ∇χ` is curl-free (divergent), `u_rot` is divergence-free (rotational),
and `u_harm` is the harmonic remainder (both div- and curl-free) carried by domain
topology / boundaries. The potentials solve Poisson equations:

    Δχ      = δ = ∇·u                       (scalar velocity potential)
    ΔR_ab   = W_ab = ∂_a u_b − ∂_b u_a      (rotation-potential matrix, a < b)

In 2D the single rotation component `R_12` is the streamfunction `ψ`; in 3D the three
components are the Hodge dual of the vector potential `A`. On the sphere the rotational
and divergent parts are reconstructed with the spherical metric (Δ on S² has eigenvalues
`−ℓ(ℓ+1)/R²`); filtering the scalar potentials commutes with `∇` on S² (Aluie 2019),
which is the package's reason for existing.

# References
- Aluie (2019): Convolutions on the sphere, doi:10.1007/s13137-019-0123-9.
- Glötzl & Richters (2023): n-dimensional Helmholtz potentials, doi:10.1016/j.jmaa.2023.127138.
- Bhatia et al. (2013): The Helmholtz-Hodge Decomposition — A Survey.
"""

export HelmholtzResult, helmholtz_decompose!, helmholtz_decompose
export streamfunction, velocity_potential, vector_potential

"""
    HelmholtzResult{N,T,AV,AS}

Result of an `N`-dimensional Helmholtz-Hodge decomposition. Velocity-like fields use the
component-last layout `(dims..., N)`; potentials and scalar diagnostics are `(dims...)`.

# Fields
- `u_rot::AV` — rotational (divergence-free) velocity, `(dims..., N)`.
- `u_div::AV` — divergent (curl-free) velocity, `(dims..., N)`.
- `u_harm::AV` — harmonic remainder `u − u_div − u_rot`, `(dims..., N)`.
- `χ::AS` — scalar velocity potential, `(dims...)`.
- `rotation_potential::AV` — rotation-potential components `(dims..., P)`, `P = N(N-1)/2`
  (the streamfunction `ψ` in 2D; the Hodge dual of `A` in 3D).
- `vorticity::AV` — rotation-tensor components `W_ab`, `(dims..., P)`.
- `divergence::AS` — divergence field `δ`, `(dims...)`.
- `harmonic_fraction::T` — `‖u_harm‖ / ‖u‖` (measure-weighted), a diagnostic of how much
  of the field lives in the harmonic (topological/boundary) subspace.
- `χ_solve::SolverResult{T}` — convergence info for the `χ` solve.
- `rot_solve::Vector{SolverResult{T}}` — convergence info for each rotation-potential solve.
"""
struct HelmholtzResult{N,T<:AbstractFloat,AV<:AbstractArray{T},AS<:AbstractArray{T,N}}
    u_rot::AV
    u_div::AV
    u_harm::AV
    χ::AS
    rotation_potential::AV
    vorticity::AV
    divergence::AS
    harmonic_fraction::T
    χ_solve::SolverResult{T}
    rot_solve::Vector{SolverResult{T}}
end

@inline Base.ndims(::HelmholtzResult{N}) where {N} = N

"""
    streamfunction(result::HelmholtzResult{2})

The 2-D streamfunction `ψ` (the single rotation-potential component). Only defined in 2D.
"""
streamfunction(r::HelmholtzResult{2}) = _component(r.rotation_potential, 1, Val(2))
streamfunction(::HelmholtzResult{N}) where {N} =
    throw(ArgumentError("streamfunction is only defined in 2D (got N=$N); use `vector_potential` in 3D or `rotation_potential` generally"))

"""
    velocity_potential(result)

The scalar velocity potential `χ` (defined in any dimension).
"""
velocity_potential(r::HelmholtzResult) = r.χ

"""
    vector_potential(result::HelmholtzResult{3}) -> (A1, A2, A3)

The 3-D vector potential `A`, the Hodge dual of the rotation potential:
`A1 = R_23`, `A2 = -R_13`, `A3 = R_12`. Only defined in 3D.
"""
function vector_potential(r::HelmholtzResult{3})
    R12 = _component(r.rotation_potential, 1, Val(3))  # pair (1,2)
    R13 = _component(r.rotation_potential, 2, Val(3))  # pair (1,3)
    R23 = _component(r.rotation_potential, 3, Val(3))  # pair (2,3)
    return (R23, -1 .* R13, R12)
end
vector_potential(::HelmholtzResult{N}) where {N} =
    throw(ArgumentError("vector_potential is only defined in 3D (got N=$N)"))

# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

"""
    HelmholtzResult(grid::StructuredGrid{N,G,T})

Pre-allocate a zeroed `HelmholtzResult` matching the grid dimensions.
"""
function HelmholtzResult(grid::StructuredGrid{N,G,T}) where {N,T<:AbstractFloat,G<:AbstractGeometry{T}}
    dims = size_tuple(grid)
    P = n_rotation_components(N)
    dummy = SolverResult{T}(false, 0, zero(T))
    return HelmholtzResult{N,T,Array{T,N + 1},Array{T,N}}(
        zeros(T, dims..., N),       # u_rot
        zeros(T, dims..., N),       # u_div
        zeros(T, dims..., N),       # u_harm
        zeros(T, dims...),          # χ
        zeros(T, dims..., P),       # rotation_potential
        zeros(T, dims..., P),       # vorticity
        zeros(T, dims...),          # divergence
        zero(T),
        dummy,
        [dummy for _ in 1:P],
    )
end

# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

"""
    helmholtz_decompose(u, grid; kwargs...) -> HelmholtzResult
    helmholtz_decompose(u, v, grid; kwargs...)         # 2D convenience
    helmholtz_decompose(u, v, w, grid; kwargs...)      # 3D convenience

Decompose a velocity field on `grid`. The primary method takes a single component-last
array `u` of size `(dims..., N)`. The 2- and 3-argument forms stack scalar component
arrays for convenience.

# Keyword Arguments
- `solver::AbstractPoissonSolver = AutoSolver()` — Poisson solver.
- `boundary_χ::Symbol = :neumann` — boundary condition for the velocity potential `χ`.
- `boundary_ψ::Symbol = :dirichlet` — boundary condition for the rotation potential.

Returns a [`HelmholtzResult`](@ref).
"""
function helmholtz_decompose(u::AbstractArray, grid::StructuredGrid; backend::AbstractExecutionBackend = AutoBackend(), kwargs...)
    return _decompose_backend(_resolve_backend(backend, u), u, grid; kwargs...)
end

"""
    _resolve_backend(backend, u) -> AbstractExecutionBackend

Resolve an `AutoBackend` to a concrete execution backend from the array type. The default
is `SerialBackend`; the CUDA extension specializes this on GPU array types.
"""
_resolve_backend(b::AbstractExecutionBackend, ::AbstractArray) = b
_resolve_backend(::AutoBackend, ::AbstractArray) = SerialBackend()

"""
    _decompose_backend(backend, u, grid; kwargs...) -> HelmholtzResult

Execution-backend dispatch. The default (serial / threaded / GPU — where the arrays
themselves carry the compute) runs the in-place core; the MPI/Distributed extensions
specialize this to partition the domain, decompose locally, and gather.
"""
function _decompose_backend(::AbstractExecutionBackend, u::AbstractArray, grid::StructuredGrid; kwargs...)
    return helmholtz_decompose!(HelmholtzResult(grid), u, grid; kwargs...)
end

function helmholtz_decompose(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}, grid::StructuredGrid{N}; kwargs...) where {N}
    U = _stack_components(grid, u, v)
    return helmholtz_decompose(U, grid; kwargs...)
end

function helmholtz_decompose(u::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}, w::AbstractArray{<:Any,N}, grid::StructuredGrid{N}; kwargs...) where {N}
    U = _stack_components(grid, u, v, w)
    return helmholtz_decompose(U, grid; kwargs...)
end

function _stack_components(grid::StructuredGrid{N,G,T}, comps::Vararg{AbstractArray,M}) where {N,G,T,M}
    dims = size_tuple(grid)
    U = Array{T,N + 1}(undef, dims..., M)
    for c in 1:M
        copyto!(_component(U, c, Val(N)), comps[c])
    end
    return U
end

"""
    helmholtz_decompose!(result, u, grid; kwargs...) -> result

In-place decomposition writing into a pre-allocated [`HelmholtzResult`](@ref). `u` is a
component-last array of size `(dims..., N)`.
"""
function helmholtz_decompose!(
    result::HelmholtzResult{N,T},
    u::AbstractArray,
    grid::StructuredGrid{N,G,T};
    solver::AbstractPoissonSolver = AutoSolver(),
    boundary_χ::Symbol = :neumann,
    boundary_ψ::Symbol = :dirichlet,
) where {N,T<:AbstractFloat,G<:AbstractGeometry{T}}
    size(u) == (size_tuple(grid)..., N) ||
        throw(DimensionMismatch("velocity array size $(size(u)) does not match (dims..., N) = $((size_tuple(grid)..., N))"))

    div_f = result.divergence
    vort = result.vorticity
    χ = result.χ
    Rpot = result.rotation_potential

    # 1. Divergence and rotation tensor.
    _compute_div_rot!(div_f, vort, u, grid)

    # 2. Divergence balancing — only for the homogeneous Neumann χ problem, where the
    #    Fredholm solvability condition ∫δ dV = 0 must hold (subtracting a nonzero mean
    #    under an inhomogeneous BC would corrupt the field).
    if boundary_χ === :neumann
        _subtract_weighted_mean!(div_f, grid)
    end

    # 3. Solve the Poisson equations.
    χ_result = solve_poisson!(χ, div_f, grid, solver; boundary = boundary_χ)
    P = n_rotation_components(N)
    rot_results = result.rot_solve
    for p in 1:P
        Rp = _component(Rpot, p, Val(N))
        Wp = _component(vort, p, Val(N))
        rot_results[p] = solve_poisson!(Rp, Wp, grid, solver; boundary = boundary_ψ)
    end

    # 4. Reconstruct velocities from the potentials.
    _reconstruct!(result.u_div, result.u_rot, χ, Rpot, grid)

    # 5. Harmonic remainder and diagnostic.
    hfrac = _harmonic_residual!(result.u_harm, u, result.u_div, result.u_rot, grid)

    return HelmholtzResult{N,T,typeof(result.u_rot),typeof(χ)}(
        result.u_rot, result.u_div, result.u_harm, χ, Rpot, vort, div_f,
        hfrac, χ_result, rot_results,
    )
end

# ---------------------------------------------------------------------------
# Geometry-dispatched operators
# ---------------------------------------------------------------------------

# Cartesian (dimension-generic) — delegate to Operators.jl.
function _compute_div_rot!(div_f, vort, u, grid::StructuredGrid{N,<:CartesianGeometry}) where {N}
    cartesian_divergence!(div_f, u, grid)
    cartesian_rotation_tensor!(vort, u, grid)
    return nothing
end

function _reconstruct!(u_div, u_rot, χ, Rpot, grid::StructuredGrid{N,<:CartesianGeometry}) where {N}
    cartesian_reconstruct_div!(u_div, χ, grid)
    cartesian_reconstruct_rot!(u_rot, Rpot, grid)
    return nothing
end

# Spherical (N == 2) — uses the spherical metric with periodic longitude.
function _compute_div_rot!(div_f, vort, u, grid::StructuredGrid{2,<:SphericalGeometry{T}}) where {T}
    Nlon, Nlat = size_tuple(grid)
    lon, lat = grid.coords_axes
    R = grid.geometry.R
    dλ = Nlon > 1 ? lon[2] - lon[1] : one(T)
    dφ = Nlat > 1 ? lat[2] - lat[1] : one(T)
    periodic = _is_periodic_longitude(lon, dλ)
    uc = _component(u, 1, Val(2))
    vc = _component(u, 2, Val(2))
    fill!(div_f, zero(T))
    fill!(vort, zero(T))
    ζ = _component(vort, 1, Val(2))
    @inbounds for j in 1:Nlat
        cosφ = cos(lat[j])
        abs(cosφ) < sqrt(eps(T)) && continue
        for i in 1:Nlon
            isactive(grid, i, j) || continue
            ip, im = _lon_neighbors(i, Nlon, grid, j, periodic)
            jp = j < Nlat && isactive(grid, i, j + 1) ? j + 1 : j
            jm = j > 1 && isactive(grid, i, j - 1) ? j - 1 : j

            h_λ = _lon_step(i, ip, im, Nlon, dλ, periodic)
            h_φ = (jp - jm) * dφ

            dudλ = h_λ == 0 ? zero(T) : (uc[ip, j] - uc[im, j]) / h_λ
            dvdλ = h_λ == 0 ? zero(T) : (vc[ip, j] - vc[im, j]) / h_λ
            vcos_jp = vc[i, jp] * cos(lat[jp])
            vcos_jm = vc[i, jm] * cos(lat[jm])
            ucos_jp = uc[i, jp] * cos(lat[jp])
            ucos_jm = uc[i, jm] * cos(lat[jm])
            d_vcos_dφ = h_φ == 0 ? zero(T) : (vcos_jp - vcos_jm) / h_φ
            d_ucos_dφ = h_φ == 0 ? zero(T) : (ucos_jp - ucos_jm) / h_φ

            div_f[i, j] = (dudλ + d_vcos_dφ) / (R * cosφ)
            ζ[i, j] = (dvdλ - d_ucos_dφ) / (R * cosφ)
        end
    end
    return nothing
end

function _reconstruct!(u_div, u_rot, χ, Rpot, grid::StructuredGrid{2,<:SphericalGeometry{T}}) where {T}
    Nlon, Nlat = size_tuple(grid)
    lon, lat = grid.coords_axes
    R = grid.geometry.R
    dλ = Nlon > 1 ? lon[2] - lon[1] : one(T)
    dφ = Nlat > 1 ? lat[2] - lat[1] : one(T)
    periodic = _is_periodic_longitude(lon, dλ)
    ψ = _component(Rpot, 1, Val(2))
    u_divc = _component(u_div, 1, Val(2)); v_divc = _component(u_div, 2, Val(2))
    u_rotc = _component(u_rot, 1, Val(2)); v_rotc = _component(u_rot, 2, Val(2))
    fill!(u_div, zero(T))
    fill!(u_rot, zero(T))
    @inbounds for j in 1:Nlat
        cosφ = cos(lat[j])
        abs(cosφ) < sqrt(eps(T)) && continue
        for i in 1:Nlon
            isactive(grid, i, j) || continue
            ip, im = _lon_neighbors(i, Nlon, grid, j, periodic)
            jp = j < Nlat && isactive(grid, i, j + 1) ? j + 1 : j
            jm = j > 1 && isactive(grid, i, j - 1) ? j - 1 : j
            h_λ = _lon_step(i, ip, im, Nlon, dλ, periodic)
            h_φ = (jp - jm) * dφ

            dχdλ = h_λ == 0 ? zero(T) : (χ[ip, j] - χ[im, j]) / h_λ
            dχdφ = h_φ == 0 ? zero(T) : (χ[i, jp] - χ[i, jm]) / h_φ
            dψdλ = h_λ == 0 ? zero(T) : (ψ[ip, j] - ψ[im, j]) / h_λ
            dψdφ = h_φ == 0 ? zero(T) : (ψ[i, jp] - ψ[i, jm]) / h_φ

            u_divc[i, j] = dχdλ / (R * cosφ)
            v_divc[i, j] = dχdφ / R
            u_rotc[i, j] = -dψdφ / R
            v_rotc[i, j] = dψdλ / (R * cosφ)
        end
    end
    return nothing
end

@inline function _lon_step(i, ip, im, Nlon, dλ::T, periodic) where {T}
    if periodic
        # Across the seam the index jumps by Nlon-1 but the physical step is one cell.
        return (ip == i || im == i) ? (ip == im ? zero(T) : dλ) : T(2) * dλ
    else
        return (ip - im) * dλ
    end
end

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

function _subtract_weighted_mean!(field, grid::StructuredGrid{N}) where {N}
    T = eltype(field)
    total = zero(T)
    weight = zero(T)
    @inbounds for I in CartesianIndices(grid.mask)
        grid.mask[I] || continue
        w = cellmeasure(grid, Tuple(I)...)
        total += field[I] * w
        weight += w
    end
    weight == 0 && return field
    m = total / weight
    @inbounds for I in CartesianIndices(grid.mask)
        grid.mask[I] || continue
        field[I] -= m
    end
    return field
end

"""
    _velocity_norm(U, grid) -> T

Measure-weighted L² norm `sqrt(∫ |U|² dV)` of a component-last velocity array over the
active cells of `grid`.
"""
function _velocity_norm(U, grid::StructuredGrid{N}) where {N}
    T = real(eltype(U))
    acc = zero(T)
    comps = ntuple(c -> _component(U, c, Val(N)), Val(N))
    @inbounds for I in CartesianIndices(grid.mask)
        grid.mask[I] || continue
        w = cellmeasure(grid, Tuple(I)...)
        for c in 1:N
            acc += w * abs2(comps[c][I])
        end
    end
    return sqrt(acc)
end

function _harmonic_residual!(u_harm, u, u_div, u_rot, grid::StructuredGrid{N}) where {N}
    T = eltype(u_harm)
    @. u_harm = u - u_div - u_rot
    # Zero masked-out cells.
    @inbounds for I in CartesianIndices(grid.mask)
        grid.mask[I] && continue
        for c in 1:N
            _component(u_harm, c, Val(N))[I] = zero(T)
        end
    end
    den = _velocity_norm(u, grid)
    return den == 0 ? zero(T) : _velocity_norm(u_harm, grid) / den
end
