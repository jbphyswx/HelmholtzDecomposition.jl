"""
    TestFields.jl — Synthetic velocity field generators for testing and examples.

Provides analytically known velocity fields with prescribed rotational and/or
divergent components, enabling verification that the Helmholtz decomposition
correctly recovers each component.
"""

export taylor_green_vortex, point_source_sink, rankine_vortex_with_source
export rossby_wave, kelvin_ekman_flow
export disk_mask, harmonic_vortex, harmonic_source

# ---------------------------------------------------------------------------
# Cartesian test fields
# ---------------------------------------------------------------------------

"""
    taylor_green_vortex(grid; A=1.0, kx=2π, ky=2π) → (u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact)

Taylor-Green vortex on a Cartesian grid. Purely non-divergent (rotational only).

    ψ = A cos(kx·x) cos(ky·y)
    u = -∂ψ/∂y =  A kỹ cos(kx·x) sin(ky·y)
    v =  ∂ψ/∂x = -A kx̃ sin(kx·x) cos(ky·y)

The divergent component is exactly zero.
"""
function taylor_green_vortex(
    grid::StructuredGrid{2,G,T};
    A::Real = one(T),
    kx::Real = T(2π) / (grid.coords_axes[1][end] - grid.coords_axes[1][1] + grid.coords_axes[1][2] - grid.coords_axes[1][1]),
    ky::Real = T(2π) / (grid.coords_axes[2][end] - grid.coords_axes[2][1] + grid.coords_axes[2][2] - grid.coords_axes[2][1])
) where {T<:AbstractFloat, G<:CartesianGeometry{2,T}}
    Nlon, Nlat = size_tuple(grid)
    xs, ys = grid.coords_axes
    A_T = T(A)
    kx_T = T(kx)
    ky_T = T(ky)

    u = Matrix{T}(undef, Nlon, Nlat)
    v = Matrix{T}(undef, Nlon, Nlat)

    for j in 1:Nlat
        y = ys[j]
        for i in 1:Nlon
            x = xs[i]
            u[i, j] =  A_T * ky_T * cos(kx_T * x) * sin(ky_T * y)
            v[i, j] = -A_T * kx_T * sin(kx_T * x) * cos(ky_T * y)
        end
    end

    u_rot_exact = copy(u)
    v_rot_exact = copy(v)
    u_div_exact = zeros(T, Nlon, Nlat)
    v_div_exact = zeros(T, Nlon, Nlat)

    return u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact
end

"""
    point_source_sink(grid; A=1.0, kx=2π, ky=2π) → (u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact)

Purely divergent field on a Cartesian grid (irrotational, zero vorticity).

    χ = A sin(kx·x) sin(ky·y)
    u = ∂χ/∂x = A kx cos(kx·x) sin(ky·y)
    v = ∂χ/∂y = A ky sin(kx·x) cos(ky·y)

The rotational component is exactly zero.
"""
function point_source_sink(
    grid::StructuredGrid{2,G,T};
    A::Real = one(T),
    kx::Real = T(2π) / (grid.coords_axes[1][end] - grid.coords_axes[1][1] + grid.coords_axes[1][2] - grid.coords_axes[1][1]),
    ky::Real = T(2π) / (grid.coords_axes[2][end] - grid.coords_axes[2][1] + grid.coords_axes[2][2] - grid.coords_axes[2][1])
) where {T<:AbstractFloat, G<:CartesianGeometry{2,T}}
    Nlon, Nlat = size_tuple(grid)
    xs, ys = grid.coords_axes
    A_T = T(A)
    kx_T = T(kx)
    ky_T = T(ky)

    u = Matrix{T}(undef, Nlon, Nlat)
    v = Matrix{T}(undef, Nlon, Nlat)

    for j in 1:Nlat
        y = ys[j]
        for i in 1:Nlon
            x = xs[i]
            u[i, j] = A_T * kx_T * cos(kx_T * x) * sin(ky_T * y)
            v[i, j] = A_T * ky_T * sin(kx_T * x) * cos(ky_T * y)
        end
    end

    u_rot_exact = zeros(T, Nlon, Nlat)
    v_rot_exact = zeros(T, Nlon, Nlat)
    u_div_exact = copy(u)
    v_div_exact = copy(v)

    return u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact
end

"""
    rankine_vortex_with_source(grid; A_rot=1.0, A_div=0.5, kx=2π, ky=2π)

Mixed field: sum of Taylor-Green vortex (rotational) and source/sink (divergent)
on a Cartesian grid.
"""
function rankine_vortex_with_source(
    grid::StructuredGrid{2,G,T};
    A_rot::Real = one(T),
    A_div::Real = T(0.5),
    kx::Real = T(2π) / (grid.coords_axes[1][end] - grid.coords_axes[1][1] + grid.coords_axes[1][2] - grid.coords_axes[1][1]),
    ky::Real = T(2π) / (grid.coords_axes[2][end] - grid.coords_axes[2][1] + grid.coords_axes[2][2] - grid.coords_axes[2][1])
) where {T<:AbstractFloat, G<:CartesianGeometry{2,T}}
    u_r, v_r, u_rot, v_rot, _, _ = taylor_green_vortex(grid; A=A_rot, kx=kx, ky=ky)
    u_d, v_d, _, _, u_div, v_div = point_source_sink(grid; A=A_div, kx=kx, ky=ky)

    u = u_r .+ u_d
    v = v_r .+ v_d

    return u, v, u_rot, v_rot, u_div, v_div
end

# ---------------------------------------------------------------------------
# Spherical test fields
# ---------------------------------------------------------------------------

"""
    rossby_wave(grid; n=3, m=2, A=1.0) → (u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact)

Non-divergent Rossby wave-like field on a spherical grid. Purely rotational.

Uses spherical harmonic-like stream function:
    ψ(λ, φ) = A · cos(m·λ) · Pₙᵐ(sin φ)   (simplified as cosⁿ φ · sin(m·λ))

The velocity is derived from ψ so the divergent component is exactly zero.
"""
function rossby_wave(
    grid::StructuredGrid{2,G,T};
    n::Int = 3,
    m::Int = 2,
    A::Real = one(T)
) where {T<:AbstractFloat, G<:SphericalGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    lons, lats = grid.coords_axes
    R = grid.geometry.R
    A_T = T(A)

    u = Matrix{T}(undef, Nlon, Nlat)
    v = Matrix{T}(undef, Nlon, Nlat)

    for j in 1:Nlat
        φ = lats[j]
        cosφ = cos(φ)
        sinφ = sin(φ)
        for i in 1:Nlon
            λ = lons[i]
            # ψ = A cos(mλ) cosⁿ(φ)
            # u = -(1/R) ∂ψ/∂φ = (A n / R) cos(mλ) cosⁿ⁻¹(φ) sin(φ)
            # v = 1/(R cosφ) ∂ψ/∂λ = -(A m / (R cosφ)) sin(mλ) cosⁿ(φ)
            u[i, j] = A_T * n / R * cos(m * λ) * cosφ^(n-1) * sinφ
            v[i, j] = -A_T * m / (R * cosφ) * sin(m * λ) * cosφ^n
        end
    end

    u_rot_exact = copy(u)
    v_rot_exact = copy(v)
    u_div_exact = zeros(T, Nlon, Nlat)
    v_div_exact = zeros(T, Nlon, Nlat)

    return u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact
end

"""
    kelvin_ekman_flow(grid; A_rot=1.0, A_div=0.3, n=3, m=2, p=2, q=1)

Mixed rotational + divergent flow on a spherical grid. Combines a Rossby-wave-like
rotational component with an Ekman-like divergent component.

The divergent part uses velocity potential:
    χ = A_div · sin(q·λ) · cosᵖ(φ)
"""
function kelvin_ekman_flow(
    grid::StructuredGrid{2,G,T};
    A_rot::Real = one(T),
    A_div::Real = T(0.3),
    n::Int = 3,
    m::Int = 2,
    p::Int = 2,
    q::Int = 1
) where {T<:AbstractFloat, G<:SphericalGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    lons, lats = grid.coords_axes
    R = grid.geometry.R
    A_div_T = T(A_div)

    # Rotational part
    _, _, u_rot, v_rot, _, _ = rossby_wave(grid; n=n, m=m, A=A_rot)

    # Divergent part from χ = A_div sin(qλ) cosᵖ(φ)
    u_div = Matrix{T}(undef, Nlon, Nlat)
    v_div = Matrix{T}(undef, Nlon, Nlat)

    for j in 1:Nlat
        φ = lats[j]
        cosφ = cos(φ)
        sinφ = sin(φ)
        for i in 1:Nlon
            λ = lons[i]
            # u_div = 1/(R cosφ) ∂χ/∂λ = A_div q / (R cosφ) cos(qλ) cosᵖ(φ)
            u_div[i, j] = A_div_T * q / (R * cosφ) * cos(q * λ) * cosφ^p
            # v_div = (1/R) ∂χ/∂φ = -A_div p / R sin(qλ) cosᵖ⁻¹(φ) sin(φ)
            v_div[i, j] = -A_div_T * p / R * sin(q * λ) * cosφ^(p-1) * sinφ
        end
    end

    u = u_rot .+ u_div
    v = v_rot .+ v_div

    return u, v, u_rot, v_rot, u_div, v_div
end

# ---------------------------------------------------------------------------
# Harmonic (multiply-connected) test fields
# ---------------------------------------------------------------------------

"""
    disk_mask(grid; center=domain center, radius) -> Array{Bool}

Boolean active mask (`true` = active) for a 2-D Cartesian grid with a circular disk of the
given `radius` masked out (marked inactive), producing an annulus / domain-with-hole. Useful
for constructing multiply-connected domains that carry a nonzero harmonic component.
"""
function disk_mask(
    grid::StructuredGrid{2,<:CartesianGeometry{2,T}};
    center::Tuple{<:Real,<:Real} = _domain_center(grid),
    radius::Real = (grid.coords_axes[1][end] - grid.coords_axes[1][1]) / 4,
) where {T}
    xs, ys = grid.coords_axes
    cx, cy = T(center[1]), T(center[2])
    r2 = T(radius)^2
    mask = trues(length(xs), length(ys))
    @inbounds for j in eachindex(ys), i in eachindex(xs)
        ((xs[i] - cx)^2 + (ys[j] - cy)^2) <= r2 && (mask[i, j] = false)
    end
    return mask
end

function _domain_center(grid::StructuredGrid{2})
    xs, ys = grid.coords_axes
    return ((xs[1] + xs[end]) / 2, (ys[1] + ys[end]) / 2)
end

"""
    harmonic_vortex(grid; Γ=1.0, center=domain center) → (u, v)

Point-vortex velocity field `u = Γ/(2π) · (−(y−y₀), (x−x₀)) / r²`. Away from the
singular center this field is **both** divergence-free and curl-free — i.e. purely
*harmonic*. On a domain with the center masked out (see [`disk_mask`](@ref)) it has net
circulation `Γ` around the hole and cannot be represented by a single-valued streamfunction
or velocity potential, so a Helmholtz decomposition must place essentially all of it in the
harmonic component. The definitive test for issue #1.
"""
function harmonic_vortex(
    grid::StructuredGrid{2,<:CartesianGeometry{2,T}};
    Γ::Real = one(T),
    center::Tuple{<:Real,<:Real} = _domain_center(grid),
) where {T}
    xs, ys = grid.coords_axes
    cx, cy = T(center[1]), T(center[2])
    g = T(Γ) / T(2π)
    u = zeros(T, length(xs), length(ys))
    v = zeros(T, length(xs), length(ys))
    @inbounds for j in eachindex(ys), i in eachindex(xs)
        dx = xs[i] - cx
        dy = ys[j] - cy
        r2 = dx^2 + dy^2
        r2 == 0 && continue
        u[i, j] = -g * dy / r2
        v[i, j] = g * dx / r2
    end
    return u, v
end

"""
    harmonic_source(grid; q=1.0, center=domain center) → (u, v)

Point-source velocity field `u = q/(2π) · (x−x₀, y−y₀) / r²`. Away from the singular
center it is both curl-free and divergence-free — purely *harmonic* — with net outward flux
`q` through any loop enclosing the center. Complementary to [`harmonic_vortex`](@ref): the
flux mode of the harmonic subspace on a multiply-connected domain.
"""
function harmonic_source(
    grid::StructuredGrid{2,<:CartesianGeometry{2,T}};
    q::Real = one(T),
    center::Tuple{<:Real,<:Real} = _domain_center(grid),
) where {T}
    xs, ys = grid.coords_axes
    cx, cy = T(center[1]), T(center[2])
    g = T(q) / T(2π)
    u = zeros(T, length(xs), length(ys))
    v = zeros(T, length(xs), length(ys))
    @inbounds for j in eachindex(ys), i in eachindex(xs)
        dx = xs[i] - cx
        dy = ys[j] - cy
        r2 = dx^2 + dy^2
        r2 == 0 && continue
        u[i, j] = g * dx / r2
        v[i, j] = g * dy / r2
    end
    return u, v
end
