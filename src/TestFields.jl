"""
    TestFields.jl — Synthetic velocity field generators for testing and examples.

Provides analytically known velocity fields with prescribed rotational and/or
divergent components, enabling verification that the Helmholtz decomposition
correctly recovers each component.
"""

export taylor_green_vortex, point_source_sink, rankine_vortex_with_source
export rossby_wave, kelvin_ekman_flow

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
    grid::StructuredGrid{G,T};
    A::Real = one(T),
    kx::Real = T(2π) / (grid.lon[end] - grid.lon[1] + grid.lon[2] - grid.lon[1]),
    ky::Real = T(2π) / (grid.lat[end] - grid.lat[1] + grid.lat[2] - grid.lat[1])
) where {T<:AbstractFloat, G<:CartesianGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    A_T = T(A)
    kx_T = T(kx)
    ky_T = T(ky)

    u = Matrix{T}(undef, Nlon, Nlat)
    v = Matrix{T}(undef, Nlon, Nlat)

    for j in 1:Nlat
        y = grid.lat[j]
        for i in 1:Nlon
            x = grid.lon[i]
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
    grid::StructuredGrid{G,T};
    A::Real = one(T),
    kx::Real = T(2π) / (grid.lon[end] - grid.lon[1] + grid.lon[2] - grid.lon[1]),
    ky::Real = T(2π) / (grid.lat[end] - grid.lat[1] + grid.lat[2] - grid.lat[1])
) where {T<:AbstractFloat, G<:CartesianGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    A_T = T(A)
    kx_T = T(kx)
    ky_T = T(ky)

    u = Matrix{T}(undef, Nlon, Nlat)
    v = Matrix{T}(undef, Nlon, Nlat)

    for j in 1:Nlat
        y = grid.lat[j]
        for i in 1:Nlon
            x = grid.lon[i]
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
    grid::StructuredGrid{G,T};
    A_rot::Real = one(T),
    A_div::Real = T(0.5),
    kx::Real = T(2π) / (grid.lon[end] - grid.lon[1] + grid.lon[2] - grid.lon[1]),
    ky::Real = T(2π) / (grid.lat[end] - grid.lat[1] + grid.lat[2] - grid.lat[1])
) where {T<:AbstractFloat, G<:CartesianGeometry{T}}
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
    grid::StructuredGrid{G,T};
    n::Int = 3,
    m::Int = 2,
    A::Real = one(T)
) where {T<:AbstractFloat, G<:SphericalGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    R = grid.geometry.R
    A_T = T(A)

    u = Matrix{T}(undef, Nlon, Nlat)
    v = Matrix{T}(undef, Nlon, Nlat)

    for j in 1:Nlat
        φ = grid.lat[j]
        cosφ = cos(φ)
        sinφ = sin(φ)
        for i in 1:Nlon
            λ = grid.lon[i]
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
    grid::StructuredGrid{G,T};
    A_rot::Real = one(T),
    A_div::Real = T(0.3),
    n::Int = 3,
    m::Int = 2,
    p::Int = 2,
    q::Int = 1
) where {T<:AbstractFloat, G<:SphericalGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    R = grid.geometry.R
    A_div_T = T(A_div)

    # Rotational part
    _, _, u_rot, v_rot, _, _ = rossby_wave(grid; n=n, m=m, A=A_rot)

    # Divergent part from χ = A_div sin(qλ) cosᵖ(φ)
    u_div = Matrix{T}(undef, Nlon, Nlat)
    v_div = Matrix{T}(undef, Nlon, Nlat)

    for j in 1:Nlat
        φ = grid.lat[j]
        cosφ = cos(φ)
        sinφ = sin(φ)
        for i in 1:Nlon
            λ = grid.lon[i]
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
