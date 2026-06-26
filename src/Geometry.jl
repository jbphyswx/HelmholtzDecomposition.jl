"""
    Geometry.jl — Coordinate system abstractions for Helmholtz decomposition.

Defines abstract and concrete geometry types for Cartesian (any dimension `N`) and
spherical (a 2-surface) coordinate systems. These are fully standalone types with no
external package dependencies beyond `LinearAlgebra`/`StaticArrays`.
"""

using LinearAlgebra: LinearAlgebra
using StaticArrays: StaticArrays, SVector

export AbstractGeometry, CartesianGeometry, SphericalGeometry
export distance, area_element, cell_measure, ndims_space
export to_planetary_cartesian, from_planetary_cartesian

"""
    AbstractGeometry{T<:AbstractFloat}

Abstract supertype for all coordinate systems and geometry metrics.

# Type Parameters
- `T`: Floating point type (`Float32`/`Float64`) used for coordinate calculations.

# Implementations
- [`CartesianGeometry{N,T}`](@ref): uniform Cartesian grid spacing in `N` dimensions.
- [`SphericalGeometry{T}`](@ref): spherical coordinates on a planet surface (a 2-surface).
"""
abstract type AbstractGeometry{T<:AbstractFloat} end

"""
    ndims_space(geometry) -> Int

Number of spatial dimensions the geometry describes (`N` for `CartesianGeometry{N}`,
always `2` for `SphericalGeometry`).
"""
function ndims_space end

"""
    CartesianGeometry{N,T<:AbstractFloat}

Uniform Cartesian grid with per-axis spacings stored as an `NTuple{N,T}`.

# Fields
- `spacing::NTuple{N,T}`: grid spacing along each axis (e.g. `(dx, dy, dz)`), in meters.

# Examples
```julia
geom = CartesianGeometry(2000.0, 2000.0)        # 2D, 2km × 2km cells
geom3 = CartesianGeometry(100.0, 100.0, 50.0)   # 3D
cell_measure(geom)                               # 4e6 (m²)
```
"""
struct CartesianGeometry{N,T<:AbstractFloat} <: AbstractGeometry{T}
    spacing::NTuple{N,T}
end

function CartesianGeometry(spacing::Vararg{Real,N}) where {N}
    T = float(promote_type(map(typeof, spacing)...))
    return CartesianGeometry{N,T}(map(x -> convert(T, x), spacing))
end
CartesianGeometry(spacing::Tuple{Vararg{Real}}) = CartesianGeometry(spacing...)
function CartesianGeometry{T}(spacing::Vararg{Real,N}) where {T<:AbstractFloat,N}
    return CartesianGeometry{N,T}(map(x -> convert(T, x), spacing))
end
CartesianGeometry{T}(spacing::Tuple{Vararg{Real}}) where {T<:AbstractFloat} = CartesianGeometry{T}(spacing...)

@inline ndims_space(::CartesianGeometry{N}) where {N} = N

"""
    SphericalGeometry{T<:AbstractFloat}

Spherical coordinates on a planet of radius `R`. Inherently describes a 2-surface
(longitude λ, latitude φ); `ndims_space` is always `2`.

# Fields
- `R::T`: planet radius in meters (default `6.371e6`, Earth).

# Notes
- Coordinates are `(longitude λ, latitude φ)` in radians.
- Uses the Haversine formula for great-circle distances.
"""
struct SphericalGeometry{T<:AbstractFloat} <: AbstractGeometry{T}
    R::T
end

SphericalGeometry() = SphericalGeometry(6.371e6)
SphericalGeometry(R::Real) = SphericalGeometry(float(R))
SphericalGeometry{T}(R::Real) where {T<:AbstractFloat} = SphericalGeometry{T}(convert(T, R))

@inline ndims_space(::SphericalGeometry) = 2

# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

"""
    distance(geo::AbstractGeometry, pt1, pt2)

Distance between two points in the given geometry.
- `CartesianGeometry`: Euclidean norm (any dimension).
- `SphericalGeometry`: great-circle (Haversine) distance.
"""
@inline function distance(::CartesianGeometry{N}, pt1::SVector{N}, pt2::SVector{N}) where {N}
    return LinearAlgebra.norm(pt1 - pt2)
end

@inline function distance(geo::SphericalGeometry{T}, coords1::SVector{2}, coords2::SVector{2}) where {T}
    λ1, φ1 = coords1[1], coords1[2]
    λ2, φ2 = coords2[1], coords2[2]

    dλ = λ2 - λ1
    dφ = φ2 - φ1

    a = sin(dφ / T(2))^2 + cos(φ1) * cos(φ2) * sin(dλ / T(2))^2
    c = T(2) * atan(sqrt(a), sqrt(max(zero(T), one(T) - a)))
    return geo.R * c
end

# ---------------------------------------------------------------------------
# Cell measures (area in 2D, volume in 3D, generally the N-d measure)
# ---------------------------------------------------------------------------

"""
    cell_measure(geo::CartesianGeometry)
    cell_measure(geo::SphericalGeometry, lat, dλ, dφ)

Local grid-cell measure: `prod(spacing)` for Cartesian (area in 2D, volume in 3D, …),
and `R²·cos(lat)·dλ·dφ` for spherical surface cells.
"""
@inline cell_measure(geo::CartesianGeometry) = prod(geo.spacing)

@inline function cell_measure(geo::SphericalGeometry{T}, lat::Real, dλ::Real, dφ::Real) where {T}
    return geo.R^2 * cos(T(lat)) * T(dλ) * T(dφ)
end

# `area_element` retained as an alias for the 2-D-flavoured name used throughout the
# fluid-dynamics literature; identical to `cell_measure`.
@inline area_element(geo::CartesianGeometry) = cell_measure(geo)
@inline area_element(geo::SphericalGeometry{T}, lat::Real, dλ::Real, dφ::Real) where {T} =
    cell_measure(geo, lat, dλ, dφ)

# ---------------------------------------------------------------------------
# Coordinate projections for spherical vector fields
# ---------------------------------------------------------------------------

"""
    to_planetary_cartesian(geo::SphericalGeometry, u_east, u_north, u_vertical, λ, φ)

Convert spherical local velocity components (East, North, Radial) into global planetary
Cartesian `X, Y, Z`.
"""
@inline function to_planetary_cartesian(
    geo::SphericalGeometry{T},
    u_east::Real,
    u_north::Real,
    u_vertical::Real,
    λ::Real,
    φ::Real,
) where {T<:AbstractFloat}
    u_east_T = convert(T, u_east)
    u_north_T = convert(T, u_north)
    u_vertical_T = convert(T, u_vertical)
    λ_T = convert(T, λ)
    φ_T = convert(T, φ)

    sinλ, cosλ = sin(λ_T), cos(λ_T)
    sinφ, cosφ = sin(φ_T), cos(φ_T)

    ux = u_east_T * (-sinλ) + u_north_T * (-sinφ * cosλ) + u_vertical_T * (cosφ * cosλ)
    uy = u_east_T * (cosλ) + u_north_T * (-sinφ * sinλ) + u_vertical_T * (cosφ * sinλ)
    uz = u_north_T * cosφ + u_vertical_T * sinφ

    return SVector{3,T}(ux, uy, uz)
end

@inline function to_planetary_cartesian(
    geo::SphericalGeometry{T},
    u_east::Real,
    u_north::Real,
    λ::Real,
    φ::Real,
) where {T<:AbstractFloat}
    return to_planetary_cartesian(geo, u_east, u_north, zero(T), λ, φ)
end

"""
    from_planetary_cartesian(geo::SphericalGeometry, ux, uy, uz, λ, φ)

Convert global planetary Cartesian velocity components back to local East, North, Radial.
"""
@inline function from_planetary_cartesian(
    geo::SphericalGeometry{T},
    ux::Real,
    uy::Real,
    uz::Real,
    λ::Real,
    φ::Real,
) where {T<:AbstractFloat}
    ux_T = convert(T, ux)
    uy_T = convert(T, uy)
    uz_T = convert(T, uz)
    λ_T = convert(T, λ)
    φ_T = convert(T, φ)

    sinλ, cosλ = sin(λ_T), cos(λ_T)
    sinφ, cosφ = sin(φ_T), cos(φ_T)

    u_east = ux_T * (-sinλ) + uy_T * cosλ
    u_north = ux_T * (-sinφ * cosλ) + uy_T * (-sinφ * sinλ) + uz_T * cosφ
    u_vertical = ux_T * (cosφ * cosλ) + uy_T * (cosφ * sinλ) + uz_T * sinφ

    return SVector{3,T}(u_east, u_north, u_vertical)
end
