"""
    Geometry.jl — Coordinate system abstractions for Helmholtz decomposition.

Defines abstract and concrete geometry types for Cartesian and spherical coordinate
systems. These are fully standalone types with no external package dependencies.
"""

using LinearAlgebra: LinearAlgebra
using StaticArrays: StaticArrays, SVector

export AbstractGeometry, CartesianGeometry, SphericalGeometry
export distance, area_element, to_planetary_cartesian, from_planetary_cartesian

"""
    AbstractGeometry{T<:AbstractFloat}

Abstract supertype for all coordinate systems and geometry metrics.

# Type Parameters
- `T`: Floating point type (Float32 or Float64) for coordinate calculations

# Implementations
- [`CartesianGeometry{T}`](@ref): Cartesian coordinates with uniform grid spacing
- [`SphericalGeometry{T}`](@ref): Spherical coordinates on a planet surface
"""
abstract type AbstractGeometry{T<:AbstractFloat} end

"""
    CartesianGeometry{T<:AbstractFloat}

Cartesian coordinates with grid spacings `dx`, `dy`, and optionally `dz`.

# Fields
- `dx::T`: Grid spacing in x-direction (meters)
- `dy::T`: Grid spacing in y-direction (meters)
- `dz::T`: Grid spacing in z-direction (meters), zero for 2D grids

# Examples
```julia
geom = CartesianGeometry(2000.0, 2000.0)  # 2km × 2km grid
area = area_element(geom)  # Returns 4e6 m²
```
"""
struct CartesianGeometry{T<:AbstractFloat} <: AbstractGeometry{T}
    dx::T
    dy::T
    dz::T
end

CartesianGeometry(dx::T, dy::T) where {T<:AbstractFloat} = CartesianGeometry{T}(dx, dy, zero(T))
CartesianGeometry{T}(dx, dy) where {T<:AbstractFloat} = CartesianGeometry{T}(convert(T, dx), convert(T, dy), zero(T))

"""
    SphericalGeometry{T<:AbstractFloat}

Spherical coordinates on a planet of radius `R`.

# Fields
- `R::T`: Planet radius in meters (default: 6.371e6 for Earth)

# Notes
- Coordinates are (longitude λ, latitude φ) in radians
- Uses Haversine formula for great-circle distance calculations

# Examples
```julia
geom = SphericalGeometry()  # Earth-like sphere
```
"""
struct SphericalGeometry{T<:AbstractFloat} <: AbstractGeometry{T}
    R::T
end

SphericalGeometry() = SphericalGeometry(6.371e6)

# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

"""
    distance(geo::AbstractGeometry, pt1, pt2)

Calculate the distance between two points in the given geometry.
- For `CartesianGeometry`, this is the Euclidean norm.
- For `SphericalGeometry`, this is the great-circle (Haversine) distance.
"""
@inline function distance(::CartesianGeometry{T}, pt1::SVector{N,T}, pt2::SVector{N,T}) where {N,T}
    return LinearAlgebra.norm(pt1 - pt2)
end

@inline function distance(geo::SphericalGeometry{T}, coords1::SVector{2,T}, coords2::SVector{2,T}) where {T}
    λ1, φ1 = coords1[1], coords1[2]
    λ2, φ2 = coords2[1], coords2[2]

    dλ = λ2 - λ1
    dφ = φ2 - φ1

    a = sin(dφ / T(2))^2 + cos(φ1) * cos(φ2) * sin(dλ / T(2))^2
    c = T(2) * atan(sqrt(a), sqrt(max(zero(T), one(T) - a)))
    return geo.R * c
end

# ---------------------------------------------------------------------------
# Area Elements
# ---------------------------------------------------------------------------

"""
    area_element(geo::CartesianGeometry)
    area_element(geo::SphericalGeometry, lat, dλ, dφ)

Compute local grid cell area.
"""
@inline area_element(geo::CartesianGeometry{T}) where {T} = geo.dx * geo.dy

@inline function area_element(geo::SphericalGeometry{T}, lat::T, dλ::T, dφ::T) where {T}
    return geo.R^2 * cos(lat) * dλ * dφ
end

# ---------------------------------------------------------------------------
# Coordinate Projections for Spherical Vector Fields
# ---------------------------------------------------------------------------

"""
    to_planetary_cartesian(geo::SphericalGeometry, u_east, u_north, u_vertical, λ, φ)

Convert spherical local velocity components (East, North, Radial) into global planetary
Cartesian X, Y, Z. Used by the direct (approximate) filtering approach.
"""
@inline function to_planetary_cartesian(
    geo::SphericalGeometry{T},
    u_east::Real,
    u_north::Real,
    u_vertical::Real,
    λ::Real,
    φ::Real
) where {T<:AbstractFloat}
    u_east_T = convert(T, u_east)
    u_north_T = convert(T, u_north)
    u_vertical_T = convert(T, u_vertical)
    λ_T = convert(T, λ)
    φ_T = convert(T, φ)

    sinλ, cosλ = sin(λ_T), cos(λ_T)
    sinφ, cosφ = sin(φ_T), cos(φ_T)

    ux = u_east_T * (-sinλ) + u_north_T * (-sinφ * cosλ) + u_vertical_T * (cosφ * cosλ)
    uy = u_east_T * (cosλ)  + u_north_T * (-sinφ * sinλ) + u_vertical_T * (cosφ * sinλ)
    uz =                      u_north_T * cosφ           + u_vertical_T * sinφ

    return SVector{3,T}(ux, uy, uz)
end

@inline function to_planetary_cartesian(
    geo::SphericalGeometry{T},
    u_east::Real,
    u_north::Real,
    λ::Real,
    φ::Real
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
    φ::Real
) where {T<:AbstractFloat}
    ux_T = convert(T, ux)
    uy_T = convert(T, uy)
    uz_T = convert(T, uz)
    λ_T = convert(T, λ)
    φ_T = convert(T, φ)

    sinλ, cosλ = sin(λ_T), cos(λ_T)
    sinφ, cosφ = sin(φ_T), cos(φ_T)

    u_east     = ux_T * (-sinλ) + uy_T * cosλ
    u_north    = ux_T * (-sinφ * cosλ) + uy_T * (-sinφ * sinλ) + uz_T * cosφ
    u_vertical = ux_T * (cosφ * cosλ)  + uy_T * (cosφ * sinλ)  + uz_T * sinφ

    return SVector{3,T}(u_east, u_north, u_vertical)
end
