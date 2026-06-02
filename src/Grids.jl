"""
    Grids.jl — Grid abstractions for Helmholtz decomposition.

Defines structured grid types with mask support for both Cartesian and spherical
geometries. Fully standalone — no dependencies on other packages.
"""

using StaticArrays: StaticArrays, SVector

export AbstractGrid, StructuredGrid
export coords, area, iswet, grid_geometry, size_tuple

"""
    AbstractGrid{G<:AbstractGeometry, T<:AbstractFloat}

Abstract supertype for all grid architectures (structured, curvilinear, unstructured).
"""
abstract type AbstractGrid{G<:AbstractGeometry, T<:AbstractFloat} end

grid_geometry(grid::AbstractGrid) = grid.geometry

# ---------------------------------------------------------------------------
# Structured Grid
# ---------------------------------------------------------------------------

"""
    StructuredGrid{G, T, V, M, B}

Structured grid where coordinates are 1D vectors along each axis (e.g. regular
latitude-longitude or uniform Cartesian).

# Fields
- `geometry::G` — Geometry type (`CartesianGeometry` or `SphericalGeometry`)
- `lon::V` — 1D coordinate vector along X/λ
- `lat::V` — 1D coordinate vector along Y/φ
- `areas::M` — 2D cell areas (Nlon × Nlat)
- `mask::B` — 2D active mask (true=water/active, false=land/inactive)
"""
struct StructuredGrid{
    G<:AbstractGeometry,
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    B<:AbstractMatrix{Bool}
} <: AbstractGrid{G, T}
    geometry::G
    lon::V
    lat::V
    areas::M
    mask::B
end

size_tuple(grid::StructuredGrid) = size(grid.mask)

@inline function coords(grid::StructuredGrid{G,T}, i::Integer, j::Integer) where {G,T}
    return SVector{2,T}(grid.lon[i], grid.lat[j])
end

@inline area(grid::StructuredGrid, i::Integer, j::Integer) = grid.areas[i, j]
@inline iswet(grid::StructuredGrid, i::Integer, j::Integer) = grid.mask[i, j]

"""
    StructuredGrid(geometry, lon, lat, mask)

Construct a `StructuredGrid` with automatically computed cell areas.

# Arguments
- `geometry::AbstractGeometry{T}` — Coordinate system
- `lon::AbstractVector` — X/longitude coordinates
- `lat::AbstractVector` — Y/latitude coordinates
- `mask::AbstractMatrix{Bool}` — Active cell mask (true=active)
"""
function StructuredGrid(
    geometry::G,
    lon::AbstractVector,
    lat::AbstractVector,
    mask::AbstractMatrix{Bool}
) where {
    T<:AbstractFloat,
    G<:AbstractGeometry{T}
}
    lon_T = convert(Vector{T}, lon)
    lat_T = convert(Vector{T}, lat)

    Nlon = length(lon_T)
    Nlat = length(lat_T)

    areas = Matrix{T}(undef, Nlon, Nlat)

    if G <: CartesianGeometry{T}
        A = area_element(geometry)
        fill!(areas, A)
    else
        dλ = Nlon > 1 ? lon_T[2] - lon_T[1] : T(0)
        dφ = Nlat > 1 ? lat_T[2] - lat_T[1] : T(0)
        for j in 1:Nlat
            A = area_element(geometry, lat_T[j], dλ, abs(dφ))
            for i in 1:Nlon
                areas[i, j] = A
            end
        end
    end

    return StructuredGrid{G, T, Vector{T}, Matrix{T}, Matrix{Bool}}(
        geometry, lon_T, lat_T, areas, mask
    )
end

"""
    StructuredGrid(geometry, lon, lat)

Construct a `StructuredGrid` with no land mask (all cells active).
"""
function StructuredGrid(
    geometry::G,
    lon::AbstractVector,
    lat::AbstractVector
) where {
    T<:AbstractFloat,
    G<:AbstractGeometry{T}
}
    mask = trues(length(lon), length(lat))
    return StructuredGrid(geometry, lon, lat, mask)
end
