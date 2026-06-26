"""
    Grids.jl — Grid abstractions for Helmholtz decomposition.

Defines dimension-generic structured grids with mask support for both Cartesian
(any dimension `N`) and spherical (`N == 2`) geometries. Fully standalone — no
dependencies on other packages beyond `StaticArrays`.
"""

using StaticArrays: StaticArrays, SVector

export AbstractGrid, StructuredGrid
export coords, area, cellmeasure, isactive, grid_geometry, size_tuple

"""
    AbstractGrid{G<:AbstractGeometry, T<:AbstractFloat}

Abstract supertype for all grid architectures (structured, curvilinear, unstructured).
"""
abstract type AbstractGrid{G<:AbstractGeometry,T<:AbstractFloat} end

grid_geometry(grid::AbstractGrid) = grid.geometry

# ---------------------------------------------------------------------------
# Structured Grid
# ---------------------------------------------------------------------------

"""
    StructuredGrid{N, G, T, C, A, B}

Structured grid whose coordinates are 1-D vectors along each of `N` axes (e.g. uniform
Cartesian, or regular longitude–latitude when `N == 2`).

# Fields
- `geometry::G` — geometry (`CartesianGeometry{N}` or `SphericalGeometry`).
- `coords_axes::C` — `NTuple{N}` of 1-D coordinate vectors, one per axis. For spherical
  grids these are `(lon, lat)`.
- `cell_measures::A` — `N`-dimensional array of per-cell measures (area in 2D, volume in 3D).
- `mask::B` — `N`-dimensional boolean active mask (`true` = active, `false` = inactive).
"""
struct StructuredGrid{
    N,
    G<:AbstractGeometry,
    T<:AbstractFloat,
    C<:NTuple{N,AbstractVector{T}},
    A<:AbstractArray{T,N},
    B<:AbstractArray{Bool,N},
} <: AbstractGrid{G,T}
    geometry::G
    coords_axes::C
    cell_measures::A
    mask::B
end

@inline Base.ndims(::StructuredGrid{N}) where {N} = N
@inline size_tuple(grid::StructuredGrid) = size(grid.mask)

"""
    coords(grid, I...) -> SVector{N,T}

Coordinate vector at the integer index `I = (i₁, …, i_N)`.
"""
@inline function coords(grid::StructuredGrid{N,G,T}, I::Vararg{Integer,N}) where {N,G,T}
    return SVector{N,T}(ntuple(d -> grid.coords_axes[d][I[d]], Val(N)))
end

"""
    cellmeasure(grid, I...) -> T

The `N`-dimensional cell measure (area/volume) at index `I`.
"""
@inline cellmeasure(grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {N} = grid.cell_measures[I...]

# `area` retained as the familiar 2-D-flavoured alias of `cellmeasure`.
@inline area(grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {N} = cellmeasure(grid, I...)

"""
    isactive(grid, I...) -> Bool

Whether the cell at index `I` is active (`true`) or masked out (`false`).
"""
@inline isactive(grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {N} = grid.mask[I...]

# Convenience accessors for the 2-D spherical/longitude–latitude case.
@inline _axis(grid::StructuredGrid, d::Integer) = grid.coords_axes[d]

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    StructuredGrid(geometry, axes...; mask=trues(map(length, axes)))

Construct a `StructuredGrid` from per-axis coordinate vectors, computing cell measures
from the geometry. Pass `mask` to mark inactive cells.

For `SphericalGeometry`, supply exactly two axes `(lon, lat)`; cell areas vary with
latitude as `R²·cos(lat)·dλ·dφ`. For `CartesianGeometry{N}`, supply `N` axes; cell
measures are the uniform `prod(spacing)`.

# Examples
```julia
grid = StructuredGrid(CartesianGeometry(1.0, 1.0), 0:0.1:1, 0:0.1:1)
grid3 = StructuredGrid(CartesianGeometry(1.0, 1.0, 1.0), xs, ys, zs)
sgrid = StructuredGrid(SphericalGeometry(), lons, lats; mask=ocean_mask)
```
"""
function StructuredGrid(
    geometry::G,
    axes::Vararg{AbstractVector,N};
    mask::Union{Nothing,AbstractArray{Bool,N}} = nothing,
) where {T<:AbstractFloat,G<:AbstractGeometry{T},N}
    _check_geometry_dims(geometry, Val(N))

    axes_T = ntuple(d -> convert(Vector{T}, collect(axes[d])), Val(N))
    dims = ntuple(d -> length(axes_T[d]), Val(N))
    mask_arr = mask === nothing ? trues(dims) : mask
    size(mask_arr) == dims ||
        throw(DimensionMismatch("mask size $(size(mask_arr)) does not match axis lengths $dims"))

    measures = _cell_measures(geometry, axes_T, dims)

    return StructuredGrid{N,G,T,typeof(axes_T),typeof(measures),typeof(mask_arr)}(
        geometry, axes_T, measures, mask_arr,
    )
end

@inline _check_geometry_dims(::CartesianGeometry{N}, ::Val{N}) where {N} = nothing
_check_geometry_dims(::CartesianGeometry{M}, ::Val{N}) where {M,N} =
    throw(DimensionMismatch("CartesianGeometry{$M} requires $M coordinate axes, got $N"))
_check_geometry_dims(::SphericalGeometry, ::Val{2}) = nothing
_check_geometry_dims(::SphericalGeometry, ::Val{N}) where {N} =
    throw(DimensionMismatch("SphericalGeometry describes a 2-surface; supply 2 axes (lon, lat), got $N"))

# Cartesian: uniform measure everywhere.
function _cell_measures(geometry::CartesianGeometry{N,T}, ::NTuple{N}, dims::NTuple{N,Int}) where {N,T}
    measures = Array{T,N}(undef, dims)
    fill!(measures, cell_measure(geometry))
    return measures
end

# Spherical (N == 2): measure varies with latitude band.
function _cell_measures(geometry::SphericalGeometry{T}, axes::NTuple{2}, dims::NTuple{2,Int}) where {T}
    lon, lat = axes
    Nlon, Nlat = dims
    dλ = Nlon > 1 ? lon[2] - lon[1] : zero(T)
    dφ = Nlat > 1 ? lat[2] - lat[1] : zero(T)
    measures = Matrix{T}(undef, Nlon, Nlat)
    for j in 1:Nlat
        A = cell_measure(geometry, lat[j], dλ, abs(dφ))
        for i in 1:Nlon
            measures[i, j] = A
        end
    end
    return measures
end
