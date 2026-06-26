"""
    Scattered.jl — Unstructured / scattered-point sample sets.

`StructuredGrid` represents a uniform tensor-product lattice. For genuinely scattered data
(arbitrary sample locations — observation networks, floats, ship/satellite tracks) use
[`ScatteredPoints`](@ref), which stores raw point coordinates and is decomposed through the
non-uniform transforms (FINUFFT on Cartesian, NUFSHT on the sphere): NUFFT analysis →
Leray projection in spectral space → NUFFT synthesis back to the points. Because the points
have no neighbor stencil, there is no finite-difference/SOR path for scattered data — the
spectral extensions are required.

Fields are stored component-last with a single sample axis: a velocity sample set is an
`(M, N)` array (`M` points, `N` components), matching the `(dims..., N)` convention with
`dims = (M,)`.
"""

export ScatteredPoints, npoints

"""
    ScatteredPoints{N, G, T, C} <: AbstractGrid{G, T}

`M` scattered sample locations in an `N`-dimensional geometry.

# Fields
- `geometry::G` — `CartesianGeometry{N}` or `SphericalGeometry`.
- `coords::C` — `(M, N)` coordinates: Cartesian `(x, y, …)` or spherical `(λ, φ)` (radians).
- `box::NTuple{N,T}` — periodic box length per axis (Cartesian); `(2π, π)` for spherical.
"""
struct ScatteredPoints{N,G<:AbstractGeometry,T<:AbstractFloat,C<:AbstractMatrix{T}} <: AbstractGrid{G,T}
    geometry::G
    coords::C
    box::NTuple{N,T}
end

@inline Base.ndims(::ScatteredPoints{N}) where {N} = N

"""
    npoints(pts::ScatteredPoints) -> Int

Number of sample points `M`.
"""
@inline npoints(pts::ScatteredPoints) = size(pts.coords, 1)

"""
    ScatteredPoints(geometry::CartesianGeometry{N}, coords; box)

Cartesian scattered points. `coords` is `(M, N)`; `box` is the periodic domain length per
axis (defaults to the coordinate span, rounded out by one mean spacing).
"""
function ScatteredPoints(
    geometry::CartesianGeometry{N,T},
    coords::AbstractMatrix;
    box::Union{Nothing,NTuple{N,<:Real}} = nothing,
) where {N,T}
    size(coords, 2) == N || throw(DimensionMismatch("coords must be (M, $N), got $(size(coords))"))
    coordsT = convert(Matrix{T}, coords)
    b = box === nothing ? ntuple(d -> _default_box(view(coordsT, :, d)), Val(N)) : map(T, box)
    return ScatteredPoints{N,typeof(geometry),T,typeof(coordsT)}(geometry, coordsT, b)
end

function _default_box(x::AbstractVector{T}) where {T}
    lo, hi = extrema(x)
    span = hi - lo
    n = length(x)
    return n > 1 ? span * T(n) / T(n - 1) : (span > 0 ? span : one(T))
end

"""
    ScatteredPoints(geometry::SphericalGeometry, lons, lats)

Spherical scattered points from per-point longitude/latitude vectors (radians).
"""
function ScatteredPoints(geometry::SphericalGeometry{T}, lons::AbstractVector, lats::AbstractVector) where {T}
    length(lons) == length(lats) || throw(DimensionMismatch("lons and lats must have equal length"))
    coords = Matrix{T}(undef, length(lons), 2)
    coords[:, 1] .= lons
    coords[:, 2] .= lats
    return ScatteredPoints{2,typeof(geometry),T,typeof(coords)}(geometry, coords, (T(2π), T(π)))
end

# Scattered points always route to the non-uniform (irregular) spectral solver.
function _resolve_auto_solver(pts::ScatteredPoints{N,G}) where {N,G<:AbstractGeometry}
    key = G <: SphericalGeometry ? :spherical_irregular : :cartesian_irregular
    haskey(_SPECTRAL_SOLVERS, key) && return _SPECTRAL_SOLVERS[key]()
    pkg = G <: SphericalGeometry ? "NUFSHT" : "FINUFFT"
    throw(ArgumentError("scattered-point decomposition requires the $pkg extension (`using $pkg`)."))
end

# Stack scalar component vectors into the (M, ncomp) component-last layout.
_stack_components(::ScatteredPoints, comps::Vararg{AbstractVector}) = reduce(hcat, comps)
