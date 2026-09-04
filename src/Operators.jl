"""
    Operators.jl — One consistent family of discrete operators.

The rotational part of an `N`-dimensional Helmholtz decomposition is encoded by an antisymmetric
rotation-potential matrix `R` with `N(N-1)/2` independent components (Glötzl & Richters, 2023).
In 2D that component is the streamfunction `ψ`; in 3D the three are the Hodge dual of the vector
potential `A`.

# Why the operators are defined as adjoints of one another

A decomposition is a decomposition only if the operators around it agree. One gradient `G` is
chosen — compact, cell centres to faces — and the rest are *defined* from it:

    D = −G*        (adjoint under the cell-measure inner product)
    L = D G        (so L = −G*G)

Then, discretely and not to truncation order: `L` is symmetric negative-semidefinite, which is
what makes conjugate gradients and multigrid applicable at all; `div(u_div) = L χ = D u` exactly;
and `⟨u_div, u_rot⟩ = 0`, so `harmonic_fraction` measures domain topology rather than
disagreement between three separately-chosen stencils.

# The metric enters exactly once

`d` is metric-free: `dχ` on an edge is `χ_head − χ_tail`, no division. `d∘d = 0` — and hence
`curl(grad χ) = 0` — is an identity of *that* operator, so a difference already divided by a
physical gap does not satisfy it wherever the metric varies along the differenced direction, as
it does on any curved geometry. All metric therefore lives in one place, the Hodge factor
`c_f = A_f / g_f`, with the areas and gaps taken from `FlowGeometries` and evaluated **at the
face**: on a sphere `measure(I)/width(I)` is not the area of the face between two φ-neighbours.

Storage and counts are in `Staggering.jl`.
"""

# ---------------------------------------------------------------------------
# Execution backend
# ---------------------------------------------------------------------------

"""
    execution_backend(backend)

The object `FlowGeometries.Execution` dispatches on, given a `ComputationalBackends` one.

The two libraries answer different questions: `ComputationalBackends` names *what kind* of
execution is wanted, while `Execution`'s device methods dispatch on the KernelAbstractions device
object itself. A `GPUBackend` therefore hands over the device it names; serial and threaded
backends pass through, FlowGeometries' own ComputationalBackends extension having methods for
them. Anything else reaches `run_indices` unchanged and raises a `MethodError` there rather than
running serially in silence.
"""
@inline execution_backend(b::ComputationalBackends.GPUBackend) = b.backend
@inline execution_backend(b) = b

"""
    resolve_execution_backend(backend) -> concrete backend

The backend a single decomposition's own loops run on, resolved once so nothing below has to ask.

`AutoBackend` reads the thread count — the defect that made every documented parallel path dead
at defaults was resolving it to serial unconditionally. A `GPUBackend` is checked here, at plan
time, against whether FlowGeometries' KernelAbstractions extension is actually loaded, so the
error names the package to load rather than surfacing as a `MethodError` from inside a loop.

`DistributedBackend` and `MPIBackend` are refused: they spread *fields across processes*, which is
[`helmholtz_decompose_batch`](@ref)'s axis, not the index space inside one field.
"""
resolve_execution_backend(b::ComputationalBackends.AbstractSerialBackend) = b
resolve_execution_backend(b::ComputationalBackends.AbstractThreadedBackend) = b
resolve_execution_backend(::ComputationalBackends.AbstractAutoBackend) =
    Threads.nthreads() > 1 ? ComputationalBackends.ThreadedBackend() :
                             ComputationalBackends.SerialBackend()

function resolve_execution_backend(b::ComputationalBackends.AbstractGPUBackend)
    Base.get_extension(FlowGeometries, :FlowGeometriesKernelAbstractionsExt) === nothing &&
        throw(ArgumentError(
            "$(typeof(b)) needs FlowGeometries' KernelAbstractions extension, which is not " *
            "loaded. Run `using KernelAbstractions` first (plus the vendor package owning the " *
            "device), or pass `backend = ComputationalBackends.SerialBackend()`."))
    return b
end

resolve_execution_backend(b::ComputationalBackends.AbstractExecutionBackend) = throw(ArgumentError(
    "$(typeof(b)) parallelises over fields, not over the index space within one field. Pass it " *
    "to `helmholtz_decompose_batch` instead; `plan_helmholtz`'s `backend` selects how the loops " *
    "inside a single decomposition run."))

"""
    allocate_zeros(backend, T, dims) -> zeroed array

Every buffer this package owns goes through here, so that a device backend gets device memory
rather than a host array a kernel cannot reach. The default is a host `Array`; the
KernelAbstractions extension adds the device method.
"""
allocate_zeros(::Any, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

"""
    to_backend(backend, x) -> x on the backend's memory

Move an already-built array — or a struct of them — to where `backend` executes.

The plan's coefficients and face metrics are built by walking the grid's geometry, which is host
work done once; only their *results* are read in the inner loops. So they are assembled on the
host and moved here, rather than every geometry accessor being made device-callable to build them
in place. Identity by default; the KernelAbstractions extension routes it through `Adapt`.
"""
to_backend(::Any, x) = x

# ---------------------------------------------------------------------------
# Rotation-component bookkeeping
# ---------------------------------------------------------------------------

"""
    n_rotation_components(N) -> Int

`N(N-1)/2`: 0 in 1D, 1 in 2D, 3 in 3D.
"""
@inline n_rotation_components(N::Integer) = (N * (N - 1)) ÷ 2

"""
    rotation_pairs(Val(N)) -> NTuple{P,Tuple{Int,Int}}

The pairs `(a, b)`, `a < b`, lexicographically — the independent components of an antisymmetric
2-tensor, and equally the directions each component is staggered in.
"""
@generated function rotation_pairs(::Val{N}) where {N}
    pairs = Tuple{Int,Int}[]
    for a in 1:(N - 1), b in (a + 1):N
        push!(pairs, (a, b))
    end
    return Expr(:tuple, (Expr(:tuple, p[1], p[2]) for p in pairs)...)
end

"""
    rotation_terms(Val(N)) -> NTuple{N,NTuple{N-1,Tuple{Int,Int,Int}}}

For each direction `c`, the `(e, p, s)` triples saying how `u_rot_c` is assembled: difference
component `p` of `R` along direction `e` with sign `s`. Inverting `rotation_pairs` this way turns
the accumulation `u_rot_a −= ∂_b R_ab`, `u_rot_b += ∂_a R_ab` — a *scatter*, where several pairs
write the same face — into a gather, so each face is written exactly once and the loop can be
handed to `Execution` unchanged. Homogeneous, so indexing it with a runtime direction is stable.
"""
@generated function rotation_terms(::Val{N}) where {N}
    pairs = Tuple{Int,Int}[]
    for a in 1:(N - 1), b in (a + 1):N
        push!(pairs, (a, b))
    end
    per_dir = map(1:N) do c
        terms = Expr[]
        for (p, (a, b)) in enumerate(pairs)
            a == c && push!(terms, Expr(:tuple, b, p, -1))
            b == c && push!(terms, Expr(:tuple, a, p, 1))
        end
        Expr(:tuple, terms...)
    end
    return Expr(:tuple, per_dir...)
end

@inline function _component(U::AbstractArray{<:Any,M}, c::Integer, ::Val{N}) where {M,N}
    return @view U[ntuple(_ -> Colon(), Val(N))..., c]
end

# ---------------------------------------------------------------------------
# Face metrics
# ---------------------------------------------------------------------------

# Physical extent of cell `I` along `e`: coordinate width times that direction's scale factor —
# `1` on a Cartesian grid, `R·cosφ` / `R` on a spherical one.
@inline function _physical_width(grid, I::CartesianIndex{N}, e::Integer) where {N}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    p = FlowGeometries.Grids.coords(Tuple, grid, Tuple(I)...)
    h = FlowGeometries.Geometry.scale_factors(geo, p)[e]
    return abs(h) * FlowGeometries.Grids.cell_width(grid, e, I[e])
end

@inline function _signed_half_gap(grid, xi::T, xj::T, d::Integer) where {T}
    δ = xj - xi
    if FlowGeometries.Grids.isperiodic(grid, d)
        p = T(FlowGeometries.Grids.period(grid, d))
        abs(δ) > p / 2 && (δ -= sign(δ) * p)   # across the seam the raw difference is the long way
    end
    return δ / 2
end

# Coordinates of face `F` along `d`: the midpoint of the two cells it separates, or half a cell
# out from the only cell it has. The metric is read here, not at either cell, because on a curved
# geometry the two disagree.
@inline function _face_point(grid, Cl, Cu, d::Integer, ::Val{N}) where {N}
    if Cl !== nothing && Cu !== nothing
        xl = FlowGeometries.Grids.coords(Tuple, grid, Tuple(Cl)...)
        xu = FlowGeometries.Grids.coords(Tuple, grid, Tuple(Cu)...)
        return ntuple(e -> e == d ? (xl[e] + _signed_half_gap(grid, xl[e], xu[e], d)) : xl[e], Val(N))
    end
    C = Cu === nothing ? Cl : Cu
    x = FlowGeometries.Grids.coords(Tuple, grid, Tuple(C)...)
    w = FlowGeometries.Grids.cell_width(grid, d, C[d]) / 2
    # The face lies below `Cu` and above `Cl`.
    return ntuple(e -> e == d ? (Cu === nothing ? x[e] + w : x[e] - w) : x[e], Val(N))
end

"""
    geometric_face_area(grid, F, d, T) -> T

Area of face `F` normal to `d`: the product of the physical cell widths in every other
direction, evaluated at the face. Purely geometric — whether flux crosses it is
[`face_area`](@ref)'s question.
"""
@inline function geometric_face_area(grid, F::CartesianIndex{N}, d::Integer, ::Type{T}) where {N,T}
    Cl = cell_below(grid, F, d)
    Cu = cell_above(grid, F, d)
    ref = Cu === nothing ? Cl : Cu
    ref === nothing && return zero(T)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    h = FlowGeometries.Geometry.scale_factors(geo, _face_point(grid, Cl, Cu, d, Val(N)))
    return prod(ntuple(e -> e == d ? one(T) :
                       abs(T(h[e])) * T(FlowGeometries.Grids.cell_width(grid, e, ref[e])), Val(N)))
end

"""
    face_area(grid, F, d, bc, T) -> T

The area that carries flux across face `F` — the coefficient the operators use, and where the
boundary condition lives, because on a flux-form operator that is what a boundary condition *is*:

- between two active cells → the geometric area;
- against a masked-out cell → `0`. No flux crosses, which is at once the no-flux condition, the
  mask treatment, and the reason `D = −G*` survives both;
- at the outer edge of a bounded direction → `0` under [`Neumann`](@ref), the geometric area
  under [`Dirichlet`](@ref), whose ghost value beyond the edge is zero.

On a covering spherical grid the poles need no special case: the φ-face area carries the `cos φ`
that vanishes there, so the metric closes the surface itself.
"""
@inline function face_area(grid, F::CartesianIndex{N}, d::Integer, bc, ::Type{T}) where {N,T}
    Cl = cell_below(grid, F, d)
    Cu = cell_above(grid, F, d)
    Cl === nothing && Cu === nothing && return zero(T)
    if Cl === nothing || Cu === nothing
        C = Cu === nothing ? Cl : Cu
        @inbounds FlowGeometries.Grids.isactive(grid, Tuple(C)...) || return zero(T)
        return bc isa Dirichlet ? geometric_face_area(grid, F, d, T) : zero(T)
    end
    @inbounds (FlowGeometries.Grids.isactive(grid, Tuple(Cl)...) &&
               FlowGeometries.Grids.isactive(grid, Tuple(Cu)...)) || return zero(T)
    return geometric_face_area(grid, F, d, T)
end

"""
    face_gap(grid, F, d, T) -> T

Physical centre-to-centre distance across face `F`: the mean of the two cells' physical widths,
or half a cell where the face has only one. Consistent with [`face_area`](@ref) — a great-circle
distance would not be, being shorter than the coordinate line the area is built on.
"""
@inline function face_gap(grid, F::CartesianIndex{N}, d::Integer, ::Type{T}) where {N,T}
    Cl = cell_below(grid, F, d)
    Cu = cell_above(grid, F, d)
    if Cl === nothing || Cu === nothing
        C = Cu === nothing ? Cl : Cu
        C === nothing && return one(T)
        return T(_physical_width(grid, C, d)) / T(2)
    end
    return (T(_physical_width(grid, Cl, d)) + T(_physical_width(grid, Cu, d))) / T(2)
end

@inline cell_measure(grid, I::CartesianIndex, ::Type{T}) where {T} =
    T(FlowGeometries.Grids.measure(grid, Tuple(I)...))

# ---------------------------------------------------------------------------
# Gradient, divergence, Laplacian
# ---------------------------------------------------------------------------

"""
    FaceMetrics{N,T,A}

Every face's flux-carrying area and centre-to-centre gap, evaluated once for a `(grid, boundary)`
pair.

[`face_area`](@ref) and [`face_gap`](@ref) each read the grid's coordinates and evaluate the
geometry's scale factors *at the face* — on a curved grid, trigonometry per face per direction.
None of it depends on the field, yet the operators call them inside their loops, so a decomposition
recomputed the same metric `2N + 4P` times per field and again for every field of a batch. Reduced
here to two array reads.
"""
struct FaceMetrics{N,T,A<:NTuple{N,AbstractArray{T,N}}}
    area::A
    gap::A
end

"""
    SeparableMetrics() / DenseMetrics()

How a face metric is stored, chosen from the geometry and mask **types** so the choice folds at
compile time.

A face area is `∏_{e≠d} |h_e(x_F)|·w_e`. Every `h_e` is `1` on a Cartesian geometry; on a spherical
one `h_λ = R cos φ` and `h_φ = R`, and each term is a function of a single index. Both are products
of one factor per axis, so `∑_d n_d` numbers carry what `∏_d n_d` numbers held. A no-flux edge is
the `d`-factor vanishing at its two end entries, so a boundary condition preserves the form. A mask
breaks it, and there the dense array is the representation.
"""
struct SeparableMetrics end
struct DenseMetrics end

@inline metric_layout(grid::FlowGeometries.Grids.StructuredGrid) =
    _metric_layout(FlowGeometries.Grids.grid_geometry(grid), FlowGeometries.Grids.mask(grid))

@inline _metric_layout(::FlowGeometries.Geometry.AbstractCartesianGeometry,
                       ::FlowGeometries.Grids.AllActive) = SeparableMetrics()
@inline _metric_layout(::FlowGeometries.Geometry.AbstractSphericalGeometry,
                       ::FlowGeometries.Grids.AllActive) = SeparableMetrics()
@inline _metric_layout(_, _) = DenseMetrics()

# An index where `f` is nonzero, to take the axis factors through. The centre is tried first: a
# vanishing entry sits at the end of an axis.
function _separable_reference(f::F, dims::NTuple{N,Int}) where {F,N}
    mid = CartesianIndex(ntuple(e -> cld(dims[e], 2), Val(N)))
    iszero(f(mid)) || return mid
    for I in CartesianIndices(dims)
        iszero(f(I)) || return I
    end
    return nothing
end

"""
    _factorise(f, dims, ::Type{T}) -> NTuple{N,Vector{T}} or nothing

Axis factors `g_e` with `f(I) = ∏_e g_e(I[e])`, read off `f` along each axis through one reference
index: `O(∑ n_e)` evaluations against `O(∏ n_e)` for the dense array.

With `G_e(i) = f(ref with e → i)` and `P = f(ref)`, expanding gives `∏_e G_e(I[e]) = f(I)·P^{N-1}`,
so dividing the trailing `N-1` factors by `P` leaves `∏_e g_e = f`.

`nothing` where `f` vanishes everywhere, which no reference factors.
"""
function _factorise(f::F, dims::NTuple{N,Int}, ::Type{T}) where {F,N,T}
    ref = _separable_reference(f, dims)
    ref === nothing && return nothing
    P = f(ref)
    return ntuple(Val(N)) do e
        g = Vector{T}(undef, dims[e])
        @inbounds for i in 1:dims[e]
            g[i] = f(CartesianIndex(ntuple(k -> k == e ? i : ref[k], Val(N))))
        end
        e == 1 ? g : (g ./= P; g)
    end
end

"""
    _metric_array(f, dims, layout, ::Type{T}) -> AbstractArray{T,N}

One face metric, held as its axis factors where the layout admits them and densely elsewhere.
Both are `AbstractArray{T,N}` indexed `a[F]`, so the operators read them the same way.

Whether the factored form reproduces `f` is a property of the geometry, checked exhaustively in
`test/operators.jl` against the scalar functions on every grid kind.
"""
function _metric_array(f::F, dims::NTuple{N,Int}, ::SeparableMetrics, ::Type{T}) where {F,N,T}
    g = _factorise(f, dims, T)
    g === nothing && return _metric_array(f, dims, DenseMetrics(), T)
    return FlowGeometries.Grids.SeparableMeasure(g)
end

function _metric_array(f::F, dims::NTuple{N,Int}, ::DenseMetrics, ::Type{T}) where {F,N,T}
    a = zeros(T, dims)
    @inbounds for I in CartesianIndices(dims)
        a[I] = f(I)
    end
    return a
end

"""
    face_metrics(grid::CurvilinearGrid, bc)

Face areas and gaps read off the grid's own cell-vertex arrays.

A curvilinear cell's shape is its own, so no product of axis factors describes it and the arrays
are dense. At `N = 2` a face is the edge between two vertices and its area is their separation; the
gap is the distance between the two cell centres it divides. `FlowGeometries` derives the
curvilinear cell measure from those same vertices, so `L`'s flux balance and the measure it divides
by come from one description of the mesh.

The boundary condition and the mask enter exactly where they do on a rectilinear grid: a face
against an inactive cell carries zero area, and so does a bounded outer edge under `Neumann`.
"""
function face_metrics(grid::FlowGeometries.Grids.CurvilinearGrid{T,G,2}, bc) where {G,T}
    FlowGeometries.Grids.has_corners(grid) || throw(ArgumentError(
        "the flux-form operators build face areas from this CurvilinearGrid's cell-vertex arrays, " *
        "which it did not retain. Rebuild it with `keep_corners = true`, or pass `corners`."))
    k = FlowGeometries.Grids.corners(grid)
    area = ntuple(d -> _curvilinear_area(grid, k, d, bc, T), Val(2))
    gap = ntuple(d -> _curvilinear_gap(grid, d, T), Val(2))
    return FaceMetrics{2,T,typeof(area)}(area, gap)
end

# A curvilinear metric is per cell, so there is nothing to factor.
@inline metric_layout(::FlowGeometries.Grids.CurvilinearGrid) = DenseMetrics()

# Face `F` normal to `d` spans the two vertices at `F` and at `F` stepped once along the transverse
# direction. Its area is their separation; `face_area`'s open/closed rules decide whether it counts.
function _curvilinear_area(grid, k, d::Integer, bc, ::Type{T}) where {T}
    e = d == 1 ? 2 : 1
    dims = face_dims(grid, d)
    a = zeros(T, dims)
    @inbounds for F in CartesianIndices(dims)
        iszero(_face_openness(grid, F, d, bc, T)) && continue
        Q = CartesianIndex(ntuple(j -> j == e ? F[j] + 1 : F[j], Val(2)))
        a[F] = sqrt((T(k[1][Q]) - T(k[1][F]))^2 + (T(k[2][Q]) - T(k[2][F]))^2)
    end
    return a
end

# Centre-to-centre distance across the face, or half a cell where the face has one side.
function _curvilinear_gap(grid, d::Integer, ::Type{T}) where {T}
    c = FlowGeometries.Grids.coordinates(grid)
    dims = face_dims(grid, d)
    g = ones(T, dims)
    @inbounds for F in CartesianIndices(dims)
        Cl = cell_below(grid, F, d)
        Cu = cell_above(grid, F, d)
        if Cl !== nothing && Cu !== nothing
            g[F] = sqrt((T(c[1][Cu]) - T(c[1][Cl]))^2 + (T(c[2][Cu]) - T(c[2][Cl]))^2)
        else
            C = Cu === nothing ? Cl : Cu
            C === nothing && continue
            # Half the cell's own extent along `d`, from its two bounding vertices.
            Q = CartesianIndex(ntuple(j -> j == d ? C[j] + 1 : C[j], Val(2)))
            kk = FlowGeometries.Grids.corners(grid)
            g[F] = sqrt((T(kk[1][Q]) - T(kk[1][C]))^2 + (T(kk[2][Q]) - T(kk[2][C]))^2) / 2
        end
    end
    return g
end

# Whether a face carries flux at all: the mask and the boundary condition, with no geometry.
@inline function _face_openness(grid, F::CartesianIndex{N}, d::Integer, bc, ::Type{T}) where {N,T}
    Cl = cell_below(grid, F, d)
    Cu = cell_above(grid, F, d)
    Cl === nothing && Cu === nothing && return zero(T)
    if Cl === nothing || Cu === nothing
        C = Cu === nothing ? Cl : Cu
        @inbounds FlowGeometries.Grids.isactive(grid, Tuple(C)...) || return zero(T)
        return bc isa Dirichlet ? one(T) : zero(T)
    end
    @inbounds (FlowGeometries.Grids.isactive(grid, Tuple(Cl)...) &&
               FlowGeometries.Grids.isactive(grid, Tuple(Cu)...)) || return zero(T)
    return one(T)
end

function face_metrics(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, bc) where {G,T,N}
    layout = metric_layout(grid)
    area = ntuple(Val(N)) do d
        _metric_array(F -> face_area(grid, F, d, bc, T), face_dims(grid, d), layout, T)
    end
    gap = ntuple(Val(N)) do d
        _metric_array(F -> face_gap(grid, F, d, T), face_dims(grid, d), layout, T)
    end
    return FaceMetrics{N,T,typeof(area)}(area, gap)
end

"""
    gradient!(g, χ, grid, bc) -> g

`g[d][F] = (χ_above − χ_below) / gap` on every face. A face with area but only one cell is a
Dirichlet edge, whose ghost value is zero — which is exactly what the condition says.
"""
function gradient!(g::NTuple{N,<:AbstractArray}, χ, grid::FaceIndexedGrid{T,G,N},
                  bc, fm::FaceMetrics{N,T} = face_metrics(grid, bc); backend = ComputationalBackends.SerialBackend()) where {G,T,N}
    for d in 1:N
        gd, ad, gpd = g[d], fm.area[d], fm.gap[d]
        fcart = CartesianIndices(face_dims(grid, d))
        FlowGeometries.Execution.run_indices(length(gd), backend) do lin
            @inbounds begin
                F = fcart[lin]
                gd[F] = iszero(ad[F]) ? zero(T) :
                        (_cell_value(χ, grid, F, d, true, T) -
                         _cell_value(χ, grid, F, d, false, T)) / gpd[F]
            end
            return nothing
        end
    end
    return g
end

# The value on one side of a face, or the zero ghost where that side has no cell — which is the
# Dirichlet condition, and the only case where a face carries area without two neighbours.
#
# The `nothing` stays *inside* this function on purpose: both branches return `T`, so the small
# union never escapes into the caller's loop, where it would be a type instability and, in a
# kernel, an unsupported one.
@inline function _cell_value(χ, grid, F::CartesianIndex{N}, d::Integer, upper::Bool,
                             ::Type{T}) where {N,T}
    C = upper ? cell_above(grid, F, d) : cell_below(grid, F, d)
    C === nothing && return zero(T)
    return @inbounds T(χ[C])
end

"""
    divergence!(δ, v, grid, bc) -> δ

`δ[I] = (1/V_I) Σ_d (A·v)[face above] − (A·v)[face below]` — the negative adjoint of
[`gradient!`](@ref) under the cell-measure inner product, which is what makes `L = D G` symmetric.
"""
function divergence!(δ, v::NTuple{N,<:AbstractArray}, grid::FaceIndexedGrid{T,G,N},
                     bc, fm::FaceMetrics{N,T} = face_metrics(grid, bc); backend = ComputationalBackends.SerialBackend()) where {G,T,N}
    msk = FlowGeometries.Grids.mask(grid)
    meas = FlowGeometries.Grids.measure(grid)
    cart = CartesianIndices(size(grid))
    area = fm.area
    # One write per index, every direction accumulated inside it. That is the shape a device launch
    # can express — with `KernelAbstractions` loaded, `run_indices` becomes a kernel — and it is
    # also strictly less work than before: `N + 2` passes over the grid collapse to one, absorbing
    # the `fill!` and the separate division by the measure.
    #
    # The grid is captured directly rather than unpacked into plain tuples: `FlowGeometries` adapts
    # a `StructuredGrid` to a device (coordinates, measure and mask move; `SeparableMeasure` travels
    # as its factors), and `isperiodic` is a type-parameter read, so `face_above` is already
    # kernel-safe.
    FlowGeometries.Execution.run_indices(length(δ), backend) do lin
        @inbounds begin
            I = cart[lin]
            if msk[I]
                acc = zero(T)
                for d in 1:N
                    vd, ad = v[d], area[d]
                    Fhi = face_above(grid, I, d)
                    acc += ad[Fhi] * vd[Fhi] - ad[I] * vd[I]
                end
                δ[I] = acc / meas[I]
            end
        end
        return nothing
    end
    return δ
end

"""
    laplacian!(out, χ, grid, bc, scratch) -> out

`L χ = D G χ`, from exactly the two operators above, so a solver inverts the same `L` the
decomposition differentiates with.
"""
function laplacian!(out, χ, grid::FlowGeometries.Grids.StructuredGrid, bc, scratch::NTuple)
    gradient!(scratch, χ, grid, bc)
    return divergence!(out, scratch, grid, bc)
end

# ---------------------------------------------------------------------------
# Curl and the rotational reconstruction
# ---------------------------------------------------------------------------

"""
    curl!(W, v, grid, bc) -> W

`W_ab = ∂_a v_b − ∂_b v_a` on the `(a,b)` corner, from face-normal velocity.

The circulation `v·g` is differenced rather than `v` itself, so what is differenced is the
metric-free `d`; the metric re-enters once, in the division at the end. A corner is included only
when all four faces bounding it are open — a loop running half through a mask edge has no zero
circulation, and requiring only two faces is what previously broke `curl(grad χ) = 0` on a masked
grid.
"""
function curl!(W::NTuple, v::NTuple{N,<:AbstractArray}, grid::FaceIndexedGrid{T,G,N},
               bc, fm::FaceMetrics{N,T} = face_metrics(grid, bc); backend = ComputationalBackends.SerialBackend()) where {G,T,N}
    for (p, (a, b)) in enumerate(rotation_pairs(Val(N)))
        Wp = W[p]
        va, vb = v[a], v[b]
        aa, ab = fm.area[a], fm.area[b]
        ga, gb_ = fm.gap[a], fm.gap[b]
        qcart = CartesianIndices(corner_dims(grid, a, b))
        FlowGeometries.Execution.run_indices(length(Wp), backend) do lin
            @inbounds begin
                # `Qc` indexes the corner array; `Q` is the face index it sits on, which is what
                # the geometry is walked from.
                Qc = qcart[lin]
                Q = corner_to_face(grid, Qc, a, b)
                Cla = cell_below(grid, Q, a)      # a-cells either side of the a-face at Q[a]
                Cua = cell_above(grid, Q, a)
                Clb = cell_below(grid, Q, b)
                Cub = cell_above(grid, Q, b)
                # A corner is in the complex only when all four faces bounding it are open;
                # requiring two lets a circulation loop run half through a mask edge. Written as a
                # branch rather than an early `continue` because the body is a closure.
                if Cla === nothing || Cua === nothing || Clb === nothing || Cub === nothing
                    Wp[Qc] = zero(T)
                elseif iszero(ab[Cla]) || iszero(ab[Cua]) || iszero(aa[Clb]) || iszero(aa[Cub])
                    Wp[Qc] = zero(T)
                else
                    # The dual cell's extent at the corner. Not `face_gap(grid, Q, …)`: `Q`'s other
                    # staggered slot is a FACE index, so walking to cells from it lands out of range
                    # on a bounded direction and half a cell off on a curved one. `Clb`/`Cla` already
                    # carry a cell index in that slot while keeping the face index in the measured one.
                    circ_b = vb[Cua] * gb_[Cua] - vb[Cla] * gb_[Cla]
                    circ_a = va[Cub] * ga[Cub] - va[Clb] * ga[Clb]
                    Wp[Qc] = (circ_b - circ_a) / (ga[Clb] * gb_[Cla])
                end
            end
            return nothing
        end
    end
    return W
end

"""
    rotational_velocity!(u_rot, R, grid, bc) -> u_rot

`u_rot_a = −Σ_b ∂_b R_ab`, moving each component of `R` from its corner onto the `a`-faces. The
adjoint of [`curl!`](@ref), so `div(u_rot) = 0` holds discretely.
"""
function rotational_velocity!(u_rot::NTuple{N,<:AbstractArray}, R::NTuple,
                              grid::FaceIndexedGrid{T,G,N},
                              bc, fm::FaceMetrics{N,T} = face_metrics(grid, bc);
                              backend = ComputationalBackends.SerialBackend()) where {G,T,N}
    all_terms = rotation_terms(Val(N))
    for c in 1:N
        uc, ac = u_rot[c], fm.area[c]
        terms = all_terms[c]
        fcart = CartesianIndices(face_dims(grid, c))
        FlowGeometries.Execution.run_indices(length(uc), backend) do lin
            @inbounds begin
                F = fcart[lin]
                if iszero(ac[F])
                    uc[F] = zero(T)
                else
                    acc = zero(T)
                    # `u_rot_c` sits on a c-face, whose e-index is a CELL index; the two corners it
                    # reads are the e-FACES either side of that cell, and the distance between them
                    # is that cell's own physical width along e.
                    for (e, p, s) in terms
                        Rp = R[p]
                        # Face indices, then shifted into the corner array. A corner that lands
                        # outside it is one of the dropped outer pair, where `R = 0` — which is
                        # exactly the Dirichlet ghost the dual grid is solved with, so reading
                        # nothing there is the condition rather than an omission.
                        Qlo = face_to_corner(grid, face_below(F, e), c, e)
                        Qhi = face_to_corner(grid, face_above(grid, F, e), c, e)
                        lo = checkbounds(Bool, Rp, Qlo) ? Rp[Qlo] : zero(T)
                        hi = checkbounds(Bool, Rp, Qhi) ? Rp[Qhi] : zero(T)
                        acc += T(s) * (hi - lo) / _transverse_width(grid, F, c, e, T)
                    end
                    uc[F] = acc
                end
            end
            return nothing
        end
    end
    return u_rot
end

# Physical width along `e` at face `F` normal to `d`. `F`'s own `d`-slot is a face index, not a
# valid cell index, so the coordinate width is taken from a cell the face touches — but the scale
# factor is evaluated **at the face**, for the same reason `geometric_face_area` does: on a curved
# metric the two adjacent cells disagree, and reading either one is wrong by half a cell.
@inline function _transverse_width(grid, F::CartesianIndex{N}, d::Integer, e::Integer, ::Type{T}) where {N,T}
    Cl = cell_below(grid, F, d)
    Cu = cell_above(grid, F, d)
    ref = Cu === nothing ? Cl : Cu
    ref === nothing && return one(T)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    h = FlowGeometries.Geometry.scale_factors(geo, _face_point(grid, Cl, Cu, d, Val(N)))[e]
    return abs(T(h)) * T(FlowGeometries.Grids.cell_width(grid, e, ref[e]))
end

# ---------------------------------------------------------------------------
# Collocated <-> staggered
# ---------------------------------------------------------------------------
#
# Velocity arrives and leaves at cell centres, because that is how fields are stored; the
# projection happens on faces, where it is exact. These two are adjoint, so the round trip does
# not bias the split.

"""
    to_faces!(vf, uc, grid, bc) -> vf

Cell-centred velocity `(dims..., N)` to face-normal velocity, averaging the two cells a face
separates. A face of zero area takes nothing; an open boundary face has only one cell to read.
"""
function to_faces!(vf::NTuple{N,<:AbstractArray}, uc, grid::FaceIndexedGrid{T,G,N},
                   bc, fm::FaceMetrics{N,T} = face_metrics(grid, bc); backend = ComputationalBackends.SerialBackend()) where {G,T,N}
    for d in 1:N
        vd, ad = vf[d], fm.area[d]
        ud = _component(uc, d, Val(N))
        fcart = CartesianIndices(face_dims(grid, d))
        FlowGeometries.Execution.run_indices(length(vd), backend) do lin
            @inbounds begin
                F = fcart[lin]
                if iszero(ad[F])
                    vd[F] = zero(T)
                else
                    # A face with only one cell — an open boundary — averages the one it has, so
                    # the weight count is computed rather than assumed to be two.
                    lo = _cell_present(grid, F, d, false)
                    hi = _cell_present(grid, F, d, true)
                    s = _cell_value(ud, grid, F, d, false, T) + _cell_value(ud, grid, F, d, true, T)
                    w = T(lo + hi)
                    vd[F] = iszero(w) ? zero(T) : s / w
                end
            end
            return nothing
        end
    end
    return vf
end

@inline _cell_present(grid, F::CartesianIndex, d::Integer, upper::Bool) =
    (upper ? cell_above(grid, F, d) : cell_below(grid, F, d)) === nothing ? 0 : 1

"""
    to_centres!(uc, vf, grid, bc) -> uc

Face-normal velocity back to cell centres, averaging a cell's two faces — the adjoint of
[`to_faces!`](@ref). A cell against a closed face averages only the faces that carry flux.
"""
function to_centres!(uc, vf::NTuple{N,<:AbstractArray}, grid::FaceIndexedGrid{T,G,N},
                     bc, fm::FaceMetrics{N,T} = face_metrics(grid, bc); backend = ComputationalBackends.SerialBackend()) where {G,T,N}
    msk = FlowGeometries.Grids.mask(grid)
    cart = CartesianIndices(size(grid))
    for d in 1:N
        vd, ad = vf[d], fm.area[d]
        ud = _component(uc, d, Val(N))
        FlowGeometries.Execution.run_indices(length(ud), backend) do lin
            @inbounds begin
                I = cart[lin]
                if msk[I]
                    Fhi = face_above(grid, I, d)
                    lo = iszero(ad[I]) ? zero(T) : vd[I]
                    hi = iszero(ad[Fhi]) ? zero(T) : vd[Fhi]
                    w = T((iszero(ad[I]) ? 0 : 1) + (iszero(ad[Fhi]) ? 0 : 1))
                    ud[I] = iszero(w) ? zero(T) : (lo + hi) / w
                else
                    ud[I] = zero(T)
                end
            end
            return nothing
        end
    end
    return uc
end
