# A geometric V-cycle preconditioner for `L`. Each level is a `FlowGeometries.Operators.coarsen` of
# the one above, its operator `L` rediscretized there from the fine coupling areas
# (`coarse_coefficients`), so the mask and the boundary condition descend the hierarchy through the
# coefficients. Restriction is the measure-weighted mean and prolongation its adjoint, injection, and the
# smoothing is equal before and after, so the cycle is symmetric in the cell measure, as conjugate
# gradients requires. The Galerkin product `R L P` of these transfers is twice the rediscretized operator
# along each halved direction, and a cycle on it converges more slowly the more levels it has.
#
# On a closed problem each coarse right-hand side is projected onto the range of its operator. The
# iterates are not projected: a constant on a closed component is in the null space of every finer
# level's `L`, and the conjugate-gradient iteration removes it from the cycle's output.

"""
    LevelBuffers

The vectors a cycle writes through at a level below the finest: the iterate, its right-hand side, a
second iterate the point smoothing alternates with, and the line solves' scratch where the smoothing is
by lines (`nothing` otherwise). One set per task, since a cycle writes them.
"""
struct LevelBuffers{A,S}
    x::A
    b::A
    r::A
    line::S
end

function LevelBuffers(c::FlowGeometries.Operators.LaplacianCoefficients{N,T}, smoother;
                      execution = ComputationalBackends.SerialBackend()) where {N,T}
    dims = FlowGeometries.Grids.size_tuple(c.grid)
    z() = _zeros(execution, T, dims)
    return LevelBuffers(z(), z(), z(), _line_scratch(smoother, z))
end

"""
    MultigridBuffers

What one task's cycles write: the finest level's second iterate and line scratch (its iterate and
right-hand side are the preconditioner's output and input) and a [`LevelBuffers`](@ref) per coarser
level.
"""
struct MultigridBuffers{A,S,L<:Tuple}
    scratch::A
    line::S
    coarse::L
end

"""
    PointJacobi(ω)

`ν` sweeps of damped Jacobi with weight `ω` on each side of a coarse correction.
"""
struct PointJacobi{T<:AbstractFloat}
    ω::T
end

"""
    ZebraLines()

`ν` passes of line Gauss–Seidel along direction 1 on each side of a coarse correction: every line of cells
along direction 1 is solved exactly, a tridiagonal system (periodic where the direction wraps) against the
current values of the lines beside it, first the lines of one parity of `i₂ + … + i_N` and then the
other, in the reverse order after the correction, so the cycle stays symmetric. Where the coupling along
direction 1 dominates, as along the latitude rows of a sphere near its poles, a point smoother leaves the
error that is smooth along a row and rough across it, and a coarser level cannot represent that error.
"""
struct ZebraLines end

"""
    AlternatingLines()

`ν` passes of [`ZebraLines`](@ref)'s line Gauss–Seidel along each direction in turn on each side of a
coarse correction, directions `1, 2, …, N` before the correction and `N, …, 1` after it, so the cycle stays
symmetric. Where the direction of strong coupling changes across the grid, as on a mesh stretched along
both directions, every cell is relaxed along its strong direction.
"""
struct AlternatingLines end

const _LineSmoother = Union{ZebraLines, AlternatingLines}

# The line solves' scratch: the Thomas factors and the second solution of the periodic correction.
_line_scratch(::PointJacobi, z) = nothing
_line_scratch(::_LineSmoother, z) = (cp = z(), dp = z(), w = z())

"""
    default_smoother(c::LaplacianCoefficients) -> PointJacobi, ZebraLines or AlternatingLines

The smoother of the operator `c` describes. [`ZebraLines`](@ref) on a sphere or spheroid of two or more
directions, whose cells narrow along longitude towards the poles. Elsewhere [`AlternatingLines`](@ref)
where some cell couples to its neighbours along one direction more than twice as strongly as along
another, and [`PointJacobi`](@ref)`(0.8)` otherwise: a damped-Jacobi sweep reduces an error that is smooth
along the strong direction of a cell with coupling ratio `ε ≤ 1` by the factor `1 − ωε/(1 + ε)`, 0.6 at
`ε = 1`, 0.73 at `ε = 1/2`, and towards 1 as `ε` falls.
"""
default_smoother(c::FlowGeometries.Operators.LaplacianCoefficients{N,T}) where {N,T} =
    _default_smoother(FlowGeometries.Grids.grid_geometry(c.grid), Val(N), T, c)
_default_smoother(::Union{FlowGeometries.Geometry.AbstractSphericalGeometry,
                          FlowGeometries.Geometry.AbstractEllipsoidalGeometry}, ::Val{N}, ::Type{T}, c) where {N,T} =
    N ≥ 2 ? ZebraLines() : PointJacobi(T(0.8))
_default_smoother(::Any, ::Val{N}, ::Type{T}, c) where {N,T} =
    N ≥ 2 && _coupling_ratio(c) > 2 ? AlternatingLines() : PointJacobi(T(0.8))

# The largest ratio, over the active cells, of a cell's strongest face coefficient along one direction to
# its strongest along another; a direction whose faces at a cell are both closed does not enter.
function _coupling_ratio(c::FlowGeometries.Operators.LaplacianCoefficients{N,T}) where {N,T}
    grid, coef = c.grid, c.coef
    msk = FlowGeometries.Grids.mask(grid)
    worst = one(T)
    @inbounds for I in CartesianIndices(FlowGeometries.Grids.size_tuple(grid))
        msk[I] || continue
        s = ntuple(d -> max(abs(T(coef[d][I])), abs(T(coef[d][FlowGeometries.Grids.face_above(grid, I, d)]))), Val(N))
        hi = maximum(s)
        lo = minimum(x -> iszero(x) ? hi : x, s)
        iszero(lo) || (worst = max(worst, hi / lo))
    end
    return worst
end

_default_sweeps(::PointJacobi) = 2
_default_sweeps(::_LineSmoother) = 1

"""
    smooth!(x, b, c, r, ω, n; execution)

`n` sweeps of damped Jacobi on `−L`, the positive-definite orientation, `L` the operator the
coefficients `c` describe, `r` the second iterate. A sweep is one pass, the application of `L` fused
with the update, from one iterate into the other; an odd `n` copies the last one back into `x`. The same
on every backend, and symmetric.
"""
function smooth!(x, b, c::FlowGeometries.Operators.LaplacianCoefficients, r, ω::T, n::Int;
                 execution = ComputationalBackends.SerialBackend()) where {T}
    lb = loop_backend(execution, length(x))
    src, dst = x, r
    for _ in 1:n
        _jacobi_sweep!(dst, src, b, c, ω, lb)
        src, dst = dst, src
    end
    src === x || copyto!(x, src)
    return x
end

"""
    smooth_from_zero!(x, b, c, r, ω, n; execution)

[`smooth!`](@ref) from `x = 0`; `x`'s entry values are not read. The first sweep is `ω b/(−d)`, which
reads no neighbour, and the sweeps alternate between `r` and `x` so the last one lands in `x`.
"""
function smooth_from_zero!(x, b, c::FlowGeometries.Operators.LaplacianCoefficients, r, ω::T, n::Int;
                           execution = ComputationalBackends.SerialBackend()) where {T}
    n ≥ 1 || return fill!(x, zero(eltype(x)))
    lb = loop_backend(execution, length(x))
    src, dst = isodd(n) ? (x, r) : (r, x)
    _jacobi_first!(src, b, c, ω, lb)
    for _ in 2:n
        _jacobi_sweep!(dst, src, b, c, ω, lb)
        src, dst = dst, src
    end
    return x
end

function _jacobi_first!(out, b, c::FlowGeometries.Operators.LaplacianCoefficients, ω::T, backend) where {T}
    d = c.diag
    FlowGeometries.Execution.run_indices(length(out), backend) do lin
        @inbounds begin
            dI = d[lin]
            out[lin] = iszero(dI) ? zero(eltype(out)) : ω * b[lin] / -dI
        end
        return nothing
    end
    return out
end

# `out = x + ω (b + L x) / (−d)`, the damped Jacobi step on `A = −L`, whose residual is `b + L x` and whose
# diagonal is `−d`; a cell with no face keeps its value.
function _jacobi_sweep!(out, x, b, c::FlowGeometries.Operators.LaplacianCoefficients, ω::T, backend) where {T}
    d = c.diag
    FlowGeometries.Operators.apply_laplacian!(out, x, c; execution = backend) do I, L
        @inbounds begin
            dI = d[I]
            return iszero(dI) ? x[I] : x[I] + ω * (b[I] + L) / -dI
        end
    end
    return out
end

# The smoothing before a coarse correction, from `x = 0`, and after it.
_presmooth!(x, b, c, r, line, sm::PointJacobi, ν::Int, backend) =
    smooth_from_zero!(x, b, c, r, sm.ω, ν; execution = backend)
_postsmooth!(x, b, c, r, line, sm::PointJacobi, ν::Int, backend) =
    smooth!(x, b, c, r, sm.ω, ν; execution = backend)

function _presmooth!(x, b, c, r, line, ::ZebraLines, ν::Int, backend)
    fill!(x, zero(eltype(x)))
    for _ in 1:ν
        _line_sweep!(x, b, c, line, 0, backend, Val(1))
        _line_sweep!(x, b, c, line, 1, backend, Val(1))
    end
    return x
end

function _postsmooth!(x, b, c, r, line, ::ZebraLines, ν::Int, backend)
    for _ in 1:ν
        _line_sweep!(x, b, c, line, 1, backend, Val(1))
        _line_sweep!(x, b, c, line, 0, backend, Val(1))
    end
    return x
end

function _presmooth!(x, b, c::FlowGeometries.Operators.LaplacianCoefficients{N}, r, line, ::AlternatingLines,
                     ν::Int, backend) where {N}
    fill!(x, zero(eltype(x)))
    for _ in 1:ν
        _alternate!(x, b, c, line, backend, Val(1), Val(N), 0)
    end
    return x
end

function _postsmooth!(x, b, c::FlowGeometries.Operators.LaplacianCoefficients{N}, r, line, ::AlternatingLines,
                      ν::Int, backend) where {N}
    for _ in 1:ν
        _alternate!(x, b, c, line, backend, Val(N), Val(1), 1)
    end
    return x
end

# Zebra passes along directions `d` to `stop`, a step of one towards `stop`, each pass taking colour
# `first` and then the other.
@inline function _alternate!(x, b, c, line, backend, ::Val{d}, ::Val{stop}, first::Int) where {d, stop}
    _line_sweep!(x, b, c, line, first, backend, Val(d))
    _line_sweep!(x, b, c, line, 1 - first, backend, Val(d))
    d == stop && return x
    return _alternate!(x, b, c, line, backend, Val(d < stop ? d + 1 : d - 1), Val(stop), first)
end

# Every line along direction `d` whose sum of the other indices has the parity of `colour`, solved for
# `−L x = b` against the current `x` elsewhere. Lines of one colour do not couple, so they run in any order
# or at once.
function _line_sweep!(x, b, c::FlowGeometries.Operators.LaplacianCoefficients{N,T}, line, colour::Int,
                      backend, ::Val{d}) where {N,T,d}
    dims = FlowGeometries.Grids.size_tuple(c.grid)
    rest = CartesianIndices(_drop(dims, Val(d)))
    lb = loop_backend(backend, length(x))
    grid, meas, diag, coef = c.grid, c.measure, c.diag, c.coef
    periodic = FlowGeometries.Grids.isperiodic(grid, d)
    cp, dp, w = line.cp, line.dp, line.w
    FlowGeometries.Execution.run_indices(length(rest), lb) do ℓ
        @inbounds begin
            J = Tuple(rest[ℓ])
            iseven(sum(J; init = 0) + colour) &&
                _solve_line!(x, b, grid, meas, diag, coef, cp, dp, w, J, dims[d], periodic, T, Val(d))
        end
        return nothing
    end
    return x
end

# `t` without its entry `d`, and the index with `i` put back at `d`.
@inline _drop(t::NTuple{N}, ::Val{d}) where {N,d} = ntuple(k -> t[k < d ? k : k + 1], Val(N - 1))
@inline _line_index(i::Int, J::NTuple{M,Int}, ::Val{d}) where {M,d} =
    CartesianIndex(ntuple(k -> k < d ? J[k] : k == d ? i : J[k - 1], Val(M + 1)))

# One line's row of `V·(−L)`, `K x = V b + (coupling to the other directions' neighbours)`: diagonal the
# sum of the cell's face coefficients, `−c` to each neighbour along direction `d`. An inactive cell, or one
# with no face, keeps its value. The Thomas factors go to `cp`, `dp` and, for a periodic line, the second
# solution of the Sherman–Morrison correction to `w` (Press et al., *Numerical Recipes*, §2.7).
@inline function _solve_line!(x, b, grid, meas, diag, coef, cp, dp, w, J::NTuple{M,Int}, n::Int,
                              periodic::Bool, ::Type{T}, dir::Val) where {M,T}
    I1 = _line_index(1, J, dir)
    In = _line_index(n, J, dir)
    if n == 1
        k, _, _, d = _line_row(x, b, grid, meas, diag, coef, I1, true, T, dir)
        x[I1] = iszero(k) ? x[I1] : d / k
        return nothing
    end
    # Periodic corners: row 1's coupling to cell `n` and row `n`'s to cell 1.
    β = periodic ? _line_row(x, b, grid, meas, diag, coef, I1, false, T, dir)[2] : zero(T)
    α = periodic ? _line_row(x, b, grid, meas, diag, coef, In, false, T, dir)[3] : zero(T)
    k1 = _line_row(x, b, grid, meas, diag, coef, I1, false, T, dir)[1]
    γ = -k1
    cprev = zero(T)
    dprev = zero(T)
    wprev = zero(T)
    @inbounds for i in 1:n
        I = _line_index(i, J, dir)
        k, a, cc, d = _line_row(x, b, grid, meas, diag, coef, I, false, T, dir)
        if periodic
            i == 1 && (k -= γ; a = zero(T))
            i == n && (k -= α * β / γ; cc = zero(T))
        else
            i == 1 && (a = zero(T))
            i == n && (cc = zero(T))
        end
        m = k - a * cprev
        cprev = cc / m
        dprev = (d - a * dprev) / m
        cp[I] = cprev
        dp[I] = dprev
        if periodic
            u = i == 1 ? γ : i == n ? α : zero(T)
            wprev = (u - a * wprev) / m
            w[I] = wprev
        end
    end
    @inbounds for i in (n - 1):-1:1
        I = _line_index(i, J, dir)
        Iu = _line_index(i + 1, J, dir)
        dp[I] -= cp[I] * dp[Iu]
        periodic && (w[I] -= cp[I] * w[Iu])
    end
    if periodic
        f = (dp[I1] + β * dp[In] / γ) / (1 + w[I1] + β * w[In] / γ)
        @inbounds for i in 1:n
            I = _line_index(i, J, dir)
            x[I] = dp[I] - f * w[I]
        end
    else
        @inbounds for i in 1:n
            I = _line_index(i, J, dir)
            x[I] = dp[I]
        end
    end
    return nothing
end

# `(K_II, K to the cell below along d, K to the cell above along d, right-hand side)` for cell `I`; with
# `self`, a single-cell line, the couplings along `d` fold onto the cell itself. An inactive cell, or one
# with no face, is the identity row on its current value.
@inline function _line_row(x, b, grid, meas, diag, coef::NTuple{N}, I::CartesianIndex{N}, self::Bool,
                           ::Type{T}, ::Val{d}) where {N,T,d}
    @inbounds begin
        V = T(meas[I])
        k = -T(diag[I]) * V
        (iszero(V) || iszero(k)) && return (one(T), zero(T), zero(T), T(x[I]))
        cd = coef[d]
        a = -T(cd[I])
        cc = -T(cd[FlowGeometries.Grids.face_above(grid, I, d)])
        self && (k += a + cc; a = zero(T); cc = zero(T))
        s = V * T(b[I]) + _cross_terms(x, grid, coef, I, Val(N), T, Val(d))
        return (k, a, cc, s)
    end
end

# `Σ c (x_neighbour)` over the faces of `I` normal to every direction but the line's `l`, a neighbour past a
# bounded end reading zero as `L` does.
@inline _cross_terms(x, grid, coef, I::CartesianIndex, ::Val{0}, ::Type{T}, ::Val) where {T} = zero(T)
@inline function _cross_terms(x, grid, coef, I::CartesianIndex, ::Val{e}, ::Type{T}, line::Val{l}) where {e,T,l}
    e == l && return _cross_terms(x, grid, coef, I, Val(e - 1), T, line)
    @inbounds begin
        ce = coef[e]
        Fhi = FlowGeometries.Grids.face_above(grid, I, e)
        lo = FlowGeometries.Grids.cell_below(grid, I, e)
        hi = FlowGeometries.Grids.cell_above(grid, Fhi, e)
        s = T(ce[I]) * (lo === nothing ? zero(T) : T(x[lo])) + T(ce[Fhi]) * (hi === nothing ? zero(T) : T(x[hi]))
    end
    return s + _cross_terms(x, grid, coef, I, Val(e - 1), T, line)
end

"""
    vcycle!(levels, x, b, r, line, coarse, smoother, ν, backend)

One V-cycle over `levels`, coarsest last, for `−L x = b` on `levels[1]` from `x = 0`: smooth, restrict
the residual, recurse on the tail, prolong the correction, smooth again. `x`'s entry values are not
read, `r` and `line` are the level's second iterate and line scratch, and `coarse` holds the
[`LevelBuffers`](@ref) of the levels below.

The recursion is on the tuple's tail, so each step's argument is a shorter tuple type and the
compiler sees the depth terminate.
"""
function vcycle!(levels::Tuple{LC}, x, b, r, line, ::Tuple{}, sm, ν::Int, backend) where {LC}
    # The coarsest grid is a handful of cells, relaxed to convergence.
    _presmooth!(x, b, levels[1], r, line, sm, 20ν, backend)
    return nothing
end

function vcycle!(levels::Tuple, x, b, r, line, coarse::Tuple, sm, ν::Int, backend)
    c, rest = levels[1], Base.tail(levels)
    cn, nxt = rest[1], coarse[1]
    lb = loop_backend(backend, length(x))
    _presmooth!(x, b, c, r, line, sm, ν, backend)
    # `R (b − A x)`, `A = −L`; on a closed problem it is projected onto the coarse operator's range.
    FlowGeometries.Operators.restrict_residual!(nxt.b, b, x, c, cn; execution = lb)
    FlowGeometries.Operators.project_out_constant!(nxt.b, cn;
                                                   execution = loop_backend(backend, length(nxt.b)))
    vcycle!(rest, nxt.x, nxt.b, nxt.r, nxt.line, Base.tail(coarse), sm, ν, backend)
    FlowGeometries.Operators.prolong_add!(x, nxt.x, c, cn; execution = lb)
    _postsmooth!(x, b, c, r, line, sm, ν, backend)
    return nothing
end

"""
    MultigridPreconditioner

A hierarchy of `FlowGeometries.Operators.LaplacianCoefficients`, finest first, applied as
`z ← M⁻¹ r` inside conjugate gradients, with its smoother ([`PointJacobi`](@ref) or
[`ZebraLines`](@ref)) and sweeps per side. A tuple, so each level keeps its own grid and mask types.
"""
struct MultigridPreconditioner{L<:Tuple,S}
    levels::L
    smoother::S
    ν::Int
end

"""
    multigrid(c, boundary; smoother = default_smoother(c), ν, maxlevels = 12) -> MultigridPreconditioner

The hierarchy below the operator `c` describes under `boundary`, each coarser level the operator
rediscretized on `FlowGeometries.Operators.coarsen` of the grid above
(`FlowGeometries.Operators.coarse_coefficients`), smoothed by `smoother` with `ν` sweeps a side: two of
[`PointJacobi`](@ref)`(0.8)`, the damped-Jacobi weight that reduces the highest-frequency error of a
five-point Laplacian, or one pass of [`ZebraLines`](@ref) or [`AlternatingLines`](@ref).
"""
function multigrid(c::FlowGeometries.Operators.LaplacianCoefficients{N,T},
                   boundary::FlowGeometries.Operators.AbstractBoundaryCondition;
                   smoother = default_smoother(c), ν::Int = _default_sweeps(smoother),
                   maxlevels::Int = 12) where {N,T}
    return MultigridPreconditioner(_levels(c, boundary, maxlevels - 1), smoother, ν)
end

# Grown by recursion, so each level's own type survives into the tuple.
function _levels(c, boundary, remaining::Int)
    remaining <= 0 && return (c,)
    gc = FlowGeometries.Operators.coarsen(c.grid; pinned = _pinned(boundary))
    gc === nothing && return (c,)
    return (c, _levels(FlowGeometries.Operators.coarse_coefficients(c, gc, boundary; ghost = _ghost(boundary)),
                       boundary, remaining - 1)...)
end

# Where a condition holds its zeros off the end faces.
_ghost(bc::CornerDirichlet) = bc.ghost
_ghost(::Any) = nothing

# A two-dimensional dual grid's inactive corners hold `R = 0` through the conductances that reach them
# (`dual_coefficients`), so its coarse levels keep a corner active only where every corner it covers is.
_pinned(::CornerDirichlet) = true
_pinned(::Any) = false

"""
    multigrid_buffers(mg; execution) -> MultigridBuffers

The vectors one task's cycles write. The hierarchy itself is shared.
"""
function multigrid_buffers(mg::MultigridPreconditioner; execution = ComputationalBackends.SerialBackend())
    top = mg.levels[1]
    z() = _zeros(execution, eltype(top.diag), FlowGeometries.Grids.size_tuple(top.grid))
    return MultigridBuffers(z(), _line_scratch(mg.smoother, z),
                            map(c -> LevelBuffers(c, mg.smoother; execution = execution), Base.tail(mg.levels)))
end

_precondition!(z, r, grid, c, mg::MultigridPreconditioner, bufs;
               execution = ComputationalBackends.SerialBackend()) =
    apply_preconditioner!(z, r, mg, bufs; execution = execution)

"""
    apply_preconditioner!(z, r, mg, bufs; execution) -> z

`z = M⁻¹ r`, one V-cycle from `z = 0` with `r` its finest right-hand side; `z` and `r` are different
arrays and `r` is not written. On a closed problem `z` carries a constant in `L`'s null space, which the
caller removes.
"""
function apply_preconditioner!(z, r, mg::MultigridPreconditioner, bufs::MultigridBuffers;
                               execution = ComputationalBackends.SerialBackend())
    vcycle!(mg.levels, z, r, bufs.scratch, bufs.line, bufs.coarse, mg.smoother, mg.ν, execution)
    return z
end
