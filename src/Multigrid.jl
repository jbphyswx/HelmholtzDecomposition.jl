# A geometric V-cycle preconditioner for `L`. Each level is a `FlowGeometries.Operators.coarsen` of
# the one above, its operator the Galerkin product `R L P` of `galerkin_coefficients`, so the mask
# and the boundary condition descend the hierarchy through the coefficients. Restriction is the
# measure-weighted mean and prolongation its adjoint, injection, and the smoothing is equal before
# and after, so the cycle is symmetric in the cell measure, as conjugate gradients requires.
#
# On a closed problem each coarse right-hand side is projected onto the range of its operator. The
# iterates are not projected: a constant on a closed component is in the null space of every finer
# level's `L`, and the conjugate-gradient iteration removes it from the cycle's output.

"""
    LevelBuffers

The three vectors a cycle writes through at a level below the finest: the iterate, its right-hand
side and a second iterate the smoothing alternates with. One set per task, since a cycle writes them.
"""
struct LevelBuffers{A}
    x::A
    b::A
    r::A
end

function LevelBuffers(c::FlowGeometries.Operators.LaplacianCoefficients{N,T};
                      execution = ComputationalBackends.SerialBackend()) where {N,T}
    dims = FlowGeometries.Grids.size_tuple(c.grid)
    return LevelBuffers(_zeros(execution, T, dims), _zeros(execution, T, dims),
                        _zeros(execution, T, dims))
end

"""
    MultigridBuffers

What one task's cycles write: the finest level's second iterate (its iterate and right-hand side are
the preconditioner's output and input) and a [`LevelBuffers`](@ref) per coarser level.
"""
struct MultigridBuffers{A,L<:Tuple}
    scratch::A
    coarse::L
end

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

"""
    vcycle!(levels, x, b, r, coarse, ω, ν, backend)

One V-cycle over `levels`, coarsest last, for `−L x = b` on `levels[1]` from `x = 0`: smooth, restrict
the residual, recurse on the tail, prolong the correction, smooth again. `x`'s entry values are not
read, `r` is the level's second iterate, and `coarse` holds the [`LevelBuffers`](@ref) of the levels
below.

The recursion is on the tuple's tail, so each step's argument is a shorter tuple type and the
compiler sees the depth terminate.
"""
function vcycle!(levels::Tuple{LC}, x, b, r, ::Tuple{}, ω::T, ν::Int, backend) where {LC,T}
    # The coarsest grid is a handful of cells, relaxed to convergence.
    smooth_from_zero!(x, b, levels[1], r, ω, 20ν; execution = backend)
    return nothing
end

function vcycle!(levels::Tuple, x, b, r, coarse::Tuple, ω::T, ν::Int, backend) where {T}
    c, rest = levels[1], Base.tail(levels)
    cn, nxt = rest[1], coarse[1]
    lb = loop_backend(backend, length(x))
    smooth_from_zero!(x, b, c, r, ω, ν; execution = backend)
    # `R (b − A x)`, `A = −L`; on a closed problem it is projected onto the coarse operator's range.
    FlowGeometries.Operators.restrict_residual!(nxt.b, b, x, c, cn; execution = lb)
    FlowGeometries.Operators.project_out_constant!(nxt.b, cn;
                                                   execution = loop_backend(backend, length(nxt.b)))
    vcycle!(rest, nxt.x, nxt.b, nxt.r, Base.tail(coarse), ω, ν, backend)
    FlowGeometries.Operators.prolong_add!(x, nxt.x, c, cn; execution = lb)
    smooth!(x, b, c, r, ω, ν; execution = backend)
    return nothing
end

"""
    MultigridPreconditioner

A hierarchy of `FlowGeometries.Operators.LaplacianCoefficients`, finest first, applied as
`z ← M⁻¹ r` inside conjugate gradients. A tuple, so each level keeps its own grid and mask types.
"""
struct MultigridPreconditioner{L<:Tuple,T<:AbstractFloat}
    levels::L
    ω::T
    ν::Int
end

"""
    multigrid(c; ω = 0.8, ν = 2, maxlevels = 12) -> MultigridPreconditioner

The hierarchy below the operator `c` describes, each coarser level the Galerkin product of the one
above on `FlowGeometries.Operators.coarsen` of its grid. `ω = 0.8` is the damped-Jacobi weight that
reduces the highest-frequency error of a five-point Laplacian.
"""
function multigrid(c::FlowGeometries.Operators.LaplacianCoefficients{N,T}; ω::Real = T(0.8),
                   ν::Int = 2, maxlevels::Int = 12) where {N,T}
    return MultigridPreconditioner(_levels(c, maxlevels - 1), T(ω), ν)
end

# Grown by recursion, so each level's own type survives into the tuple.
function _levels(c, remaining::Int)
    remaining <= 0 && return (c,)
    gc = FlowGeometries.Operators.coarsen(c.grid)
    gc === nothing && return (c,)
    return (c, _levels(FlowGeometries.Operators.galerkin_coefficients(c, gc), remaining - 1)...)
end

"""
    multigrid_buffers(mg; execution) -> MultigridBuffers

The vectors one task's cycles write. The hierarchy itself is shared.
"""
function multigrid_buffers(mg::MultigridPreconditioner; execution = ComputationalBackends.SerialBackend())
    top = mg.levels[1]
    scratch = _zeros(execution, eltype(top.diag), FlowGeometries.Grids.size_tuple(top.grid))
    return MultigridBuffers(scratch,
                            map(c -> LevelBuffers(c; execution = execution), Base.tail(mg.levels)))
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
    vcycle!(mg.levels, z, r, bufs.scratch, bufs.coarse, eltype(z)(mg.ω), mg.ν, execution)
    return z
end
