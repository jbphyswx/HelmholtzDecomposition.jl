# A geometric V-cycle preconditioner for `L`. Each level is a `FlowGeometries.Operators.coarsen` of
# the one above, its operator the Galerkin product `R L P` of `galerkin_coefficients`, so the mask
# and the boundary condition descend the hierarchy through the coefficients. Restriction is the
# measure-weighted mean and prolongation its adjoint, injection, and the smoothing is equal before
# and after, so the cycle is symmetric in the cell measure, as conjugate gradients requires.

"""
    LevelBuffers

The three vectors a cycle writes through at one level: the iterate, its right-hand side and the
residual. One set per task, since a cycle writes them.
"""
struct LevelBuffers{A}
    x::A
    b::A
    r::A
end

function LevelBuffers(c::FlowGeometries.Operators.LaplacianCoefficients{N,T};
                      backend = ComputationalBackends.SerialBackend()) where {N,T}
    dims = FlowGeometries.Grids.size_tuple(c.grid)
    return LevelBuffers(_zeros(backend, T, dims), _zeros(backend, T, dims), _zeros(backend, T, dims))
end

"""
    smooth!(x, b, c, r, ω, n; backend)

`n` sweeps of damped Jacobi on `−L`, the positive-definite orientation, `L` the operator the
coefficients `c` describe, using `r` as the residual scratch. Jacobi is one application of `L` and an
update per sweep, the same on every backend, and symmetric.
"""
function smooth!(x, b, c::FlowGeometries.Operators.LaplacianCoefficients, r, ω::T, n::Int;
                 backend = ComputationalBackends.SerialBackend()) where {T}
    d = c.diag
    lb = loop_backend(backend, length(x))
    for _ in 1:n
        FlowGeometries.Operators.apply_laplacian!(r, x, c; backend = lb)
        # `A = −L`, so the residual is `b − A x = b + L x` and `A`'s diagonal is `−d`.
        FlowGeometries.Execution.run_indices(length(x), lb) do lin
            @inbounds begin
                dI = d[lin]
                iszero(dI) || (x[lin] += ω * (b[lin] + r[lin]) / -dI)
            end
            return nothing
        end
    end
    return x
end

"""
    vcycle!(levels, bufs, ω, ν, backend)

One V-cycle over `levels`, coarsest last: smooth, restrict the residual, recurse on the tail, prolong
the correction, smooth again.

The recursion is on the tuple's tail, so each step's argument is a shorter tuple type and the
compiler sees the depth terminate.
"""
function vcycle!(levels::Tuple{LC}, bufs::Tuple{BF}, ω::T, ν::Int, backend) where {LC,BF,T}
    c, bf = levels[1], bufs[1]
    _project!(c, bf, backend)
    # The coarsest grid is a handful of cells, relaxed to convergence.
    smooth!(bf.x, bf.b, c, bf.r, ω, 20ν; backend = backend)
    _project!(c, bf, backend)
    return nothing
end

function vcycle!(levels::Tuple, bufs::Tuple, ω::T, ν::Int, backend) where {T}
    c, rest = levels[1], Base.tail(levels)
    bf, brest = bufs[1], Base.tail(bufs)
    cn, bnxt = rest[1], brest[1]
    lb = loop_backend(backend, length(bf.x))
    # A closed problem keeps the constants in `L`'s null space at every level, and the coarse
    # correction is taken orthogonal to them.
    _project!(c, bf, backend)
    smooth!(bf.x, bf.b, c, bf.r, ω, ν; backend = backend)
    FlowGeometries.Operators.apply_laplacian!(bf.r, bf.x, c; backend = lb)
    @. bf.r = bf.b + bf.r                    # b − Ax with A = −L

    FlowGeometries.Operators.restrict!(bnxt.b, bf.r, c, cn;
                                       backend = loop_backend(backend, length(bnxt.b)))
    # The coarse right-hand side lies in the range of the coarse operator.
    FlowGeometries.Operators.project_out_constant!(bnxt.b, cn;
                                                   backend = loop_backend(backend, length(bnxt.b)))
    fill!(bnxt.x, zero(T))
    vcycle!(rest, brest, ω, ν, backend)
    FlowGeometries.Operators.prolong_add!(bf.x, bnxt.x, c, cn; backend = lb)

    smooth!(bf.x, bf.b, c, bf.r, ω, ν; backend = backend)
    _project!(c, bf, backend)
    return nothing
end

@inline _project!(c, bf::LevelBuffers, backend) =
    FlowGeometries.Operators.project_out_constant!(bf.x, c; backend = loop_backend(backend, length(bf.x)))

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
    multigrid_buffers(mg; backend) -> Tuple{Vararg{LevelBuffers}}

One `LevelBuffers` per level, for one task. The hierarchy itself is shared.
"""
multigrid_buffers(mg::MultigridPreconditioner; backend = ComputationalBackends.SerialBackend()) =
    map(c -> LevelBuffers(c; backend = backend), mg.levels)

_precondition!(z, r, grid, c, mg::MultigridPreconditioner, bufs;
               backend = ComputationalBackends.SerialBackend()) =
    apply_preconditioner!(z, r, mg, bufs; backend = backend)

function apply_preconditioner!(z, r, mg::MultigridPreconditioner, bufs;
                               backend = ComputationalBackends.SerialBackend())
    T = eltype(z)
    top = bufs[1]
    copyto!(top.b, r)
    fill!(top.x, zero(T))
    vcycle!(mg.levels, bufs, T(mg.ω), mg.ν, backend)
    copyto!(z, top.x)
    return z
end
