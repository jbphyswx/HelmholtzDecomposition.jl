"""
    Multigrid.jl — A geometric V-cycle preconditioner for `L`.

Conjugate gradients on `L` converges in a number of iterations that grows with the grid, because
`L`'s condition number does: relaxation kills the oscillatory error quickly and the smooth error
hardly at all. Multigrid removes that growth by representing the smooth error where it is *not*
smooth — on a coarser grid — so the iteration count becomes roughly resolution-independent.

This matters most exactly where nothing else applies. A masked domain admits no transform, so it
is the iterative solver or nothing, and it is also the case this package is for.

# Why it composes with the rest

Every level is an ordinary `FlowGeometries` grid with its own [`LaplacianCoefficients`](@ref), so
the coarse operator is the *same* `L = D G` construction re-discretized, not a second operator
chosen independently — the discipline that A5 was about. A coarse face inherits its area from
whether the cells it separates are active, so the mask and the boundary condition descend the
hierarchy by construction rather than by special-casing.

The cycle is symmetric — equal pre- and post-smoothing, and prolongation is the transpose of
restriction up to a positive constant — because a preconditioner for conjugate gradients must be,
or the method is no longer minimizing what it reports.
"""

"""
    MultigridLevel

One rung: the grid, its Laplacian coefficients, and the buffers a cycle needs there.
"""
struct MultigridLevel{G,LC,A}
    grid::G
    coefficients::LC
    x::A          # iterate / correction
    b::A          # right-hand side
    r::A          # residual
end

function MultigridLevel(grid, bc::AbstractBoundaryCondition, ::Type{T}) where {T}
    c = laplacian_coefficients(grid, bc)
    dims = size(grid)
    return MultigridLevel(grid, c, zeros(T, dims), zeros(T, dims), zeros(T, dims))
end

MultigridLevel(grid, c::LaplacianCoefficients, ::Type{T}) where {T} =
    MultigridLevel(grid, c, zeros(T, size(grid)), zeros(T, size(grid)), zeros(T, size(grid)))

"""
    galerkin_coefficients(fine, fgrid, cgrid) -> LaplacianCoefficients

The coarse operator as `R A P` rather than a fresh discretization of the coarse grid.

Re-discretizing is what breaks a masked domain. The coarse mask cannot reproduce the fine one —
mark a coarse cell active if *any* child is and the domain grows outward a cell per level, require
*all* and it erodes — so the coarse operator ends up solving a differently-shaped problem and its
correction is inconsistent with the fine one. That showed up as the rotation-potential solves
running to their iteration cap while the primal solve converged in 13.

For this transfer pair — restriction that averages a cell's children, prolongation that adds a
coarse value to each of them — the Galerkin product has a closed form and needs no matrix: the
conductance of a coarse face is the **sum of the fine conductances crossing it**, and a coarse
cell's measure is the sum of its children's. Nothing about the geometry is recomputed, so nothing
can disagree with the fine level.
"""
function galerkin_coefficients(
    fine::LaplacianCoefficients{N,T}, fgrid, cgrid,
) where {N,T}
    fdims = size(fgrid)
    cdims = size(cgrid)
    coef = ntuple(d -> zeros(T, face_dims(cgrid, d)), Val(N))
    meas = zeros(T, cdims)
    diag = zeros(T, cdims)

    @inbounds for d in 1:N
        cd, fd = coef[d], fine.coef[d]
        for Fc in CartesianIndices(face_dims(cgrid, d))
            # The coarse face sits at the fine face below this coarse cell, and spans its
            # children in every other direction.
            base = ntuple(e -> e == d ? (2Fc[e] - 1) : 0, Val(N))
            fdims[d] < 4 && (base = ntuple(e -> e == d ? Fc[e] : 0, Val(N)))
            acc = zero(T)
            for J in CartesianIndices(ntuple(e -> e == d ? (base[e]:base[e]) :
                                                  _children(Fc[e], fdims[e]), Val(N)))
                checkbounds(Bool, fd, J) || continue
                acc += fd[J]
            end
            cd[Fc] = acc
        end
    end
    @inbounds for Ic in CartesianIndices(cdims)
        m = zero(T)
        for J in CartesianIndices(ntuple(e -> _children(Ic[e], fdims[e]), Val(N)))
            m += fine.measure[J]
        end
        meas[Ic] = m
        iszero(m) && continue
        acc = zero(T)
        for d in 1:N
            acc += coef[d][face_below(Ic, d)] + coef[d][face_above(cgrid, Ic, d)]
        end
        diag[Ic] = -acc / m
    end
    built = LaplacianCoefficients{N,T,typeof(coef),typeof(diag)}(coef, diag, meas, false)
    return LaplacianCoefficients{N,T,typeof(coef),typeof(diag)}(
        coef, diag, meas, _detect_singular(cgrid, built),
    )
end

"""
    coarsen(grid, bc) -> grid or nothing

The next grid down: every other coordinate in each direction that still has enough cells to halve.

A coarse cell is active when **any** of the fine cells it covers is. The alternative — requiring
all of them — erodes the domain by a cell per level, so a narrow channel or a coastline would
vanish partway down the hierarchy and the correction there would be identically zero.
"""
function coarsen(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, bc) where {G,T,N}
    dims = size(grid)
    # Nothing left to coarsen once every direction is too small to halve usefully.
    all(n -> n < 4, dims) && return nothing
    axes_c = ntuple(d -> _coarse_axis(FlowGeometries.Grids.coordinates(grid, d), dims[d]), Val(N))
    cdims = ntuple(d -> length(axes_c[d]), Val(N))
    fine_mask = FlowGeometries.Grids.mask(grid)
    cmask = trues(cdims)
    @inbounds for I in CartesianIndices(cdims)
        any_active = false
        for J in CartesianIndices(ntuple(d -> _children(I[d], dims[d]), Val(N)))
            fine_mask[J] && (any_active = true; break)
        end
        cmask[I] = any_active
    end
    topo = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(N))
    per = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d) ?
                      FlowGeometries.Grids.period(grid, d) : nothing, Val(N))
    return FlowGeometries.Grids.StructuredGrid(
        FlowGeometries.Grids.grid_geometry(grid), axes_c...;
        mask = cmask, topology = topo, period = per,
    )
end

# Coarse cell centres sit midway between the pair of fine centres they cover, so the coarse grid is
# a genuine cell-centred grid rather than a subsample of the fine one.
#
# A uniform axis coarsens to a `UniformAxis`, not a `Vector`. FlowGeometries proves uniformity from
# the axis TYPE, so collecting it would make every coarse level report itself as stretched — losing
# the `O(1)` spacing lookup, and with it the fast paths, at every level of the hierarchy.
function _coarse_axis(x::AbstractRange, n::Int)
    n < 4 && return x
    nc = cld(n, 2)
    h = step(x)
    return FlowGeometries.Axes.UniformAxis(first(x) + h / 2, 2h, nc)
end

function _coarse_axis(x, n::Int)
    n < 4 && return collect(float(eltype(x)), x)
    nc = cld(n, 2)
    out = Vector{float(eltype(x))}(undef, nc)
    @inbounds for i in 1:nc
        out[i] = (x[2i - 1] + x[min(2i, n)]) / 2
    end
    return out
end

@inline _children(i::Int, n::Int) = n < 4 ? (i:i) : ((2i - 1):min(2i, n))

"""
    restrict!(coarse, fine, cgrid, fgrid, fc, cc)

**Measure-weighted** average of each coarse cell's fine children.

The weighting is not a refinement — it is what makes `R` the adjoint of prolongation, and a
non-adjoint pair makes the preconditioner non-symmetric, which invalidates conjugate gradients.
With prolongation by injection and the coarse measure defined as the sum of its children's,

    ⟨P c, f⟩_fine = Σ_I Σ_{J∈children(I)} V_J c_I f_J = ⟨c, R f⟩_coarse
    ⟹  (R f)_I = Σ_J V_J f_J / Σ_J V_J

An unweighted average satisfies that only when every `V_J` is equal, which is why it worked on a
Cartesian grid and failed outright on a sphere, where the measure varies as `cos φ`: multigrid ran
to the iteration cap at every resolution while plain Jacobi converged.

No mask term is needed — an inactive cell's measure is already zero, so it carries no weight.
"""
function restrict!(coarse, fine, cgrid, fgrid::FlowGeometries.Grids.StructuredGrid{G,T,N},
                   fc::LaplacianCoefficients{N,T}, cc::LaplacianCoefficients{N,T}) where {G,T,N}
    fdims = size(fgrid)
    fmeas, cmeas = fc.measure, cc.measure
    @inbounds for I in CartesianIndices(size(cgrid))
        w = cmeas[I]
        if iszero(w)
            coarse[I] = zero(T)
            continue
        end
        acc = zero(T)
        for J in CartesianIndices(ntuple(d -> _children(I[d], fdims[d]), Val(N)))
            acc += fmeas[J] * fine[J]
        end
        coarse[I] = acc / w
    end
    return coarse
end

"""
    prolong_add!(fine, coarse, cgrid, fgrid, fc)

Add each coarse value to the fine cells it covers — injection, the adjoint of the measure-weighted
[`restrict!`](@ref), which is what keeps the cycle symmetric.
"""
function prolong_add!(fine, coarse, cgrid, fgrid::FlowGeometries.Grids.StructuredGrid{G,T,N},
                      fc::LaplacianCoefficients{N,T}) where {G,T,N}
    fdims = size(fgrid)
    fmeas = fc.measure
    @inbounds for I in CartesianIndices(size(cgrid))
        v = coarse[I]
        iszero(v) && continue
        for J in CartesianIndices(ntuple(d -> _children(I[d], fdims[d]), Val(N)))
            iszero(fmeas[J]) && continue
            fine[J] += v
        end
    end
    return fine
end

"""
    smooth!(x, b, level, ω, n)

`n` sweeps of damped Jacobi on `−L`, the positive-definite orientation.

Damped Jacobi rather than Gauss–Seidel: it is a pure `Ax` plus an axpy, so it is the same
expression serial, threaded and on a device, and it is symmetric — which Gauss–Seidel is not
unless the sweep order is reversed between the pre- and post-smoothing, and a non-symmetric
preconditioner breaks conjugate gradients.
"""
function smooth!(x, b, level::MultigridLevel, ω::T, n::Int; backend = ComputationalBackends.SerialBackend()) where {T}
    grid, c = level.grid, level.coefficients
    r = level.r
    d = c.diag
    for _ in 1:n
        apply_laplacian!(r, x, grid, c; backend = backend)
        # `A = −L`, so the residual is `b − A x = b + L x` and `A`'s diagonal is `−d`.
        x .= ifelse.(iszero.(d), x, x .+ ω .* (b .+ r) ./ .-d)
    end
    return x
end

"""
    vcycle!(levels, ω, ν, backend)

One V-cycle over `levels`, coarsest last: smooth, restrict the residual, recurse on the tail,
prolong the correction, smooth again. Equal pre- and post-smoothing keeps the cycle symmetric.

The recursion is on the tuple's **tail**, not on an index. Recursing with `Val(l + 1)` gives the
compiler no proof the depth terminates, so it abandons inference at the recursive call and boxes
the arguments — 1 KiB per level per cycle, which at 81 conjugate-gradient iterations was most of a
megabyte per solve. A shorter tuple type each step terminates structurally.
"""
function vcycle!(levels::Tuple{LC}, ω::T, ν::Int, backend) where {LC,T}
    lev = levels[1]
    _project!(lev)
    # Coarsest: smooth hard rather than solve exactly; the grid is tiny by here.
    smooth!(lev.x, lev.b, lev, ω, 20ν; backend = backend)
    _project!(lev)
    return nothing
end

function vcycle!(levels::Tuple, ω::T, ν::Int, backend) where {T}
    lev, rest = levels[1], Base.tail(levels)
    nxt = rest[1]
    # A closed problem — every face of the active region shut, which is what a Neumann domain and
    # the dual grid's masked-off boundary ring both are — leaves the constants in `L`'s null space
    # at EVERY level. Without projecting them out the coarse correction drifts along the null
    # space and the preconditioner stops being one: the rotation-potential solves ran to their
    # iteration cap rather than converging.
    _project!(lev)
    smooth!(lev.x, lev.b, lev, ω, ν; backend = backend)
    apply_laplacian!(lev.r, lev.x, lev.grid, lev.coefficients; backend = backend)
    @. lev.r = lev.b + lev.r                    # b − Ax with A = −L

    restrict!(nxt.b, lev.r, nxt.grid, lev.grid, lev.coefficients, nxt.coefficients)
    # The restricted residual must be in the range of the coarse operator, or the coarse solve is
    # inconsistent and has no solution to converge to.
    nxt.coefficients.singular && project_out_constant!(nxt.b, nxt.grid, nxt.coefficients)
    fill!(nxt.x, zero(T))
    vcycle!(rest, ω, ν, backend)
    prolong_add!(lev.x, nxt.x, nxt.grid, lev.grid, lev.coefficients)

    smooth!(lev.x, lev.b, lev, ω, ν; backend = backend)
    _project!(lev)
    return nothing
end

@inline function _project!(lev::MultigridLevel)
    lev.coefficients.singular && project_out_constant!(lev.x, lev.grid, lev.coefficients)
    return lev
end

"""
    MultigridPreconditioner

A hierarchy built once for a `(grid, boundary)` pair, applied as `z ← M⁻¹ r` inside conjugate
gradients.
"""
struct MultigridPreconditioner{L<:Tuple,T<:AbstractFloat}
    # A tuple, not a `Vector`: the levels genuinely differ in type — a coarse grid's mask is a
    # `BitArray` where an unmasked fine grid's is `AllActive` — and a tuple carries that difference
    # in the type rather than erasing it to `Any`. Erasing it made every operation inside the
    # V-cycle a dynamic dispatch, which allocated a box per intermediate: 37 KiB per application on
    # a 24×19 grid, several times that per conjugate-gradient iteration.
    levels::L
    ω::T
    ν::Int
end

"""
    multigrid(grid, bc, T; ω = 0.8, ν = 2, maxlevels = 12) -> MultigridPreconditioner

Build the hierarchy. `ω = 0.8` is the usual damped-Jacobi factor for a 2-D five-point Laplacian —
undamped Jacobi does not reduce the highest-frequency error at all, which is the error a smoother
exists to remove.
"""
function multigrid(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, bc::AbstractBoundaryCondition,
                   ::Type{T} = T; ω::T = T(0.8), ν::Int = 2, maxlevels::Int = 12) where {G,T,N}
    return MultigridPreconditioner(_levels(MultigridLevel(grid, bc, T), grid, bc, maxlevels - 1),
                                   ω, ν)
end

# Grown by recursion rather than `push!`, so each level's own type survives into the tuple.
function _levels(lev, g, bc, remaining::Int)
    remaining <= 0 && return (lev,)
    gc = coarsen(g, bc)
    gc === nothing && return (lev,)
    # Galerkin, not a fresh discretization: the coarse mask cannot reproduce the fine one, so
    # re-discretizing solves a differently-shaped problem — see `galerkin_coefficients`.
    cc = galerkin_coefficients(lev.coefficients, g, gc)
    return (lev, _levels(MultigridLevel(gc, cc, eltype(g)), gc, bc, remaining - 1)...)
end

_precondition!(z, r, grid, c, mg::MultigridPreconditioner; backend = ComputationalBackends.SerialBackend()) =
    apply_preconditioner!(z, r, mg; backend = backend)

function apply_preconditioner!(z, r, mg::MultigridPreconditioner; backend = ComputationalBackends.SerialBackend())
    T = eltype(z)
    top = mg.levels[1]
    copyto!(top.b, r)
    fill!(top.x, zero(T))
    vcycle!(mg.levels, T(mg.ω), mg.ν, backend)
    copyto!(z, top.x)
    return z
end
