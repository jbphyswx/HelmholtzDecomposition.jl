"""
    HelmholtzDecompositionFINUFFTExt — Cartesian spectral decomposition at scattered points.

Inverse NUFFT to recover the velocity's Fourier coefficients → exact Leray projection in mode
space → type-2 NUFFT back to the points.

The transforms go through FINUFFT's **guru** interface, so the plans and the node set are built
once and only `exec` runs per call. The simple interface builds a plan and re-sends the points on
every invocation, and the inverse transform is a conjugate-gradient iteration that applies two
transforms per step — so at the default 200 iterations, per component, that was up to 800 plan
constructions and node uploads for one field, none of which depend on the data.
"""
module HelmholtzDecompositionFINUFFTExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra

"""
    CartesianNUFFTSolver(; nk = (64, 64), tol = 1e-8, rtol = 1e-10, maxiter = 1000)

Spectral Poisson/Helmholtz solver for scattered Cartesian samples in any dimension. `nk` is the
mode count per axis and its length fixes the dimension; `rtol`/`maxiter` govern the conjugate
gradients inside the exact inverse transform.

The iterations a fit needs grow with the mode-to-point ratio, and any ratio below one is accepted.
`maxiter` is sized so that conditioning is what an unconvergeable fit runs into: a budget set for a
well-sampled fit refuses a legitimate one near the top of that range, and the refusal then names
`nk` when the cap was the only thing missing.

`condition_limit` is the `κ(A)` above which the fit reports that the nodes determine the
coefficients too poorly for `rtol`. It defaults to `√(rtol/eps(T))`, the level where a
normal-equations solve stops reaching that tolerance; pass `Inf` for silence, or a number to
override. The estimate itself costs two scalars per iteration — see `HD.condition_estimate`.

`atol` is the norm of the measurement noise in `vals`, and setting it stops the fit once
`‖vals − A F‖` reaches it — the discrepancy principle. Iterations past that point fit the noise:
on a well-conditioned point set they achieve nothing, and on an ill-conditioned one the coefficients
travel a long way from their best value while the residual keeps falling, so `rtol` alone never
halts them. A caller who knows the noise in their data should set this.
"""
struct CartesianNUFFTSolver{D,T<:AbstractFloat} <: HD.AbstractPoissonSolver
    nk::NTuple{D,Int}
    tol::T
    rtol::T
    atol::T
    maxiter::Int
    condition_limit::T
end

function CartesianNUFFTSolver(; nk::NTuple{D,Int} = (64, 64), tol::AbstractFloat = 1e-8,
                              rtol::AbstractFloat = 1e-10, atol::Real = 0,
                              maxiter::Int = 1000, condition_limit::Real = -1) where {D}
    T = promote_type(typeof(float(tol)), typeof(float(rtol)))
    lim = condition_limit < 0 ? HD.condition_limit(rtol, T) : float(condition_limit)
    return CartesianNUFFTSolver{D,T}(nk, promote(tol, rtol)..., T(atol), maxiter, T(lim))
end

# A NUFFT expands in the same periodic basis an FFT does, over every sample it is handed.
HD.supports_boundary(::CartesianNUFFTSolver, ::HD.AbstractBoundaryCondition) = true
HD.requires_full_domain(::CartesianNUFFTSolver) = true

"""
    NUFFTPair{T,P}

The two FINUFFT plans one node set needs, and the buffers they exec through.

An owned, parameterised handle rather than the bare foreign plan: an extension that names a
dependency-defined type in a signature fails to precompile, and the plans must also be freed
deterministically. `Base.show` is defined for a second reason — the default `show` of a struct
holding a FINUFFT plan can segfault while printing the plan's internals.
"""
struct NUFFTPair{D,T,P}
    analysis::P            # type 1: points → modes (the exact adjoint of synthesis)
    synthesis::P           # type 2: modes → points
    nk::NTuple{D,Int}
    npoints::Int
end

Base.show(io::IO, p::NUFFTPair{D,T}) where {D,T} =
    print(io, "NUFFTPair{", D, ",", T, "}(", p.npoints, " points → ",
          join(p.nk, "×"), " modes)")

function nufft_pair(::Type{T}, coords::NTuple{D,AbstractVector{T}},
                    nk::NTuple{D,Int}, tol::Real, ntrans::Int) where {D,T}
    ms = FINUFFT.BIGINT[nk...]
    # `iflag` is opposite between the two so that analysis is the exact adjoint of synthesis,
    # which is what makes `AᴴA` symmetric positive definite and conjugate gradients valid.
    #
    # `ntrans` transforms per `exec`. The velocity's components share a node set, so they go
    # through together. This batches full-size complex transforms; it does not shrink them.
    a = FINUFFT.finufft_makeplan(1, ms, -1, ntrans, T(tol); dtype = T)
    s = FINUFFT.finufft_makeplan(2, ms, +1, ntrans, T(tol); dtype = T)
    c = map(collect, coords)                 # guru setpts requires contiguous storage
    FINUFFT.finufft_setpts!(a, c...)
    FINUFFT.finufft_setpts!(s, c...)
    return NUFFTPair{D,T,typeof(a)}(a, s, nk, length(first(c)))
end

function close_pair!(p::NUFFTPair)
    FINUFFT.finufft_destroy!(p.analysis)
    FINUFFT.finufft_destroy!(p.synthesis)
    return p
end

# `ntrans` transforms per `exec`, so the mode array is `(nk..., ntrans)` and the value array
# `(npoints, ntrans)`. Both must be contiguous, so they are whole arrays and never views.
@inline function analyze!(F::AbstractArray{Complex{T}}, f::AbstractMatrix{Complex{T}},
                          p::NUFFTPair{D,T}) where {D,T}
    FINUFFT.finufft_exec!(p.analysis, f, F)
    return F
end

@inline function synthesize!(f::AbstractMatrix{Complex{T}}, F::AbstractArray{Complex{T}},
                             p::NUFFTPair{D,T}) where {D,T}
    FINUFFT.finufft_exec!(p.synthesis, F, f)
    return f
end

# Column-wise inner product: the columns are independent systems sharing a plan, so each keeps its
# own conjugate-gradient scalars. The mode axes are all but the last.
@inline function _coldot(A::AbstractArray{Complex{T}}, B::AbstractArray{Complex{T}},
                         out::AbstractVector{T}) where {T}
    modes = CartesianIndices(Base.front(size(A)))
    @inbounds for c in eachindex(out)
        acc = zero(T)
        for I in modes
            acc += real(conj(A[I, c]) * B[I, c])
        end
        out[c] = acc
    end
    return out
end

"""
    inverse_nufft!(F, vals, p; rtol, maxiter)

Least-squares Fourier coefficients of the bandlimited fields matching `vals` at the plan's nodes:
conjugate gradients on `(AᴴA) F = Aᴴ vals`, with `A` the type-2 synthesis and `Aᴴ` its exact
adjoint.

This is the genuine non-uniform inverse. A single adjoint — one type-1 transform — is only the
inverse when the nodes carry quadrature weights, i.e. on a uniform set; at scattered points it is a
different operator, and using it is the same class of error as using an adjoint SHT off the
Clenshaw–Curtis grid.

`vals` carries the two real velocity components packed as `u + iv`, so this is one complex system
rather than two real ones and each iteration costs one transform pair instead of two. `A` is
complex-linear and the syntheses of Hermitian spectra are real, so the solution is exactly
`F_u + i F_v`; [`_unpack!`](@ref) splits it once at the end.

The column loop remains because `F` may still carry several independent systems.

Its working set comes from [`CGBuffers`](@ref), so repeated fits over one point set allocate none.
"""

"""
    CGBuffers{A,S,V}

The mode-space vectors and the point-space scratch one normal-equations solve iterates through.

Held apart from [`inverse_nufft!`](@ref) so a caller with many fields on one point set allocates
them once: the sizes depend on the node count and the mode count, and on nothing that varies
between calls.
"""
struct CGBuffers{A,S,V,L}
    b::A
    r::A
    d::A
    Ap::A
    scratch::S
    # `b − A F` in point space, carried alongside the mode-space residual. `A d` is already formed
    # each iteration, so this follows by one axpy and costs no transform. It is what the
    # discrepancy principle tests, and the mode-space residual cannot answer that.
    rpt::S
    rs::V
    rs_new::V
    den::V
    bn::V
    rn::V
    # The iteration's own `α` and `β`, which are the Lanczos recurrence for `AᴴA` and carry the
    # conditioning of the fit — see `HD.condition_estimate`.
    lanczos::L
end

function CGBuffers(::Type{T}, proto, nk::NTuple{D,Int}, npoints::Int, nc::Int,
                   maxiter::Int) where {D,T}
    m() = similar(proto, Complex{T}, nk..., nc)
    v() = zeros(T, nc)
    pt() = similar(proto, Complex{T}, npoints, nc)
    return CGBuffers(m(), m(), m(), m(), pt(), pt(),
                     v(), v(), v(), v(), v(), HD.CGLanczos(Float64, maxiter, nc))
end

function inverse_nufft!(F::AbstractArray{Complex{T}}, vals::AbstractMatrix{Complex{T}},
                        p::NUFFTPair{D,T}, cg::CGBuffers;
                        rtol::Real, maxiter::Int, atol::Real = 0) where {D,T}
    nc = size(F)[end]
    npt = size(vals, 1)
    modes = CartesianIndices(Base.front(size(F)))
    b = cg.b; analyze!(b, vals, p)
    r = cg.r; copyto!(r, b)
    d = cg.d; copyto!(d, b)
    Ap = cg.Ap
    scratch = cg.scratch
    rpt = cg.rpt; copyto!(rpt, vals)        # `b − A F` at `F = 0`
    fill!(F, zero(Complex{T}))
    rs = cg.rs; rs_new = cg.rs_new; den = cg.den; bn = cg.bn; rn = cg.rn
    _coldot(r, r, rs)
    _coldot(b, b, bn)
    all(iszero, bn) && return true
    check_atol = atol > 0
    converged = false
    for iter in 1:maxiter
        synthesize!(scratch, d, p)          # A d
        analyze!(Ap, scratch, p)            # Aᴴ A d
        _coldot(d, Ap, den)
        @inbounds for c in 1:nc
            iszero(den[c]) && continue
            α = rs[c] / den[c]
            for I in modes
                F[I, c] += α * d[I, c]
                r[I, c] -= α * Ap[I, c]
            end
            # `A F` advanced by `α A d`, so the point-space residual follows the same step.
            for i in 1:npt
                rpt[i, c] -= α * scratch[i, c]
            end
        end
        _coldot(r, r, rs_new)
        @inbounds for c in 1:nc
            β = iszero(rs[c]) ? zero(T) : rs_new[c] / rs[c]
            α = iszero(den[c]) ? zero(T) : rs[c] / den[c]
            HD.record!(cg.lanczos, c, iter, α, β)
        end
        # The discrepancy principle: once the samples are matched to their own noise, further
        # iterations fit the noise, and on an ill-conditioned point set that undoes the answer.
        if check_atol
            _coldot(rpt, rpt, rn)
            if all(c -> sqrt(rn[c]) <= atol, 1:nc)
                converged = true
                break
            end
        end
        if all(c -> sqrt(rs_new[c]) <= rtol * sqrt(bn[c]), 1:nc)
            converged = true
            break
        end
        @inbounds for c in 1:nc
            β = iszero(rs[c]) ? zero(T) : rs_new[c] / rs[c]
            for I in modes
                d[I, c] = r[I, c] + β * d[I, c]
            end
            rs[c] = rs_new[c]
        end
    end
    return converged
end

"""
    _reflect(i, n) -> Int

Index of mode `−k` given the index of `k`, in FINUFFT's centred layout `−n÷2 : (n−1)÷2`.

Index 1 holds `−n÷2`, which for even `n` is its own negative on the lattice (`n/2 ≡ −n/2 mod n`),
so it maps to itself.
"""
@inline _reflect(i::Int, n::Int) = i == 1 ? 1 : n + 2 - i

"""
    _unpack!(Fu, Fv, H)

Split the spectrum of `u + iv` into the spectra of the real fields `u` and `v`.

For real `u`, `û(−k) = conj(û(k))`, so with `ĥ = û + i v̂`

    ĥ(k) + conj(ĥ(−k)) = 2 û(k)
    ĥ(k) − conj(ĥ(−k)) = 2i v̂(k)

Both results are exactly Hermitian by construction, which is what the projection downstream needs.
This runs once after the fit, not once per iteration.
"""
function _unpack!(Fu::AbstractArray{Complex{T},D}, Fv::AbstractArray{Complex{T},D},
                  H::AbstractArray{Complex{T},D}) where {T,D}
    n = size(H)
    half = Complex{T}(0.5)
    @inbounds for I in CartesianIndices(n)
        J = CartesianIndex(ntuple(d -> _reflect(I[d], n[d]), Val(D)))
        c = conj(H[J])
        Fu[I] = half * (H[I] + c)
        Fv[I] = -im * half * (H[I] - c)
    end
    return Fu, Fv
end

# The fit recovers `prod(nk)` coefficients from `M` samples; fewer samples than modes leaves the
# normal equations singular and the result is a minimum-norm field, not the sampled one.
@inline function _require_resolvable(nk::NTuple{D,Int}, M::Int) where {D}
    prod(nk) <= M || throw(ArgumentError(
        "$(join(nk, "×")) = $(prod(nk)) modes cannot be determined from $M samples; reduce `nk`."))
    return nothing
end

"""
    _require_converged(ok, nk, M, solver)

Refuse a fit that did not reach its tolerance.

`prod(nk) ≤ M` is necessary and nowhere near sufficient: the normal equations lose conditioning as
the mode count approaches the sample count, well before they become singular. Measured on random
points at `nk = 48²`, the fit residual is `~2e-12` up to 0.58 modes per point and jumps to `6.9e-5`
at 0.77 — and that `6.9e-5` fit residual came out as a **16%** error in the decomposition.

The test is therefore whether the fit converged, which the iteration already knows, rather than a
density threshold with an invented constant.
"""
@inline function _require_converged(ok::Bool, nk::NTuple{D,Int}, M::Int, solver) where {D}
    ok && return nothing
    throw(ArgumentError(
        "the non-uniform fit did not reach rtol=$(solver.rtol) in $(solver.maxiter) iterations " *
        "for $(join(nk, "×")) modes at $M points ($(round(prod(nk) / M; digits = 2)) modes per " *
        "point). The normal equations lose conditioning as that ratio approaches 1; reduce `nk`, " *
        "supply more points, or raise `maxiter`. Returning the unconverged fit would give a " *
        "smooth field that does not match the samples."))
end

# Points are scaled to `[0, 2π)` by the period; an aperiodic grid reports `0` and would hand
# FINUFFT `Inf` coordinates, which segfaults in its spreader rather than raising.
@inline function _require_period(grid, d::Integer, ::Type{T}) where {T}
    L = T(FG.Grids.period(grid, d))
    (isfinite(L) && L > 0) || throw(ArgumentError(
        "direction $d of this point set has period $L; a non-uniform FFT expands in a periodic " *
        "basis. Pass `periodic = (true, …)` alongside `period` when building the grid."))
    return L
end

# Modes are laid out `-nk÷2 : (nk-1)÷2`, matching FINUFFT's ordering.
_mode_wavenumbers(::Type{T}, nk::Int, period::T) where {T} =
    T[T(2π) * m / period for m in (-div(nk, 2)):div(nk - 1, 2)]

"""
    ScatteredState

Everything a scattered decomposition reuses: the two guru plans with their node set already
uploaded, the mode-space and point-space buffers, the conjugate-gradient working set, and the
wavenumbers.

None of it depends on the field. Building the plans and re-sending the points costs milliseconds
per call, and a caller decomposing a time series over one point set pays that once here.

Plan-owning, so it carries an explicit `show` and a [`close!`](@ref): the default `show` of a
struct holding a FINUFFT plan can segfault while printing the plan's internals.

**One per task.** Every buffer here is written through during a solve, so concurrent decompositions
need one each; the plans inside are the shareable part and are what rebuilding costs.
"""
struct ScatteredState{D,T,H,A,B,C,K,G}
    pair::H
    packed::A          # (nk…, 1) complex, the packed pair in mode space
    buf::B             # (M, 1) complex, the packed pair at the points
    velocity_hat::C    # (nk…, D) complex
    cg::G
    ks::K
    L::NTuple{D,T}
end

Base.show(io::IO, s::ScatteredState{D,T}) where {D,T} =
    print(io, "ScatteredState{", D, ",", T, "}(", s.pair.npoints, " points → ",
          join(s.pair.nk, "×"), " modes)")

close!(s::ScatteredState) = (close_pair!(s.pair); s)

function HD.prepare_solver(
    solver::CartesianNUFFTSolver{D},
    grid::FG.Grids.UnstructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,D},
    ::HD.AbstractBoundaryCondition;
    backend = CB.SerialBackend(), shared = nothing,
) where {D,T}
    # One coordinate vector per direction is FlowGeometries' native layout and is what FINUFFT
    # takes, so the nodes reach the transform with no repacking.
    L = ntuple(d -> _require_period(grid, d, T), Val(D))
    coords = ntuple(d -> T(2π) .* collect(FG.Grids.coordinates(grid, d)) ./ L[d], Val(D))
    M = length(first(coords))
    nk = solver.nk
    _require_resolvable(nk, M)

    proto = first(coords)
    pair = nufft_pair(T, coords, nk, solver.tol, 1)
    packed = similar(proto, Complex{T}, nk..., 1)
    buf = similar(proto, Complex{T}, M, 1)
    velocity_hat = similar(proto, Complex{T}, nk..., D)
    cg = CGBuffers(T, proto, nk, M, 1, solver.maxiter)
    ks = ntuple(d -> _mode_wavenumbers(T, nk[d], L[d]), Val(D))
    return ScatteredState{D,T,typeof(pair),typeof(packed),typeof(buf),typeof(velocity_hat),
                          typeof(ks),typeof(cg)}(pair, packed, buf, velocity_hat, cg, ks, L)
end

function HD._decompose_spectral(
    solver::CartesianNUFFTSolver{D},
    ::FG.Geometry.AbstractCartesianGeometry,
    U::AbstractMatrix{T},
    grid::FG.Grids.UnstructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,D};
    state::Union{Nothing,ScatteredState} = nothing,
    kwargs...,
) where {D,T}
    # A state the caller supplies is reused; one built here is closed on the way out, so a one-shot
    # call frees its plans and a repeated one keeps them.
    owned = state === nothing
    st = owned ? HD.prepare_solver(solver, grid, HD.Neumann()) : state
    try
        return _decompose_scattered(solver, U, st)
    finally
        owned && close!(st)
    end
end

function _decompose_scattered(solver::CartesianNUFFTSolver{D}, U::AbstractMatrix{T},
                              st::ScatteredState{D,T}) where {D,T}
    p = st.pair
    nk = p.nk
    M = p.npoints
    size(U, 2) == D || throw(DimensionMismatch(
        "velocity has $(size(U, 2)) components for a $D-dimensional point set"))
    size(U, 1) == M || throw(DimensionMismatch(
        "velocity has $(size(U, 1)) rows against $M nodes in the plan"))

    # Components ride in pairs through one COMPLEX transform as `u + iv`. A NUFFT's cost is
    # dominated by spreading, which is per transform, so a pair costs what one component would;
    # batching by contrast merges calls without shrinking them. An odd component count leaves the
    # last one alone in the real part.
    npair = cld(D, 2)
    packed = st.packed
    buf = st.buf
    velocity_hat = st.velocity_hat
    modes = CartesianIndices(nk)
    colons = ntuple(_ -> Colon(), Val(D))

    for q in 1:npair
        a = 2q - 1
        b = 2q
        @inbounds for i in 1:M
            buf[i, 1] = Complex{T}(U[i, a], b <= D ? U[i, b] : zero(T))
        end
        ok = inverse_nufft!(packed, buf, p, st.cg; rtol = solver.rtol, maxiter = solver.maxiter,
                            atol = solver.atol)
        _require_converged(ok, nk, M, solver)
        # A converged fit says the samples are reproduced; the conditioning says whether they pick
        # out one coefficient vector or many.
        HD.warn_conditioning(HD.condition_estimate(st.cg.lanczos, 1), solver.rtol,
                             solver.condition_limit, M, prod(nk))
        Fa = view(velocity_hat, colons..., a)
        Fb = b <= D ? view(velocity_hat, colons..., b) : similar(Fa)
        _unpack!(Fa, Fb, view(packed, colons..., 1))
    end

    rot = similar(velocity_hat)
    div = similar(velocity_hat)
    harm = similar(velocity_hat)
    HD.helmholtz_project_spectral!(rot, div, harm, velocity_hat, st.ks)

    # The outputs follow the caller's array type.
    u_rot = similar(U, T, M, D)
    u_div = similar(U, T, M, D)
    u_harm = similar(U, T, M, D)
    # Packed on the way back too, so each pair of components is one synthesis.
    for (out, spec) in ((u_rot, rot), (u_div, div), (u_harm, harm)), q in 1:npair
        a = 2q - 1
        b = 2q
        @inbounds for I in modes
            packed[I, 1] = spec[I, a] + im * (b <= D ? spec[I, b] : zero(T))
        end
        synthesize!(buf, packed, p)
        @inbounds for i in 1:M
            out[i, a] = real(buf[i, 1])
            b <= D && (out[i, b] = imag(buf[i, 1]))
        end
    end
    return (; u_rot, u_div, u_harm)
end

function __init__()
    HD.register_spectral_solver!(SB.NUFFTSpectralBackend, CartesianNUFFTSolver; priority = 10)
end

end # module
