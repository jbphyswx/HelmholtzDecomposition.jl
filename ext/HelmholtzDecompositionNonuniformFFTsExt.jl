"""
    HelmholtzDecompositionNonuniformFFTsExt — scattered Cartesian decomposition, pure Julia.

Same algorithm as the FINUFFT extension — inverse NUFFT to recover the Fourier coefficients, exact
Leray projection in mode space, type-2 NUFFT back to the points — with two differences that decide
when it is preferred.

`PlanNUFFT` takes the element type positionally, and a **real** one selects the real-input
transform: the spectrum is then a half-spectrum, `(n₁÷2 + 1, n₂)`, exactly as `rfft` gives. Half the
spectral storage and no imaginary part carried through the fit.

It also takes a `KernelAbstractions.Backend` directly, and needs no binary, so the same code runs
on host and device.
"""
module HelmholtzDecompositionNonuniformFFTsExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using NonuniformFFTs: NonuniformFFTs as NU
using LinearAlgebra: LinearAlgebra

"""
    CartesianNonuniformFFTSolver(; nk = (64, 64), halfsupport = 8, rtol = 1e-10, maxiter = 1000)

Spectral Poisson/Helmholtz solver for scattered Cartesian samples in any dimension. `nk` is the
mode count per axis before the first is halved by the real transform, and its length fixes the
dimension; `rtol`/`maxiter` govern the conjugate gradients inside the exact inverse transform.

The iterations a fit needs grow with the mode-to-point ratio, and any ratio below one is accepted.
`maxiter` is sized so that conditioning is what an unconvergeable fit runs into: a budget set for a
well-sampled fit refuses a legitimate one near the top of that range, and the refusal then names
`nk` when the cap was the only thing missing.

`condition_limit` is the `κ(A)` above which the fit reports that the nodes determine the
coefficients too poorly for `rtol`. It defaults to `√(rtol/eps(T))`, the level where a
normal-equations solve stops reaching that tolerance; pass `Inf` for silence, or a number to
override.

`atol` is the norm of the measurement noise in `vals`, and setting it stops the fit once
`‖vals − A F‖` reaches it — the discrepancy principle. Iterations past that point fit the noise,
and on an ill-conditioned point set the coefficients travel a long way from their best value while
the residual keeps falling, so `rtol` alone never halts them.

`halfsupport` is the spreading kernel's half-width, which sets the transform's own accuracy — the
library's default of 4 gives about `1e-7`, so it is raised here: a solver whose answer stops at
`1e-7` while its iteration is asked for `1e-10` is reporting a tolerance it does not deliver.
"""
struct CartesianNonuniformFFTSolver{D,T<:AbstractFloat} <: HD.AbstractPoissonSolver
    nk::NTuple{D,Int}
    halfsupport::Int
    rtol::T
    atol::T
    maxiter::Int
    condition_limit::T
end

function CartesianNonuniformFFTSolver(; nk::NTuple{D,Int} = (64, 64), halfsupport::Int = 8,
                                      rtol::AbstractFloat = 1e-10, atol::Real = 0,
                                      maxiter::Int = 1000, condition_limit::Real = -1) where {D}
    T = typeof(float(rtol))
    lim = condition_limit < 0 ? HD.condition_limit(rtol, T) : float(condition_limit)
    return CartesianNonuniformFFTSolver{D,T}(nk, halfsupport, float(rtol), T(atol), maxiter,
                                             T(lim))
end

# A NUFFT expands in the same periodic basis an FFT does, over every sample it is handed.
HD.supports_boundary(::CartesianNonuniformFFTSolver, ::HD.AbstractBoundaryCondition) = true
HD.requires_full_domain(::CartesianNonuniformFFTSolver) = true

"""
    NUHandle{T,P}

The plan and its node count, in a type this package owns.

Naming a dependency-defined type in a signature is what makes an extension fail to precompile, so
the foreign plan is wrapped rather than passed around bare. `Base.show` is explicit for the usual
reason: the default would print the plan's internals.
"""
struct NUHandle{D,T,P}
    plan::P
    nk::NTuple{D,Int}
    npoints::Int
end

Base.show(io::IO, h::NUHandle{D,T}) where {D,T} =
    print(io, "NUHandle{", D, ",", T, "}(", h.npoints, " points → ", join(h.nk, "×"), " modes)")

# Half-spectrum shape: the real transform halves the first direction, as `rfft` does.
@inline _spectral_dims(nk::NTuple{D,Int}) where {D} =
    ntuple(d -> d == 1 ? nk[1] ÷ 2 + 1 : nk[d], Val(D))

function nu_handle(::Type{T}, coords::NTuple{D,AbstractVector{T}},
                   nk::NTuple{D,Int}, halfsupport::Int) where {D,T}
    plan = NU.PlanNUFFT(T, nk; m = NU.HalfSupport(halfsupport))
    NU.set_points!(plan, coords)
    return NUHandle{D,T,typeof(plan)}(plan, nk, length(first(coords)))
end

# Type 1 is the ADJOINT of type 2, not its inverse — the same relation the FINUFFT extension
# depends on, and why the fit below is a normal-equations solve rather than a single transform.
@inline analyze!(F, f, h::NUHandle) = (NU.exec_type1!(F, h.plan, f); F)
@inline synthesize!(f, F, h::NUHandle) = (NU.exec_type2!(f, h.plan, F); f)

"""
    inverse_nufft!(F, vals, h; rtol, maxiter)

Least-squares Fourier coefficients of the bandlimited field matching `vals` at the plan's nodes:
conjugate gradients on `(AᴴA) F = Aᴴ vals`.

A single adjoint is the inverse only when the nodes carry quadrature weights, i.e. on a uniform
set. At scattered points it is a different operator.
"""
function inverse_nufft!(F::AbstractArray{Complex{T}}, vals::AbstractVector{T},
                        h::NUHandle{D,T}, lanczos;
                        rtol::Real, maxiter::Int, atol::Real = 0) where {D,T}
    b = similar(F)
    analyze!(b, vals, h)
    r = copy(b)
    d = copy(b)
    Ap = similar(F)
    scratch = similar(vals)
    rpt = copy(vals)                    # `b − A F` at `F = 0`
    fill!(F, zero(Complex{T}))
    rs = real(LinearAlgebra.dot(r, r))
    bnorm = sqrt(real(LinearAlgebra.dot(b, b)))
    iszero(bnorm) && return true
    check_atol = atol > 0
    converged = false
    for iter in 1:maxiter
        synthesize!(scratch, d, h)      # A d
        analyze!(Ap, scratch, h)        # Aᴴ A d
        denom = real(LinearAlgebra.dot(d, Ap))
        iszero(denom) && break
        α = rs / denom
        @. F += α * d
        @. r -= α * Ap
        # `A F` advanced by `α A d`, so the point-space residual follows the same step. It costs
        # one axpy and is what the discrepancy principle tests.
        @. rpt -= α * scratch
        rs_new = real(LinearAlgebra.dot(r, r))
        # `α` and `β` are the Lanczos recurrence for `AᴴA`; they carry the fit's conditioning.
        HD.record!(lanczos, 1, iter, α, iszero(rs) ? zero(T) : rs_new / rs)
        # Once the samples are matched to their own noise, further iterations fit the noise, and on
        # an ill-conditioned point set that undoes the answer.
        if check_atol && sqrt(real(LinearAlgebra.dot(rpt, rpt))) <= atol
            converged = true
            break
        end
        if sqrt(rs_new) <= rtol * bnorm
            converged = true
            break
        end
        @. d = r + (rs_new / rs) * d
        rs = rs_new
    end
    return converged
end

# `prod(nk) ≤ M` is necessary and nowhere near sufficient: the normal equations lose conditioning
# well before they go singular, and an unconverged fit is a smooth field that does not match the
# samples. Measured on the FINUFFT path, a 6.9e-5 fit residual came out as a 16% decomposition
# error, so this refuses rather than warns.
@inline function _require_converged(ok::Bool, nk::NTuple{D,Int}, M::Int, solver) where {D}
    ok && return nothing
    throw(ArgumentError(
        "the non-uniform fit did not reach rtol=$(solver.rtol) in $(solver.maxiter) iterations " *
        "for $(join(nk, "×")) modes at $M points ($(round(prod(nk) / M; digits = 2)) modes per " *
        "point). Reduce `nk`, supply more points, or raise `maxiter`."))
end

# The fit recovers `prod(nk)` coefficients from `M` samples; fewer samples than modes leaves the
# normal equations singular and the result is a minimum-norm field, not the sampled one.
@inline function _require_resolvable(nk::NTuple{D,Int}, M::Int) where {D}
    prod(nk) <= M || throw(ArgumentError(
        "$(join(nk, "×")) = $(prod(nk)) modes cannot be determined from $M samples; reduce `nk`."))
    return nothing
end

# Points are scaled to `[0, 2π)` by the period; an aperiodic grid reports `0` and would hand the
# transform `Inf` coordinates, which segfaults rather than raising.
@inline function _require_period(grid, d::Integer, ::Type{T}) where {T}
    L = T(FG.Grids.period(grid, d))
    (isfinite(L) && L > 0) || throw(ArgumentError(
        "direction $d of this point set has period $L; a non-uniform FFT expands in a periodic " *
        "basis. Pass `periodic = (true, …)` alongside `period` when building the grid."))
    return L
end

# The halved direction runs 0 : n÷2; the other carries the usual FFT order, non-negative
# frequencies first and negative ones after.
_rfft_wavenumbers(::Type{T}, n::Int, period::T) where {T} =
    T[T(2π) * m / period for m in 0:(n ÷ 2)]

_fft_wavenumbers(::Type{T}, n::Int, period::T) where {T} =
    T[T(2π) * (m < cld(n, 2) ? m : m - n) / period for m in 0:(n - 1)]

"""
    ScatteredState

The plan with its node set uploaded, the mode-space and point-space buffers, and the wavenumbers.

None of it depends on the field, so a caller decomposing many fields over one point set builds it
once. Plan-owning, hence the explicit `show`.

**One per task.** Every buffer here is written through during a solve, so concurrent decompositions
need one each.
"""
struct ScatteredState{D,T,H,A,C,B,K,L}
    handle::H
    velocity_hat::A    # (kdims…, D)
    Fc::C              # (kdims…)
    buf::B
    ks::K
    L::NTuple{D,T}
    # The fit's own `α` and `β`, which carry its conditioning — see `HD.condition_estimate`.
    lanczos::L
end

Base.show(io::IO, s::ScatteredState{D,T}) where {D,T} =
    print(io, "ScatteredState{", D, ",", T, "}(", s.handle.npoints, " points → ",
          join(s.handle.nk, "×"), " modes)")

function HD.prepare_solver(
    solver::CartesianNonuniformFFTSolver{D},
    grid::FG.Grids.UnstructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,D},
    ::HD.AbstractBoundaryCondition;
    backend = CB.SerialBackend(), shared = nothing,
) where {D,T}
    L = ntuple(d -> _require_period(grid, d, T), Val(D))
    # One coordinate vector per direction is FlowGeometries' own layout and is what `set_points!`
    # takes, so nothing is repacked.
    coords = ntuple(d -> collect(T(2π) .* FG.Grids.coordinates(grid, d) ./ L[d]), Val(D))
    M = length(first(coords))
    nk = solver.nk
    _require_resolvable(nk, M)
    kdims = _spectral_dims(nk)

    proto = first(coords)
    h = nu_handle(T, coords, nk, solver.halfsupport)
    velocity_hat = similar(proto, Complex{T}, kdims..., D)
    Fc = similar(proto, Complex{T}, kdims)
    buf = similar(proto, T, M)
    ks = ntuple(d -> d == 1 ? _rfft_wavenumbers(T, nk[1], L[1]) :
                     _fft_wavenumbers(T, nk[d], L[d]), Val(D))
    lanczos = HD.CGLanczos(Float64, solver.maxiter, 1)
    return ScatteredState{D,T,typeof(h),typeof(velocity_hat),typeof(Fc),typeof(buf),typeof(ks),
                          typeof(lanczos)}(h, velocity_hat, Fc, buf, ks, L, lanczos)
end

function HD._decompose_spectral(
    solver::CartesianNonuniformFFTSolver{D},
    ::FG.Geometry.AbstractCartesianGeometry,
    U::AbstractMatrix{T},
    grid::FG.Grids.UnstructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,D};
    state::Union{Nothing,ScatteredState} = nothing,
    kwargs...,
) where {D,T}
    st = state === nothing ? HD.prepare_solver(solver, grid, HD.Neumann()) : state
    h = st.handle
    nk = h.nk
    M = h.npoints
    size(U, 2) == D || throw(DimensionMismatch(
        "velocity has $(size(U, 2)) components for a $D-dimensional point set"))
    size(U, 1) == M || throw(DimensionMismatch(
        "velocity has $(size(U, 1)) rows against $M nodes in the plan"))
    colons = ntuple(_ -> Colon(), Val(D))

    velocity_hat = st.velocity_hat
    buf = st.buf
    Fc = st.Fc
    # The real-input transform carries no imaginary part, so the components go through one at a
    # time; the `u + iv` packing the FINUFFT path uses buys nothing against a real transform.
    for c in 1:D
        @inbounds for i in 1:M
            buf[i] = U[i, c]
        end
        ok = inverse_nufft!(Fc, buf, h, st.lanczos; rtol = solver.rtol, maxiter = solver.maxiter,
                            atol = solver.atol)
        _require_converged(ok, nk, M, solver)
        # A converged fit says the samples are reproduced; the conditioning says whether they pick
        # out one coefficient vector or many.
        HD.warn_conditioning(HD.condition_estimate(st.lanczos, 1), solver.rtol,
                             solver.condition_limit, M, prod(nk))
        copyto!(view(velocity_hat, colons..., c), Fc)
    end

    rot = similar(velocity_hat)
    div = similar(velocity_hat)
    harm = similar(velocity_hat)
    HD.helmholtz_project_spectral!(rot, div, harm, velocity_hat, st.ks)

    u_rot = similar(U, T, M, D)
    u_div = similar(U, T, M, D)
    u_harm = similar(U, T, M, D)
    for (out, spec) in ((u_rot, rot), (u_div, div), (u_harm, harm)), c in 1:D
        copyto!(Fc, view(spec, colons..., c))
        synthesize!(buf, Fc, h)
        @inbounds for i in 1:M
            out[i, c] = buf[i]
        end
    end
    return (; u_rot, u_div, u_harm)
end

function __init__()
    # Behind FINUFFT's 10; name this solver explicitly to select it instead.
    HD.register_spectral_solver!(SB.NUFFTSpectralBackend, CartesianNonuniformFFTSolver;
                                 priority = 20)
end

end # module
