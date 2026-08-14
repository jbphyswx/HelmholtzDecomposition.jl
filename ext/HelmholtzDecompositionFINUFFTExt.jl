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
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra

"""
    CartesianNUFFTSolver(; nk = (64, 64), tol = 1e-8, rtol = 1e-10, maxiter = 200)

Spectral Poisson/Helmholtz solver for scattered 2-D Cartesian samples. `nk` is the mode count per
axis; `rtol`/`maxiter` govern the conjugate gradients inside the exact inverse transform.
"""
struct CartesianNUFFTSolver{T<:AbstractFloat} <: HD.AbstractPoissonSolver
    nk::NTuple{2,Int}
    tol::T
    rtol::T
    maxiter::Int
end

CartesianNUFFTSolver(; nk::NTuple{2,Int} = (64, 64), tol::AbstractFloat = 1e-8,
                     rtol::AbstractFloat = 1e-10, maxiter::Int = 200) =
    CartesianNUFFTSolver(nk, promote(tol, rtol)..., maxiter)

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
mutable struct NUFFTPair{T,P}
    analysis::P            # type 1: points → modes (the exact adjoint of synthesis)
    synthesis::P           # type 2: modes → points
    nk::NTuple{2,Int}
    npoints::Int
    open::Bool
end

Base.show(io::IO, p::NUFFTPair{T}) where {T} =
    print(io, "NUFFTPair{", T, "}(", p.npoints, " points → ", p.nk[1], "×", p.nk[2], " modes)")

function nufft_pair(::Type{T}, x::AbstractVector{T}, y::AbstractVector{T},
                    nk::NTuple{2,Int}, tol::Real, ntrans::Int) where {T}
    ms = FINUFFT.BIGINT[nk[1], nk[2]]
    # `iflag` is opposite between the two so that analysis is the exact adjoint of synthesis,
    # which is what makes `AᴴA` symmetric positive definite and conjugate gradients valid.
    #
    # `ntrans` transforms per `exec`. The velocity's components share a node set, so they go
    # through together. This batches full-size complex transforms; it does not shrink them.
    a = FINUFFT.finufft_makeplan(1, ms, -1, ntrans, T(tol); dtype = T)
    s = FINUFFT.finufft_makeplan(2, ms, +1, ntrans, T(tol); dtype = T)
    xc, yc = collect(x), collect(y)          # guru setpts requires contiguous storage
    FINUFFT.finufft_setpts!(a, xc, yc)
    FINUFFT.finufft_setpts!(s, xc, yc)
    return NUFFTPair{T,typeof(a)}(a, s, nk, length(xc), true)
end

function close_pair!(p::NUFFTPair)
    p.open || return p
    FINUFFT.finufft_destroy!(p.analysis)
    FINUFFT.finufft_destroy!(p.synthesis)
    p.open = false
    return p
end

# `ntrans` transforms per `exec`, so the mode array is `(nk..., ntrans)` and the value array
# `(npoints, ntrans)`. Both must be contiguous, which is why they are whole arrays and not views.
@inline function analyze!(F::AbstractArray{Complex{T},3}, f::AbstractMatrix{Complex{T}},
                          p::NUFFTPair{T}) where {T}
    FINUFFT.finufft_exec!(p.analysis, f, F)
    return F
end

@inline function synthesize!(f::AbstractMatrix{Complex{T}}, F::AbstractArray{Complex{T},3},
                             p::NUFFTPair{T}) where {T}
    FINUFFT.finufft_exec!(p.synthesis, F, f)
    return f
end

# Column-wise inner product: the components are independent systems sharing a plan, so each keeps
# its own conjugate-gradient scalars.
@inline function _coldot(A::AbstractArray{Complex{T},3}, B::AbstractArray{Complex{T},3},
                         out::Vector{T}) where {T}
    nk1, nk2, nc = size(A)
    @inbounds for c in 1:nc
        acc = zero(T)
        for j in 1:nk2, i in 1:nk1
            acc += real(conj(A[i, j, c]) * B[i, j, c])
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
"""
function inverse_nufft!(F::AbstractArray{Complex{T},3}, vals::AbstractMatrix{Complex{T}},
                        p::NUFFTPair{T}; rtol::Real, maxiter::Int) where {T}
    nc = size(F, 3)
    b = similar(F); analyze!(b, vals, p)
    r = copy(b); d = copy(b); Ap = similar(F)
    scratch = Matrix{Complex{T}}(undef, p.npoints, nc)
    fill!(F, zero(Complex{T}))
    rs = zeros(T, nc); rs_new = zeros(T, nc); den = zeros(T, nc); bn = zeros(T, nc)
    _coldot(r, r, rs)
    _coldot(b, b, bn)
    all(iszero, bn) && return true
    converged = false
    for _ in 1:maxiter
        synthesize!(scratch, d, p)          # A d
        analyze!(Ap, scratch, p)            # Aᴴ A d
        _coldot(d, Ap, den)
        @inbounds for c in 1:nc
            iszero(den[c]) && continue
            α = rs[c] / den[c]
            for j in axes(F, 2), i in axes(F, 1)
                F[i, j, c] += α * d[i, j, c]
                r[i, j, c] -= α * Ap[i, j, c]
            end
        end
        _coldot(r, r, rs_new)
        if all(c -> sqrt(rs_new[c]) <= rtol * sqrt(bn[c]), 1:nc)
            converged = true
            break
        end
        @inbounds for c in 1:nc
            β = iszero(rs[c]) ? zero(T) : rs_new[c] / rs[c]
            for j in axes(F, 2), i in axes(F, 1)
                d[i, j, c] = r[i, j, c] + β * d[i, j, c]
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
function _unpack!(Fu::AbstractMatrix{Complex{T}}, Fv::AbstractMatrix{Complex{T}},
                  H::AbstractMatrix{Complex{T}}) where {T}
    n1, n2 = size(H)
    half = Complex{T}(0.5)
    @inbounds for j in 1:n2, i in 1:n1
        c = conj(H[_reflect(i, n1), _reflect(j, n2)])
        Fu[i, j] = half * (H[i, j] + c)
        Fv[i, j] = -im * half * (H[i, j] - c)
    end
    return Fu, Fv
end

# The fit recovers `prod(nk)` coefficients from `M` samples; fewer samples than modes leaves the
# normal equations singular and the result is a minimum-norm field, not the sampled one.
@inline function _require_resolvable(nk::NTuple{2,Int}, M::Int)
    prod(nk) <= M || throw(ArgumentError(
        "$(nk[1])×$(nk[2]) = $(prod(nk)) modes cannot be determined from $M samples; reduce `nk`."))
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
@inline function _require_converged(ok::Bool, nk::NTuple{2,Int}, M::Int, solver)
    ok && return nothing
    throw(ArgumentError(
        "the non-uniform fit did not reach rtol=$(solver.rtol) in $(solver.maxiter) iterations " *
        "for $(nk[1])×$(nk[2]) modes at $M points ($(round(prod(nk) / M; digits = 2)) modes per " *
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
    _decompose_spectral(solver, geometry, U, grid::UnstructuredGrid)

Helmholtz decomposition of a velocity field sampled at scattered Cartesian points. Returns the
three physical parts at those points.
"""
function HD._decompose_spectral(
    solver::CartesianNUFFTSolver,
    ::FG.Geometry.AbstractCartesianGeometry,
    U::AbstractMatrix{T},
    grid::FG.Grids.UnstructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,2};
    kwargs...,
) where {T}
    # One coordinate vector per direction is FlowGeometries' native layout and is exactly what
    # FINUFFT takes; the previous point-major `(M, 2)` storage needed a copy per call to produce it.
    Lx = _require_period(grid, 1, T)
    Ly = _require_period(grid, 2, T)
    x = T(2π) .* collect(FG.Grids.coordinates(grid, 1)) ./ Lx
    y = T(2π) .* collect(FG.Grids.coordinates(grid, 2)) ./ Ly
    M = length(x)
    nk = solver.nk
    _require_resolvable(nk, M)

    # `ntrans = 1`: the two components ride in one COMPLEX transform as `u + iv`, so the fit is a
    # single system. A NUFFT's cost is dominated by spreading, which is per transform, so one
    # transform instead of two halves the work — unlike batching, which only merges calls.
    p = nufft_pair(T, x, y, nk, solver.tol, 1)
    try
        packed = similar(U, Complex{T}, nk[1], nk[2], 1)
        buf = similar(U, Complex{T}, M, 1)
        @inbounds for i in 1:M
            buf[i, 1] = Complex{T}(U[i, 1], U[i, 2])
        end
        ok = inverse_nufft!(packed, buf, p; rtol = solver.rtol, maxiter = solver.maxiter)
        _require_converged(ok, nk, M, solver)

        velocity_hat = similar(U, Complex{T}, nk[1], nk[2], 2)
        _unpack!(view(velocity_hat, :, :, 1), view(velocity_hat, :, :, 2),
                 view(packed, :, :, 1))

        ks = (_mode_wavenumbers(T, nk[1], Lx), _mode_wavenumbers(T, nk[2], Ly))
        rot = similar(velocity_hat)
        div = similar(velocity_hat)
        harm = similar(velocity_hat)
        HD.helmholtz_project_spectral!(rot, div, harm, velocity_hat, ks)

        # Follow the caller's array type rather than naming `Matrix`: these are returned.
        u_rot = similar(U, T, M, 2)
        u_div = similar(U, T, M, 2)
        u_harm = similar(U, T, M, 2)
        # Packed on the way back too: each part's two components are the real and imaginary parts
        # of one complex synthesis, so three transforms rather than six.
        packed_out = packed
        for (out, spec) in ((u_rot, rot), (u_div, div), (u_harm, harm))
            @inbounds for j in 1:nk[2], i in 1:nk[1]
                packed_out[i, j, 1] = spec[i, j, 1] + im * spec[i, j, 2]
            end
            synthesize!(buf, packed_out, p)
            @inbounds for i in 1:M
                out[i, 1] = real(buf[i, 1])
                out[i, 2] = imag(buf[i, 1])
            end
        end
        return (; u_rot, u_div, u_harm)
    finally
        close_pair!(p)
    end
end

function __init__()
    HD.register_spectral_solver!(SB.NUFFTSpectralBackend, CartesianNUFFTSolver; priority = 10)
end

end # module
