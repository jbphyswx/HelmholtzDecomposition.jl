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
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using NonuniformFFTs: NonuniformFFTs as NU
using LinearAlgebra: LinearAlgebra

"""
    CartesianNonuniformFFTSolver(; nk = (64, 64), halfsupport = 8, rtol = 1e-10, maxiter = 200)

Spectral Poisson/Helmholtz solver for scattered 2-D Cartesian samples. `nk` is the mode count per
axis before the first is halved by the real transform; `rtol`/`maxiter` govern the conjugate
gradients inside the exact inverse transform.

`halfsupport` is the spreading kernel's half-width, which sets the transform's own accuracy — the
library's default of 4 gives about `1e-7`, so it is raised here: a solver whose answer stops at
`1e-7` while its iteration is asked for `1e-10` is reporting a tolerance it does not deliver.
"""
struct CartesianNonuniformFFTSolver{T<:AbstractFloat} <: HD.AbstractPoissonSolver
    nk::NTuple{2,Int}
    halfsupport::Int
    rtol::T
    maxiter::Int
end

CartesianNonuniformFFTSolver(; nk::NTuple{2,Int} = (64, 64), halfsupport::Int = 8,
                             rtol::AbstractFloat = 1e-10, maxiter::Int = 200) =
    CartesianNonuniformFFTSolver(nk, halfsupport, rtol, maxiter)

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
struct NUHandle{T,P}
    plan::P
    nk::NTuple{2,Int}
    npoints::Int
end

Base.show(io::IO, h::NUHandle{T}) where {T} =
    print(io, "NUHandle{", T, "}(", h.npoints, " points → ", h.nk[1], "×", h.nk[2], " modes)")

# Half-spectrum shape: the real transform halves the first direction, as `rfft` does.
@inline _spectral_dims(nk::NTuple{2,Int}) = (nk[1] ÷ 2 + 1, nk[2])

function nu_handle(::Type{T}, x::AbstractVector{T}, y::AbstractVector{T},
                   nk::NTuple{2,Int}, halfsupport::Int) where {T}
    plan = NU.PlanNUFFT(T, nk; m = NU.HalfSupport(halfsupport))
    NU.set_points!(plan, (x, y))
    return NUHandle{T,typeof(plan)}(plan, nk, length(x))
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
function inverse_nufft!(F::AbstractMatrix{Complex{T}}, vals::AbstractVector{T},
                        h::NUHandle{T}; rtol::Real, maxiter::Int) where {T}
    b = similar(F)
    analyze!(b, vals, h)
    r = copy(b)
    d = copy(b)
    Ap = similar(F)
    scratch = similar(vals)
    fill!(F, zero(Complex{T}))
    rs = real(LinearAlgebra.dot(r, r))
    bnorm = sqrt(real(LinearAlgebra.dot(b, b)))
    iszero(bnorm) && return true
    converged = false
    for _ in 1:maxiter
        synthesize!(scratch, d, h)      # A d
        analyze!(Ap, scratch, h)        # Aᴴ A d
        denom = real(LinearAlgebra.dot(d, Ap))
        iszero(denom) && break
        α = rs / denom
        @. F += α * d
        @. r -= α * Ap
        rs_new = real(LinearAlgebra.dot(r, r))
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
@inline function _require_converged(ok::Bool, nk::NTuple{2,Int}, M::Int, solver)
    ok && return nothing
    throw(ArgumentError(
        "the non-uniform fit did not reach rtol=$(solver.rtol) in $(solver.maxiter) iterations " *
        "for $(nk[1])×$(nk[2]) modes at $M points ($(round(prod(nk) / M; digits = 2)) modes per " *
        "point). Reduce `nk`, supply more points, or raise `maxiter`."))
end

# The fit recovers `prod(nk)` coefficients from `M` samples; fewer samples than modes leaves the
# normal equations singular and the result is a minimum-norm field, not the sampled one.
@inline function _require_resolvable(nk::NTuple{2,Int}, M::Int)
    prod(nk) <= M || throw(ArgumentError(
        "$(nk[1])×$(nk[2]) = $(prod(nk)) modes cannot be determined from $M samples; reduce `nk`."))
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

function HD._decompose_spectral(
    solver::CartesianNonuniformFFTSolver,
    ::FG.Geometry.AbstractCartesianGeometry,
    U::AbstractMatrix{T},
    grid::FG.Grids.UnstructuredGrid{T,<:FG.Geometry.AbstractCartesianGeometry,2};
    kwargs...,
) where {T}
    Lx = _require_period(grid, 1, T)
    Ly = _require_period(grid, 2, T)
    # One coordinate vector per direction is FlowGeometries' own layout and is what `set_points!`
    # takes, so nothing is repacked.
    x = T(2π) .* FG.Grids.coordinates(grid, 1) ./ Lx
    y = T(2π) .* FG.Grids.coordinates(grid, 2) ./ Ly
    M = length(x)
    nk = solver.nk
    _require_resolvable(nk, M)
    kdims = _spectral_dims(nk)

    h = nu_handle(T, collect(x), collect(y), nk, solver.halfsupport)
    velocity_hat = similar(U, Complex{T}, kdims..., 2)
    buf = similar(U, T, M)
    Fc = similar(U, Complex{T}, kdims)
    for c in 1:2
        @inbounds for i in 1:M
            buf[i] = U[i, c]
        end
        ok = inverse_nufft!(Fc, buf, h; rtol = solver.rtol, maxiter = solver.maxiter)
        _require_converged(ok, nk, M, solver)
        copyto!(view(velocity_hat, :, :, c), Fc)
    end

    ks = (_rfft_wavenumbers(T, nk[1], Lx), _fft_wavenumbers(T, nk[2], Ly))
    rot = similar(velocity_hat)
    div = similar(velocity_hat)
    harm = similar(velocity_hat)
    HD.helmholtz_project_spectral!(rot, div, harm, velocity_hat, ks)

    u_rot = similar(U, T, M, 2)
    u_div = similar(U, T, M, 2)
    u_harm = similar(U, T, M, 2)
    for (out, spec) in ((u_rot, rot), (u_div, div), (u_harm, harm))
        for c in 1:2
            copyto!(Fc, view(spec, :, :, c))
            synthesize!(buf, Fc, h)
            @inbounds for i in 1:M
                out[i, c] = buf[i]
            end
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
