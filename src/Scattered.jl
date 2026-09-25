"""
    Scattered.jl — spectral decomposition of scattered Cartesian samples.

The velocity's Fourier coefficients are the least-squares fit to the samples: conjugate gradients
on `AᴴA F = Aᴴ u`, with `A` the type-2 NUFFT at the sample points and `Aᴴ` its adjoint, the type 1.
The fit is projected exactly in mode space, and each part returns to the points through one type 2.
The transforms are FlowTransformBindings plans over the library the solver names.

A single adjoint is the inverse only when the nodes carry quadrature weights, as on a uniform set;
at scattered points it is a different operator, the same error as an adjoint SHT off its quadrature
grid.
"""

"""
    CartesianNUFFTSolver(nufft = SpectralBackends.AutoSpectralBackend(); nk = (64, 64),
                         tol = nothing, rtol = 1e-10, atol = 0, maxiter = 1000,
                         condition_limit = -1)

Spectral Poisson/Helmholtz solver for scattered Cartesian samples in any dimension, over the NUFFT
library `nufft` names: `FlowTransformBindings.NonuniformFFTsBackend()`,
`FlowTransformBindings.FINUFFTBackend()`, or `AutoSpectralBackend()` for the first of the two that
is loaded, in that order. `nk` is the mode count per axis and its length fixes the dimension;
`rtol`/`maxiter` govern the conjugate gradients of the fit, and `tol` is the transforms' relative
accuracy, by default the `rtol` the fit is asked for.

The iterations a fit needs grow with the mode-to-point ratio, and any ratio below one is accepted.
`maxiter` is sized so that conditioning is what an unconvergeable fit runs into: a budget set for a
well-sampled fit refuses a legitimate one near the top of that range, and the refusal then names
`nk` when the cap was the only thing missing.

`condition_limit` is the `κ(A)` above which the fit reports that the nodes determine the
coefficients too poorly for `rtol`. It defaults to `√(rtol/eps(T))`, the level where a
normal-equations solve stops reaching that tolerance; pass `Inf` for silence, or a number to
override. The estimate itself costs two scalars per iteration — see [`condition_estimate`](@ref).

`atol` is the norm of the measurement noise in the samples, and setting it stops the fit once
`‖u − A F‖` reaches it — the discrepancy principle. Iterations past that point fit the noise: on a
well-conditioned point set they achieve nothing, and on an ill-conditioned one the coefficients
travel a long way from their best value while the residual keeps falling, so `rtol` alone never
halts them. A caller who knows the noise in their data should set this.
"""
struct CartesianNUFFTSolver{L<:SpectralBackends.AbstractSpectralBackend,D,T<:AbstractFloat} <:
       AbstractPoissonSolver
    nufft::L
    nk::NTuple{D,Int}
    tol::T
    rtol::T
    atol::T
    maxiter::Int
    condition_limit::T
end

function CartesianNUFFTSolver(nufft::SpectralBackends.AbstractSpectralBackend =
                                  SpectralBackends.AutoSpectralBackend();
                              nk::NTuple{D,Int} = (64, 64),
                              tol::Union{Nothing,AbstractFloat} = nothing,
                              rtol::AbstractFloat = 1e-10, atol::Real = 0, maxiter::Int = 1000,
                              condition_limit::Real = -1) where {D}
    _require_nufft_tag(nufft)
    t = tol === nothing ? rtol : tol
    T = promote_type(typeof(float(t)), typeof(float(rtol)))
    # The keyword shadows the function of the same name, so the module qualifies it.
    lim = condition_limit < 0 ? HelmholtzDecomposition.condition_limit(rtol, T) :
          float(condition_limit)
    return CartesianNUFFTSolver{typeof(nufft),D,T}(nufft, nk, T(t), T(rtol), T(atol), maxiter,
                                                   T(lim))
end

_require_nufft_tag(::SpectralBackends.AbstractNUFFTSpectralBackend) = nothing
_require_nufft_tag(::SpectralBackends.AbstractAutoSpectralBackend) = nothing
_require_nufft_tag(t::SpectralBackends.AbstractSpectralBackend) = throw(ArgumentError(
    "CartesianNUFFTSolver runs a nonuniform FFT; got $(nameof(typeof(t))). Pass " *
    "FlowTransformBindings.NonuniformFFTsBackend(), FlowTransformBindings.FINUFFTBackend() or " *
    "SpectralBackends.AutoSpectralBackend()."))

"""
    _nufft_library(tag) -> FlowTransformBindings tag or nothing

The loaded library a solver's transforms run on: the one `tag` names, or under
`AutoSpectralBackend` NonuniformFFTs, then FINUFFT. `nothing` when that library is not loaded.
"""
_nufft_library(t::SpectralBackends.AbstractSpectralBackend) =
    FlowTransformBindings.is_available(t) ? t : nothing
function _nufft_library(::SpectralBackends.AbstractAutoSpectralBackend)
    FlowTransformBindings.is_available(FlowTransformBindings.NonuniformFFTsBackend()) &&
        return FlowTransformBindings.NonuniformFFTsBackend()
    FlowTransformBindings.is_available(FlowTransformBindings.FINUFFTBackend()) &&
        return FlowTransformBindings.FINUFFTBackend()
    return nothing
end

const _CartesianCloud{T,D} =
    FlowGeometries.Grids.UnstructuredGrid{T,<:FlowGeometries.Geometry.AbstractCartesianGeometry,D}

# A NUFFT expands in the same periodic basis an FFT does, over every sample it is handed.
supports_boundary(::CartesianNUFFTSolver, ::AbstractBoundaryCondition) = true
requires_full_domain(::CartesianNUFFTSolver) = true

# The solver expands a point cloud, and its methods take an `UnstructuredGrid` at a matching
# dimension. A rectilinear grid reaches it only through selection, which then has no
# `solve_poisson!` to call, and a solver whose `D` disagrees with the grid carries the wrong mode
# count in its own type.
supports_sampling(::CartesianNUFFTSolver, grid) = false
supports_sampling(::CartesianNUFFTSolver{L,D},
                  ::_CartesianCloud{T,D}) where {L<:SpectralBackends.AbstractSpectralBackend,D,T} =
    true

_sampling_message(::CartesianNUFFTSolver{L,D}, grid) where {L,D} =
    "CartesianNUFFTSolver expands a cloud of $D-dimensional samples and takes an " *
    "`UnstructuredGrid` over that many directions; this grid is $(nameof(typeof(grid))). Build " *
    "the points with `FlowGeometries.Grids.UnstructuredGrid`, or solve a rectilinear grid with a " *
    "transform defined on it or with the iterative solver."

library_loaded(s::CartesianNUFFTSolver) = _nufft_library(s.nufft) !== nothing

_unavailable_message(s::CartesianNUFFTSolver) =
    "CartesianNUFFTSolver over $(nameof(typeof(s.nufft))) needs its NUFFT library: run " *
    "`using NonuniformFFTs` or `using FINUFFT`, and name it with " *
    "`CartesianNUFFTSolver(FlowTransformBindings.NonuniformFFTsBackend(); …)` or leave the " *
    "default."

"""
    _auto_nufft_solver(grid) -> CartesianNUFFTSolver

The scattered solver [`AutoSolver`](@ref) tries on `grid`: on a Cartesian point cloud one mode
count per direction, over the library `AutoSpectralBackend` resolves; on any other grid the
default, which refuses it.

The dimension rides in the solver's own type and the mode count has to be resolvable from the
samples, so neither can come from a constructor default. The count leaves a factor of two between
`prod(nk)` and the node count, which is the margin `_require_resolvable` asks for.

The direction count is the grid's own type parameter. `ndims` of a point cloud is `1`: its cells
carry one flat address each, and that is unrelated to how many directions they sit in.
"""
function _auto_nufft_solver(grid::_CartesianCloud{T,D}) where {T,D}
    M = length(FlowGeometries.Grids.mask(grid))
    n = max(2, 2 * fld(floor(Int, (M / 2)^(1 / D)), 2))
    return CartesianNUFFTSolver(; nk = ntuple(_ -> n, Val(D)))
end
_auto_nufft_solver(grid::FlowGeometries.Grids.AbstractGrid) = CartesianNUFFTSolver()

# ---------------------------------------------------------------------------
# How the real velocity components ride the transforms
# ---------------------------------------------------------------------------

# A library with a real-data transform takes one component per fit. A library that transforms
# complex values takes two, as `u + iv`: its cost is the spreading, which is per transform, so a
# pair costs what one component does.
struct _RealComponents end
struct _ComplexPairs end

_packing(lib::SpectralBackends.AbstractNUFFTSpectralBackend) =
    FlowTransformBindings.has_real_transform(lib) ? _RealComponents() : _ComplexPairs()

_value_type(::_RealComponents, ::Type{T}) where {T} = T
_value_type(::_ComplexPairs, ::Type{T}) where {T} = Complex{T}

# A real-data transform's half spectrum sits in FFT order; the complex pairs are unpacked through
# the reflection `k → −k`, which the centered order makes an index formula.
_mode_order(::_RealComponents) = FlowTransformBindings.FFTModes()
_mode_order(::_ComplexPairs) = FlowTransformBindings.CenteredModes()

# The fits a velocity of `D` components takes.
_nfits(::_RealComponents, D::Int) = D
_nfits(::_ComplexPairs, D::Int) = cld(D, 2)

# The library's own thread count under the backend a decomposition runs on.
_library_threads(::ComputationalBackends.AbstractThreadedBackend) = Threads.nthreads()
_library_threads(::ComputationalBackends.AbstractExecutionBackend) = 1

# ---------------------------------------------------------------------------
# The normal-equations fit
# ---------------------------------------------------------------------------

"""
    CGBuffers{A,S,V,L}

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
    # conditioning of the fit — see [`condition_estimate`](@ref).
    lanczos::L
end

function CGBuffers(::Type{T}, modes::AbstractArray, values::AbstractArray,
                   maxiter::Int) where {T}
    nc = size(modes)[end]
    v() = zeros(T, nc)
    return CGBuffers(similar(modes), similar(modes), similar(modes), similar(modes),
                     similar(values), similar(values), v(), v(), v(), v(), v(),
                     CGLanczos(Float64, maxiter, nc))
end

"""
    _coldot(A, B, out, weight) -> out

Column-wise inner products of arrays whose last axis holds independent systems sharing a plan:
`out[c] = Σ_I w(I) Re(conj(A[I, c]) B[I, c])`, with `w(I) = weight[I[1]]`, or `1` for `nothing`.

A real-data transform stores the half spectrum, and its type 1 is the adjoint of its type 2 under
the inner product that weights each row `k₁ > 0` by two, the row standing for its conjugate at
`−k` as well (`FlowTransformBindings.plan_nufft`). The fit's operator `type 1 ∘ type 2` is
self-adjoint under that product, so conjugate gradients take their scalars from it.
"""
@inline function _coldot(A::AbstractArray, B::AbstractArray, out::AbstractVector{T},
                         ::Nothing) where {T}
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

@inline function _coldot(A::AbstractArray, B::AbstractArray, out::AbstractVector{T},
                         weight::AbstractVector) where {T}
    modes = CartesianIndices(Base.front(size(A)))
    @inbounds for c in eachindex(out)
        acc = zero(T)
        for I in modes
            acc += weight[I[1]] * real(conj(A[I, c]) * B[I, c])
        end
        out[c] = acc
    end
    return out
end

"""
    inverse_nufft!(F, vals, plan, cg, weight; rtol, maxiter, atol = 0) -> Bool

Least-squares Fourier coefficients of the bandlimited fields matching `vals` at the plan's nodes:
conjugate gradients on `(AᴴA) F = Aᴴ vals`, with `A` the type-2 synthesis and `Aᴴ` its adjoint,
in the mode-space inner product `weight` names (see [`_coldot`](@ref)). Returns whether the fit
reached its tolerance.

Its working set comes from [`CGBuffers`](@ref), so repeated fits over one point set allocate none.
"""
function inverse_nufft!(F::AbstractArray, vals::AbstractArray,
                        plan::FlowTransformBindings.AbstractNUFFTPlan, cg::CGBuffers, weight;
                        rtol::Real, maxiter::Int, atol::Real = 0)
    nc = size(F)[end]
    npt = size(vals, 1)
    modes = CartesianIndices(Base.front(size(F)))
    b = cg.b
    FlowTransformBindings.nufft_type1!(b, plan, vals)
    r = cg.r; copyto!(r, b)
    d = cg.d; copyto!(d, b)
    Ap = cg.Ap
    scratch = cg.scratch
    rpt = cg.rpt; copyto!(rpt, vals)        # `b − A F` at `F = 0`
    fill!(F, zero(eltype(F)))
    rs = cg.rs; rs_new = cg.rs_new; den = cg.den; bn = cg.bn; rn = cg.rn
    _coldot(r, r, rs, weight)
    _coldot(b, b, bn, weight)
    all(iszero, bn) && return true
    check_atol = atol > 0
    converged = false
    for iter in 1:maxiter
        FlowTransformBindings.nufft_type2!(scratch, plan, d)      # A d
        FlowTransformBindings.nufft_type1!(Ap, plan, scratch)     # Aᴴ A d
        _coldot(d, Ap, den, weight)
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
        _coldot(r, r, rs_new, weight)
        @inbounds for c in 1:nc
            β = iszero(rs[c]) ? zero(eltype(rs)) : rs_new[c] / rs[c]
            α = iszero(den[c]) ? zero(eltype(rs)) : rs[c] / den[c]
            record!(cg.lanczos, c, iter, α, β)
        end
        # The discrepancy principle: once the samples are matched to their own noise, further
        # iterations fit the noise, and on an ill-conditioned point set that undoes the answer.
        if check_atol
            _coldot(rpt, rpt, rn, nothing)
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
            β = iszero(rs[c]) ? zero(eltype(rs)) : rs_new[c] / rs[c]
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

Index of mode `−k` given the index `i` of `k`, in the centered layout `−⌊n/2⌋ : ⌈n/2⌉ − 1`.

On an even axis index 1 holds `−n/2`, whose partner `+n/2` lies outside the layout; it maps to
itself, `n/2 ≡ −n/2 mod n` on the lattice.
"""
@inline function _reflect(i::Int, n::Int)
    j = 2 * (n ÷ 2) + 2 - i
    return j > n ? i : j
end

"""
    _unpack!(Fu, Fv, H)

Split the spectrum of `u + iv` into the spectra of the real fields `u` and `v`.

For real `u`, `û(−k) = conj(û(k))`, so with `ĥ = û + i v̂`

    ĥ(k) + conj(ĥ(−k)) = 2 û(k)
    ĥ(k) − conj(ĥ(−k)) = 2i v̂(k)

Both results are exactly Hermitian by construction, which the projection downstream needs.
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

# The fit recovers `prod(nk)` coefficients from `M` samples. Fewer samples than modes leave the
# normal equations singular, and the fit returns the minimum-norm field among the many that match
# the samples.
@inline function _require_resolvable(nk::NTuple{D,Int}, M::Int) where {D}
    prod(nk) <= M || throw(ArgumentError(
        "$(join(nk, "×")) = $(prod(nk)) modes cannot be determined from $M samples; reduce `nk`."))
    return nothing
end

"""
    _require_converged(ok, nk, M, solver)

Refuse a fit that did not reach its tolerance.

`prod(nk) ≤ M` is necessary and nowhere near sufficient: the normal equations lose conditioning as
the mode count approaches the sample count, well before they become singular, and an unconverged
fit is a smooth field that does not match the samples. The test is therefore whether the fit
converged, which the iteration already knows.
"""
@inline function _require_converged(ok::Bool, nk::NTuple{D,Int}, M::Int, solver) where {D}
    ok && return nothing
    throw(ArgumentError(
        "the non-uniform fit did not reach rtol=$(solver.rtol) in $(solver.maxiter) iterations " *
        "for $(join(nk, "×")) modes at $M points ($(round(prod(nk) / M; digits = 2)) modes per " *
        "point). The normal equations lose conditioning as that ratio approaches 1; reduce `nk`, " *
        "supply more points, or raise `maxiter`."))
end

# A non-uniform FFT expands in a periodic basis over the grid's period, and an aperiodic grid
# reports a period of `0`.
@inline function _require_period(grid, d::Integer, ::Type{T}) where {T}
    L = T(FlowGeometries.Grids.period(grid, d))
    (isfinite(L) && L > 0) || throw(ArgumentError(
        "direction $d of this point set has period $L; a non-uniform FFT expands in a periodic " *
        "basis. Pass `periodic = (true, …)` alongside `period` when building the grid."))
    return L
end

# ---------------------------------------------------------------------------
# State and decomposition
# ---------------------------------------------------------------------------

"""
    ScatteredState

Everything a scattered decomposition reuses: the library plan with its node set, the mode-space
and point-space buffers of one fit, the conjugate-gradient working set, the wavenumbers, and the
inner-product weights of the mode space.

None of it depends on the field, so a caller decomposing a time series over one point set builds
it once. It owns the library plan, which [`close!`](@ref) releases.

**One per task.** Every buffer here is written through during a solve, so concurrent
decompositions need one each.
"""
struct ScatteredState{D,T,S,P,A,B,C,G,K,W}
    packing::S
    plan::P
    packed::A          # one fit's coefficients: (mode_size…, 1)
    buf::B             # one fit's samples: (M, 1), real for one component, complex for a pair
    velocity_hat::C    # (mode_size…, D)
    cg::G
    ks::K
    weight::W          # axis-1 inner-product weights of a half spectrum; `nothing` otherwise
end

Base.show(io::IO, s::ScatteredState{D,T}) where {D,T} =
    print(io, "ScatteredState{", D, ",", T, "}(", FlowTransformBindings.npoints(s.plan),
          " points → ", join(FlowTransformBindings.nmodes(s.plan), "×"), " modes)")

"""
    close!(state::ScatteredState) -> state

Release the library plan the state holds. Idempotent.
"""
close!(s::ScatteredState) = (FlowTransformBindings.close!(s.plan); s)

# The axis-1 weights of a half spectrum's inner product: one where `k₁ = 0`, two where `k₁ > 0`.
_inner_weights(::_ComplexPairs, plan, ::Type{T}) where {T} = nothing
_inner_weights(::_RealComponents, plan, ::Type{T}) where {T} =
    T[k == 0 ? one(T) : T(2) for k in FlowTransformBindings.mode_frequencies(plan, 1)]

function prepare_solver(solver::CartesianNUFFTSolver{L,D}, grid::_CartesianCloud{T,D},
                        ::AbstractBoundaryCondition;
                        backend = ComputationalBackends.SerialBackend(),
                        shared = nothing) where {L,D,T}
    lib = _nufft_library(solver.nufft)
    lib === nothing && throw(ArgumentError(_unavailable_message(solver)))
    return _scattered_state(_packing(lib), lib, solver, grid, T, Val(D), backend)
end

function _scattered_state(packing, lib, solver::CartesianNUFFTSolver, grid, ::Type{T}, ::Val{D},
                          backend) where {T,D}
    periods = ntuple(d -> _require_period(grid, d, T), Val(D))
    # One coordinate vector per direction is FlowGeometries' own layout and is what the plan
    # takes, so the nodes reach it with no repacking.
    coords = ntuple(d -> FlowGeometries.Grids.coordinates(grid, d), Val(D))
    M = length(first(coords))
    nk = solver.nk
    _require_resolvable(nk, M)
    VT = _value_type(packing, T)
    plan = FlowTransformBindings.plan_nufft(lib, VT, coords, nk; tol = solver.tol,
        order = _mode_order(packing), period = periods, origin = zero(T),
        nthreads = _library_threads(backend))
    kdims = FlowTransformBindings.mode_size(plan)
    proto = first(coords)
    packed = similar(proto, Complex{T}, kdims..., 1)
    buf = similar(proto, VT, M, 1)
    velocity_hat = similar(proto, Complex{T}, kdims..., D)
    cg = CGBuffers(T, packed, buf, solver.maxiter)
    ks = ntuple(Val(D)) do d
        T[T(2π) * k / periods[d] for k in FlowTransformBindings.mode_frequencies(plan, d)]
    end
    weight = _inner_weights(packing, plan, T)
    return ScatteredState{D,T,typeof(packing),typeof(plan),typeof(packed),typeof(buf),
                          typeof(velocity_hat),typeof(cg),typeof(ks),typeof(weight)}(
        packing, plan, packed, buf, velocity_hat, cg, ks, weight)
end

function _decompose_spectral(solver::CartesianNUFFTSolver{L,D},
                             ::FlowGeometries.Geometry.AbstractCartesianGeometry,
                             U::AbstractMatrix{T}, grid::_CartesianCloud{T,D};
                             state::Union{Nothing,ScatteredState} = nothing,
                             backend = ComputationalBackends.AutoBackend(),
                             kwargs...) where {L,D,T}
    # A state the caller supplies is reused; one built here is closed on the way out, so a one-shot
    # call frees its plan and a repeated one keeps it.
    owned = state === nothing
    st = owned ?
         prepare_solver(solver, grid, Neumann(); backend = resolve_execution_backend(backend)) :
         state
    try
        return _decompose_scattered(solver, U, st)
    finally
        owned && close!(st)
    end
end

function _decompose_scattered(solver::CartesianNUFFTSolver{L,D}, U::AbstractMatrix{T},
                              st::ScatteredState{D,T}) where {L,D,T}
    nk = solver.nk
    M = FlowTransformBindings.npoints(st.plan)
    size(U, 2) == D || throw(DimensionMismatch(
        "velocity has $(size(U, 2)) components for a $D-dimensional point set"))
    size(U, 1) == M || throw(DimensionMismatch(
        "velocity has $(size(U, 1)) rows against $M nodes in the plan"))

    velocity_hat = st.velocity_hat
    for q in 1:_nfits(st.packing, D)
        _load_samples!(st.buf, st.packing, U, q, D)
        ok = inverse_nufft!(st.packed, st.buf, st.plan, st.cg, st.weight; rtol = solver.rtol,
                            maxiter = solver.maxiter, atol = solver.atol)
        _require_converged(ok, nk, M, solver)
        # A converged fit says the samples are reproduced; the conditioning says whether they pick
        # out one coefficient vector or many.
        warn_conditioning(condition_estimate(st.cg.lanczos, 1), solver.rtol,
                          solver.condition_limit, M, prod(nk))
        _store_fit!(velocity_hat, st.packing, st.packed, q, D)
    end

    rot = similar(velocity_hat)
    div = similar(velocity_hat)
    harm = similar(velocity_hat)
    helmholtz_project_spectral!(rot, div, harm, velocity_hat, st.ks)

    # The outputs follow the caller's array type.
    u_rot = similar(U, T, M, D)
    u_div = similar(U, T, M, D)
    u_harm = similar(U, T, M, D)
    for (out, spec) in ((u_rot, rot), (u_div, div), (u_harm, harm)), q in 1:_nfits(st.packing, D)
        _load_modes!(st.packed, st.packing, spec, q, D)
        FlowTransformBindings.nufft_type2!(st.buf, st.plan, st.packed)
        _store_samples!(out, st.packing, st.buf, q, D)
    end
    return (; u_rot, u_div, u_harm)
end

# Fit `q`'s samples: component `q`, or components `2q − 1` and `2q` as `u + iv`. An odd component
# count leaves the last one alone in the real part.
function _load_samples!(buf, ::_RealComponents, U, q::Int, D::Int)
    @inbounds for i in axes(U, 1)
        buf[i, 1] = U[i, q]
    end
    return buf
end
function _load_samples!(buf, ::_ComplexPairs, U, q::Int, D::Int)
    a, b = 2q - 1, 2q
    @inbounds for i in axes(U, 1)
        buf[i, 1] = complex(U[i, a], b <= D ? U[i, b] : zero(eltype(U)))
    end
    return buf
end

function _store_fit!(velocity_hat, ::_RealComponents, packed, q::Int, D::Int)
    copyto!(selectdim(velocity_hat, ndims(velocity_hat), q), selectdim(packed, ndims(packed), 1))
    return velocity_hat
end
function _store_fit!(velocity_hat, ::_ComplexPairs, packed, q::Int, D::Int)
    a, b = 2q - 1, 2q
    Fa = selectdim(velocity_hat, ndims(velocity_hat), a)
    Fb = b <= D ? selectdim(velocity_hat, ndims(velocity_hat), b) : similar(Fa)
    _unpack!(Fa, Fb, selectdim(packed, ndims(packed), 1))
    return velocity_hat
end

# Fit `q`'s coefficients for one part's synthesis: component `q`, or the pair packed as `u + iv`,
# so each pair of components returns through one type 2.
function _load_modes!(packed, ::_RealComponents, spec, q::Int, D::Int)
    copyto!(selectdim(packed, ndims(packed), 1), selectdim(spec, ndims(spec), q))
    return packed
end
function _load_modes!(packed, ::_ComplexPairs, spec, q::Int, D::Int)
    a, b = 2q - 1, 2q
    modes = CartesianIndices(Base.front(size(packed)))
    @inbounds for I in modes
        packed[I, 1] = spec[I, a] + im * (b <= D ? spec[I, b] : zero(eltype(spec)))
    end
    return packed
end

function _store_samples!(out, ::_RealComponents, buf, q::Int, D::Int)
    @inbounds for i in axes(out, 1)
        out[i, q] = buf[i, 1]
    end
    return out
end
function _store_samples!(out, ::_ComplexPairs, buf, q::Int, D::Int)
    a, b = 2q - 1, 2q
    @inbounds for i in axes(out, 1)
        out[i, a] = real(buf[i, 1])
        b <= D && (out[i, b] = imag(buf[i, 1]))
    end
    return out
end
