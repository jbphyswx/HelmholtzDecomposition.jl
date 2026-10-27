# Spectral decomposition of scattered Cartesian samples. The velocity's Fourier coefficients are the
# least-squares fit to the samples: conjugate gradients on `AᴴA F = Aᴴ u`, with `A` the type-2 NUFFT at
# the sample points and `Aᴴ` its adjoint, the type 1. The fit is projected exactly in mode space, and each
# part returns to the points through one type 2. The transforms are FlowTransformBindings plans over the
# library the solver names.
#
# A single adjoint is the inverse only when the nodes carry quadrature weights, as on a uniform set; at
# scattered points it is a different operator.

"""
    CartesianNUFFTSolver(nufft = SpectralBackends.AutoSpectralBackend(); nk = (64, 64),
                         tol = nothing, rtol = 1e-10, atol = 0, maxiter = 1000,
                         condition_limit = -1)

Spectral Poisson/Helmholtz solver for scattered Cartesian samples in any dimension, over the NUFFT
library `nufft` names: `FlowTransformBindings.NonuniformFFTsBackend()`,
`FlowTransformBindings.FINUFFTBackend()`, or `AutoSpectralBackend()` for NonuniformFFTs when it is
loaded and FINUFFT when only it is. `nk` is the mode count per axis and its length fixes the dimension;
the fitted series spans `|k_d| ≤ ⌊nk_d/2⌋`. The fit minimizes the squared misfit at the nodes weighted
by each node's measure, the grid's quadrature of the misfit's integral. `rtol`/`maxiter` govern the
conjugate gradients of the fit, and `tol` is the transforms' relative accuracy, by default the `rtol`
the fit is asked for.

The iterations a fit needs grow with the mode-to-point ratio, and any ratio below one is accepted. A
fit that stops at `maxiter` short of `rtol` is returned as it stands, with `converged == false` in the
result's `χ_solve` and `rot_solve`.

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

# A NUFFT expands in the same periodic basis an FFT does, over every sample it is handed. The fit's
# working set is host memory.
supports_boundary(::CartesianNUFFTSolver, ::FlowGeometries.Operators.AbstractBoundaryCondition) = true
requires_full_domain(::CartesianNUFFTSolver) = true
supports_backend(::CartesianNUFFTSolver, backend, grid) = backend isa _HostBackend

# The solver expands a point cloud, and its methods take an `UnstructuredGrid` at a matching
# dimension. A rectilinear grid reaches it only through selection, which then has no
# `solve_poisson!` to call, and a solver whose `D` disagrees with the grid carries the wrong mode
# count in its own type.
supports_sampling(::CartesianNUFFTSolver, grid) = false
supports_sampling(::CartesianNUFFTSolver{L,D},
                  ::_CartesianCloud{T,D}) where {L<:SpectralBackends.AbstractSpectralBackend,D,T} =
    true
sampling_known(s::CartesianNUFFTSolver, grid) = supports_sampling(s, grid)

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
samples, so neither can come from a constructor default. The count leaves four nodes per mode: for `M`
random nodes and `n` modes the fit's matrix has condition number near `(1 + √(n/M)) / (1 − √(n/M))`
(Marchenko–Pastur), 3 at a quarter, and unbounded as `n` approaches `M`.

The direction count is the grid's own type parameter. `ndims` of a point cloud is `1`: its cells
carry one flat address each, and that is unrelated to how many directions they sit in.
"""
function _auto_nufft_solver(grid::_CartesianCloud{T,D}) where {T,D}
    M = length(FlowGeometries.Grids.mask(grid))
    n = max(2, 2 * fld(floor(Int, (M / 4)^(1 / D)), 2))
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

# The modes a fit's transform holds for the requested counts `nk`. A real-data transform's series spans
# `|k_d| ≤ nk_d/2` on every axis. The complex pairs take one more mode on an even axis, so `+nk_d/2`
# sits beside `−nk_d/2`: the same space, and every mode has its reflection.
_plan_modes(::_RealComponents, nk::NTuple) = nk
_plan_modes(::_ComplexPairs, nk::NTuple) = map(n -> n + iseven(n), nk)

# The fits a velocity of `D` components takes.
_nfits(::_RealComponents, D::Int) = D
_nfits(::_ComplexPairs, D::Int) = cld(D, 2)

# The library's own thread count under the backend a decomposition runs on.
_library_threads(::ComputationalBackends.AbstractThreadedBackend) = Threads.nthreads()
_library_threads(::ComputationalBackends.AbstractExecutionBackend) = 1

# Keywords for an `AbstractFFTs` plan over `a` under `backend`. An extension whose library takes a
# thread count adds the method for the arrays it plans, so each plan runs on the backend's threads.
_plan_kwargs(a, backend) = (;)

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
    r::A
    d::A
    Ap::A
    scratch::S
    # `b − A F` in point space, carried alongside the mode-space residual where the discrepancy principle
    # is tested (`atol > 0`), and empty otherwise. `A d` is already formed each iteration, so this follows
    # by one axpy and costs no transform; the mode-space residual cannot answer that test.
    rpt::S
    # A point-space vector times the node weights, which the adjoint is applied to.
    wpt::S
    rs::V
    rs_new::V
    den::V
    bn::V
    rn::V
    # The iteration's own `α` and `β`, which are the Lanczos recurrence for `AᴴA` and carry the
    # conditioning of the fit — see [`condition_estimate`](@ref).
    lanczos::L
end

function CGBuffers(::Type{T}, modes::AbstractArray, values::AbstractArray, maxiter::Int,
                   noise::Bool) where {T}
    nc = size(modes)[end]
    v() = zeros(T, nc)
    return CGBuffers(similar(modes), similar(modes), similar(modes), similar(values),
                     similar(values, noise ? size(values, 1) : 0, nc), similar(values), v(), v(), v(), v(), v(),
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
@inline function _coldot(A::AbstractArray, B::AbstractArray, out::AbstractVector{T}, weight) where {T}
    modes = CartesianIndices(Base.front(size(A)))
    @inbounds for c in eachindex(out)
        acc = zero(T)
        for I in modes
            acc += _row_weight(weight, I[1]) * real(conj(A[I, c]) * B[I, c])
        end
        out[c] = acc
    end
    return out
end

# The inner-product weight of axis-1 row `i`: one throughout a full spectrum (`nothing`).
@inline _row_weight(::Nothing, ::Int) = true
@inline _row_weight(w::AbstractVector, i::Int) = @inbounds w[i]

"""
    inverse_nufft!(F, vals, plan, cg, weight, w; rtol, maxiter, atol = 0) -> SolverResult

Least-squares Fourier coefficients of the bandlimited fields matching `vals` at the plan's nodes,
each node's residual weighted by `w`: conjugate gradients on `(AᴴWA) F = AᴴW vals`, with `A` the
type-2 synthesis, `Aᴴ` its adjoint and `W = Diagonal(w)`, in the mode-space inner product `weight`
names (see [`_coldot`](@ref)). Returns whether every column reached its tolerance, the iterations
taken, and the largest relative mode-space residual.

Its working set comes from [`CGBuffers`](@ref), so repeated fits over one point set allocate none.
"""
function inverse_nufft!(F::AbstractArray, vals::AbstractArray,
                        plan::FlowTransformBindings.AbstractNUFFTPlan, cg::CGBuffers, weight,
                        w::AbstractVector; rtol::Real, maxiter::Int, atol::Real = 0)
    nc = size(F)[end]
    npt = size(vals, 1)
    modes = CartesianIndices(Base.front(size(F)))
    r, d, Ap, scratch, wpt, rpt = cg.r, cg.d, cg.Ap, cg.scratch, cg.wpt, cg.rpt
    check_atol = atol > 0
    check_atol && size(rpt, 1) != npt && throw(ArgumentError(
        "these buffers carry no point-space residual; build them for a fit with `atol > 0`"))
    # The residual at `F = 0` is `b = AᴴW u`, written straight into `r`.
    wpt .= w .* vals
    FlowTransformBindings.nufft_type1!(r, plan, wpt)
    copyto!(d, r)
    check_atol && copyto!(rpt, vals)        # `u − A F` at `F = 0`
    fill!(F, zero(eltype(F)))
    rs = cg.rs; rs_new = cg.rs_new; den = cg.den; bn = cg.bn; rn = cg.rn
    RT = eltype(rs)
    _coldot(r, r, rs, weight)
    copyto!(bn, rs)
    all(iszero, bn) && return SolverResult{RT}(true, 0, zero(RT))
    copyto!(rs_new, rs)
    converged = false
    iters = 0
    for iter in 1:maxiter
        iters = iter
        FlowTransformBindings.nufft_type2!(scratch, plan, d)      # A d
        wpt .= w .* scratch
        FlowTransformBindings.nufft_type1!(Ap, plan, wpt)         # Aᴴ W A d
        _coldot(d, Ap, den, weight)
        @inbounds for c in 1:nc
            iszero(den[c]) && (rs_new[c] = rs[c]; continue)
            α = rs[c] / den[c]
            acc = zero(RT)
            for I in modes
                F[I, c] += α * d[I, c]
                rI = r[I, c] - α * Ap[I, c]
                r[I, c] = rI
                acc += _row_weight(weight, I[1]) * abs2(rI)
            end
            rs_new[c] = acc
            # `A F` advanced by `α A d`, so the point-space residual follows the same step.
            if check_atol
                for i in 1:npt
                    rpt[i, c] -= α * scratch[i, c]
                end
            end
        end
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
    resid = zero(RT)
    for c in 1:nc
        iszero(bn[c]) || (resid = max(resid, sqrt(rs_new[c] / bn[c])))
    end
    return SolverResult{RT}(converged, iters, resid)
end

"""
    _reflect(i, n) -> Int

Index of mode `−k` given the index `i` of `k`, in the centered layout `−(n−1)/2 : (n−1)/2` of an odd
axis.
"""
@inline _reflect(i::Int, n::Int) = n + 1 - i

"""
    _unpack!(Fu, Fv, H)

Split the spectrum of `u + iv`, odd along every axis (see `_plan_modes`), into the spectra of the
real fields `u` and `v`.

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

# A non-uniform FFT expands in a periodic basis over the grid's period, and an aperiodic grid
# reports a period of `0`.
@inline function _require_period(grid, d::Integer, ::Type{T}) where {T}
    L = T(FlowGeometries.Grids.period(grid, d))
    (isfinite(L) && L > 0) || throw(ArgumentError(
        "direction $d of this point set has period $L; a non-uniform FFT expands in a periodic " *
        "basis. Pass `topology = (FlowGeometries.Grids.Periodic(), …)` alongside `period` when building the " *
        "grid."))
    return L
end

# ---------------------------------------------------------------------------
# State and decomposition
# ---------------------------------------------------------------------------

"""
    ScatteredState

Everything a scattered decomposition reuses: the library plan with its node set, the mode-space
and point-space buffers of one fit, the conjugate-gradient working set, the wavenumbers and their
derivative symbol, the inner-product weights of the mode space, each node's measure, which weights its
residual, and the spectra of the three parts and of the `P + 2` scalars.

None of it depends on the field, so a caller decomposing a time series over one point set builds
it once with [`prepare_spectral`](@ref), passes it as `helmholtz_decompose_spectral`'s `state`, and
releases its library plan with [`close!`](@ref).

**One per task.** Every buffer here is written through during a solve, so concurrent
decompositions need one each.
"""
struct ScatteredState{D,T,S,P,A,B,C,G,K,KD,W,V<:AbstractVector{T},Q,R}
    packing::S
    plan::P
    packed::A          # one fit's coefficients: (mode_size…, 1)
    buf::B             # one fit's samples: (M, 1), real for one component, complex for a pair
    velocity_hat::C    # (mode_size…, D)
    cg::G
    ks::K
    K::KD              # the derivative symbol, one array per axis shaped to broadcast along it
    weight::W          # axis-1 inner-product weights of a half spectrum; `nothing` otherwise
    measure::V         # each node's measure
    parts::Q           # the rotational, divergent and harmonic spectra, each like `velocity_hat`
    spectra::R         # (mode_size…, P + 2): χ̂, the rotation potential, the divergence
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

function prepare_spectral(solver::CartesianNUFFTSolver{L,D}, grid::_CartesianCloud{T,D};
                          execution = ComputationalBackends.AutoBackend()) where {L,D,T}
    lib = _nufft_library(solver.nufft)
    lib === nothing && throw(ArgumentError(_unavailable_message(solver)))
    return _scattered_state(_packing(lib), lib, solver, grid, T, Val(D), resolve_execution_backend(execution))
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
    plan = FlowTransformBindings.plan_nufft(lib, VT, coords, _plan_modes(packing, nk); tol = solver.tol,
        order = _mode_order(packing), period = periods, origin = zero(T),
        nthreads = _library_threads(backend))
    kdims = FlowTransformBindings.mode_size(plan)
    proto = first(coords)
    packed = similar(proto, Complex{T}, kdims..., 1)
    buf = similar(proto, VT, M, 1)
    velocity_hat = similar(proto, Complex{T}, kdims..., D)
    cg = CGBuffers(T, packed, buf, solver.maxiter, solver.atol > 0)
    ks = ntuple(Val(D)) do d
        T[T(2π) * k / periods[d] for k in FlowTransformBindings.mode_frequencies(plan, d)]
    end
    K = _derivative_symbol(velocity_hat, T, ks)
    weight = _inner_weights(packing, plan, T)
    measure = copyto!(similar(proto, T, M), FlowGeometries.Grids.measure(grid))
    parts = ntuple(_ -> similar(velocity_hat), Val(3))
    spectra = similar(proto, Complex{T}, kdims..., FlowGeometries.Operators.n_rotation_components(D) + 2)
    return ScatteredState{D,T,typeof(packing),typeof(plan),typeof(packed),typeof(buf),
                          typeof(velocity_hat),typeof(cg),typeof(ks),typeof(K),typeof(weight),typeof(measure),
                          typeof(parts),typeof(spectra)}(
        packing, plan, packed, buf, velocity_hat, cg, ks, K, weight, measure, parts, spectra)
end

allocate_result(st::ScatteredState{D,T}, grid::_CartesianCloud{T,D}) where {D,T} =
    _spectral_result(st.measure, T, (length(st.measure),), Val(D), Val(FlowGeometries.Operators.n_rotation_components(D)))

_decompose_spectral!(solver::CartesianNUFFTSolver{L,D}, ::FlowGeometries.Geometry.AbstractCartesianGeometry,
                     result::HelmholtzResult, U::AbstractMatrix{T}, grid::_CartesianCloud{T,D}, execution,
                     state::ScatteredState) where {L,D,T} =
    _decompose_scattered!(result, solver, U, state, grid)

function _decompose_scattered!(result::HelmholtzResult{D,P,T}, solver::CartesianNUFFTSolver{L,D},
                               U::AbstractMatrix{T}, st::ScatteredState{D,T}, grid) where {D,P,L,T}
    nk = solver.nk
    M = FlowTransformBindings.npoints(st.plan)
    size(U, 2) == D || throw(DimensionMismatch(
        "velocity has $(size(U, 2)) components for a $D-dimensional point set"))
    size(U, 1) == M || throw(DimensionMismatch(
        "velocity has $(size(U, 1)) rows against $M nodes in the plan"))
    _check_spectral_result(result, (M,), Val(D))

    velocity_hat = st.velocity_hat
    fit = SolverResult{T}(true, 0, zero(T))
    for q in 1:_nfits(st.packing, D)
        _load_samples!(st.buf, st.packing, U, q, D)
        s = inverse_nufft!(st.packed, st.buf, st.plan, st.cg, st.weight, st.measure; rtol = solver.rtol,
                           maxiter = solver.maxiter, atol = solver.atol)
        fit = _worst(fit, s)
        # Convergence says the samples are reproduced; the conditioning says whether they pick out
        # one coefficient vector or many.
        warn_conditioning(condition_estimate(st.cg.lanczos, 1), solver.rtol,
                          solver.condition_limit, M, prod(nk))
        _store_fit!(velocity_hat, st.packing, st.packed, q, D)
    end

    rot, div, harm = st.parts
    helmholtz_project_spectral!(rot, div, harm, velocity_hat, st.ks)
    _synthesize!(result.u_rot, rot, st)
    _synthesize!(result.u_div, div, st)
    _synthesize!(result.u_harm, harm, st)

    # χ, the `P` rotation-potential components and the divergence, as `P + 2` real scalars.
    _scalar_spectra!(st.spectra, velocity_hat, st.K, Val(D))
    _synthesize!((result.χ, result.rotation_potential..., result.divergence), st.spectra, st)

    result.harmonic_fraction = _measure_ratio(result.u_harm, U, st.measure)
    result.residual_fraction =
        _measure_ratio(Broadcast.broadcasted((u, a, b, h) -> u - a - b - h, U, result.u_rot, result.u_div, result.u_harm),
                       U, st.measure)
    result.χ_solve = fit
    result.rot_solve = ntuple(_ -> fit, Val(P))
    return result
end

# The `n` real fields whose spectra are the last axis of `spec`, at the plan's points, into the columns
# of `out`, a matrix or a tuple of vectors: one type 2 per real-transform field, one per pair of fields as
# `a + ib`.
function _synthesize!(out, spec::AbstractArray, st::ScatteredState)
    n = _ncolumns(out)
    for q in 1:_nfits(st.packing, n)
        _load_modes!(st.packed, st.packing, spec, q, n)
        FlowTransformBindings.nufft_type2!(st.buf, st.plan, st.packed)
        _store_samples!(out, st.packing, st.buf, q, n)
    end
    return out
end

_ncolumns(out::AbstractMatrix) = size(out, 2)
_ncolumns(out::Tuple) = length(out)
@inline _column(out::AbstractMatrix, a::Int) = view(out, :, a)
@inline _column(out::Tuple, a::Int) = out[a]

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
    copyto!(_column(out, q), view(buf, :, 1))
    return out
end
function _store_samples!(out, ::_ComplexPairs, buf, q::Int, D::Int)
    a, b = 2q - 1, 2q
    _column(out, a) .= real.(view(buf, :, 1))
    b <= D && (_column(out, b) .= imag.(view(buf, :, 1)))
    return out
end
