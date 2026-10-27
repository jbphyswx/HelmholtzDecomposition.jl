# Spherical Poisson solves and the spectral decomposition through FastSphericalHarmonics: forward
# SHT, divide by the eigenvalue `−ℓ(ℓ+1)/R²`, inverse SHT. The transform is defined on Fejér's
# first-rule node set `θᵢ = π(i−½)/N_θ`, `λⱼ = 2π(j−1)/(2N_θ−1)` (`FastSphericalHarmonics.sph_points`)
# alone, which a grid states by its sampling type and size. FastTransforms computes in Float64, so the
# transforms run on Float64 buffers and a field of another element type converts on the way in and out.
# The set spans the whole sphere; a regional patch is a boundary value problem, which the iterative
# solver takes.
module HelmholtzDecompositionFSHExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries: FlowGeometries as FG
using FlowTransformBindings: FlowTransformBindings as FTB
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FFTW: FFTW

# A grid this solver accepts covers the closed sphere, which has no boundary, so every boundary
# condition is accepted and none acts. The transform reads every node, so no cell may be masked out.
HD.supports_boundary(::HD.SphericalSpectralSolver, ::FG.Operators.AbstractBoundaryCondition) = true
HD.requires_full_domain(::HD.SphericalSpectralSolver) = true

"""
    _is_fsh_grid(grid) -> Bool

Whether `grid`'s nodes are the set this transform is defined on, from its type and size: a grid whose
sampling is Fejér's first rule with `N_λ = 2N_θ − 1`, of any float element type. Either latitude order
is accepted: the reflection `θ → π − θ` is an isometry of `S²` and commutes with `Δ`.
"""
_is_fsh_grid(grid) = false
_is_fsh_grid(grid::FG.Grids.StructuredGrid{<:AbstractFloat,<:FG.Geometry.AbstractSphericalGeometry,2}) =
    FG.Grids.sampling(grid) isa FG.SphericalSampling.AbstractFejer1Sampling && size(grid, 1) == 2 * size(grid, 2) - 1

HD.library_loaded(::HD.SphericalSpectralSolver) = true
HD.supports_sampling(::HD.SphericalSpectralSolver, grid) = _is_fsh_grid(grid)
HD.sampling_known(::HD.SphericalSpectralSolver, grid) = _is_fsh_grid(grid)

function HD._sampling_message(::HD.SphericalSpectralSolver, grid)
    sz = size(grid)
    nlat = length(sz) == 2 ? sz[2] : 0
    return "FastSphericalHarmonics is defined on Fejér's first-rule node set, θᵢ = π(i−½)/N_θ and " *
           "λⱼ = 2π(j−1)/(2N_θ−1), which a grid states by its sampling; this grid ($(join(sz, "×"))) does " *
           "not. Build it with `FlowGeometries.Connectivity.structured_grid(T, Fejer1Sampling(), $nlat)`, " *
           "load `NUFSHT` for an arbitrary node set, or use the iterative solver for a regional patch."
end

"""
    SphericalSpectralState

The transposed coefficient array and the per-mode eigenvalues, in Float64, built once.

`FastSphericalHarmonics` wants `(nlat, nlon)` where the grid is `(nlon, nlat)`, so a solve has to
transpose in and out, converting the grid's element type to Float64 and back. Allocating that array
inside `solve_poisson!` allocates it once per potential per field of a batch.
"""
struct SphericalSpectralState{A<:AbstractMatrix{Float64},E<:AbstractVector{Float64},C}
    coeffs::A
    inv_eig::E      # 1/λ_ℓ per ℓ, with ℓ = 0 zeroed
    cache::C        # this task's own FastTransforms plans
end

# FastTransforms builds its sphere plans with `fftw_plan_many_r2r` on the libfftw3 FFTW.jl loads, whose
# planner keeps one process-global table; two tasks planning at once abort in `malloc`. So every plan
# here is built under FFTW.jl's planner lock, `FFTW.set_num_threads(f, n)`, at the thread count of the
# backend the plan runs on. Execution needs no lock when each task owns its plans; two tasks executing
# one plan corrupt each other through its scratch, so the cache lives on the per-task state.

HD._backend_message(::HD.SphericalSpectralSolver, backend) =
    "SphericalSpectralSolver runs on the host: FastSphericalHarmonics wraps FastTransforms, which has " *
    "no device implementation, and cannot run under $(nameof(typeof(backend))). Plan this grid with a " *
    "host `execution`, or on the device with `CGSolver()`."

function HD.prepare_solver(solver::HD.SphericalSpectralSolver,
                           grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
                           ::FG.Operators.AbstractBoundaryCondition;
                           execution = CB.SerialBackend(), shared = nothing) where {T}
    HD._require_backend(solver, execution, grid)
    nlon, nlat = size(grid)
    R = Float64(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    coeffs = zeros(Float64, nlat, nlon)
    # Column `|m|` holds the degrees `|m| … |m| + nlat − 1`, so the stored degrees run to
    # `nlon ÷ 2 + nlat − 1`. ℓ = 0 is the constant the Laplacian annihilates, and stays zero.
    inv_eig = zeros(Float64, nlon ÷ 2 + nlat)
    @inbounds for ℓ in 1:(length(inv_eig) - 1)
        inv_eig[ℓ + 1] = -R^2 / (ℓ * (ℓ + 1))
    end
    # Both plans are built here, on this task, so a solve executes and plans nothing. The warm-up
    # runs one transform and one evaluation over `coeffs`, which is left zeroed for the first solve.
    cache = FSH.SphPlanCache{Float64}()
    FFTW.set_num_threads(HD._library_threads(execution)) do
        FTB.with_fasttransforms_threads() do
            FSH.sph_transform!(coeffs; cache = cache)
            FSH.sph_evaluate!(coeffs; cache = cache)
        end
    end
    fill!(coeffs, zero(Float64))
    return SphericalSpectralState(coeffs, inv_eig, cache)
end

function HD.solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
    solver::HD.SphericalSpectralSolver;
    boundary::FG.Operators.AbstractBoundaryCondition = FG.Operators.Neumann(),
    state::SphericalSpectralState = HD.prepare_solver(solver, grid, boundary),
    kwargs...,
) where {T<:AbstractFloat}
    HD.spectral_inverse!(Φ, RHS, grid, solver, state)
    return HD.SolverResult{T}(true, 1, zero(T))
end

function HD.spectral_inverse!(
    Φ::AbstractMatrix{T}, RHS::AbstractMatrix{T},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
    ::HD.SphericalSpectralSolver, state::SphericalSpectralState,
) where {T<:AbstractFloat}
    nlon, nlat = size(grid)

    C = state.coeffs
    @inbounds for j in 1:nlat, i in 1:nlon
        C[j, i] = RHS[i, j]
    end
    FTB.with_fasttransforms_threads(() -> FSH.sph_transform!(C; cache = state.cache))
    # Every stored mode, `ℓ = row + |m| − 1` in column `2|m| + (m ≥ 0)` (`FSH.sph_mode`); `ℓ = 0` is the
    # constant, defined up to which the solve is.
    @inbounds for col in 1:nlon, row in 1:nlat
        C[row, col] *= state.inv_eig[row + col ÷ 2]
    end
    FTB.with_fasttransforms_threads(() -> FSH.sph_evaluate!(C; cache = state.cache))
    @inbounds for j in 1:nlat, i in 1:nlon
        Φ[i, j] = C[j, i]
    end
    return Φ
end

# ---------------------------------------------------------------------------
# Vector Hodge decomposition on the Fejér first-rule grid
# ---------------------------------------------------------------------------
#
# `ð̄` lowers spin weight and, on a spin-1 tangent field, is the divergence; `ð` raises it and, on a
# real spin-0 field, is the gradient. So `u_div = ∇Δ⁻¹(∇·u)` is three coefficient-space steps with
# no stencil, and `u_rot = u − u_div` is exact: `S²` is simply connected, so a tangent field has no
# harmonic part.
#
# The radius cancels. Both `∇·` and `∇` carry one factor of `1/R` and `Δ` carries `1/R²`, so the
# split of the field is the same on a sphere of any radius; only the potentials carry `R`.

# FSH's grid runs colatitude ascending, so its latitudes descend. A grid built with ascending
# latitude is the same node set read the other way round, and its rows are reversed here.
@inline _fsh_row(j::Int, nlat::Int, flip::Bool) = flip ? nlat - j + 1 : j

# Column `2|m| + (m ≥ 0)` of a coefficient array holds one `m`, so an even column is `m < 0`.
# `spinsph_eth` and `spinsph_ethbar` negate those columns when they convert between the raw complex
# packing and the real one, and the projection below is stated in the real packing.
@inline _spin_phase(col::Int) = iseven(col) ? -1.0 : 1.0

"""
    _invert_and_project!(S) -> S

Turn the spin-0 coefficients of `∇·u` into those of the velocity potential `χ`, in place.

`Δ⁻¹` on spin 0 is `−1/(ℓ(ℓ+1))`, and `ℓ = 0` is the constant `Δ` annihilates. A column holds one
`|m|` and a row steps `ℓ` up from it, so `ℓ = row + |m| − 1` reaches every stored mode, including
those above `nlat − 1` the transform resolves at high `|m|`.

`V = ð(χ + iψ)` for real `χ`, `ψ`, so `Δ⁻¹ð̄V` is the complex potential and the velocity potential
is its real part. Realness of a spin-0 field is `c₋ₘ = conj(c₊ₘ)` in this storage, so averaging each
`±m` pair against its partner's conjugate is that projection — the one step here that is not
diagonal in `(ℓ, m)`, and the whole of the Hodge split.
"""
function _invert_and_project!(S::AbstractMatrix{ComplexF64})
    nrow = size(S, 1)
    @inbounds for row in 1:nrow
        l = row - 1                             # column 1 is m = 0
        S[row, 1] = iszero(l) ? zero(ComplexF64) : complex(-real(S[row, 1]) / (l * (l + 1)))
    end
    @inbounds for col in 2:2:size(S, 2)
        am = col ÷ 2                            # columns `2|m|` and `2|m|+1` hold ∓|m|
        αlo, αhi = _spin_phase(col), _spin_phase(col + 1)
        for row in 1:nrow
            l = row + am - 1                    # `am ≥ 1` here, so `ℓ ≥ 1`
            w = -1.0 / (l * (l + 1))
            a = αlo * w * S[row, col]
            b = αhi * w * S[row, col + 1]
            p = (a + conj(b)) / 2
            S[row, col] = αlo * p
            S[row, col + 1] = αhi * conj(p)
        end
    end
    return S
end

"""
    _potential_coefficients(S) -> (P, St)

From `S`, the spin-0 coefficients of `ð̄V` in FastSphericalHarmonics' packing: `St`, the same
coefficients in FastTransforms' own packing (`spinsph_eth`/`spinsph_ethbar` differ from it by the sign
of every `m < 0` column), and `P = Δ⁻¹St`, `−1/(ℓ(ℓ+1))` per mode with `ℓ = 0` zero.

FastTransforms' ð on spin 0 is `−(∂θ + i cscθ ∂φ)` (its `examples/spinweighted.jl`), so a tangent field
`V = u_θ + i u_φ = (∇χ + k̂×∇ψ)/R` on the unit-sphere operators is `V = −ð(χ + iψ)/R`, and `P` evaluates
to `−(χ + iψ)/R` and `St` to `−R(∇·u + i ζ)`.
"""
function _potential_coefficients(S::AbstractMatrix{ComplexF64})
    P = similar(S)
    St = similar(S)
    @inbounds for col in axes(S, 2)
        am = col ÷ 2
        α = _spin_phase(col)
        for row in axes(S, 1)
            l = row + am - 1
            s = α * S[row, col]
            St[row, col] = s
            P[row, col] = iszero(l) ? zero(ComplexF64) : -s / (l * (l + 1))
        end
    end
    return P, St
end

"""
    FSHSpectralState

The spin-weighted transforms' plans on one Fejér first-rule grid, for spin 1 and spin 0, and the spin-1
field the forward transform runs in place on, in Float64. `spinsph_transform!` and `spinsph_evaluate!`
key their plans on `(s, N)` and share the `spinsph2fourier` plan; all of them are built here, under
FFTW.jl's planner lock, so a decomposition plans nothing.
"""
struct FSHSpectralState{C,V<:AbstractMatrix{ComplexF64}}
    cache::C
    V::V
end

HD.close!(s::FSHSpectralState) = s

function HD.prepare_spectral(solver::HD.SphericalSpectralSolver,
                             grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2};
                             execution = CB.AutoBackend()) where {T}
    exec = HD.resolve_execution_backend(execution)
    HD._require_backend(solver, exec, grid)
    nlon, nlat = size(grid)
    V = zeros(ComplexF64, nlat, nlon)
    cache = FSH.SpinSphPlanCache{ComplexF64}()
    FFTW.set_num_threads(HD._library_threads(exec)) do
        FTB.with_fasttransforms_threads() do
            C = FSH.spinsph_transform!(V, 1; cache = cache)
            FSH.spinsph_evaluate!(C, 1; cache = cache)
            FSH.spinsph_evaluate!(zeros(ComplexF64, nlat, nlon), 0; cache = cache)
        end
    end
    return FSHSpectralState(cache, fill!(V, zero(ComplexF64)))
end

HD.allocate_result(::FSHSpectralState, grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2}) where {T} =
    HD._spectral_result(Array{T}(undef, 0), T, size(grid), Val(2), Val(1))

"""
    _decompose_spectral!(solver, geometry, result, U, grid, execution, state) -> result

Helmholtz decomposition of a tangent velocity field on the Fejér first-rule grid, by spin-weighted
quadrature through `state`'s plans, into `result`. `U` is component-last carrying `(u_east, u_north)`.
`S²` is simply connected, so `u_harm` is zero; `χ`, the streamfunction `ψ` (`u_rot = k̂ × ∇ψ`) and `∇·u`
come from the same coefficients.
"""
function HD._decompose_spectral!(
    ::HD.SphericalSpectralSolver,
    ::FG.Geometry.AbstractSphericalGeometry,
    result::HD.HelmholtzResult{2,1,T},
    U::AbstractArray{T,3},
    grid::FG.Grids.StructuredGrid{T,<:FG.Geometry.AbstractSphericalGeometry,2},
    execution, st::FSHSpectralState,
) where {T<:AbstractFloat}
    HD._check_spectral_result(result, size(grid), Val(2))
    nlon, nlat = size(grid)
    R = Float64(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))
    φ = FG.Grids.coordinates(grid, 2)
    flip = nlat > 1 && φ[1] < φ[end]

    # `θ̂` points south and `φ̂` east, and FSH indexes `(nlat, nlon)` where this package uses
    # `(nlon, nlat)`. The spin-1 field is carried as the complex combination `u_θ + i u_φ`: FSH's
    # `SVector{2,Float64}` spin-1 interface requires the `m = 0` coefficient to be real, which a
    # tangent field with a zonal mean — solid-body rotation — does not satisfy.
    V = st.V
    @inbounds for i in 1:nlon, j in 1:nlat
        V[_fsh_row(j, nlat, flip), i] = complex(-U[i, j, 2], U[i, j, 1])
    end

    cache = st.cache
    D, Φ, Δ = FTB.with_fasttransforms_threads() do
        C = FSH.spinsph_transform!(V, 1; cache = cache)   # spin-1 coefficients, in place over `V`
        S = FSH.spinsph_ethbar(C, 1)                      # spin-0 coefficients of ð̄V
        P, St = _potential_coefficients(S)
        _invert_and_project!(S)
        Dv = FSH.spinsph_evaluate!(FSH.spinsph_eth(S, 0), 1; cache = cache)   # ∇χ at the nodes
        return Dv, FSH.spinsph_evaluate!(P, 0; cache = cache), FSH.spinsph_evaluate!(St, 0; cache = cache)
    end

    u_div, u_rot, χ, δ = result.u_div, result.u_rot, result.χ, result.divergence
    ψ = result.rotation_potential[1]
    fill!(result.u_harm, zero(T))
    @inbounds for i in 1:nlon, j in 1:nlat
        row = _fsh_row(j, nlat, flip)
        d = D[row, i]
        u_div[i, j, 1] = imag(d)     # u_east = u_φ
        u_div[i, j, 2] = -real(d)    # u_north = −u_θ
        u_rot[i, j, 1] = U[i, j, 1] - u_div[i, j, 1]
        u_rot[i, j, 2] = U[i, j, 2] - u_div[i, j, 2]
        χ[i, j] = -R * real(Φ[row, i])
        ψ[i, j] = -R * imag(Φ[row, i])
        δ[i, j] = -real(Δ[row, i]) / R
    end
    ok = HD.SolverResult{T}(true, 1, zero(T))
    result.harmonic_fraction = zero(T)
    result.residual_fraction = HD._measure_ratio(Broadcast.broadcasted((u, a, b) -> u - a - b, U, u_rot, u_div), U, grid)
    result.χ_solve = ok
    result.rot_solve = (ok,)
    return result
end

end # module
