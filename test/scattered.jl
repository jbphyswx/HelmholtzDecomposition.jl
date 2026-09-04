"""
Scattered Cartesian decomposition, on both non-uniform transforms.

A manufactured field with an exact split, sampled at random points:

    χ = sin x cos y      →  ∇χ  = ( cos x cos y, −sin x sin y )
    ψ = cos 2x sin y     →  ∇×ψ = ( cos 2x cos y,  2 sin 2x sin y )

Every assertion here is a case that returned a plausible, smooth, wrong field before — or, for the
period guard, a hard crash inside the transform library rather than an error.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using NearestNeighbors: NearestNeighbors
using Logging: Logging
using Random: Random

const FIExt = Base.get_extension(HD, :HelmholtzDecompositionFINUFFTExt)
const NUExt = Base.get_extension(HD, :HelmholtzDecompositionNonuniformFFTsExt)

# `areas` is supplied rather than triangulated: these solvers read only the coordinates, and the
# Voronoi construction would pull in DelaunayTriangulation for a quantity none of them uses.
function scattered_case(M::Int; periodic = true)
    T = Float64
    L = 2π
    Random.seed!(21)
    x = L .* rand(T, M)
    y = L .* rand(T, M)
    kw = periodic ? (; period = (L, L), periodic = (true, true)) : (;)
    grid = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{T}(), (x, y);
                                     areas = fill(L^2 / M, M), kw...)
    U = similar(x, M, 2)
    @. U[:, 1] = cos(x) * cos(y) + cos(2x) * cos(y)
    @. U[:, 2] = -sin(x) * sin(y) + 2 * sin(2x) * sin(y)
    exact_div = similar(U)
    @. exact_div[:, 1] = cos(x) * cos(y)
    @. exact_div[:, 2] = -sin(x) * sin(y)
    return grid, U, exact_div
end

# Same construction in `D` dimensions. `χ = sin x₁ ∏_{d>1} cos x_d` gives an exact gradient, and in
# 3-D a curl adds an exactly divergence-free part.
function scattered_nd(::Val{D}, M::Int) where {D}
    T = Float64
    L = 2π
    Random.seed!(23)
    coords = ntuple(_ -> L .* rand(T, M), Val(D))
    grid = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{T}(), coords;
                                     areas = fill(L^D / M, M),
                                     period = ntuple(_ -> L, Val(D)),
                                     periodic = ntuple(_ -> true, Val(D)))
    U = zeros(T, M, D)
    exact_div = zeros(T, M, D)
    x = coords[1]
    if D == 1
        @. exact_div[:, 1] = cos(x)
    elseif D == 2
        y = coords[2]
        @. exact_div[:, 1] = cos(x) * cos(y)
        @. exact_div[:, 2] = -sin(x) * sin(y)
        @. U[:, 1] += cos(2x) * cos(y)          # curl part, divergence-free
        @. U[:, 2] += 2 * sin(2x) * sin(y)
    else
        y, z = coords[2], coords[3]
        @. exact_div[:, 1] = cos(x) * cos(y) * cos(z)
        @. exact_div[:, 2] = -sin(x) * sin(y) * cos(z)
        @. exact_div[:, 3] = -sin(x) * cos(y) * sin(z)
        @. U[:, 1] += cos(2x) * cos(y)          # ∇×(0,0,ψ), ψ = cos 2x sin y
        @. U[:, 2] += 2 * sin(2x) * sin(y)
    end
    U .+= exact_div
    return grid, U, exact_div
end

Test.@testset "scattered Cartesian in 1-D, 2-D and 3-D" begin
    # `D = 1` and `D = 3` have an odd component count, which is the branch where the FINUFFT path's
    # `u + iv` packing carries its last component alone.
    cases = ((Val(1), 2000, (32,)), (Val(2), 4000, (16, 16)), (Val(3), 6000, (8, 8, 8)))
    for (v, M, nk) in cases
        D = length(nk)
        grid, U, exact_div = scattered_nd(v, M)
        geo = FG.Grids.grid_geometry(grid)
        scale = maximum(abs, U)
        for (name, s) in (("FINUFFT",
                           FIExt.CartesianNUFFTSolver(; nk = nk, tol = 1e-12, rtol = 1e-12,
                                                      maxiter = 300)),
                          ("NonuniformFFTs",
                           NUExt.CartesianNonuniformFFTSolver(; nk = nk, halfsupport = 8,
                                                              rtol = 1e-12, maxiter = 300)))
            Test.@testset "$D-D/$name" begin
                r = HD._decompose_spectral(s, geo, U, grid)
                Test.@test size(r.u_div) == (M, D)
                Test.@test maximum(abs, r.u_div .- exact_div) / scale < 1e-7
                Test.@test maximum(abs, (r.u_rot .+ r.u_div .+ r.u_harm) .- U) / scale < 1e-7
            end
        end
    end
end

Test.@testset "scattered Cartesian" begin
    M = 4000
    nk = (16, 16)
    grid, U, exact_div = scattered_case(M)
    geo = FG.Grids.grid_geometry(grid)

    solvers = (("FINUFFT", FIExt.CartesianNUFFTSolver(; nk = nk, tol = 1e-12, rtol = 1e-12,
                                                       maxiter = 200)),
               ("NonuniformFFTs", NUExt.CartesianNonuniformFFTSolver(; nk = nk, halfsupport = 8,
                                                                      rtol = 1e-12, maxiter = 200)))
    results = Dict{String,Any}()
    for (name, s) in solvers
        r = HD._decompose_spectral(s, geo, U, grid)
        results[name] = r
        scale = maximum(abs, U)
        Test.@testset "$name" begin
            # The divergent part is recovered, not merely something of the right size.
            Test.@test maximum(abs, r.u_div .- exact_div) / scale < 1e-8
            # And the three parts reconstruct the input.
            Test.@test maximum(abs, (r.u_rot .+ r.u_div .+ r.u_harm) .- U) / scale < 1e-8
        end
    end

    # Two independent transform libraries, one answer.
    Test.@test maximum(abs, results["FINUFFT"].u_div .- results["NonuniformFFTs"].u_div) /
          maximum(abs, U) < 1e-8

    Test.@testset "guards" begin
        s = FIExt.CartesianNUFFTSolver(; nk = nk, tol = 1e-12, rtol = 1e-12, maxiter = 200)

        # A grid with no period reports 0; dividing by it fed the transform `Inf` coordinates,
        # which segfaults in the spreader rather than raising.
        aperiodic, Ua, _ = scattered_case(M; periodic = false)
        Test.@test_throws ArgumentError HD._decompose_spectral(s, geo, Ua, aperiodic)

        # More modes than samples leaves the normal equations singular.
        toobig = FIExt.CartesianNUFFTSolver(; nk = (128, 128), tol = 1e-12, rtol = 1e-12,
                                            maxiter = 50)
        Test.@test_throws ArgumentError HD._decompose_spectral(toobig, geo, U, grid)

        # `prod(nk) ≤ M` is necessary and not sufficient: conditioning degrades well before the
        # normal equations go singular, so an unconverged fit is refused rather than returned.
        dense, Ud, _ = scattered_case(3000)
        tight = FIExt.CartesianNUFFTSolver(; nk = (48, 48), tol = 1e-12, rtol = 1e-12,
                                           maxiter = 400)
        Test.@test_throws ArgumentError HD._decompose_spectral(tight, geo, Ud, dense)
    end

    Test.@testset "a node set that cannot determine the modes is reported" begin
        # Samples confined to a band of the torus — a ship track, a satellite swath. `144` modes
        # from `2000` samples clears every counting test by a wide margin, the fit converges, and
        # the coefficients are still not determined: the field is unconstrained across the empty
        # part of the domain. The residual cannot see that. `κ(A)` can, and comes out of the
        # iteration's own scalars.
        T = Float64
        L = 2π
        Mb = 2000
        Random.seed!(5)
        xb = L .* rand(T, Mb)
        yb = 0.3π .* rand(T, Mb)
        banded = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{T}(), (xb, yb);
                                           areas = fill(L^2 / Mb, Mb),
                                           period = (L, L), periodic = (true, true))
        Ub = similar(xb, Mb, 2)
        @. Ub[:, 1] = cos(xb) * cos(yb)
        @. Ub[:, 2] = -sin(xb) * sin(yb)

        nk = (12, 12)
        sb = FIExt.CartesianNUFFTSolver(; nk = nk, tol = 1e-12, rtol = 1e-10, maxiter = 2000)
        Test.@test prod(nk) <= Mb                       # the counting test it passes
        Test.@test_logs (:warn,) match_mode = :any HD._decompose_spectral(sb, geo, Ub, banded)

        # `condition_limit` is the caller's: `Inf` says nothing at all.
        quiet = FIExt.CartesianNUFFTSolver(; nk = nk, tol = 1e-12, rtol = 1e-10,
                                           maxiter = 2000, condition_limit = Inf)
        Test.@test_logs min_level = Logging.Warn HD._decompose_spectral(quiet, geo, Ub, banded)

        # The same count of well-spread samples is quiet, so the report tracks the layout.
        uni, Uu, _ = scattered_case(Mb)
        su = FIExt.CartesianNUFFTSolver(; nk = nk, tol = 1e-12, rtol = 1e-10, maxiter = 2000)
        Test.@test_logs min_level = Logging.Warn HD._decompose_spectral(su, geo, Uu, uni)

        # With noise on such a point set the residual keeps falling long after the coefficients
        # have passed their best value, so `rtol` alone runs on and the answer travels. `atol`
        # stops at the noise level — the discrepancy principle. `Ub` is a pure gradient, so every
        # bit of rotational part the fit reports is error.
        Random.seed!(11)
        noise = 1e-2
        Un = Ub .+ noise .* randn(Mb, 2)
        δ = noise * sqrt(2 * Mb)
        common = (; nk = nk, tol = 1e-12, rtol = 1e-10, maxiter = 4000, condition_limit = Inf)
        loose = HD._decompose_spectral(
            FIExt.CartesianNUFFTSolver(; common..., atol = 0.0), geo, Un, banded)
        stopped = HD._decompose_spectral(
            FIExt.CartesianNUFFTSolver(; common..., atol = δ), geo, Un, banded)
        nrm(A) = sqrt(sum(abs2, A))
        Test.@test nrm(stopped.u_rot) < nrm(loose.u_rot) / 100
        Test.@test nrm(stopped.u_rot) / nrm(Ub) < 1.0
        # Both still reproduce the samples: the residual is what cannot tell them apart.
        for r in (loose, stopped)
            Test.@test nrm(r.u_rot .+ r.u_div .+ r.u_harm .- Un) / nrm(Un) < 1e-1
        end
    end
end
