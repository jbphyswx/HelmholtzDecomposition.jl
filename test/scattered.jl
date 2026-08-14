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
end
