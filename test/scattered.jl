"""
Scattered Cartesian decomposition on both NUFFT libraries: the split and the potentials at the points
against closed forms, flagged fits, conditioning reported by the layout of the points, and the
discrepancy principle.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FlowTransformBindings: FlowTransformBindings as FTB
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using Logging: Logging
using Random: Random

const LIBS = (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())

# `M` random points on the `D`-torus of side `2π`, each standing for an equal share of its volume.
function cloud(D, M; seed = 21, band = 2π)
    Random.seed!(seed)
    x = ntuple(d -> (d == 2 ? band : 2π) .* rand(M), D)
    grid = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), x, fill((2π)^D / M, M);
                                     period = ntuple(_ -> 2π, D), topology = ntuple(_ -> FG.Grids.Periodic(), D))
    return grid, x
end

# `u = ∇χ + k̂ × ∇R` with `χ = sin x₁ ∏ cos x_d` and, from two dimensions, `R = −cos 2x sin y` in the
# `(x, y)` plane: velocity, `∇χ`, `χ`, `∇·u = −D χ`, and `R`.
function field(x)
    D = length(x)
    χ = sin.(x[1])
    for d in 2:D
        χ = χ .* cos.(x[d])
    end
    G = zeros(length(χ), D)
    for d in 1:D
        g = d == 1 ? cos.(x[1]) : sin.(x[1])
        for e in 2:D
            g = g .* (e == d ? -sin.(x[e]) : cos.(x[e]))
        end
        G[:, d] .= g
    end
    U = copy(G)
    R = nothing
    if D >= 2
        R = -cos.(2 .* x[1]) .* sin.(x[2])
        U[:, 1] .+= cos.(2 .* x[1]) .* cos.(x[2])
        U[:, 2] .+= 2 .* sin.(2 .* x[1]) .* sin.(x[2])
    end
    return U, G, χ, -D .* χ, R
end

Test.@testset "split and potentials against closed forms" begin
    # Even, odd and mixed mode counts; an odd component count carries its last component alone in the
    # complex-pair packing; `AutoSolver` sizes the mode count from the points.
    cases = ((1, 300, ((16,), (15,))), (2, 1200, ((12, 12), (11, 12))), (3, 2000, ((6, 6, 6),)))
    for (D, M, nks) in cases
        grid, x = cloud(D, M)
        U, G, χ, δ, R = field(x)
        solvers = vcat([HD.CartesianNUFFTSolver(lib; nk, tol = 1e-12, rtol = 1e-12, maxiter = 400) for lib in LIBS, nk in nks]...,
                       HD.AutoSolver())
        for s in solvers
            Test.@testset "$D-D $(s isa HD.AutoSolver ? "Auto" : "$(nameof(typeof(s.nufft))) $(s.nk)")" begin
                r = HD.helmholtz_decompose_spectral(U, grid; solver = s)
                Test.@test r.χ_solve.converged
                Test.@test maximum(abs, r.u_div .- G) < 1e-7
                Test.@test maximum(abs, r.χ .- χ) < 1e-7
                Test.@test maximum(abs, r.divergence .- δ) < 1e-6
                D == 2 && Test.@test maximum(abs, HD.streamfunction(r) .- R) < 1e-7
                Test.@test r.residual_fraction < 1e-8
                Test.@test r.harmonic_fraction < 1e-8
            end
        end
    end

    # At scattered points `e^{8ix}` and `e^{−8ix}` are different functions, and `nk = 16` holds both: a
    # gradient along a Nyquist wavevector and a rotational Nyquist mode across it split exactly.
    grid, (x, y) = cloud(2, 1200; seed = 29)
    gd = hcat(cos.(8 .* x .+ 3 .* y), (3 / 8) .* cos.(8 .* x .+ 3 .* y))    # ∇ sin(8x + 3y) / 8
    gr = hcat(-sin.(8 .* y), zeros(length(x)))                              # k̂ × ∇ cos(8y) / 8
    for lib in LIBS
        r = HD.helmholtz_decompose_spectral(gd .+ gr, grid;
                                            solver = HD.CartesianNUFFTSolver(lib; nk = (16, 16), tol = 1e-12, rtol = 1e-12, maxiter = 400))
        Test.@test maximum(abs, r.u_div .- gd) < 1e-7
        Test.@test maximum(abs, r.u_rot .+ r.u_harm .- gr) < 1e-7
    end
end

Test.@testset "a held state gives the one-shot result" begin
    grid, x = cloud(2, 1200)
    U = first(field(x))
    s = HD.CartesianNUFFTSolver(FTB.NonuniformFFTsBackend(); nk = (12, 12), tol = 1e-12, rtol = 1e-12, maxiter = 400)
    st = HD.prepare_spectral(s, grid)
    for V in (U, 2 .* U)
        held = HD.helmholtz_decompose_spectral(V, grid; solver = s, state = st)
        once = HD.helmholtz_decompose_spectral(V, grid; solver = s)
        Test.@test isapprox(held.χ, once.χ; rtol = 1e-12) && isapprox(held.u_rot, once.u_rot; rtol = 1e-12)
    end
    HD.close!(st)
end

Test.@testset "a capped fit returns flagged" begin
    grid, x = cloud(2, 1200)
    U = first(field(x))
    r = HD.helmholtz_decompose_spectral(U, grid; solver = HD.CartesianNUFFTSolver(FTB.FINUFFTBackend(); nk = (30, 30),
                                                                                    tol = 1e-12, rtol = 1e-12, maxiter = 40))
    Test.@test !r.χ_solve.converged && r.χ_solve.iterations == 40
    Test.@test all(isfinite, r.u_rot) && all(isfinite, r.u_div) && all(isfinite, r.χ) && isfinite(r.residual_fraction)
end

Test.@testset "points confined to a band" begin
    # 144 modes from 1200 samples confined to a band of the torus: the fit converges and the
    # coefficients are not determined across the empty part. The conditioning reports it, and the same
    # count of well-spread samples does not; `condition_limit = Inf` silences it.
    banded, (xb, yb) = cloud(2, 1200; seed = 5, band = 0.3π)
    spread, xs = cloud(2, 1200; seed = 5)
    grad(x, y) = hcat(cos.(x) .* cos.(y), -sin.(x) .* sin.(y))
    fit(; kw...) = HD.CartesianNUFFTSolver(FTB.FINUFFTBackend(); nk = (12, 12), tol = 1e-12, rtol = 1e-10, maxiter = 4000, kw...)
    Ub = grad(xb, yb)
    Test.@test_logs (:warn,) match_mode = :any HD.helmholtz_decompose_spectral(Ub, banded; solver = fit())
    Test.@test_logs min_level = Logging.Warn HD.helmholtz_decompose_spectral(Ub, banded; solver = fit(condition_limit = Inf))
    Test.@test_logs min_level = Logging.Warn HD.helmholtz_decompose_spectral(grad(xs...), spread; solver = fit())

    # With noise of norm `δ` the residual keeps falling past the coefficients' best value; `atol = δ`
    # stops once the samples are matched to `δ`, the two components forming one complex column. `Ub` is
    # a gradient, so every rotational part reported is error.
    Random.seed!(11)
    noise = 1e-2
    Un = Ub .+ noise .* randn(size(Ub))
    δ = noise * sqrt(length(Un))
    loose = HD.helmholtz_decompose_spectral(Un, banded; solver = fit(condition_limit = Inf))
    stopped = HD.helmholtz_decompose_spectral(Un, banded; solver = fit(condition_limit = Inf, atol = δ))
    nrm(A) = sqrt(sum(abs2, A))
    Test.@test nrm(stopped.u_rot) < nrm(loose.u_rot) / 100
    Test.@test stopped.residual_fraction <= (1 + 1e-6) * δ / nrm(Un)
end
