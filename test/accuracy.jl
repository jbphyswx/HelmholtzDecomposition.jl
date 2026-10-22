"""
The spectral path against closed forms and at the Nyquist slots, and the harmonic part about a hole.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FlowTransformBindings: FlowTransformBindings as FTB
using FFTW: FFTW
using Random: Random

include(joinpath(@__DIR__, "..", "reference_flows", "reference_flows.jl"))
using .ReferenceFlows: ReferenceFlows

const CARTA = FG.Geometry.CartesianGeometry{Float64}()
nrm(a) = sqrt(sum(abs2, a))

Test.@testset "spectral path against closed forms" begin
    n = 16
    xr = range(0.0, 1.0 - 1 / n; length = n)
    grid = FG.Grids.StructuredGrid(CARTA, xr, xr; topology = (true, true), period = (1.0, 1.0))
    k = 2π
    f = [sin(k * x) * sin(k * y) for x in xr, y in xr]
    fx = [k * cos(k * x) * sin(k * y) for x in xr, y in xr]
    fy = [k * sin(k * x) * cos(k * y) for x in xr, y in xr]
    # `∇f` has potential `f` and divergence `−2k² f`; `k̂ × ∇f` has streamfunction `f`.
    g = HD.helmholtz_decompose_spectral(cat(fx, fy; dims = 3), grid)
    Test.@test nrm(g.u_rot) / nrm(f) < 1e-12 && g.harmonic_fraction < 1e-12
    Test.@test maximum(abs, g.χ .- f) < 1e-12
    Test.@test maximum(abs, g.divergence .+ 2k^2 .* f) < 1e-10
    r = HD.helmholtz_decompose_spectral(cat(-fy, fx; dims = 3), grid)
    Test.@test nrm(r.u_div) / nrm(f) < 1e-12
    Test.@test maximum(abs, HD.streamfunction(r) .- f) < 1e-12

    # The ABC flow is Beltrami, `∇ × u = u`, so its divergence-free vector potential is `A = u`.
    m = 8
    x3 = range(0.0, 2π - 2π / m; length = m)
    g3 = FG.Grids.StructuredGrid(CARTA, x3, x3, x3; topology = (true, true, true), period = (2π, 2π, 2π))
    U = zeros(m, m, m, 3)
    for (k3, z) in enumerate(x3), (j, y) in enumerate(x3), (i, x) in enumerate(x3)
        U[i, j, k3, :] .= (sin(z) + cos(y), sin(x) + cos(z), sin(y) + cos(x))
    end
    a = HD.helmholtz_decompose_spectral(U, g3)
    Test.@test nrm(a.u_div) / nrm(U) < 1e-12
    A = HD.vector_potential(a)
    Test.@test all(c -> maximum(abs, A[c] .- U[:, :, :, c]) < 1e-12, 1:3)
end

# White noise fills every Nyquist slot. The first derivative of a real field is zero at an even axis's
# Nyquist slot (Trefethen, *Spectral Methods in MATLAB*, ch. 3); under that derivative `u_rot` is
# divergence-free, `u_div` curl-free and the two orthogonal, checked by full-length FFTs, and the
# harmonic part is the mean alone.
Test.@testset "spectral path at the Nyquist slots" begin
    rng = Random.Xoshiro(3)
    for dims in ((8, 6), (7, 5), (6, 4, 5))
        N = length(dims)
        Ls = ntuple(d -> 1.0 + d / 4, N)
        xs = ntuple(d -> range(0.0, Ls[d] - Ls[d] / dims[d]; length = dims[d]), N)
        grid = FG.Grids.StructuredGrid(CARTA, xs...; topology = ntuple(_ -> true, N), period = Ls)
        U = randn(rng, dims..., N)
        res = HD.helmholtz_decompose_spectral(U, grid)
        K = FTB.wavenumber_arrays(zeros(dims), Float64, FTB.wavenumber_axes(Ls, dims, Val(false)); derivative = true)
        comp(A, c) = selectdim(A, N + 1, c)
        ûr = [FFTW.fft(comp(res.u_rot, c)) for c in 1:N]
        ûd = [FFTW.fft(comp(res.u_div, c)) for c in 1:N]
        scale = maximum(abs, FFTW.fft(comp(U, 1))) * maximum(k -> maximum(abs, k), K)
        Test.@test maximum(abs, sum(K[c] .* ûr[c] for c in 1:N)) < 1e-12 * scale
        Test.@test all(((a, b),) -> maximum(abs, K[a] .* ûd[b] .- K[b] .* ûd[a]) < 1e-12 * scale,
                       [(a, b) for a in 1:N for b in (a + 1):N])
        Test.@test abs(sum(res.u_rot .* res.u_div)) < 1e-12 * sum(abs2, U)
        Test.@test all(c -> all(≈(sum(comp(U, c)) / length(comp(U, c))), comp(res.u_harm, c)), 1:N)
    end
end

Test.@testset "circulation about a hole is harmonic" begin
    # A box with a masked hole has one loop. The face-space harmonic part of a vortex about the hole is
    # divergence-free on every cell, curl-free on every corner of the complex, carries no flux through
    # the edges, and is nonzero.
    n = 21
    xs = range(-1.0, 1.0; length = n)
    mask = [x^2 + y^2 > 0.3^2 for x in xs, y in xs]
    grid = FG.Grids.StructuredGrid(CARTA, xs, xs; mask = mask)
    u, v = ReferenceFlows.harmonic_vortex(grid; Γ = 1.0)
    U = cat(u, v; dims = 3) .* mask
    plan = HD.plan_helmholtz(grid; solver = HD.CGSolver())
    ws = HD.allocate_workspace(plan)
    res = HD.helmholtz_decompose!(HD.allocate_result(plan), U, plan, ws)
    h = HD.face_harmonic(ws)
    Dh = FG.Operators.divergence!(zeros(n, n), h, FG.Operators.face_metrics(grid, FG.Operators.Dirichlet()))
    Wh = FG.Operators.curl!(FG.Operators.allocate_corners(Float64, grid), h, plan.metrics)
    scale = maximum(abs, U)
    Test.@test maximum(abs, Dh[mask]) <= 1e-9 * scale
    Test.@test maximum(abs, Wh[1]) <= 1e-9 * scale
    Test.@test all(iszero, h[1][[1, n + 1], :]) && all(iszero, h[2][:, [1, n + 1]])
    Test.@test res.harmonic_fraction > 0.1
end
