# Spherical spectral decomposition: the split and the potentials against closed forms on every
# spherical route, coverage of the sphere, threaded batches, and flagged fits.

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FlowTransformBindings: FlowTransformBindings as FTB
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using Random: Random

const SPH = FG.Geometry.SphericalGeometry(1.0)
const SB_AUTO = HD.SpectralBackends.AutoSpectralBackend()

nrm(A) = sqrt(sum(abs2, A))

# A longitude–latitude grid whose cells tile the sphere.
latlon(nlat::Int) = FG.Grids.StructuredGrid(SPH, FG.Axes.PeriodicAxis(0.0, 2π, 2nlat),
                                            range(-π / 2 + π / (2nlat), π / 2 - π / (2nlat); length = nlat))

# A sampling's own grid: its longitude a periodic axis, its recipe kept.
sampled(s, nlat) = FG.Connectivity.structured_grid(s, nlat; geometry = SPH)

# `χ = cosᵃφ sinᵇφ cos(mλ)`, which is `P_ℓ^m(sin φ) cos(mλ)` up to normalization for the `(m, a, b)`
# used here, with `∇χ` and `k̂ × ∇χ` on the unit sphere as (east, north) components.
function harmonic_fields(λ, φ, m, a, b)
    χ = [cos(p)^a * sin(p)^b * cos(m * l) for l in λ, p in φ]
    G = zeros(length(λ), length(φ), 2)
    R = similar(G)
    for (j, p) in enumerate(φ), (i, l) in enumerate(λ)
        c, s = cos(p), sin(p)
        dλ = -m * c^a * s^b * sin(m * l)
        dφ = (-a * c^(a - 1) * s^(b + 1) + (b == 0 ? 0.0 : b * c^(a + 1) * s^(b - 1))) * cos(m * l)
        G[i, j, 1], G[i, j, 2] = dλ / c, dφ
        R[i, j, 1], R[i, j, 2] = -dφ, dλ / c
    end
    return χ, G, R
end

Test.@testset "split and potentials against closed forms" begin
    # `∇χ` is divergent with velocity potential `χ` and divergence `Δχ = −ℓ(ℓ+1)χ`; `k̂ × ∇χ` is rotational
    # with streamfunction `χ`. The sphere is simply connected, so the harmonic part is zero.
    nlat = 16
    cc = sampled(FG.SphericalSampling.ClenshawCurtisSampling(), nlat)
    flipped = FG.Grids.StructuredGrid(SPH, FG.Grids.coordinates(cc, 1), reverse(FG.Grids.coordinates(cc, 2)))
    nu(lib = SB_AUTO) = HD.SphericalNUSHTSolver(; lmax = nlat - 1, tol = 1e-12, rtol = 1e-12, nufft = lib)
    quad(s) = sampled(s, nlat)
    routes = [
        ("FSH, Clenshaw–Curtis", cc, HD.SphericalSpectralSolver()),
        ("FSH, Clenshaw–Curtis, latitudes reversed", flipped, HD.SphericalSpectralSolver()),
        ("NUFSHT, Clenshaw–Curtis", cc, nu()),
        ("NUFSHT over FINUFFT, longitude–latitude", latlon(nlat), nu(FTB.FINUFFTBackend())),
        ("NUFSHT over NonuniformFFTs, longitude–latitude", latlon(nlat), nu(FTB.NonuniformFFTsBackend())),
        ("Auto, Gauss–Legendre", quad(FG.SphericalSampling.GaussLegendreSampling()), HD.AutoSolver()),
        ("Auto, Driscoll–Healy", quad(FG.SphericalSampling.DriscollHealySampling()), HD.AutoSolver()),
    ]
    for (label, grid, solver) in routes
        Test.@testset "$label" begin
            resolved = HD._resolve_spectral_solver(grid, solver)
            # The accuracy the route asks for: the NUFFT and least-squares tolerances of a NUFSHT fit;
            # FastSphericalHarmonics' quadrature is exact to round-off.
            ε = resolved isa HD.SphericalNUSHTSolver ? max(resolved.tol, resolved.rtol) : 1e-12
            λ, φ = FG.Grids.coordinates(grid, 1), FG.Grids.coordinates(grid, 2)
            for (m, a, b, l) in ((0, 0, 1, 1), (2, 2, 1, 3), (4, 4, 0, 4))
                χ, G, R = harmonic_fields(λ, φ, m, a, b)
                s = maximum(abs, χ)
                g = HD.helmholtz_decompose_spectral(G, grid; solver)
                Test.@test g.χ_solve.converged
                Test.@test nrm(g.u_rot) / nrm(G) < 100ε
                Test.@test nrm(g.u_div .- G) / nrm(G) < 100ε
                Test.@test maximum(abs, g.u_harm) == 0
                Test.@test maximum(abs, g.χ .- χ) <= 1000ε * s
                Test.@test maximum(abs, g.divergence .+ l * (l + 1) .* χ) <= 1e4 * ε * l * (l + 1) * s
                Test.@test maximum(abs, HD.streamfunction(g)) <= 1000ε * s
                r = HD.helmholtz_decompose_spectral(R, grid; solver)
                Test.@test nrm(r.u_div) / nrm(R) < 100ε
                Test.@test nrm(r.u_rot .- R) / nrm(R) < 100ε
                Test.@test maximum(abs, HD.streamfunction(r) .- χ) <= 1000ε * s
                Test.@test maximum(abs, r.χ) <= 1000ε * s
                Test.@test r.residual_fraction < 100ε
            end
        end
    end
end

Test.@testset "the NUFSHT route on GPUBackend(KernelAbstractions.CPU()) equals the serial one" begin
    # The plans built under the device backend and every buffer in their memory: a field with both parts
    # decomposes as on the host, to the fits' tolerance.
    nlat = 16
    grid = latlon(nlat)
    λ, φ = FG.Grids.coordinates(grid, 1), FG.Grids.coordinates(grid, 2)
    _, G, _ = harmonic_fields(λ, φ, 2, 2, 1)
    _, _, Rr = harmonic_fields(λ, φ, 4, 4, 0)
    U = G .+ Rr
    solver = HD.SphericalNUSHTSolver(; lmax = nlat - 1, tol = 1e-12, rtol = 1e-12)
    ε = max(solver.tol, solver.rtol)
    s = HD.helmholtz_decompose_spectral(U, grid; solver, execution = CB.SerialBackend())
    d = HD.helmholtz_decompose_spectral(U, grid; solver, execution = CB.GPUBackend(KA.CPU()))
    Test.@test s.χ_solve.converged && d.χ_solve.converged
    Test.@test nrm(d.u_rot .- s.u_rot) / nrm(U) < 100ε
    Test.@test nrm(d.u_div .- s.u_div) / nrm(U) < 100ε
    Test.@test maximum(abs, d.χ .- s.χ) <= 1000ε * maximum(abs, s.χ)
    Test.@test maximum(abs, HD.streamfunction(d) .- HD.streamfunction(s)) <= 1000ε * maximum(abs, HD.streamfunction(s))
    Test.@test maximum(abs, d.divergence .- s.divergence) <= 1e4 * ε * maximum(abs, s.divergence)
    Test.@test abs(d.residual_fraction - s.residual_fraction) < 100ε
end

Test.@testset "coverage of the sphere" begin
    # A grid covers the sphere when each pole lies within one and a half spacings of its two outermost
    # latitudes: a dual grid's `Δ`, nudged either way by a rounding, does; a ±74.5° band does not.
    nlat = 16
    λ = FG.Axes.PeriodicAxis(0.0, 2π, 2nlat)
    Δ = π / nlat
    nu = HD.SphericalNUSHTSolver()
    for nudge in (-4eps(), 0.0, 4eps())
        Test.@test HD.supports_sampling(nu, FG.Grids.StructuredGrid(SPH, λ, range(-π / 2 + Δ + nudge, π / 2 - Δ - nudge;
                                                                                  length = nlat - 1)))
    end
    let g = latlon(nlat)
        Test.@test HD.supports_sampling(nu, HD.dual_grid(g, 1, 2, FG.Operators.face_metrics(g, FG.Operators.Neumann())))
    end
    Test.@test !HD.supports_sampling(nu, FG.Grids.StructuredGrid(SPH, λ, range(-1.3, 1.3; length = nlat)))

    # On a stretched axis: cell centres 0.071 from each pole with outermost spacing 0.12, and their dual
    # 0.131 from it with spacing 0.15, cover the sphere in either latitude order; rows 0.02 apart that
    # stop 0.171 short of each pole, 8.5 of their spacings, do not.
    λs = FG.Axes.PeriodicAxis(0.0, 2π, 22)
    φc = [-1.50, -1.38, -1.20, -0.9, -0.5, 0.0, 0.5, 0.9, 1.20, 1.38, 1.50]
    for φs in (φc, reverse(φc))
        g = FG.Grids.StructuredGrid(SPH, λs, φs)
        Test.@test HD.supports_sampling(nu, g)
        Test.@test HD.supports_sampling(nu, HD.dual_grid(g, 1, 2, FG.Operators.face_metrics(g, FG.Operators.Neumann())))
    end
    φh = [-1.40, -1.38, -1.36, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.36, 1.38, 1.40]
    Test.@test !HD.supports_sampling(nu, FG.Grids.StructuredGrid(SPH, λs, φh))
end

Test.@testset "a threaded batch on a Clenshaw–Curtis sphere equals the serial one" begin
    # The staggered solve here is conjugate gradients preconditioned by FastTransforms, which a batch
    # reaches from worker tasks; pinned to one thread there it returns what the root task returns.
    nlat = 12
    cc = sampled(FG.SphericalSampling.ClenshawCurtisSampling(), nlat)
    plan = HD.plan_helmholtz(cc; execution = CB.SerialBackend())
    Test.@test plan.solver.spectral isa HD.SphericalSpectralSolver
    Random.seed!(31)
    fields = [randn(size(cc)..., 2) for _ in 1:4]
    bser = HD.helmholtz_decompose_batch(plan, fields; execution = CB.SerialBackend())
    bthr = HD.helmholtz_decompose_batch(plan, fields; execution = CB.ThreadedBackend())
    Test.@test bser.χ == bthr.χ
    # The dual solves are preconditioned by NUFSHT, whose NUFFT is reproducible to round-off only.
    Test.@test maximum(abs, bser.u_rot .- bthr.u_rot) <= 1e-12 * maximum(abs, bser.u_rot)
end

Test.@testset "fits past their tolerance or their nodes" begin
    grid = latlon(12)
    U = zeros(size(grid)..., 2)
    U[:, :, 1] .= cos.(FG.Grids.coordinates(grid, 2))'             # solid-body rotation
    # One iteration cannot reach `rtol`: the fit comes back as it stands, flagged.
    r = HD.helmholtz_decompose_spectral(U, grid; solver = HD.SphericalNUSHTSolver(; lmax = 8, rtol = 1e-14, maxiter = 1))
    Test.@test !r.χ_solve.converged && r.χ_solve.iterations == 1
    Test.@test all(isfinite, r.u_rot) && all(isfinite, r.u_div) && all(isfinite, r.χ) && isfinite(r.residual_fraction)
    # 288 nodes against 1680 coefficients: the fit reproduces the samples to its least-squares tolerance
    # and warns, and the split it reads off the undetermined coefficients does not hold; sized from the
    # grid it does.
    wide = HD.SphericalNUSHTSolver(; lmax = 40)
    over = Test.@test_logs (:warn,) match_mode = :any HD.helmholtz_decompose_spectral(U, grid; solver = wide)
    Test.@test over.residual_fraction < wide.rtol
    Test.@test nrm(over.u_div) / nrm(U) > 0.1
    Test.@test nrm(HD.helmholtz_decompose_spectral(U, grid; solver = HD.SphericalNUSHTSolver()).u_div) / nrm(U) < 1e-10
end
