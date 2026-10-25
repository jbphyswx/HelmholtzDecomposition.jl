# Solvers and backends: each transform inverts the discrete `L` the operators compose, the spherical
# transforms precondition conjugate gradients on it, a solve stopped short returns flagged, the null
# space is one constant per closed component, and threading a batch changes no arithmetic.

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads
using Random: Random

const CART = FG.Geometry.CartesianGeometry{Float64}()
const OPS = FG.Operators

Test.@testset "transforms invert L" begin
    # `L Φ = f` with `L` applied from the coefficients, for the periodic FFT, the cosine and sine
    # transforms of a bounded box under either condition, a channel, and DST-I on a bounded dual grid.
    n = 12
    xp = FG.Axes.PeriodicAxis(0.0, 1.0, n)
    xb = range(0.0, 1.0; length = n)
    box = FG.Grids.StructuredGrid(CART, xb, xb)
    dplan = HD.plan_helmholtz(box; solver = HD.CGSolver(), execution = CB.SerialBackend())
    fft, r2r = HD.CartesianSpectralSolver(), HD.CartesianBoundedSolver()
    cases = [
        ("periodic", FG.Grids.StructuredGrid(CART, xp, xp), OPS.Neumann(), nothing, fft),
        ("box, Neumann", box, OPS.Neumann(), nothing, r2r),
        ("box, Dirichlet", box, OPS.Dirichlet(), nothing, r2r),
        ("channel, Dirichlet", FG.Grids.StructuredGrid(CART, xp, xb), OPS.Dirichlet(), nothing, r2r),
        ("dual of a box", dplan.dual_grids[1], dplan.dual_boundaries[1], dplan.dual_coefficients[1], r2r),
    ]
    for (name, grid, bc, coef, s) in cases
        Test.@testset "$name" begin
            c = coef === nothing ? OPS.laplacian_coefficients(OPS.face_metrics(grid, bc)) : coef
            Random.seed!(4)
            f = randn(size(grid))
            OPS.project_out_constant!(f, c)
            Φ = zeros(size(grid))
            HD.solve_poisson!(Φ, f, grid, s; boundary = bc, state = HD.prepare_solver(s, grid, bc))
            LΦ = OPS.apply_laplacian!(zeros(size(grid)), Φ, c)
            Test.@test maximum(abs, LΦ .- f) <= 1e-11 * maximum(abs, f)
        end
    end
end

Test.@testset "spherical transforms" begin
    sph = FG.Geometry.SphericalGeometry(1.0)
    nlat = 24
    cc = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), nlat; geometry = sph)
    ll = FG.Grids.StructuredGrid(sph, FG.Axes.PeriodicAxis(0.0, 2π, 48),
                                 range(-π / 2 + π / 48, π / 2 - π / 48; length = 24))
    for (name, grid, solver) in (("Clenshaw–Curtis, FastSphericalHarmonics", cc, HD.SphericalSpectralSolver()),
                                 ("longitude–latitude, NUFSHT", ll,
                                  HD.SphericalNUSHTSolver(; tol = 1e-10, rtol = 1e-12)))
        Test.@testset "$name" begin
            φ = FG.Grids.coordinates(grid, 2)
            # `ΔY₁₀ = −2 Y₁₀` on the unit sphere, by the transform alone.
            f = repeat(sin.(φ)', size(grid, 1))
            st = HD.prepare_solver(solver, grid, OPS.Neumann())
            Φ = zeros(size(grid))
            HD.solve_poisson!(Φ, -2 .* f, grid, solver; boundary = OPS.Neumann(), state = st)
            Test.@test maximum(abs, Φ .- f) <= 1e-9
            solver isa HD.SphericalNUSHTSolver && HD.close!(st)

            # On the staggered path, under `AutoSolver` and under multigrid-preconditioned conjugate
            # gradients, the divergent part carries the field's divergence to the solver tolerance.
            # Under a device backend `AutoSolver` keeps multigrid.
            Test.@test HD.select_solver(HD.AutoSolver(), grid, OPS.Neumann();
                                        execution = CB.GPUBackend(nothing)).spectral === nothing
            Random.seed!(43)
            u = randn(size(grid)..., 2)
            for s in (HD.AutoSolver(), HD.CGSolver())
                plan = HD.plan_helmholtz(grid; solver = s, execution = CB.SerialBackend())
                ws = HD.allocate_workspace(plan)
                r = HD.helmholtz_decompose!(HD.allocate_result(plan), u, plan, ws)
                fmD = OPS.face_metrics(grid, OPS.Dirichlet())
                Dv = OPS.divergence!(zeros(size(grid)), HD.face_velocity(ws), fmD)
                Dg = OPS.divergence!(zeros(size(grid)), HD.face_divergent(ws), fmD)
                Test.@test r.χ_solve.converged
                Test.@test maximum(abs, Dg .- Dv) <= 1e-9 * maximum(abs, Dv)
                HD.close!(ws)
            end
        end
    end
end

# A V-cycle whose coarse levels are the operator rediscretized, smoothed along the directions that couple
# strongly, reduces the error by a factor that does not depend on the resolution, so conjugate gradients
# take as many iterations at twice the cells per direction: on a lat–lon sphere, on a box with a hole,
# whose rotation potential holds zero on the corners inside the hole, and on a stretched curvilinear mesh.
Test.@testset "multigrid iterations do not grow with resolution" begin
    sph = FG.Geometry.SphericalGeometry(1.0)
    latlon(n) = FG.Grids.StructuredGrid(sph, FG.Axes.PeriodicAxis(0.0, 2π, 2n),
                                        range(-π / 2 + π / (2n), π / 2 - π / (2n); length = n))
    function holed(n)
        x = collect(range(0.0, 1.0; length = n))
        m = trues(n, n)
        m[(3n ÷ 8):(5n ÷ 8), (3n ÷ 8):(5n ÷ 8)] .= false
        return FG.Grids.StructuredGrid(CART, x, x, m)
    end
    # A curvilinear mesh stretched along both directions, its cells' aspect ratio between about 1/3 and 3, so
    # the direction of strong coupling changes across it.
    function stretched(n)
        s(t) = t + 0.15 * sin(π * t)
        xv = [s(i / n) + 0.1 * (j / n) for i in 0:n, j in 0:n]
        yv = [0.8 * s(j / n) + 0.06 * sin(2π * i / n) for i in 0:n, j in 0:n]
        mid(q) = [(q[i, j] + q[i + 1, j] + q[i, j + 1] + q[i + 1, j + 1]) / 4 for i in 1:n, j in 1:n]
        return FG.Grids.CurvilinearGrid(CART, mid(xv), mid(yv), trues(n, n); corners = (xv, yv))
    end
    for (name, grids) in (("lat–lon sphere", (latlon(32), latlon(64))), ("box with a hole", (holed(64), holed(128))),
                          ("stretched curvilinear mesh", (stretched(64), stretched(128))))
        Test.@testset "$name" begin
            its = map(grids) do g
                plan = HD.plan_helmholtz(g; solver = HD.CGSolver(), execution = CB.SerialBackend())
                Random.seed!(3)
                r = HD.helmholtz_decompose!(HD.allocate_result(plan), randn(size(g)..., 2), plan)
                Test.@test r.χ_solve.converged && all(s -> s.converged, r.rot_solve)
                (r.χ_solve.iterations, maximum(s -> s.iterations, r.rot_solve))
            end
            Test.@test its[2][1] <= its[1][1] + 2
            Test.@test its[2][2] <= its[1][2] + 2
        end
    end
end

Test.@testset "Float32 converges; a capped solve returns flagged" begin
    n = 16
    xr = collect(range(0.0f0, 1.0f0; length = n))
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float32}(), xr, xr)
    Random.seed!(12)
    u = randn(Float32, n, n, 2)
    for mg in (true, false)
        plan = HD.plan_helmholtz(grid; solver = HD.CGSolver(; multigrid = mg), execution = CB.SerialBackend())
        ws = HD.allocate_workspace(plan)
        res = HD.helmholtz_decompose!(HD.allocate_result(plan), u, plan, ws)
        Test.@test res.χ_solve.converged && all(s -> s.converged, res.rot_solve)
        # `D u_div = D u` to the true residual, of order `eps·κ(L)` in Float32, `κ ≈ 8n²/π²`.
        δ = OPS.divergence!(zeros(Float32, n, n), HD.face_divergent(ws), OPS.face_metrics(grid, OPS.Dirichlet()))
        Test.@test sqrt(sum(abs2, δ .- res.divergence)) <= 10 * eps(Float32) * (8 * n^2 / π^2) * sqrt(sum(abs2, res.divergence))
    end
    capped = HD.plan_helmholtz(grid; solver = HD.CGSolver(; max_iter = 2, multigrid = false),
                               execution = CB.SerialBackend())
    rc = HD.helmholtz_decompose!(HD.allocate_result(capped), u, capped)
    Test.@test !rc.χ_solve.converged && rc.χ_solve.iterations == 2
    Test.@test !all(s -> s.converged, rc.rot_solve)
    Test.@test all(isfinite, rc.u_rot) && all(isfinite, rc.u_div) && all(isfinite, rc.χ)
    Test.@test isfinite(rc.harmonic_fraction) && isfinite(rc.residual_fraction)
end

Test.@testset "one constant per closed component" begin
    # Both basins of a walled Neumann box, and a lake cut off inside a Dirichlet box by an inactive ring,
    # are closed: `χ` has zero measure-weighted mean on each, and `D u_div = D u` holds.
    n = 32
    xr = collect(range(0.0, 1.0; length = n))
    wall = trues(n, n); wall[n ÷ 2, :] .= false
    ring = [abs(hypot(x - 0.5, y - 0.5) - 0.25) < 0.05 for x in xr, y in xr]
    lake = .!ring
    inside = [hypot(x - 0.5, y - 0.5) < 0.2 for x in xr, y in xr]
    Random.seed!(13)
    for (mask, bc, parts) in ((wall, OPS.Neumann(), ((1:(n ÷ 2 - 1), :), ((n ÷ 2 + 1):n, :))),
                              (lake, OPS.Dirichlet(), (inside,)))
        grid = FG.Grids.StructuredGrid(CART, xr, xr; mask = mask)
        plan = HD.plan_helmholtz(grid; boundary = bc, solver = HD.CGSolver(), execution = CB.SerialBackend())
        ws = HD.allocate_workspace(plan)
        res = HD.helmholtz_decompose!(HD.allocate_result(plan), randn(n, n, 2) .* mask, plan, ws)
        Test.@test res.χ_solve.converged
        V = [FG.Grids.measure(grid, Tuple(I)...) for I in CartesianIndices((n, n))]
        for part in parts
            sel = part isa Tuple ? part : (part,)
            m = sum((res.χ .* V)[sel...] .* mask[sel...])
            Test.@test abs(m) <= 1e3 * eps() * sum(abs.(res.χ .* V)[sel...])
        end
        δ = OPS.divergence!(zeros(n, n), HD.face_divergent(ws), OPS.face_metrics(grid, OPS.Dirichlet()))
        Test.@test maximum(abs, (δ .- res.divergence)[mask]) <= 1e-9 * maximum(abs, res.divergence)
    end
end

Test.@testset "a threaded batch equals the serial one" begin
    # Both batches run their fields with a serial inner backend, so the comparison isolates the batch
    # axis; the tasks share one plan and each writes its own workspace and slice.
    n = 10
    grid = FG.Grids.StructuredGrid(CART, range(0.0, 1.0; length = n), range(0.0, 1.0; length = n))
    plan = HD.plan_helmholtz(grid; execution = CB.SerialBackend())
    Random.seed!(2)
    fields = [randn(n, n, 2) for _ in 1:4]
    serial = HD.helmholtz_decompose_batch(plan, fields; execution = CB.SerialBackend())
    threaded = HD.helmholtz_decompose_batch(plan, fields; execution = CB.ThreadedBackend())
    Test.@test serial.u_rot == threaded.u_rot && serial.χ == threaded.χ
    Test.@test serial.harmonic_fraction == threaded.harmonic_fraction

    # Inside one field, `AutoBackend`'s loop-sized threading changes only the grouping of reductions.
    # At one index per thread it threads every loop.
    sized = HD.plan_helmholtz(grid; execution = HD.LoopSizedThreads(1))
    rs = HD.helmholtz_decompose!(HD.allocate_result(sized), fields[1], sized)
    Test.@test maximum(abs, rs.u_rot .- serial.u_rot[:, :, :, 1]) <= 1e-12 * maximum(abs, serial.u_rot)
end
