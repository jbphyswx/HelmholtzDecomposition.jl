"""
Capability routing: what each solver and backend will and will not accept.

Every case here is a request that used to be answered with a plausible, smooth, *wrong* field, or
with a silent drop to serial execution. They are regression tests in the strict sense — each one
fails on the code as it was, for a reason that no amount of reading the output would reveal.
"""

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

Test.@testset "Cartesian spectral solver" begin
    n = 16; L = 1.0
    uniform = range(0.0, L - L/n; length = n)
    periodic = FG.Grids.StructuredGrid(CART, uniform, uniform; topology = (true, true), period = (L, L))
    bounded  = FG.Grids.StructuredGrid(CART, uniform, uniform)
    masked   = let m = trues(n, n); m[5, 5] = false
        FG.Grids.StructuredGrid(CART, uniform, uniform; mask = m, topology = (true, true), period = (L, L))
    end
    # A `Vector` axis is not PROVABLY uniform — FlowGeometries answers that from the type, never
    # by scanning values — so the FFT is not selected for it. Pass ranges to keep the fast path.
    collected = FG.Grids.StructuredGrid(CART, collect(uniform), collect(uniform);
                                        topology = (true, true), period = (L, L))
    fft = HD.CartesianSpectralSolver()
    bc = HD.Neumann()

    Test.@test HD.select_solver(fft, periodic, bc) === fft
    # Dividing by the symbol in an exponential basis inverts the PERIODIC Laplacian; on a bounded
    # grid that answer solves a different boundary value problem, to machine precision.
    Test.@test_throws ArgumentError HD.select_solver(fft, bounded, bc)
    Test.@test_throws ArgumentError HD.select_solver(fft, masked, bc)
    Test.@test_throws ArgumentError HD.select_solver(fft, collected, bc)

    Test.@test HD._resolve_auto_solver(periodic, bc) isa typeof(fft)
    # A bounded uniform grid gets the cosine/sine transform, which diagonalizes the *bounded*
    # Laplacian — not the iterative fallback, and not this periodic solver.
    Test.@test HD._resolve_auto_solver(bounded, bc) isa HD.CartesianBoundedSolver
    # A `Vector` axis is not provably uniform, so no transform applies and CG is correct.
    Test.@test HD._resolve_auto_solver(collected, bc) isa HD.CGSolver

    # The transform must invert the SAME L the operators compose, so it agrees with the iterative
    # solver to round-off rather than to discretisation order.
    Random.seed!(9)
    f = randn(n, n)
    c = HD.laplacian_coefficients(periodic, bc)
    HD.project_out_constant!(f, periodic, c)
    Φf = zeros(n, n); HD.solve_poisson!(Φf, f, periodic, fft; boundary = bc)
    Φc = zeros(n, n); HD.solve_poisson!(Φc, f, periodic, HD.CGSolver(; rtol = 1e-14);
                                        boundary = bc, coefficients = c)
    HD.project_out_constant!(Φf, periodic, c); HD.project_out_constant!(Φc, periodic, c)
    Test.@test sqrt(sum(abs2, Φf .- Φc)) / sqrt(sum(abs2, Φc)) < 1e-12
end

Test.@testset "bounded Cartesian: cosine and sine transforms" begin
    n = 24; L = 1.0
    xr = range(0.0, L; length = n)
    bounded = FG.Grids.StructuredGrid(CART, xr, xr)
    mixed = FG.Grids.StructuredGrid(CART, range(0.0, L - L/n; length = n), xr;
                                    topology = (true, false), period = L)
    for bc in (HD.Neumann(), HD.Dirichlet())
        chosen = HD.select_solver(HD.AutoSolver(), bounded, bc)
        # A bounded domain gets a direct transform, not the iterative fallback.
        Test.@test !(chosen isa HD.CGSolver)

        c = HD.laplacian_coefficients(bounded, bc)
        Random.seed!(4)
        f = randn(n, n)
        c.singular && HD.project_out_constant!(f, bounded, c)
        Φt = zeros(n, n); HD.solve_poisson!(Φt, f, bounded, chosen; boundary = bc)
        Φc = zeros(n, n); HD.solve_poisson!(Φc, f, bounded, HD.CGSolver(; rtol = 1e-14);
                                            boundary = bc, coefficients = c)
        if c.singular
            HD.project_out_constant!(Φt, bounded, c)
            HD.project_out_constant!(Φc, bounded, c)
        end
        # The transform must invert the SAME discrete L the operators compose, so it agrees with
        # the iterative solver to round-off rather than to discretisation order.
        Test.@test sqrt(sum(abs2, Φt .- Φc)) / sqrt(sum(abs2, Φc)) < 1e-12
        r = zeros(n, n); HD.apply_laplacian!(r, Φt, bounded, c)
        Test.@test sqrt(sum(abs2, r .- f)) / sqrt(sum(abs2, f)) < 1e-12

        # A channel — periodic in one direction, bounded in the other — is one plan with a
        # different transform kind per direction, not a case to refuse.
        mixed_solver = HD.CartesianBoundedSolver()
        cm = HD.laplacian_coefficients(mixed, bc)
        Random.seed!(6)
        fm = randn(n, n)
        cm.singular && HD.project_out_constant!(fm, mixed, cm)
        Φm = zeros(n, n)
        HD.solve_poisson!(Φm, fm, mixed, mixed_solver; boundary = bc,
                          state = HD.prepare_solver(mixed_solver, mixed, bc))
        rm = zeros(n, n); HD.apply_laplacian!(rm, Φm, mixed, cm)
        Test.@test sqrt(sum(abs2, rm .- fm)) / sqrt(sum(abs2, fm)) < 1e-12
    end
end

Test.@testset "spherical solvers" begin
    sph = FG.Geometry.SphericalGeometry(1.0)
    nlat = 12
    ax = FG.SphericalSampling.spherical_axes(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), nlat)
    cc = FG.Grids.StructuredGrid(sph, ax.λ, ax.φ)
    fsh = HD.SphericalSpectralSolver()
    bc = HD.Neumann()

    Test.@test HD.select_solver(fsh, cc, bc) === fsh
    # Auto takes FastSphericalHarmonics on its own grid and NUFSHT on any other covering set.
    Test.@test HD._resolve_auto_solver(cc, bc) isa HD.SphericalSpectralSolver
    # `N_λ = 2N_θ − 1` is necessary and nowhere near sufficient: the node POSITIONS decide.
    shaped_but_wrong = FG.Grids.StructuredGrid(sph, range(0, 2π - 2π/(2nlat-1); length = 2nlat-1),
                                               range(-1.2, 1.2; length = nlat))
    Test.@test size(shaped_but_wrong) == size(cc)
    Test.@test_throws ArgumentError HD.select_solver(fsh, shaped_but_wrong, bc)

    # ΔY₁₀ = −2/R² · Y₁₀ on the unit sphere.
    f = [sin(ax.φ[j]) for i in eachindex(ax.λ), j in eachindex(ax.φ)]
    Φ = zeros(size(cc))
    HD.solve_poisson!(Φ, -2.0 .* f, cc, fsh; boundary = bc)
    Test.@test maximum(abs.(Φ .- f)) < 1e-12

    # The non-uniform transform accepts any covering node set and refuses a regional patch:
    # analysis integrates over S² and the inverse Laplacian there is nonlocal.
    nu = HD.SphericalNUSHTSolver(; lmax = 12, tol = 1e-10, rtol = 1e-12, maxiter = 500)
    λ = range(0, 2π - 2π/(2nlat); length = 2nlat)
    φ = range(-π/2 + π/(2nlat), π/2 - π/(2nlat); length = nlat)
    covering = FG.Grids.StructuredGrid(sph, λ, φ)
    Test.@test HD.select_solver(nu, covering, bc) === nu
    Test.@test HD._resolve_auto_solver(covering, bc) isa HD.SphericalNUSHTSolver
    Test.@test_throws ArgumentError HD.select_solver(nu, FG.Grids.StructuredGrid(sph, λ, range(-1.3, 1.3; length = nlat)), bc)

    # Exact inversion off the Clenshaw–Curtis grid, where a single adjoint is a different operator.
    fN = [sin(φ[j]) for i in eachindex(λ), j in eachindex(φ)]
    ΦN = zeros(size(covering))
    HD.solve_poisson!(ΦN, -2.0 .* fN, covering, nu; boundary = bc)
    Test.@test maximum(abs.(ΦN .- fN)) < 1e-9
end

Test.@testset "iterative solves converge in Float32 and refuse to return unconverged" begin
    n = 32
    xr = collect(range(0.0f0, 1.0f0; length = n))       # a `Vector` axis, so the solver is CG
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float32}(), xr, xr)
    Random.seed!(12)
    u = randn(Float32, n, n, 2)
    for mg in (true, false)
        plan = HD.plan_helmholtz(grid; boundary = HD.Neumann(), solver = HD.CGSolver(; multigrid = mg),
                                 backend = CB.SerialBackend())
        Test.@test plan.solver isa HD.CGSolver
        ws = HD.allocate_workspace(plan)
        res = HD.helmholtz_decompose!(HD.allocate_result(plan), u, plan, ws)
        Test.@test res.χ_solve.converged
        Test.@test all(s -> s.converged, res.rot_solve)
        Test.@test eltype(res.χ) == Float32
        # `D Gχ = D u` up to the true residual, which in Float32 is of order `eps·κ(L)`, with
        # `κ ≈ 8n²/π²` for this Laplacian.
        δ = zeros(Float32, n, n)
        HD.divergence!(δ, HD.face_divergent(ws), grid, plan.boundary, plan.metrics)
        κ = 8 * n^2 / π^2
        Test.@test sqrt(sum(abs2, δ .- res.divergence)) <= 10 * eps(Float32) * κ * sqrt(sum(abs2, res.divergence))
    end

    # A solve stopped by its iteration cap raises.
    capped = HD.plan_helmholtz(grid; boundary = HD.Neumann(),
                               solver = HD.CGSolver(; max_iter = 2, multigrid = false),
                               backend = CB.SerialBackend())
    Test.@test_throws ArgumentError HD.helmholtz_decompose!(HD.allocate_result(capped), u, capped)
end

Test.@testset "each closed component of the mask keeps its own constant" begin
    # `L`'s null space is one constant per component no Dirichlet face reaches: both basins of a
    # walled Neumann box, and a lake cut off by an inactive ring inside a Dirichlet box.
    n = 40
    xr = collect(range(0.0, 1.0; length = n))
    wall = trues(n, n)
    wall[n ÷ 2, :] .= false
    lake = trues(n, n)
    for j in 1:n, i in 1:n
        abs(hypot(xr[i] - 0.5, xr[j] - 0.5) - 0.25) < 0.04 && (lake[i, j] = false)
    end
    Random.seed!(13)
    κ = 8 * n^2 / π^2
    for (mask, bc, nclosed) in ((wall, HD.Neumann(), 2), (lake, HD.Dirichlet(), 1))
        grid = FG.Grids.StructuredGrid(CART, xr, xr; mask = mask)
        plan = HD.plan_helmholtz(grid; boundary = bc, solver = HD.CGSolver(),
                                 backend = CB.SerialBackend())
        c = plan.coefficients
        ns = c.nullspace
        Test.@test c.singular
        Test.@test length(ns.totals) == nclosed
        Test.@test !ns.full
        u = randn(n, n, 2) .* mask
        ws = HD.allocate_workspace(plan)
        res = HD.helmholtz_decompose!(HD.allocate_result(plan), u, plan, ws)
        Test.@test res.χ_solve.converged
        # χ is unique up to one constant per closed component, fixed by zero mean on each: the
        # weighted sum vanishes to the round-off of an n-term sum.
        for k in eachindex(ns.totals)
            cells = ns.perm[ns.offsets[k]:(ns.offsets[k + 1] - 1)]
            Test.@test abs(sum(res.χ[cells] .* c.measure[cells])) <=
                  length(cells) * eps() * sum(abs.(res.χ[cells]) .* c.measure[cells])
        end
        δ = zeros(n, n)
        HD.divergence!(δ, HD.face_divergent(ws), grid, plan.boundary, plan.metrics)
        Test.@test sqrt(sum(abs2, δ .- res.divergence)) <= 10 * eps() * κ * sqrt(sum(abs2, res.divergence))
    end
end

Test.@testset "backend honesty" begin
    n = 12
    xr = range(0.0, 1.0; length = n)
    grid = FG.Grids.StructuredGrid(CART, xr, xr)
    # The inner backend is named, so both calls below run identical arithmetic and the comparison
    # isolates the batch axis. `AutoBackend` resolves to `ThreadedBackend` at more than one thread,
    # which threads the loops *within* each field for the serial batch and leaves them serial for
    # the threaded one — two different reduction groupings, differing in the last bits.
    plan = HD.plan_helmholtz(grid; boundary = HD.Neumann(), backend = CB.SerialBackend())
    Random.seed!(2)
    fields = [randn(n, n, 2) for _ in 1:4]

    # Auto chooses on real capability: batch size, thread count, extension loaded.
    Test.@test HD._resolve_batch_backend(CB.AutoBackend(), fields[1:1]) isa CB.SerialBackend
    expected = Threads.nthreads() > 1 ? CB.ThreadedBackend : CB.SerialBackend
    Test.@test HD._resolve_batch_backend(CB.AutoBackend(), fields) isa expected

    # A backend named explicitly is honoured or refused, never quietly downgraded to serial.
    Test.@test_throws ArgumentError HD.helmholtz_decompose_batch(plan, fields;
                                                            backend = CB.MPIBackend())

    serial = HD.helmholtz_decompose_batch(plan, fields; backend = CB.SerialBackend())
    threaded = HD.helmholtz_decompose_batch(plan, fields; backend = CB.ThreadedBackend())
    # Threading changes scheduling, never arithmetic. The tasks share one plan and write disjoint
    # slices of one batch, so this also says the solver state they write through is per task.
    Test.@test all(i -> serial[i].u_rot == threaded[i].u_rot, eachindex(fields))
    Test.@test all(i -> serial[i].χ == threaded[i].χ, eachindex(fields))
    Test.@test all(i -> serial[i].harmonic_fraction == threaded[i].harmonic_fraction, eachindex(fields))

    # The batch is contiguous: one array per output for the whole batch, batch axis last.
    Test.@test size(serial.u_rot) == (size(grid)..., 2, length(fields))
    Test.@test size(serial.χ) == (size(grid)..., length(fields))
end
