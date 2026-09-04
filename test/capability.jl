"""
Capability routing: what each solver and backend will and will not accept.

Every case here is a request that used to be answered with a plausible, smooth, *wrong* field, or
with a silent drop to serial execution. They are regression tests in the strict sense — each one
fails on the code as it was, for a reason that no amount of reading the output would reveal.
"""

using Test: @testset, @test, @test_throws
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads
using Random: Random



# Extensions are not properties of the parent module; `Base.get_extension` is how they are reached.
const FFTWExt = Base.get_extension(HD, :HelmholtzDecompositionFFTWExt)
const FSHExt = Base.get_extension(HD, :HelmholtzDecompositionFSHExt)
const NUFSHTExt = Base.get_extension(HD, :HelmholtzDecompositionNUFSHTExt)

const CART = FG.Geometry.CartesianGeometry{Float64}()

@testset "Cartesian spectral solver" begin
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
    fft = FFTWExt.CartesianSpectralSolver()
    bc = HD.Neumann()

    @test HD.select_solver(fft, periodic, bc) === fft
    # Dividing by the symbol in an exponential basis inverts the PERIODIC Laplacian; on a bounded
    # grid that answer solves a different boundary value problem, to machine precision.
    @test_throws ArgumentError HD.select_solver(fft, bounded, bc)
    @test_throws ArgumentError HD.select_solver(fft, masked, bc)
    @test_throws ArgumentError HD.select_solver(fft, collected, bc)

    @test HD._resolve_auto_solver(periodic, bc) isa typeof(fft)
    # A bounded uniform grid gets the cosine/sine transform, which diagonalizes the *bounded*
    # Laplacian — not the iterative fallback, and not this periodic solver.
    @test HD._resolve_auto_solver(bounded, bc) isa FFTWExt.CartesianBoundedSolver
    # A `Vector` axis is not provably uniform, so no transform applies and CG is correct.
    @test HD._resolve_auto_solver(collected, bc) isa HD.CGSolver

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
    @test sqrt(sum(abs2, Φf .- Φc)) / sqrt(sum(abs2, Φc)) < 1e-12
end

@testset "bounded Cartesian: cosine and sine transforms" begin
    n = 24; L = 1.0
    xr = range(0.0, L; length = n)
    bounded = FG.Grids.StructuredGrid(CART, xr, xr)
    mixed = FG.Grids.StructuredGrid(CART, range(0.0, L - L/n; length = n), xr;
                                    topology = (true, false), period = L)
    for bc in (HD.Neumann(), HD.Dirichlet())
        chosen = HD.select_solver(HD.AutoSolver(), bounded, bc)
        # A bounded domain gets a direct transform, not the iterative fallback.
        @test !(chosen isa HD.CGSolver)

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
        @test sqrt(sum(abs2, Φt .- Φc)) / sqrt(sum(abs2, Φc)) < 1e-12
        r = zeros(n, n); HD.apply_laplacian!(r, Φt, bounded, c)
        @test sqrt(sum(abs2, r .- f)) / sqrt(sum(abs2, f)) < 1e-12

        # A channel — periodic in one direction, bounded in the other — is one plan with a
        # different transform kind per direction, not a case to refuse.
        mixed_solver = FFTWExt.CartesianBoundedSolver()
        cm = HD.laplacian_coefficients(mixed, bc)
        Random.seed!(6)
        fm = randn(n, n)
        cm.singular && HD.project_out_constant!(fm, mixed, cm)
        Φm = zeros(n, n)
        HD.solve_poisson!(Φm, fm, mixed, mixed_solver; boundary = bc,
                          state = HD.prepare_solver(mixed_solver, mixed, bc))
        rm = zeros(n, n); HD.apply_laplacian!(rm, Φm, mixed, cm)
        @test sqrt(sum(abs2, rm .- fm)) / sqrt(sum(abs2, fm)) < 1e-12
    end
end

@testset "spherical solvers" begin
    sph = FG.Geometry.SphericalGeometry(1.0)
    nlat = 12
    ax = FG.SphericalSampling.spherical_axes(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), nlat)
    cc = FG.Grids.StructuredGrid(sph, ax.λ, ax.φ)
    fsh = FSHExt.SphericalSpectralSolver()
    bc = HD.Neumann()

    @test HD.select_solver(fsh, cc, bc) === fsh
    # `N_λ = 2N_θ − 1` is necessary and nowhere near sufficient: the node POSITIONS decide.
    shaped_but_wrong = FG.Grids.StructuredGrid(sph, range(0, 2π - 2π/(2nlat-1); length = 2nlat-1),
                                               range(-1.2, 1.2; length = nlat))
    @test size(shaped_but_wrong) == size(cc)
    @test_throws ArgumentError HD.select_solver(fsh, shaped_but_wrong, bc)

    # ΔY₁₀ = −2/R² · Y₁₀ on the unit sphere.
    f = [sin(ax.φ[j]) for i in eachindex(ax.λ), j in eachindex(ax.φ)]
    Φ = zeros(size(cc))
    HD.solve_poisson!(Φ, -2.0 .* f, cc, fsh; boundary = bc)
    @test maximum(abs.(Φ .- f)) < 1e-12

    # The non-uniform transform accepts any covering node set and refuses a regional patch:
    # analysis integrates over S² and the inverse Laplacian there is nonlocal.
    nu = NUFSHTExt.SphericalNUSHTSolver(; lmax = 12, tol = 1e-10, rtol = 1e-12, maxiter = 500)
    λ = range(0, 2π - 2π/(2nlat); length = 2nlat)
    φ = range(-π/2 + π/(2nlat), π/2 - π/(2nlat); length = nlat)
    covering = FG.Grids.StructuredGrid(sph, λ, φ)
    @test HD.select_solver(nu, covering, bc) === nu
    @test_throws ArgumentError HD.select_solver(nu, FG.Grids.StructuredGrid(sph, λ, range(-1.3, 1.3; length = nlat)), bc)

    # Exact inversion off the Clenshaw–Curtis grid, where a single adjoint is a different operator.
    fN = [sin(φ[j]) for i in eachindex(λ), j in eachindex(φ)]
    ΦN = zeros(size(covering))
    HD.solve_poisson!(ΦN, -2.0 .* fN, covering, nu; boundary = bc)
    @test maximum(abs.(ΦN .- fN)) < 1e-9
end

@testset "backend honesty" begin
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
    @test HD._resolve_batch_backend(CB.AutoBackend(), fields[1:1]) isa CB.SerialBackend
    expected = Threads.nthreads() > 1 ? CB.ThreadedBackend : CB.SerialBackend
    @test HD._resolve_batch_backend(CB.AutoBackend(), fields) isa expected

    # A backend named explicitly is honoured or refused, never quietly downgraded to serial.
    @test_throws ArgumentError HD.helmholtz_decompose_batch(plan, fields;
                                                            backend = CB.MPIBackend())

    serial = HD.helmholtz_decompose_batch(plan, fields; backend = CB.SerialBackend())
    threaded = HD.helmholtz_decompose_batch(plan, fields; backend = CB.ThreadedBackend())
    # Threading changes scheduling, never arithmetic. The tasks share one plan and write disjoint
    # slices of one batch, so this also says the solver state they write through is per task.
    @test all(i -> serial[i].u_rot == threaded[i].u_rot, eachindex(fields))
    @test all(i -> serial[i].χ == threaded[i].χ, eachindex(fields))
    @test all(i -> serial[i].harmonic_fraction == threaded[i].harmonic_fraction, eachindex(fields))

    # The batch is contiguous: one array per output for the whole batch, batch axis last.
    @test size(serial.u_rot) == (size(grid)..., 2, length(fields))
    @test size(serial.χ) == (size(grid)..., length(fields))
end
