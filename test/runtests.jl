using HelmholtzDecomposition: HelmholtzDecomposition as HD
using Test: Test, @testset, @test, @test_throws
using Aqua: Aqua
using Statistics: Statistics
using Random: Random
using LinearAlgebra: LinearAlgebra
using FFTW: FFTW
using OhMyThreads: OhMyThreads
using Distributed: Distributed
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT

# Component-last helpers for tests.
comp(A, c) = selectdim(A, ndims(A), c)
relnorm(x) = sqrt(sum(abs2, x))

@testset "HelmholtzDecomposition.jl" begin

    @testset "Aqua.jl" begin
        Aqua.test_all(HD; ambiguities = false)
    end

    @testset "Geometry" begin
        geom = HD.CartesianGeometry(1000.0, 2000.0)
        @test geom.spacing == (1000.0, 2000.0)
        @test HD.ndims_space(geom) == 2
        @test HD.cell_measure(geom) == 2e6
        @test HD.area_element(geom) == 2e6

        geom3 = HD.CartesianGeometry(2.0, 3.0, 4.0)
        @test HD.ndims_space(geom3) == 3
        @test HD.cell_measure(geom3) == 24.0

        sgeom = HD.SphericalGeometry()
        @test sgeom.R ≈ 6.371e6
        @test HD.ndims_space(sgeom) == 2
        @test HD.SphericalGeometry(1.0).R == 1.0
    end

    @testset "Grids" begin
        @testset "2D Cartesian" begin
            grid = HD.StructuredGrid(HD.CartesianGeometry(100.0, 100.0),
                collect(0.0:100.0:9900.0), collect(0.0:100.0:9900.0))
            @test HD.size_tuple(grid) == (100, 100)
            @test ndims(grid) == 2
            @test all(grid.mask)
            @test HD.cellmeasure(grid, 1, 1) ≈ 10000.0
            @test HD.coords(grid, 2, 3) == [100.0, 200.0]
        end
        @testset "3D Cartesian" begin
            grid = HD.StructuredGrid(HD.CartesianGeometry(1.0, 1.0, 1.0),
                collect(0.0:1.0:4.0), collect(0.0:1.0:3.0), collect(0.0:1.0:2.0))
            @test HD.size_tuple(grid) == (5, 4, 3)
            @test ndims(grid) == 3
        end
        @testset "mask" begin
            mask = trues(10, 10); mask[1, 1] = false
            grid = HD.StructuredGrid(HD.CartesianGeometry(100.0, 100.0),
                collect(0.0:100.0:900.0), collect(0.0:100.0:900.0); mask = mask)
            @test !HD.isactive(grid, 1, 1)
            @test HD.isactive(grid, 2, 2)
        end
        @testset "spherical areas vary with latitude" begin
            grid = HD.StructuredGrid(HD.SphericalGeometry(),
                collect(range(0.0, 2π, length = 72)), collect(range(-π / 2 + 0.01, π / 2 - 0.01, length = 36)))
            @test HD.size_tuple(grid) == (72, 36)
            @test all(grid.cell_measures .> 0)
        end
        @testset "dimension mismatch errors" begin
            @test_throws DimensionMismatch HD.StructuredGrid(HD.CartesianGeometry(1.0, 1.0), 1:3)
            @test_throws DimensionMismatch HD.StructuredGrid(HD.SphericalGeometry(), 1:3, 1:3, 1:3)
        end
    end

    @testset "Spectral Leray projector (pure, ND)" begin
        Random.seed!(1)
        for dims in ((8, 8), (6, 6, 6), (10,), (5, 4, 3))
            N = length(dims)
            vh = randn(ComplexF64, dims..., N)
            ks = ntuple(d -> randn(dims[d]), N)
            res = HD.helmholtz_project_spectral(vh, ks)
            @test maximum(abs.(res.u_rot .+ res.u_div .- vh)) < 1e-12
            # divergence-free rotational part: k·û_rot ≈ 0
            K = ntuple(d -> reshape(ks[d], ntuple(i -> i == d ? dims[d] : 1, N)), N)
            kdot = zeros(ComplexF64, dims...)
            for a in 1:N
                kdot .+= K[a] .* comp(res.u_rot, a)
            end
            @test maximum(abs.(kdot)) < 1e-11
        end
    end

    @testset "SOR Poisson (2D Dirichlet)" begin
        N = 32
        dx = 1.0 / (N + 1)
        xs = collect(range(dx, 1.0 - dx, length = N)); ys = copy(xs)
        grid = HD.StructuredGrid(HD.CartesianGeometry(dx, dx), xs, ys)
        RHS = [-(2π^2) * sin(π * xs[i]) * sin(π * ys[j]) for i in 1:N, j in 1:N]
        Φex = [sin(π * xs[i]) * sin(π * ys[j]) for i in 1:N, j in 1:N]
        Φ = zeros(N, N)
        r = HD.solve_poisson!(Φ, RHS, grid, HD.SORSolver(; max_iter = 50_000, tol = 1e-10, boundary = HD.Dirichlet()))
        @test r.converged
        @test maximum(abs.(Φ .- Φex)) < 0.05
    end

    @testset "FFTW spectral Poisson (2D periodic)" begin
        N = 64; L = 1.0; dx = L / N
        xs = collect(range(0.0, L - dx, length = N)); ys = copy(xs)
        grid = HD.StructuredGrid(HD.CartesianGeometry(dx, dx), xs, ys)
        kx = 2π / L
        RHS = [-(2kx^2) * sin(kx * xs[i]) * sin(kx * ys[j]) for i in 1:N, j in 1:N]
        Φex = [sin(kx * xs[i]) * sin(kx * ys[j]) for i in 1:N, j in 1:N]
        Φ = zeros(N, N)
        HD.solve_poisson!(Φ, RHS, grid, HD.AutoSolver())
        Φs = Φ .- Statistics.mean(Φ) .+ Statistics.mean(Φex)
        @test maximum(abs.(Φs .- Φex)) < 1e-10
    end

    @testset "Physical decomposition (manufactured, SOR)" begin
        n = 48; L = 1.0; h = L / (n + 1)
        xs = collect(range(h, L - h, length = n)); ys = copy(xs)
        grid = HD.StructuredGrid(HD.CartesianGeometry(h, h), xs, ys)
        solver = HD.SORSolver(; max_iter = 50_000, tol = 1e-10, boundary = HD.Dirichlet())

        # pure gradient χ = sin(πx)sin(πy)
        Ug = zeros(n, n, 2)
        for j in 1:n, i in 1:n
            Ug[i, j, 1] = π * cos(π * xs[i]) * sin(π * ys[j])
            Ug[i, j, 2] = π * sin(π * xs[i]) * cos(π * ys[j])
        end
        rg = HD.helmholtz_decompose(Ug, grid; solver = solver, boundary_χ = HD.Dirichlet(), boundary_ψ = HD.Dirichlet())
        @test relnorm(rg.u_rot) / relnorm(Ug) < 1e-3
        @test rg.harmonic_fraction < 1e-2

        # pure rotational ψ = sin(πx)sin(πy)
        Ur = zeros(n, n, 2)
        for j in 1:n, i in 1:n
            Ur[i, j, 1] = -π * sin(π * xs[i]) * cos(π * ys[j])
            Ur[i, j, 2] = π * cos(π * xs[i]) * sin(π * ys[j])
        end
        rr = HD.helmholtz_decompose(Ur, grid; solver = solver, boundary_χ = HD.Dirichlet(), boundary_ψ = HD.Dirichlet())
        @test relnorm(rr.u_div) / relnorm(Ur) < 1e-3
        @test size(HD.streamfunction(rr)) == (n, n)
    end

    @testset "Spectral decomposition (FFTW)" begin
        N = 64; L = 1.0; dx = L / N
        xs = collect(range(0.0, L - dx, length = N)); ys = copy(xs)
        grid = HD.StructuredGrid(HD.CartesianGeometry(dx, dx), xs, ys)
        u, v, = HD.taylor_green_vortex(grid)

        # coefficient-space projection identity
        u_hat = FFTW.rfft(u); v_hat = FFTW.rfft(v)
        resc = HD.helmholtz_project_spectral(u_hat, v_hat, grid)
        @test resc isa HD.SpectralCartesianResult
        @test maximum(abs.(comp(resc.u_rot, 1) .+ comp(resc.u_div, 1) .- u_hat)) < 1e-12
        @test maximum(abs.(comp(resc.u_div, 1))) / maximum(abs.(comp(resc.u_rot, 1))) < 1e-10

        # physical decomposition: Taylor-Green is purely rotational
        res = HD.helmholtz_decompose_spectral(u, v, grid)
        @test res isa HD.HelmholtzResult
        U = cat(u, v; dims = 3)
        @test maximum(abs.(res.u_rot .+ res.u_div .+ res.u_harm .- U)) < 1e-10
        @test relnorm(res.u_div) / relnorm(res.u_rot) < 1e-8
        @test res.harmonic_fraction < 1e-8
    end

    @testset "3D spectral decomposition (FFTW, ABC flow)" begin
        n = 16; L = 2π; h = L / n
        xs = collect(range(0, L - h, length = n)); ys = copy(xs); zs = copy(xs)
        grid = HD.StructuredGrid(HD.CartesianGeometry(h, h, h), xs, ys, zs)
        U = zeros(n, n, n, 3)
        for k in 1:n, j in 1:n, i in 1:n
            x, y, z = xs[i], ys[j], zs[k]
            U[i, j, k, 1] = sin(z) + cos(y)
            U[i, j, k, 2] = sin(x) + cos(z)
            U[i, j, k, 3] = sin(y) + cos(x)
        end
        res = HD.helmholtz_decompose_spectral(U, grid)
        @test res isa HD.HelmholtzResult
        @test relnorm(res.u_div) / relnorm(U) < 1e-8   # ABC is solenoidal
        @test res.harmonic_fraction < 1e-8
        @test maximum(abs.(res.u_rot .+ res.u_div .+ res.u_harm .- U)) < 1e-9
        @test length(HD.vector_potential(res)) == 3
    end

    @testset "Harmonic component (issue #1)" begin
        # Annulus: a central disk masked out → multiply-connected (b₁ = 1).
        n = 41; L = 2.0; h = L / (n - 1)
        xs = collect(range(-1.0, 1.0, length = n)); ys = copy(xs)
        base = HD.StructuredGrid(HD.CartesianGeometry(h, h), xs, ys)
        mask = HD.disk_mask(base; center = (0.0, 0.0), radius = 0.3)
        grid = HD.StructuredGrid(HD.CartesianGeometry(h, h), xs, ys; mask = mask)
        @test HD.count_holes(grid) == 1
        @test HD.betti1_estimate(grid) == 1

        # Pure circulation about the hole: harmonic (div-free AND curl-free in the annulus).
        u, v = HD.harmonic_vortex(grid; Γ = 1.0)
        solver = HD.SORSolver(; max_iter = 20_000, tol = 1e-9, boundary = HD.Dirichlet())
        res = HD.helmholtz_decompose(u, v, grid; solver = solver, boundary_χ = HD.Dirichlet(), boundary_ψ = HD.Dirichlet())
        U = cat(u, v; dims = 3)
        # Essentially all of the field is harmonic.
        @test res.harmonic_fraction > 0.9
        @test relnorm(res.u_rot) / relnorm(U) < 0.1
        @test relnorm(res.u_div) / relnorm(U) < 0.1
        # Exact reconstruction by construction, on active cells (the hole is masked out).
        active = repeat(mask, 1, 1, 2)
        @test maximum(abs.((res.u_rot .+ res.u_div .+ res.u_harm .- U)[active])) < 1e-8

        # Pure source (flux mode) is likewise harmonic on the annulus.
        us, vs = HD.harmonic_source(grid; q = 1.0)
        ress = HD.helmholtz_decompose(us, vs, grid; solver = solver, boundary_χ = HD.Dirichlet(), boundary_ψ = HD.Dirichlet())
        @test ress.harmonic_fraction > 0.9

        # A fully active rectangle is simply-connected.
        @test HD.count_holes(base) == 0
    end

    @testset "Execution backends" begin
        @test HD.local_backend(HD.MPIBackend(HD.SerialBackend())) isa HD.SerialBackend
        @test HD.is_distributed(HD.MPIBackend())
        @test !HD.is_distributed(HD.SerialBackend())
        @test HD.local_backend(HD.GPUBackend(:dummy)) isa HD.GPUBackend
        # backend= kwarg routes; SerialBackend reproduces the default result.
        grid = HD.StructuredGrid(HD.CartesianGeometry(0.05, 0.05),
            collect(range(0.05, 0.95, length = 19)), collect(range(0.05, 0.95, length = 19)))
        U = zeros(19, 19, 2)
        for j in 1:19, i in 1:19
            U[i, j, 1] = -π * sin(π * (0.05i)) * cos(π * (0.05j))
            U[i, j, 2] = π * cos(π * (0.05i)) * sin(π * (0.05j))
        end
        solver = HD.SORSolver(; max_iter = 5_000, tol = 1e-8, boundary = HD.Dirichlet())
        ra = HD.helmholtz_decompose(U, grid; solver = solver, boundary_χ = HD.Dirichlet(), boundary_ψ = HD.Dirichlet())
        rb = HD.helmholtz_decompose(U, grid; backend = HD.SerialBackend(), solver = solver, boundary_χ = HD.Dirichlet(), boundary_ψ = HD.Dirichlet())
        @test ra.u_rot == rb.u_rot
    end

    @testset "AutoSolver is mask-aware" begin
        # With a mask, AutoSolver must not pick the periodic FFT solver.
        mask = trues(16, 16); mask[8, 8] = false
        grid = HD.StructuredGrid(HD.CartesianGeometry(0.1, 0.1),
            collect(0.0:0.1:1.5), collect(0.0:0.1:1.5); mask = mask)
        @test HD._resolve_auto_solver(grid) isa HD.SORSolver
    end

    @testset "Spherical decomposition" begin
        lons = collect(range(0, 2π, length = 49)[1:48])   # periodic longitude
        lats = collect(range(-1.3, 1.3, length = 40))
        grid = HD.StructuredGrid(HD.SphericalGeometry(1.0), lons, lats)
        solver = HD.SORSolver(; max_iter = 30_000, tol = 1e-9, boundary = HD.Dirichlet())

        # Purely rotational Rossby wave → divergent part is small.
        u, v, = HD.rossby_wave(grid; n = 3, m = 2)
        res = HD.helmholtz_decompose(u, v, grid; solver = solver, boundary_χ = HD.Neumann(), boundary_ψ = HD.Dirichlet())
        U = cat(u, v; dims = 3)
        @test relnorm(res.u_div) / relnorm(U) < 0.05
        @test maximum(abs.(res.u_rot .+ res.u_div .+ res.u_harm .- U)) < 1e-8
        @test size(HD.streamfunction(res)) == (48, 40)

        # Select the spectral backends explicitly (both FSH + NUFSHT are loaded).
        fsh = HD._SPECTRAL_SOLVERS[:spherical_regular]()
        nusht = HD._SPECTRAL_SOLVERS[:spherical_irregular]()

        # FSH spectral path (Clenshaw–Curtis grid Nlon = 2·Nlat − 1).
        Nlat = 24; Nlon = 2 * Nlat - 1
        cclons = collect(range(0, 2π, length = Nlon + 1)[1:Nlon])
        cclats = collect(range(-1.2, 1.2, length = Nlat))
        ccgrid = HD.StructuredGrid(HD.SphericalGeometry(1.0), cclons, cclats)
        us, vs, = HD.rossby_wave(ccgrid; n = 2, m = 1)
        # Spectral decomposition returns a physical HelmholtzResult. (Accuracy on a true
        # Clenshaw–Curtis node set is validated separately; here the evenly-spaced grid only
        # exercises the FSH plumbing, and the SOR path above checks spherical accuracy.)
        sres = HD.helmholtz_decompose_spectral(us, vs, ccgrid; solver = fsh)
        @test sres isa HD.HelmholtzResult
        @test all(isfinite, sres.u_rot)

        # NUFSHT returns a physical result on an arbitrary grid.
        nres = HD.helmholtz_decompose_spectral(u, v, grid; solver = nusht)
        @test nres isa HD.HelmholtzResult

        # FSH grid validation rejects a non-CC grid.
        @test_throws ArgumentError HD.helmholtz_decompose_spectral(u, v, grid; solver = fsh)
    end

    @testset "Batch decomposition (serial/threaded/distributed)" begin
        grid = HD.StructuredGrid(HD.CartesianGeometry(0.05, 0.05),
            collect(range(0.05, 0.95, length = 19)), collect(range(0.05, 0.95, length = 19)))
        mkfield(s) = begin
            U = zeros(19, 19, 2)
            for j in 1:19, i in 1:19
                U[i, j, 1] = -π * s * sin(π * (0.05i)) * cos(π * (0.05j))
                U[i, j, 2] = π * s * cos(π * (0.05i)) * sin(π * (0.05j))
            end
            U
        end
        fields = [mkfield(s) for s in (1.0, 2.0, 0.5, 1.5)]
        solver = HD.SORSolver(; max_iter = 3_000, tol = 1e-8, boundary = HD.Dirichlet())
        kw = (; solver = solver, boundary_χ = HD.Dirichlet(), boundary_ψ = HD.Dirichlet())

        serial = HD.helmholtz_decompose_batch(grid, fields; kw...)
        @test length(serial) == 4
        @test serial[2].u_rot == HD.helmholtz_decompose(fields[2], grid; kw...).u_rot

        # Threaded (OhMyThreads ext) reproduces the serial batch.
        threaded = HD.helmholtz_decompose_batch(grid, fields; backend = HD.ThreadedBackend(), kw...)
        @test all(threaded[i].u_rot == serial[i].u_rot for i in 1:4)

        # Distributed (Distributed ext) reproduces the serial batch.
        Distributed.addprocs(2)
        try
            Distributed.@everywhere using HelmholtzDecomposition
            dist = HD.helmholtz_decompose_batch(grid, fields; backend = HD.DistributedBackend(), kw...)
            @test all(dist[i].u_rot == serial[i].u_rot for i in 1:4)
        finally
            Distributed.rmprocs(Distributed.workers())
        end
    end

    @testset "TestFields generation" begin
        grid = HD.StructuredGrid(HD.CartesianGeometry(0.01, 0.01),
            collect(range(0.0, 1.0 - 0.01, length = 100)), collect(range(0.0, 1.0 - 0.01, length = 100)))
        u, v, ur, vr, ud, vd = HD.taylor_green_vortex(grid)
        @test all(ud .== 0) && all(vd .== 0) && maximum(abs.(u)) > 0
        u, v, ur, vr, ud, vd = HD.point_source_sink(grid)
        @test all(ur .== 0) && all(vr .== 0) && maximum(abs.(u)) > 0

        sgrid = HD.StructuredGrid(HD.SphericalGeometry(1.0),
            collect(range(0.0, 2π, length = 72)), collect(range(-π / 4, π / 4, length = 36)))
        u, v, ur, vr, ud, vd = HD.rossby_wave(sgrid)
        @test all(ud .== 0) && all(vd .== 0) && maximum(abs.(u)) > 0
    end
end
