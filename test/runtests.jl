using HelmholtzDecomposition: HelmholtzDecomposition
using Test: Test, @testset, @test
using Aqua: Aqua
using Statistics: Statistics
using Random: Random
using FFTW: FFTW

@testset "HelmholtzDecomposition.jl" begin

    @testset "Aqua.jl" begin
        Aqua.test_all(HelmholtzDecomposition)
    end

    @testset "Geometry" begin
        @testset "CartesianGeometry" begin
            geom = HelmholtzDecomposition.CartesianGeometry(1000.0, 2000.0)
            @test geom.dx == 1000.0
            @test geom.dy == 2000.0
            @test geom.dz == 0.0
            @test HelmholtzDecomposition.area_element(geom) == 2e6
        end

        @testset "SphericalGeometry" begin
            geom = HelmholtzDecomposition.SphericalGeometry()
            @test geom.R ≈ 6.371e6
            geom2 = HelmholtzDecomposition.SphericalGeometry(1.0)
            @test geom2.R == 1.0
        end
    end

    @testset "Grids" begin
        @testset "CartesianGrid construction" begin
            geom = HelmholtzDecomposition.CartesianGeometry(100.0, 100.0)
            xs = collect(0.0:100.0:9900.0)
            ys = collect(0.0:100.0:9900.0)
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)
            @test HelmholtzDecomposition.size_tuple(grid) == (100, 100)
            @test all(grid.mask)
            @test HelmholtzDecomposition.area(grid, 1, 1) ≈ 10000.0
        end

        @testset "CartesianGrid with mask" begin
            geom = HelmholtzDecomposition.CartesianGeometry(100.0, 100.0)
            xs = collect(0.0:100.0:900.0)
            ys = collect(0.0:100.0:900.0)
            mask = trues(10, 10)
            mask[1, 1] = false
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys, mask)
            @test !HelmholtzDecomposition.iswet(grid, 1, 1)
            @test HelmholtzDecomposition.iswet(grid, 2, 2)
        end

        @testset "SphericalGrid construction" begin
            geom = HelmholtzDecomposition.SphericalGeometry()
            lons = collect(range(0.0, 2π, length=72))
            lats = collect(range(-π/2 + 0.01, π/2 - 0.01, length=36))
            grid = HelmholtzDecomposition.StructuredGrid(geom, lons, lats)
            @test HelmholtzDecomposition.size_tuple(grid) == (72, 36)
            @test all(grid.areas .> 0)
        end
    end

    @testset "SOR Solver" begin
        @testset "Cartesian Poisson solve (Dirichlet)" begin
            # Use Dirichlet BCs with a solution that is zero on boundaries
            N = 32
            dx = 1.0 / (N + 1)
            geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
            xs = collect(range(dx, 1.0 - dx, length=N))
            ys = collect(range(dx, 1.0 - dx, length=N))
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

            # Exact: Φ = sin(πx) sin(πy), satisfies Φ=0 on boundaries x=0,1, y=0,1
            # ∇²Φ = -2π² sin(πx) sin(πy)
            kx = π
            ky = π
            RHS = Matrix{Float64}(undef, N, N)
            Φ_exact = Matrix{Float64}(undef, N, N)
            for j in 1:N, i in 1:N
                RHS[i, j] = -(kx^2 + ky^2) * sin(kx * xs[i]) * sin(ky * ys[j])
                Φ_exact[i, j] = sin(kx * xs[i]) * sin(ky * ys[j])
            end

            Φ = zeros(N, N)
            solver = HelmholtzDecomposition.SORSolver(; max_iter=50_000, tol=1e-10, boundary=:dirichlet)
            result = HelmholtzDecomposition.solve_poisson!(Φ, RHS, grid, solver)

            @test result.converged
            max_err = maximum(abs.(Φ .- Φ_exact))
            @test max_err < 0.05
        end
    end

    @testset "FFTW Spectral Solver" begin
        @testset "Cartesian periodic Poisson solve" begin
            N = 64
            L = 1.0
            dx = L / N
            geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
            xs = collect(range(0.0, L - dx, length=N))
            ys = collect(range(0.0, L - dx, length=N))
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

            kx = 2π / L
            ky = 2π / L
            RHS = Matrix{Float64}(undef, N, N)
            Φ_exact = Matrix{Float64}(undef, N, N)
            for j in 1:N, i in 1:N
                RHS[i, j] = -(kx^2 + ky^2) * sin(kx * xs[i]) * sin(ky * ys[j])
                Φ_exact[i, j] = sin(kx * xs[i]) * sin(ky * ys[j])
            end

            Φ = zeros(N, N)
            # AutoSolver should pick FFTW extension since we loaded FFTW above
            result = HelmholtzDecomposition.solve_poisson!(Φ, RHS, grid, HelmholtzDecomposition.AutoSolver())

            # FFT solver should be very accurate for periodic fields
            Φ_shifted = Φ .- Statistics.mean(Φ) .+ Statistics.mean(Φ_exact)
            max_err = maximum(abs.(Φ_shifted .- Φ_exact))
            @test max_err < 1e-10
        end
    end

    @testset "Helmholtz Decomposition" begin
        @testset "Cartesian: pure rotational (Taylor-Green)" begin
            N = 64
            L = 1.0
            dx = L / N
            geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
            xs = collect(range(0.0, L - dx, length=N))
            ys = collect(range(0.0, L - dx, length=N))
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

            u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact =
                HelmholtzDecomposition.taylor_green_vortex(grid)

            result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

            # Divergent component should be near zero
            div_mag = sqrt.(result.u_div.^2 .+ result.v_div.^2)
            rot_mag = sqrt.(result.u_rot.^2 .+ result.v_rot.^2)
            @test Statistics.mean(div_mag) / Statistics.mean(rot_mag) < 0.01

            # Reconstruction: u_rot + u_div ≈ u
            recon_err_u = maximum(abs.(result.u_rot .+ result.u_div .- u))
            recon_err_v = maximum(abs.(result.v_rot .+ result.v_div .- v))
            @test recon_err_u < 0.1 * maximum(abs.(u))
            @test recon_err_v < 0.1 * maximum(abs.(v))
        end

        @testset "Cartesian: pure divergent (source/sink)" begin
            N = 64
            L = 1.0
            dx = L / N
            geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
            xs = collect(range(0.0, L - dx, length=N))
            ys = collect(range(0.0, L - dx, length=N))
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

            u, v, u_rot_exact, v_rot_exact, u_div_exact, v_div_exact =
                HelmholtzDecomposition.point_source_sink(grid)

            result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

            # Rotational component should be near zero
            rot_mag = sqrt.(result.u_rot.^2 .+ result.v_rot.^2)
            div_mag = sqrt.(result.u_div.^2 .+ result.v_div.^2)
            @test Statistics.mean(rot_mag) / Statistics.mean(div_mag) < 0.01

            # Reconstruction
            recon_err_u = maximum(abs.(result.u_rot .+ result.u_div .- u))
            recon_err_v = maximum(abs.(result.v_rot .+ result.v_div .- v))
            @test recon_err_u < 0.1 * maximum(abs.(u))
            @test recon_err_v < 0.1 * maximum(abs.(v))
        end

        @testset "Cartesian: mixed field reconstruction" begin
            N = 64
            L = 1.0
            dx = L / N
            geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
            xs = collect(range(0.0, L - dx, length=N))
            ys = collect(range(0.0, L - dx, length=N))
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

            u, v, _, _, _, _ = HelmholtzDecomposition.rankine_vortex_with_source(grid)

            result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

            # Reconstruction should be accurate
            recon_err_u = maximum(abs.(result.u_rot .+ result.u_div .- u))
            recon_err_v = maximum(abs.(result.v_rot .+ result.v_div .- v))
            @test recon_err_u < 0.1 * maximum(abs.(u))
            @test recon_err_v < 0.1 * maximum(abs.(v))

            # Both components should be non-negligible
            rot_mag = Statistics.mean(sqrt.(result.u_rot.^2 .+ result.v_rot.^2))
            div_mag = Statistics.mean(sqrt.(result.u_div.^2 .+ result.v_div.^2))
            @test rot_mag > 0.01
            @test div_mag > 0.01
        end
    end

    @testset "HelmholtzResult construction" begin
        geom = HelmholtzDecomposition.CartesianGeometry(100.0, 100.0)
        xs = collect(0.0:100.0:900.0)
        ys = collect(0.0:100.0:900.0)
        grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)
        result = HelmholtzDecomposition.HelmholtzResult(grid)
        @test size(result.u_rot) == (10, 10)
        @test size(result.ψ) == (10, 10)
    end

    @testset "TestFields generation" begin
        @testset "Taylor-Green" begin
            geom = HelmholtzDecomposition.CartesianGeometry(0.01, 0.01)
            xs = collect(range(0.0, 1.0 - 0.01, length=100))
            ys = collect(range(0.0, 1.0 - 0.01, length=100))
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)
            u, v, ur, vr, ud, vd = HelmholtzDecomposition.taylor_green_vortex(grid)
            @test all(ud .== 0)
            @test all(vd .== 0)
            @test maximum(abs.(u)) > 0
        end

        @testset "Point source/sink" begin
            geom = HelmholtzDecomposition.CartesianGeometry(0.01, 0.01)
            xs = collect(range(0.0, 1.0 - 0.01, length=100))
            ys = collect(range(0.0, 1.0 - 0.01, length=100))
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)
            u, v, ur, vr, ud, vd = HelmholtzDecomposition.point_source_sink(grid)
            @test all(ur .== 0)
            @test all(vr .== 0)
            @test maximum(abs.(u)) > 0
        end

        @testset "Rossby wave (spherical)" begin
            geom = HelmholtzDecomposition.SphericalGeometry(1.0)
            lons = collect(range(0.0, 2π, length=72))
            lats = collect(range(-π/4, π/4, length=36))
            grid = HelmholtzDecomposition.StructuredGrid(geom, lons, lats)
            u, v, ur, vr, ud, vd = HelmholtzDecomposition.rossby_wave(grid)
            @test all(ud .== 0)
            @test all(vd .== 0)
            @test maximum(abs.(u)) > 0
        end
    end

    @testset "Spectral Decomposition" begin
        @testset "Base complex Cartesian projection" begin
            N = 64
            L = 1.0
            dx = L / N
            geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
            xs = collect(range(0.0, L - dx, length=N))
            ys = collect(range(0.0, L - dx, length=N))
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

            u, v, _, _, _, _ = HelmholtzDecomposition.taylor_green_vortex(grid)
            u_hat = FFTW.rfft(u)
            v_hat = FFTW.rfft(v)

            res = HelmholtzDecomposition.helmholtz_project_spectral(u_hat, v_hat, grid)
            @test res isa HelmholtzDecomposition.SpectralCartesianResult

            # Exact reconstruction: u_rot_hat + u_div_hat ≈ u_hat
            @test maximum(abs.(res.u_rot .+ res.u_div .- u_hat)) < 1e-12
            @test maximum(abs.(res.v_rot .+ res.v_div .- v_hat)) < 1e-12

            # For Taylor-Green (purely rotational), divergent component should be very small
            @test maximum(abs.(res.u_div)) / maximum(abs.(res.u_rot)) < 1e-10
        end

        @testset "Physical Cartesian spectral decomposition (FFTW)" begin
            N = 64
            L = 1.0
            dx = L / N
            geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
            xs = collect(range(0.0, L - dx, length=N))
            ys = collect(range(0.0, L - dx, length=N))
            grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

            u, v, _, _, _, _ = HelmholtzDecomposition.taylor_green_vortex(grid)
            res = HelmholtzDecomposition.helmholtz_decompose_spectral(u, v, grid)
            @test res isa HelmholtzDecomposition.SpectralCartesianResult

            # Test reconstruction
            u_hat = FFTW.rfft(u)
            @test maximum(abs.(res.u_rot .+ res.u_div .- u_hat)) < 1e-12
        end
    end
end
