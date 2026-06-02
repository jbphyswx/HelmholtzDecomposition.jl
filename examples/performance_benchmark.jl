"""
Performance Benchmark

Compare wall-clock time of SOR vs FFTW spectral solver across grid sizes.
"""

using HelmholtzDecomposition: HelmholtzDecomposition
using FFTW: FFTW

function benchmark_solver(N, solver)
    L = 1.0
    dx = L / N
    geom = HelmholtzDecomposition.CartesianGeometry(dx, dx)
    xs = collect(range(0.0, L - dx, length=N))
    ys = collect(range(0.0, L - dx, length=N))
    grid = HelmholtzDecomposition.StructuredGrid(geom, xs, ys)

    u, v, _, _, _, _ = HelmholtzDecomposition.rankine_vortex_with_source(grid)

    # Warmup
    HelmholtzDecomposition.helmholtz_decompose(u, v, grid; solver=solver)

    # Timed run
    t = @elapsed begin
        result = HelmholtzDecomposition.helmholtz_decompose(u, v, grid; solver=solver)
    end
    return t, result
end

println("=== Performance Benchmark: SOR vs FFTW ===")
println()

for N in [16, 32, 64, 128]
    # FFTW
    t_fft, res_fft = benchmark_solver(N, HelmholtzDecomposition.AutoSolver())
    
    # SOR (only for small grids, too slow for large)
    if N <= 64
        sor = HelmholtzDecomposition.SORSolver(; max_iter=50_000, tol=1e-6)
        t_sor, res_sor = benchmark_solver(N, sor)
        speedup = t_sor / t_fft
        println("N=$N: FFTW=$(round(t_fft*1000, digits=1))ms, SOR=$(round(t_sor*1000, digits=1))ms, speedup=$(round(speedup, digits=1))x")
    else
        println("N=$N: FFTW=$(round(t_fft*1000, digits=1))ms, SOR=skipped (too slow)")
    end
end
