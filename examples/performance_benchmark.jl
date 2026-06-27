"""
Performance Benchmark

Compare wall-clock time of the SOR fallback vs the FFTW spectral Poisson solve across grid
sizes (both via the physical-space `helmholtz_decompose`, which differs only in the
Poisson solver used internally).
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FFTW: FFTW

function benchmark_solver(N, solver)
    L = 1.0
    dx = L / N
    grid = HD.StructuredGrid(HD.CartesianGeometry(dx, dx),
        collect(range(0.0, L - dx, length = N)), collect(range(0.0, L - dx, length = N)))
    u, v, = HD.rankine_vortex_with_source(grid)
    HD.helmholtz_decompose(u, v, grid; solver = solver)            # warmup
    t = @elapsed HD.helmholtz_decompose(u, v, grid; solver = solver)
    return t
end

println("=== Performance Benchmark: SOR vs FFTW spectral Poisson ===\n")
for N in (16, 32, 64, 128)
    t_fft = benchmark_solver(N, HD.AutoSolver())   # FFTW (regular) when loaded
    if N <= 64
        t_sor = benchmark_solver(N, HD.SORSolver(; max_iter = 50_000, tol = 1e-6))
        println("N=$N: FFTW=$(round(t_fft*1000, digits=1))ms, SOR=$(round(t_sor*1000, digits=1))ms, speedup=$(round(t_sor/t_fft, digits=1))x")
    else
        println("N=$N: FFTW=$(round(t_fft*1000, digits=1))ms, SOR=skipped (too slow)")
    end
end
