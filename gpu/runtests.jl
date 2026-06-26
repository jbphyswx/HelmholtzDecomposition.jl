# GPU tests — run on a CUDA-capable machine: `julia --project=gpu gpu/runtests.jl`.
# Skipped (not failed) when no functional GPU is present.
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using CUDA: CUDA
using FFTW: FFTW
using Test: Test, @testset, @test

@testset "GPU spectral decomposition" begin
    if !CUDA.functional()
        @info "CUDA not functional — skipping GPU tests."
    else
        N = 64; L = 1.0; dx = L / N
        xs = collect(range(0.0, L - dx, length = N)); ys = copy(xs)
        grid = HD.StructuredGrid(HD.CartesianGeometry(dx, dx), xs, ys)
        u, v = HD.taylor_green_vortex(grid)
        U = cat(u, v; dims = 3)

        # AutoBackend resolves to GPU for CuArray input.
        @test HD._resolve_backend(HD.AutoBackend(), CUDA.cu(U)) isa HD.GPUBackend

        # GPU velocity split (physical) matches the CPU FFTW result.
        cpu = HD.helmholtz_decompose_spectral(U, grid)                       # CPU HelmholtzResult
        gpu = HD.helmholtz_decompose_spectral(CUDA.cu(U), grid; output = :physical)
        @test maximum(abs.(Array(gpu.u_rot) .- cpu.u_rot)) < 1e-6
        @test maximum(abs.(Array(gpu.u_div) .- cpu.u_div)) < 1e-6

        # Pure-rotational field: divergent part vanishes on the GPU too.
        @test maximum(abs.(Array(gpu.u_div))) / maximum(abs.(Array(gpu.u_rot))) < 1e-6
    end
end
