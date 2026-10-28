# Device parity on CUDA hardware: on a periodic grid the staggered decomposition and the periodic
# spectral route, each run on `GPUBackend(CUDA.CUDABackend())`, equal the serial host result. Every loop
# runs through `FlowGeometries.Execution` and every transform through `AbstractFFTs`, which CUFFT
# implements for a `CuArray`. Run with `julia --project=gpu gpu/runtests.jl` on a machine with a device.

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions
using CUDA: CUDA
using FFTW: FFTW
using Random: Random

Test.@testset "GPU parity" begin
    Test.@test CUDA.functional()
    N, L = 64, 1.0
    axis = FG.Axes.PeriodicUniformAxis(0.0, L, N)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), axis, axis)
    Random.seed!(7)
    U = randn(N, N, 2)
    gpu = CB.GPUBackend(CUDA.CUDABackend())
    close_to(h, d) = maximum(abs, h .- Array(d)) < 1e-10 * max(maximum(abs, h), 1.0)

    Test.@testset "staggered" begin
        host = HD.plan_helmholtz(grid; execution = CB.SerialBackend())
        dev = HD.plan_helmholtz(grid; execution = gpu)
        r_host = HD.helmholtz_decompose!(HD.allocate_result(host), U, host)
        r_dev = HD.helmholtz_decompose!(HD.allocate_result(dev), CUDA.CuArray(U), dev)
        Test.@test r_dev.u_rot isa CUDA.CuArray
        for (h, d) in ((r_host.u_rot, r_dev.u_rot), (r_host.u_div, r_dev.u_div),
                       (r_host.u_harm, r_dev.u_harm), (r_host.χ, r_dev.χ))
            Test.@test close_to(h, d)
        end
        Test.@test isapprox(r_host.harmonic_fraction, r_dev.harmonic_fraction; rtol = 1e-10)
    end

    Test.@testset "spectral" begin
        s_host = HD.helmholtz_decompose_spectral(U, grid; execution = CB.SerialBackend())
        for s_dev in (HD.helmholtz_decompose_spectral(U, grid; execution = gpu),
                      HD.helmholtz_decompose_spectral(CUDA.CuArray(U), grid; execution = gpu))
            Test.@test s_dev.u_rot isa CUDA.CuArray
            for (h, d) in ((s_host.u_rot, s_dev.u_rot), (s_host.u_div, s_dev.u_div),
                           (s_host.u_harm, s_dev.u_harm), (s_host.χ, s_dev.χ),
                           (s_host.divergence, s_dev.divergence))
                Test.@test close_to(h, d)
            end
            Test.@test isapprox(s_host.harmonic_fraction, s_dev.harmonic_fraction; rtol = 1e-10)
        end
    end
end
