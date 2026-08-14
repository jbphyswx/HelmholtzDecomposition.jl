"""
Device parity, on real hardware.

Nothing here is CUDA-specific beyond obtaining the backend. The operators, the solver and the
reductions all run through `FlowGeometries.Execution`, and the transforms through `AbstractFFTs`,
so a device array needs no vendor code — which is why this file tests *parity* rather than a
separate GPU implementation.

The same parity is checked without hardware in the main suite through
`GPUBackend(KernelAbstractions.CPU())`; this runs it where the arrays are genuinely elsewhere.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using CUDA: CUDA
using AbstractFFTs: AbstractFFTs
using Random: Random

Test.@testset "GPU parity" begin
    if !CUDA.functional()
        @info "no CUDA device; skipping"
    else
        N, L = 64, 1.0
        axis = range(0.0, L - L / N; length = N)
        grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), axis, axis;
                                       topology = (true, true), period = (L, L))
        Random.seed!(7)
        U = randn(N, N, 2)

        host = HD.plan_helmholtz(grid; backend = CB.SerialBackend())
        dev = HD.plan_helmholtz(grid; backend = CB.GPUBackend(CUDA.CUDABackend()))

        r_host = HD.helmholtz_decompose!(HD.allocate_result(host), U, host,
                                         HD.allocate_workspace(host))
        r_dev = HD.helmholtz_decompose!(HD.allocate_result(dev), CUDA.cu(U), dev,
                                        HD.allocate_workspace(dev))

        # The result is a `HelmholtzResult` of device arrays — the same type the host path gives,
        # not a different API.
        Test.@test r_dev.u_rot isa CUDA.CuArray
        for (h, d) in ((r_host.u_rot, r_dev.u_rot), (r_host.u_div, r_dev.u_div),
                       (r_host.u_harm, r_dev.u_harm), (r_host.χ, r_dev.χ))
            Test.@test maximum(abs, h .- Array(d)) < 1e-10 * max(maximum(abs, h), 1.0)
        end
        Test.@test isapprox(r_host.harmonic_fraction, r_dev.harmonic_fraction; rtol = 1e-10)
    end
end
