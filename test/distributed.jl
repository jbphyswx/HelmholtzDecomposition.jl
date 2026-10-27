# Distributed batches equal the serial one: over a caller's plan, which each chunk carries, and over a plan
# every worker builds and holds, which only the fields reach. A worker runs the plan's solver, so it loads
# the package that solver's extension needs (FFTW for the bounded box's transform solver).

using Test: Test
using Distributed: Distributed
Distributed.nworkers() == 1 && Distributed.workers() == [1] &&
    Distributed.addprocs(2; exeflags = "--project=$(Base.active_project())")
Distributed.@everywhere using HelmholtzDecomposition: HelmholtzDecomposition
Distributed.@everywhere using FFTW: FFTW
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using Random: Random

Test.@testset "Distributed batches equal the serial one" begin
    n = 10
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), range(0.0, 1.0; length = n),
                                   range(0.0, 1.0; length = n))
    plan = HD.plan_helmholtz(grid; execution = CB.SerialBackend())
    Test.@test plan.solver isa HD.CartesianBoundedSolver
    Random.seed!(41)
    fields = [randn(n, n, 2) for _ in 1:5]
    serial = HD.helmholtz_decompose_batch(plan, fields; execution = CB.SerialBackend())
    sent = HD.helmholtz_decompose_batch(plan, fields; execution = CB.DistributedBackend())
    held = HD.helmholtz_decompose_batch(HD.plan_helmholtz(grid; execution = CB.DistributedBackend()), fields)
    for b in (sent, held)
        Test.@test b.u_rot == serial.u_rot && b.χ == serial.χ
        Test.@test b.harmonic_fraction == serial.harmonic_fraction
        Test.@test b.χ_solve == serial.χ_solve
    end
end
