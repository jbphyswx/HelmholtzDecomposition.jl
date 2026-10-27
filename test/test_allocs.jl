# Allocation gates: a steady-state call, every buffer already built, allocates nothing, on
# `SerialBackend()` (a threaded backend carries a task-spawn floor). Each gate warms both branches and
# calls through a fixed-arity function barrier capturing no global. FlowGeometries gates its own
# operators.

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using Random: Random

include("cases.jl")

function gate(f::F, a, b, c, d) where {F}
    f(a, b, c, d)
    f(a, b, c, d)
    return @allocated f(a, b, c, d)
end

Test.@testset "allocations: decomposition" begin
    for (name, grid) in _grids(), (sname, solver) in _solvers(grid), bc in _conditions()
        plan = HD.plan_helmholtz(grid; boundary = bc, solver = solver, execution = CB.SerialBackend())
        ws, res = HD.allocate_workspace(plan), HD.allocate_result(plan)
        Random.seed!(6)
        u = randn(size(grid)..., ndims(grid))
        f_dec(a, b, c, d) = HD.helmholtz_decompose!(a, b, c, d)
        Test.@testset "$name/$sname/$(nameof(typeof(bc)))" begin
            Test.@test gate(f_dec, res, u, plan, ws) == 0
        end
    end
end
