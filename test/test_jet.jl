# Dispatch gates over the grids, solvers and conditions of `cases.jl`: building a plan and a decomposition
# through it make no runtime dispatch. JET refuses pre-release Julia.

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using JET: JET

include("cases.jl")

Test.@testset "dispatch: decomposition and plan" begin
    if isempty(VERSION.prerelease)
        exts = map(n -> Base.get_extension(HD, n), (:HelmholtzDecompositionFFTWExt, :HelmholtzDecompositionAbstractFFTsExt))
        Test.@test all(!isnothing, exts)
        targets = (HD, FG, exts...)
        for (name, grid) in _grids(), (sname, solver) in _solvers(grid), bc in _conditions()
            plan = HD.plan_helmholtz(grid; boundary = bc, solver = solver, execution = CB.SerialBackend())
            ws, res = HD.allocate_workspace(plan), HD.allocate_result(plan)
            u = randn(size(grid)..., ndims(grid))
            Test.@testset "$name/$sname/$(nameof(typeof(bc)))" begin
                JET.@test_opt target_modules = targets HD.helmholtz_decompose!(res, u, plan, ws)
                JET.@test_opt target_modules = targets HD.plan_helmholtz(
                    grid; boundary = bc, solver = solver, execution = CB.SerialBackend())
            end
        end
    end
end

Test.@testset "dispatch: held periodic spectral decomposition" begin
    if isempty(VERSION.prerelease)
        targets = (HD, FG, Base.get_extension(HD, :HelmholtzDecompositionAbstractFFTsExt))
        grid = Dict(_grids())["periodic"]
        solver, serial = HD.CartesianSpectralSolver(), CB.SerialBackend()
        st = HD.prepare_spectral(solver, grid; execution = serial)
        u = randn(size(grid)..., ndims(grid))
        JET.@test_opt target_modules = targets HD.prepare_spectral(solver, grid; execution = serial)
        JET.@test_opt target_modules = targets HD.helmholtz_decompose_spectral(
            u, grid; solver, execution = serial, state = st)
        res = HD.allocate_result(st, grid)
        JET.@test_opt target_modules = targets HD.allocate_result(st, grid)
        JET.@test_opt target_modules = targets HD.helmholtz_decompose_spectral!(
            res, u, grid; solver, execution = serial, state = st)
    end
end
