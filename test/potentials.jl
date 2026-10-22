"""
The potentials-only path returns the full decomposition's potentials and holds none of its
collocated velocity arrays.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using Random: Random

const CARTP = FG.Geometry.CartesianGeometry{Float64}()

function _cases()
    xs = range(0.0, 1.0; length = 16)
    ys = range(0.0, 0.8; length = 13)
    mask = trues(16, 13); mask[7:9, 6:8] .= false
    return (
        ("bounded", FG.Grids.StructuredGrid(CARTP, xs, ys)),
        ("masked", FG.Grids.StructuredGrid(CARTP, xs, ys; mask = mask)),
        ("3d", FG.Grids.StructuredGrid(CARTP, range(0.0, 1.0; length = 8),
                                       range(0.0, 0.9; length = 7),
                                       range(0.0, 0.8; length = 6))),
    )
end

Test.@testset "potentials-only path" begin
    for (name, grid) in _cases()
        Test.@testset "$name" begin
            N = ndims(grid)
            Random.seed!(17)
            u = randn(size(grid)..., N)
            plan = HD.plan_helmholtz(grid; execution = CB.SerialBackend())
            ws = HD.allocate_workspace(plan)

            full = HD.helmholtz_decompose!(HD.allocate_result(plan), u, plan, ws)
            pot = HD.helmholtz_decompose!(HD.allocate_potentials(plan), u, plan, ws)

            # Both results run the same potential solves through one workspace, so they agree exactly.
            Test.@test pot.χ == full.χ
            Test.@test pot.divergence == full.divergence
            Test.@test all(p == f for (p, f) in zip(pot.rotation_potential, full.rotation_potential))
            Test.@test pot.χ_solve.iterations == full.χ_solve.iterations

            # `u_rot`, `u_div` and `u_harm` are `(dims..., N)` each; none of them is allocated.
            Test.@test Base.summarysize(full) - Base.summarysize(pot) >=
                  3 * prod(size(grid)) * N * sizeof(Float64)
        end
    end
end

Test.@testset "potentials entry point" begin
    grid = FG.Grids.StructuredGrid(CARTP, range(0.0, 1.0; length = 12),
                                   range(0.0, 1.0; length = 12))
    Random.seed!(18)
    u = randn(12, 12, 2)
    pot = HD.helmholtz_potentials(u, grid)
    Test.@test HD.helmholtz_decompose(u, grid).χ ≈ pot.χ
end