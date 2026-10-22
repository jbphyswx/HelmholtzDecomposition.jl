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

function gate(f::F, a, b, c, d) where {F}
    f(a, b, c, d)
    f(a, b, c, d)
    return @allocated f(a, b, c, d)
end

function _grids()
    cart = FG.Geometry.CartesianGeometry{Float64}()
    xs = range(0.0, 1.0; length = 12)
    ys = range(0.0, 0.8; length = 9)
    mask = trues(12, 9); mask[5:7, 4:5] .= false
    return (
        ("bounded", FG.Grids.StructuredGrid(cart, xs, ys)),
        ("periodic", FG.Grids.StructuredGrid(cart, range(0.0, 1.0 - 1 / 12; length = 12), range(0.0, 0.9; length = 10);
                                             topology = (true, true), period = (1.0, 1.0))),
        ("masked", FG.Grids.StructuredGrid(cart, xs, ys; mask = mask)),
        ("spherical", FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), FG.Axes.PeriodicAxis(0.0, 2π, 16),
                                              range(-π / 2 + π / 20, π / 2 - π / 20; length = 10))),
        ("3d", FG.Grids.StructuredGrid(cart, range(0.0, 1.0; length = 6), range(0.0, 0.9; length = 5),
                                       range(0.0, 0.6; length = 4))),
    )
end

"""
    _solvers(grid)

Jacobi- and multigrid-preconditioned conjugate gradients on every grid, and on a Cartesian grid
`AutoSolver`, which with FFTW loaded resolves to the periodic FFT, the bounded cosine/sine transforms
(DST-I on the duals) or conjugate gradients on a masked grid.
"""
function _solvers(grid)
    cg = (("jacobi", HD.CGSolver(; multigrid = false)), ("mg", HD.CGSolver(; multigrid = true)))
    FG.Grids.grid_geometry(grid) isa FG.Geometry.AbstractCartesianGeometry || return cg
    return (cg..., ("auto", HD.AutoSolver()))
end

Test.@testset "allocations: decomposition" begin
    for (name, grid) in _grids(), (sname, solver) in _solvers(grid),
        bc in (FG.Operators.Neumann(), FG.Operators.Dirichlet())
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
