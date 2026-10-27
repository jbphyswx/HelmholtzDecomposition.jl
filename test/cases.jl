# The grids, solvers and conditions the allocation and dispatch gates run over.

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG

function _grids()
    cart = FG.Geometry.CartesianGeometry{Float64}()
    xs = range(0.0, 1.0; length = 12)
    ys = range(0.0, 0.8; length = 9)
    mask = trues(12, 9); mask[5:7, 4:5] .= false
    return (
        ("bounded", FG.Grids.StructuredGrid(cart, xs, ys)),
        ("periodic", FG.Grids.StructuredGrid(cart, range(0.0, 1.0 - 1 / 12; length = 12), range(0.0, 0.9; length = 10);
                                             topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (1.0, 1.0))),
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
    cg = (("jacobi", HD.CGSolver(; preconditioner = HD.DiagonalScaling())),
          ("mg", HD.CGSolver(; preconditioner = HD.MultigridCycle())))
    FG.Grids.grid_geometry(grid) isa FG.Geometry.AbstractCartesianGeometry || return cg
    return (cg..., ("auto", HD.AutoSolver()))
end

_conditions() = (FG.Operators.Neumann(), FG.Operators.Dirichlet())
