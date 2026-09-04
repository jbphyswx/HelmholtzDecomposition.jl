"""
Allocation gates.

A steady-state call — one where every buffer already exists — must allocate nothing. This file is
the gate that says so, and it exists because its absence let a broadcast that materialises a full
grid array per conjugate-gradient iteration sit in the solver unnoticed while the package reported
its performance work as done.

Two measurement traps this avoids, both of which make the harness rather than the code the thing
being measured: the closure must capture no `Module` local and take a fixed arity, and both
branches must be warmed before the count is taken. The backend is named `SerialBackend()`
explicitly — a threaded backend carries a task-spawn floor, so `== 0` is only meaningful serially.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using Random: Random

# Fixed arity, no splatting: `f(args...)` costs ~240 B of its own.
function gate(f::F, a, b, c, d) where {F}
    f(a, b, c, d)
    f(a, b, c, d)
    return @allocated f(a, b, c, d)
end

function gate5(f::F, a, b, c, d, e) where {F}
    f(a, b, c, d, e)
    f(a, b, c, d, e)
    return @allocated f(a, b, c, d, e)
end

_grids() = begin
    cart = FG.Geometry.CartesianGeometry{Float64}()
    xs = range(0.0, 1.0; length = 24)
    ys = range(0.0, 0.8; length = 19)
    xp = range(0.0, 1.0 - 1 / 24; length = 24)
    yp = range(0.0, 1.0 - 1 / 19; length = 19)
    mask = trues(24, 19)
    mask[10:14, 8:11] .= false
    sph = FG.Geometry.SphericalGeometry(1.0)
    λ = range(0, 2π - 2π / 20; length = 20)
    φ = range(-π / 2 + π / 24, π / 2 - π / 24; length = 12)
    zs = range(0.0, 0.6; length = 6)
    return (
        ("bounded", FG.Grids.StructuredGrid(cart, xs, ys)),
        ("periodic", FG.Grids.StructuredGrid(cart, xp, yp; topology = (true, true),
                                             period = (1.0, 1.0))),
        ("masked", FG.Grids.StructuredGrid(cart, xs, ys; mask = mask)),
        ("spherical", FG.Grids.StructuredGrid(sph, λ, φ)),
        # Three rotation components, which exercises `rotation_terms` and the per-pair corner
        # machinery at the arity 2-D leaves untested.
        ("3d", FG.Grids.StructuredGrid(cart, range(0.0, 1.0; length = 8),
                                       range(0.0, 0.9; length = 7), zs)),
    )
end

"""
Solvers to gate a decomposition on.

`AutoSolver` is here because it is what a caller gets, and it resolves to a different concrete
solver per grid — the transform on a periodic or bounded uniform grid, conjugate gradients on a
masked one. Gating `CGSolver` alone leaves every transform solver unmeasured, and a transform that
plans inside `solve_poisson!` allocates its whole spectral working set on each of the `P + 1`
solves.
"""
_solvers() = (
    ("jacobi", HD.CGSolver(; multigrid = false)),
    ("mg", HD.CGSolver(; multigrid = true)),
    ("auto", HD.AutoSolver()),
)

Test.@testset "allocations: operators" begin
    for (name, grid) in _grids(), bc in (HD.Neumann(), HD.Dirichlet())
        T = Float64
        N = ndims(grid)
        fm = HD.face_metrics(grid, bc)
        v = HD.allocate_faces(T, grid)
        gχ = HD.allocate_faces(T, grid)
        urot = HD.allocate_faces(T, grid)
        W = HD.allocate_corners(T, grid)
        R = HD.allocate_corners(T, grid)
        δ = zeros(T, size(grid))
        χ = zeros(T, size(grid))
        uc = zeros(T, size(grid)..., N)
        Random.seed!(4)
        foreach(a -> a .= randn.(), v)
        χ .= randn.()

        # `fm` is passed, as the plan does. Omitting it takes the default argument, which rebuilds
        # every face's area and gap.
        Test.@testset "$name/$(nameof(typeof(bc)))" begin
            Test.@test gate5(HD.divergence!, δ, v, grid, bc, fm) == 0
            Test.@test gate5(HD.gradient!, gχ, χ, grid, bc, fm) == 0
            Test.@test gate5(HD.curl!, W, v, grid, bc, fm) == 0
            Test.@test gate5(HD.rotational_velocity!, urot, R, grid, bc, fm) == 0
            Test.@test gate5(HD.to_faces!, v, uc, grid, bc, fm) == 0
            Test.@test gate5(HD.to_centres!, uc, v, grid, bc, fm) == 0
        end
    end
end

Test.@testset "allocations: solver inner loop" begin
    for (name, grid) in _grids()
        bc = HD.Neumann()
        T = Float64
        c = HD.laplacian_coefficients(grid, bc)
        x = zeros(T, size(grid))
        y = zeros(T, size(grid))
        z = zeros(T, size(grid))
        Random.seed!(5)
        x .= randn.()

        Test.@testset "$name" begin
            # `L` itself, the innermost operation in the package.
            Test.@test gate(HD.apply_laplacian!, y, x, grid, c) == 0
            # The three reductions the conjugate-gradient iteration evaluates every step.
            Test.@test gate(HD._dot_measure, x, y, grid, c) == 0
            f_proj(a, b, d, e) = HD.project_out_constant!(a, d, e)
            Test.@test gate(f_proj, x, y, grid, c) == 0
            f_jac(a, b, d, e) = HD._jacobi_precondition!(a, b, d, e)
            Test.@test gate(f_jac, z, x, grid, c) == 0
        end
    end
end

Test.@testset "allocations: decomposition" begin
    for (name, grid) in _grids(), mg in (false, true)
        bc = HD.Neumann()
        N = ndims(grid)
        solver = mg ? HD.CGSolver(; multigrid = true) : HD.CGSolver()
        plan = HD.plan_helmholtz(grid; boundary = bc, solver = solver,
                                 backend = CB.SerialBackend())
        ws = HD.allocate_workspace(plan)
        res = HD.allocate_result(plan)
        Random.seed!(6)
        u = randn(size(grid)..., N)

        # A whole decomposition, buffers preallocated: the solver's own iteration count varies with
        # the grid, so this gates the *rate* — nothing may be allocated per iteration.
        f_dec(a, b, c, d) = HD.helmholtz_decompose!(a, b, c, d)
        Test.@testset "$name/$(mg ? "mg" : "jacobi")" begin
            Test.@test gate(f_dec, res, u, plan, ws) == 0
        end
    end
end

Test.@testset "allocations: norms" begin
    for (name, grid) in _grids()
        T = Float64
        N = ndims(grid)
        U = randn(size(grid)..., N)
        f_norm(a, b, c, d) = HD.velocity_norm(a, b)
        Test.@testset "$name" begin
            Test.@test gate(f_norm, U, grid, nothing, nothing) == 0
        end
    end
end
