# The staggered decomposition on every grid kind, over FlowGeometries' mimetic operators (whose own
# identities FlowGeometries tests): in two dimensions the rotation potential's operator is `curl ∘ rot`;
# the decomposition reproduces the field's divergence, flux through the edges included; its three face
# parts are orthogonal; and its harmonic part has the dimension of the domain's loops.

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using Random: Random

const CART = FG.Geometry.CartesianGeometry{Float64}()
const OPS = FG.Operators

# `Σ w · p q` over faces, `w = area · gap` with the operator's or the data's area.
function dot_faces(p, q, grid, area, gap)
    acc = 0.0
    for d in 1:ndims(grid), F in CartesianIndices(FG.Grids.face_dims(grid, d))
        acc += area[d][F] * gap[d][F] * p[d][F] * q[d][F]
    end
    return acc
end

# A mildly warped quadrilateral mesh, given its cell vertices; the centres are their averages.
function warped_grid(nx::Int, ny::Int)
    kx = [(i - 1) / nx + 0.06 * sin(π * (j - 1) / ny) / nx for i in 1:(nx + 1), j in 1:(ny + 1)]
    ky = [(j - 1) / ny + 0.05 * sin(2π * (i - 1) / nx) / ny for i in 1:(nx + 1), j in 1:(ny + 1)]
    avg(k) = [(k[i, j] + k[i + 1, j] + k[i, j + 1] + k[i + 1, j + 1]) / 4 for i in 1:nx, j in 1:ny]
    return FG.Grids.CurvilinearGrid(CART, avg(kx), avg(ky); corners = (kx, ky))
end

# (name, grid, condition, simply connected) for every grid kind.
function grid_kinds()
    xs = range(0.0, 1.0; length = 12)
    ys = range(0.0, 0.8; length = 9)
    stretched = cumsum(vcat(0.0, 0.05 .+ 0.03 .* sin.(range(0, 3π; length = 11))))
    hole = trues(12, 9); hole[5:7, 4:5] .= false
    xp = range(0.0, 1.0 - 1 / 12; length = 12)
    return [
        ("bounded, Neumann", FG.Grids.StructuredGrid(CART, xs, ys), OPS.Neumann(), true),
        ("bounded, Dirichlet", FG.Grids.StructuredGrid(CART, xs, ys), OPS.Dirichlet(), true),
        ("stretched", FG.Grids.StructuredGrid(CART, stretched, ys), OPS.Neumann(), true),
        ("channel", FG.Grids.StructuredGrid(CART, xp, ys; topology = (true, false), period = 1.0), OPS.Neumann(), false),
        ("masked hole", FG.Grids.StructuredGrid(CART, xs, ys; mask = hole), OPS.Neumann(), false),
        ("spherical lat-lon", FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), FG.Axes.PeriodicAxis(0.0, 2π, 16),
                                                      range(-π / 2 + π / 20, π / 2 - π / 20; length = 10)), OPS.Neumann(), false),
        ("3-D periodic", FG.Grids.StructuredGrid(CART, xp, range(0.0, 0.8 - 0.1; length = 8), range(0.0, 0.5 - 0.1; length = 5);
                                                 topology = (true, true, true), period = (1.0, 0.8, 0.5)), OPS.Neumann(), false),
        ("3-D bounded", FG.Grids.StructuredGrid(CART, xs, ys, range(0.0, 0.5; length = 6)), OPS.Neumann(), true),
        ("3-D spherical shell", FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), FG.Axes.PeriodicAxis(0.0, 2π, 12),
                                                        range(-1.0, 1.0; length = 6), range(1.0, 1.6; length = 4)),
         OPS.Neumann(), false),
        ("curvilinear, Neumann", warped_grid(11, 9), OPS.Neumann(), true),
        ("curvilinear, Dirichlet", warped_grid(11, 9), OPS.Dirichlet(), true),
    ]
end

Test.@testset "the rotation potential's operator is curl ∘ rot" begin
    # On every two-dimensional grid kind, on every active corner, with `R = 0` on the corners the dual
    # drops or masks.
    for (name, grid, bc, _) in grid_kinds()
        ndims(grid) == 2 || continue
        Test.@testset "$name" begin
            Random.seed!(11)
            plan = HD.plan_helmholtz(grid; boundary = bc, solver = HD.CGSolver(), backend = CB.SerialBackend())
            dg = plan.dual_grids[1]
            dact = FG.Grids.mask(dg)
            R = (randn(size(dg)) .* dact,)
            LR = OPS.apply_laplacian!(zeros(size(dg)), R[1], plan.dual_coefficients[1])
            uf = OPS.rotational_velocity!(OPS.allocate_faces(Float64, grid), R, plan.metrics)
            Wr = OPS.curl!(OPS.allocate_corners(Float64, grid), uf, plan.metrics)
            Test.@test maximum(abs, (LR .- Wr[1])[dact]) <= 1e-12 * maximum(abs, LR)
        end
    end
end

Test.@testset "the staggered decomposition" begin
    # On every grid kind, for a random field: the divergent part has the face velocity's divergence
    # over the data areas, flux through the edges included, and that is the divergence reported; under
    # `Neumann` it carries the face velocity on the edges, where the rotational part is zero; the three
    # face parts are orthogonal; on a simply connected domain under `Neumann` the harmonic part is zero.
    # In three dimensions each rotation component is solved with its own dual Laplacian under
    # `Dirichlet`, which is `curl ∘ rot` on a periodic Cartesian grid only; elsewhere `⟨u_rot, u_harm⟩`
    # and the harmonic part are not gated.
    for (name, grid, bc, simple) in grid_kinds()
        Test.@testset "$name" begin
            N = ndims(grid)
            act = FG.Grids.mask(grid)
            Random.seed!(31)
            u = randn(size(grid)..., N) .* act
            plan = HD.plan_helmholtz(grid; boundary = bc, solver = HD.CGSolver(), backend = CB.SerialBackend())
            ws = HD.allocate_workspace(plan)
            r = HD.helmholtz_decompose!(HD.allocate_result(plan), u, plan, ws)
            Test.@test r.χ_solve.converged && all(s -> s.converged, r.rot_solve)
            v, g, w, h = HD.face_velocity(ws), HD.face_divergent(ws), HD.face_rotational(ws), HD.face_harmonic(ws)
            fmD = OPS.face_metrics(grid, OPS.Dirichlet())
            Dv = OPS.divergence!(zeros(size(grid)), v, fmD)
            Dg = OPS.divergence!(zeros(size(grid)), g, fmD)
            Test.@test maximum(abs, (Dg .- Dv)[act]) <= 1e-9 * maximum(abs, Dv)
            Test.@test maximum(abs, (r.divergence .- Dv)[act]) <= 1e-9 * maximum(abs, Dv)
            if bc isa OPS.Neumann
                for d in 1:N
                    FG.Grids.isperiodic(grid, d) && continue
                    for side in (1, size(grid, d) + 1)
                        sel = ntuple(e -> e == d ? side : Colon(), N)
                        Test.@test g[d][sel...] == v[d][sel...]
                        Test.@test all(iszero, w[d][sel...])
                    end
                end
            end
            fm = plan.metrics
            scale = dot_faces(v, v, grid, fm.area_flux, fm.gap)
            Test.@test abs(dot_faces(g, w, grid, fm.area_flux, fm.gap)) <= 1e-12 * scale
            Test.@test abs(dot_faces(g, h, grid, fm.area_flux, fm.gap)) <= 1e-10 * scale
            exact_rot = N == 2 || all(d -> FG.Grids.isperiodic(grid, d), 1:N)
            exact_rot && Test.@test abs(dot_faces(w, h, grid, fm.area_flux, fm.gap)) <= 1e-10 * scale
            (exact_rot && simple && bc isa OPS.Neumann) && Test.@test r.harmonic_fraction <= 1e-9
        end
    end
end

Test.@testset "closed forms" begin
    n = 10
    xs = range(0.0, 1.0; length = n)
    run(u, grid; bc = OPS.Neumann()) =
        HD.helmholtz_decompose(u, grid; boundary = bc, solver = HD.CGSolver(), backend = CB.SerialBackend())

    # A uniform flow through a box is the gradient of `x`: wholly divergent, divergence zero in the
    # edge cells as everywhere (`∂χ/∂n = u·n̂`, Bhatia et al. 2013).
    u = zeros(n, n, 2); u[:, :, 1] .= 1.0
    r = run(u, FG.Grids.StructuredGrid(CART, xs, xs))
    Test.@test maximum(abs, r.u_div .- u) <= 1e-12
    Test.@test maximum(abs, r.divergence) <= 1e-9
    Test.@test r.harmonic_fraction <= 1e-12

    # A channel, periodic in `x` and walled in `y`, has one loop: a uniform `(U, V)` is the gradient of
    # `V y` plus the harmonic `(U, 0)`, harmonic fraction `|U| / |(U, V)|`.
    ch = FG.Grids.StructuredGrid(CART, FG.Axes.PeriodicAxis(0.0, 1.0, n), xs)
    U, V = 0.6, 0.8
    uc = zeros(n, n, 2); uc[:, :, 1] .= U; uc[:, :, 2] .= V
    rc = run(uc, ch)
    Test.@test abs(rc.harmonic_fraction - U / hypot(U, V)) <= 1e-12
    Test.@test maximum(abs, rc.u_div[:, :, 2] .- V) <= 1e-12
    Test.@test maximum(abs, rc.u_harm[:, :, 1] .- U) <= 1e-12

    # A zonal flow `u_east = C / cos φ` on a latitude band is divergence-free and curl-free, and the band
    # is an annulus: the field is its own harmonic part.
    band = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), FG.Axes.PeriodicAxis(0.0, 2π, 16),
                                   range(-1.0, 1.0; length = 7))
    ub = zeros(16, 7, 2)
    ub[:, :, 1] .= 0.7 ./ cos.(FG.Grids.coordinates(band, 2))'
    for bc in (OPS.Neumann(), OPS.Dirichlet())
        rb = run(ub, band; bc)
        Test.@test abs(rb.harmonic_fraction - 1) <= 1e-12
        Test.@test maximum(abs, rb.u_harm .- ub) <= 1e-12
    end

    # The collocated round trip has the symbol `cos²(kh/2)`: `(sin kx, 0)` on a periodic grid comes back
    # scaled by it, so `residual_fraction = sin²(kh/2)`.
    for (m, np) in ((1, 16), (3, 12))
        ax = FG.Axes.PeriodicAxis(0.0, 1.0, np)
        grid = FG.Grids.StructuredGrid(CART, ax, ax)
        us = zeros(np, np, 2)
        us[:, :, 1] .= sin.(2π * m .* collect(ax))
        rs = run(us, grid)
        Test.@test abs(rs.residual_fraction - sin(π * m / np)^2) <= 1e-10
        Test.@test rs.harmonic_fraction <= 1e-10
    end
end

Test.@testset "count_holes" begin
    n = 21
    xs = range(-1.0, 1.0; length = n)
    disk = [x^2 + y^2 > 0.3^2 for x in xs, y in xs]
    two = trues(n, n); two[4:6, 4:6] .= false; two[15:17, 15:17] .= false
    notch = trues(n, n); notch[1:3, 9:11] .= false             # open to the edge: not enclosed
    Test.@test HD.count_holes(FG.Grids.StructuredGrid(CART, xs, xs)) == 0
    Test.@test HD.count_holes(FG.Grids.StructuredGrid(CART, xs, xs; mask = disk)) == 1
    Test.@test HD.count_holes(FG.Grids.StructuredGrid(CART, xs, xs; mask = two)) == 2
    Test.@test HD.count_holes(FG.Grids.StructuredGrid(CART, xs, xs; mask = notch)) == 0
end
