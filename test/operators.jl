"""
Properties of the discrete operator family.

These are the statements the decomposition rests on, and each is checked as an identity rather
than to a tolerance chosen by eye: `D` is exactly `−G*`, `L = D G` is symmetric and negative
semidefinite, `curl ∘ G = 0`, and `div(u_div) = div(u)`. If any of them stops holding, the
decomposition stops being a decomposition — the parts cease to be orthogonal and the residue is
reported as `harmonic_fraction`, which is a wrong answer rather than an imprecise one.

Every grid architecture the package supports is exercised, because several of these held on a flat
unmasked grid while failing on a curved or masked one.
"""

using Test: @testset, @test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using Random: Random

const CART = FG.Geometry.CartesianGeometry{Float64}()

# The weight that makes D the negative adjoint of G is the face area times the face gap.
function dot_faces(p, q, grid, bc, ::Type{T}, ::Val{N}) where {T,N}
    acc = zero(T)
    for d in 1:N, F in CartesianIndices(HD.face_dims(grid, d))
        A = HD.face_area(grid, F, d, bc, T)
        iszero(A) && continue
        acc += A * HD.face_gap(grid, F, d, T) * p[d][F] * q[d][F]
    end
    return acc
end

function dot_cells(a, b, grid, ::Type{T}) where {T}
    acc = zero(T)
    for I in CartesianIndices(size(grid))
        FG.Grids.isactive(grid, Tuple(I)...) || continue
        acc += a[I] * b[I] * HD.cell_measure(grid, I, T)
    end
    return acc
end

rel(got, want) = abs(got - want) / max(abs(want), 1.0)

function operator_properties(grid, bc)
    T = Float64
    N = ndims(grid)
    dims = size(grid)
    Random.seed!(11)
    χ = randn(T, dims)
    ψ = randn(T, dims)
    v = HD.allocate_faces(T, grid)
    foreach(a -> copyto!(a, randn(T, size(a))), v)
    g = HD.allocate_faces(T, grid)
    Dv = zeros(T, dims)

    HD.gradient!(g, χ, grid, bc)
    HD.divergence!(Dv, v, grid, bc)
    # D = −G*, under the cell-measure inner product on one side and A·g on the other.
    @test rel(dot_faces(g, v, grid, bc, T, Val(N)), -dot_cells(χ, Dv, grid, T)) < 1e-10

    c = HD.laplacian_coefficients(grid, bc)
    Lχ = zeros(T, dims); HD.apply_laplacian!(Lχ, χ, grid, c)
    Lψ = zeros(T, dims); HD.apply_laplacian!(Lψ, ψ, grid, c)
    # The assembled coefficients must be the same operator as applying D and G in turn.
    scratch = HD.allocate_faces(T, grid)
    LDG = zeros(T, dims); HD.laplacian!(LDG, χ, grid, bc, scratch)
    @test sqrt(dot_cells(Lχ .- LDG, Lχ .- LDG, grid, T)) < 1e-8
    # Self-adjoint, and negative semidefinite.
    @test rel(dot_cells(Lχ, ψ, grid, T), dot_cells(χ, Lψ, grid, T)) < 1e-10
    @test dot_cells(Lχ, χ, grid, T) <= 1e-9

    # curl ∘ G = 0: the two parts are independent, not merely nearly so.
    if N >= 2
        W = HD.allocate_corners(T, grid)
        HD.curl!(W, g, grid, bc)
        @test sqrt(sum(a -> sum(abs2, a), W)) < 1e-8
    end
end

@testset "operator family" begin
    xs = range(0.0, 1.0; length = 12)
    ys = range(0.0, 0.8; length = 9)
    stretched = cumsum(vcat(0.0, 0.05 .+ 0.03 .* sin.(range(0, 3π; length = 11))))
    mask = trues(12, 9); mask[5:7, 4:5] .= false

    cases = [
        ("Cartesian bounded, Neumann",   FG.Grids.StructuredGrid(CART, xs, ys), HD.Neumann()),
        ("Cartesian bounded, Dirichlet", FG.Grids.StructuredGrid(CART, xs, ys), HD.Dirichlet()),
        ("stretched axis",               FG.Grids.StructuredGrid(CART, stretched, ys), HD.Neumann()),
        ("x-periodic",                   FG.Grids.StructuredGrid(CART, range(0.0, 1.0 - 1/12; length = 12), ys;
                                                                 topology = (true, false), period = 1.0), HD.Neumann()),
        ("masked hole",                  FG.Grids.StructuredGrid(CART, xs, ys; mask = mask), HD.Neumann()),
        ("spherical lat-lon",            FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0),
                                                                 range(0, 2π - 2π/16; length = 16),
                                                                 range(-π/2 + π/20, π/2 - π/20; length = 10)), HD.Neumann()),
        ("Cartesian 3-D",                FG.Grids.StructuredGrid(CART, xs, ys, range(0.0, 0.5; length = 6)), HD.Neumann()),
    ]
    for (name, grid, bc) in cases
        @testset "$name" begin
            operator_properties(grid, bc)
        end
    end
end

@testset "dual Laplacian is curl of the codifferential" begin
    # The rotation potential's operator must be the one the reconstruction composes, or the solve
    # inverts something else — which showed up as a rotational part that would not converge.
    n = 16; L = 1.0
    xr = range(0.0, L - L/n; length = n)
    grid = FG.Grids.StructuredGrid(CART, xr, xr; topology = (true, true), period = (L, L))
    bc = HD.Neumann()
    plan = HD.plan_helmholtz(grid; boundary = bc)
    Random.seed!(5)
    R = (randn(size(plan.dual_grids[1])),)
    LA = zeros(size(plan.dual_grids[1]))
    HD.apply_laplacian!(LA, R[1], plan.dual_grids[1], plan.dual_coefficients[1])
    uf = HD.allocate_faces(Float64, grid); HD.rotational_velocity!(uf, R, grid, bc)
    W = HD.allocate_corners(Float64, grid); HD.curl!(W, uf, grid, bc)
    @test sqrt(sum(abs2, LA .- W[1])) / sqrt(sum(abs2, LA)) < 1e-12
end

@testset "decomposition identities" begin
    xs = range(0.0, 1.0; length = 14)
    ys = range(0.0, 0.8; length = 11)
    mask = trues(14, 11); mask[6:8, 5:6] .= false
    for (name, grid, bc) in [
        ("bounded Neumann",   FG.Grids.StructuredGrid(CART, xs, ys), HD.Neumann()),
        ("bounded Dirichlet", FG.Grids.StructuredGrid(CART, xs, ys), HD.Dirichlet()),
        ("masked hole",       FG.Grids.StructuredGrid(CART, xs, ys; mask = mask), HD.Neumann()),
    ]
        @testset "$name" begin
            T = Float64; N = ndims(grid); dims = size(grid)
            Random.seed!(3)
            u = randn(T, dims..., N)
            plan = HD.plan_helmholtz(grid; boundary = bc)
            ws = HD.allocate_workspace(plan)
            res = HD.helmholtz_decompose!(HD.allocate_result(plan), u, plan, ws)

            δu = zeros(T, dims); HD.divergence!(δu, ws.v, grid, bc)
            plan.coefficients.singular && HD.project_out_constant!(δu, grid, plan.coefficients)
            δd = zeros(T, dims); HD.divergence!(δd, ws.gχ, grid, bc)
            # The divergent part reproduces the divergence it was built from.
            @test sqrt(sum(abs2, δd .- δu)) < 1e-7
            δr = zeros(T, dims); HD.divergence!(δr, ws.urot, grid, bc)
            @test sqrt(sum(abs2, δr)) < 1e-7
            @test res.χ_solve.converged
        end
    end
end

@testset "count_holes" begin
    n = 41
    xs = range(-1.0, 1.0; length = n)
    disk = trues(n, n)
    for j in 1:n, i in 1:n
        (xs[i]^2 + xs[j]^2) <= 0.3^2 && (disk[i, j] = false)
    end
    two = trues(n, n); two[8:12, 8:12] .= false; two[28:32, 28:32] .= false
    notch = trues(n, n); notch[1:5, 18:22] .= false
    @test HD.count_holes(FG.Grids.StructuredGrid(CART, xs, xs)) == 0
    @test HD.count_holes(FG.Grids.StructuredGrid(CART, xs, xs; mask = disk)) == 1
    @test HD.count_holes(FG.Grids.StructuredGrid(CART, xs, xs; mask = two)) == 2
    # A notch open to the edge is not enclosed, so it is not a hole.
    @test HD.count_holes(FG.Grids.StructuredGrid(CART, xs, xs; mask = notch)) == 0
end
