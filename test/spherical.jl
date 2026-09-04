"""
Spherical spectral decomposition.

The pre-migration suite covered these and the rebuild dropped them along with the capability, so
they are regression tests in the strict sense: each fails on a package where
`_decompose_spectral` has no spherical method.

The split is exact. The tangent field `V = u_θ + i u_φ` has spin weight 1; the symmetric and
antisymmetric combinations of its spin(+1) and spin(−1) coefficients are the rotational and
divergent parts. Both manufactured fields below therefore have a known answer that holds to the
transform's own tolerance, with no discretisation order to allow for.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads
using Random: Random

const SPH = FG.Geometry.SphericalGeometry(1.0)

# Extensions are reached through `Base.get_extension`; they are not properties of the module.
nufsht_ext() = Base.get_extension(HD, :HelmholtzDecompositionNUFSHTExt)
fsh_ext() = Base.get_extension(HD, :HelmholtzDecompositionFSHExt)

nrm(A) = sqrt(sum(abs2, A))

# A longitude–latitude grid whose cells tile the sphere.
function latlon(nlat::Int)
    nlon = 2nlat
    λ = range(0.0, 2π - 2π / nlon; length = nlon)
    φ = range(-π / 2 + π / (2nlat), π / 2 - π / (2nlat); length = nlat)
    return FG.Grids.StructuredGrid(SPH, λ, φ; topology = (true, false), period = (2π, 0.0))
end

# Solid-body rotation about the pole: `u_east = cos φ`, `u_north = 0`. Divergence-free.
function solid_body(grid)
    nlon, nlat = size(grid)
    φ = FG.Grids.coordinates(grid, 2)
    U = zeros(nlon, nlat, 2)
    @inbounds for j in 1:nlat, i in 1:nlon
        U[i, j, 1] = cos(φ[j])
    end
    return U
end

# The gradient of `sin φ` on the unit sphere: `u_north = ∂φ(sin φ) = cos φ`. Curl-free.
function pure_gradient(grid)
    nlon, nlat = size(grid)
    φ = FG.Grids.coordinates(grid, 2)
    U = zeros(nlon, nlat, 2)
    @inbounds for j in 1:nlat, i in 1:nlon
        U[i, j, 2] = cos(φ[j])
    end
    return U
end

Test.@testset "spherical spectral Hodge" begin
    grid = latlon(20)

    Test.@testset "solid-body rotation is rotational" begin
        U = solid_body(grid)
        r = HD.helmholtz_decompose_spectral(U, grid)
        Test.@test nrm(r.u_rot) / nrm(U) > 0.99
        Test.@test nrm(r.u_div) / nrm(U) < 1e-2
        Test.@test nrm(r.u_rot .+ r.u_div .+ r.u_harm .- U) / nrm(U) < 1e-8
    end

    Test.@testset "a gradient is divergent" begin
        U = pure_gradient(grid)
        r = HD.helmholtz_decompose_spectral(U, grid)
        Test.@test nrm(r.u_div) / nrm(U) > 0.99
        Test.@test nrm(r.u_rot) / nrm(U) < 1e-2
        Test.@test nrm(r.u_rot .+ r.u_div .+ r.u_harm .- U) / nrm(U) < 1e-8
    end
end

Test.@testset "Clenshaw–Curtis spectral Hodge" begin
    nlat = 24
    ax = FG.SphericalSampling.spherical_axes(Float64, FG.SphericalSampling.ClenshawCurtisSampling(),
                                             nlat)
    cc = FG.Grids.StructuredGrid(SPH, ax.λ, ax.φ)
    nlon = length(ax.λ)
    φ = ax.φ

    # Solid-body rotation is divergence-free; the gradient of `sin φ` is curl-free.
    rot = zeros(nlon, nlat, 2)
    grad = zeros(nlon, nlat, 2)
    for j in 1:nlat, i in 1:nlon
        rot[i, j, 1] = cos(φ[j])
        grad[i, j, 2] = cos(φ[j])
    end

    fsh = fsh_ext().SphericalSpectralSolver()
    Test.@testset "rotational" begin
        r = HD.helmholtz_decompose_spectral(rot, cc; solver = fsh)
        Test.@test nrm(r.u_div) / nrm(rot) < 1e-8
        Test.@test nrm(r.u_rot .+ r.u_div .+ r.u_harm .- rot) / nrm(rot) < 1e-10
    end
    Test.@testset "divergent" begin
        r = HD.helmholtz_decompose_spectral(grad, cc; solver = fsh)
        Test.@test nrm(r.u_div) / nrm(grad) > 0.99
        Test.@test nrm(r.u_rot) / nrm(grad) < 1e-8
        Test.@test nrm(r.u_rot .+ r.u_div .+ r.u_harm .- grad) / nrm(grad) < 1e-10
    end

    # `rot` and `grad` are even in `φ` and independent of `λ`, so reading the latitude axis
    # backwards maps each into its own class and neither can see a row-order error. Reflection
    # carries these off the gradients, because the `λ` derivative keeps its sign while the `φ` one
    # changes.
    Test.@testset "latitude order" begin
        λ = ax.λ
        gmix = zeros(nlon, nlat, 2)
        rmix = zeros(nlon, nlat, 2)
        for j in 1:nlat, i in 1:nlon
            gmix[i, j, 1] = -sin(φ[j]) * sin(λ[i])      # ∇χ with χ = ½ sin2φ cos λ
            gmix[i, j, 2] = cos(2φ[j]) * cos(λ[i])
            rmix[i, j, 1] = -cos(2φ[j]) * cos(λ[i])     # k̂ × ∇ψ with the same ψ
            rmix[i, j, 2] = -sin(φ[j]) * sin(λ[i])
        end
        a = HD.helmholtz_decompose_spectral(gmix, cc; solver = fsh)
        Test.@test nrm(a.u_rot) / nrm(gmix) < 1e-9
        b = HD.helmholtz_decompose_spectral(rmix, cc; solver = fsh)
        Test.@test nrm(b.u_div) / nrm(rmix) < 1e-9
    end

    # Two independent implementations — exact quadrature and a spin±1 least-squares fit — on the
    # same nodes and the same field.
    Test.@testset "agrees with the non-uniform transform" begin
        mixed = rot .+ grad
        a = HD.helmholtz_decompose_spectral(mixed, cc; solver = fsh)
        nu = nufsht_ext().SphericalNUSHTSolver(; lmax = nlat - 1, tol = 1e-12, rtol = 1e-11)
        b = HD.helmholtz_decompose_spectral(mixed, cc; solver = nu)
        Test.@test nrm(a.u_div .- b.u_div) / nrm(mixed) < 1e-6
    end
end

Test.@testset "quadrature samplings resolve and split" begin
    # Gauss–Legendre and Driscoll–Healy are the samplings whose quadrature is exact at the stated
    # band limit. `FastSphericalHarmonics` implements neither, so they resolve to the non-uniform
    # fit; both cover the sphere, so that resolution is a solver. Solid-body rotation is
    # divergence-free, so everything reported as divergent is artifact.
    nlat = 24
    for s in (FG.SphericalSampling.GaussLegendreSampling(),
              FG.SphericalSampling.DriscollHealySampling())
        Test.@test FG.SphericalSampling.admits_exact_bandlimited_quadrature(s)
        ax = FG.SphericalSampling.spherical_axes(Float64, s, nlat)
        g = FG.Grids.StructuredGrid(SPH, ax.λ, ax.φ)
        nlon = length(ax.λ)
        Test.@test HD.select_solver(HD.AutoSolver(), g, HD.Neumann()) isa
                   nufsht_ext().SphericalNUSHTSolver

        U = zeros(nlon, nlat, 2)
        for j in 1:nlat, i in 1:nlon
            U[i, j, 1] = cos(ax.φ[j])
        end
        r = HD.helmholtz_decompose_spectral(U, g)
        Test.@test nrm(r.u_div) / nrm(U) < 1e-10
        Test.@test nrm(r.u_rot .+ r.u_div .+ r.u_harm .- U) / nrm(U) < 1e-10
    end
end

Test.@testset "spherical coverage" begin
    nlat = 16
    nlon = 2nlat
    λ = range(0.0, 2π - 2π / nlon; length = nlon)

    # A ±74.5° band is a regional patch: analysis integrates over S² and the inverse Laplacian
    # there is nonlocal, so the band does not determine the solution even on the band.
    band = FG.Grids.StructuredGrid(SPH, λ, range(-1.3, 1.3; length = nlat);
                                   topology = (true, false), period = (2π, 0.0))
    nu = nufsht_ext().SphericalNUSHTSolver()
    Test.@test !HD.supports_sampling(nu, band)
    Test.@test_throws ArgumentError HD.select_solver(nu, band, HD.Neumann())
    Test.@test HD.select_solver(nu, latlon(nlat), HD.Neumann()) === nu
end

Test.@testset "Clenshaw–Curtis sampling is read from the grid" begin
    nlat = 16
    ax = FG.SphericalSampling.spherical_axes(Float64, FG.SphericalSampling.ClenshawCurtisSampling(),
                                             nlat)
    cc = FG.Grids.StructuredGrid(SPH, ax.λ, ax.φ)
    fsh = fsh_ext().SphericalSpectralSolver()
    Test.@test HD.supports_sampling(fsh, cc)

    # The node positions decide, and `N_λ = 2N_θ − 1` alone does not: an evenly spaced latitude
    # grid of the same shape returns a smooth, plausible, wrong field.
    shaped = FG.Grids.StructuredGrid(SPH, range(0, 2π - 2π / (2nlat - 1); length = 2nlat - 1),
                                     range(-1.2, 1.2; length = nlat))
    Test.@test size(shaped) == size(cc)
    Test.@test !HD.supports_sampling(fsh, shaped)
    Test.@test_throws ArgumentError HD.select_solver(fsh, shaped, HD.Neumann())
end

Test.@testset "a threaded batch on a Clenshaw–Curtis sphere matches serial" begin
    # `FastTransforms` returns a different result when its parallel region is entered from a
    # non-root Julia task, at any Julia thread count. `AutoSolver` resolves the primal solve on this
    # grid to that transform, so a batch reaches it from a worker task.
    #
    # The assertion is equality: both paths run the same arithmetic in the same order on the same
    # slices, and the transform pinned to one thread returns what the root task returns.
    nlat = 16
    ax = FG.SphericalSampling.spherical_axes(Float64, FG.SphericalSampling.ClenshawCurtisSampling(),
                                             nlat)
    cc = FG.Grids.StructuredGrid(SPH, ax.λ, ax.φ)
    nlon = length(ax.λ)

    plan = HD.plan_helmholtz(cc; boundary = HD.Neumann(), backend = CB.SerialBackend())
    # The exposure exists only if the primal solve is the Clenshaw–Curtis transform.
    Test.@test plan.solver isa fsh_ext().SphericalSpectralSolver

    Random.seed!(31)
    fields = [randn(nlon, nlat, 2) for _ in 1:6]
    bser = HD.helmholtz_decompose_batch(plan, fields; backend = CB.SerialBackend())
    bthr = HD.helmholtz_decompose_batch(plan, fields; backend = CB.ThreadedBackend())
    # A second serial batch, through its own workspaces: the same work with no threading anywhere.
    bctl = HD.helmholtz_decompose_batch(plan, fields; backend = CB.SerialBackend())

    # `χ` is the Clenshaw–Curtis transform, and it is asserted exactly — pinning FastTransforms to
    # one thread returns what the root task returns.
    Test.@test all(i -> bser[i].χ == bthr[i].χ, eachindex(fields))

    # `u_rot` and `u_div` come through the dual solve, whose non-uniform transform is not bitwise
    # reproducible once a fast backend is loaded. `bctl` measures that floor with no threading
    # involved, and threading may not exceed it.
    for f in (:u_rot, :u_div)
        scale = maximum(i -> maximum(abs, getfield(bser[i], f)), eachindex(fields))
        base = maximum(i -> maximum(abs, getfield(bser[i], f) .- getfield(bctl[i], f)),
                       eachindex(fields))
        thr = maximum(i -> maximum(abs, getfield(bser[i], f) .- getfield(bthr[i], f)),
                      eachindex(fields))
        Test.@test thr <= max(base, 8 * eps(Float64) * scale)
    end
end

Test.@testset "an unconverged spherical fit is refused" begin
    # One iteration cannot reach `rtol`, and a fit that stopped short synthesises a smooth field
    # that does not match the samples.
    grid = latlon(12)
    U = solid_body(grid)
    short = nufsht_ext().SphericalNUSHTSolver(; lmax = 8, rtol = 1e-14, maxiter = 1)
    Test.@test_throws ArgumentError HD.helmholtz_decompose_spectral(U, grid; solver = short)
end

Test.@testset "a degree past the node count warns and runs" begin
    # 288 nodes against 1680 coefficients. The fit reproduces the samples and the three parts sum
    # back to the input; the split does not survive, and the caller keeps the choice.
    grid = latlon(12)
    U = solid_body(grid)
    over = nufsht_ext().SphericalNUSHTSolver(; lmax = 40)
    r = Test.@test_logs (:warn,) match_mode = :any HD.helmholtz_decompose_spectral(U, grid;
                                                                                   solver = over)
    Test.@test nrm(r.u_rot .+ r.u_div .+ r.u_harm .- U) / nrm(U) < 1e-12
    Test.@test nrm(r.u_div) / nrm(U) > 0.1

    # Sized from the grid instead, the same field splits exactly.
    d = HD.helmholtz_decompose_spectral(U, grid; solver = nufsht_ext().SphericalNUSHTSolver())
    Test.@test nrm(d.u_div) / nrm(U) < 1e-10
end