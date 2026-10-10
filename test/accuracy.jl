"""
Accuracy on flows whose decomposition is known in closed form.

The other suites check that the operators satisfy their identities and that solvers refuse what
they cannot do. Neither would notice an operator that is self-consistent and consistently wrong,
which is what these catch: a field built from an analytic `ψ` or `χ` must come back with the other
part at zero.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using FlowTransformBindings: FlowTransformBindings as FTB
using FFTW: FFTW
using Random: Random

include(joinpath(@__DIR__, "..", "reference_flows", "reference_flows.jl"))
using .ReferenceFlows: ReferenceFlows

const CARTA = FG.Geometry.CartesianGeometry{Float64}()
nrm(a) = sqrt(sum(abs2, a))

Test.@testset "spectral path is exact on its own basis" begin
    n = 32; L = 1.0
    xr = range(0.0, L - L / n; length = n)
    grid = FG.Grids.StructuredGrid(CARTA, xr, xr; topology = (true, true), period = (L, L))
    k = 2π / L

    # u = ∇χ with χ = sin(kx)sin(ky): curl-free, so the rotational part must vanish.
    ug = [k * cos(k * xr[i]) * sin(k * xr[j]) for i in 1:n, j in 1:n]
    vg = [k * sin(k * xr[i]) * cos(k * xr[j]) for i in 1:n, j in 1:n]
    Ug = cat(ug, vg; dims = 3)
    rg = HD.helmholtz_decompose_spectral(Ug, grid)
    Test.@test nrm(rg.u_rot) / nrm(Ug) < 1e-12
    Test.@test rg.harmonic_fraction < 1e-12

    # u = ∇⊥ψ: divergence-free, so the divergent part must vanish.
    ur = [-k * sin(k * xr[i]) * cos(k * xr[j]) for i in 1:n, j in 1:n]
    vr = [k * cos(k * xr[i]) * sin(k * xr[j]) for i in 1:n, j in 1:n]
    Ur = cat(ur, vr; dims = 3)
    rr = HD.helmholtz_decompose_spectral(Ur, grid)
    Test.@test nrm(rr.u_div) / nrm(Ur) < 1e-12
    Test.@test maximum(abs.(rr.u_rot .+ rr.u_div .+ rr.u_harm .- Ur)) < 1e-12
end

# White noise fills every Nyquist slot. The first derivative of a real field is zero at an even axis's
# Nyquist slot (Trefethen, *Spectral Methods in MATLAB*, ch. 3), and under that derivative `u_rot` is
# divergence-free, `u_div` curl-free and the two orthogonal. Checked by full-length FFTs.
Test.@testset "spectral path is exact at the Nyquist slots" begin
    rng = Random.Xoshiro(3)
    for dims in ((8, 6), (7, 6), (8, 5), (7, 5), (6, 4, 5))
        N = length(dims)
        Ls = ntuple(d -> 1.0 + d / 4, N)
        xs = ntuple(d -> range(0.0, Ls[d] - Ls[d] / dims[d]; length = dims[d]), N)
        grid = FG.Grids.StructuredGrid(CARTA, xs...; topology = ntuple(_ -> true, N), period = Ls)
        U = randn(rng, dims..., N)
        res = HD.helmholtz_decompose_spectral(U, grid)
        K = FTB.wavenumber_arrays(zeros(dims), Float64, FTB.wavenumber_axes(Ls, dims, Val(false));
                                  derivative = true)
        comp(A, c) = selectdim(A, N + 1, c)
        ûr = [FFTW.fft(comp(res.u_rot, c)) for c in 1:N]
        ûd = [FFTW.fft(comp(res.u_div, c)) for c in 1:N]
        scale = maximum(abs, FFTW.fft(comp(U, 1))) * maximum(k -> maximum(abs, k), K)
        Test.@test maximum(abs, sum(K[c] .* ûr[c] for c in 1:N)) < 1e-12 * scale
        for a in 1:N, b in (a + 1):N
            Test.@test maximum(abs, K[a] .* ûd[b] .- K[b] .* ûd[a]) < 1e-12 * scale
        end
        Test.@test abs(sum(res.u_rot .* res.u_div)) < 1e-12 * sum(abs2, U)
        # The harmonic part is the mean alone; a mode at Nyquist on every axis has a zero derivative
        # symbol and stays in the rotational part.
        V̂ = cat((FFTW.rfft(comp(U, c)) for c in 1:N)...; dims = N + 1)
        p = HD.helmholtz_project_spectral(V̂, grid)
        mean_slot = ntuple(_ -> 1, N)
        Test.@test p.u_harm[mean_slot..., :] == V̂[mean_slot..., :]
        Test.@test count(!iszero, p.u_harm) == count(!iszero, V̂[mean_slot..., :])
        Test.@test p.u_rot .+ p.u_div .+ p.u_harm ≈ V̂ rtol = 1e-14
    end
end

Test.@testset "Taylor-Green, an independent analytic field" begin
    n = 32; L = 1.0
    xr = range(0.0, L - L / n; length = n)
    grid = FG.Grids.StructuredGrid(CARTA, xr, xr; topology = (true, true), period = (L, L))
    u, v, _, _, ud, vd = ReferenceFlows.taylor_green_vortex(grid)
    # The reference flow states its own answer: no divergent part at all.
    Test.@test nrm(ud) + nrm(vd) == 0
    U = cat(u, v; dims = 3)
    res = HD.helmholtz_decompose_spectral(U, grid)
    Test.@test nrm(res.u_div) / nrm(U) < 1e-12
end

Test.@testset "3-D ABC flow is solenoidal" begin
    n = 16; L = 2π
    xr = range(0.0, L - L / n; length = n)
    grid = FG.Grids.StructuredGrid(CARTA, xr, xr, xr;
                                   topology = (true, true, true), period = (L, L, L))
    U = zeros(n, n, n, 3)
    for k in 1:n, j in 1:n, i in 1:n
        x, y, z = xr[i], xr[j], xr[k]
        U[i, j, k, 1] = sin(z) + cos(y)
        U[i, j, k, 2] = sin(x) + cos(z)
        U[i, j, k, 3] = sin(y) + cos(x)
    end
    res = HD.helmholtz_decompose_spectral(U, grid)
    Test.@test nrm(res.u_div) / nrm(U) < 1e-10
    Test.@test maximum(abs.(res.u_rot .+ res.u_div .+ res.u_harm .- U)) < 1e-10
    Test.@test length(HD.vector_potential(res)) == 3
end

Test.@testset "circulation on an annulus is harmonic" begin
    # A pure vortex around a masked hole is both divergence-free and curl-free in the annulus, so
    # it belongs to neither part — it is the harmonic component, and it exists only because the
    # domain is multiply connected. This is the case the whole staggered/flux-form rebuild is for.
    n = 41
    xs = range(-1.0, 1.0; length = n)
    mask = trues(n, n)
    for j in 1:n, i in 1:n
        (xs[i]^2 + xs[j]^2) <= 0.3^2 && (mask[i, j] = false)
    end
    grid = FG.Grids.StructuredGrid(CARTA, xs, xs; mask = mask)
    Test.@test HD.count_holes(grid) == 1

    u, v = ReferenceFlows.harmonic_vortex(grid; Γ = 1.0)
    U = cat(u, v; dims = 3)
    for I in CartesianIndices((n, n))
        mask[I] || (U[I, 1] = 0.0; U[I, 2] = 0.0)
    end
    plan = HD.plan_helmholtz(grid; boundary = HD.Neumann())
    res = HD.helmholtz_decompose!(HD.allocate_result(plan), U, plan)
    Test.@test res.harmonic_fraction > 0.9
    Test.@test nrm(res.u_rot) / nrm(U) < 0.05
end
