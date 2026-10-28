"""
Device execution: a decomposition planned on a device backend holds no host memory, equals the serial
host result, keeps its batch diagnostics on the host, and the periodic spectral result is assembled
on the array's own device.

`GPUBackend(KernelAbstractions.CPU())` runs the kernels through KernelAbstractions over plain `Array`s;
`JLArray` is a distinct array type over host memory with scalar indexing disallowed, so a buffer built
with `zeros` where the backend's allocator belongs is found by the walk below.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using JLArrays: JLArrays
using FFTW: FFTW
using FlowTransformBindings: FlowTransformBindings as FTB
using Random: Random

# `JLArrays` defines no `KernelAbstractions.synchronize` for its backend; its kernels have completed by
# the time a launch returns.
KA.synchronize(::JLArrays.JLBackend) = nothing

module DeviceWalk

# Every `Array` or `BitArray` of values reachable from `x`, with the field path that reached it. An
# `Array` of structs (a list of multigrid levels) is walked into; FlowGeometries' lazy wrappers are walked
# into; any other array type is device storage and a leaf; a foreign transform plan owns memory this
# package does not allocate. A `ComponentNullspace` keeps `offsets` and `totals` on the host by contract.
using FlowGeometries: FlowGeometries as FG

function host_arrays(x)
    out = Tuple{String,Type}[]
    _walk!(out, Base.IdSet{Any}(), x, "")
    return out
end

function _walk!(out, seen, x, path)
    (x === nothing || isbits(x) || x in seen) && return nothing
    push!(seen, x)
    x isa FG.Operators.ComponentNullspace && return _walk!(out, seen, x.perm, path * ".perm")
    if x isa Array && !isbitstype(eltype(x))
        foreach(i -> _walk!(out, seen, x[i], string(path, "[", i, "]")), eachindex(x))
        return nothing
    end
    if x isa AbstractArray
        (x isa Array || x isa BitArray) && return (push!(out, (path, typeof(x))); nothing)
        (x isa SubArray || x isa Base.ReshapedArray) && return _walk!(out, seen, parent(x), path * "→parent")
        startswith(string(parentmodule(typeof(x))), "FlowGeometries") || return nothing
    end
    if x isa Tuple || x isa NamedTuple
        foreach(((k, v),) -> _walk!(out, seen, v, string(path, ".", k)), pairs(x))
        return nothing
    end
    T = typeof(x)
    foreign_plan = occursin("Plan", string(nameof(T))) &&
                   !startswith(string(parentmodule(T)), "HelmholtzDecomposition")
    (isstructtype(T) && !foreign_plan) || return nothing
    for f in fieldnames(T)
        isdefined(x, f) && _walk!(out, seen, getfield(x, f), string(path, ".", f))
    end
    return nothing
end

end # module DeviceWalk

const CART = FG.Geometry.CartesianGeometry{Float64}()

Test.@testset "device parity with the serial host path" begin
    Random.seed!(21)
    for dims in ((10, 10), (6, 6, 6)),
        (sname, solver) in (("jacobi", HD.CGSolver(; preconditioner = HD.DiagonalScaling())),
                            ("multigrid", HD.CGSolver(; preconditioner = HD.MultigridCycle())))
        N = length(dims)
        grid = FG.Grids.StructuredGrid(CART, ntuple(d -> range(0.0, 1.0; length = dims[d]), N)...)
        fields = [randn(dims..., N) for _ in 1:2]
        hplan = HD.plan_helmholtz(grid; solver, execution = CB.SerialBackend())
        href = HD.helmholtz_decompose_batch(hplan, fields; execution = CB.SerialBackend())
        for (bname, execution, move) in (("JLArrays", CB.GPUBackend(JLArrays.JLBackend()), JLArrays.JLArray),
                                         ("KA.CPU", CB.GPUBackend(KA.CPU()), identity))
            Test.@testset "$N-D $sname on $bname" begin
                plan = HD.plan_helmholtz(grid; solver, execution)
                if bname == "JLArrays"
                    for obj in (plan, HD.allocate_workspace(plan), HD.allocate_result(plan))
                        hits = DeviceWalk.host_arrays(obj)
                        isempty(hits) || println("  host arrays reachable: ", first(hits, 6))
                        Test.@test isempty(hits)
                    end
                end
                # The batch axis serial, or on the plan's device.
                for batch_execution in (CB.SerialBackend(), execution)
                    b = HD.helmholtz_decompose_batch(plan, map(move, fields); execution = batch_execution)
                    Test.@test b.harmonic_fraction isa Vector{Float64} && b.residual_fraction isa Vector{Float64}
                    Test.@test Array(b.χ) ≈ href.χ
                    Test.@test Array(b.u_rot) ≈ href.u_rot
                    Test.@test Array(b.u_div) ≈ href.u_div
                    Test.@test b.harmonic_fraction ≈ href.harmonic_fraction atol = 1e-12
                    Test.@test b.residual_fraction ≈ href.residual_fraction
                end
            end
        end
    end
end

Test.@testset "AutoSolver on a periodic grid, on each device backend" begin
    # Whatever solver each backend's arrays take, the decomposition is the serial host one, which the FFT
    # solves exactly, to conjugate gradients' accuracy: `κ(L) · 100eps`, `κ(L) ≈ 30` on this grid.
    n = 12
    ax = FG.Axes.PeriodicUniformAxis(0.0, 1.0, n)
    grid = FG.Grids.StructuredGrid(CART, ax, ax)
    Random.seed!(23)
    fields = [randn(n, n, 2) for _ in 1:2]
    href = HD.helmholtz_decompose_batch(HD.plan_helmholtz(grid; execution = CB.SerialBackend()), fields;
                                        execution = CB.SerialBackend())
    for (bname, execution, move) in (("JLArrays", CB.GPUBackend(JLArrays.JLBackend()), JLArrays.JLArray),
                                     ("KA.CPU", CB.GPUBackend(KA.CPU()), identity))
        Test.@testset "$bname" begin
            plan = HD.plan_helmholtz(grid; execution)
            b = HD.helmholtz_decompose_batch(plan, map(move, fields); execution)
            Test.@test isapprox(Array(b.χ), href.χ; rtol = 1e-10)
            Test.@test isapprox(Array(b.u_rot), href.u_rot; rtol = 1e-10)
            Test.@test isapprox(Array(b.u_div), href.u_div; rtol = 1e-10)
            Test.@test b.harmonic_fraction ≈ href.harmonic_fraction atol = 1e-10
        end
    end
end

Test.@testset "the periodic spectral result is assembled on the device" begin
    # Every step after the forward transform runs on JLArrays; the inverse transforms are the host FFT
    # wrapped, since JLArrays has none.
    n = 12
    xr = range(0.0, 1.0 - 1 / n; length = n)
    grid = FG.Grids.StructuredGrid(CART, xr, xr; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (1.0, 1.0))
    Random.seed!(25)
    U = randn(n, n, 2)
    ks = FTB.wavenumber_axes(Float64, (1.0, 1.0), (n, n))
    V̂ = cat(FFTW.rfft(U[:, :, 1]), FFTW.rfft(U[:, :, 2]); dims = 3)
    assemble(U, V̂, invert!) = HD._cartesian_result!(HD._spectral_result(U, Float64, (n, n), Val(2), Val(1)), U, V̂,
                                                     HD._derivative_symbol(V̂, Float64, ks),
                                                     similar(V̂, size(V̂)[1:2]), similar(V̂, size(V̂)[1:2]),
                                                     1 / n^2, invert!)
    href = assemble(U, V̂, (out, x) -> copyto!(out, FFTW.brfft(x, n)))
    dres = assemble(JLArrays.JLArray(U), JLArrays.JLArray(V̂),
                    (out, x) -> copyto!(out, JLArrays.JLArray(FFTW.brfft(Array(x), n))))
    Test.@test dres.u_rot isa JLArrays.JLArray
    Test.@test Array(dres.u_rot) ≈ href.u_rot
    Test.@test Array(dres.u_div) ≈ href.u_div
    Test.@test Array(dres.χ) ≈ href.χ
    Test.@test dres.harmonic_fraction ≈ href.harmonic_fraction
end
