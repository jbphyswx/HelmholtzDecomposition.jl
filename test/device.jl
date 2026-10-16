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

# Every `Array` or `BitArray` reachable from `x`, with the field path that reached it. FlowGeometries'
# lazy wrappers are walked into; any other array type is device storage and a leaf; a foreign
# transform plan owns memory this package does not allocate.
function host_arrays(x)
    out = Tuple{String,Type}[]
    _walk!(out, Base.IdSet{Any}(), x, "")
    return out
end

function _walk!(out, seen, x, path)
    (x === nothing || isbits(x) || x in seen) && return nothing
    push!(seen, x)
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
    for dims in ((10, 10), (6, 6, 6)), (sname, solver) in (("jacobi", HD.CGSolver(; multigrid = false)),
                                                           ("multigrid", HD.CGSolver(; multigrid = true)))
        N = length(dims)
        grid = FG.Grids.StructuredGrid(CART, ntuple(d -> range(0.0, 1.0; length = dims[d]), N)...)
        fields = [randn(dims..., N) for _ in 1:2]
        hplan = HD.plan_helmholtz(grid; solver, backend = CB.SerialBackend())
        href = HD.helmholtz_decompose_batch(hplan, fields; backend = CB.SerialBackend())
        for (bname, backend, move) in (("JLArrays", CB.GPUBackend(JLArrays.JLBackend()), JLArrays.JLArray),
                                       ("KA.CPU", CB.GPUBackend(KA.CPU()), identity))
            Test.@testset "$N-D $sname on $bname" begin
                plan = HD.plan_helmholtz(grid; solver, backend)
                if bname == "JLArrays"
                    for obj in (plan, HD.allocate_workspace(plan), HD.allocate_result(plan))
                        hits = DeviceWalk.host_arrays(obj)
                        isempty(hits) || println("  host arrays reachable: ", first(hits, 6))
                        Test.@test isempty(hits)
                    end
                end
                b = HD.helmholtz_decompose_batch(plan, map(move, fields); backend = CB.SerialBackend())
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

Test.@testset "the periodic spectral result is assembled on the device" begin
    # Every step after the forward transform runs on JLArrays; the transforms are the host FFT wrapped,
    # since JLArrays has none.
    n = 12
    xr = range(0.0, 1.0 - 1 / n; length = n)
    grid = FG.Grids.StructuredGrid(CART, xr, xr; topology = (true, true), period = (1.0, 1.0))
    Random.seed!(25)
    U = randn(n, n, 2)
    ks = FTB.wavenumber_axes((1.0, 1.0), (n, n), Val(true))
    V̂ = cat(FFTW.rfft(U[:, :, 1]), FFTW.rfft(U[:, :, 2]); dims = 3)
    href = HD.build_cartesian_result(grid, U, V̂, ks, x -> FFTW.irfft(x, n))
    dres = HD.build_cartesian_result(grid, JLArrays.JLArray(U), JLArrays.JLArray(V̂), ks,
                                     x -> JLArrays.JLArray(FFTW.irfft(Array(x), n)))
    Test.@test dres.u_rot isa JLArrays.JLArray
    Test.@test Array(dres.u_rot) ≈ href.u_rot
    Test.@test Array(dres.u_div) ≈ href.u_div
    Test.@test Array(dres.χ) ≈ href.χ
    Test.@test dres.harmonic_fraction ≈ href.harmonic_fraction
end
