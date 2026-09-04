"""
Device gate: every buffer a plan, workspace or result reaches lives where the backend runs.

`GPUBackend(KernelAbstractions.CPU())` allocates plain `Array`s. A host buffer and a device buffer
have the same type there, so a solver that builds its state with `zeros` passes a parity test
against the serial path while holding host memory a kernel cannot reach. `JLArray` is a distinct
array type over host memory, so the same walk over the same structures separates the two.

The walk records any `Array` or `BitArray` it reaches. `FlowGeometries`' lazy wrappers are recursed
into, since a `SeparableMeasure` holding host `Vector`s is host memory wearing an abstract type;
an array from another module is device storage and is a leaf.
"""

using Test: Test
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using JLArrays: JLArrays
using Random: Random

# `JLArrays` implements `get_backend`, `allocate`, `launch_config` and the `Adapt` rules for
# `JLBackend`, but not `KernelAbstractions.synchronize`. Its kernels are converted to CPU kernels
# and have completed by the time the launch returns, so the method is a no-op.
KA.synchronize(::JLArrays.JLBackend) = nothing

module DeviceWalk

"""
    host_arrays(x) -> Vector{Tuple{String,Type}}

Every `Array` or `BitArray` reachable from `x`, with the field path that reached it.
"""
function host_arrays(x)
    out = Tuple{String,Type}[]
    _walk!(out, Base.IdSet{Any}(), x, "")
    return out
end

_is_flowgeometries(T) = startswith(string(parentmodule(T)), "FlowGeometries")

function _walk!(out, seen, x, path)
    x === nothing && return nothing
    isbits(x) && return nothing
    x in seen && return nothing
    push!(seen, x)

    if x isa AbstractArray
        if x isa Array || x isa BitArray
            push!(out, (path, typeof(x)))
            return nothing
        end
        if x isa SubArray || x isa Base.ReshapedArray
            return _walk!(out, seen, parent(x), path * "→parent")
        end
        # A lazy wrapper carries its storage in its fields; anything else is the device's own array.
        _is_flowgeometries(typeof(x)) || return nothing
    end

    if x isa Tuple || x isa NamedTuple
        for (k, v) in pairs(x)
            _walk!(out, seen, v, string(path, ".", k))
        end
        return nothing
    end

    T = typeof(x)
    isstructtype(T) || return nothing
    # A foreign transform handle owns memory this package does not allocate.
    occursin("Plan", string(nameof(T))) && return nothing
    for f in fieldnames(T)
        isdefined(x, f) || continue
        _walk!(out, seen, getfield(x, f), string(path, ".", f))
    end
    return nothing
end

end # module DeviceWalk

const _JL = CB.GPUBackend(JLArrays.JLBackend())
const CART = FG.Geometry.CartesianGeometry{Float64}()

function _report(label, hits)
    isempty(hits) && return
    println("\n  ", label, ": ", length(hits), " host arrays reachable")
    for (path, T) in first(hits, 12)
        println("    ", path, "  ::  ", T)
    end
    length(hits) > 12 && println("    … and ", length(hits) - 12, " more")
end

Test.@testset "device gate: the walker itself" begin
    # A host array is found through a tuple, a struct field, and a view.
    Test.@test !isempty(DeviceWalk.host_arrays((zeros(2),)))
    Test.@test !isempty(DeviceWalk.host_arrays(view(zeros(4), 1:2)))
    # A device array is a leaf, and its own host-side storage is not reported.
    Test.@test isempty(DeviceWalk.host_arrays(JLArrays.JLArray(zeros(2))))
    # An isbits lazy mask carries no storage.
    Test.@test isempty(DeviceWalk.host_arrays(FG.Grids.AllActive((2, 2))))
end

Test.@testset "device: no host memory reachable from a device plan" begin
    n = 12
    xr = range(0.0, 1.0; length = n)
    grid = FG.Grids.StructuredGrid(CART, xr, xr)

    for (label, solver) in (("jacobi", HD.CGSolver(; multigrid = false)),
                            ("multigrid", HD.CGSolver(; multigrid = true)))
        plan = HD.plan_helmholtz(grid; boundary = HD.Neumann(), solver = solver, backend = _JL)
        ws = HD.allocate_workspace(plan)
        res = HD.allocate_result(plan)

        Test.@testset "$label" begin
            for (what, obj) in (("plan", plan), ("workspace", ws), ("result", res))
                hits = DeviceWalk.host_arrays(obj)
                _report("$label/$what", hits)
                Test.@test isempty(hits)
            end
        end
    end
end

Test.@testset "device: a decomposition runs and matches the host" begin
    n = 12
    xr = range(0.0, 1.0; length = n)
    grid = FG.Grids.StructuredGrid(CART, xr, xr)
    solver = HD.CGSolver(; multigrid = true)
    Random.seed!(21)
    u = randn(n, n, 2)

    hplan = HD.plan_helmholtz(grid; boundary = HD.Neumann(), solver, backend = CB.SerialBackend())
    href = HD.helmholtz_decompose!(HD.allocate_result(hplan), u, hplan,
                                   HD.allocate_workspace(hplan))

    dplan = HD.plan_helmholtz(grid; boundary = HD.Neumann(), solver, backend = _JL)
    dres = HD.helmholtz_decompose!(HD.allocate_result(dplan), JLArrays.JLArray(u), dplan,
                                   HD.allocate_workspace(dplan))

    Test.@test Array(dres.χ) ≈ href.χ
    Test.@test Array(dres.u_rot) ≈ href.u_rot
    Test.@test Array(dres.u_div) ≈ href.u_div
end
