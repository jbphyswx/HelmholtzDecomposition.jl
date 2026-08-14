"""
Cost of a decomposition, and what the plan/workspace split buys.

Three things are separated here, because they are charged differently:

- `plan_helmholtz` — the grid's Laplacian coefficients, face metrics and dual grids. Once per grid.
- `allocate_workspace` / `allocate_result` — the buffers a decomposition writes through. Once per
  task, and once per batch rather than once per field.
- `helmholtz_decompose!` — the steady-state call, which allocates nothing.

The allocating `helmholtz_decompose(u, grid)` does all three every time, which is what makes it the
convenience entry point rather than the one to loop over.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using Printf: Printf
using Random: Random

best(f, n = 3) = (f(); minimum(f() for _ in 1:n))

function report(N::Int)
    axis = range(0.0, 1.0; length = N)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), axis, axis)
    Random.seed!(1)
    u = randn(N, N, 2)

    tplan = best(() -> @elapsed HD.plan_helmholtz(grid; backend = CB.SerialBackend()))
    plan = HD.plan_helmholtz(grid; backend = CB.SerialBackend())
    tws = best(() -> @elapsed HD.allocate_workspace(plan))
    ws, res = HD.allocate_workspace(plan), HD.allocate_result(plan)

    step = () -> @elapsed HD.helmholtz_decompose!(res, u, plan, ws)
    tstep = best(step)
    HD.helmholtz_decompose!(res, u, plan, ws)
    alloc = @allocated HD.helmholtz_decompose!(res, u, plan, ws)
    tall = best(() -> @elapsed HD.helmholtz_decompose(u, grid))

    Printf.@printf("%5d² | plan %7.2f ms | workspace %6.2f ms | decompose! %7.2f ms (%d B) | all-in-one %7.2f ms | solver %s\n",
            N, 1e3tplan, 1e3tws, 1e3tstep, alloc, 1e3tall, nameof(typeof(plan.solver)))
    return nothing
end

println("=== Cost of a decomposition ===")
for N in (64, 128, 256, 512)
    report(N)
end

println()
println("=== Batch: one plan, one contiguous result ===")
for (N, B) in ((128, 32), (256, 16))
    axis = range(0.0, 1.0; length = N)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), axis, axis)
    plan = HD.plan_helmholtz(grid; backend = CB.SerialBackend())
    Random.seed!(2)
    fields = [randn(N, N, 2) for _ in 1:B]
    batch = HD.allocate_batch(plan, B)
    run = () -> @elapsed HD.helmholtz_decompose_batch!(batch, fields, plan;
                                                       backend = CB.SerialBackend())
    t = best(run, 2)
    Printf.@printf("%5d² × %-4d | %8.1f ms total | %6.2f ms/field\n", N, B, 1e3t, 1e3t / B)
end
