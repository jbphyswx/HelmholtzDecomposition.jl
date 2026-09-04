# MPI batch-decomposition smoke test. Launch with:
#   mpiexec -n 3 julia --project=test test/mpi_batch.jl
# MPI.jl ships its own launcher, so this is reachable from a test run as
# `run(`$(MPI.mpiexec()) -n 3 julia --project=test test/mpi_batch.jl`)`.
#
# Verifies the MPIBackend batch reproduces the serial result on every rank, in input order.
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using MPI: MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

const N = 19
xs = range(0.05, 0.95; length = N)
grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, xs)

function mkfield(s)
    U = zeros(N, N, 2)
    for j in 1:N, i in 1:N
        U[i, j, 1] = -π * s * sin(π * xs[i]) * cos(π * xs[j])
        U[i, j, 2] = π * s * cos(π * xs[i]) * sin(π * xs[j])
    end
    return U
end

fields = [mkfield(s) for s in (1.0, 2.0, 0.5, 1.5, 0.8)]

# One plan for the whole batch; every rank builds the same one and each takes its own workspace.
plan = HD.plan_helmholtz(grid; boundary = HD.Dirichlet(), backend = CB.SerialBackend())

serial = HD.helmholtz_decompose_batch(plan, fields; backend = CB.SerialBackend())
mpi = HD.helmholtz_decompose_batch(plan, fields; backend = CB.MPIBackend())

ok = length(mpi) == length(fields) &&
     all(mpi[i].u_rot == serial[i].u_rot for i in eachindex(fields)) &&
     all(mpi[i].χ == serial[i].χ for i in eachindex(fields))

if rank == 0
    println(ok ? "MPI batch OK on $(MPI.Comm_size(comm)) ranks" : "MPI batch MISMATCH")
end
ok || error("MPI batch mismatch on rank $rank")
MPI.Finalize()
