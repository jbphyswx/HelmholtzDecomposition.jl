# MPI batch-decomposition smoke test. Launch with:
#   mpiexec -n 3 julia --project=test test/mpi_batch.jl
# Verifies the MPIBackend batch reproduces the serial result on every rank.
using HelmholtzDecomposition: HelmholtzDecomposition as HD
using MPI: MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

grid = HD.StructuredGrid(HD.CartesianGeometry(0.05, 0.05),
    collect(range(0.05, 0.95, length = 19)), collect(range(0.05, 0.95, length = 19)))

mkfield(s) = begin
    U = zeros(19, 19, 2)
    for j in 1:19, i in 1:19
        U[i, j, 1] = -π * s * sin(π * (0.05i)) * cos(π * (0.05j))
        U[i, j, 2] = π * s * cos(π * (0.05i)) * sin(π * (0.05j))
    end
    U
end
fields = [mkfield(s) for s in (1.0, 2.0, 0.5, 1.5, 0.8)]
solver = HD.SORSolver(; max_iter = 3_000, tol = 1e-8, boundary = HD.Dirichlet())
kw = (; solver = solver, boundary_χ = HD.Dirichlet(), boundary_ψ = HD.Dirichlet())

serial = HD.helmholtz_decompose_batch(grid, fields; kw...)
mpi = HD.helmholtz_decompose_batch(grid, fields; backend = HD.MPIBackend(), kw...)

ok = length(mpi) == length(fields) && all(mpi[i].u_rot == serial[i].u_rot for i in eachindex(fields))
if rank == 0
    println(ok ? "MPI batch OK on $(MPI.Comm_size(comm)) ranks" : "MPI batch MISMATCH")
end
ok || error("MPI batch mismatch on rank $rank")
MPI.Finalize()
