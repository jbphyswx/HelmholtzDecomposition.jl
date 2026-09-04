"""
    benchmark/benchmarks.jl — the baseline every performance claim is measured against.

    julia --project=benchmark benchmark/benchmarks.jl [quick|full|huge]

Three costs are reported separately because they are paid at different rates: building a plan is
once per grid, building a workspace is once per task, and `helmholtz_decompose!` is once per field.
Timing the allocating entry point sums all three and hides which one moved.

`Base.summarysize(plan)` is reported alongside, in the same table. The plan holds one array per
face direction for the area, one for the gap, one per direction for the Laplacian coefficients, and
the cell diagonal and measure, on the primal grid and on each of the `P` dual grids. On a 3-D grid
that total is several times the field it operates on.

A case that raises is reported as its exception type and the run continues, so the table records
which paths execute.

Sizes:

| set | 2-D | 3-D | batch | scattered |
|---|---|---|---|---|
| `quick` | 128² | 32³ | 1, 8 | 2000 |
| `full` | 512² | 128³ | 1, 64, 256 | 2000, 100000 |
| `huge` | 512² | 256³ | 1, 64, 256 | 2000, 100000 |

`huge` is sized by its fields: a `256³` plan is about 0.11 MiB and one velocity field is 384 MiB.
"""

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using AbstractFFTs: AbstractFFTs
using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra
using NUFSHT: NUFSHT
using NonuniformFFTs: NonuniformFFTs
using OhMyThreads: OhMyThreads
using Printf: Printf
using Random: Random

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

"""
    bench(f; samples) -> (; time, bytes)

Minimum wall time over `samples` runs, and the bytes one run allocates, after two warm-up calls.

`f` takes no arguments and captures what it needs, so the measurement carries no splat: `f(args...)`
costs about 240 B of its own, which at the small end is the thing being measured.
"""
function bench(f::F; samples::Int = 5) where {F}
    f()
    f()
    bytes = @allocated f()
    t = Inf
    for _ in 1:samples
        t = min(t, @elapsed f())
    end
    return (; time = t, bytes = bytes)
end

const NA = (; time = NaN, bytes = -1)

_fmt_time(t) = isnan(t) ? "     —" :
               t < 1e-3 ? Printf.@sprintf("%6.1fµs", t * 1e6) :
               t < 1.0 ? Printf.@sprintf("%6.2fms", t * 1e3) : Printf.@sprintf("%6.2f s", t)

function _fmt_bytes(b)
    b < 0 && return "     —"
    b == 0 && return "     0"
    b < 1024 && return Printf.@sprintf("%5d B", b)
    b < 1024^2 && return Printf.@sprintf("%5.1fKiB", b / 1024)
    b < 1024^3 && return Printf.@sprintf("%5.1fMiB", b / 1024^2)
    return Printf.@sprintf("%5.2fGiB", b / 1024^3)
end

const ROWS = Vector{NamedTuple}()

function record!(; case, solver, plan, workspace, call, planbytes)
    push!(ROWS, (; case, solver, plan, workspace, call, planbytes))
    Printf.@printf("  %-34s %-26s %s %s %s %s  %s\n", case, solver,
            _fmt_time(plan.time), _fmt_time(workspace.time), _fmt_time(call.time),
            _fmt_bytes(call.bytes), _fmt_bytes(planbytes))
    return nothing
end

function header()
    Printf.@printf("  %-34s %-26s %7s %7s %7s %7s  %7s\n",
            "case", "solver", "plan", "wspace", "call", "call B", "plan B")
    println("  ", "-"^115)
end

# ---------------------------------------------------------------------------
# Configuration, printed first
# ---------------------------------------------------------------------------

const EXTENSIONS = (:HelmholtzDecompositionFFTWExt, :HelmholtzDecompositionAbstractFFTsExt,
                    :HelmholtzDecompositionFSHExt, :HelmholtzDecompositionNUFSHTExt,
                    :HelmholtzDecompositionFINUFFTExt, :HelmholtzDecompositionNonuniformFFTsExt,
                    :HelmholtzDecompositionOhMyThreadsExt,
                    :HelmholtzDecompositionKernelAbstractionsExt)

function configuration()
    println("HelmholtzDecomposition.jl benchmark")
    println("  julia          ", VERSION)
    println("  threads        ", Threads.nthreads())
    println("  BLAS threads   ", LinearAlgebra.BLAS.get_num_threads())
    println("  FFTW threads   ", FFTW.get_num_threads())
    println("  total memory   ", Printf.@sprintf("%.1f GiB", Sys.total_memory() / 1024^3))
    loaded = filter(e -> Base.get_extension(HD, e) !== nothing, EXTENSIONS)
    println("  extensions     ", isempty(loaded) ? "none" :
                                 join(replace.(String.(loaded), "HelmholtzDecomposition" => "",
                                               "Ext" => ""), ", "))
    println()
end

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

const CART = FG.Geometry.CartesianGeometry{Float64}()
const SPH = FG.Geometry.SphericalGeometry(1.0)

_bounded(n, d) = FG.Grids.StructuredGrid(CART, ntuple(_ -> range(0.0, 1.0; length = n), d)...)

_periodic(n, d) = FG.Grids.StructuredGrid(
    CART, ntuple(_ -> range(0.0, 1.0 - 1 / n; length = n), d)...;
    topology = ntuple(_ -> true, d), period = ntuple(_ -> 1.0, d))

function _masked(n)
    m = trues(n, n)
    c = n ÷ 2
    r = n ÷ 8
    for j in 1:n, i in 1:n
        (i - c)^2 + (j - c)^2 < r^2 && (m[i, j] = false)
    end
    return FG.Grids.StructuredGrid(CART, range(0.0, 1.0; length = n),
                                   range(0.0, 1.0; length = n); mask = m)
end

# Clenshaw–Curtis node positions, which is the node set the spherical transform is defined on.
function _clenshaw_curtis(nlat)
    ax = FG.SphericalSampling.spherical_axes(Float64, FG.SphericalSampling.ClenshawCurtisSampling(),
                                             nlat)
    return FG.Grids.StructuredGrid(SPH, ax.λ, ax.φ)
end

# Longitude as a range, so its uniformity is provable from the axis type.
function _latlon(nlat)
    nlon = 2nlat
    λ = range(0.0, 2π - 2π / nlon; length = nlon)
    φ = range(-π / 2 + π / (2nlat), π / 2 - π / (2nlat); length = nlat)
    return FG.Grids.StructuredGrid(SPH, λ, φ; topology = (true, false), period = (2π, 0.0))
end

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

"""
    structured_case(label, grid; boundary, solver, backend)

Plan, workspace and steady-state cost of one decomposition, plus the plan's footprint.
"""
function structured_case(label, grid; boundary = HD.Neumann(), solver = HD.AutoSolver(),
                         backend = CB.SerialBackend())
    N = ndims(grid)
    plan = HD.plan_helmholtz(grid; boundary, solver, backend)
    tplan = bench(() -> HD.plan_helmholtz(grid; boundary, solver, backend); samples = 3)
    tws = bench(() -> HD.allocate_workspace(plan); samples = 3)

    ws = HD.allocate_workspace(plan)
    res = HD.allocate_result(plan)
    Random.seed!(11)
    u = randn(size(grid)..., N)

    tcall = bench(() -> HD.helmholtz_decompose!(res, u, plan, ws); samples = 3)
    record!(; case = label, solver = string(nameof(typeof(plan.solver))),
            plan = tplan, workspace = tws, call = tcall, planbytes = Base.summarysize(plan))
    return nothing
end

"""
    batch_case(label, grid, nfields; backend)

Per-field cost across a batch. One plan is shared and each task takes its own workspace, so the
number to read is `call / nfields` against the single-field `call` above.
"""
function batch_case(label, grid, nfields::Int; backend = CB.SerialBackend())
    N = ndims(grid)
    plan = HD.plan_helmholtz(grid; boundary = HD.Neumann(), backend = CB.SerialBackend())
    Random.seed!(12)
    fields = [randn(size(grid)..., N) for _ in 1:nfields]
    batch = HD.allocate_batch(plan, nfields)

    t = bench(() -> HD.helmholtz_decompose_batch!(batch, fields, plan; backend); samples = 3)
    per = (; time = t.time / nfields, bytes = t.bytes ÷ nfields)
    record!(; case = label, solver = string(nameof(typeof(plan.solver))) * " /field",
            plan = NA, workspace = NA, call = per, planbytes = Base.summarysize(plan))
    return nothing
end

"""
    spectral_case(label, grid)

The fully spectral entry point, which is a different code path from `helmholtz_decompose!` and is
the one the spherical geometries are advertised on.
"""
function spectral_case(label, grid)
    N = ndims(grid)
    Random.seed!(13)
    U = randn(size(grid)..., N)
    t = bench(() -> HD.helmholtz_decompose_spectral(U, grid); samples = 3)
    record!(; case = label, solver = "spectral", plan = NA, workspace = NA, call = t,
            planbytes = -1)
    return nothing
end

"""
    scattered_case(label, npoints, nk)

Scattered Cartesian samples through the non-uniform transform. Needs `FINUFFT` or `NonuniformFFTs`.

`nk` sets the solver's mode count, so the label names what was measured. The default mode count is
fixed at 64², and the fit is a conjugate-gradient solve on the normal equations whose cost and
conditioning both track `prod(nk) / npoints`.
"""
function scattered_case(label, npoints::Int, nk::Int)
    Random.seed!(14)
    Lx = Ly = 2π
    x = Lx .* rand(npoints)
    y = Ly .* rand(npoints)
    # The measure is positional. This form leaves every node with degree 0, which the non-uniform
    # transform does not read: it takes the coordinates and the period.
    measure = fill(Lx * Ly / npoints, npoints)
    grid = FG.Grids.UnstructuredGrid(CART, (x, y), measure;
                                     periodic = (true, true), period = (Lx, Ly))
    ext = Base.get_extension(HD, :HelmholtzDecompositionFINUFFTExt)
    solver = ext.CartesianNUFFTSolver(; nk = (nk, nk))
    U = randn(npoints, 2)
    t = bench(() -> HD.helmholtz_decompose_spectral(U, grid; solver); samples = 3)
    record!(; case = label, solver = "NUFFT", plan = NA, workspace = NA, call = t, planbytes = -1)
    return nothing
end

# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

function suite(set::Symbol)
    n2 = set === :quick ? 128 : 512
    n3 = set === :quick ? 32 : set === :huge ? 256 : 128
    batches = set === :quick ? (1, 8) : (1, 64, 256)
    scatter = set === :quick ? ((2_000, 32),) : ((2_000, 32), (100_000, 128))
    nlat = set === :quick ? 32 : 128

    configuration()

    println("Cartesian 2-D, n = $n2")
    header()
    structured_case("bounded $(n2)²", _bounded(n2, 2))
    structured_case("periodic $(n2)²", _periodic(n2, 2))
    structured_case("masked $(n2)²", _masked(n2))
    structured_case("bounded $(n2)², Dirichlet", _bounded(n2, 2); boundary = HD.Dirichlet())
    structured_case("masked $(n2)², jacobi", _masked(n2); solver = HD.CGSolver(; multigrid = false))
    println()

    println("Cartesian 3-D, n = $n3")
    header()
    structured_case("bounded $(n3)³", _bounded(n3, 3))
    structured_case("periodic $(n3)³", _periodic(n3, 3))
    println()

    println("Spherical, nlat = $nlat")
    header()
    structured_case("clenshaw-curtis $nlat", _clenshaw_curtis(nlat))
    structured_case("lat-lon $nlat", _latlon(nlat))
    # `_decompose_spectral` has no spherical method, so `helmholtz_decompose_spectral` on a sphere
    # raises. Add the two rows here once it does.
    println("  (spherical spectral entry point: no method)")
    println()

    println("Spectral entry point, Cartesian")
    header()
    spectral_case("periodic $(n2)², spectral", _periodic(n2, 2))
    println()

    println("Scattered Cartesian")
    header()
    for (m, nk) in scatter
        scattered_case("scattered M=$m, nk=$(nk)²", m, nk)
    end
    println()

    println("Batch, periodic $(n2)² (per field)")
    header()
    for b in batches
        batch_case("serial B=$b", _periodic(n2, 2), b)
        Threads.nthreads() > 1 &&
            batch_case("threaded B=$b", _periodic(n2, 2), b; backend = CB.ThreadedBackend())
    end
    println()

    println("Intra-field backend, masked $(n2)²")
    header()
    structured_case("masked serial", _masked(n2); backend = CB.SerialBackend())
    Threads.nthreads() > 1 &&
        structured_case("masked threaded", _masked(n2); backend = CB.ThreadedBackend())
    println()
    println("BENCHMARK_COMPLETE")
    return ROWS
end

if abspath(PROGRAM_FILE) == @__FILE__
    set = isempty(ARGS) ? :quick : Symbol(ARGS[1])
    set in (:quick, :full, :huge) ||
        error("unknown size set $set; choose quick, full or huge")
    suite(set)
end
