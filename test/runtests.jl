using Test: @testset

@testset "HelmholtzDecomposition.jl" begin
    include("operators.jl")
    include("capability.jl")
    include("accuracy.jl")
    include("scattered.jl")

    include("test_allocs.jl")
    include("test_quality.jl")
end
