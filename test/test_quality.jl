using Test: Test
using Aqua: Aqua
using HelmholtzDecomposition: HelmholtzDecomposition as HD

Test.@testset "Aqua" begin
    Aqua.test_all(HD)
end