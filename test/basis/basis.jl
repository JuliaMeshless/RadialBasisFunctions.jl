using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "Basis - General Utils" begin
    m = MonomialBasis(2, 3)
    @test dim(m) == 2
    @test degree(m) == 3

    ∂m = RBF.∂(m, 1)
    @test dim(∂m) == 2
    @test degree(∂m) == 3
end
