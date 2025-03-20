using RadialBasisFunctions

@testset "Constructors" begin
    c = Custom(identity)
    @test c(1) == 1
end

@testset "Printing" begin
    c = Custom(identity)
    @test RadialBasisFunctions.print_op(c) == "Custom Operator"
end
