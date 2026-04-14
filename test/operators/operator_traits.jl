using RadialBasisFunctions
using StaticArraysCore

@testset "output_rank" begin
    @test output_rank(Laplacian()) == 0
    @test output_rank(Partial(1, 1)) == 0
    @test output_rank(MixedPartial(1, 2)) == 0
    @test output_rank(Directional{2}(ones(2))) == 0
    @test output_rank(Divergence{2}()) == 0
    @test output_rank(Curl{2}()) == 0
    @test output_rank(StrainRate{2}()) == 0
    @test output_rank(RotationRate{2}()) == 0
    @test output_rank(Identity()) == 0
    @test output_rank(Jacobian{2}()) == 1
    @test output_rank(Hessian{2}()) == 2
    @test output_rank(2.0 * Laplacian()) == 0
    @test output_rank(3.0 * Jacobian{2}()) == 1
end

@testset "requires_vector_input" begin
    @test !requires_vector_input(Laplacian())
    @test !requires_vector_input(Partial(1, 1))
    @test !requires_vector_input(Jacobian{2}())
    @test !requires_vector_input(Hessian{2}())
    @test !requires_vector_input(Identity())
    @test requires_vector_input(Divergence{2}())
    @test requires_vector_input(Curl{3}())
    @test requires_vector_input(StrainRate{2}())
    @test requires_vector_input(RotationRate{3}())
end

@testset "is_symmetric" begin
    @test !is_symmetric(Laplacian())
    @test !is_symmetric(Jacobian{2}())
    @test !is_symmetric(Divergence{2}())
    @test is_symmetric(Hessian{2}())
    @test is_symmetric(StrainRate{2}())
end

@testset "is_antisymmetric" begin
    @test !is_antisymmetric(Laplacian())
    @test !is_antisymmetric(StrainRate{2}())
    @test is_antisymmetric(RotationRate{2}())
end

@testset "is_self_adjoint" begin
    @test is_self_adjoint(Laplacian())
    @test is_self_adjoint(Identity())
    @test !is_self_adjoint(Partial(1, 1))
    @test !is_self_adjoint(Jacobian{2}())
    @test !is_self_adjoint(Regrid())
end

@testset "derivative_order" begin
    @test derivative_order(Identity()) == 0
    @test derivative_order(Regrid()) == 0
    @test derivative_order(Partial(1, 1)) == 1
    @test derivative_order(Partial(2, 1)) == 2
    @test derivative_order(MixedPartial(1, 2)) == 2
    @test derivative_order(Laplacian()) == 2
    @test derivative_order(Jacobian{2}()) == 1
    @test derivative_order(Hessian{2}()) == 2
    @test derivative_order(Directional{2}(ones(2))) == 1
    @test derivative_order(Divergence{2}()) == 1
    @test derivative_order(Curl{3}()) == 1
    @test derivative_order(StrainRate{2}()) == 1
    @test derivative_order(RotationRate{3}()) == 1
    @test derivative_order(2.0 * Laplacian()) == 2
end
