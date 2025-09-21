using Test
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "BoundaryCondition" begin
    @testset "Constructor and type promotion" begin
        # Basic constructors
        bc1 = BoundaryCondition(1.0, 0.0)
        @test α(bc1) == 1.0
        @test β(bc1) == 0.0
        @test bc1 isa BoundaryCondition{Float64}

        # Type promotion
        bc2 = BoundaryCondition(1, 0.5)  # Int and Float64
        @test bc2 isa BoundaryCondition{Float64}
        @test α(bc2) == 1.0
        @test β(bc2) == 0.5

        # Float32 precision
        bc3 = BoundaryCondition(1.0f0, 0.0f0)
        @test bc3 isa BoundaryCondition{Float32}
    end

    @testset "Predicate functions" begin
        # Dirichlet condition: α=1, β=0
        dirichlet = BoundaryCondition(1.0, 0.0)
        @test is_dirichlet(dirichlet)
        @test !is_neumann(dirichlet)
        @test !is_robin(dirichlet)

        # Neumann condition: α=0, β=1  
        neumann = BoundaryCondition(0.0, 1.0)
        @test !is_dirichlet(neumann)
        @test is_neumann(neumann)
        @test !is_robin(neumann)

        # Robin condition: α≠0, β≠0
        robin = BoundaryCondition(0.5, 0.3)
        @test !is_dirichlet(robin)
        @test !is_neumann(robin)
        @test is_robin(robin)

        # Edge cases
        edge1 = BoundaryCondition(0.0, 0.0)  # Neither Dirichlet nor Neumann
        @test !is_dirichlet(edge1)
        @test !is_neumann(edge1)
        @test !is_robin(edge1)  # Both coefficients are zero
    end

    @testset "Constructor helpers" begin
        # Default Float64 constructors
        d1 = Dirichlet()
        @test d1 isa BoundaryCondition{Float64}
        @test is_dirichlet(d1)
        @test α(d1) == 1.0
        @test β(d1) == 0.0

        n1 = Neumann()
        @test n1 isa BoundaryCondition{Float64}
        @test is_neumann(n1)
        @test α(n1) == 0.0
        @test β(n1) == 1.0

        # Typed constructors
        d2 = Dirichlet(Float32)
        @test d2 isa BoundaryCondition{Float32}
        @test is_dirichlet(d2)

        n2 = Neumann(Float32)
        @test n2 isa BoundaryCondition{Float32}
        @test is_neumann(n2)

        # Robin constructor
        r1 = Robin(2.0, 3.0)
        @test r1 isa BoundaryCondition{Float64}
        @test is_robin(r1)
        @test α(r1) == 2.0
        @test β(r1) == 3.0

        # Robin with type promotion
        r2 = Robin(2, 3.0f0)
        @test r2 isa BoundaryCondition{Float32}
    end
end

@testset "HermiteStencilData" begin
    @testset "Constructor with data" begin
        # 2D example with 3 points
        data = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        is_boundary = [true, false, true]
        bcs = [Dirichlet(), Dirichlet(), Neumann()]
        normals = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]

        hsd = RBF.HermiteStencilData(data, is_boundary, bcs, normals)
        @test hsd isa RBF.HermiteStencilData{Float64}
        @test length(hsd.data) == 3
        @test length(hsd.is_boundary) == 3
        @test length(hsd.boundary_conditions) == 3
        @test length(hsd.normals) == 3
        @test hsd.data[1] == [0.0, 0.0]
        @test hsd.is_boundary[2] == false
        @test is_neumann(hsd.boundary_conditions[3])
    end

    @testset "Pre-allocation constructor" begin
        # 2D, 4 points
        hsd = RBF.HermiteStencilData{Float64}(4, 2)
        @test hsd isa RBF.HermiteStencilData{Float64}
        @test length(hsd.data) == 4
        @test length(hsd.is_boundary) == 4
        @test length(hsd.boundary_conditions) == 4
        @test length(hsd.normals) == 4

        # Check pre-allocated dimensions
        @test length(hsd.data[1]) == 2
        @test length(hsd.normals[1]) == 2

        # Check defaults
        @test all(hsd.is_boundary .== false)
        @test all(is_dirichlet.(hsd.boundary_conditions))

        # Float32 version
        hsd32 = RBF.HermiteStencilData{Float32}(3, 1)
        @test hsd32 isa RBF.HermiteStencilData{Float32}
        @test length(hsd32.data[1]) == 1
    end

    @testset "Constructor validation" begin
        # Mismatched lengths should fail
        data = [[0.0, 0.0], [1.0, 0.0]]
        is_boundary = [true, false, true]  # Wrong length
        bcs = [Dirichlet(), Neumann()]
        normals = [[1.0, 0.0], [0.0, 1.0]]

        @test_throws AssertionError RBF.HermiteStencilData(data, is_boundary, bcs, normals)
    end
end

@testset "update_stencil_data!" begin
    @testset "Basic functionality" begin
        # Setup global data
        global_data = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
        is_boundary = [false, true, false, true]
        boundary_conditions = [Dirichlet(), Neumann()]  # Only for boundary points
        normals = [[1.0, 0.0], [0.0, 1.0]]  # Only for boundary points
        global_to_boundary = [0, 1, 0, 2]  # Maps global idx to boundary idx (0 for interior)

        # Create pre-allocated structure
        hsd = RBF.HermiteStencilData{Float64}(3, 2)

        # Update with neighbors [2, 3, 4] (1-indexed)
        neighbors = [2, 3, 4]
        RBF.update_stencil_data!(
            hsd,
            global_data,
            neighbors,
            is_boundary,
            boundary_conditions,
            normals,
            global_to_boundary,
        )

        # Check data copying
        @test hsd.data[1] == [1.0, 0.0]  # global_data[2]
        @test hsd.data[2] == [2.0, 0.0]  # global_data[3]  
        @test hsd.data[3] == [0.0, 1.0]  # global_data[4]

        # Check boundary flags
        @test hsd.is_boundary[1] == true   # is_boundary[2]
        @test hsd.is_boundary[2] == false  # is_boundary[3]
        @test hsd.is_boundary[3] == true   # is_boundary[4]

        # Check boundary conditions  
        # neighbors[1] = 2 -> is_boundary[2] = true -> boundary_idx = global_to_boundary[2] = 1 -> boundary_conditions[1] = Dirichlet()
        @test is_dirichlet(hsd.boundary_conditions[1])  # boundary_conditions[1] is Dirichlet  
        @test is_dirichlet(hsd.boundary_conditions[2])  # Default for interior
        # neighbors[3] = 4 -> is_boundary[4] = true -> boundary_idx = global_to_boundary[4] = 2 -> boundary_conditions[2] = Neumann()
        @test is_neumann(hsd.boundary_conditions[3])  # boundary_conditions[2] is Neumann

        # Check normals
        @test hsd.normals[1] == [1.0, 0.0]  # normals[1]
        @test hsd.normals[2] == [0.0, 0.0]  # Default for interior
        @test hsd.normals[3] == [0.0, 1.0]  # normals[2]
    end
end

@testset "StencilType dispatch" begin
    @testset "Type definitions" begin
        @test InternalStencil() isa RBF.StencilType
        @test RBF.DirichletStencil() isa RBF.StencilType
        @test HermiteStencil() isa RBF.StencilType
    end

    @testset "stencil_type function" begin
        # Setup test data
        is_boundary = [false, true, false, true, true]
        boundary_conditions = [Dirichlet(), Neumann(), Robin(1.0, 2.0)]
        global_to_boundary = [0, 1, 0, 2, 3]

        # Internal stencil (no boundary points in neighbors)
        neighbors = [1, 3]  # All interior points
        result = RBF.stencil_type(
            is_boundary, boundary_conditions, 1, neighbors, global_to_boundary
        )
        @test result isa InternalStencil

        # Dirichlet stencil (eval point is boundary with Dirichlet condition)
        neighbors = [1, 2, 3]
        result = RBF.stencil_type(
            is_boundary, boundary_conditions, 2, neighbors, global_to_boundary
        )
        @test result isa RBF.DirichletStencil

        # Hermite stencil (eval point is boundary with Neumann condition)
        neighbors = [1, 3, 4]
        result = RBF.stencil_type(
            is_boundary, boundary_conditions, 4, neighbors, global_to_boundary
        )
        @test result isa HermiteStencil

        # Hermite stencil (eval point is boundary with Robin condition)
        neighbors = [1, 3, 5]
        result = RBF.stencil_type(
            is_boundary, boundary_conditions, 5, neighbors, global_to_boundary
        )
        @test result isa HermiteStencil

        # Hermite stencil (eval point is interior but has boundary neighbors)
        neighbors = [2, 4]  # Contains boundary points
        result = RBF.stencil_type(
            is_boundary, boundary_conditions, 1, neighbors, global_to_boundary
        )
        @test result isa HermiteStencil
    end
end
