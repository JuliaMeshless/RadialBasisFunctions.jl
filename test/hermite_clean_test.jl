"""
Test file for the new clean Hermite implementation using multiple dispatch.
This tests the single stencil case without region data or full system assembly.
"""

using StaticArraysCore
using LinearAlgebra
using Test

import RadialBasisFunctions as RBF

@testset "Clean Hermite Implementation - Single Stencil" begin
    println("Testing clean Hermite implementation with multiple dispatch...")
    
    # Create a simple 1D problem with 3 points: 2 interior + 1 Neumann boundary
    data = [SVector(0.0), SVector(0.5), SVector(1.0)]  # points along line
    eval_point = SVector(0.5)  # evaluate at middle point
    
    # Create basis and operators
    basis = RBF.PHS(3; poly_deg=1)  # PHS3 with linear polynomial
    mon = RBF.MonomialBasis(1, 1)  # 1D linear monomial basis
    
    # Test 1: Standard stencil (no boundary points)
    println("  Test 1: Standard stencil (no boundary points)")
    boundary_info_none = nothing
    
    # Create system matrices
    k = 3
    nmon = 2  # 1D linear: [1, x]
    n = k + nmon
    A_std = Symmetric(zeros(Float64, n, n), :U)
    b_std = zeros(Float64, n, 1)
    
    # Create simple identity operators for testing  
    identity_rbf = RBF.Custom(basis -> (x1, x2) -> basis(x1, x2))
    identity_mon = RBF.Custom(mon -> (arr, x) -> mon(arr, x))
    ℒrbf = identity_rbf(basis)  # Get the actual operator function
    ℒmon = identity_mon(mon)    # Get the actual operator function
    
    # Test standard path
    weights_std = RBF._build_stencil!(A_std, b_std, ℒrbf, ℒmon, data, eval_point, basis, mon, k, RBF.StandardStencil())
    
    @test size(weights_std) == (k, 1)
    @test all(isfinite.(weights_std))
    println("    ✓ Standard stencil completed")
    
    # Test 2: Hermite stencil with Neumann boundary condition
    println("  Test 2: Hermite stencil with Neumann boundary condition")
    
    # Set up boundary info: point 3 (index 3) is Neumann boundary
    is_boundary = [false, false, true]
    boundary_conditions = [
        RBF.BoundaryCondition(1.0, 0.0),     # placeholder Dirichlet for interior point 1
        RBF.BoundaryCondition(1.0, 0.0),     # placeholder Dirichlet for interior point 2  
        RBF.BoundaryCondition(0.0, 1.0)      # Neumann condition for boundary point 3
    ]
    normals = [
        [0.0],   # placeholder normal for point 1
        [0.0],   # placeholder normal for point 2
        [1.0]    # outward normal for point 3 (rightward)
    ]
    
    boundary_info = RBF.HermiteBoundaryInfo(is_boundary, boundary_conditions, normals)
    
    # Create fresh matrices for Hermite test
    A_herm = Symmetric(zeros(Float64, n, n), :U)
    b_herm = zeros(Float64, n, 1)
    
    # Test Hermite path - this should dispatch to Hermite functions
    try
        weights_herm = RBF._build_stencil!(A_herm, b_herm, ℒrbf, ℒmon, data, eval_point, basis, mon, k, RBF.HermiteStencil(), boundary_info)
        
        @test size(weights_herm) == (k, 1)
        @test all(isfinite.(weights_herm))
        
        # The weights should be different from standard case due to boundary modification
        @test !isapprox(weights_std, weights_herm, rtol=1e-12)
        
        println("    ✓ Hermite stencil completed")
        println("    ✓ Weights differ from standard case as expected")
        
    catch e
        println("    ✗ Hermite test failed with error: $e")
        @test false
    end
    
    # Test 3: Trait dispatch functionality  
    println("  Test 3: Trait dispatch functionality")
    
    # Test stencil type detection
    @test RBF.stencil_type(nothing) isa RBF.StandardStencil
    @test RBF.stencil_type(boundary_info) isa RBF.HermiteStencil
    
    # Test boundary detection
    @test RBF.has_boundary_points(nothing) == false
    @test RBF.has_boundary_points(boundary_info) == true
    
    println("    ✓ Trait dispatch works correctly")
    
    println("✓ All clean Hermite implementation tests passed!")
end