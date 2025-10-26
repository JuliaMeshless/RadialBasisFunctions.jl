"""
Common utilities for end-to-end integration tests with Hermite interpolation.

This module provides reusable functions for setting up 2D test domains,
boundary conditions, and target functions for testing RBF operators.
"""

using Test
using StaticArraysCore
using LinearAlgebra

"""
    create_2d_unit_square_domain(spacing::Float64=0.05)

Create a uniform 2D grid on the unit square [0,1]² with the given spacing.

# Arguments
- `spacing`: Grid spacing (default: 0.05, resulting in 441 points)

# Returns
- `domain_2d`: Vector of SVector{2,Float64} points
"""
function create_2d_unit_square_domain(spacing::Float64=0.05)
    domain_2d = SVector{2,Float64}[]
    for x in 0.0:spacing:1.0
        for y in 0.0:spacing:1.0
            push!(domain_2d, SVector(x, y))
        end
    end
    return domain_2d
end

"""
    identify_boundary_points(domain_2d)

Identify which points in the domain are on the boundary of the unit square.

# Arguments
- `domain_2d`: Vector of 2D points

# Returns
- `is_boundary`: Boolean vector indicating boundary points
"""
function identify_boundary_points(domain_2d)
    is_boundary = zeros(Bool, length(domain_2d))
    for i in eachindex(domain_2d)
        if domain_2d[i][1] == 0.0 ||
            domain_2d[i][1] == 1.0 ||
            domain_2d[i][2] == 0.0 ||
            domain_2d[i][2] == 1.0
            is_boundary[i] = true
        end
    end
    return is_boundary
end

"""
    calculate_normal(point::SVector{2,Float64})

Calculate the outward unit normal vector at a boundary point of the unit square.

# Arguments
- `point`: 2D point on the boundary

# Returns
- `n`: Outward unit normal vector
"""
function calculate_normal(point::SVector{2,Float64})
    n = SVector(0.0, 0.0)
    if point[1] == 0.0
        n += SVector(-1.0, 0.0)
    elseif point[1] == 1.0
        n += SVector(1.0, 0.0)
    end
    if point[2] == 0.0
        n += SVector(0.0, -1.0)
    elseif point[2] == 1.0
        n += SVector(0.0, 1.0)
    end
    return normalize(n)
end

"""
    compute_normals(domain_2d, is_boundary)

Compute normal vectors for all boundary points.

# Arguments
- `domain_2d`: Vector of 2D points
- `is_boundary`: Boolean vector indicating boundary points

# Returns
- `normals`: Vector of normal vectors (only for boundary points)
"""
function compute_normals(domain_2d, is_boundary)
    normals = SVector{2,Float64}[]
    for i in eachindex(domain_2d)
        if is_boundary[i]
            push!(normals, calculate_normal(domain_2d[i]))
        end
    end
    return normals
end

"""
Target function: u(x,y) = x² + y² + 1
"""
target_function(x, y) = x^2 + y^2 + 1.0

"""
Gradient of target function: ∇u = (2x, 2y)
"""
target_gradient(x, y) = SVector(2.0 * x, 2.0 * y)

"""
Laplacian of target function: ∇²u = 4
"""
target_laplacian(x, y) = 4.0

"""
    target_Neumann_bc(x, y, normal)

Compute Neumann boundary condition value: ∂u/∂n = ∇u · n

# Arguments
- `x, y`: Point coordinates
- `normal`: Outward unit normal vector

# Returns
- Directional derivative value
"""
function target_Neumann_bc(x, y, normal)
    return dot(target_gradient(x, y), normal)
end

"""
    setup_mixed_boundary_conditions(domain_2d, is_boundary, RBF)

Set up mixed boundary conditions for the unit square:
- x=1: Neumann BC
- x=0: Robin BC (α=0.5, β=0.5)
- y=0,1: Dirichlet BC

# Arguments
- `domain_2d`: Vector of 2D points
- `is_boundary`: Boolean vector indicating boundary points
- `RBF`: RadialBasisFunctions module

# Returns
- `boundary_conditions`: Vector of boundary condition objects
"""
function setup_mixed_boundary_conditions(domain_2d, is_boundary, RBF)
    boundary_conditions = RBF.BoundaryCondition{Float64}[]
    for i in eachindex(domain_2d)
        if is_boundary[i] && domain_2d[i][1] == 1.0
            push!(boundary_conditions, RBF.Neumann())
        elseif is_boundary[i] && domain_2d[i][1] == 0.0
            push!(boundary_conditions, RBF.Robin(0.5, 0.5))
        elseif is_boundary[i]
            push!(boundary_conditions, RBF.Dirichlet())
        end
    end
    return boundary_conditions
end

"""
    construct_u_values_hermite(domain_2d, is_boundary, boundary_conditions, normals, RBF)

Construct u_values vector for Hermite interpolation with proper BC handling.

For Hermite interpolation, u_values at boundaries should contain BC-transformed values:
- Interior: u_values[i] = f(x,y)
- Dirichlet: u_values[i] = f(x,y)
- Neumann: u_values[i] = ∂f/∂n
- Robin: u_values[i] = α*f + β*∂f/∂n

# Arguments
- `domain_2d`: Vector of 2D points
- `is_boundary`: Boolean vector indicating boundary points
- `boundary_conditions`: Vector of boundary condition objects
- `normals`: Vector of normal vectors for boundary points
- `RBF`: RadialBasisFunctions module

# Returns
- `u_values`: Vector of values for Hermite interpolation
"""
function construct_u_values_hermite(
    domain_2d, is_boundary, boundary_conditions, normals, RBF
)
    u_values = [target_function(p[1], p[2]) for p in domain_2d]
    bnd_counter = 0
    for i in eachindex(domain_2d)
        if is_boundary[i]
            bnd_counter += 1
            normal = normals[bnd_counter]
            if RBF.is_neumann(boundary_conditions[bnd_counter])
                u_values[i] = target_Neumann_bc(domain_2d[i][1], domain_2d[i][2], normal)
            elseif RBF.is_robin(boundary_conditions[bnd_counter])
                u_values[i] =
                    0.5 * target_function(domain_2d[i][1], domain_2d[i][2]) +
                    0.5 * target_Neumann_bc(domain_2d[i][1], domain_2d[i][2], normal)
            elseif RBF.is_dirichlet(boundary_conditions[bnd_counter])
                u_values[i] = target_function(domain_2d[i][1], domain_2d[i][2])
            end
        end
    end
    return u_values
end

# Export functions
export create_2d_unit_square_domain
export identify_boundary_points
export calculate_normal
export compute_normals
export target_function
export target_gradient
export target_laplacian
export target_Neumann_bc
export setup_mixed_boundary_conditions
export construct_u_values_hermite
