using Test
using StaticArraysCore
using LinearAlgebra

function create_2d_unit_square_domain(spacing::Float64=0.05; randomize=false)
    domain_2d = SVector{2,Float64}[]
    for x in 0.0:spacing:1.0
        for y in 0.0:spacing:1.0
            # Check if point is on boundary
            is_on_boundary = (x == 0.0 || x == 1.0 || y == 0.0 || y == 1.0)

            if randomize && !is_on_boundary
                # Only add noise to interior points
                noise_x = (rand() - 0.5) * spacing * 0.3
                noise_y = (rand() - 0.5) * spacing * 0.3
                push!(domain_2d, SVector(x + noise_x, y + noise_y))
            else
                # Keep boundary points exact
                push!(domain_2d, SVector(x, y))
            end
        end
    end
    return domain_2d
end

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

function compute_normals(domain_2d, is_boundary)
    normals = SVector{2,Float64}[]
    for i in eachindex(domain_2d)
        if is_boundary[i]
            push!(normals, calculate_normal(domain_2d[i]))
        end
    end
    return normals
end

target_function(x, y) = x^2 + y^2 + 1.0

target_gradient(x, y) = SVector(2.0 * x, 2.0 * y)

target_laplacian(x, y) = 4.0

function target_Neumann_bc(x, y, normal)
    return dot(target_gradient(x, y), normal)
end

function setup_test_boundary_conditions(domain_2d, is_boundary, RBF)
    boundary_conditions = RBF.BoundaryCondition{Float64}[]
    for i in eachindex(domain_2d)
        if is_boundary[i]
            x, y = domain_2d[i][1], domain_2d[i][2]

            # Priority 1: Dirichlet on y=0 and y=1 (including corners)
            if y == 0.0 || y == 1.0
                push!(boundary_conditions, RBF.Dirichlet())
                # Priority 2: Robin on x=0 (excluding corners already handled)
            elseif x == 0.0
                push!(boundary_conditions, RBF.Robin(0.5, 0.5))
                # Priority 3: Neumann on x=1 (excluding corners already handled)
            elseif x == 1.0
                push!(boundary_conditions, RBF.Neumann())
            end
        end
    end
    return boundary_conditions
end

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
export setup_test_boundary_conditions
export construct_u_values_hermite
