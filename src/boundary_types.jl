"""
Boundary condition types and utilities for Hermite interpolation.
"""

# Simple boundary condition type
struct BoundaryCondition{T<:Real}
    α::T
    β::T

    function BoundaryCondition(α::A, β::B) where {A,B}
        T = promote_type(A, B)
        return new{T}(α, β)
    end
end

# Accessors
α(bc::BoundaryCondition) = bc.α
β(bc::BoundaryCondition) = bc.β

# Predicate functions
is_dirichlet(bc::BoundaryCondition) = isone(bc.α) && iszero(bc.β)
is_neumann(bc::BoundaryCondition) = iszero(bc.α) && isone(bc.β)
is_robin(bc::BoundaryCondition) = !iszero(bc.α) && !iszero(bc.β)

# Constructor helpers
Dirichlet(::Type{T}=Float64) where {T<:Real} = BoundaryCondition(one(T), zero(T))
Neumann(::Type{T}=Float64) where {T<:Real} = BoundaryCondition(zero(T), one(T))
Robin(α::Real, β::Real) = BoundaryCondition(α, β)

# Boundary information for a local stencil
"""
This struct is meant to be used to correctly broadcast the build_stencil() function
When Hermite scheme is used, it can be given to _build_stencil!() in place of the sole data.
"""
struct HermiteStencilData{T<:Real}
    data::AbstractVector{Vector{T}}  # Coordinates of stencil points
    is_boundary::Vector{Bool}
    boundary_conditions::Vector{BoundaryCondition{T}}
    normals::Vector{Vector{T}}

    function HermiteStencilData(
        data::AbstractVector{Vector{T}},
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{BoundaryCondition{T}},
        normals::Vector{Vector{T}},
    ) where {T<:Real}
        @assert length(data) ==
            length(is_boundary) ==
            length(boundary_conditions) ==
            length(normals)
        return new{T}(data, is_boundary, boundary_conditions, normals)
    end
end

#pre-allocation constructor
function HermiteStencilData{T}(k::Int, dim::Int) where {T<:Real}
    data = [Vector{T}(undef, dim) for _ in 1:k]  # Pre-allocate with correct dimension
    is_boundary = falses(k)
    boundary_conditions = [Dirichlet(T) for _ in 1:k]
    normals = [Vector{T}(undef, dim) for _ in 1:k]  # Pre-allocate with correct dimension
    return HermiteStencilData(data, is_boundary, boundary_conditions, normals)
end

"""
Populate local boundary information for a specific stencil within a kernel.
This function extracts boundary data for the neighbors of eval_idx and fills 
the pre-allocated HermiteBoundaryInfo structure.

# Arguments
- `boundary_info`: Pre-allocated HermiteBoundaryInfo structure to fill (for this batch)
- `eval_idx`: Current evaluation point index
- `adjl`: Adjacency list for eval_idx (the neighbors)  
- `is_boundary`: Global is_boundary vector for all points
- `boundary_conditions`: Global boundary_conditions vector for all points
- `normals`: Global normals vector for all points
"""
function update_stencil_data!(
    hermite_data::HermiteStencilData{T},  # Pre-allocated structure passed in
    global_data::AbstractVector{Vector{T}},
    neighbors::Vector{Int},  # adjl[eval_idx]
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{BoundaryCondition{T}},
    normals::Vector{Vector{T}},
    global_to_boundary::Vector{Int},
) where {T}
    k = length(neighbors)

    # Fill local boundary info for each neighbor (in-place, no allocation)
    @inbounds for local_idx in 1:k
        global_idx = neighbors[local_idx]
        hermite_data.data[local_idx] .= global_data[global_idx]
        hermite_data.is_boundary[local_idx] = is_boundary[global_idx]

        if is_boundary[global_idx]
            boundary_idx = global_to_boundary[global_idx]
            hermite_data.boundary_conditions[local_idx] = boundary_conditions[boundary_idx]
            hermite_data.normals[local_idx] .= normals[boundary_idx]
        else
            # Set default Dirichlet for interior points (not used but keeps type consistency)
            hermite_data.boundary_conditions[local_idx] = Dirichlet(T)
            fill!(hermite_data.normals[local_idx], zero(T))
        end
    end

    return nothing
end

# Trait types for dispatch
abstract type StencilType end
struct InternalStencil <: StencilType end
struct DirichletStencil <: StencilType end
struct HermiteStencil <: StencilType end

function stencil_type(
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{BoundaryCondition},
    eval_idx::Int,
    neighbors::Vector{Int},
    global_to_boundary::Vector{Int},
)
    if sum(is_boundary[neighbors]) == 0
        return InternalStencil()
    elseif is_boundary[eval_idx] &&
        is_dirichlet(boundary_conditions[global_to_boundary[eval_idx]])
        return DirichletStencil()
    else
        return HermiteStencil()
    end
end
