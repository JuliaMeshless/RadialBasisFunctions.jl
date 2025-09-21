"""
Boundary condition types and utilities for Hermite interpolation.
"""

using StaticArraysCore: StaticVector

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
    data::AbstractVector{Vector{T}}  # Coordinates of stencil points (stored as Vector{T} for efficiency)
    is_boundary::Vector{Bool}
    boundary_conditions::Vector{BoundaryCondition{T}}
    normals::Vector{Vector{T}}  # Normals stored as Vector{T}

    # Generic constructor that accepts both Vector{T} and StaticVector{N,T} inputs
    function HermiteStencilData(
        data::AbstractVector{<:AbstractVector{T}},  # Accept both Vector{T} and StaticVector{N,T}
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{BoundaryCondition{T}},
        normals::AbstractVector{<:AbstractVector{T}},  # Accept both Vector{T} and StaticVector{N,T}
    ) where {T<:Real}
        @assert length(data) ==
            length(is_boundary) ==
            length(boundary_conditions) ==
            length(normals)

        # Convert input data to Vector{Vector{T}} for internal storage
        # This handles both Vector{T} and StaticVector{N,T} inputs
        data_vectors = [Vector{T}(point) for point in data]
        normals_vectors = [Vector{T}(normal) for normal in normals]

        return new{T}(data_vectors, is_boundary, boundary_conditions, normals_vectors)
    end
end

#pre-allocation constructor
function HermiteStencilData{T}(k::Int, dim::Int) where {T<:Real}
    data = [Vector{T}(undef, dim) for _ in 1:k]  # Pre-allocate with correct dimension
    is_boundary = Vector{Bool}(falses(k))
    boundary_conditions = [Dirichlet(T) for _ in 1:k]
    normals = [Vector{T}(undef, dim) for _ in 1:k]  # Pre-allocate with correct dimension
    return HermiteStencilData(data, is_boundary, boundary_conditions, normals)
end

"""
Unified function that handles both Vector{T} and StaticVector{N,T} data types.
The key insight is that both types support broadcasting (`.=`) operations,
so we can write one generic implementation.

This function populates local boundary information for a specific stencil within a kernel,
extracting boundary data for the neighbors of eval_idx and filling the pre-allocated
HermiteStencilData structure.

# Arguments  
- `hermite_data`: Pre-allocated HermiteStencilData structure to fill
- `global_data`: Global data vector (accepts both Vector{T} and StaticVector{N,T})
- `neighbors`: Adjacency list for eval_idx (the neighbors)
- `is_boundary`: Global is_boundary vector for all points
- `boundary_conditions`: Global boundary_conditions vector for all points
- `normals`: Global normals vector for all points
"""
function update_stencil_data!(
    hermite_data::HermiteStencilData{T},  # Pre-allocated structure passed in
    global_data::AbstractVector{<:AbstractVector{T}},  # Accepts both Vector{T} and StaticVector{N,T}
    neighbors::Vector{Int},  # adjl[eval_idx]
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{BoundaryCondition{T}},
    normals::AbstractVector{<:AbstractVector{T}},  # Accepts both Vector{T} and StaticVector{N,T}
    global_to_boundary::Vector{Int},
) where {T}
    k = length(neighbors)

    # Fill local boundary info for each neighbor (in-place, no allocation)
    # This works for both Vector{T} and StaticVector{N,T} thanks to broadcasting
    @inbounds for local_idx in 1:k
        global_idx = neighbors[local_idx]
        hermite_data.data[local_idx] .= global_data[global_idx]  # Broadcasting works for both types
        hermite_data.is_boundary[local_idx] = is_boundary[global_idx]

        if is_boundary[global_idx]
            boundary_idx = global_to_boundary[global_idx]
            hermite_data.boundary_conditions[local_idx] = boundary_conditions[boundary_idx]
            hermite_data.normals[local_idx] .= normals[boundary_idx]  # Broadcasting works for both types
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
    boundary_conditions::Vector{BoundaryCondition{T}},
    eval_idx::Int,
    neighbors::Vector{Int},
    global_to_boundary::Vector{Int},
) where {T}
    if sum(is_boundary[neighbors]) == 0
        return InternalStencil()
    elseif is_boundary[eval_idx] &&
        is_dirichlet(boundary_conditions[global_to_boundary[eval_idx]])
        return DirichletStencil()
    else
        return HermiteStencil()
    end
end
