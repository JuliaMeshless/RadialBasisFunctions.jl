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
is_internal(bc::BoundaryCondition) = iszero(bc.α) && iszero(bc.β)

# Constructor helpers
Dirichlet(::Type{T}=Float64) where {T<:Real} = BoundaryCondition(one(T), zero(T))
Neumann(::Type{T}=Float64) where {T<:Real} = BoundaryCondition(zero(T), one(T))
Robin(α::Real, β::Real) = BoundaryCondition(α, β)
Internal(::Type{T}=Float64) where {T<:Real} = BoundaryCondition(zero(T), zero(T))

# Boundary information for a local stencil
"""
    HermiteStencilData{T}

Local stencil data structure for Hermite interpolation with boundary conditions.

This struct is meant to be used to correctly broadcast the build_stencil() function.
When Hermite scheme is used, it can be given to _build_stencil!() in place of the sole data.

# Fields
- `data`: Coordinates of stencil points (stored as Vector{T} for efficiency)
- `is_boundary`: Boolean flags indicating which points are on the boundary
- `boundary_conditions`: Boundary condition for each point. For interior points 
  (where `is_boundary[i] == false`), contains `Internal()` sentinel values that 
  should NOT be used. Always check `is_boundary[i]` before accessing.
- `normals`: Normal vectors for boundary points. For interior points, contains 
  zero vectors. Always check `is_boundary[i]` before accessing.

# Note
The `boundary_conditions` and `normals` arrays have the same length as `data` and
`is_boundary`, but only the entries where `is_boundary[i] == true` contain meaningful
data. Interior point entries are filled with sentinel `Internal()` and zero vectors
for type stability and array reuse in parallel kernels.
"""
struct HermiteStencilData{T<:Real}
    data::AbstractVector{Vector{T}}
    is_boundary::Vector{Bool}
    boundary_conditions::Vector{BoundaryCondition{T}}
    normals::Vector{Vector{T}}

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
    boundary_conditions = [Internal(T) for _ in 1:k]
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
            # Set default Internal for interior points (not used but keeps type consistency)
            hermite_data.boundary_conditions[local_idx] = Internal(T)
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

"""
Determine stencil type based on evaluation point properties.

NOTE: Classification is based ONLY on the evaluation point, not the stencil neighbors.
This simplifies dispatch:
- InternalStencil: eval_point is interior (use standard RBF)
- DirichletStencil: eval_point has Dirichlet BC (identity row)
- HermiteStencil: eval_point has Neumann/Robin BC (apply boundary operator)

Interior points with boundary neighbors are handled by InternalStencil with standard solve.jl.
"""
function stencil_type(
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{BoundaryCondition{T}},
    eval_idx::Int,
    # neighbors::Vector{Int},
    global_to_boundary::Vector{Int},
)::StencilType where {T}
    !is_boundary[eval_idx] && return InternalStencil()

    boundary_idx = global_to_boundary[eval_idx]
    is_dirichlet(boundary_conditions[boundary_idx]) && return DirichletStencil()

    return HermiteStencil()  # Neumann or Robin
end