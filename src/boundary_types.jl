"""
Boundary condition types and utilities for Hermite interpolation.
"""

# Simple boundary condition type
struct BoundaryCondition{T<:Real}
    α::T
    β::T
    
    function BoundaryCondition(α::A, β::B) where {A,B}
        T = promote_type(A, B)
        new{T}(α, β)
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
struct HermiteBoundaryInfo{T<:Real}
    is_boundary::Vector{Bool}
    boundary_conditions::Vector{BoundaryCondition{T}}
    normals::Vector{Vector{T}}
    
    function HermiteBoundaryInfo(
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{BoundaryCondition{T}},
        normals::Vector{Vector{T}}
    ) where {T<:Real}
        @assert length(is_boundary) == length(boundary_conditions) == length(normals)
        new{T}(is_boundary, boundary_conditions, normals)
    end
end

# Trait types for dispatch
abstract type StencilType end
struct StandardStencil <: StencilType end
struct HermiteStencil <: StencilType end

# Trait function to determine stencil type
stencil_type(boundary_info::Nothing) = StandardStencil()
stencil_type(boundary_info::HermiteBoundaryInfo) = any(boundary_info.is_boundary) ? HermiteStencil() : StandardStencil()

# Convenience function to check if any boundary points in stencil
has_boundary_points(boundary_info::Nothing) = false
has_boundary_points(boundary_info::HermiteBoundaryInfo) = any(boundary_info.is_boundary)