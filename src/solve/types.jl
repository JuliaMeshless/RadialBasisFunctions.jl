using StaticArraysCore: StaticVector

# ============================================================================
# Boundary Condition Types
# ============================================================================

"""
    BoundaryCondition{T}

Unified boundary condition representation: Bu = α*u + β*∂ₙu

Special cases:
- Dirichlet: α=1, β=0
- Neumann: α=0, β=1
- Robin: α≠0, β≠0
- Internal: α=0, β=0 (sentinel for interior points)
"""
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

# Predicates
is_dirichlet(bc::BoundaryCondition) = isone(bc.α) && iszero(bc.β)
is_neumann(bc::BoundaryCondition) = iszero(bc.α) && isone(bc.β)
is_robin(bc::BoundaryCondition) = !iszero(bc.α) && !iszero(bc.β)
is_internal(bc::BoundaryCondition) = iszero(bc.α) && iszero(bc.β)

# Constructors
Dirichlet((::Type{T})=Float64) where {T<:Real} = BoundaryCondition(one(T), zero(T))
Neumann((::Type{T})=Float64) where {T<:Real} = BoundaryCondition(zero(T), one(T))
Robin(α::Real, β::Real) = BoundaryCondition(α, β)
Internal((::Type{T})=Float64) where {T<:Real} = BoundaryCondition(zero(T), zero(T))

# ============================================================================
# Hermite Stencil Data
# ============================================================================

"""
    HermiteStencilData{T}

Local stencil data for Hermite interpolation with boundary conditions.

Fields:
- `data`: Coordinates of k stencil points
- `is_boundary`: Boolean flags for each point
- `boundary_conditions`: BC for each point (use Internal() for interior)
- `normals`: Normal vectors (zero for interior points)
- `poly_workspace`: Pre-allocated buffer for polynomial operations (avoids allocations in hot path)

Note: For interior points (is_boundary[i] == false), boundary_conditions[i]
and normals[i] contain sentinel values and should not be accessed.
"""
struct HermiteStencilData{T<:Real}
    data::AbstractVector{Vector{T}}
    is_boundary::Vector{Bool}
    boundary_conditions::Vector{BoundaryCondition{T}}
    normals::Vector{Vector{T}}
    poly_workspace::Vector{T}  # Pre-allocated buffer for polynomial operations

    function HermiteStencilData(
        data::AbstractVector{<:AbstractVector{T}},
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{BoundaryCondition{T}},
        normals::AbstractVector{<:AbstractVector{T}},
        poly_workspace::Vector{T}=Vector{T}(undef, 0),
    ) where {T<:Real}
        @assert length(data) ==
            length(is_boundary) ==
            length(boundary_conditions) ==
            length(normals)

        # Convert to Vector{Vector{T}} for internal storage
        data_vectors = [Vector{T}(point) for point in data]
        normals_vectors = [Vector{T}(normal) for normal in normals]

        return new{T}(
            data_vectors, is_boundary, boundary_conditions, normals_vectors, poly_workspace
        )
    end
end

"""Pre-allocation constructor for HermiteStencilData"""
function HermiteStencilData{T}(k::Int, dim::Int, nmon::Int=0) where {T<:Real}
    data = [Vector{T}(undef, dim) for _ in 1:k]
    is_boundary = Vector{Bool}(falses(k))
    boundary_conditions = [Internal(T) for _ in 1:k]
    normals = [Vector{T}(undef, dim) for _ in 1:k]
    poly_workspace = Vector{T}(undef, nmon)
    return HermiteStencilData(
        data, is_boundary, boundary_conditions, normals, poly_workspace
    )
end

"""
    update_hermite_stencil_data!(hermite_data, global_data, neighbors,
                                 is_boundary, boundary_conditions, normals,
                                 global_to_boundary)

Populate local Hermite stencil data from global arrays.
Used within kernels to extract boundary info for specific neighborhoods.
"""
function update_hermite_stencil_data!(
    hermite_data::HermiteStencilData{T},
    global_data::AbstractVector{<:AbstractVector{T}},
    neighbors::Vector{Int},
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{BoundaryCondition{T}},
    normals::AbstractVector{<:AbstractVector{T}},
    global_to_boundary::Vector{Int},
) where {T}
    k = length(neighbors)

    @inbounds for local_idx in 1:k
        global_idx = neighbors[local_idx]
        hermite_data.data[local_idx] .= global_data[global_idx]
        hermite_data.is_boundary[local_idx] = is_boundary[global_idx]

        if is_boundary[global_idx]
            boundary_idx = global_to_boundary[global_idx]
            hermite_data.boundary_conditions[local_idx] = boundary_conditions[boundary_idx]
            hermite_data.normals[local_idx] .= normals[boundary_idx]
        else
            hermite_data.boundary_conditions[local_idx] = Internal(T)
            fill!(hermite_data.normals[local_idx], zero(T))
        end
    end

    return nothing
end

# ============================================================================
# Boundary Data Wrapper
# ============================================================================

"""
    BoundaryData{T}

Wrapper for global boundary information (replaces fragile tuples).
"""
struct BoundaryData{T}
    is_boundary::Vector{Bool}
    boundary_conditions::Vector{BoundaryCondition{T}}
    normals::Vector{<:AbstractVector{T}}
end

# ============================================================================
# Stencil Classification Types
# ============================================================================

"""Trait types for stencil classification"""
abstract type StencilType end
struct InteriorStencil <: StencilType end  # All neighbors are interior
struct DirichletStencil <: StencilType end  # Eval point is Dirichlet BC
struct HermiteStencil <: StencilType end    # Mixed interior/boundary

"""
    classify_stencil(is_boundary, boundary_conditions, eval_idx,
                    neighbors, global_to_boundary)

Classify stencil type for dispatch in kernel execution.
"""
function classify_stencil(
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{BoundaryCondition{T}},
    eval_idx::Int,
    neighbors::Vector{Int},
    global_to_boundary::Vector{Int},
) where {T}
    if sum(is_boundary[neighbors]) == 0
        return InteriorStencil()
    elseif is_boundary[eval_idx] &&
        is_dirichlet(boundary_conditions[global_to_boundary[eval_idx]])
        return DirichletStencil()
    else
        return HermiteStencil()
    end
end

# Convenience wrapper for BoundaryData
function classify_stencil(
    boundary_data::BoundaryData,
    eval_idx::Int,
    neighbors::Vector{Int},
    global_to_boundary::Vector{Int},
)
    return classify_stencil(
        boundary_data.is_boundary,
        boundary_data.boundary_conditions,
        eval_idx,
        neighbors,
        global_to_boundary,
    )
end

# ============================================================================
# Point Type Classification (for Hermite dispatch)
# ============================================================================

"""Trait types for individual point boundary classification"""
abstract type BoundaryPointType end
struct InteriorPoint <: BoundaryPointType end
struct DirichletPoint <: BoundaryPointType end
struct NeumannRobinPoint <: BoundaryPointType end

"""Determine boundary type of a single point"""
@inline function point_type(is_bound::Bool, bc::BoundaryCondition)
    return if is_bound
        (is_dirichlet(bc) ? DirichletPoint() : NeumannRobinPoint())
    else
        InteriorPoint()
    end
end

# ============================================================================
# Operator Arity Traits (for type-stable buffer allocation)
# ============================================================================

"""Operator arity traits for compile-time dispatch"""
abstract type OperatorArity end
struct SingleOperator <: OperatorArity end
struct MultiOperator{N} <: OperatorArity end

"""Extract operator arity at compile time"""
operator_arity(::T) where {T} = operator_arity(T)
operator_arity(::Type{<:Tuple{Vararg{Any,N}}}) where {N} = MultiOperator{N}()
operator_arity(::Type) = SingleOperator()

"""Get number of operators (type-stable)"""
_num_ops(::SingleOperator) = 1
_num_ops(::MultiOperator{N}) where {N} = N
_num_ops(ℒ) = _num_ops(operator_arity(ℒ))

"""Prepare RHS buffer with correct type (type-stable)"""
_prepare_buffer(::SingleOperator, T, n) = zeros(T, n)
_prepare_buffer(::MultiOperator{N}, T, n) where {N} = zeros(T, n, N)
_prepare_buffer(ℒ, T, n) = _prepare_buffer(operator_arity(ℒ), T, n)
