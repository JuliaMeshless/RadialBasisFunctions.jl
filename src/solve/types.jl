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
struct BoundaryCondition{T <: Real}
    α::T
    β::T

    function BoundaryCondition(α::A, β::B) where {A, B}
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
Dirichlet((::Type{T}) = Float64) where {T <: Real} = BoundaryCondition(one(T), zero(T))
Neumann((::Type{T}) = Float64) where {T <: Real} = BoundaryCondition(zero(T), one(T))
Robin(α::Real, β::Real) = BoundaryCondition(α, β)
Internal((::Type{T}) = Float64) where {T <: Real} = BoundaryCondition(zero(T), zero(T))

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
- `normal_workspace`: Pre-allocated scratch for in-place normal-derivative evaluation
- `eval_local_idx`: Local index of the evaluation point among the stencil points
  (set by `update_hermite_stencil_data!`; 0 when absent/unset)

Note: For interior points (is_boundary[i] == false), boundary_conditions[i]
and normals[i] contain sentinel values and should not be accessed.
"""
struct HermiteStencilData{T <: Real}
    data::Vector{Vector{T}}
    is_boundary::Vector{Bool}
    boundary_conditions::Vector{BoundaryCondition{T}}
    normals::Vector{Vector{T}}
    poly_workspace::Vector{T}  # Pre-allocated buffer for polynomial operations
    normal_workspace::Vector{T}  # Scratch for in-place normal-derivative evaluation
    eval_local_idx::Base.RefValue{Int}

    function HermiteStencilData(
            data::AbstractVector{<:AbstractVector{T}},
            is_boundary::Vector{Bool},
            boundary_conditions::Vector{BoundaryCondition{T}},
            normals::AbstractVector{<:AbstractVector{T}},
            poly_workspace::Vector{T} = Vector{T}(undef, 0),
        ) where {T <: Real}
        if !(
                length(data) ==
                    length(is_boundary) ==
                    length(boundary_conditions) ==
                    length(normals)
            )
            throw(
                DimensionMismatch(
                    "HermiteStencilData requires equal lengths, got data = " *
                        "$(length(data)), is_boundary = $(length(is_boundary)), " *
                        "boundary_conditions = $(length(boundary_conditions)), " *
                        "normals = $(length(normals))",
                ),
            )
        end

        # Convert to Vector{Vector{T}} for internal storage
        data_vectors = [Vector{T}(point) for point in data]
        normals_vectors = [Vector{T}(normal) for normal in normals]

        return new{T}(
            data_vectors, is_boundary, boundary_conditions, normals_vectors,
            poly_workspace, Vector{T}(undef, length(poly_workspace)), Ref(0),
        )
    end
end

"""Pre-allocation constructor for HermiteStencilData"""
function HermiteStencilData{T}(k::Int, dim::Int, nmon::Int = 0) where {T <: Real}
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
                                 global_to_boundary[, eval_point])

Populate local Hermite stencil data from global arrays.
Used within kernels to extract boundary info for specific neighborhoods.

When `eval_point` is given, caches its local index (first stencil point equal by
value, 0 if absent) in `hermite_data.eval_local_idx` so RHS assembly avoids a
per-operator search. When it is `nothing` the cache is set to `-1`, which tells
`_mono_rhs!` the evaluation point's own boundary condition must not be applied
and suppresses its fallback search.
"""
function update_hermite_stencil_data!(
        hermite_data::HermiteStencilData{T},
        global_data::AbstractVector{<:AbstractVector{T}},
        neighbors::Vector{Int},
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{BoundaryCondition{T}},
        normals::AbstractVector{<:AbstractVector{T}},
        global_to_boundary::Vector{Int},
        eval_point = nothing,
    ) where {T}
    k = length(neighbors)
    # -1: no eval point supplied, so its BC must NOT be applied (eval_bc = false).
    #  0: eval point supplied but not yet located; `_mono_rhs!` may scan for it.
    hermite_data.eval_local_idx[] = eval_point === nothing ? -1 : 0

    @inbounds for local_idx in 1:k
        global_idx = neighbors[local_idx]
        hermite_data.data[local_idx] .= global_data[global_idx]
        hermite_data.is_boundary[local_idx] = is_boundary[global_idx]

        if eval_point !== nothing && hermite_data.eval_local_idx[] == 0 &&
                global_data[global_idx] == eval_point
            hermite_data.eval_local_idx[] = local_idx
        end

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
    BoundaryData{T,V}

Wrapper for global boundary information (replaces fragile tuples).
"""
struct BoundaryData{T, V <: AbstractVector{T}}
    is_boundary::Vector{Bool}
    boundary_conditions::Vector{BoundaryCondition{T}}
    normals::Vector{V}
end

# ============================================================================
# Stencil Classification Types
# ============================================================================

"""Trait types for stencil classification"""
abstract type StencilType end
struct InteriorStencil <: StencilType end  # All neighbors are interior
struct DirichletStencil <: StencilType end  # Eval point is Dirichlet BC
struct HermiteStencil <: StencilType end    # Mixed interior/boundary

"""Early-exit scan; avoids materializing `is_boundary[neighbors]` per stencil."""
@inline function _any_boundary(is_boundary, neighbors)
    for n in neighbors
        is_boundary[n] && return true
    end
    return false
end

"""
    build_eval_maps(data, is_boundary, eval_points) -> (eval_bnd, eval_data)

Value-based maps for the evaluation points:

- `eval_bnd[i] > 0` is the position of evaluation point `i` in the boundary
  arrays (`boundary_conditions`/`normals`) when it coincides with a boundary
  data point, 0 otherwise.
- `eval_data[i] > 0` is the index of evaluation point `i` in `data` when it
  coincides with ANY data point (boundary or interior), 0 otherwise — e.g. for
  oversampled collocation points, which are interior by definition.

Exact-equality lookup: safe only because callers build `eval_points` from the
same point objects (e.g. vcat of subsets of `data`). A recomputed coordinate
would compare unequal and silently classify as interior/not-found.
"""
function build_eval_maps(
        data::AbstractVector, is_boundary::Vector{Bool}, eval_points::AbstractVector
    )
    data_index = Dict{eltype(data), Int}()
    for i in eachindex(data)
        data_index[data[i]] = i
    end
    g2b = construct_global_to_boundary(is_boundary)
    eval_bnd = Vector{Int}(undef, length(eval_points))
    eval_data = Vector{Int}(undef, length(eval_points))
    for i in eachindex(eval_points)
        di = get(data_index, eval_points[i], 0)
        eval_data[i] = di
        eval_bnd[i] = di > 0 ? g2b[di] : 0
    end
    return eval_bnd, eval_data
end

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
    if !_any_boundary(is_boundary, neighbors)
        return InteriorStencil()
    elseif is_boundary[eval_idx] &&
            is_dirichlet(boundary_conditions[global_to_boundary[eval_idx]])
        return DirichletStencil()
    else
        return HermiteStencil()
    end
end

"""
    classify_stencil(is_boundary, boundary_conditions, eval_bnd, eval_idx,
                    neighbors, global_to_boundary)

Value-based variant: the evaluation point's boundary status comes from
`eval_bnd` (see [`build_eval_boundary_map`](@ref)) rather than from indexing
`is_boundary` by `eval_idx`, which is only valid when `eval_points === data`.
"""
function classify_stencil(
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{<:BoundaryCondition},
        eval_bnd::Vector{Int},
        eval_idx::Int,
        neighbors::Vector{Int},
        global_to_boundary::Vector{Int},
    )
    if !_any_boundary(is_boundary, neighbors)
        return InteriorStencil()
    elseif eval_bnd[eval_idx] > 0 &&
            is_dirichlet(boundary_conditions[eval_bnd[eval_idx]])
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
# Operator Arity Helpers (direct Tuple dispatch)
# ============================================================================

_num_ops(::Tuple{Vararg{Any, N}}) where {N} = N
_num_ops(_) = 1

_prepare_buffer(::Tuple{Vararg{Any, N}}, T, n) where {N} = zeros(T, n, N)
_prepare_buffer(_, T, n) = zeros(T, n)

# ============================================================================
# Basis Operators Bundle (for hot loop optimization)
# ============================================================================

"""
    BasisOperators{B,G,Hess}

Bundle of pre-constructed basis operators for efficient evaluation in hot loops.
Avoids repeated functor construction inside `hermite_rbf_dispatch`.

Fields:
- `φ`: The basis function itself
- `∇φ`: Gradient operator (pre-constructed ∇(basis))
- `Hφ`: Hessian operator (pre-constructed H(basis))

Usage:
```julia
ops = BasisOperators(basis)
# In hot loop:
φ_val = ops.φ(x, xᵢ)
grad = ops.∇φ(x, xᵢ)      # Returns vector
hess = ops.Hφ(x, xᵢ)      # Returns matrix
Dφ = dot(n, grad)         # Directional derivative
D²φ = dot(ni, hess * nj)  # Second directional derivative
```
"""
struct BasisOperators{B <: AbstractRadialBasis, G, Hess}
    φ::B
    ∇φ::G
    Hφ::Hess
end

"""Construct BasisOperators from a basis function."""
BasisOperators(basis::AbstractRadialBasis) = BasisOperators(basis, ∇(basis), H(basis))
