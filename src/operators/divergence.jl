"""
    Divergence{Dim} <: AbstractGradientOperator{Dim,0}

Operator for the divergence of a vector field (∇⋅u = ∑ᵢ ∂uᵢ/∂xᵢ).

Takes a vector field (Matrix N×D) as input and produces a scalar field (Vector N).
Weights are stored as `NTuple{Dim, SparseMatrixCSC}`, one partial derivative matrix
per spatial dimension, reusing the Jacobian weight-building infrastructure.
"""
struct Divergence{Dim} <: AbstractGradientOperator{Dim, 0} end

# ============================================================================
# Evaluation — vector field input only
# ============================================================================

_alloc_output(::Divergence, x, ::Type{T}, n) where {T} = similar(x, T, n)

# In-place
function _eval_op(op::RadialBasisOperator{<:Divergence}, y::AbstractVector, x::AbstractMatrix)
    D = length(op.weights)
    T = eltype(y)
    mul!(y, op.weights[1], view(x, :, 1))
    for d in 2:D
        mul!(y, op.weights[d], view(x, :, d), one(T), one(T))
    end
    return y
end

# SparseVector weights (single eval point)
function _eval_op(
        op::RadialBasisOperator{<:Divergence, <:NTuple{<:Any, <:SparseVector}},
        x::AbstractMatrix,
    )
    D = length(op.weights)
    result = dot(op.weights[1], view(x, :, 1))
    for d in 2:D
        result += dot(op.weights[d], view(x, :, d))
    end
    return result
end

# ============================================================================
# Convenience constructors
# ============================================================================

"""
    divergence(data; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for the divergence (∇⋅u).

# Arguments
- `data`: Vector of data points

# Keyword Arguments
$(KWARG_DOCS)

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
div_op = divergence(points)

# Vector field as matrix (N × D)
u = hcat(sin.(getindex.(points, 1)), cos.(getindex.(points, 2)))
div_u = div_op(u)  # Vector (N,)
```

See also: [`curl`](@ref), [`gradient`](@ref), [`jacobian`](@ref)
"""
function divergence(data::AbstractVector{<:AbstractVector}; kw...)
    Dim = length(first(data))
    return RadialBasisOperator(Divergence{Dim}(), data; kw...)
end

"""
    divergence(data, x; basis=PHS(3; poly_deg=2), k, adjl)

One-shot convenience function that creates a divergence operator and applies it to vector field `x`.

For repeated evaluations on the same points, prefer creating the operator once with
[`divergence(data)`](@ref) and calling it via functor syntax `op(x)`.
"""
function divergence(data::AbstractVector{<:AbstractVector}, x::AbstractMatrix; kw...)
    op = divergence(data; kw...)
    return op(x)
end

# pretty printing
print_op(::Divergence) = "Divergence (∇⋅)"
