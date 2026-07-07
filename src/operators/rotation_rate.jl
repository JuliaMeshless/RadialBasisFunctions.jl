"""
    RotationRate{Dim} <: AbstractJacobianOperator{Dim,0}

Operator for the anti-symmetric rotation rate tensor ωᵢⱼ = ½(∂uᵢ/∂xⱼ − ∂uⱼ/∂xᵢ).

Takes a vector field (Matrix N×D) as input and produces an anti-symmetric tensor
(Array N_eval×D×D). Diagonal entries are zero. Weights are stored as
`NTuple{Dim, SparseMatrixCSC}`, reusing the Jacobian weight-building infrastructure.
"""
struct RotationRate{Dim} <: AbstractJacobianOperator{Dim, 0} end

# ============================================================================
# Evaluation — vector field input only
# ============================================================================

_alloc_output(::RotationRate{Dim}, x, ::Type{T}, n) where {Dim, T} = similar(x, T, n, Dim, Dim)

# In-place; ωᵢⱼ = ½(∂uᵢ/∂xⱼ − ∂uⱼ/∂xᵢ), ωⱼᵢ = −ωᵢⱼ
function _eval_op(
        op::RadialBasisOperator{<:RotationRate}, y::AbstractArray{<:Any, 3}, x::AbstractMatrix
    )
    D = length(op.weights)
    T = eltype(y)
    half = T(0.5)
    fill!(y, zero(T))
    for i in 1:D, j in (i + 1):D
        mul!(view(y, :, i, j), op.weights[j], view(x, :, i))
        mul!(view(y, :, i, j), op.weights[i], view(x, :, j), -one(T), one(T))
        view(y, :, i, j) .*= half
        view(y, :, j, i) .= .-view(y, :, i, j)
    end
    return y
end

# SparseVector weights (single eval point)
function _eval_op(
        op::RadialBasisOperator{<:RotationRate, <:NTuple{<:Any, <:SparseVector}},
        x::AbstractMatrix,
    )
    D = length(op.weights)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = fill!(similar(x, T, D, D), zero(T))
    half = T(0.5)
    @inbounds for i in 1:D, j in (i + 1):D
        val = half * (dot(op.weights[j], view(x, :, i)) - dot(op.weights[i], view(x, :, j)))
        out[i, j] = val
        out[j, i] = -val
    end
    return out
end

# ============================================================================
# Convenience constructors
# ============================================================================

"""
    rotation_rate(data; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for the anti-symmetric rotation rate tensor
ωᵢⱼ = ½(∂uᵢ/∂xⱼ − ∂uⱼ/∂xᵢ).

# Arguments
- `data`: Vector of data points

# Keyword Arguments
$(KWARG_DOCS)

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
ω_op = rotation_rate(points)

# Solid body rotation: u = (-y, x) → ω₁₂ = -1
u = hcat(-getindex.(points, 2), getindex.(points, 1))
ω = ω_op(u)  # Array (1000 × 2 × 2), ω₁₂ ≈ -1.0
```

See also: [`strain_rate`](@ref), [`jacobian`](@ref), [`curl`](@ref)
"""
function rotation_rate(data::AbstractVector{<:AbstractVector}; kw...)
    Dim = length(first(data))
    return RadialBasisOperator(RotationRate{Dim}(), data; kw...)
end

"""
    rotation_rate(data, x; basis=PHS(3; poly_deg=2), k, adjl)

One-shot convenience function that creates a rotation rate operator and applies it to vector field `x`.
"""
function rotation_rate(data::AbstractVector{<:AbstractVector}, x::AbstractMatrix; kw...)
    op = rotation_rate(data; kw...)
    return op(x)
end

# pretty printing
print_op(::RotationRate) = "Rotation Rate (ω = ½(∇u − (∇u)ᵀ))"
