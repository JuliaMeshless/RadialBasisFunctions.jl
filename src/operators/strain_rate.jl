"""
    StrainRate{Dim} <: AbstractGradientOperator{Dim,0}

Operator for the symmetric strain rate tensor εᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ).

Takes a vector field (Matrix N×D) as input and produces a symmetric tensor
(Array N_eval×D×D). Weights are stored as `NTuple{Dim, SparseMatrixCSC}`,
reusing the Jacobian weight-building infrastructure.
"""
struct StrainRate{Dim} <: AbstractGradientOperator{Dim, 0} end

# ============================================================================
# Evaluation — vector field input only
# ============================================================================

function _eval_op(op::RadialBasisOperator{<:StrainRate}, x::AbstractMatrix)
    D = length(op.weights)
    N_eval = length(op.eval_points)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, N_eval, D, D)
    half = T(0.5)
    n_unique = _num_sym(D)
    for k in 1:n_unique
        i, j = _kth_sym_pair(k, D)
        if i == j
            # εᵢᵢ = ∂uᵢ/∂xᵢ
            mul!(view(out, :, i, i), op.weights[i], view(x, :, i))
        else
            # εᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
            mul!(view(out, :, i, j), op.weights[j], view(x, :, i))
            mul!(view(out, :, i, j), op.weights[i], view(x, :, j), one(T), one(T))
            view(out, :, i, j) .*= half
            copyto!(view(out, :, j, i), view(out, :, i, j))
        end
    end
    return out
end

# In-place
function _eval_op(
        op::RadialBasisOperator{<:StrainRate}, y::AbstractArray{<:Any, 3}, x::AbstractMatrix
    )
    D = length(op.weights)
    T = eltype(y)
    half = T(0.5)
    n_unique = _num_sym(D)
    for k in 1:n_unique
        i, j = _kth_sym_pair(k, D)
        if i == j
            mul!(view(y, :, i, i), op.weights[i], view(x, :, i))
        else
            mul!(view(y, :, i, j), op.weights[j], view(x, :, i))
            mul!(view(y, :, i, j), op.weights[i], view(x, :, j), one(T), one(T))
            view(y, :, i, j) .*= half
            copyto!(view(y, :, j, i), view(y, :, i, j))
        end
    end
    return y
end

# SparseVector weights (single eval point)
function _eval_op(
        op::RadialBasisOperator{<:StrainRate, <:NTuple{<:Any, <:SparseVector}},
        x::AbstractMatrix,
    )
    D = length(op.weights)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, D, D)
    half = T(0.5)
    n_unique = _num_sym(D)
    @inbounds for k in 1:n_unique
        i, j = _kth_sym_pair(k, D)
        if i == j
            out[i, i] = dot(op.weights[i], view(x, :, i))
        else
            val = half * (dot(op.weights[j], view(x, :, i)) + dot(op.weights[i], view(x, :, j)))
            out[i, j] = val
            out[j, i] = val
        end
    end
    return out
end

# ============================================================================
# Convenience constructors
# ============================================================================

"""
    strain_rate(data; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for the symmetric strain rate tensor
εᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ).

# Arguments
- `data`: Vector of data points

# Keyword Arguments
$(KWARG_DOCS)

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
ε_op = strain_rate(points)

# Vector field as matrix (N × D)
u = hcat(getindex.(points, 2), getindex.(points, 1))  # u = (y, x)
ε = ε_op(u)  # Array (1000 × 2 × 2), ε₁₂ = ε₂₁ ≈ 1.0
```

See also: [`rotation_rate`](@ref), [`jacobian`](@ref), [`divergence`](@ref)
"""
function strain_rate(data::AbstractVector{<:AbstractVector}; kw...)
    Dim = length(first(data))
    return RadialBasisOperator(StrainRate{Dim}(), data; kw...)
end

"""
    strain_rate(data, x; basis=PHS(3; poly_deg=2), k, adjl)

One-shot convenience function that creates a strain rate operator and applies it to vector field `x`.
"""
function strain_rate(data::AbstractVector{<:AbstractVector}, x::AbstractMatrix; kw...)
    op = strain_rate(data; kw...)
    return op(x)
end

# pretty printing
print_op(::StrainRate) = "Strain Rate (ε = ½(∇u + (∇u)ᵀ))"
