"""
    Curl{Dim} <: AbstractOperator{0}

Operator for the curl of a vector field (∇×u).

- **2D**: Matrix `(N×2)` → Vector `(N)`: `∂u₂/∂x₁ − ∂u₁/∂x₂`
- **3D**: Matrix `(N×3)` → Matrix `(N×3)`: standard curl vector

Only defined for `Dim ∈ {2, 3}`. Weights are stored as `NTuple{Dim, SparseMatrixCSC}`,
reusing the Jacobian weight-building infrastructure.
"""
struct Curl{Dim} <: AbstractOperator{0}
    function Curl{Dim}() where {Dim}
        (Dim == 2 || Dim == 3) || throw(ArgumentError("Curl is only defined for 2D and 3D, got $(Dim)D"))
        return new{Dim}()
    end
end

function (::Curl{Dim})(basis) where {Dim}
    return ntuple(dim -> ∂(basis, dim), Dim)
end

# ============================================================================
# Evaluation
# ============================================================================

function _eval_op(op::RadialBasisOperator{<:Curl}, x::AbstractVector)
    throw(ArgumentError(
        "Curl requires a vector field (Matrix input, N×D), got a Vector. " *
        "Each column should be a component of the vector field."
    ))
end

# 2D curl: (N×2) → (N,)  =  ∂u₂/∂x₁ − ∂u₁/∂x₂
function _eval_op(op::RadialBasisOperator{<:Curl{2}}, x::AbstractMatrix)
    N_eval = length(op.eval_points)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, N_eval)
    # out = ∂/∂x₁ * u₂
    mul!(out, op.weights[1], view(x, :, 2))
    # out -= ∂/∂x₂ * u₁
    mul!(out, op.weights[2], view(x, :, 1), -one(T), one(T))
    return out
end

# 3D curl: (N×3) → (N×3)
function _eval_op(op::RadialBasisOperator{<:Curl{3}}, x::AbstractMatrix)
    N_eval = length(op.eval_points)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, N_eval, 3)
    # component 1: ∂u₃/∂x₂ − ∂u₂/∂x₃
    mul!(view(out, :, 1), op.weights[2], view(x, :, 3))
    mul!(view(out, :, 1), op.weights[3], view(x, :, 2), -one(T), one(T))
    # component 2: ∂u₁/∂x₃ − ∂u₃/∂x₁
    mul!(view(out, :, 2), op.weights[3], view(x, :, 1))
    mul!(view(out, :, 2), op.weights[1], view(x, :, 3), -one(T), one(T))
    # component 3: ∂u₂/∂x₁ − ∂u₁/∂x₂
    mul!(view(out, :, 3), op.weights[1], view(x, :, 2))
    mul!(view(out, :, 3), op.weights[2], view(x, :, 1), -one(T), one(T))
    return out
end

# In-place 2D
function _eval_op(op::RadialBasisOperator{<:Curl{2}}, y::AbstractVector, x::AbstractMatrix)
    T = eltype(y)
    mul!(y, op.weights[1], view(x, :, 2))
    mul!(y, op.weights[2], view(x, :, 1), -one(T), one(T))
    return y
end

# In-place 3D
function _eval_op(op::RadialBasisOperator{<:Curl{3}}, y::AbstractMatrix, x::AbstractMatrix)
    T = eltype(y)
    mul!(view(y, :, 1), op.weights[2], view(x, :, 3))
    mul!(view(y, :, 1), op.weights[3], view(x, :, 2), -one(T), one(T))
    mul!(view(y, :, 2), op.weights[3], view(x, :, 1))
    mul!(view(y, :, 2), op.weights[1], view(x, :, 3), -one(T), one(T))
    mul!(view(y, :, 3), op.weights[1], view(x, :, 2))
    mul!(view(y, :, 3), op.weights[2], view(x, :, 1), -one(T), one(T))
    return y
end

# SparseVector weights (single eval point) — 2D
function _eval_op(
        op::RadialBasisOperator{<:Curl{2}, <:NTuple{<:Any, <:SparseVector}},
        x::AbstractMatrix,
    )
    return dot(op.weights[1], view(x, :, 2)) - dot(op.weights[2], view(x, :, 1))
end

# SparseVector weights (single eval point) — 3D
function _eval_op(
        op::RadialBasisOperator{<:Curl{3}, <:NTuple{<:Any, <:SparseVector}},
        x::AbstractMatrix,
    )
    T = promote_type(eltype(x), eltype(first(op.weights)))
    return SVector{3, T}(
        dot(op.weights[2], view(x, :, 3)) - dot(op.weights[3], view(x, :, 2)),
        dot(op.weights[3], view(x, :, 1)) - dot(op.weights[1], view(x, :, 3)),
        dot(op.weights[1], view(x, :, 2)) - dot(op.weights[2], view(x, :, 1)),
    )
end

# ============================================================================
# Convenience constructors
# ============================================================================

"""
    curl(data; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for the curl (∇×u).

Only defined for 2D and 3D data.

- **2D**: Returns scalar field `∂u₂/∂x₁ − ∂u₁/∂x₂`
- **3D**: Returns vector field with standard curl components

# Arguments
- `data`: Vector of data points

# Keyword Arguments
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation

# Examples
```julia
# 2D curl (scalar output)
points = [SVector{2}(rand(2)) for _ in 1:1000]
curl_op = curl(points)
u = hcat(-getindex.(points, 2), getindex.(points, 1))  # u = (-y, x)
ω = curl_op(u)  # ≈ 2.0 everywhere

# 3D curl (vector output)
points3d = [SVector{3}(rand(3)) for _ in 1:1000]
curl_op3d = curl(points3d)
u3d = hcat(-getindex.(points3d, 2), getindex.(points3d, 1), zeros(1000))
ω3d = curl_op3d(u3d)  # ≈ [0, 0, 2] everywhere
```

See also: [`divergence`](@ref), [`gradient`](@ref), [`jacobian`](@ref)
"""
function curl(data::AbstractVector{<:AbstractVector}; kw...)
    Dim = length(first(data))
    return RadialBasisOperator(Curl{Dim}(), data; kw...)
end

"""
    curl(data, x; basis=PHS(3; poly_deg=2), k, adjl)

One-shot convenience function that creates a curl operator and applies it to vector field `x`.

For repeated evaluations on the same points, prefer creating the operator once with
[`curl(data)`](@ref) and calling it via functor syntax `op(x)`.
"""
function curl(data::AbstractVector{<:AbstractVector}, x::AbstractMatrix; kw...)
    op = curl(data; kw...)
    return op(x)
end

# pretty printing
print_op(::Curl{2}) = "Curl 2D (∇×)"
print_op(::Curl{3}) = "Curl 3D (∇×)"
