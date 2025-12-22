"""
    jacobian(op::RadialBasisOperator{<:VectorValuedOperator}, x)

Compute the Jacobian of field `x` using the operator `op`.

Differentiation increases tensor rank by 1. The output gains a trailing dimension of size D
(the spatial dimension).

# Input/Output Shapes
- Scalar field `Vector{T}` (N,) → Gradient `Matrix{T}` (N_eval × D)
- Vector field `Matrix{T}` (N × D) → Jacobian `Array{T,3}` (N_eval × D × D)
- Matrix field `Array{T,3}` (N × D × D) → 3-tensor `Array{T,4}` (N_eval × D × D × D)
- General: input shape `(N, dims...)` → output shape `(N_eval, dims..., D)`

# Examples
```julia
# Create gradient/jacobian operator for 2D points
points = [SVector{2}(rand(2)) for _ in 1:1000]
op = gradient(points, PHS(3; poly_deg=2))

# Scalar field → gradient (Matrix)
u = sin.(getindex.(points, 1))
∇u = jacobian(op, u)  # Matrix (1000 × 2)

# Vector field → Jacobian (3-tensor)
v = hcat(u, cos.(getindex.(points, 2)))  # (1000 × 2)
J = jacobian(op, v)  # Array (1000 × 2 × 2)
# J[:, i, j] = ∂vᵢ/∂xⱼ
```

See also: [`gradient`](@ref), [`jacobian!`](@ref)
"""
function jacobian(op::RadialBasisOperator{<:VectorValuedOperator}, x)
    !is_cache_valid(op) && update_weights!(op)
    return _eval_op(op, x)
end

"""
    jacobian!(out, op::RadialBasisOperator{<:VectorValuedOperator}, x)

In-place version of [`jacobian`](@ref). Compute the Jacobian and store in preallocated `out`.

# Required Output Shapes
- For scalar input `x::Vector{T}` (N,): `out::Matrix{T}` (N_eval × D)
- For vector input `x::Matrix{T}` (N × D): `out::Array{T,3}` (N_eval × D × D)

# Examples
```julia
op = gradient(points, PHS(3; poly_deg=2))
u = sin.(getindex.(points, 1))
out = Matrix{Float64}(undef, length(points), 2)
jacobian!(out, op, u)
```
"""
function jacobian!(out, op::RadialBasisOperator{<:VectorValuedOperator}, x)
    !is_cache_valid(op) && update_weights!(op)
    return _eval_op(op, out, x)
end

"""
    jacobian(data, x, basis=PHS(3; poly_deg=2); k, adjl)

Convenience function that creates a gradient operator and applies it to field `x`.

This combines operator construction and evaluation in a single call. For repeated
evaluations on the same points, prefer creating the operator once with [`gradient`](@ref)
and calling [`jacobian`](@ref) multiple times.

# Arguments
- `data`: Vector of points (e.g., `Vector{SVector{2,Float64}}`)
- `x`: Field values to differentiate
- `basis`: RBF basis function (default: `PHS(3; poly_deg=2)`)

# Keyword Arguments
- `k`: Stencil size (neighbors per point)
- `adjl`: Precomputed adjacency list

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
u = sin.(getindex.(points, 1))
∇u = jacobian(points, u)  # One-shot gradient computation
```
"""
function jacobian(
    data::AbstractVector,
    x,
    basis::B=PHS(3; poly_deg=2);
    k::Int=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {B<:AbstractRadialBasis}
    op = gradient(data, basis; k=k, adjl=adjl)
    return jacobian(op, x)
end

# pretty printing
print_op(::typeof(jacobian)) = "Jacobian (J)"
