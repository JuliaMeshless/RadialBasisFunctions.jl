"""
    Jacobian{Dim} <: VectorValuedOperator

Operator type for computing Jacobians (and gradients as a special case).

The Jacobian is the fundamental differential operator that computes all partial derivatives.
When applied to a scalar field, it produces the gradient. When applied to a vector field,
it produces the full Jacobian matrix.

Differentiation increases tensor rank by 1. The output gains a trailing dimension of size D
(the spatial dimension).

# Input/Output Shapes
- Scalar field `Vector{T}` (N,) → Gradient `Matrix{T}` (N_eval × D)
- Vector field `Matrix{T}` (N × D) → Jacobian `Array{T,3}` (N_eval × D × D)
- Matrix field `Array{T,3}` (N × D × D) → 3-tensor `Array{T,4}` (N_eval × D × D × D)
- General: input shape `(N, dims...)` → output shape `(N_eval, dims..., D)`
"""
struct Jacobian{Dim} <: VectorValuedOperator{Dim} end

function (op::Jacobian{Dim})(basis) where {Dim}
    return ntuple(dim -> ∂(basis, dim), Dim)
end

# Primary interface using unified keyword constructor
"""
    jacobian(data; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for computing Jacobians (or gradients for scalar fields).

The Jacobian is the fundamental differential operator. For a scalar field, it computes
the gradient. For a vector field, it computes the full Jacobian matrix. The spatial
dimension is automatically inferred from the data.

# Arguments
- `data`: Vector of points (e.g., `Vector{SVector{2,Float64}}`)

# Keyword Arguments
- `basis`: RBF basis function (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
op = jacobian(points)

# Scalar field → gradient
u = sin.(getindex.(points, 1))
∇u = op(u)  # Matrix (1000 × 2)

# Vector field → Jacobian matrix
v = hcat(u, cos.(getindex.(points, 2)))
J = op(v)  # Array (1000 × 2 × 2)
```

See also: [`gradient`](@ref)
"""
function jacobian(data::AbstractVector{<:AbstractVector}; kw...)
    Dim = length(first(data))
    return RadialBasisOperator(Jacobian{Dim}(), data; kw...)
end

# Backward compatible positional signatures
function jacobian(data::AbstractVector{<:AbstractVector}, basis::AbstractRadialBasis; kw...)
    Dim = length(first(data))
    return RadialBasisOperator(Jacobian{Dim}(), data; basis = basis, kw...)
end

# Note: Type constraint {<:AbstractVector} ensures this only matches vector-of-points,
# not field values like Vector{Float64}
function jacobian(
        data::AbstractVector{<:AbstractVector},
        eval_points::AbstractVector{<:AbstractVector},
        basis::AbstractRadialBasis = PHS(3; poly_deg = 2);
        kw...,
    )
    Dim = length(first(data))
    return RadialBasisOperator(
        Jacobian{Dim}(), data; eval_points = eval_points, basis = basis, kw...
    )
end

# Hermite backward compatibility (positional boundary arguments)
function jacobian(
        data::AbstractVector{<:AbstractVector},
        eval_points::AbstractVector{<:AbstractVector},
        basis::AbstractRadialBasis,
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{<:BoundaryCondition},
        normals::Vector{<:AbstractVector};
        kw...,
    )
    Dim = length(first(data))
    hermite = (is_boundary = is_boundary, bc = boundary_conditions, normals = normals)
    return RadialBasisOperator(
        Jacobian{Dim}(), data; eval_points = eval_points, basis = basis, hermite = hermite, kw...
    )
end

# One-shot convenience: create operator and apply immediately
"""
    jacobian(data, x; basis=PHS(3; poly_deg=2), k, adjl)

One-shot convenience function that creates a Jacobian operator and applies it to field `x`.

For repeated evaluations on the same points, prefer creating the operator once with
[`jacobian(data)`](@ref) and calling it via functor syntax `op(x)`.

# Arguments
- `data`: Vector of points
- `x`: Field values to differentiate

# Keyword Arguments
- `basis`: RBF basis function (default: `PHS(3; poly_deg=2)`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
u = sin.(getindex.(points, 1))
∇u = jacobian(points, u)  # One-shot gradient computation
```
"""
function jacobian(data::AbstractVector{<:AbstractVector}, x; kw...)
    op = jacobian(data; kw...)
    return op(x)
end

# pretty printing
print_op(::Jacobian) = "Jacobian (J)"
