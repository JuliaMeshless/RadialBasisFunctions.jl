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

"""
    jacobian(data, basis; k, adjl)

Build a `RadialBasisOperator` for computing Jacobians (or gradients for scalar fields).

The Jacobian is the fundamental differential operator. For a scalar field, it computes
the gradient. For a vector field, it computes the full Jacobian matrix.

# Arguments
- `data`: Vector of points (e.g., `Vector{SVector{2,Float64}}`)
- `basis`: RBF basis function (default: `PHS(3; poly_deg=2)`)

# Keyword Arguments
- `k`: Stencil size (neighbors per point)
- `adjl`: Precomputed adjacency list

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
op = jacobian(points, PHS(3; poly_deg=2))

# Scalar field → gradient
u = sin.(getindex.(points, 1))
∇u = op(u)  # Matrix (1000 × 2)

# Vector field → Jacobian matrix
v = hcat(u, cos.(getindex.(points, 2)))
J = op(v)  # Array (1000 × 2 × 2)
```

See also: [`gradient`](@ref)
"""
function jacobian(
    data::AbstractVector{TD},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {TD<:AbstractVector,B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Jacobian{Dim}()
    return RadialBasisOperator(ℒ, data, basis; k=k, adjl=adjl)
end

"""
    jacobian(data, eval_points, basis; k, adjl)

Build a `RadialBasisOperator` for computing Jacobians at specified evaluation points.

# Arguments
- `data`: Vector of data points
- `eval_points`: Vector of points where the Jacobian will be evaluated
- `basis`: RBF basis function (default: `PHS(3; poly_deg=2)`)

# Keyword Arguments
- `k`: Stencil size (neighbors per point)
- `adjl`: Precomputed adjacency list
"""
function jacobian(
    data::AbstractVector{TD},
    eval_points::AbstractVector{TE},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {TD<:AbstractVector,TE<:AbstractVector,B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Jacobian{Dim}()
    return RadialBasisOperator(ℒ, data, eval_points, basis; k=k, adjl=adjl)
end

"""
    jacobian(data, eval_points, basis, is_boundary, boundary_conditions, normals; k, adjl)

Build a Hermite-compatible `RadialBasisOperator` for computing Jacobians.
The additional boundary information enables Hermite interpolation with proper boundary condition handling.

# Arguments
- `data`: Vector of data points
- `eval_points`: Vector of points where the Jacobian will be evaluated
- `basis`: RBF basis function
- `is_boundary`: Boolean vector indicating boundary points
- `boundary_conditions`: Vector of boundary conditions
- `normals`: Vector of normal vectors at boundary points

# Keyword Arguments
- `k`: Stencil size (neighbors per point)
- `adjl`: Precomputed adjacency list
"""
function jacobian(
    data::AbstractVector{TD},
    eval_points::AbstractVector{TE},
    basis::B,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {TD<:AbstractVector,TE<:AbstractVector,B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Jacobian{Dim}()
    return RadialBasisOperator(
        ℒ,
        data,
        eval_points,
        basis,
        is_boundary,
        boundary_conditions,
        normals;
        k=k,
        adjl=adjl,
    )
end

"""
    jacobian(data, x, basis; k, adjl)

One-shot convenience function that creates a Jacobian operator and applies it to field `x`.

For repeated evaluations on the same points, prefer creating the operator once with
[`jacobian(data, basis)`](@ref) and calling it via functor syntax `op(x)`.

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
    op = jacobian(data, basis; k=k, adjl=adjl)
    return op(x)
end

# pretty printing
print_op(::Jacobian) = "Jacobian (J)"
