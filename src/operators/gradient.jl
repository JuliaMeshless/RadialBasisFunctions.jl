# Gradient is a convenience alias for Jacobian applied to scalar fields.
# The Jacobian is the fundamental operator; gradient provides familiar naming
# for users working with scalar fields.

"""
    gradient(data; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for computing gradients of scalar fields.

This is a convenience alias for [`jacobian`](@ref). The gradient of a scalar field
is mathematically the Jacobian (a 1×D row vector, returned as a length-D vector per point).

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
op = gradient(points)

u = sin.(getindex.(points, 1))
∇u = op(u)  # Matrix (1000 × 2)
∂u_∂x = ∇u[:, 1]
∂u_∂y = ∇u[:, 2]
```

See also: [`jacobian`](@ref)
"""
gradient(data::AbstractVector{<:AbstractVector}; kw...) = jacobian(data; kw...)

# Backward compatible positional signatures (all delegate to jacobian)
gradient(data::AbstractVector{<:AbstractVector}, basis::AbstractRadialBasis; kw...) =
    jacobian(data, basis; kw...)

gradient(data::AbstractVector{<:AbstractVector}, eval_points::AbstractVector{<:AbstractVector},
         basis::AbstractRadialBasis=PHS(3; poly_deg=2); kw...) =
    jacobian(data, eval_points, basis; kw...)

# Hermite backward compatibility
function gradient(
    data::AbstractVector{<:AbstractVector},
    eval_points::AbstractVector{<:AbstractVector},
    basis::AbstractRadialBasis,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    kw...,
)
    return jacobian(data, eval_points, basis, is_boundary, boundary_conditions, normals; kw...)
end

# One-shot convenience (delegate to jacobian's one-shot)
"""
    gradient(data, x; basis=PHS(3; poly_deg=2), k, adjl)

One-shot convenience function that creates a gradient operator and applies it to scalar field `x`.

For repeated evaluations, prefer creating the operator once with [`gradient(data)`](@ref).

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
u = sin.(getindex.(points, 1))
∇u = gradient(points, u)  # One-shot gradient computation
```
"""
gradient(data::AbstractVector{<:AbstractVector}, x; kw...) = jacobian(data, x; kw...)

# Note: No print_op needed - gradient returns a Jacobian operator, which has its own print_op
