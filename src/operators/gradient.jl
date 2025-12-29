# Gradient is a convenience alias for Jacobian applied to scalar fields.
# The Jacobian is the fundamental operator; gradient provides familiar naming
# for users working with scalar fields.

"""
    gradient(data, basis; k, adjl)

Build a `RadialBasisOperator` for computing gradients of scalar fields.

This is a convenience alias for [`jacobian`](@ref). The gradient of a scalar field
is mathematically the Jacobian (a 1×D row vector, returned as a length-D vector per point).

# Arguments
- `data`: Vector of points (e.g., `Vector{SVector{2,Float64}}`)
- `basis`: RBF basis function (default: `PHS(3; poly_deg=2)`)

# Keyword Arguments
- `k`: Stencil size (neighbors per point)
- `adjl`: Precomputed adjacency list

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
op = gradient(points, PHS(3; poly_deg=2))

u = sin.(getindex.(points, 1))
∇u = op(u)  # Matrix (1000 × 2)
∂u_∂x = ∇u[:, 1]
∂u_∂y = ∇u[:, 2]
```

See also: [`jacobian`](@ref)
"""
function gradient(
    data::AbstractVector{TD},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {TD<:AbstractVector,B<:AbstractRadialBasis,T<:Int}
    return jacobian(data, basis; k=k, adjl=adjl)
end

"""
    gradient(data, eval_points, basis; k, adjl)

Build a `RadialBasisOperator` for computing gradients at specified evaluation points.

This is a convenience alias for [`jacobian`](@ref).
"""
function gradient(
    data::AbstractVector{TD},
    eval_points::AbstractVector{TE},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {TD<:AbstractVector,TE<:AbstractVector,B<:AbstractRadialBasis,T<:Int}
    return jacobian(data, eval_points, basis; k=k, adjl=adjl)
end

"""
    gradient(data, eval_points, basis, is_boundary, boundary_conditions, normals; k, adjl)

Build a Hermite-compatible `RadialBasisOperator` for computing gradients.
This is a convenience alias for [`jacobian`](@ref) with Hermite interpolation support.
"""
function gradient(
    data::AbstractVector{TD},
    eval_points::AbstractVector{TE},
    basis::B,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {TD<:AbstractVector,TE<:AbstractVector,B<:AbstractRadialBasis,T<:Int}
    return jacobian(
        data, eval_points, basis, is_boundary, boundary_conditions, normals; k=k, adjl=adjl
    )
end

"""
    gradient(data, x, basis; k, adjl)

One-shot convenience function that creates a gradient operator and applies it to scalar field `x`.

For repeated evaluations on the same points, prefer creating the operator once with
[`gradient(data, basis)`](@ref) and calling it via functor syntax `op(x)`.

# Examples
```julia
points = [SVector{2}(rand(2)) for _ in 1:1000]
u = sin.(getindex.(points, 1))
∇u = gradient(points, u)  # One-shot gradient computation
```
"""
function gradient(
    data::AbstractVector,
    x,
    basis::B=PHS(3; poly_deg=2);
    k::Int=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {B<:AbstractRadialBasis}
    return jacobian(data, x, basis; k=k, adjl=adjl)
end

# pretty printing - reuse Jacobian's print_op since gradient returns a Jacobian operator
# The operator itself is Jacobian, so print_op(::Jacobian) handles it
