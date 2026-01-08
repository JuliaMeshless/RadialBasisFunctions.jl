"""
    Partial{T<:Int} <: ScalarValuedOperator

Operator for a partial derivative of specified order with respect to a dimension.
"""
struct Partial{T <: Int} <: ScalarValuedOperator
    order::T
    dim::T
end
(op::Partial)(basis) = ∂(basis, op.order, op.dim)

# Primary interface using unified keyword constructor
"""
    partial(data, order, dim; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for a partial derivative.

# Arguments
- `data`: Vector of data points
- `order`: Derivative order (1 or 2)
- `dim`: Dimension index to differentiate

# Keyword Arguments
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation

# Examples
```julia
# First derivative in x-direction
∂x = partial(data, 1, 1)

# Second derivative in y-direction
∂²y = partial(data, 2, 2; basis=PHS(5; poly_deg=4))
```
"""
function partial(data::AbstractVector, order::Int, dim::Int; kw...)
    return RadialBasisOperator(Partial(order, dim), data; kw...)
end

# Backward compatible positional signatures
function partial(
        data::AbstractVector, order::Int, dim::Int, basis::AbstractRadialBasis; kw...
    )
    return RadialBasisOperator(Partial(order, dim), data; basis = basis, kw...)
end

function partial(
        data::AbstractVector,
        eval_points::AbstractVector,
        order::Int,
        dim::Int,
        basis::AbstractRadialBasis = PHS(3; poly_deg = 2);
        kw...,
    )
    return RadialBasisOperator(
        Partial(order, dim), data; eval_points = eval_points, basis = basis, kw...
    )
end

# Hermite backward compatibility (positional boundary arguments)
function partial(
        data::AbstractVector,
        eval_points::AbstractVector,
        order::Int,
        dim::Int,
        basis::AbstractRadialBasis,
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{<:BoundaryCondition},
        normals::Vector{<:AbstractVector};
        kw...,
    )
    hermite = (is_boundary = is_boundary, bc = boundary_conditions, normals = normals)
    return RadialBasisOperator(
        Partial(order, dim),
        data;
        eval_points = eval_points,
        basis = basis,
        hermite = hermite,
        kw...,
    )
end

# Helper: dispatch to ∂ or ∂² based on order
function ∂(basis::AbstractBasis, order::T, dim::T) where {T <: Int}
    if order == 1
        return ∂(basis, dim)
    elseif order == 2
        return ∂²(basis, dim)
    else
        throw(
            ArgumentError(
                "Only first and second order derivatives are supported. Use the custom operator for higher orders.",
            ),
        )
    end
end

# pretty printing
print_op(op::Partial) = "∂ⁿf/∂xᵢ (n = $(op.order), i = $(op.dim))"
