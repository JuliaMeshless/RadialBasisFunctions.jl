"""
    MixedPartial{T<:Int} <: AbstractOperator{0}

Operator for the mixed second partial derivative ∂²f/(∂xᵢ ∂xⱼ).
When `dim1 == dim2`, delegates to `Partial(2, dim)`.
"""
struct MixedPartial{T <: Int} <: AbstractOperator{0}
    dim1::T
    dim2::T
end

function (op::MixedPartial)(basis::AbstractRadialBasis)
    op.dim1 == op.dim2 && return ∂²(basis, op.dim1)
    return ∂mixed(basis, op.dim1, op.dim2)
end

function (op::MixedPartial)(basis::MonomialBasis)
    return ∂mixed(basis, op.dim1, op.dim2)
end

# Primary interface
"""
    mixed_partial(data, dim1, dim2; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for the mixed partial derivative ∂²f/(∂xᵢ ∂xⱼ).

# Arguments
- `data`: Vector of data points
- `dim1`: First dimension index
- `dim2`: Second dimension index

# Keyword Arguments
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation

# Examples
```julia
∂²xy = mixed_partial(data, 1, 2)
```

See also: [`partial`](@ref), [`hessian`](@ref)
"""
function mixed_partial(data::AbstractVector, dim1::Int, dim2::Int; kw...)
    return RadialBasisOperator(MixedPartial(dim1, dim2), data; kw...)
end

# Backward compatible positional signatures
function mixed_partial(
        data::AbstractVector, dim1::Int, dim2::Int, basis::AbstractRadialBasis; kw...
    )
    return RadialBasisOperator(MixedPartial(dim1, dim2), data; basis = basis, kw...)
end

function mixed_partial(
        data::AbstractVector,
        eval_points::AbstractVector,
        dim1::Int,
        dim2::Int,
        basis::AbstractRadialBasis = PHS(3; poly_deg = 2);
        kw...,
    )
    return RadialBasisOperator(
        MixedPartial(dim1, dim2), data; eval_points = eval_points, basis = basis, kw...
    )
end

# Hermite backward compatibility (positional boundary arguments)
function mixed_partial(
        data::AbstractVector,
        eval_points::AbstractVector,
        dim1::Int,
        dim2::Int,
        basis::AbstractRadialBasis,
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{<:BoundaryCondition},
        normals::Vector{<:AbstractVector};
        kw...,
    )
    hermite = (is_boundary = is_boundary, bc = boundary_conditions, normals = normals)
    return RadialBasisOperator(
        MixedPartial(dim1, dim2),
        data;
        eval_points = eval_points,
        basis = basis,
        hermite = hermite,
        kw...,
    )
end

# One-shot convenience
"""
    mixed_partial(data, dim1, dim2, x; basis=PHS(3; poly_deg=2), k, adjl)

One-shot convenience function that creates a mixed partial operator and applies it to field `x`.
"""
function mixed_partial(data::AbstractVector, dim1::Int, dim2::Int, x::AbstractVector{<:Number}; kw...)
    op = mixed_partial(data, dim1, dim2; kw...)
    return op(x)
end

# pretty printing
print_op(op::MixedPartial) = "∂²f/∂x$(op.dim1)∂x$(op.dim2)"
