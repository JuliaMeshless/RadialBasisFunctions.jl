"""
    Custom{N, F<:Function} <: AbstractOperator{N}

Custom operator that applies a user-defined function to basis functions.
The function `ℒ` should accept a basis and return a callable `(x, xᵢ) -> value`.

`N` is the tensor rank added to the output: `Custom{0}(ℒ)` for rank-preserving
or `Custom{1}(ℒ)` for rank+1.
"""
struct Custom{N, F <: Function} <: AbstractOperator{N}
    ℒ::F
end
Custom{N}(ℒ::F) where {N, F <: Function} = Custom{N, F}(ℒ)
(op::Custom)(basis) = op.ℒ(basis)

"""
    _infer_rank(ℒ)

Infer the tensor rank of a custom operator function by probing it with a default basis.
Returns `1` if `ℒ(basis)` produces a `Tuple` (one callable per dimension), `0` otherwise.
"""
function _infer_rank(ℒ)
    probe = ℒ(PHS(3; poly_deg=2))
    return probe isa Tuple ? 1 : 0
end

# Primary interface using unified keyword constructor
"""
    custom(data, ℒ::Function; rank=<auto>, basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)
    custom(data, op::AbstractOperator; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` with a custom operator.

# Arguments
- `data`: Vector of data points
- `ℒ`: Custom function that accepts a basis and returns a callable `(x, xᵢ) -> value`
- `op`: An `AbstractOperator` (e.g. from [`@operator`](@ref) or operator algebra)

# Keyword Arguments
- `rank::Int`: Tensor rank added to the output (0 = rank-preserving, 1 = rank+1).
  Auto-inferred when omitted: from the type parameter for `AbstractOperator`, or by
  probing the closure for `Function` (tuple return → rank 1, scalar → rank 0).
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation

# Examples
```julia
# Custom operator that returns the basis function itself (rank-preserving)
op = custom(data, basis -> (x, xᵢ) -> basis(x, xᵢ))

# Custom second partial derivative ∂²f/∂x₁² using the ∂² functor
op = custom(data, basis -> ∂²(basis, 1))

# Using @operator macro (rank inferred from type parameter)
op = custom(data, @operator ∇² + k² * f)
```
"""
function custom(data::AbstractVector, ℒ::Function; rank::Int=_infer_rank(ℒ), kw...)
    return RadialBasisOperator(Custom{rank}(ℒ), data; kw...)
end

# Accept AbstractOperator directly (from operator algebra or @operator macro)
function custom(data::AbstractVector, op::AbstractOperator; kw...)
    return RadialBasisOperator(op, data; kw...)
end

# Backward compatible positional signatures
function custom(data::AbstractVector, ℒ::Function, basis::AbstractRadialBasis; rank::Int=_infer_rank(ℒ), kw...)
    return RadialBasisOperator(Custom{rank}(ℒ), data; basis = basis, kw...)
end

function custom(
        data::AbstractVector,
        eval_points::AbstractVector,
        ℒ::Function,
        basis::AbstractRadialBasis = PHS(3; poly_deg = 2);
        rank::Int=_infer_rank(ℒ),
        kw...,
    )
    return RadialBasisOperator(Custom{rank}(ℒ), data; eval_points = eval_points, basis = basis, kw...)
end

# Hermite backward compatibility (positional boundary arguments)
function custom(
        data::AbstractVector,
        eval_points::AbstractVector,
        ℒ::Function,
        basis::AbstractRadialBasis,
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{<:BoundaryCondition},
        normals::Vector{<:AbstractVector};
        rank::Int=_infer_rank(ℒ),
        kw...,
    )
    hermite = (is_boundary = is_boundary, bc = boundary_conditions, normals = normals)
    return RadialBasisOperator(
        Custom{rank}(ℒ), data; eval_points = eval_points, basis = basis, hermite = hermite, kw...
    )
end

# pretty printing
print_op(::Custom) = "Custom Operator"
