"""
    Custom{N, F<:Function} <: AbstractOperator{N}

Custom operator that applies a user-defined function to basis functions.
The function `ℒ` should accept a basis and return a callable `(x, xᵢ) -> value`.

`N` is the tensor rank added to the output and must be specified explicitly,
e.g. `Custom{0}(ℒ)` for rank-preserving or `Custom{1}(ℒ)` for rank+1.
"""
struct Custom{N, F <: Function} <: AbstractOperator{N}
    ℒ::F
end
Custom{N}(ℒ::F) where {N, F <: Function} = Custom{N, F}(ℒ)
(op::Custom)(basis) = op.ℒ(basis)

# Primary interface using unified keyword constructor
"""
    custom(data, ℒ; rank, basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` with a custom operator function.

# Arguments
- `data`: Vector of data points
- `ℒ`: Custom function that accepts a basis and returns a callable `(x, xᵢ) -> value`

# Keyword Arguments
- `rank::Int` (required): Tensor rank added to the output (0 = rank-preserving, 1 = rank+1)
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation

# Examples
```julia
# Custom operator that returns the basis function itself (rank-preserving)
op = custom(data, basis -> (x, xᵢ) -> basis(x, xᵢ); rank=0)

# Custom second partial derivative ∂²f/∂x₁² using the ∂² functor
op = custom(data, basis -> ∂²(basis, 1); rank=0)
```
"""
function custom(data::AbstractVector, ℒ::Function; rank::Int, kw...)
    return RadialBasisOperator(Custom{rank}(ℒ), data; kw...)
end

# Backward compatible positional signatures
function custom(data::AbstractVector, ℒ::Function, basis::AbstractRadialBasis; rank::Int, kw...)
    return RadialBasisOperator(Custom{rank}(ℒ), data; basis = basis, kw...)
end

function custom(
        data::AbstractVector,
        eval_points::AbstractVector,
        ℒ::Function,
        basis::AbstractRadialBasis = PHS(3; poly_deg = 2);
        rank::Int,
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
        rank::Int,
        kw...,
    )
    hermite = (is_boundary = is_boundary, bc = boundary_conditions, normals = normals)
    return RadialBasisOperator(
        Custom{rank}(ℒ), data; eval_points = eval_points, basis = basis, hermite = hermite, kw...
    )
end

# pretty printing
print_op(::Custom) = "Custom Operator"
