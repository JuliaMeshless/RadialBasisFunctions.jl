"""
    Custom{F<:Function} <: AbstractOperator

Custom operator that applies a user-defined function to basis functions.
The function `ℒ` should accept a basis and return a callable `(x, xᵢ) -> value`.
"""
struct Custom{F<:Function} <: AbstractOperator
    ℒ::F
end
(op::Custom)(basis) = op.ℒ(basis)

# Primary interface using unified keyword constructor
"""
    custom(data, ℒ; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` with a custom operator function.

# Arguments
- `data`: Vector of data points
- `ℒ`: Custom function that accepts a basis and returns a callable `(x, xᵢ) -> value`

# Keyword Arguments
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation

# Examples
```julia
# Custom operator that returns the basis function itself
op = custom(data, basis -> (x, xᵢ) -> basis(x, xᵢ))

# Custom biharmonic operator (∇⁴)
op = custom(data, basis -> ∇²(basis) ∘ ∇²(basis))
```
"""
custom(data::AbstractVector, ℒ::Function; kw...) =
    RadialBasisOperator(Custom(ℒ), data; kw...)

# Backward compatible positional signatures
custom(data::AbstractVector, ℒ::Function, basis::AbstractRadialBasis; kw...) =
    RadialBasisOperator(Custom(ℒ), data; basis=basis, kw...)

custom(data::AbstractVector, eval_points::AbstractVector, ℒ::Function,
       basis::AbstractRadialBasis=PHS(3; poly_deg=2); kw...) =
    RadialBasisOperator(Custom(ℒ), data; eval_points=eval_points, basis=basis, kw...)

# Hermite backward compatibility (positional boundary arguments)
function custom(
    data::AbstractVector,
    eval_points::AbstractVector,
    ℒ::Function,
    basis::AbstractRadialBasis,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    kw...,
)
    hermite = (is_boundary=is_boundary, bc=boundary_conditions, normals=normals)
    return RadialBasisOperator(Custom(ℒ), data;
        eval_points=eval_points, basis=basis, hermite=hermite, kw...)
end

# pretty printing
print_op(::Custom) = "Custom Operator"
