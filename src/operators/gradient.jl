"""
    Gradient{Dim} <: VectorValuedOperator

Builds an operator for the gradient of a function.
"""
struct Gradient{Dim} <: VectorValuedOperator{Dim} end

function (op::Gradient{Dim})(basis) where {Dim}
    return ntuple(dim -> ∂(basis, dim), Dim)
end

# convienience constructors
"""
    function gradient(data, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the gradient, `Gradient`.
"""
function gradient(
    data::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Gradient{Dim}()
    return RadialBasisOperator(ℒ, data, basis; k=k, adjl=adjl)
end

"""
    function gradient(data, eval_points, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the gradient, `Gradient`. The resulting operator will only evaluate at `eval_points`.
"""
function gradient(
    data::AbstractVector,
    eval_points::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Gradient{Dim}()
    return RadialBasisOperator(ℒ, data, eval_points, basis; k=k, adjl=adjl)
end

"""
    function gradient(data, eval_points, basis, is_boundary, boundary_conditions, normals; k=autoselect_k(data, basis))

Builds a Hermite-compatible `RadialBasisOperator` where the operator is the gradient, `Gradient`.
The additional boundary information enables Hermite interpolation with proper boundary condition handling.
"""
function gradient(
    data::AbstractVector,
    eval_points::AbstractVector,
    basis::B,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Gradient{Dim}()
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

# Hermite-compatible method now uses the generic dispatcher in solve_hermite.jl

# pretty printing
print_op(op::Gradient) = "Gradient (∇f)"

# Evaluation methods - delegate to jacobian for unified implementation

"""
    gradient(op::RadialBasisOperator{<:Gradient}, x::AbstractVector)

Evaluate the gradient of scalar field `x`. Returns `Matrix{T}` of size (N_eval × D).

This is equivalent to `jacobian(op, x)` but validates that the input is a scalar field.

!!! note "Breaking change"
    In previous versions, `gradient` returned a tuple of vectors `(∂f/∂x₁, ∂f/∂x₂, ...)`.
    It now returns a Matrix for better autodiff compatibility and consistency with Jacobian semantics.
    Migrate code from `result[1]` to `result[:, 1]`.

# Examples
```julia
op = gradient(points, PHS(3; poly_deg=2))
u = sin.(getindex.(points, 1))
∇u = gradient(op, u)  # Matrix (N × D)
∂u_∂x = ∇u[:, 1]
∂u_∂y = ∇u[:, 2]
```
"""
function gradient(op::RadialBasisOperator{<:Gradient}, x::AbstractVector)
    return jacobian(op, x)
end

"""
    gradient!(out::AbstractMatrix, op::RadialBasisOperator{<:Gradient}, x::AbstractVector)

In-place gradient evaluation. Stores result in preallocated `out` matrix.

# Arguments
- `out`: Preallocated output matrix of size (N_eval × D)
- `op`: Gradient operator
- `x`: Scalar field values

# Examples
```julia
op = gradient(points, PHS(3; poly_deg=2))
out = Matrix{Float64}(undef, length(points), 2)
gradient!(out, op, u)
```
"""
function gradient!(
    out::AbstractMatrix, op::RadialBasisOperator{<:Gradient}, x::AbstractVector
)
    return jacobian!(out, op, x)
end
