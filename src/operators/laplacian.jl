"""
    Laplacian <: ScalarValuedOperator

Operator for the sum of the second derivatives w.r.t. each independent variable (∇²f).
"""
struct Laplacian <: ScalarValuedOperator end
(::Laplacian)(basis) = ∇²(basis)

# Primary interface using unified keyword constructor
"""
    laplacian(data; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for the Laplacian operator (∇²f).

# Arguments
- `data`: Vector of data points

# Keyword Arguments
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation

# Examples
```julia
# Basic usage
op = laplacian(data)

# With custom basis
op = laplacian(data; basis=PHS(5; poly_deg=3))

# With different evaluation points
op = laplacian(data; eval_points=eval_pts)
```
"""
laplacian(data::AbstractVector; kw...) = RadialBasisOperator(Laplacian(), data; kw...)

# Backward compatible positional signatures
function laplacian(data::AbstractVector, basis::AbstractRadialBasis; kw...)
    return RadialBasisOperator(Laplacian(), data; basis = basis, kw...)
end

function laplacian(
        data::AbstractVector,
        eval_points::AbstractVector,
        basis::AbstractRadialBasis = PHS(3; poly_deg = 2);
        kw...,
    )
    return RadialBasisOperator(Laplacian(), data; eval_points = eval_points, basis = basis, kw...)
end

# Hermite backward compatibility (positional boundary arguments)
"""
    laplacian(data, eval_points, basis, is_boundary, boundary_conditions, normals; k, adjl)

Build a Hermite-compatible `RadialBasisOperator` for the Laplacian.
Maintains backward compatibility with the positional argument API.
"""
function laplacian(
        data::AbstractVector,
        eval_points::AbstractVector,
        basis::AbstractRadialBasis,
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{<:BoundaryCondition},
        normals::Vector{<:AbstractVector};
        kw...,
    )
    hermite = (is_boundary = is_boundary, bc = boundary_conditions, normals = normals)
    return RadialBasisOperator(
        Laplacian(), data; eval_points = eval_points, basis = basis, hermite = hermite, kw...
    )
end

# pretty printing
print_op(::Laplacian) = "Laplacian (∇²f)"
