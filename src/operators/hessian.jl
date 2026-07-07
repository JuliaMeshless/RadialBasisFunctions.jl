"""
    Hessian{Dim} <: AbstractOperator{2}

Operator for computing the full Hessian matrix ∂²f/(∂xᵢ ∂xⱼ) at each point.

The Hessian is a rank-2 operator: it adds two trailing dimensions of size D
(the spatial dimension) to the output. Only `D*(D+1)/2` unique weight matrices
are stored (upper-triangular entries); symmetric entries are filled via `copyto!`.

# Input/Output Shapes
- Scalar field `Vector{T}` (N,) → Hessian `Array{T,3}` (N_eval × D × D)
- Vector field `Matrix{T}` (N × D_in) → `Array{T,4}` (N_eval × D_in × D × D)
"""
struct Hessian{Dim} <: AbstractOperator{2} end

function (op::Hessian{Dim})(basis::AbstractRadialBasis) where {Dim}
    n_unique = _num_sym(Dim)
    return ntuple(n_unique) do k
        i, j = _kth_sym_pair(k, Dim)
        ∂mixed(basis, i, j)
    end
end

function (op::Hessian{Dim})(basis::MonomialBasis) where {Dim}
    return H(basis)
end

# Primary interface
"""
    hessian(data; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for computing the full Hessian matrix.

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
op = hessian(points)

# Scalar field → Hessian tensor
u = sin.(getindex.(points, 1)) .* cos.(getindex.(points, 2))
H = op(u)  # Array (1000 × 2 × 2)
# H[:, 1, 1] = ∂²u/∂x², H[:, 1, 2] = ∂²u/∂x∂y, etc.
```

See also: [`jacobian`](@ref), [`laplacian`](@ref), [`mixed_partial`](@ref)
"""
function hessian(data::AbstractVector{<:AbstractVector}; kw...)
    Dim = length(first(data))
    return RadialBasisOperator(Hessian{Dim}(), data; kw...)
end

# Backward compatible positional signatures
function hessian(data::AbstractVector{<:AbstractVector}, basis::AbstractRadialBasis; kw...)
    Dim = length(first(data))
    return RadialBasisOperator(Hessian{Dim}(), data; basis = basis, kw...)
end

# One-shot convenience
"""
    hessian(data, x; basis=PHS(3; poly_deg=2), k, adjl)

One-shot convenience function that creates a Hessian operator and applies it to field `x`.
"""
function hessian(data::AbstractVector{<:AbstractVector}, x; kw...)
    op = hessian(data; kw...)
    return op(x)
end

# pretty printing
print_op(::Hessian) = "Hessian (H)"
