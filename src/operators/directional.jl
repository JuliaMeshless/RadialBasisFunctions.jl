"""
    Directional{Dim,T} <: AbstractOperator{0}

Operator for the directional derivative (∇f⋅v), the inner product of the gradient and a direction vector.
"""
struct Directional{Dim, T} <: AbstractOperator{0}
    v::T
end
Directional{Dim}(v) where {Dim} = Directional{Dim, typeof(v)}(v)

# Primary interface using unified keyword constructor
"""
    directional(data, v; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl, hermite)

Build a `RadialBasisOperator` for the directional derivative (∇f⋅v).

# Arguments
- `data`: Vector of data points
- `v`: Direction vector. Can be:
  - A single vector of length `Dim` (constant direction)
  - A vector of vectors matching `length(data)` (spatially-varying direction)

# Keyword Arguments
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation

# Examples
```julia
# Constant direction
∂_x = directional(data, [1.0, 0.0])

# Spatially-varying direction (e.g., radial)
normals = [normalize(p) for p in data]
∂_n = directional(data, normals)
```

See also: [`gradient`](@ref), [`partial`](@ref), [`laplacian`](@ref)
"""
function directional(data::AbstractVector, v::AbstractVector; kw...)
    Dim = length(first(data))
    return RadialBasisOperator(Directional{Dim}(v), data; kw...)
end

# Backward compatible positional signatures
function directional(
        data::AbstractVector, v::AbstractVector, basis::AbstractRadialBasis; kw...
    )
    Dim = length(first(data))
    return RadialBasisOperator(Directional{Dim}(v), data; basis = basis, kw...)
end

# ============================================================================
# Custom weight-building logic for Directional operator
# The directional derivative is computed by combining Jacobian weights with
# the direction vector: ∇f⋅v = Σᵢ (∂f/∂xᵢ) * vᵢ
# ============================================================================

# Helper: validate direction vector dimensions
function _validate_directional_vector(v, Dim::Int, data_length::Int)
    return if !(length(v) == Dim || length(v) == data_length)
        throw(
            DomainError(
                "Direction vector length $(length(v)) must equal dimension $Dim or data length $data_length",
            ),
        )
    end
end

# Helper: combine gradient weights with direction vector
function _combine_directional_weights(weights, v, Dim::Int)
    if length(v) == Dim
        # Constant direction: simple weighted sum
        return mapreduce(+, zip(weights, v)) do (w, vᵢ)
            w * vᵢ
        end
    else
        # Spatially-varying direction: diagonal scaling
        vv = ntuple(i -> getindex.(v, i), Dim)
        return mapreduce(+, zip(weights, vv)) do (w, vᵢ)
            Diagonal(vᵢ) * w
        end
    end
end

# Custom _build_weights: standard (non-Hermite) path
function _build_weights(ℒ::Directional{Dim}, data, eval_points, adjl, basis; device = CPU()) where {Dim}
    v = ℒ.v
    _validate_directional_vector(v, Dim, length(data))
    weights = _build_weights(Jacobian{Dim}(), data, eval_points, adjl, basis; device = device)
    return _combine_directional_weights(weights, v, Dim)
end

# Custom _build_weights: Hermite path (with boundary conditions)
function _build_weights(
        ℒ::Directional{Dim},
        data::AbstractVector,
        eval_points::AbstractVector,
        adjl::AbstractVector,
        basis::AbstractRadialBasis,
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{<:BoundaryCondition},
        normals::Vector{<:AbstractVector};
        device = CPU(),
    ) where {Dim}
    v = ℒ.v
    _validate_directional_vector(v, Dim, length(data))

    # Build jacobian weights using Hermite method
    dim = length(first(data))
    mon = MonomialBasis(dim, basis.poly_deg)
    jacobian_op = Jacobian{Dim}()
    ℒmon = jacobian_op(mon)
    ℒrbf = jacobian_op(basis)

    weights = _build_weights(
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        is_boundary,
        boundary_conditions,
        normals;
        device = device,
    )

    return _combine_directional_weights(weights, v, Dim)
end

# pretty printing
print_op(::Directional) = "Directional Derivative (∇f⋅v)"
