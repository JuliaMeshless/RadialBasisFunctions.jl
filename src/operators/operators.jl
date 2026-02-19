abstract type AbstractOperator end
abstract type ScalarValuedOperator <: AbstractOperator end
abstract type VectorValuedOperator{Dim} <: AbstractOperator end

"""
    struct RadialBasisOperator

Operator of data using a radial basis with potential monomial augmentation.
"""
struct RadialBasisOperator{L, W, D, C, A, B <: AbstractRadialBasis, Dev}
    ℒ::L
    weights::W
    data::D
    eval_points::C
    adjl::A
    basis::B
    device::Dev
    valid_cache::Base.RefValue{Bool}
    function RadialBasisOperator(
            ℒ::L,
            weights::W,
            data::D,
            eval_points::C,
            adjl::A,
            basis::B,
            cache_status::Bool = false;
            device::Dev = CPU(),
        ) where {L, W, D, C, A, B <: AbstractRadialBasis, Dev}
        return new{L, W, D, C, A, B, Dev}(
            ℒ, weights, data, eval_points, adjl, basis, device, Ref(cache_status)
        )
    end
end

# ============================================================================
# Unified RadialBasisOperator Constructor
# ============================================================================

"""
    RadialBasisOperator(ℒ, data; eval_points, basis, k, adjl, hermite, device)

Unified constructor with keyword arguments.

# Arguments
- `ℒ`: The operator type (e.g., `Laplacian()`, `Partial(1, 2)`)
- `data`: Vector of data points

# Keyword Arguments
- `eval_points`: Evaluation points (default: `data`)
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)
- `hermite`: Optional NamedTuple for Hermite interpolation with fields:
  - `is_boundary::Vector{Bool}`
  - `bc::Vector{<:BoundaryCondition}`
  - `normals::Vector{<:AbstractVector}`
- `device`: KernelAbstractions backend for weight computation (default: auto-detected from `data` via `get_backend`)

# Examples
```julia
# Basic usage
op = RadialBasisOperator(Laplacian(), data)

# With custom basis and stencil size
op = RadialBasisOperator(Laplacian(), data; basis=PHS(5; poly_deg=3), k=40)

# With different evaluation points
op = RadialBasisOperator(Laplacian(), data; eval_points=eval_pts)

# With Hermite boundary conditions
op = RadialBasisOperator(Laplacian(), data;
    hermite=(is_boundary=is_bound, bc=boundary_conds, normals=normal_vecs))

# With explicit device
using KernelAbstractions
op = RadialBasisOperator(Laplacian(), data; device=CPU())
```
"""
function RadialBasisOperator(
        ℒ,
        data::AbstractVector;
        eval_points::AbstractVector = data,
        basis::AbstractRadialBasis = PHS(3; poly_deg = 2),
        k::Int = autoselect_k(data, basis),
        adjl = find_neighbors(data, eval_points, k),
        hermite::Union{Nothing, NamedTuple} = nothing,
        device = get_backend(data),
    )
    weights = if isnothing(hermite)
        _build_weights(ℒ, data, eval_points, adjl, basis; device = device)
    else
        _build_weights(
            ℒ,
            data,
            eval_points,
            adjl,
            basis,
            hermite.is_boundary,
            hermite.bc,
            hermite.normals;
            device = device,
        )
    end
    return RadialBasisOperator(ℒ, weights, data, eval_points, adjl, basis, true; device = device)
end

# ============================================================================
# Backward compatible positional constructors (delegate to unified constructor)
# ============================================================================

# Data + basis (positional)
function RadialBasisOperator(
        ℒ,
        data::AbstractVector,
        basis::AbstractRadialBasis;
        k::Int = autoselect_k(data, basis),
        adjl = find_neighbors(data, k),
        device = get_backend(data),
    )
    return RadialBasisOperator(ℒ, data; basis = basis, k = k, adjl = adjl, device = device)
end

# Data + eval_points + basis (positional)
function RadialBasisOperator(
        ℒ,
        data::AbstractVector,
        eval_points::AbstractVector,
        basis::AbstractRadialBasis;
        k::Int = autoselect_k(data, basis),
        adjl = find_neighbors(data, eval_points, k),
        device = get_backend(data),
    )
    return RadialBasisOperator(
        ℒ, data; eval_points = eval_points, basis = basis, k = k, adjl = adjl, device = device
    )
end

# Hermite-compatible constructor (positional boundary arguments)
function RadialBasisOperator(
        ℒ,
        data::AbstractVector,
        eval_points::AbstractVector,
        basis::AbstractRadialBasis,
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{<:BoundaryCondition},
        normals::Vector{<:AbstractVector};
        k::Int = autoselect_k(data, basis),
        adjl = find_neighbors(data, eval_points, k),
        device = get_backend(data),
    )
    hermite = (is_boundary = is_boundary, bc = boundary_conditions, normals = normals)
    return RadialBasisOperator(
        ℒ, data; eval_points = eval_points, basis = basis, k = k, adjl = adjl, hermite = hermite, device = device
    )
end

dim(op::RadialBasisOperator) = length(first(op.data))

# caching
invalidate_cache!(op::RadialBasisOperator) = op.valid_cache[] = false
validate_cache!(op::RadialBasisOperator) = op.valid_cache[] = true
is_cache_valid(op::RadialBasisOperator) = op.valid_cache[]

"""
    function (op::RadialBasisOperator)(x)

Evaluate the operator at `x`.
"""
function (op::RadialBasisOperator)(x)
    !is_cache_valid(op) && update_weights!(op)
    return _eval_op(op, x)
end

"""
    function (op::RadialBasisOperator)(y, x)

Evaluate the operator at `x` in-place and store the result in `y`.
"""
function (op::RadialBasisOperator)(y, x)
    !is_cache_valid(op) && update_weights!(op)
    return _eval_op(op, y, x)
end

# dispatches for evaluation
_eval_op(op::RadialBasisOperator, x) = op.weights * x
_eval_op(op::RadialBasisOperator, y, x) = mul!(y, op.weights, x)

# VectorValuedOperator: Scalar field input → Matrix output (N×D)
function _eval_op(
        op::RadialBasisOperator{<:VectorValuedOperator{D}}, x::AbstractVector
    ) where {D}
    N_eval = length(op.eval_points)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, N_eval, D)
    for d in 1:D
        mul!(view(out, :, d), op.weights[d], x)
    end
    return out
end

# VectorValuedOperator: Vector field input (N×D_in) → 3-tensor output (N×D_in×D)
function _eval_op(
        op::RadialBasisOperator{<:VectorValuedOperator{D}}, x::AbstractMatrix
    ) where {D}
    N_eval = length(op.eval_points)
    D_in = size(x, 2)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, N_eval, D_in, D)
    for d_out in 1:D, d_in in 1:D_in
        mul!(view(out, :, d_in, d_out), op.weights[d_out], view(x, :, d_in))
    end
    return out
end

# VectorValuedOperator: General tensor input → tensor output with extra dimension
function _eval_op(
        op::RadialBasisOperator{<:VectorValuedOperator{D}}, x::AbstractArray
    ) where {D}
    N_eval = length(op.eval_points)
    trailing_dims = size(x)[2:end]
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, N_eval, trailing_dims..., D)
    for idx in CartesianIndices(trailing_dims), d in 1:D
        mul!(view(out, :, idx, d), op.weights[d], view(x, :, idx))
    end
    return out
end

# VectorValuedOperator with SparseVector weights (single eval point)
# W is 2nd type param in RadialBasisOperator{L,W,D,C,A,B}
function _eval_op(
        op::RadialBasisOperator{<:VectorValuedOperator{D}, <:NTuple{D, <:SparseVector}},
        x::AbstractVector,
    ) where {D}
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, D)
    @inbounds for d in 1:D
        out[d] = dot(op.weights[d], x)
    end
    return out
end

function _eval_op(
        op::RadialBasisOperator{<:VectorValuedOperator{D}, <:NTuple{D, <:SparseVector}},
        x::AbstractMatrix,
    ) where {D}
    D_in = size(x, 2)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, D_in, D)
    @inbounds for d in 1:D, d_in in 1:D_in
        out[d_in, d] = dot(op.weights[d], view(x, :, d_in))
    end
    return out
end

function _eval_op(
        op::RadialBasisOperator{<:VectorValuedOperator{D}, <:NTuple{D, <:SparseVector}},
        x::AbstractArray,
    ) where {D}
    trailing_dims = size(x)[2:end]
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = similar(x, T, trailing_dims..., D)
    @inbounds for idx in CartesianIndices(trailing_dims), d in 1:D
        out[idx, d] = dot(op.weights[d], view(x, :, idx))
    end
    return out
end

# In-place: Scalar field → Matrix
function _eval_op(
        op::RadialBasisOperator{<:VectorValuedOperator{D}}, y::AbstractMatrix, x::AbstractVector
    ) where {D}
    for d in 1:D
        mul!(view(y, :, d), op.weights[d], x)
    end
    return y
end

# In-place: Vector field → 3-tensor
function _eval_op(
        op::RadialBasisOperator{<:VectorValuedOperator{D}},
        y::AbstractArray{<:Any, 3},
        x::AbstractMatrix,
    ) where {D}
    D_in = size(x, 2)
    for d_out in 1:D, d_in in 1:D_in
        mul!(view(y, :, d_in, d_out), op.weights[d_out], view(x, :, d_in))
    end
    return y
end

# LinearAlgebra methods - divergence (dot with gradient operator)
function LinearAlgebra.:⋅(
        op::RadialBasisOperator{<:VectorValuedOperator}, x::AbstractVector
    )
    !is_cache_valid(op) && update_weights!(op)
    result = op(x)  # Now Matrix (N×D)
    return vec(sum(result; dims = 2))  # Sum across derivative dimension → Vector (N,)
end

# update weights
function update_weights!(op::RadialBasisOperator)
    op.weights .= _build_weights(op.ℒ, op)
    validate_cache!(op)
    return nothing
end

function update_weights!(op::RadialBasisOperator{<:VectorValuedOperator{Dim}}) where {Dim}
    new_weights = _build_weights(op.ℒ, op)
    for i in 1:Dim
        op.weights[i] .= new_weights[i]
    end
    validate_cache!(op)
    return nothing
end

# ============================================================================
# Adapt.jl support (GPU array conversion)
# ============================================================================

function Adapt.adapt_structure(to, op::RadialBasisOperator)
    adapted_weights = Adapt.adapt(to, op.weights)
    return RadialBasisOperator(
        op.ℒ, adapted_weights, op.data, op.eval_points, op.adjl, op.basis,
        is_cache_valid(op); device = op.device,
    )
end

function Adapt.adapt_structure(to, op::RadialBasisOperator{<:VectorValuedOperator})
    adapted_weights = map(w -> Adapt.adapt(to, w), op.weights)
    return RadialBasisOperator(
        op.ℒ, adapted_weights, op.data, op.eval_points, op.adjl, op.basis,
        is_cache_valid(op); device = op.device,
    )
end

# pretty printing
function Base.show(io::IO, op::RadialBasisOperator)
    println(io, "RadialBasisOperator")
    println(io, "├─Operator: " * print_op(op.ℒ))
    println(io, "├─Data type: ", typeof(first(op.data)))
    println(io, "├─Number of points: ", length(op.data))
    println(io, "├─Stencil size: ", length(first(op.adjl)))
    if !(op.device isa CPU)
        println(io, "├─Device: ", op.device)
    end
    return println(
        io,
        "└─Basis: ",
        print_basis(op.basis),
        " with degree $(op.basis.poly_deg) polynomial augmentation",
    )
end
