"""
    @operator expr

Create an operator from mathematical notation. Returns an `AbstractOperator`
suitable for passing to [`custom`](@ref) or [`RadialBasisOperator`](@ref).

# Recognized symbols
- `∇²` / `Δ` — [`Laplacian`](@ref)
- `∂(dim)` — first partial derivative in dimension `dim`
- `∂²(dim)` — second partial derivative in dimension `dim`
- `∇ ⋅ (κ * ∇)` — anisotropic diffusion (scalar or vector `κ`)
- `f` / `I` — [`Identity`](@ref) operator
- Everything else — scalar coefficient

# Examples
```julia
helm = custom(x, @operator(∇² + k^2 * f); rank=0)
aniso = custom(x, @operator(κx * ∂²(1) + κy * ∂²(2)); rank=0)
advdiff = custom(x, @operator(ν * ∇² - c[1] * ∂(1) - c[2] * ∂(2)); rank=0)
diff = custom(x, @operator(∇ ⋅ (κ * ∇)); rank=0)
```
"""
macro operator(expr)
    return esc(_transform_operator_expr(expr))
end

function _transform_operator_expr(expr)
    if expr isa Symbol
        return _resolve_operator_symbol(expr)
    elseif expr isa Expr && expr.head == :call
        return _transform_operator_call(expr)
    end
    return expr  # scalar literal or other expression (e.g., array indexing)
end

function _resolve_operator_symbol(s::Symbol)
    s === :∇² && return :(Laplacian())
    s === :Δ  && return :(Laplacian())
    s === :f  && return :(Identity())
    s === :I  && return :(Identity())
    return s  # scalar variable
end

function _transform_operator_call(expr)
    op = expr.args[1]
    args = expr.args[2:end]

    if op === :+ && length(args) >= 2
        result = _transform_operator_expr(args[1])
        for i in 2:length(args)
            result = Expr(:call, :+, result, _transform_operator_expr(args[i]))
        end
        return result
    elseif op === :- && length(args) == 2
        a = _transform_operator_expr(args[1])
        b = _transform_operator_expr(args[2])
        return Expr(:call, :-, a, b)
    elseif op === :- && length(args) == 1
        a = _transform_operator_expr(args[1])
        return Expr(:call, :-, a)
    elseif op === :* && length(args) >= 2
        result = _transform_operator_expr(args[1])
        for i in 2:length(args)
            result = Expr(:call, :*, result, _transform_operator_expr(args[i]))
        end
        return result
    elseif op === :∂ && length(args) == 1
        return :(Partial(1, $(args[1])))
    elseif op === :∂² && length(args) == 1
        return :(Partial(2, $(args[1])))
    elseif op === :⋅ && length(args) == 2
        return _transform_dot(args[1], args[2])
    end

    return expr  # fallback: scalar expression (k^2, sin(x), etc.)
end

function _transform_dot(lhs, rhs)
    if lhs === :∇ && _is_scaled_nabla(rhs)
        coeff = _extract_nabla_coefficient(rhs)
        return Expr(:call, GlobalRef(@__MODULE__, :_expand_div_grad), coeff)
    end
    return Expr(:call, :⋅, _transform_operator_expr(lhs), _transform_operator_expr(rhs))
end

function _is_scaled_nabla(expr)
    return expr isa Expr && expr.head == :call && expr.args[1] === :* &&
           any(a -> a === :∇, expr.args[2:end])
end

function _extract_nabla_coefficient(expr)
    coeff_args = filter(a -> a !== :∇, expr.args[2:end])
    return length(coeff_args) == 1 ? coeff_args[1] : Expr(:call, :*, coeff_args...)
end

# ============================================================================
# Expansion functions for textbook notation
# ============================================================================

"""
    _expand_div_grad(κ::Number)

Expand `∇⋅(κ∇)` with scalar diffusivity into `κ * Laplacian()`.
"""
_expand_div_grad(κ::Number) = κ * Laplacian()

"""
    _expand_div_grad(κ::AbstractVector)

Expand `∇⋅(κ∇)` with per-dimension diffusivity into `∑ κ[i] * ∂²(i)`.
"""
function _expand_div_grad(κ::AbstractVector)
    return sum(κ[i] * Partial(2, i) for i in 1:length(κ))
end

# ============================================================================
# Convenience constructor
# ============================================================================

"""
    diffusion(data, κ; kw...)

Build a `RadialBasisOperator` for the diffusion operator `∇⋅(κ∇f)`.

Scalar `κ` produces `κ * Laplacian()`. Vector `κ` (length must match data
dimension) produces `∑ κ[i] * ∂²f/∂xᵢ²`.

# Examples
```julia
op = diffusion(points, 0.5)            # isotropic: 0.5 * ∇²f
op = diffusion(points, [1.0, 2.0])     # anisotropic: ∂²f/∂x² + 2∂²f/∂y²
```
"""
function diffusion(data::AbstractVector, κ; kw...)
    if κ isa AbstractVector && length(κ) != length(first(data))
        throw(DimensionMismatch(
            "diffusivity vector length ($(length(κ))) must match data dimension ($(length(first(data))))"
        ))
    end
    op = _expand_div_grad(κ)
    return RadialBasisOperator(op, data; kw...)
end
