"""
    @operator expr

Create an operator from mathematical notation. Returns an `AbstractOperator`
suitable for passing to [`custom`](@ref) or [`RadialBasisOperator`](@ref).

# Recognized symbols
- `∇²` / `Δ` — [`Laplacian`](@ref)
- `∂(dim)` — first partial derivative in dimension `dim`
- `∂²(dim)` — second partial derivative in dimension `dim`
- `∂(dim1, dim2)` — mixed partial derivative ∂²f/(∂xᵢ ∂xⱼ)
- `∇ ⋅ (κ * ∇)` — anisotropic diffusion (scalar or vector `κ`)
- `c ⋅ ∇` — advection operator (vector `c`)
- `f` / `I` — [`Identity`](@ref) operator
- Everything else — scalar coefficient

# Examples
```julia
helm = custom(x, @operator(∇² + k^2 * f))
aniso = custom(x, @operator(κx * ∂²(1) + κy * ∂²(2)))
advdiff = custom(x, @operator(ν * ∇² - c ⋅ ∇))
diff = custom(x, @operator(∇ ⋅ (κ * ∇)))
```
"""
macro operator(expr)
    return esc(_transform_operator_expr(expr))
end

_transform_operator_expr(s::Symbol) = _resolve_operator_symbol(s)
function _transform_operator_expr(expr::Expr)
    expr.head == :call && return _transform_operator_call(expr)
    return expr
end
_transform_operator_expr(expr) = expr  # scalar literal or other expression

function _resolve_operator_symbol(s::Symbol)
    s === :∇² && return :(Laplacian())
    s === :Δ  && return :(Laplacian())
    s === :f  && return :(Identity())
    s === :I  && return :(Identity())
    return s  # scalar variable
end

function _fold_left(op_sym::Symbol, args)
    result = _transform_operator_expr(args[1])
    for i in 2:length(args)
        result = Expr(:call, op_sym, result, _transform_operator_expr(args[i]))
    end
    return result
end

function _transform_operator_call(expr)
    op = expr.args[1]
    args = expr.args[2:end]

    if op === :+ && length(args) >= 2
        return _fold_left(:+, args)
    elseif op === :- && length(args) == 2
        return Expr(:call, :-, _transform_operator_expr(args[1]), _transform_operator_expr(args[2]))
    elseif op === :- && length(args) == 1
        return Expr(:call, :-, _transform_operator_expr(args[1]))
    elseif op === :* && length(args) >= 2
        return _fold_left(:*, args)
    elseif op === :/ && length(args) == 2
        return Expr(:call, :/, _transform_operator_expr(args[1]), _transform_operator_expr(args[2]))
    elseif op === :∂
        if length(args) == 1
            return :(Partial(1, $(args[1])))
        elseif length(args) == 2
            return :(MixedPartial($(args[1]), $(args[2])))
        else
            throw(ArgumentError("∂ expects 1 or 2 arguments, got $(length(args))"))
        end
    elseif op === :∂²
        length(args) == 1 || throw(ArgumentError("∂²(dim) expects exactly 1 argument, got $(length(args))"))
        return :(Partial(2, $(args[1])))
    elseif op === :⋅ && length(args) == 2
        return _transform_dot(args[1], args[2])
    end

    return expr  # fallback: scalar expression (k^2, sin(x), etc.)
end

function _transform_dot(lhs, rhs)
    if lhs === :∇ && rhs === :∇
        return :(Laplacian())
    elseif lhs === :∇ && _is_scaled_nabla(rhs)
        coeff = _extract_nabla_coefficient(rhs)
        return Expr(:call, GlobalRef(@__MODULE__, :_expand_div_grad), coeff)
    elseif rhs === :∇
        return Expr(
            :call, GlobalRef(@__MODULE__, :_expand_dot_grad),
            _transform_operator_expr(lhs)
        )
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
    _expand_dot_grad(c::AbstractVector)

Expand `c ⋅ ∇` into `∑ c[i] * ∂/∂xᵢ` (advection operator).
"""
_expand_dot_grad(c::AbstractVector) = sum(c[i] * Partial(1, i) for i in 1:length(c))

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
