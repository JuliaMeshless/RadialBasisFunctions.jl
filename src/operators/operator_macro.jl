"""
    @operator expr

Create an operator from mathematical notation. Returns an `AbstractOperator`
suitable for passing to [`custom`](@ref) or [`RadialBasisOperator`](@ref).

# Recognized symbols
- `∇²` / `Δ` — [`Laplacian`](@ref)
- `∂(dim)` — first partial derivative in dimension `dim`
- `∂²(dim)` — second partial derivative in dimension `dim`
- `f` / `I` — [`Identity`](@ref) operator
- Everything else — scalar coefficient

# Examples
```julia
helm = custom(x, @operator(∇² + k^2 * f); rank=0)
aniso = custom(x, @operator(κx * ∂²(1) + κy * ∂²(2)); rank=0)
advdiff = custom(x, @operator(ν * ∇² - c[1] * ∂(1) - c[2] * ∂(2)); rank=0)
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
    end

    return expr  # fallback: scalar expression (k^2, sin(x), etc.)
end
