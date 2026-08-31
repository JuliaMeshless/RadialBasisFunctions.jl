"""
    normal_derivative(data, normals; basis=PHS(3; poly_deg=2), eval_points=data, k, adjl)

Build a `RadialBasisOperator` for the normal derivative (∇f⋅n̂).

The input `normals` are automatically normalized to unit vectors. This is a
convenience wrapper around [`directional`](@ref).

# Arguments
- `data`: Vector of data points
- `normals`: Normal vectors at each point (will be normalized)

# Keyword Arguments
$(KWARG_DOCS)

# Examples
```julia
points = rand(SVector{2,Float64}, 1000)
normals = normalize.(points)  # radial normals
∂ₙ = normal_derivative(points, normals)
result = ∂ₙ(sin.(getindex.(points, 1)))
```

See also: [`directional`](@ref), [`gradient`](@ref)
"""
function normal_derivative(data::AbstractVector, normals::AbstractVector; kw...)
    n̂ = normalize.(normals)
    return directional(data, n̂; kw...)
end

"""
    normal_derivative(data, normals, x; basis=PHS(3; poly_deg=2), k, adjl)

One-shot convenience function that creates a normal derivative operator and applies it to field `x`.

For repeated evaluations on the same points, prefer creating the operator once with
[`normal_derivative(data, normals)`](@ref) and calling it via functor syntax `op(x)`.
"""
function normal_derivative(data::AbstractVector, normals::AbstractVector, x; kw...)
    op = normal_derivative(data, normals; kw...)
    return op(x)
end
