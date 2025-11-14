# Polyharmonic Spline

"""
   abstract type AbstractPHS{M<:Metric} <: AbstractRadialBasis{M}

Supertype of all Polyharmonic Splines with distance metric type parameter `M`.
"""
abstract type AbstractPHS{M<:Metric} <: AbstractRadialBasis{M} end

"""
    function PHS(n::T=3; poly_deg::T=2, metric::M=Euclidean()) where {T<:Int, M<:Metric}

Convenience constructor for polyharmonic splines.

# Arguments
- `n::Int`: Order of the polyharmonic spline (must be 1, 3, 5, or 7)
- `poly_deg::Int`: Degree of polynomial augmentation
- `metric::Metric`: Distance metric to use (defaults to Euclidean)

# Returns
Polyharmonic spline basis function of order `n` with the specified metric.
"""
function PHS(n::T=3; poly_deg::T=2, metric::M=Euclidean()) where {T<:Int, M<:Metric}
    check_poly_deg(poly_deg)
    if iseven(n) || n > 7
        throw(ArgumentError("n must be 1, 3, 5, or 7. (n = $n)"))
    end
    n == 1 && return PHS1(poly_deg; metric=metric)
    n == 3 && return PHS3(poly_deg; metric=metric)
    n == 5 && return PHS5(poly_deg; metric=metric)
    return PHS7(poly_deg; metric=metric)
end

"""
    struct PHS1{T<:Int, M<:Metric} <: AbstractPHS{M}

Polyharmonic spline radial basis function: `ϕ(r) = r`

# Fields
- `poly_deg::Int`: Degree of polynomial augmentation
- `metric::Metric`: Distance metric (default: Euclidean)
"""
struct PHS1{T<:Int, M<:Metric} <: AbstractPHS{M}
    poly_deg::T
    metric::M
    function PHS1(poly_deg::T; metric::M=Euclidean()) where {T<:Int, M<:Metric}
        check_poly_deg(poly_deg)
        return new{T, M}(poly_deg, metric)
    end
end

# Keyword-only constructor for backward compatibility
PHS1(; poly_deg::T, metric::M=Euclidean()) where {T<:Int, M<:Metric} = PHS1(poly_deg; metric=metric)

(phs::PHS1)(x, xᵢ) = phs.metric(x, xᵢ)

# Analytical derivative for Euclidean metric (fast path)
function ∂(phs::PHS1{T, Euclidean}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return (x[dim] - xᵢ[dim]) / (r + AVOID_INF)
        else
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -normal_arg[dim] / (r + AVOID_INF) +
                   dot_normal * (x[dim] - xᵢ[dim]) / (r^3 + AVOID_INF)
        end
    end
    return ∂ℒ
end

# AD-based derivative for non-Euclidean metrics
function ∂(phs::PHS1{T, M}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        if normal_arg === nothing
            # Use ForwardDiff for partial derivative
            return ForwardDiff.derivative(t -> phs(x + t * _unit_vector(x, dim), xᵢ), 0.0)
        else
            # Hermite case - will be validated elsewhere, but provide AD fallback
            ϕ = t -> phs(x + t * _unit_vector(x, dim), xᵢ)
            return ForwardDiff.derivative(ϕ, 0.0)
        end
    end
    return ∂ℒ
end
# Analytical gradient for Euclidean metric (fast path)
function ∇(phs::PHS1{T, Euclidean}, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∇ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return (x .- xᵢ) / r
        else
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -normal_arg / (r + AVOID_INF) +
                   dot_normal * (x .- xᵢ) / (r^3 + AVOID_INF)
        end
    end
    return ∇ℒ
end

# AD-based gradient for non-Euclidean metrics
function ∇(phs::PHS1{T, M}, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∇ℒ(x, xᵢ, normal_arg=normal)
        # Use ForwardDiff for gradient
        return ForwardDiff.gradient(ξ -> phs(ξ, xᵢ), x)
    end
    return ∇ℒ
end
# Analytical directional second derivative for Euclidean metric
function directional∂²(phs::PHS1{T, Euclidean}, v1::AbstractVector, v2::AbstractVector) where T
    function directional₂ℒ(x, xᵢ)
        r = phs.metric(x, xᵢ)
        dot_v1_v2 = LinearAlgebra.dot(v1, v2)
        dot_v1_r = LinearAlgebra.dot(v1, x .- xᵢ)
        dot_v2_r = LinearAlgebra.dot(v2, x .- xᵢ)
        return -dot_v1_v2 / r + (dot_v1_r * dot_v2_r) / r^3
    end
    return directional₂ℒ
end

# AD-based directional second derivative for non-Euclidean metrics
function directional∂²(phs::PHS1{T, M}, v1::AbstractVector, v2::AbstractVector) where {T, M<:Metric}
    function directional₂ℒ(x, xᵢ)
        # ∂²ϕ/∂v₁∂v₂ = v₂ᵀ·H·v₁ where H is the Hessian
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return LinearAlgebra.dot(v2, hess * v1)
    end
    return directional₂ℒ
end
# Analytical second partial derivative for Euclidean metric
function ∂²(phs::PHS1{T, Euclidean}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∂²ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        r² = evaluate(SqEuclidean(), x, xᵢ)
        if normal_arg === nothing
            return (-(x[dim] - xᵢ[dim])^2 + r²) / (r^3 + AVOID_INF)
        else
            n_d = normal_arg[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return (2 * n_d * Δ_d + dot_normal) / (r^3 + AVOID_INF) -
                   3 * Δ_d^2 * dot_normal / (r^5 + AVOID_INF)
        end
    end
    return ∂²ℒ
end

# AD-based second partial derivative for non-Euclidean metrics
function ∂²(phs::PHS1{T, M}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∂²ℒ(x, xᵢ, normal_arg=normal)
        # Compute ∂²ϕ/∂x_dim² using Hessian
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return hess[dim, dim]
    end
    return ∂²ℒ
end
# Analytical Laplacian for Euclidean metric
function ∇²(phs::PHS1{T, Euclidean}, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∇²ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 2 / (r + AVOID_INF)
        else
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return 2 * dot_normal / (r^3 + AVOID_INF)
        end
    end
    return ∇²ℒ
end

# AD-based Laplacian for non-Euclidean metrics
function ∇²(phs::PHS1{T, M}, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∇²ℒ(x, xᵢ, normal_arg=normal)
        # Compute Laplacian as trace of Hessian
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return LinearAlgebra.tr(hess)
    end
    return ∇²ℒ
end

"""
    struct PHS3{T<:Int, M<:Metric} <: AbstractPHS{M}

Polyharmonic spline radial basis function: `ϕ(r) = r³`

# Fields
- `poly_deg::Int`: Degree of polynomial augmentation
- `metric::Metric`: Distance metric (default: Euclidean)
"""
struct PHS3{T<:Int, M<:Metric} <: AbstractPHS{M}
    poly_deg::T
    metric::M
    function PHS3(poly_deg::T; metric::M=Euclidean()) where {T<:Int, M<:Metric}
        check_poly_deg(poly_deg)
        return new{T, M}(poly_deg, metric)
    end
end

# Keyword-only constructor for backward compatibility
PHS3(; poly_deg::T, metric::M=Euclidean()) where {T<:Int, M<:Metric} = PHS3(poly_deg; metric=metric)

(phs::PHS3)(x, xᵢ) = phs.metric(x, xᵢ)^3

# Analytical derivative for Euclidean metric (fast path)
function ∂(phs::PHS3{T, Euclidean}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 3 * (x[dim] - xᵢ[dim]) * r
        else
            n_d = normal_arg[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -3 * (n_d * r + dot_normal * Δ_d / (r + AVOID_INF))
        end
    end
    return ∂ℒ
end

# AD-based derivative for non-Euclidean metrics
function ∂(phs::PHS3{T, M}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        return ForwardDiff.derivative(t -> phs(x + t * _unit_vector(x, dim), xᵢ), 0.0)
    end
    return ∂ℒ
end
# Analytical gradient for Euclidean metric
function ∇(phs::PHS3{T, Euclidean}, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∇ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 3 * (x .- xᵢ) * r
        else
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -3 * (normal_arg * r + dot_normal * (x .- xᵢ) / (r + AVOID_INF))
        end
    end
    return ∇ℒ
end

# AD-based gradient for non-Euclidean metrics
function ∇(phs::PHS3{T, M}, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∇ℒ(x, xᵢ, normal_arg=normal)
        return ForwardDiff.gradient(ξ -> phs(ξ, xᵢ), x)
    end
    return ∇ℒ
end

# Analytical directional second derivative for Euclidean metric
function directional∂²(phs::PHS3{T, Euclidean}, v1::AbstractVector, v2::AbstractVector) where T
    function directional₂ℒ(x, xᵢ)
        r = phs.metric(x, xᵢ)
        dot_v1_v2 = LinearAlgebra.dot(v1, v2)
        dot_v1_r = LinearAlgebra.dot(v1, x .- xᵢ)
        dot_v2_r = LinearAlgebra.dot(v2, x .- xᵢ)
        return -3 * (dot_v1_v2 * r + dot_v1_r * dot_v2_r / (r + AVOID_INF))
    end
    return directional₂ℒ
end

# AD-based directional second derivative for non-Euclidean metrics
function directional∂²(phs::PHS3{T, M}, v1::AbstractVector, v2::AbstractVector) where {T, M<:Metric}
    function directional₂ℒ(x, xᵢ)
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return LinearAlgebra.dot(v2, hess * v1)
    end
    return directional₂ℒ
end

# Analytical second partial derivative for Euclidean metric
function ∂²(phs::PHS3{T, Euclidean}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∂²ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 3 * (r + (x[dim] - xᵢ[dim])^2 / (r + AVOID_INF))
        else
            r² = evaluate(SqEuclidean(), x, xᵢ)
            n_d = normal_arg[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -3 *
                   (2 * Δ_d * n_d + dot_normal - dot_normal * Δ_d^2 / (r² + AVOID_INF)) /
                   (r + AVOID_INF)
        end
    end
    return ∂²ℒ
end

# AD-based second partial derivative for non-Euclidean metrics
function ∂²(phs::PHS3{T, M}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∂²ℒ(x, xᵢ, normal_arg=normal)
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return hess[dim, dim]
    end
    return ∂²ℒ
end

# Analytical Laplacian for Euclidean metric
function ∇²(phs::PHS3{T, Euclidean}, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∇²ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 12 * r
        else
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -12 * dot_normal / (r + AVOID_INF)
        end
    end
    return ∇²ℒ
end

# AD-based Laplacian for non-Euclidean metrics
function ∇²(phs::PHS3{T, M}, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∇²ℒ(x, xᵢ, normal_arg=normal)
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return LinearAlgebra.tr(hess)
    end
    return ∇²ℒ
end

"""
    struct PHS5{T<:Int, M<:Metric} <: AbstractPHS{M}

Polyharmonic spline radial basis function: `ϕ(r) = r⁵`

# Fields
- `poly_deg::Int`: Degree of polynomial augmentation
- `metric::Metric`: Distance metric (default: Euclidean)
"""
struct PHS5{T<:Int, M<:Metric} <: AbstractPHS{M}
    poly_deg::T
    metric::M
    function PHS5(poly_deg::T; metric::M=Euclidean()) where {T<:Int, M<:Metric}
        check_poly_deg(poly_deg)
        return new{T, M}(poly_deg, metric)
    end
end

# Keyword-only constructor for backward compatibility
PHS5(; poly_deg::T, metric::M=Euclidean()) where {T<:Int, M<:Metric} = PHS5(poly_deg; metric=metric)

(phs::PHS5)(x, xᵢ) = phs.metric(x, xᵢ)^5

# Analytical derivative for Euclidean metric (fast path)
function ∂(phs::PHS5{T, Euclidean}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 5 * (x[dim] - xᵢ[dim]) * r^3
        else
            n_d = normal_arg[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -5 * (n_d * r^3 + 3 * dot_normal * Δ_d * r)
        end
    end
    return ∂ℒ
end

# AD-based derivative for non-Euclidean metrics
function ∂(phs::PHS5{T, M}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        return ForwardDiff.derivative(t -> phs(x + t * _unit_vector(x, dim), xᵢ), 0.0)
    end
    return ∂ℒ
end

# Analytical gradient for Euclidean metric
function ∇(phs::PHS5{T, Euclidean}, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∇ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 5 * (x .- xᵢ) * r^3
        else
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -5 * (normal_arg * r^3 + 3 * dot_normal * (x .- xᵢ) * r)
        end
    end
    return ∇ℒ
end

# AD-based gradient for non-Euclidean metrics
function ∇(phs::PHS5{T, M}, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∇ℒ(x, xᵢ, normal_arg=normal)
        return ForwardDiff.gradient(ξ -> phs(ξ, xᵢ), x)
    end
    return ∇ℒ
end

# Analytical directional second derivative for Euclidean metric
function directional∂²(phs::PHS5{T, Euclidean}, v1::AbstractVector, v2::AbstractVector) where T
    function directional₂ℒ(x, xᵢ)
        r = phs.metric(x, xᵢ)
        dot_v1_v2 = LinearAlgebra.dot(v1, v2)
        dot_v1_r = LinearAlgebra.dot(v1, x .- xᵢ)
        dot_v2_r = LinearAlgebra.dot(v2, x .- xᵢ)
        return -5 * (dot_v1_v2 * r^3 + 3 * dot_v1_r * dot_v2_r * r)
    end
    return directional₂ℒ
end

# AD-based directional second derivative for non-Euclidean metrics
function directional∂²(phs::PHS5{T, M}, v1::AbstractVector, v2::AbstractVector) where {T, M<:Metric}
    function directional₂ℒ(x, xᵢ)
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return LinearAlgebra.dot(v2, hess * v1)
    end
    return directional₂ℒ
end

# Analytical second partial derivative for Euclidean metric
function ∂²(phs::PHS5{T, Euclidean}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∂²ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            r² = evaluate(SqEuclidean(), x, xᵢ)
            return 5 * r * (3 * (x[dim] - xᵢ[dim])^2 + r²)
        else
            n_d = normal_arg[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -5 * (6 * n_d * Δ_d * r + 3 * dot_normal * (r + Δ_d^2 / (r + AVOID_INF)))
        end
    end
    return ∂²ℒ
end

# AD-based second partial derivative for non-Euclidean metrics
function ∂²(phs::PHS5{T, M}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∂²ℒ(x, xᵢ, normal_arg=normal)
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return hess[dim, dim]
    end
    return ∂²ℒ
end

# Analytical Laplacian for Euclidean metric
function ∇²(phs::PHS5{T, Euclidean}, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∇²ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 30 * r^3
        else
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -5 * (18) * dot_normal * r
        end
    end
    return ∇²ℒ
end

# AD-based Laplacian for non-Euclidean metrics
function ∇²(phs::PHS5{T, M}, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∇²ℒ(x, xᵢ, normal_arg=normal)
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return LinearAlgebra.tr(hess)
    end
    return ∇²ℒ
end

"""
    struct PHS7{T<:Int, M<:Metric} <: AbstractPHS{M}

Polyharmonic spline radial basis function: `ϕ(r) = r⁷`

# Fields
- `poly_deg::Int`: Degree of polynomial augmentation
- `metric::Metric`: Distance metric (default: Euclidean)
"""
struct PHS7{T<:Int, M<:Metric} <: AbstractPHS{M}
    poly_deg::T
    metric::M
    function PHS7(poly_deg::T; metric::M=Euclidean()) where {T<:Int, M<:Metric}
        check_poly_deg(poly_deg)
        return new{T, M}(poly_deg, metric)
    end
end

# Keyword-only constructor for backward compatibility
PHS7(; poly_deg::T, metric::M=Euclidean()) where {T<:Int, M<:Metric} = PHS7(poly_deg; metric=metric)

(phs::PHS7)(x, xᵢ) = phs.metric(x, xᵢ)^7

# Analytical derivative for Euclidean metric (fast path)
function ∂(phs::PHS7{T, Euclidean}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 7 * (x[dim] - xᵢ[dim]) * r^5
        else
            n_d = normal_arg[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -7 * (n_d * r^5 + 5 * r^3 * Δ_d * dot_normal)
        end
    end
    return ∂ℒ
end

# AD-based derivative for non-Euclidean metrics
function ∂(phs::PHS7{T, M}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        return ForwardDiff.derivative(t -> phs(x + t * _unit_vector(x, dim), xᵢ), 0.0)
    end
    return ∂ℒ
end

# Analytical gradient for Euclidean metric
function ∇(phs::PHS7{T, Euclidean}, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∇ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 7 * (x .- xᵢ) * r^5
        else
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -7 * (normal_arg * r^5 + 5 * dot_normal * (x .- xᵢ) * r^3)
        end
    end
    return ∇ℒ
end

# AD-based gradient for non-Euclidean metrics
function ∇(phs::PHS7{T, M}, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∇ℒ(x, xᵢ, normal_arg=normal)
        return ForwardDiff.gradient(ξ -> phs(ξ, xᵢ), x)
    end
    return ∇ℒ
end

# Analytical directional second derivative for Euclidean metric
function directional∂²(phs::PHS7{T, Euclidean}, v1::AbstractVector, v2::AbstractVector) where T
    function directional₂ℒ(x, xᵢ)
        r = phs.metric(x, xᵢ)
        dot_v1_v2 = LinearAlgebra.dot(v1, v2)
        dot_v1_r = LinearAlgebra.dot(v1, x .- xᵢ)
        dot_v2_r = LinearAlgebra.dot(v2, x .- xᵢ)
        return -7 * (dot_v1_v2 * r^5 + 5 * dot_v1_r * dot_v2_r * r^3)
    end
    return directional₂ℒ
end

# AD-based directional second derivative for non-Euclidean metrics
function directional∂²(phs::PHS7{T, M}, v1::AbstractVector, v2::AbstractVector) where {T, M<:Metric}
    function directional₂ℒ(x, xᵢ)
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return LinearAlgebra.dot(v2, hess * v1)
    end
    return directional₂ℒ
end

# Analytical second partial derivative for Euclidean metric
function ∂²(phs::PHS7{T, Euclidean}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∂²ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            r² = evaluate(SqEuclidean(), x, xᵢ)
            return 7 * r^3 * (5 * (x[dim] - xᵢ[dim])^2 + r²)
        else
            n_d = normal_arg[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -7 * (10 * n_d * Δ_d * r^3 + 5 * dot_normal * (3 * r * Δ_d^2 + r^3))
        end
    end
    return ∂²ℒ
end

# AD-based second partial derivative for non-Euclidean metrics
function ∂²(phs::PHS7{T, M}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∂²ℒ(x, xᵢ, normal_arg=normal)
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return hess[dim, dim]
    end
    return ∂²ℒ
end

# Analytical Laplacian for Euclidean metric
function ∇²(phs::PHS7{T, Euclidean}, normal::Union{Nothing,AbstractVector}=nothing) where T
    function ∇²ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return 56 * r^5
        else
            dot_normal = LinearAlgebra.dot(normal_arg, x .- xᵢ)
            return -7 * (40 * dot_normal * r^3)
        end
    end
    return ∇²ℒ
end

# AD-based Laplacian for non-Euclidean metrics
function ∇²(phs::PHS7{T, M}, normal::Union{Nothing,AbstractVector}=nothing) where {T, M<:Metric}
    function ∇²ℒ(x, xᵢ, normal_arg=normal)
        hess = ForwardDiff.hessian(ξ -> phs(ξ, xᵢ), x)
        return LinearAlgebra.tr(hess)
    end
    return ∇²ℒ
end

function Base.show(io::IO, rbf::R) where {R<:AbstractPHS}
    print(io, print_basis(rbf))
    print(io, "\n└─Polynomial augmentation: degree $(rbf.poly_deg)")
    return nothing
end

print_basis(::PHS1) = "Polyharmonic spline (r¹)"
print_basis(::PHS3) = "Polyharmonic spline (r³)"
print_basis(::PHS5) = "Polyharmonic spline (r⁵)"
print_basis(::PHS7) = "Polyharmonic spline (r⁷)"
