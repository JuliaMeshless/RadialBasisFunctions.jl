for N in (:1, :2, :3)
    for Dim in (:1, :2, :3)
        @eval begin
            function ∂(::MonomialBasis{$Dim, 0}, ::Val{$N})
                function basis!(b, x)
                    b[1] = zero(_get_underlying_type(x))
                    return nothing
                end
                return ℒMonomialBasis($Dim, 0, basis!)
            end
        end
    end
end

function ∂(::MonomialBasis{1, Deg}, ::Val{N}) where {Deg, N}
    function basis!(b, x)
        x_type = _get_underlying_type(x)
        b[1] = zero(x_type)
        b[2] = one(x_type)
        if Deg > 1
            for d in 1:(Deg - 1)
                b[d + 2] = (d + 1) * only(x)^d
            end
        end
        return nothing
    end
    return ℒMonomialBasis(1, Deg, basis!)
end

function ∂(::MonomialBasis{2, 1}, ::Val{1})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = one(T)
        b[3] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(2, 1, basis!)
end

function ∂(::MonomialBasis{2, 1}, ::Val{2})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = one(T)
        return nothing
    end
    return ℒMonomialBasis(2, 1, basis!)
end

function ∂(::MonomialBasis{2, 2}, ::Val{1})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = one(T)
        b[3] = zero(T)
        b[4] = x[2]
        b[5] = 2 * x[1]
        b[6] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(2, 2, basis!)
end

function ∂(::MonomialBasis{2, 2}, ::Val{2})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = one(T)
        b[4] = x[1]
        b[5] = zero(T)
        b[6] = 2 * x[2]
        return nothing
    end
    return ℒMonomialBasis(2, 2, basis!)
end

function ∂(::MonomialBasis{3, 1}, ::Val{1})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = one(T)
        b[3] = zero(T)
        b[4] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(3, 1, basis!)
end

function ∂(::MonomialBasis{3, 1}, ::Val{2})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = one(T)
        b[4] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(3, 1, basis!)
end

function ∂(::MonomialBasis{3, 1}, ::Val{3})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = one(T)
        return nothing
    end
    return ℒMonomialBasis(3, 1, basis!)
end

function ∂(::MonomialBasis{3, 2}, ::Val{1})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = one(T)
        b[3] = zero(T)
        b[4] = zero(T)
        b[5] = x[2]
        b[6] = x[3]
        b[7] = zero(T)
        b[8] = 2 * x[1]
        b[9] = zero(T)
        b[10] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(3, 2, basis!)
end

function ∂(::MonomialBasis{3, 2}, ::Val{2})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = one(T)
        b[4] = zero(T)
        b[5] = x[1]
        b[6] = zero(T)
        b[7] = x[3]
        b[8] = zero(T)
        b[9] = 2 * x[2]
        b[10] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(3, 2, basis!)
end

function ∂(::MonomialBasis{3, 2}, ::Val{3})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = one(T)
        b[5] = zero(T)
        b[6] = x[1]
        b[7] = x[2]
        b[8] = zero(T)
        b[9] = zero(T)
        b[10] = 2 * x[3]
        return nothing
    end
    return ℒMonomialBasis(3, 2, basis!)
end

## ∂²

∂²(m::MonomialBasis{Dim, 0}, n::Val{N}) where {Dim, N} = ∂(m, n)
∂²(m::MonomialBasis{1, 0}, n::Val{N}) where {N} = ∂(m, n)

_∂²(x, n) = (n - 1) * n * x^(n - 2)
_∂²(x::Union{AbstractArray, Tuple}, n) = (n - 1) * n * x[1]^(n - 2)
function ∂²(::MonomialBasis{1, Deg}, ::Val{N}) where {Deg, N}
    function basis!(b, x)
        x_type = _get_underlying_type(x)
        b[1] = zero(x_type)
        b[2] = zero(x_type)
        b[3] = 2 * one(x_type)
        if Deg > 2
            for d in 1:(Deg - 2)
                b[d + 3] = (d + 1) * (d + 2) * only(x)^d
            end
        end
        return nothing
    end
    return ℒMonomialBasis(1, Deg, basis!)
end
function ∂²(::MonomialBasis{1, 1}, ::Val{N}) where {N}
    function basis!(b, x)
        x_type = _get_underlying_type(x)
        b[1] = zero(x_type)
        b[2] = zero(x_type)
        return nothing
    end
    return ℒMonomialBasis(1, 1, basis!)
end

function ∂²(::MonomialBasis{2, 1}, ::Val{1})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(2, 1, basis!)
end

function ∂²(::MonomialBasis{2, 1}, ::Val{2})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(2, 1, basis!)
end

function ∂²(::MonomialBasis{2, 2}, ::Val{1})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = zero(T)
        b[5] = 2 * one(T)
        b[6] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(2, 2, basis!)
end

function ∂²(::MonomialBasis{2, 2}, ::Val{2})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = zero(T)
        b[5] = zero(T)
        b[6] = 2 * one(T)
        return nothing
    end
    return ℒMonomialBasis(2, 2, basis!)
end

function ∂²(::MonomialBasis{3, 1}, ::Val{1})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(3, 1, basis!)
end

function ∂²(::MonomialBasis{3, 1}, ::Val{2})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(3, 1, basis!)
end

function ∂²(::MonomialBasis{3, 1}, ::Val{3})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(3, 1, basis!)
end

function ∂²(::MonomialBasis{3, 2}, ::Val{1})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = zero(T)
        b[5] = zero(T)
        b[6] = zero(T)
        b[7] = zero(T)
        b[8] = 2 * one(T)
        b[9] = zero(T)
        b[10] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(3, 2, basis!)
end

function ∂²(::MonomialBasis{3, 2}, ::Val{2})
    function basis!(b, x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = zero(T)
        b[5] = zero(T)
        b[6] = zero(T)
        b[7] = zero(T)
        b[8] = zero(T)
        b[9] = 2 * one(T)
        b[10] = zero(T)
        return nothing
    end
    return ℒMonomialBasis(3, 2, basis!)
end

function ∂²(::MonomialBasis{3, 2}, ::Val{3})
    function basis!(b, x)
        T = eltype(x)
        T = eltype(x)
        b[1] = zero(T)
        b[2] = zero(T)
        b[3] = zero(T)
        b[4] = zero(T)
        b[5] = zero(T)
        b[6] = zero(T)
        b[7] = zero(T)
        b[8] = zero(T)
        b[9] = zero(T)
        b[10] = 2 * one(T)
        return nothing
    end
    return ℒMonomialBasis(3, 2, basis!)
end

# Add normal derivative functionality
function ∂_normal(mb::MonomialBasis{Dim, Deg}, normal::AbstractVector) where {Dim, Deg}
    function basis!(b, x)
        T = eltype(x)

        for i in eachindex(b)
            b[i] = zero(T)
        end

        tmp = zeros(T, length(b)) #allocating (TODO: optimize)
        for d in 1:Dim
            if !iszero(normal[d])
                # Get the partial derivative operator
                op = ∂(mb, d)

                # Apply partial derivative
                fill!(tmp, zero(T))
                op.f(tmp, x)

                # Add scaled contribution to result
                for i in eachindex(b)
                    b[i] += normal[d] * tmp[i]
                end
            end
        end

        return nothing
    end

    return ℒMonomialBasis(Dim, Deg, basis!)
end

# Hermite derivative - works for both regular and Neumann boundaries
function ∂_Hermite(mb::MonomialBasis{Dim, Deg}, dim::Int) where {Dim, Deg}
    function hermite_derivative(x, normal = nothing)
        result = ones(eltype(x), binomial(Dim + Deg, Dim))

        if normal === nothing || all(iszero, normal)
            # Regular case - just use the partial derivative
            ∂(mb, dim).f(result, x)
        else
            # Neumann boundary case - use normal derivative
            ∂_normal(mb, normal).f(result, x)
        end

        return result
    end

    return hermite_derivative
end
