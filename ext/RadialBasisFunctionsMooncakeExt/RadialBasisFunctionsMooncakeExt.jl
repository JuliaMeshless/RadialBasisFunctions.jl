"""
    RadialBasisFunctionsMooncakeExt

Package extension providing native Mooncake.jl rrule!! implementations for
RadialBasisFunctions.jl. All rules are native Mooncake rrule!! — no CRC bridge.

Includes rrule!! for:
- Basis function evaluation (all AbstractRadialBasis types)
- Operator evaluation: `_eval_op(op, x)` for scalar and vector-valued
- Operator call syntax: `op(x)`
- Interpolator evaluation
- Weight construction: `_build_weights` for Partial and Laplacian operators
"""
module RadialBasisFunctionsMooncakeExt

using RadialBasisFunctions
using Mooncake
using Mooncake: CoDual, NoFData, NoRData, primal, zero_fcodual
using StaticArrays: SVector
using SparseArrays: SparseMatrixCSC

# Import types and functions we need
import RadialBasisFunctions: _eval_op, RadialBasisOperator, Interpolator
import RadialBasisFunctions: PHS, PHS1, PHS3, PHS5, PHS7, AbstractPHS, IMQ, Gaussian
import RadialBasisFunctions: AbstractRadialBasis, VectorValuedOperator
import RadialBasisFunctions: _build_weights, Partial, Laplacian, MonomialBasis
import RadialBasisFunctions: _interpolator_point_gradient!

# Import backward pass support from main package
import RadialBasisFunctions: _forward_with_cache
import RadialBasisFunctions: extract_stencil_cotangent_from_nzval, _get_grad_funcs
import RadialBasisFunctions: build_weights_pullback_loop!

# Import gradient function
const ∇ = RadialBasisFunctions.∇

# =============================================================================
# Custom increment_and_get_rdata! for SVector types
# =============================================================================
# Mooncake represents SVector{N,T} tangents as Tangent{@NamedTuple{data::NTuple{N,T}}}
# Our pullbacks return Vector{SVector{N,T}} tangents for point gradients
# This method bridges the gap by incrementing fdata in-place

function Mooncake.increment_and_get_rdata!(
        f::Vector{<:Mooncake.Tangent}, ::Mooncake.NoRData, t::Vector{SVector{N, T}}
    ) where {N, T}
    for i in eachindex(f, t)
        old_data = f[i].fields.data
        sv = t[i]
        new_data = ntuple(j -> old_data[j] + sv[j], Val(N))
        f[i] = typeof(f[i])((data = new_data,))
    end
    return Mooncake.NoRData()
end

# =============================================================================
# Basis Function Rules (native rrule!!)
# =============================================================================
# A single @is_primitive + rrule!! covers all AbstractRadialBasis types.
# Output is Float64 scalar → rdata is Float64, fdata is NoFData.

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{<:AbstractRadialBasis, Vector{Float64}, Vector{Float64}}

function Mooncake.rrule!!(
        basis_cd::CoDual{<:AbstractRadialBasis},
        x::CoDual{Vector{Float64}},
        xi::CoDual{Vector{Float64}},
    )
    basis_val = primal(basis_cd)
    x_val, xi_val = primal(x), primal(xi)
    y = basis_val(x_val, xi_val)

    function basis_pb!!(Δy)
        grad_fn = ∇(basis_val)
        ∇φ = grad_fn(x_val, xi_val)
        for d in eachindex(x.dx)
            x.dx[d] += Δy * ∇φ[d]
            xi.dx[d] -= Δy * ∇φ[d]
        end
        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(y), basis_pb!!
end

# =============================================================================
# Scalar Operator Rules: _eval_op(op, x)
# =============================================================================
# Output is Vector{Float64} → fdata is Vector{Float64}, rdata is NoRData.

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_eval_op), RadialBasisOperator, Vector{Float64}}

function Mooncake.rrule!!(
        ::CoDual{typeof(_eval_op)},
        op::CoDual{<:RadialBasisOperator},
        x::CoDual{Vector{Float64}},
    )
    operator = primal(op)
    y = _eval_op(operator, primal(x))
    y_codual = zero_fcodual(y)

    function eval_op_pb!!(::NoRData)
        x.dx .+= operator.weights' * y_codual.dx
        return NoRData(), NoRData(), NoRData()
    end

    return y_codual, eval_op_pb!!
end

# =============================================================================
# Vector-Valued Operator Rules: _eval_op(op, x)
# =============================================================================
# Output is Matrix{Float64} → fdata is Matrix{Float64}, rdata is NoRData.

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_eval_op), RadialBasisOperator{<:VectorValuedOperator}, Vector{Float64}}

function Mooncake.rrule!!(
        ::CoDual{typeof(_eval_op)},
        op::CoDual{<:RadialBasisOperator{<:VectorValuedOperator{D}}},
        x::CoDual{Vector{Float64}},
    ) where {D}
    operator = primal(op)
    y = _eval_op(operator, primal(x))
    y_codual = zero_fcodual(y)

    function eval_op_vec_pb!!(::NoRData)
        Δy = y_codual.dx
        for d in 1:D
            x.dx .+= operator.weights[d]' * view(Δy, :, d)
        end
        return NoRData(), NoRData(), NoRData()
    end

    return y_codual, eval_op_vec_pb!!
end

# =============================================================================
# Operator Call Syntax Rules: op(x)
# =============================================================================

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{RadialBasisOperator, Vector{Float64}}

function Mooncake.rrule!!(
        op::CoDual{<:RadialBasisOperator},
        x::CoDual{Vector{Float64}},
    )
    operator = primal(op)
    !RadialBasisFunctions.is_cache_valid(operator) && RadialBasisFunctions.update_weights!(operator)
    y = _eval_op(operator, primal(x))
    y_codual = zero_fcodual(y)

    function op_call_pb!!(::NoRData)
        x.dx .+= operator.weights' * y_codual.dx
        return NoRData(), NoRData()
    end

    return y_codual, op_call_pb!!
end

# Vector-valued operator call syntax
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{RadialBasisOperator{<:VectorValuedOperator}, Vector{Float64}}

function Mooncake.rrule!!(
        op::CoDual{<:RadialBasisOperator{<:VectorValuedOperator{D}}},
        x::CoDual{Vector{Float64}},
    ) where {D}
    operator = primal(op)
    !RadialBasisFunctions.is_cache_valid(operator) && RadialBasisFunctions.update_weights!(operator)
    y = _eval_op(operator, primal(x))
    y_codual = zero_fcodual(y)

    function op_call_vec_pb!!(::NoRData)
        Δy = y_codual.dx
        for d in 1:D
            x.dx .+= operator.weights[d]' * view(Δy, :, d)
        end
        return NoRData(), NoRData()
    end

    return y_codual, op_call_vec_pb!!
end

# =============================================================================
# Interpolator Rule
# =============================================================================
# Output is Float64 scalar → rdata is Float64.

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{<:Interpolator, Vector{Float64}}

function Mooncake.rrule!!(
        interp_cd::CoDual{<:Interpolator},
        x::CoDual{Vector{Float64}},
    )
    interp = primal(interp_cd)
    x_val = primal(x)
    y = interp(x_val)

    function interp_pb!!(Δy)
        _interpolator_point_gradient!(x.dx, interp, x_val, Δy)
        return NoRData(), NoRData()
    end

    return zero_fcodual(y), interp_pb!!
end

# =============================================================================
# Native rrule!! for _build_weights
# =============================================================================

# Declare _build_weights as primitive (consolidated using abstract types)
# PHS types
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, <:AbstractPHS}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), <:Partial, AbstractVector, AbstractVector, AbstractVector, <:AbstractPHS}

# IMQ and Gaussian
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, <:Union{IMQ, Gaussian}}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), <:Partial, AbstractVector, AbstractVector, AbstractVector, <:Union{IMQ, Gaussian}}

# =============================================================================
# Shared helpers for _build_weights rrule!! implementations
# =============================================================================

"""
    _mooncake_build_weights_forward(ℒ, pts, eval_pts, adj, bas, OpType)

Shared forward pass: build monomial basis, apply operator, compute weights with cache,
and create zero CoDual for the output.
"""
function _mooncake_build_weights_forward(ℒ, pts, eval_pts, adj, bas, ::Type{OpType}) where {OpType}
    dim_space = length(first(pts))
    mon = MonomialBasis(dim_space, bas.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(bas)
    W, cache = _forward_with_cache(pts, eval_pts, adj, bas, ℒrbf, ℒmon, mon, OpType)
    W_codual = zero_fcodual(W)
    return W, W_codual, cache, mon, dim_space
end

"""
    _accumulate_into_mooncake_fdata!(fdata, Δraw, dim_space)

Accumulate raw gradient vectors into Mooncake fdata (mutable tangent storage).
Mooncake represents SVector{N,T} tangents as Tangent{@NamedTuple{data::NTuple{N,T}}},
so we reconstruct the tangent with incremented tuple values.
"""
function _accumulate_into_mooncake_fdata!(
        fdata::Vector{<:Mooncake.Tangent}, Δraw::Vector{Vector{T}}, dim_space::Int
    ) where {T}
    for i in eachindex(fdata)
        old_data = fdata[i].fields.data
        Δ = Δraw[i]
        new_data = ntuple(d -> old_data[d] + Δ[d], Val(dim_space))
        fdata[i] = typeof(fdata[i])((data = new_data,))
    end
end

"""
    _mooncake_build_weights_pullback(...)

Create shared pullback closure that runs `build_weights_pullback_loop!` and accumulates
gradients into Mooncake fdata. Returns ε gradient value for basis rdata construction.
"""
function _mooncake_build_weights_pullback(
        W_fdata, W, data, eval_points, cache, adj, pts, eval_pts, bas, mon, ℒ,
        ::Type{OpType}, dim_space,
    ) where {OpType}
    grad_Lφ_x, grad_Lφ_xi = _get_grad_funcs(OpType, bas, ℒ)

    function _shared_pb!!(ΔW_rdata)
        ΔW_nzval = W_fdata.data.nzval
        TD = eltype(first(pts))
        N_data = length(pts)
        N_eval = length(eval_pts)

        Δdata_raw = [zeros(TD, dim_space) for _ in 1:N_data]
        Δeval_raw = [zeros(TD, dim_space) for _ in 1:N_eval]
        Δε_acc = Ref(zero(TD))

        build_weights_pullback_loop!(
            Δdata_raw, Δeval_raw, Δε_acc,
            (eval_idx, neighbors, k) -> extract_stencil_cotangent_from_nzval(ΔW_nzval, W, eval_idx, neighbors, k),
            cache, adj, eval_pts, pts, bas, mon, ℒ, OpType,
            grad_Lφ_x, grad_Lφ_xi
        )

        _accumulate_into_mooncake_fdata!(data.dx, Δdata_raw, dim_space)
        _accumulate_into_mooncake_fdata!(eval_points.dx, Δeval_raw, dim_space)

        return Δε_acc[]
    end

    return _shared_pb!!
end

# =============================================================================
# rrule!! wrappers for _build_weights (2 per basis group × 2 op types = 4 total)
# =============================================================================

# PHS bases (Laplacian + Partial)
function Mooncake.rrule!!(
        ::CoDual{typeof(_build_weights)},
        op::CoDual{<:Union{Laplacian, Partial}},
        data::CoDual{<:AbstractVector},
        eval_points::CoDual{<:AbstractVector},
        adjl::CoDual{<:AbstractVector},
        basis::CoDual{<:AbstractPHS},
    )
    ℒ, pts, eval_pts, adj, bas = primal(op), primal(data), primal(eval_points), primal(adjl), primal(basis)
    OpType = typeof(ℒ) <: Partial ? Partial : Laplacian
    W, W_codual, cache, mon, dim_space = _mooncake_build_weights_forward(ℒ, pts, eval_pts, adj, bas, OpType)
    shared_pb!! = _mooncake_build_weights_pullback(W_codual.dx, W, data, eval_points, cache, adj, pts, eval_pts, bas, mon, ℒ, OpType, dim_space)

    function pb!!(ΔW_rdata)
        shared_pb!!(ΔW_rdata)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return W_codual, pb!!
end

# IMQ/Gaussian bases (Laplacian + Partial) — need ε gradient in rdata
function Mooncake.rrule!!(
        ::CoDual{typeof(_build_weights)},
        op::CoDual{<:Union{Laplacian, Partial}},
        data::CoDual{<:AbstractVector},
        eval_points::CoDual{<:AbstractVector},
        adjl::CoDual{<:AbstractVector},
        basis::CoDual{<:Union{IMQ, Gaussian}},
    )
    ℒ, pts, eval_pts, adj, bas = primal(op), primal(data), primal(eval_points), primal(adjl), primal(basis)
    OpType = typeof(ℒ) <: Partial ? Partial : Laplacian
    W, W_codual, cache, mon, dim_space = _mooncake_build_weights_forward(ℒ, pts, eval_pts, adj, bas, OpType)
    shared_pb!! = _mooncake_build_weights_pullback(W_codual.dx, W, data, eval_points, cache, adj, pts, eval_pts, bas, mon, ℒ, OpType, dim_space)
    TD = eltype(first(pts))

    function pb!!(ΔW_rdata)
        Δε = shared_pb!!(ΔW_rdata)
        basis_rdata = Mooncake.RData{@NamedTuple{ε::TD, poly_deg::Mooncake.NoRData}}(
            (ε = Δε, poly_deg = NoRData())
        )
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), basis_rdata
    end

    return W_codual, pb!!
end

end # module
