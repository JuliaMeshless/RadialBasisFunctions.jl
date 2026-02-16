"""
    RadialBasisFunctionsEnzymeExt

Package extension that provides native Enzyme.jl AD support for RadialBasisFunctions.jl
using EnzymeRules (augmented_primal + reverse).

This extension requires Enzyme to be loaded.

Native rules are provided for:
- Basis function evaluation: all AbstractRadialBasis subtypes
- Operator evaluation: `_eval_op(op, x)` for scalar and vector-valued operators
- Operator call syntax: `op(x)`
- Interpolator evaluation: single point and batch
- Weight construction: `_build_weights` for Partial and Laplacian operators
"""
module RadialBasisFunctionsEnzymeExt

using RadialBasisFunctions
using Enzyme
using EnzymeCore
using EnzymeCore.EnzymeRules
using LinearAlgebra
using SparseArrays

# Import internal functions
import RadialBasisFunctions: _eval_op, RadialBasisOperator, Interpolator
import RadialBasisFunctions: IMQ, Gaussian
import RadialBasisFunctions: AbstractRadialBasis, VectorValuedOperator
import RadialBasisFunctions: _build_weights, Partial, Laplacian, _optype
import RadialBasisFunctions: MonomialBasis

# Import backward pass support from main package
import RadialBasisFunctions: _forward_with_cache

# Import gradient function and shared helpers
const ∇ = RadialBasisFunctions.∇
import RadialBasisFunctions: _interpolator_point_gradient!

# =============================================================================
# Basis Function Rules
# =============================================================================

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{<:AbstractRadialBasis},
        ::Type{<:EnzymeCore.Active},
        x::EnzymeCore.Duplicated,
        xi::EnzymeCore.Duplicated,
    )
    basis = func.val
    y = basis(x.val, xi.val)
    tape = (copy(x.val), copy(xi.val))
    return EnzymeRules.AugmentedReturn(y, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{<:AbstractRadialBasis},
        dret::EnzymeCore.Active,
        tape,
        x::EnzymeCore.Duplicated,
        xi::EnzymeCore.Duplicated,
    )
    basis = func.val
    x_val, xi_val = tape
    grad_fn = ∇(basis)
    ∇φ = grad_fn(x_val, xi_val)
    x.dval .+= dret.val .* ∇φ
    xi.dval .-= dret.val .* ∇φ
    return (nothing, nothing)
end

# Variant for xi::Const (e.g. captured in closure with function_annotation=Const)
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{<:AbstractRadialBasis},
        ::Type{<:EnzymeCore.Active},
        x::EnzymeCore.Duplicated,
        xi::EnzymeCore.Const,
    )
    basis = func.val
    y = basis(x.val, xi.val)
    tape = (copy(x.val), copy(xi.val))
    return EnzymeRules.AugmentedReturn(y, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{<:AbstractRadialBasis},
        dret::EnzymeCore.Active,
        tape,
        x::EnzymeCore.Duplicated,
        xi::EnzymeCore.Const,
    )
    basis = func.val
    x_val, xi_val = tape
    grad_fn = ∇(basis)
    ∇φ = grad_fn(x_val, xi_val)
    x.dval .+= dret.val .* ∇φ
    return (nothing, nothing)
end

# =============================================================================
# Operator Evaluation Rules: _eval_op(op, x)
# =============================================================================

# Scalar-valued operator: y = W * x
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        ::Type{RT},
        op::EnzymeCore.Const{<:RadialBasisOperator},
        x::EnzymeCore.Duplicated,
    ) where {RT}
    y = _eval_op(op.val, x.val)
    shadow = _make_shadow_for_return(RT, y)
    return EnzymeRules.AugmentedReturn(y, shadow, (op.val, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        dret,
        tape,
        op::EnzymeCore.Const{<:RadialBasisOperator},
        x::EnzymeCore.Duplicated,
    )
    operator, shadow = tape
    Δy = _extract_dret_with_shadow(dret, shadow)
    # Pullback: Δx = W' * Δy
    x.dval .+= operator.weights' * Δy
    return (nothing, nothing)
end

# Vector-valued operator: y[:,d] = W[d] * x
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        ::Type{RT},
        op::EnzymeCore.Const{<:RadialBasisOperator{<:VectorValuedOperator{D}}},
        x::EnzymeCore.Duplicated,
    ) where {D, RT}
    y = _eval_op(op.val, x.val)
    shadow = _make_shadow_for_return(RT, y)
    return EnzymeRules.AugmentedReturn(y, shadow, (op.val, D, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        dret,
        tape,
        op::EnzymeCore.Const{<:RadialBasisOperator{<:VectorValuedOperator}},
        x::EnzymeCore.Duplicated,
    )
    operator, D, shadow = tape
    Δy = _extract_dret_with_shadow(dret, shadow)
    # Pullback: Δx = Σ_d W[d]' * Δy[:,d]
    for d in 1:D
        x.dval .+= operator.weights[d]' * view(Δy, :, d)
    end
    return (nothing, nothing)
end

# =============================================================================
# Operator Call Rules: op(x)
# =============================================================================

# Scalar-valued operator call
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator},
        ::Type{RT},
        x::EnzymeCore.Duplicated,
    ) where {RT}
    operator = op.val
    # Ensure weights are computed
    !RadialBasisFunctions.is_cache_valid(operator) && RadialBasisFunctions.update_weights!(operator)
    y = _eval_op(operator, x.val)
    shadow = _make_shadow_for_return(RT, y)
    return EnzymeRules.AugmentedReturn(y, shadow, (operator, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator},
        dret,
        tape,
        x::EnzymeCore.Duplicated,
    )
    operator, shadow = tape
    Δy = _extract_dret_with_shadow(dret, shadow)
    x.dval .+= operator.weights' * Δy
    return (nothing,)
end

# Vector-valued operator call
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator{<:VectorValuedOperator{D}}},
        ::Type{RT},
        x::EnzymeCore.Duplicated,
    ) where {D, RT}
    operator = op.val
    !RadialBasisFunctions.is_cache_valid(operator) && RadialBasisFunctions.update_weights!(operator)
    y = _eval_op(operator, x.val)
    shadow = _make_shadow_for_return(RT, y)
    return EnzymeRules.AugmentedReturn(y, shadow, (operator, D, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator{<:VectorValuedOperator}},
        dret,
        tape,
        x::EnzymeCore.Duplicated,
    )
    operator, D, shadow = tape
    Δy = _extract_dret_with_shadow(dret, shadow)
    for d in 1:D
        x.dval .+= operator.weights[d]' * view(Δy, :, d)
    end
    return (nothing,)
end

# =============================================================================
# Interpolator Rules
# =============================================================================

# Single point evaluation
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        interp::EnzymeCore.Const{<:Interpolator},
        ::Type{<:EnzymeCore.Active},
        x::EnzymeCore.Duplicated,
    )
    y = interp.val(x.val)
    tape = (interp.val, copy(x.val))
    return EnzymeRules.AugmentedReturn(y, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        interp_const::EnzymeCore.Const{<:Interpolator},
        dret::EnzymeCore.Active,
        tape,
        x::EnzymeCore.Duplicated,
    )
    interp, x_val = tape
    _interpolator_point_gradient!(x.dval, interp, x_val, dret.val)
    return (nothing,)
end

# Batch evaluation
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        interp::EnzymeCore.Const{<:Interpolator},
        ::Type{RT},
        xs::EnzymeCore.Duplicated{<:Vector{<:AbstractVector}},
    ) where {RT}
    ys = interp.val(xs.val)
    shadow = _make_shadow_for_return(RT, ys)
    tape = (interp.val, deepcopy(xs.val), shadow)
    return EnzymeRules.AugmentedReturn(ys, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        interp_const::EnzymeCore.Const{<:Interpolator},
        dret,
        tape,
        xs::EnzymeCore.Duplicated{<:Vector{<:AbstractVector}},
    )
    interp, xs_val, shadow = tape
    Δys = _extract_dret_with_shadow(dret, shadow)

    for (i, x_val) in enumerate(xs_val)
        _interpolator_point_gradient!(xs.dval[i], interp, x_val, Δys[i])
    end

    return (nothing,)
end

# =============================================================================
# _build_weights Rules for Shape Optimization
# =============================================================================

# Import shared utilities from main package
import RadialBasisFunctions: extract_stencil_cotangent, _get_grad_funcs
import RadialBasisFunctions: build_weights_pullback_loop!

# =============================================================================
# Helper functions
# =============================================================================

# For Duplicated return types, we need to allocate a shadow
function _make_shadow_for_return(::Type{<:EnzymeCore.Duplicated}, W::SparseMatrixCSC)
    return SparseMatrixCSC(W.m, W.n, copy(W.colptr), copy(W.rowval), zeros(eltype(W), length(W.nzval)))
end
function _make_shadow_for_return(::Type{<:EnzymeCore.Duplicated}, y::AbstractArray)
    return zero(y)
end
_make_shadow_for_return(::Type, _W) = nothing

# Helper to extract cotangent from dret (differs between Active and Duplicated return)
_extract_dret_with_shadow(dret::EnzymeCore.Active, _shadow) = dret.val
_extract_dret_with_shadow(::Type, shadow::AbstractArray) = shadow
_extract_dret_with_shadow(::Type, ::Nothing) = nothing

# Helper to construct Enzyme tangent for basis types
_make_enzyme_tangent(::Type{<:AbstractRadialBasis}, _basis, _Δε) = nothing  # PHS has no ε

function _make_enzyme_tangent(::Type{Gaussian{T, D}}, _basis::Gaussian{T, D}, Δε) where {T, D}
    return Gaussian(convert(T, Δε); poly_deg = D(0))
end

function _make_enzyme_tangent(::Type{IMQ{T, D}}, _basis::IMQ{T, D}, Δε) where {T, D}
    return IMQ(convert(T, Δε); poly_deg = D(0))
end

# Shared augmented_primal body for all _build_weights rules
function _enzyme_augmented_primal_body(ℒ_arg, data, eval_points, adjl, basis, RT, OpType; copy_data = false)
    op_val = ℒ_arg.val
    data_val = data.val
    eval_points_val = eval_points.val
    adjl_val = adjl.val
    basis_val = basis.val

    dim_space = length(first(data_val))
    mon = MonomialBasis(dim_space, basis_val.poly_deg)
    op_mon = op_val(mon)
    op_rbf = op_val(basis_val)

    W, cache = _forward_with_cache(
        data_val, eval_points_val, adjl_val, basis_val, op_rbf, op_mon, mon, OpType
    )

    shadow = _make_shadow_for_return(RT, W)
    stored_data = copy_data ? deepcopy(data_val) : data_val
    stored_eval = copy_data ? deepcopy(eval_points_val) : eval_points_val
    tape = (op_val, cache, adjl_val, basis_val, mon, stored_data, stored_eval, shadow)
    return EnzymeRules.AugmentedReturn(W, shadow, tape)
end

# Shared pullback core for all _build_weights reverse rules
function _enzyme_run_pullback_loop(dret, tape, OpType)
    op_cached, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow = tape
    ΔW = _extract_dret_with_shadow(dret, shadow)

    TD = eltype(first(data_val))
    N_data = length(data_val)
    N_eval = length(eval_points_val)
    dim_space = length(first(data_val))

    grad_Lφ_x, grad_Lφ_xi = _get_grad_funcs(OpType, basis_val, op_cached)

    Δdata_raw = [zeros(TD, dim_space) for _ in 1:N_data]
    Δeval_raw = [zeros(TD, dim_space) for _ in 1:N_eval]
    Δε_acc = Ref(zero(TD))

    build_weights_pullback_loop!(
        Δdata_raw, Δeval_raw, Δε_acc,
        (eval_idx, neighbors, k) -> extract_stencil_cotangent(ΔW, eval_idx, neighbors, k, cache.num_ops),
        cache, adjl_val, eval_points_val, data_val, basis_val, mon, op_cached, OpType,
        grad_Lφ_x, grad_Lφ_xi
    )

    return Δdata_raw, Δeval_raw, Δε_acc
end

# =============================================================================
# Explicit rules: Duplicated data, Const basis
# =============================================================================

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        ::Type{RT},
        ℒ_arg::EnzymeCore.Const{<:Union{Partial, Laplacian}},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    ) where {RT}
    return _enzyme_augmented_primal_body(ℒ_arg, data, eval_points, adjl, basis, RT, _optype(ℒ_arg.val); copy_data = true)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        dret, tape,
        ℒ_arg::EnzymeCore.Const{<:Union{Partial, Laplacian}},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    )
    return _enzyme_reverse_duplicated_data!(dret, tape, data, eval_points, _optype(ℒ_arg.val))
end

# Shared reverse for Duplicated data, Const basis
function _enzyme_reverse_duplicated_data!(dret, tape, data, eval_points, OpType)
    Δdata_raw, Δeval_raw, _ = _enzyme_run_pullback_loop(dret, tape, OpType)

    SV_data = eltype(data.dval)
    for i in eachindex(Δdata_raw)
        data.dval[i] = data.dval[i] + SV_data(Δdata_raw[i])
    end
    SV_eval = eltype(eval_points.dval)
    for i in eachindex(Δeval_raw)
        eval_points.dval[i] = eval_points.dval[i] + SV_eval(Δeval_raw[i])
    end

    return (nothing, nothing, nothing, nothing, nothing)
end

# =============================================================================
# Explicit rules: Const data, Active basis (shape parameter optimization)
# =============================================================================

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        ::Type{RT},
        ℒ_arg::EnzymeCore.Const{<:Union{Partial, Laplacian}},
        data::EnzymeCore.Const,
        eval_points::EnzymeCore.Const,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Active{B},
    ) where {RT, B <: AbstractRadialBasis}
    return _enzyme_augmented_primal_body(ℒ_arg, data, eval_points, adjl, basis, RT, _optype(ℒ_arg.val))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        dret, tape,
        ℒ_arg::EnzymeCore.Const{<:Union{Partial, Laplacian}},
        data::EnzymeCore.Const,
        eval_points::EnzymeCore.Const,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Active{B},
    ) where {B <: AbstractRadialBasis}
    return _enzyme_reverse_active_basis!(dret, tape, B, _optype(ℒ_arg.val))
end

# Shared reverse for Const data, Active basis
function _enzyme_reverse_active_basis!(dret, tape, ::Type{B}, OpType) where {B <: AbstractRadialBasis}
    _, _, Δε_acc = _enzyme_run_pullback_loop(dret, tape, OpType)
    Δbasis = _make_enzyme_tangent(B, tape[4], Δε_acc[])
    return (nothing, nothing, nothing, nothing, Δbasis)
end

end # module
