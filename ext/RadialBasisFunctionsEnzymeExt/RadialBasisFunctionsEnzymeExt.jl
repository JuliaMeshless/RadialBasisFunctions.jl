"""
    RadialBasisFunctionsEnzymeExt

Package extension that provides native Enzyme.jl AD support for RadialBasisFunctions.jl
using EnzymeRules (augmented_primal + reverse).

This extension requires Enzyme to be loaded.

Native rules are provided for:
- Basis function evaluation: all AbstractRadialBasis subtypes
- Operator evaluation: `_eval_op(op, x)` for scalar and vector-valued operators
- Operator call syntax: `op(x)`
- Interpolator constructor: `Interpolator(x, y, basis)`
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
import RadialBasisFunctions: IMQ, Gaussian, _tangent_basis
import RadialBasisFunctions: AbstractRadialBasis, Jacobian
import RadialBasisFunctions: _build_weights, Partial, Laplacian, _optype
import RadialBasisFunctions: MonomialBasis

# Import backward pass support from main package
import RadialBasisFunctions: _forward_with_cache

# Import gradient function and shared helpers
const ∇ = RadialBasisFunctions.∇
import RadialBasisFunctions: _interpolator_point_gradient!
import RadialBasisFunctions: _interpolator_constructor_backward, _interpolator_weight_cotangents!
import RadialBasisFunctions: _build_collocation_matrix!

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
    return EnzymeRules.AugmentedReturn(y, shadow, shadow)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        dret,
        tape,
        op::EnzymeCore.Const{<:RadialBasisOperator},
        x::EnzymeCore.Duplicated,
    )
    shadow = tape
    Δy = _extract_dret_with_shadow(dret, shadow)
    # Pullback: Δx = W' * Δy
    x.dval .+= op.val.weights' * Δy
    return (nothing, nothing)
end

# Vector-valued operator: y[:,d] = W[d] * x
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        ::Type{RT},
        op::EnzymeCore.Const{<:RadialBasisOperator{<:Jacobian{D}}},
        x::EnzymeCore.Duplicated,
    ) where {D, RT}
    y = _eval_op(op.val, x.val)
    shadow = _make_shadow_for_return(RT, y)
    return EnzymeRules.AugmentedReturn(y, shadow, shadow)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        dret,
        tape,
        op::EnzymeCore.Const{<:RadialBasisOperator{<:Jacobian}},
        x::EnzymeCore.Duplicated,
    )
    operator = op.val
    D = length(operator.weights)
    shadow = tape
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
    y = operator.weights * x.val
    shadow = _make_shadow_for_return(RT, y)
    return EnzymeRules.AugmentedReturn(y, shadow, shadow)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator},
        dret,
        tape,
        x::EnzymeCore.Duplicated,
    )
    shadow = tape
    Δy = _extract_dret_with_shadow(dret, shadow)
    x.dval .+= op.val.weights' * Δy
    return (nothing,)
end

# Vector-valued operator call
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator{<:Jacobian{D}}},
        ::Type{RT},
        x::EnzymeCore.Duplicated,
    ) where {D, RT}
    operator = op.val
    y = _eval_op(operator, x.val)
    shadow = _make_shadow_for_return(RT, y)
    return EnzymeRules.AugmentedReturn(y, shadow, shadow)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator{<:Jacobian}},
        dret,
        tape,
        x::EnzymeCore.Duplicated,
    )
    operator = op.val
    D = length(operator.weights)
    shadow = tape
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
# Interpolator Constructor Rule (#147; mirrors the Mooncake rrule!!)
# =============================================================================
# Makes the constructor opaque so Enzyme never traces the linear solve. The
# shadow Interpolator's y-field aliases y.dval; its weight fields collect
# cotangents deposited by the Duplicated-Interpolator evaluation rules, which
# the reverse pass pulls back to Δy via the implicit function theorem.

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{Type{Interpolator}},
        ::Type{RT},
        x::EnzymeCore.Const{<:AbstractVector},
        y::EnzymeCore.Duplicated{<:AbstractVector},
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    ) where {RT}
    x_val, y_val, basis_val = x.val, y.val, basis.val

    # Forward pass: reproduce the Interpolator constructor (interpolation.jl)
    dim = length(first(x_val))
    k = length(x_val)
    npoly = binomial(dim + basis_val.poly_deg, basis_val.poly_deg)
    n = k + npoly
    mon = MonomialBasis(dim, basis_val.poly_deg)
    data_type = promote_type(eltype(first(x_val)), eltype(y_val))
    A = Symmetric(zeros(data_type, n, n))
    _build_collocation_matrix!(A, x_val, basis_val, mon, k)
    b = vcat(y_val, zeros(data_type, npoly))
    F = bunchkaufman!(A, true)
    w = F \ b

    interp = Interpolator(x_val, y_val, w[1:k], w[(k + 1):end], basis_val, mon)
    shadow = Interpolator(
        zero(x_val), y.dval, zeros(data_type, k), zeros(data_type, npoly), basis_val, mon
    )
    tape = (F, k, shadow)
    # RT's primal is a UnionAll (MonomialBasis hides a function-type parameter),
    # so Enzyme requires the explicitly-typed AugmentedReturn with an Any tape
    PT = _primal_type(RT)
    return EnzymeRules.AugmentedReturn{PT, PT, Any}(interp, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{Type{Interpolator}},
        dret,
        tape,
        x::EnzymeCore.Const{<:AbstractVector},
        y::EnzymeCore.Duplicated{<:AbstractVector},
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    )
    F, k, shadow = tape
    # F is the cached Bunch-Kaufman factorization: O(n²) solve, no refactorization
    Δy = _interpolator_constructor_backward(shadow.rbf_weights, shadow.monomial_weights, F, k)
    y.dval .+= Δy
    fill!(shadow.rbf_weights, zero(eltype(shadow.rbf_weights)))
    fill!(shadow.monomial_weights, zero(eltype(shadow.monomial_weights)))
    return (nothing, nothing, nothing)
end

# =============================================================================
# Duplicated-Interpolator Evaluation Rules
# =============================================================================
# For an Interpolator constructed inside the differentiated region evaluated at
# constant points: the cotangent flows into the interpolator's weight shadows
# instead of the evaluation point. Enzyme annotates the interpolator Duplicated
# when the basis is float-free (PHS) but MixedDuplicated (shadow boxed in a
# RefValue) when the basis carries an active float (IMQ/Gaussian ε).

const DupInterp{T} = Union{EnzymeCore.Duplicated{T}, EnzymeCore.MixedDuplicated{T}}

_interp_shadow(interp::EnzymeCore.Duplicated) = interp.dval
_interp_shadow(interp::EnzymeCore.MixedDuplicated) = interp.dval[]

# Single point evaluation
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        interp::DupInterp{<:Interpolator},
        ::Type{<:EnzymeCore.Active},
        x::EnzymeCore.Const,
    )
    y = interp.val(x.val)
    tape = copy(x.val)
    return EnzymeRules.AugmentedReturn(y, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        interp::DupInterp{<:Interpolator},
        dret::EnzymeCore.Active,
        tape,
        x::EnzymeCore.Const,
    )
    x_val = tape
    shadow = _interp_shadow(interp)
    _interpolator_weight_cotangents!(
        shadow.rbf_weights, shadow.monomial_weights, interp.val, x_val, dret.val
    )
    return (nothing,)
end

# Single point evaluation at an isbits point (e.g. SVector): Enzyme passes it
# Active, so the point cotangent is returned by value alongside the weight
# cotangents deposited into the interpolator shadow
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        interp::DupInterp{<:Interpolator},
        ::Type{<:EnzymeCore.Active},
        x::EnzymeCore.Active,
    )
    y = interp.val(x.val)
    tape = x.val
    return EnzymeRules.AugmentedReturn(y, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        interp::DupInterp{<:Interpolator},
        dret::EnzymeCore.Active,
        tape,
        x::EnzymeCore.Active{XT},
    ) where {XT}
    x_val = tape
    shadow = _interp_shadow(interp)
    _interpolator_weight_cotangents!(
        shadow.rbf_weights, shadow.monomial_weights, interp.val, x_val, dret.val
    )
    Δx = zeros(eltype(x_val), length(x_val))
    _interpolator_point_gradient!(Δx, interp.val, x_val, dret.val)
    return (XT(Tuple(Δx))::XT,)
end

# Batch evaluation
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        interp::DupInterp{<:Interpolator},
        ::Type{RT},
        xs::EnzymeCore.Const{<:Vector{<:AbstractVector}},
    ) where {RT}
    ys = interp.val(xs.val)
    shadow = _make_shadow_for_return(RT, ys)
    tape = (deepcopy(xs.val), shadow)
    return EnzymeRules.AugmentedReturn(ys, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        interp::DupInterp{<:Interpolator},
        dret,
        tape,
        xs::EnzymeCore.Const{<:Vector{<:AbstractVector}},
    )
    xs_val, shadow = tape
    Δys = _extract_dret_with_shadow(dret, shadow)
    ishadow = _interp_shadow(interp)
    for (i, x_val) in enumerate(xs_val)
        _interpolator_weight_cotangents!(
            ishadow.rbf_weights, ishadow.monomial_weights, interp.val, x_val, Δys[i]
        )
    end
    return (nothing,)
end

# =============================================================================
# _build_weights Rules for Shape Optimization
# =============================================================================

# Import shared utilities from main package
import RadialBasisFunctions: extract_stencil_cotangent!, _get_grad_funcs
import RadialBasisFunctions: run_build_weights_pullback

# =============================================================================
# Helper functions
# =============================================================================

# For Duplicated/DuplicatedNoNeed return types, we need to allocate a shadow
function _make_shadow_for_return(::Type{<:EnzymeCore.Duplicated}, W::SparseMatrixCSC)
    return SparseMatrixCSC(W.m, W.n, copy(W.colptr), copy(W.rowval), zeros(eltype(W), length(W.nzval)))
end
function _make_shadow_for_return(::Type{<:EnzymeCore.Duplicated}, y::AbstractArray)
    return zero(y)
end
function _make_shadow_for_return(::Type{<:EnzymeCore.DuplicatedNoNeed}, W::SparseMatrixCSC)
    return SparseMatrixCSC(W.m, W.n, copy(W.colptr), copy(W.rowval), zeros(eltype(W), length(W.nzval)))
end
function _make_shadow_for_return(::Type{<:EnzymeCore.DuplicatedNoNeed}, y::AbstractArray)
    return zero(y)
end
_make_shadow_for_return(::Type, _W) = nothing

# Extract primal type T from Enzyme annotation types like Duplicated{T}, Active{T}, etc.
_primal_type(::Type{<:EnzymeCore.Duplicated{T}}) where {T} = T
_primal_type(::Type{<:EnzymeCore.DuplicatedNoNeed{T}}) where {T} = T
_primal_type(::Type{<:EnzymeCore.Active{T}}) where {T} = T
_primal_type(::Type{<:EnzymeCore.Const{T}}) where {T} = T

# Helper to extract cotangent from dret (differs between Active and Duplicated return)
_extract_dret_with_shadow(dret::EnzymeCore.Active, _shadow) = dret.val
_extract_dret_with_shadow(::Type, shadow::AbstractArray) = shadow
_extract_dret_with_shadow(::Type, ::Nothing) = nothing

# Helper to construct Enzyme tangent for basis types. Uses the non-validating
# _tangent_basis constructor: Δε is a derivative and may be negative.
_make_enzyme_tangent(::Type{<:AbstractRadialBasis}, _basis, _Δε) = nothing  # PHS has no ε

function _make_enzyme_tangent(::Type{Gaussian{T, D}}, _basis::Gaussian{T, D}, Δε) where {T, D}
    return _tangent_basis(Gaussian{T, D}, Δε)
end

function _make_enzyme_tangent(::Type{IMQ{T, D}}, _basis::IMQ{T, D}, Δε) where {T, D}
    return _tangent_basis(IMQ{T, D}, Δε)
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
    PT = _primal_type(RT)
    ST = shadow === nothing ? Nothing : PT
    return EnzymeRules.AugmentedReturn{PT, ST, typeof(tape)}(W, shadow, tape)
end

# Shared pullback core for all _build_weights reverse rules
function _enzyme_run_pullback_loop(dret, tape, OpType)
    op_cached, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow = tape
    ΔW = _extract_dret_with_shadow(dret, shadow)

    grad_Lφ_x, grad_Lφ_xi = _get_grad_funcs(OpType, basis_val, op_cached)

    return run_build_weights_pullback(
        (Δw, eval_idx, neighbors, k) -> extract_stencil_cotangent!(Δw, ΔW, eval_idx, neighbors, k),
        cache, adjl_val, eval_points_val, data_val, basis_val, mon, op_cached, OpType,
        grad_Lφ_x, grad_Lφ_xi
    )
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
