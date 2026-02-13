"""
    RadialBasisFunctionsEnzymeExt

Package extension that provides native Enzyme.jl AD support for RadialBasisFunctions.jl
using EnzymeRules (augmented_primal + reverse).

This extension requires Enzyme to be loaded.

Native rules are provided for:
- Basis function evaluation: PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian
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
import RadialBasisFunctions: PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian
import RadialBasisFunctions: AbstractRadialBasis, VectorValuedOperator
import RadialBasisFunctions: _build_weights, Partial, Laplacian
import RadialBasisFunctions: MonomialBasis

# Import backward pass support from main package
import RadialBasisFunctions: StencilForwardCache, WeightsBuildForwardCache
import RadialBasisFunctions: backward_stencil_partial!, backward_stencil_laplacian!
import RadialBasisFunctions: backward_stencil_partial_with_ε!, backward_stencil_laplacian_with_ε!
import RadialBasisFunctions: _forward_with_cache
import RadialBasisFunctions: grad_applied_partial_wrt_x, grad_applied_partial_wrt_xi
import RadialBasisFunctions: grad_applied_laplacian_wrt_x, grad_applied_laplacian_wrt_xi

# Import gradient function
const ∇ = RadialBasisFunctions.∇

# =============================================================================
# Basis Function Rules
# =============================================================================

# Helper macro to define basis function rules for a given type
macro define_basis_rule(BasisType)
    return quote
        # Both x and xi are Duplicated (both being differentiated)
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::EnzymeCore.Const{<:$(esc(BasisType))},
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
                func::EnzymeCore.Const{<:$(esc(BasisType))},
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

        # x is Duplicated, xi is Const (xi captured in closure)
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::EnzymeCore.Const{<:$(esc(BasisType))},
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
                func::EnzymeCore.Const{<:$(esc(BasisType))},
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
    end
end

@define_basis_rule PHS1
@define_basis_rule PHS3
@define_basis_rule PHS5
@define_basis_rule PHS7
@define_basis_rule IMQ
@define_basis_rule Gaussian

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
    shadow = RT <: EnzymeCore.Duplicated ? zero(y) : nothing
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
    dy = dret isa EnzymeCore.Active ? dret.val : shadow
    if dy !== nothing
        x.dval .+= operator.weights' * dy
        shadow !== nothing && fill!(shadow, 0)
    end
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
    shadow = RT <: EnzymeCore.Duplicated ? zero(y) : nothing
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
    dy = dret isa EnzymeCore.Active ? dret.val : shadow
    if dy !== nothing
        for d in 1:D
            x.dval .+= operator.weights[d]' * view(dy, :, d)
        end
        shadow !== nothing && fill!(shadow, 0)
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
    shadow = RT <: EnzymeCore.Duplicated ? zero(y) : nothing
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
    dy = dret isa EnzymeCore.Active ? dret.val : shadow
    if dy !== nothing
        x.dval .+= operator.weights' * dy
        shadow !== nothing && fill!(shadow, 0)
    end
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
    y = _eval_op(operator, x.val)
    shadow = RT <: EnzymeCore.Duplicated ? zero(y) : nothing
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
    dy = dret isa EnzymeCore.Active ? dret.val : shadow
    if dy !== nothing
        for d in 1:D
            x.dval .+= operator.weights[d]' * view(dy, :, d)
        end
        shadow !== nothing && fill!(shadow, 0)
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
    Δy = dret.val

    # RBF contribution: Σᵢ wᵢ ∇φ(x, xᵢ)
    grad_fn = ∇(interp.rbf_basis)
    for i in eachindex(interp.rbf_weights)
        ∇φ = grad_fn(x_val, interp.x[i])
        x.dval .+= (interp.rbf_weights[i] * Δy) .* ∇φ
    end

    # Polynomial contribution: Σⱼ wⱼ ∇pⱼ(x)
    if !isempty(interp.monomial_weights)
        dim = length(x_val)
        n_terms = length(interp.monomial_weights)
        ∇mon = ∇(interp.monomial_basis)
        ∇p = zeros(eltype(x_val), n_terms, dim)
        ∇mon(∇p, x_val)

        for j in eachindex(interp.monomial_weights)
            x.dval .+= (interp.monomial_weights[j] * Δy) .* view(∇p, j, :)
        end
    end

    return (nothing,)
end

# Batch evaluation
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        interp::EnzymeCore.Const{<:Interpolator},
        ::Type{<:EnzymeCore.Active},
        xs::EnzymeCore.Duplicated{<:Vector{<:AbstractVector}},
    )
    ys = interp.val(xs.val)
    tape = (interp.val, deepcopy(xs.val))
    return EnzymeRules.AugmentedReturn(ys, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        interp_const::EnzymeCore.Const{<:Interpolator},
        dret::EnzymeCore.Active,
        tape,
        xs::EnzymeCore.Duplicated{<:Vector{<:AbstractVector}},
    )
    interp, xs_val = tape
    Δys = dret.val

    for (i, x_val) in enumerate(xs_val)
        Δy = Δys[i]

        # RBF contribution
        grad_fn = ∇(interp.rbf_basis)
        for j in eachindex(interp.rbf_weights)
            ∇φ = grad_fn(x_val, interp.x[j])
            xs.dval[i] = xs.dval[i] + (interp.rbf_weights[j] * Δy) .* ∇φ
        end

        # Polynomial contribution
        if !isempty(interp.monomial_weights)
            dim = length(x_val)
            n_terms = length(interp.monomial_weights)
            ∇mon = ∇(interp.monomial_basis)
            ∇p = zeros(eltype(x_val), n_terms, dim)
            ∇mon(∇p, x_val)

            for k in eachindex(interp.monomial_weights)
                xs.dval[i] = xs.dval[i] + (interp.monomial_weights[k] * Δy) .* view(∇p, k, :)
            end
        end
    end

    return (nothing,)
end

# =============================================================================
# _build_weights Rules for Shape Optimization
# =============================================================================

# Helper to materialize sparse matrix tangent for Enzyme
function materialize_sparse_tangent_enzyme(ΔW, W::SparseMatrixCSC)
    if ΔW isa SparseMatrixCSC
        return ΔW
    end
    # If ΔW is dense or some other form, convert appropriately
    return SparseMatrixCSC(W.m, W.n, copy(W.colptr), copy(W.rowval), copy(nonzeros(W)))
end

# Extract stencil cotangent from sparse matrix
function extract_stencil_cotangent_enzyme(
        ΔW::AbstractMatrix{T}, eval_idx::Int, neighbors::Vector{Int}, k::Int, num_ops::Int
    ) where {T}
    Δw = zeros(T, k, num_ops)
    for (local_idx, global_idx) in enumerate(neighbors)
        Δw[local_idx, 1] = ΔW[eval_idx, global_idx]
    end
    return Δw
end

# =============================================================================
# Unified _build_weights rule generation via macro
# =============================================================================

# Helper functions to get gradient functions and call backward stencil for each operator type
_get_grad_funcs(::Type{<:Partial}, basis, ℒ) = (
    grad_applied_partial_wrt_x(basis, ℒ.dim),
    grad_applied_partial_wrt_xi(basis, ℒ.dim),
)
_get_grad_funcs(::Type{<:Laplacian}, basis, ℒ) = (
    grad_applied_laplacian_wrt_x(basis),
    grad_applied_laplacian_wrt_xi(basis),
)

# Backward stencil dispatch (without ε)
function _call_backward_stencil!(
        ::Type{<:Partial}, Δlocal_data, Δeval_pt, Δw, stencil_cache, neighbors,
        eval_point, local_data, basis, mon, k, ℒ, grad_Lφ_x, grad_Lφ_xi
    )
    return backward_stencil_partial!(
        Δlocal_data, Δeval_pt, Δw, stencil_cache, neighbors,
        eval_point, local_data, basis, mon, k, ℒ.dim, grad_Lφ_x, grad_Lφ_xi
    )
end

function _call_backward_stencil!(
        ::Type{<:Laplacian}, Δlocal_data, Δeval_pt, Δw, stencil_cache, neighbors,
        eval_point, local_data, basis, mon, k, ℒ, grad_Lφ_x, grad_Lφ_xi
    )
    return backward_stencil_laplacian!(
        Δlocal_data, Δeval_pt, Δw, stencil_cache, neighbors,
        eval_point, local_data, basis, mon, k, grad_Lφ_x, grad_Lφ_xi
    )
end

# Backward stencil dispatch (with ε)
function _call_backward_stencil_with_ε!(
        ::Type{<:Partial}, Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache, neighbors,
        eval_point, local_data, basis, mon, k, ℒ, grad_Lφ_x, grad_Lφ_xi
    )
    return backward_stencil_partial_with_ε!(
        Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache, neighbors,
        eval_point, local_data, basis, mon, k, ℒ.dim, grad_Lφ_x, grad_Lφ_xi
    )
end

function _call_backward_stencil_with_ε!(
        ::Type{<:Laplacian}, Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache, neighbors,
        eval_point, local_data, basis, mon, k, ℒ, grad_Lφ_x, grad_Lφ_xi
    )
    return backward_stencil_laplacian_with_ε!(
        Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache, neighbors,
        eval_point, local_data, basis, mon, k, grad_Lφ_x, grad_Lφ_xi
    )
end

"""
Generate augmented_primal and reverse rules for _build_weights with different argument activities.

Arguments:
- OpType: Partial or Laplacian
- data_activity: :Duplicated or :Const
- basis_activity: :Const, :Active, or :Duplicated
"""
macro define_build_weights_rule(OpType, data_activity, basis_activity)
    # Determine type annotations for signature
    data_type = data_activity == :Duplicated ? :(EnzymeCore.Duplicated) : :(EnzymeCore.Const)
    eval_type = data_activity == :Duplicated ? :(EnzymeCore.Duplicated) : :(EnzymeCore.Const)

    if basis_activity == :Const
        basis_type = :(EnzymeCore.Const{<:AbstractRadialBasis})
        basis_type_param = nothing
    elseif basis_activity == :Active
        basis_type = :(EnzymeCore.Active{B})
        basis_type_param = :(where{B <: AbstractRadialBasis})
    else  # :Duplicated
        basis_type = :(EnzymeCore.Duplicated{B})
        basis_type_param = :(where{B <: AbstractRadialBasis})
    end

    # Determine if we need RT type parameter for shadow allocation
    needs_rt = basis_activity != :Const || data_activity == :Const
    rt_param = needs_rt ? :(::Type{RT}) : :(::Type{<:EnzymeCore.Active})
    rt_where = needs_rt ? :(where{RT}) : nothing

    # Build augmented_primal signature
    aug_sig = if basis_activity == :Const && !needs_rt
        quote
            function EnzymeRules.augmented_primal(
                    config::EnzymeRules.RevConfigWidth{1},
                    func::EnzymeCore.Const{typeof(_build_weights)},
                    $rt_param,
                    ℒ_arg::EnzymeCore.Const{<:$OpType},
                    data::$data_type,
                    eval_points::$eval_type,
                    adjl::EnzymeCore.Const,
                    basis::$basis_type,
                )
            end
        end
    else
        quote
            function EnzymeRules.augmented_primal(
                    config::EnzymeRules.RevConfigWidth{1},
                    func::EnzymeCore.Const{typeof(_build_weights)},
                    $rt_param,
                    ℒ_arg::EnzymeCore.Const{<:$OpType},
                    data::$data_type,
                    eval_points::$eval_type,
                    adjl::EnzymeCore.Const,
                    basis::$basis_type,
                )
                return $rt_where $ (basis_type_param...)
            end
        end
    end

    # Generate the actual code
    return quote
        # Augmented primal
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::EnzymeCore.Const{typeof(_build_weights)},
                $(needs_rt ? :(::Type{RT}) : :(::Type{<:EnzymeCore.Active})),
                ℒ_arg::EnzymeCore.Const{<:$(esc(OpType))},
                data::$(esc(data_type)),
                eval_points::$(esc(eval_type)),
                adjl::EnzymeCore.Const,
                basis::$(esc(basis_type)),
            )
            $(
                needs_rt && basis_activity != :Const ? :(where{RT, B <: AbstractRadialBasis}) :
                    needs_rt ? :(where{RT}) :
                    basis_activity != :Const ? :(where{B <: AbstractRadialBasis}) : nothing
            )

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
                data_val, eval_points_val, adjl_val, basis_val, op_rbf, op_mon, mon, $(esc(OpType))
            )

            $(
                if basis_activity == :Active
                    quote
                        shadow = _make_shadow_for_return(RT, W)
                        tape = (op_val, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow)
                        return EnzymeRules.AugmentedReturn(W, shadow, tape)
                    end
                elseif data_activity == :Duplicated
                    quote
                        tape = (op_val, cache, adjl_val, basis_val, mon, deepcopy(data_val), deepcopy(eval_points_val))
                        return EnzymeRules.AugmentedReturn(W, nothing, tape)
                    end
                else  # Const data, Const or Duplicated basis
                    quote
                        tape = (op_val, cache, adjl_val, basis_val, mon, data_val, eval_points_val)
                        return EnzymeRules.AugmentedReturn(W, nothing, tape)
                    end
                end
            )
        end

        # Reverse pass
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::EnzymeCore.Const{typeof(_build_weights)},
                $(basis_activity == :Active ? :dret : :(dret::EnzymeCore.Active)),
                tape,
                ℒ_arg::EnzymeCore.Const{<:$(esc(OpType))},
                data::$(esc(data_type)),
                eval_points::$(esc(eval_type)),
                adjl::EnzymeCore.Const,
                basis::$(esc(basis_type)),
            )
            $(basis_activity != :Const ? :(where{B <: AbstractRadialBasis}) : nothing)

            $(
                if basis_activity == :Active
                    :((op_cached, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow) = tape)
                else
                    :((op_cached, cache, adjl_val, basis_val, mon, data_val, eval_points_val) = tape)
                end
            )

            $(
                if basis_activity == :Active
                    :(ΔW = _extract_dret_with_shadow(dret, shadow))
                else
                    :(ΔW = dret.val)
                end
            )

            TD = eltype(first(data_val))
            N_eval = length(eval_points_val)
            k = cache.k

            grad_Lφ_x, grad_Lφ_xi = _get_grad_funcs($(esc(OpType)), basis_val, op_cached)

            $(
                if basis_activity != :Const
                    :(Δε_acc = Ref(zero(TD)))
                else
                    nothing
                end
            )

            for eval_idx in 1:N_eval
                neighbors = adjl_val[eval_idx]
                eval_point = eval_points_val[eval_idx]
                stencil_cache = cache.stencil_caches[eval_idx]

                Δw = extract_stencil_cotangent_enzyme(ΔW, eval_idx, neighbors, k, cache.num_ops)

                if sum(abs, Δw) > 0
                    local_data = [data_val[i] for i in neighbors]
                    Δlocal_data = [zeros(TD, length(first(data_val))) for _ in 1:k]
                    Δeval_pt = zeros(TD, length(eval_point))

                    $(
                        if basis_activity == :Const
                            quote
                                _call_backward_stencil!(
                                    $(esc(OpType)), Δlocal_data, Δeval_pt, Δw, stencil_cache, collect(1:k),
                                    eval_point, local_data, basis_val, mon, k, op_cached, grad_Lφ_x, grad_Lφ_xi
                                )
                            end
                        else
                            quote
                                _call_backward_stencil_with_ε!(
                                    $(esc(OpType)), Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache, collect(1:k),
                                    eval_point, local_data, basis_val, mon, k, op_cached, grad_Lφ_x, grad_Lφ_xi
                                )
                            end
                        end
                    )

                    $(
                        if data_activity == :Duplicated
                            quote
                                for (local_idx, global_idx) in enumerate(neighbors)
                                    data.dval[global_idx] = data.dval[global_idx] + Δlocal_data[local_idx]
                                end
                                eval_points.dval[eval_idx] = eval_points.dval[eval_idx] + Δeval_pt
                            end
                        else
                            nothing
                        end
                    )
                end
            end

            $(
                if basis_activity == :Active
                    quote
                        Δbasis = _make_enzyme_tangent(B, basis_val, Δε_acc[])
                        return (nothing, nothing, nothing, nothing, Δbasis)
                    end
                elseif basis_activity == :Duplicated
                    quote
                        _accumulate_basis_gradient!(basis.dval, Δε_acc[])
                        return (nothing, nothing, nothing, nothing, nothing)
                    end
                else
                    :(return (nothing, nothing, nothing, nothing, nothing))
                end
            )
        end
    end
end

# =============================================================================
# Helper functions (must be defined before macro invocation)
# =============================================================================

# For Duplicated return types, we need to allocate a shadow matrix
function _make_shadow_for_return(::Type{<:EnzymeCore.Duplicated}, W::SparseMatrixCSC)
    return SparseMatrixCSC(W.m, W.n, copy(W.colptr), copy(W.rowval), zeros(eltype(W), length(W.nzval)))
end
_make_shadow_for_return(::Type, _W) = nothing

# Helper to extract cotangent from dret (differs between Active and Duplicated return)
_extract_dret_with_shadow(dret::EnzymeCore.Active, _shadow) = dret.val
_extract_dret_with_shadow(::Type, shadow::AbstractMatrix) = shadow
_extract_dret_with_shadow(::Type, ::Nothing) = nothing

# Helper to construct Enzyme tangent for basis types
_make_enzyme_tangent(::Type{<:AbstractRadialBasis}, _basis, _Δε) = nothing  # PHS has no ε

function _make_enzyme_tangent(::Type{Gaussian{T, D}}, _basis::Gaussian{T, D}, Δε) where {T, D}
    return Gaussian(convert(T, Δε); poly_deg = D(0))
end

function _make_enzyme_tangent(::Type{IMQ{T, D}}, _basis::IMQ{T, D}, Δε) where {T, D}
    return IMQ(convert(T, Δε); poly_deg = D(0))
end

# Helper to accumulate gradient into basis shadow
_accumulate_basis_gradient!(::Gaussian{T}, _Δε) where {T <: Number} = nothing
_accumulate_basis_gradient!(::IMQ{T}, _Δε) where {T <: Number} = nothing

function _accumulate_basis_gradient!(shadow::Gaussian{T}, Δε) where {T <: AbstractVector}
    shadow.ε[1] += Δε
    return nothing
end

function _accumulate_basis_gradient!(shadow::IMQ{T}, Δε) where {T <: AbstractVector}
    shadow.ε[1] += Δε
    return nothing
end

_accumulate_basis_gradient!(_shadow, _Δε) = nothing

# =============================================================================
# Explicit rules (replacing macro due to Julia version compatibility issues)
# =============================================================================

# Partial with Duplicated data, Const basis
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        ::Type{RT},
        ℒ_arg::EnzymeCore.Const{<:Partial},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    ) where {RT}
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
        data_val, eval_points_val, adjl_val, basis_val, op_rbf, op_mon, mon, Partial
    )

    shadow = _make_shadow_for_return(RT, W)
    tape = (op_val, cache, adjl_val, basis_val, mon, deepcopy(data_val), deepcopy(eval_points_val), shadow)
    return EnzymeRules.AugmentedReturn(W, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        dret,
        tape,
        ℒ_arg::EnzymeCore.Const{<:Partial},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    )
    op_cached, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow = tape
    ΔW = _extract_dret_with_shadow(dret, shadow)

    TD = eltype(first(data_val))
    N_eval = length(eval_points_val)
    k = cache.k

    grad_Lφ_x, grad_Lφ_xi = _get_grad_funcs(Partial, basis_val, op_cached)

    for eval_idx in 1:N_eval
        neighbors = adjl_val[eval_idx]
        eval_point = eval_points_val[eval_idx]
        stencil_cache = cache.stencil_caches[eval_idx]

        Δw = extract_stencil_cotangent_enzyme(ΔW, eval_idx, neighbors, k, cache.num_ops)

        if sum(abs, Δw) > 0
            local_data = [data_val[i] for i in neighbors]
            Δlocal_data = [zeros(TD, length(first(data_val))) for _ in 1:k]
            Δeval_pt = zeros(TD, length(eval_point))

            _call_backward_stencil!(
                Partial, Δlocal_data, Δeval_pt, Δw, stencil_cache, collect(1:k),
                eval_point, local_data, basis_val, mon, k, op_cached, grad_Lφ_x, grad_Lφ_xi
            )

            for (local_idx, global_idx) in enumerate(neighbors)
                data.dval[global_idx] = data.dval[global_idx] + Δlocal_data[local_idx]
            end
            eval_points.dval[eval_idx] = eval_points.dval[eval_idx] + Δeval_pt
        end
    end

    return (nothing, nothing, nothing, nothing, nothing)
end

# Laplacian with Duplicated data, Const basis
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        ::Type{RT},
        ℒ_arg::EnzymeCore.Const{<:Laplacian},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    ) where {RT}
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
        data_val, eval_points_val, adjl_val, basis_val, op_rbf, op_mon, mon, Laplacian
    )

    shadow = _make_shadow_for_return(RT, W)
    tape = (op_val, cache, adjl_val, basis_val, mon, deepcopy(data_val), deepcopy(eval_points_val), shadow)
    return EnzymeRules.AugmentedReturn(W, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        dret,
        tape,
        ℒ_arg::EnzymeCore.Const{<:Laplacian},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    )
    op_cached, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow = tape
    ΔW = _extract_dret_with_shadow(dret, shadow)

    TD = eltype(first(data_val))
    N_eval = length(eval_points_val)
    k = cache.k

    grad_Lφ_x, grad_Lφ_xi = _get_grad_funcs(Laplacian, basis_val, op_cached)

    for eval_idx in 1:N_eval
        neighbors = adjl_val[eval_idx]
        eval_point = eval_points_val[eval_idx]
        stencil_cache = cache.stencil_caches[eval_idx]

        Δw = extract_stencil_cotangent_enzyme(ΔW, eval_idx, neighbors, k, cache.num_ops)

        if sum(abs, Δw) > 0
            local_data = [data_val[i] for i in neighbors]
            Δlocal_data = [zeros(TD, length(first(data_val))) for _ in 1:k]
            Δeval_pt = zeros(TD, length(eval_point))

            _call_backward_stencil!(
                Laplacian, Δlocal_data, Δeval_pt, Δw, stencil_cache, collect(1:k),
                eval_point, local_data, basis_val, mon, k, op_cached, grad_Lφ_x, grad_Lφ_xi
            )

            for (local_idx, global_idx) in enumerate(neighbors)
                data.dval[global_idx] = data.dval[global_idx] + Δlocal_data[local_idx]
            end
            eval_points.dval[eval_idx] = eval_points.dval[eval_idx] + Δeval_pt
        end
    end

    return (nothing, nothing, nothing, nothing, nothing)
end

# Partial with Const data, Active basis (for shape parameter)
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        ::Type{RT},
        ℒ_arg::EnzymeCore.Const{<:Partial},
        data::EnzymeCore.Const,
        eval_points::EnzymeCore.Const,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Active{B},
    ) where {RT, B <: AbstractRadialBasis}
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
        data_val, eval_points_val, adjl_val, basis_val, op_rbf, op_mon, mon, Partial
    )

    shadow = _make_shadow_for_return(RT, W)
    tape = (op_val, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow)
    return EnzymeRules.AugmentedReturn(W, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        dret,
        tape,
        ℒ_arg::EnzymeCore.Const{<:Partial},
        data::EnzymeCore.Const,
        eval_points::EnzymeCore.Const,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Active{B},
    ) where {B <: AbstractRadialBasis}
    op_cached, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow = tape
    ΔW = _extract_dret_with_shadow(dret, shadow)

    TD = eltype(first(data_val))
    N_eval = length(eval_points_val)
    k = cache.k

    grad_Lφ_x, grad_Lφ_xi = _get_grad_funcs(Partial, basis_val, op_cached)
    Δε_acc = Ref(zero(TD))

    for eval_idx in 1:N_eval
        neighbors = adjl_val[eval_idx]
        eval_point = eval_points_val[eval_idx]
        stencil_cache = cache.stencil_caches[eval_idx]

        Δw = extract_stencil_cotangent_enzyme(ΔW, eval_idx, neighbors, k, cache.num_ops)

        if sum(abs, Δw) > 0
            local_data = [data_val[i] for i in neighbors]
            Δlocal_data = [zeros(TD, length(first(data_val))) for _ in 1:k]
            Δeval_pt = zeros(TD, length(eval_point))

            _call_backward_stencil_with_ε!(
                Partial, Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache, collect(1:k),
                eval_point, local_data, basis_val, mon, k, op_cached, grad_Lφ_x, grad_Lφ_xi
            )
        end
    end

    Δbasis = _make_enzyme_tangent(B, basis_val, Δε_acc[])
    return (nothing, nothing, nothing, nothing, Δbasis)
end

# Laplacian with Const data, Active basis (for shape parameter)
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        ::Type{RT},
        ℒ_arg::EnzymeCore.Const{<:Laplacian},
        data::EnzymeCore.Const,
        eval_points::EnzymeCore.Const,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Active{B},
    ) where {RT, B <: AbstractRadialBasis}
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
        data_val, eval_points_val, adjl_val, basis_val, op_rbf, op_mon, mon, Laplacian
    )

    shadow = _make_shadow_for_return(RT, W)
    tape = (op_val, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow)
    return EnzymeRules.AugmentedReturn(W, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        dret,
        tape,
        ℒ_arg::EnzymeCore.Const{<:Laplacian},
        data::EnzymeCore.Const,
        eval_points::EnzymeCore.Const,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Active{B},
    ) where {B <: AbstractRadialBasis}
    op_cached, cache, adjl_val, basis_val, mon, data_val, eval_points_val, shadow = tape
    ΔW = _extract_dret_with_shadow(dret, shadow)

    TD = eltype(first(data_val))
    N_eval = length(eval_points_val)
    k = cache.k

    grad_Lφ_x, grad_Lφ_xi = _get_grad_funcs(Laplacian, basis_val, op_cached)
    Δε_acc = Ref(zero(TD))

    for eval_idx in 1:N_eval
        neighbors = adjl_val[eval_idx]
        eval_point = eval_points_val[eval_idx]
        stencil_cache = cache.stencil_caches[eval_idx]

        Δw = extract_stencil_cotangent_enzyme(ΔW, eval_idx, neighbors, k, cache.num_ops)

        if sum(abs, Δw) > 0
            local_data = [data_val[i] for i in neighbors]
            Δlocal_data = [zeros(TD, length(first(data_val))) for _ in 1:k]
            Δeval_pt = zeros(TD, length(eval_point))

            _call_backward_stencil_with_ε!(
                Laplacian, Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache, collect(1:k),
                eval_point, local_data, basis_val, mon, k, op_cached, grad_Lφ_x, grad_Lφ_xi
            )
        end
    end

    Δbasis = _make_enzyme_tangent(B, basis_val, Δε_acc[])
    return (nothing, nothing, nothing, nothing, Δbasis)
end

end # module
