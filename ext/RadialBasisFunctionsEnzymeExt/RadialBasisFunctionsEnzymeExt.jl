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
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::EnzymeCore.Const{<:$(esc(BasisType))},
                ::Type{<:EnzymeCore.Active},
                x::EnzymeCore.Duplicated,
                xi::EnzymeCore.Duplicated,
            )
            basis = func.val
            y = basis(x.val, xi.val)
            # Tape stores copies of x and xi for the reverse pass
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
            # d/dx[φ(x, xi)] = ∇φ, d/dxi[φ(x, xi)] = -∇φ
            x.dval .+= dret.val .* ∇φ
            xi.dval .-= dret.val .* ∇φ
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
        ::Type{<:EnzymeCore.Active},
        op::EnzymeCore.Const{<:RadialBasisOperator},
        x::EnzymeCore.Duplicated,
    )
    y = _eval_op(op.val, x.val)
    # Tape stores reference to op for the reverse pass
    return EnzymeRules.AugmentedReturn(y, nothing, op.val)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        dret::EnzymeCore.Active,
        tape,
        op::EnzymeCore.Const{<:RadialBasisOperator},
        x::EnzymeCore.Duplicated,
    )
    operator = tape
    # Pullback: Δx = W' * Δy
    x.dval .+= operator.weights' * dret.val
    return (nothing, nothing)
end

# Vector-valued operator: y[:,d] = W[d] * x
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        ::Type{<:EnzymeCore.Active},
        op::EnzymeCore.Const{<:RadialBasisOperator{<:VectorValuedOperator{D}}},
        x::EnzymeCore.Duplicated,
    ) where {D}
    y = _eval_op(op.val, x.val)
    return EnzymeRules.AugmentedReturn(y, nothing, (op.val, D))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_eval_op)},
        dret::EnzymeCore.Active,
        tape,
        op::EnzymeCore.Const{<:RadialBasisOperator{<:VectorValuedOperator}},
        x::EnzymeCore.Duplicated,
    )
    operator, D = tape
    # Pullback: Δx = Σ_d W[d]' * Δy[:,d]
    for d in 1:D
        x.dval .+= operator.weights[d]' * view(dret.val, :, d)
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
        ::Type{<:EnzymeCore.Active},
        x::EnzymeCore.Duplicated,
    )
    operator = op.val
    # Ensure weights are computed
    !RadialBasisFunctions.is_cache_valid(operator) && RadialBasisFunctions.update_weights!(operator)
    y = _eval_op(operator, x.val)
    return EnzymeRules.AugmentedReturn(y, nothing, operator)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator},
        dret::EnzymeCore.Active,
        tape,
        x::EnzymeCore.Duplicated,
    )
    operator = tape
    x.dval .+= operator.weights' * dret.val
    return (nothing,)
end

# Vector-valued operator call
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator{<:VectorValuedOperator{D}}},
        ::Type{<:EnzymeCore.Active},
        x::EnzymeCore.Duplicated,
    ) where {D}
    operator = op.val
    !RadialBasisFunctions.is_cache_valid(operator) && RadialBasisFunctions.update_weights!(operator)
    y = _eval_op(operator, x.val)
    return EnzymeRules.AugmentedReturn(y, nothing, (operator, D))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        op::EnzymeCore.Const{<:RadialBasisOperator{<:VectorValuedOperator}},
        dret::EnzymeCore.Active,
        tape,
        x::EnzymeCore.Duplicated,
    )
    operator, D = tape
    for d in 1:D
        x.dval .+= operator.weights[d]' * view(dret.val, :, d)
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
            xs.dval[i] .+= (interp.rbf_weights[j] * Δy) .* ∇φ
        end

        # Polynomial contribution
        if !isempty(interp.monomial_weights)
            dim = length(x_val)
            n_terms = length(interp.monomial_weights)
            ∇mon = ∇(interp.monomial_basis)
            ∇p = zeros(eltype(x_val), n_terms, dim)
            ∇mon(∇p, x_val)

            for k in eachindex(interp.monomial_weights)
                xs.dval[i] .+= (interp.monomial_weights[k] * Δy) .* view(∇p, k, :)
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

# Partial operator rule
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        ::Type{<:EnzymeCore.Active},
        ℒ_arg::EnzymeCore.Const{<:Partial},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    )
    ℒ = ℒ_arg.val
    data_val = data.val
    eval_points_val = eval_points.val
    adjl_val = adjl.val
    basis_val = basis.val

    # Build monomial basis and apply operator
    dim_space = length(first(data_val))
    mon = MonomialBasis(dim_space, basis_val.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis_val)

    # Forward pass with caching
    W, cache = _forward_with_cache(
        data_val, eval_points_val, adjl_val, basis_val, ℒrbf, ℒmon, mon, Partial
    )

    tape = (ℒ, cache, adjl_val, basis_val, mon, deepcopy(data_val), deepcopy(eval_points_val))
    return EnzymeRules.AugmentedReturn(W, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        dret::EnzymeCore.Active,
        tape,
        ℒ_arg::EnzymeCore.Const{<:Partial},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    )
    ℒ, cache, adjl_val, basis_val, mon, data_val, eval_points_val = tape
    ΔW = dret.val

    TD = eltype(first(data_val))
    N_data = length(data_val)
    N_eval = length(eval_points_val)
    k = cache.k

    # Get gradient functions
    grad_Lφ_x = grad_applied_partial_wrt_x(basis_val, ℒ.dim)
    grad_Lφ_xi = grad_applied_partial_wrt_xi(basis_val, ℒ.dim)

    # Process each stencil
    for eval_idx in 1:N_eval
        neighbors = adjl_val[eval_idx]
        eval_point = eval_points_val[eval_idx]
        stencil_cache = cache.stencil_caches[eval_idx]

        # Extract cotangent for this stencil
        Δw = extract_stencil_cotangent_enzyme(ΔW, eval_idx, neighbors, k, cache.num_ops)

        if sum(abs, Δw) > 0
            local_data = [data_val[i] for i in neighbors]

            Δlocal_data = [zeros(TD, length(first(data_val))) for _ in 1:k]
            Δeval_pt = zeros(TD, length(eval_point))

            backward_stencil_partial!(
                Δlocal_data,
                Δeval_pt,
                Δw,
                stencil_cache,
                collect(1:k),
                eval_point,
                local_data,
                basis_val,
                mon,
                k,
                ℒ.dim,
                grad_Lφ_x,
                grad_Lφ_xi,
            )

            # Accumulate gradients
            for (local_idx, global_idx) in enumerate(neighbors)
                data.dval[global_idx] .+= Δlocal_data[local_idx]
            end
            eval_points.dval[eval_idx] .+= Δeval_pt
        end
    end

    return (nothing, nothing, nothing, nothing, nothing)
end

# Laplacian operator rule
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        ::Type{<:EnzymeCore.Active},
        ℒ_arg::EnzymeCore.Const{<:Laplacian},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    )
    ℒ = ℒ_arg.val
    data_val = data.val
    eval_points_val = eval_points.val
    adjl_val = adjl.val
    basis_val = basis.val

    dim_space = length(first(data_val))
    mon = MonomialBasis(dim_space, basis_val.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis_val)

    W, cache = _forward_with_cache(
        data_val, eval_points_val, adjl_val, basis_val, ℒrbf, ℒmon, mon, Laplacian
    )

    tape = (ℒ, cache, adjl_val, basis_val, mon, deepcopy(data_val), deepcopy(eval_points_val))
    return EnzymeRules.AugmentedReturn(W, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_build_weights)},
        dret::EnzymeCore.Active,
        tape,
        ℒ_arg::EnzymeCore.Const{<:Laplacian},
        data::EnzymeCore.Duplicated,
        eval_points::EnzymeCore.Duplicated,
        adjl::EnzymeCore.Const,
        basis::EnzymeCore.Const{<:AbstractRadialBasis},
    )
    ℒ, cache, adjl_val, basis_val, mon, data_val, eval_points_val = tape
    ΔW = dret.val

    TD = eltype(first(data_val))
    N_data = length(data_val)
    N_eval = length(eval_points_val)
    k = cache.k

    grad_Lφ_x = grad_applied_laplacian_wrt_x(basis_val)
    grad_Lφ_xi = grad_applied_laplacian_wrt_xi(basis_val)

    for eval_idx in 1:N_eval
        neighbors = adjl_val[eval_idx]
        eval_point = eval_points_val[eval_idx]
        stencil_cache = cache.stencil_caches[eval_idx]

        Δw = extract_stencil_cotangent_enzyme(ΔW, eval_idx, neighbors, k, cache.num_ops)

        if sum(abs, Δw) > 0
            local_data = [data_val[i] for i in neighbors]

            Δlocal_data = [zeros(TD, length(first(data_val))) for _ in 1:k]
            Δeval_pt = zeros(TD, length(eval_point))

            backward_stencil_laplacian!(
                Δlocal_data,
                Δeval_pt,
                Δw,
                stencil_cache,
                collect(1:k),
                eval_point,
                local_data,
                basis_val,
                mon,
                k,
                grad_Lφ_x,
                grad_Lφ_xi,
            )

            for (local_idx, global_idx) in enumerate(neighbors)
                data.dval[global_idx] .+= Δlocal_data[local_idx]
            end
            eval_points.dval[eval_idx] .+= Δeval_pt
        end
    end

    return (nothing, nothing, nothing, nothing, nothing)
end

end # module
