#=
Shared AD utilities that depend on operator types (Partial, MixedPartial, Laplacian).

Must be included AFTER operator definitions since it dispatches on Partial/Laplacian types.
Used by Enzyme and Mooncake extensions via import.
=#

"""
    _optype(ℒ)

Map operator instance to its abstract type for dispatch in AD rules.
"""
_optype(::Partial) = Partial
_optype(::MixedPartial) = MixedPartial
_optype(::Laplacian) = Laplacian

"""
    _get_grad_funcs(OpType, basis, ℒ)

Get gradient functions for the given operator type and basis.
Returns (grad_Lφ_x, grad_Lφ_xi) tuple.
"""
function _get_grad_funcs(::Type{<:Partial}, basis, ℒ)
    if ℒ.order == 1
        return (grad_applied_partial_wrt_x(basis, ℒ.dim), grad_applied_partial_wrt_xi(basis, ℒ.dim))
    else
        return (grad_applied_second_partial_wrt_x(basis, ℒ.dim), grad_applied_second_partial_wrt_xi(basis, ℒ.dim))
    end
end
_get_grad_funcs(::Type{<:MixedPartial}, basis, ℒ) = (
    grad_applied_mixed_partial_wrt_x(basis, ℒ.dim1, ℒ.dim2),
    grad_applied_mixed_partial_wrt_xi(basis, ℒ.dim1, ℒ.dim2),
)
_get_grad_funcs(::Type{<:Laplacian}, basis, ℒ) = (
    grad_applied_laplacian_wrt_x(basis),
    grad_applied_laplacian_wrt_xi(basis),
)

"""
    _get_rhs_closures(OpType, ℒ, basis)

Get operator-specific closures for `backward_stencil_with_ε!`.
Returns `(poly_backward!, ∂Lφ_∂ε_fn)` keyword arguments.

- Partial: polynomial section backward + partial ε derivative
- Laplacian: no polynomial backward + laplacian ε derivative
"""
function _get_rhs_closures(::Type{<:Partial}, ℒ, basis)
    dim = ℒ.dim
    if ℒ.order == 1
        poly_backward! = (Δeval_point, Δb, k, nmon, num_ops, eval_point) ->
            _backward_partial_polynomial_section!(Δeval_point, Δb, k, nmon, dim, num_ops, eval_point)
        ∂Lφ_∂ε_fn = (x, xi) -> ∂Partial_φ_∂ε(basis, dim, x, xi)
        return poly_backward!, ∂Lφ_∂ε_fn
    else
        poly_backward! = (Δeval_point, Δb, k, nmon, num_ops, eval_point) ->
            _backward_second_partial_polynomial_section!(Δeval_point, Δb, k, nmon, dim, num_ops)
        ∂Lφ_∂ε_fn = (x, xi) -> ∂SecondPartial_φ_∂ε(basis, dim, x, xi)
        return poly_backward!, ∂Lφ_∂ε_fn
    end
end

function _get_rhs_closures(::Type{<:MixedPartial}, ℒ, basis)
    d1, d2 = ℒ.dim1, ℒ.dim2
    poly_backward! = (Δeval_point, Δb, k, nmon, num_ops, eval_point) ->
        _backward_mixed_partial_polynomial_section!(Δeval_point, Δb, k, nmon, d1, d2, num_ops)
    ∂Lφ_∂ε_fn = (x, xi) -> ∂MixedPartial_φ_∂ε(basis, d1, d2, x, xi)
    return poly_backward!, ∂Lφ_∂ε_fn
end

function _get_rhs_closures(::Type{<:Laplacian}, ℒ, basis)
    ∂Lφ_∂ε_fn = (x, xi) -> ∂Laplacian_φ_∂ε(basis, x, xi)
    return nothing, ∂Lφ_∂ε_fn
end

"""
    build_weights_pullback_loop!(Δdata, Δeval, Δε_acc, ΔW_extractor, cache, adjl,
        eval_points, data, basis, mon, ℒ, OpType, grad_Lφ_x, grad_Lφ_xi)

Shared stencil iteration loop for _build_weights pullback across all AD backends.

`ΔW_extractor(eval_idx, neighbors, k)` is a callable that returns the stencil cotangent
matrix `Δw` given the eval index, neighbor list, and stencil size. This abstracts over
the different ways each AD framework stores cotangents (dense matrix, nzval vector, etc.).
"""
function build_weights_pullback_loop!(
    Δdata::Vector{Vector{T}},
    Δeval::Vector{Vector{T}},
    Δε_acc::Base.RefValue{T},
    ΔW_extractor,
    cache::WeightsBuildForwardCache,
    adjl::AbstractVector,
    eval_points::AbstractVector,
    data::AbstractVector,
    basis::AbstractRadialBasis,
    mon::MonomialBasis,
    ℒ,
    ::Type{OpType},
    grad_Lφ_x,
    grad_Lφ_xi,
) where {T,OpType}
    N_eval = length(eval_points)
    k = cache.k
    dim_space = length(first(data))

    poly_backward!, ∂Lφ_∂ε_fn = _get_rhs_closures(OpType, ℒ, basis)

    for eval_idx in 1:N_eval
        neighbors = adjl[eval_idx]
        eval_point = eval_points[eval_idx]
        stencil_cache = cache.stencil_caches[eval_idx]

        Δw = ΔW_extractor(eval_idx, neighbors, k)

        if sum(abs, Δw) > 0
            local_data = [data[i] for i in neighbors]
            Δlocal_data = [zeros(T, dim_space) for _ in 1:k]
            Δeval_pt = zeros(T, dim_space)

            backward_stencil_with_ε!(
                Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache, collect(1:k),
                eval_point, local_data, basis, mon, k,
                grad_Lφ_x, grad_Lφ_xi;
                (poly_backward!)=poly_backward!,
                ∂Lφ_∂ε_fn=∂Lφ_∂ε_fn,
            )

            for (local_idx, global_idx) in enumerate(neighbors)
                Δdata[global_idx] .+= Δlocal_data[local_idx]
            end
            Δeval[eval_idx] .+= Δeval_pt
        end
    end

    return nothing
end

# ============================================================================
# Cached forward/backward API — no Mooncake required
# ============================================================================

"""
    _build_weights_and_cache(ℒ, data, eval_points, adjl, basis)

Like `_build_weights(ℒ, data, eval_points, adjl, basis)` but returns
`(W, cache)` where `cache` is a `WeightsBuildForwardCache` for use with
`_pullback_weights!`.

Assumes CPU execution, no boundary conditions.
"""
function _build_weights_and_cache(ℒ, data, eval_points, adjl, basis)
    dim = length(first(data))
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)
    OpType = _optype(ℒ)
    return _forward_with_cache(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, OpType)
end

"""
    _pullback_weights!(
        Δdata, Δeval_points, ΔW_nzval, W, cache, data, eval_points, adjl, basis, ℒ,
    )

Propagate gradient from weight-matrix nonzeros back to point coordinates.
Computes `∂L/∂data` and `∂L/∂eval_points` given `∂L/∂W`.

- `Δdata`, `Δeval_points`: output buffers (`Vector{Vector{Float64}}`), populated in-place
  (caller must zero them first or allocate fresh).
- `ΔW_nzval`: gradient w.r.t. each nonzero entry of `W`.
- `W`, `cache`: from a prior `_build_weights_and_cache` call on the same operator.
- `ℒ`: operator (e.g. `Partial(2, 1)`).

Does **not** use Mooncake — pure Julia linear algebra.
"""
function _pullback_weights!(
    Δdata::Vector{Vector{T}},
    Δeval_points::Vector{Vector{T}},
    ΔW_nzval::Vector{Float64},
    W::SparseMatrixCSC,
    cache::WeightsBuildForwardCache{T},
    data::AbstractVector,
    eval_points::AbstractVector,
    adjl::AbstractVector,
    basis::AbstractRadialBasis,
    ℒ,
) where {T}
    dim_space = length(first(data))
    mon = MonomialBasis(dim_space, basis.poly_deg)
    OpType = _optype(ℒ)
    grad_Lφ_x, grad_Lφ_xi = _get_grad_funcs(OpType, basis, ℒ)
    Δε_acc = Ref(zero(T))

    ΔW_extractor = (eval_idx, neighbors, k) ->
        extract_stencil_cotangent_from_nzval(ΔW_nzval, W, eval_idx, neighbors, k)

    build_weights_pullback_loop!(
        Δdata, Δeval_points, Δε_acc, ΔW_extractor,
        cache, adjl, eval_points, data, basis, mon, ℒ, OpType,
        grad_Lφ_x, grad_Lφ_xi,
    )
    return nothing
end
