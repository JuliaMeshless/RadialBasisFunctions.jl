#=
Shared AD utilities that depend on operator types (Partial, Laplacian).

Must be included AFTER operator definitions since it dispatches on Partial/Laplacian types.
Used by Enzyme and Mooncake extensions via import.
=#

"""
    _optype(ℒ)

Map operator instance to its abstract type for dispatch in AD rules.
"""
_optype(::Partial) = Partial
_optype(::Laplacian) = Laplacian

"""
    _get_grad_funcs(OpType, basis, ℒ)

Get gradient functions for the given operator type and basis.
Returns (grad_Lφ_x, grad_Lφ_xi) tuple.
"""
_get_grad_funcs(::Type{<:Partial}, basis, ℒ) = (
    grad_applied_partial_wrt_x(basis, ℒ.dim),
    grad_applied_partial_wrt_xi(basis, ℒ.dim),
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
    poly_backward! = (Δeval_point, Δb, k, nmon, num_ops) ->
    _backward_partial_polynomial_section!(Δeval_point, Δb, k, nmon, dim, num_ops)
    ∂Lφ_∂ε_fn = (x, xi) -> ∂Partial_φ_∂ε(basis, dim, x, xi)
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
    ) where {T, OpType}
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
                poly_backward! = poly_backward!,
                ∂Lφ_∂ε_fn = ∂Lφ_∂ε_fn,
            )

            for (local_idx, global_idx) in enumerate(neighbors)
                Δdata[global_idx] .+= Δlocal_data[local_idx]
            end
            Δeval[eval_idx] .+= Δeval_pt
        end
    end

    return nothing
end
