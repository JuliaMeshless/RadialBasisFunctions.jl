"""
    RadialBasisFunctionsMooncakeExt

Package extension providing native Mooncake.jl rrule!! implementations for
RadialBasisFunctions.jl. Uses native rrule!! instead of @from_rrule bridge
to properly handle Vector{Vector/SVector} tangent types.

This extension requires both ChainRulesCore and Mooncake to be loaded.
"""
module RadialBasisFunctionsMooncakeExt

using RadialBasisFunctions
using ChainRulesCore
using Mooncake
using Mooncake: CoDual, NoFData, NoRData, primal, zero_fcodual
using StaticArrays: SVector
using SparseArrays: SparseMatrixCSC
using Combinatorics: binomial

# Import the _eval_op function we need to wrap
import RadialBasisFunctions: _eval_op, RadialBasisOperator, Interpolator
import RadialBasisFunctions: PHS, PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian
import RadialBasisFunctions: VectorValuedOperator
import RadialBasisFunctions: _build_weights, Partial, Laplacian, AbstractRadialBasis, MonomialBasis

# Import backward pass support from main package
import RadialBasisFunctions: StencilForwardCache, WeightsBuildForwardCache
import RadialBasisFunctions: backward_stencil_partial_with_ε!, backward_stencil_laplacian_with_ε!
import RadialBasisFunctions: _forward_with_cache
import RadialBasisFunctions: grad_applied_partial_wrt_x, grad_applied_partial_wrt_xi
import RadialBasisFunctions: grad_applied_laplacian_wrt_x, grad_applied_laplacian_wrt_xi

# =============================================================================
# Custom increment_and_get_rdata! for SVector types
# =============================================================================
# Mooncake represents SVector{N,T} tangents as Tangent{@NamedTuple{data::NTuple{N,T}}}
# ChainRulesCore rrules return Vector{SVector{N,T}} tangents
# This method bridges the gap by incrementing fdata in-place

# Generic version that handles any SVector dimension
function Mooncake.increment_and_get_rdata!(
        f::Vector{<:Mooncake.Tangent}, ::Mooncake.NoRData, t::Vector{SVector{N, T}}
    ) where {N, T}
    for i in eachindex(f, t)
        # Mooncake.Tangent has a `fields` field containing the NamedTuple
        # The NamedTuple has a `data` field with the tuple of values
        old_data = f[i].fields.data
        # Create new tuple with incremented values
        sv = t[i]
        new_data = ntuple(j -> old_data[j] + sv[j], Val(N))
        # Reconstruct the tangent with the same type
        f[i] = typeof(f[i])((data = new_data,))
    end
    return Mooncake.NoRData()
end

# =============================================================================
# increment_and_get_rdata! for Gaussian/IMQ Tangent types from ChainRulesCore
# =============================================================================
# When rrules return ChainRulesCore.Tangent{Gaussian/IMQ,...}, Mooncake needs to
# know how to accumulate these into its internal RData representation.

# Gaussian tangent: extract ε from ChainRulesCore.Tangent and add to RData
function Mooncake.increment_and_get_rdata!(
        ::Mooncake.NoFData,
        r::Mooncake.RData{@NamedTuple{ε::T, poly_deg::Mooncake.NoRData}},
        t::ChainRulesCore.Tangent{<:Gaussian},
    ) where {T}
    # Extract ε from ChainRulesCore tangent and add to Mooncake RData
    Δε = t.ε
    if !(Δε isa ChainRulesCore.NoTangent) && !(Δε isa ChainRulesCore.ZeroTangent)
        new_ε = r.data.ε + T(Δε)
        return Mooncake.RData{@NamedTuple{ε::T, poly_deg::Mooncake.NoRData}}((ε = new_ε, poly_deg = Mooncake.NoRData()))
    end
    return r
end

# IMQ tangent: same pattern as Gaussian
function Mooncake.increment_and_get_rdata!(
        ::Mooncake.NoFData,
        r::Mooncake.RData{@NamedTuple{ε::T, poly_deg::Mooncake.NoRData}},
        t::ChainRulesCore.Tangent{<:IMQ},
    ) where {T}
    Δε = t.ε
    if !(Δε isa ChainRulesCore.NoTangent) && !(Δε isa ChainRulesCore.ZeroTangent)
        new_ε = r.data.ε + T(Δε)
        return Mooncake.RData{@NamedTuple{ε::T, poly_deg::Mooncake.NoRData}}((ε = new_ε, poly_deg = Mooncake.NoRData()))
    end
    return r
end

# Import ChainRulesCore rules into Mooncake using @from_rrule
# The DefaultCtx is used for standard (non-debug) differentiation

# Operator evaluation rules - these are the most commonly differentiated
# Note: @from_rrule requires explicit signatures

# Scalar operator: _eval_op(op, x) -> vector
Mooncake.@from_rrule(
    Mooncake.DefaultCtx, Tuple{typeof(_eval_op), RadialBasisOperator, Vector{Float64}}
)

# Vector-valued operator (Gradient): _eval_op(op, x) -> matrix
# This covers gradient, jacobian operators
Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_eval_op), RadialBasisOperator{<:VectorValuedOperator}, Vector{Float64}}
)

# Operator call syntax: op(x) - bypasses cache check issues
Mooncake.@from_rrule(Mooncake.DefaultCtx, Tuple{RadialBasisOperator, Vector{Float64}})

Mooncake.@from_rrule(
    Mooncake.DefaultCtx, Tuple{RadialBasisOperator{<:VectorValuedOperator}, Vector{Float64}}
)

# Basis function rules for common types (Float64 vectors)
# These enable differentiating through weight computation if needed

Mooncake.@from_rrule(Mooncake.DefaultCtx, Tuple{PHS1, Vector{Float64}, Vector{Float64}})

Mooncake.@from_rrule(Mooncake.DefaultCtx, Tuple{PHS3, Vector{Float64}, Vector{Float64}})

Mooncake.@from_rrule(Mooncake.DefaultCtx, Tuple{PHS5, Vector{Float64}, Vector{Float64}})

Mooncake.@from_rrule(Mooncake.DefaultCtx, Tuple{PHS7, Vector{Float64}, Vector{Float64}})

Mooncake.@from_rrule(Mooncake.DefaultCtx, Tuple{IMQ, Vector{Float64}, Vector{Float64}})

Mooncake.@from_rrule(Mooncake.DefaultCtx, Tuple{Gaussian, Vector{Float64}, Vector{Float64}})

# Interpolator rules
Mooncake.@from_rrule(Mooncake.DefaultCtx, Tuple{Interpolator, Vector{Float64}})

# =============================================================================
# Native rrule!! for _build_weights
# =============================================================================
# Uses native rrule!! instead of @from_rrule to handle Vector{Vector/SVector} tangents

# Declare _build_weights as primitive for all signatures we support
# PHS types with Laplacian
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS1}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS3}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS5}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS7}

# PHS types with Partial
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), <:Partial, AbstractVector, AbstractVector, AbstractVector, PHS1}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), <:Partial, AbstractVector, AbstractVector, AbstractVector, PHS3}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), <:Partial, AbstractVector, AbstractVector, AbstractVector, PHS5}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), <:Partial, AbstractVector, AbstractVector, AbstractVector, PHS7}

# IMQ and Gaussian with Laplacian
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, IMQ}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, Gaussian}

# IMQ and Gaussian with Partial
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), <:Partial, AbstractVector, AbstractVector, AbstractVector, IMQ}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), <:Partial, AbstractVector, AbstractVector, AbstractVector, Gaussian}

"""
    extract_stencil_cotangent_mooncake(ΔW_nzval, W, eval_idx, neighbors, k)

Extract cotangent values for a single stencil from sparse matrix nzval gradient.
"""
function extract_stencil_cotangent_mooncake(
        ΔW_nzval::Vector{T}, W::SparseMatrixCSC, eval_idx::Int, neighbors::Vector{Int}, k::Int
    ) where {T}
    Δw = zeros(T, k, 1)
    for (local_idx, global_idx) in enumerate(neighbors)
        # Find the position in nzval for (eval_idx, global_idx)
        col_start = W.colptr[global_idx]
        col_end = W.colptr[global_idx + 1] - 1
        for pos in col_start:col_end
            if W.rowval[pos] == eval_idx
                Δw[local_idx, 1] = ΔW_nzval[pos]
                break
            end
        end
    end
    return Δw
end

# Generic implementation for Laplacian with any PHS basis
function Mooncake.rrule!!(
        ::CoDual{typeof(_build_weights)},
        op::CoDual{Laplacian},
        data::CoDual{<:AbstractVector},
        eval_points::CoDual{<:AbstractVector},
        adjl::CoDual{<:AbstractVector},
        basis::CoDual{<:Union{PHS1, PHS3, PHS5, PHS7}},
    )
    # Extract primals
    ℒ = primal(op)
    pts = primal(data)
    eval_pts = primal(eval_points)
    adj = primal(adjl)
    bas = primal(basis)

    # Build monomial basis and apply operator
    dim_space = length(first(pts))
    mon = MonomialBasis(dim_space, bas.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(bas)

    # Forward pass with caching
    W, cache = _forward_with_cache(pts, eval_pts, adj, bas, ℒrbf, ℒmon, mon, Laplacian)

    # Create output CoDual with FData (gradients accumulate into fdata.nzval)
    W_codual = zero_fcodual(W)
    W_fdata = W_codual.dx  # FData{...} with mutable nzval

    # Get gradient functions
    grad_Lφ_x = grad_applied_laplacian_wrt_x(bas)
    grad_Lφ_xi = grad_applied_laplacian_wrt_xi(bas)

    function _build_weights_laplacian_pb!!(ΔW_rdata)
        # For SparseMatrixCSC, gradients are in FData.nzval (mutable), not RData
        # The downstream operations accumulate into W_fdata.data.nzval
        ΔW_nzval = W_fdata.data.nzval

        TD = eltype(first(pts))
        N_data = length(pts)
        N_eval = length(eval_pts)
        k = cache.k

        # Get mutable fdata for accumulation
        Δdata_fdata = data.dx
        Δeval_fdata = eval_points.dx

        # Initialize local gradient accumulators
        Δdata_raw = [zeros(TD, dim_space) for _ in 1:N_data]
        Δeval_raw = [zeros(TD, dim_space) for _ in 1:N_eval]
        Δε_acc = Ref(zero(TD))

        # Process each stencil
        for eval_idx in 1:N_eval
            neighbors = adj[eval_idx]
            eval_point = eval_pts[eval_idx]
            stencil_cache = cache.stencil_caches[eval_idx]

            # Extract cotangent for this stencil
            Δw = extract_stencil_cotangent_mooncake(ΔW_nzval, W, eval_idx, neighbors, k)

            if sum(abs, Δw) > 0
                local_data = [pts[i] for i in neighbors]
                Δlocal_data = [zeros(TD, dim_space) for _ in 1:k]
                Δeval_pt = zeros(TD, dim_space)

                backward_stencil_laplacian_with_ε!(
                    Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache,
                    collect(1:k), eval_point, local_data, bas, mon, k,
                    grad_Lφ_x, grad_Lφ_xi
                )

                for (local_idx, global_idx) in enumerate(neighbors)
                    Δdata_raw[global_idx] .+= Δlocal_data[local_idx]
                end
                Δeval_raw[eval_idx] .+= Δeval_pt
            end
        end

        # Accumulate into fdata (in-place)
        for i in 1:N_data
            for d in 1:dim_space
                Δdata_fdata[i][d] += Δdata_raw[i][d]
            end
        end
        for i in 1:N_eval
            for d in 1:dim_space
                Δeval_fdata[i][d] += Δeval_raw[i][d]
            end
        end

        # Return rdata for each argument
        # For mutable Vector{Vector}, gradients are accumulated in fdata, rdata is NoRData
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return W_codual, _build_weights_laplacian_pb!!
end

# Generic implementation for Partial with any PHS basis
function Mooncake.rrule!!(
        ::CoDual{typeof(_build_weights)},
        op::CoDual{<:Partial},
        data::CoDual{<:AbstractVector},
        eval_points::CoDual{<:AbstractVector},
        adjl::CoDual{<:AbstractVector},
        basis::CoDual{<:Union{PHS1, PHS3, PHS5, PHS7}},
    )
    ℒ = primal(op)
    pts = primal(data)
    eval_pts = primal(eval_points)
    adj = primal(adjl)
    bas = primal(basis)

    dim_space = length(first(pts))
    mon = MonomialBasis(dim_space, bas.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(bas)

    W, cache = _forward_with_cache(pts, eval_pts, adj, bas, ℒrbf, ℒmon, mon, Partial)

    W_codual = zero_fcodual(W)
    W_fdata = W_codual.dx

    grad_Lφ_x = grad_applied_partial_wrt_x(bas, ℒ.dim)
    grad_Lφ_xi = grad_applied_partial_wrt_xi(bas, ℒ.dim)

    function _build_weights_partial_pb!!(ΔW_rdata)
        ΔW_nzval = W_fdata.data.nzval

        TD = eltype(first(pts))
        N_data = length(pts)
        N_eval = length(eval_pts)
        k = cache.k

        Δdata_fdata = data.dx
        Δeval_fdata = eval_points.dx

        Δdata_raw = [zeros(TD, dim_space) for _ in 1:N_data]
        Δeval_raw = [zeros(TD, dim_space) for _ in 1:N_eval]
        Δε_acc = Ref(zero(TD))

        for eval_idx in 1:N_eval
            neighbors = adj[eval_idx]
            eval_point = eval_pts[eval_idx]
            stencil_cache = cache.stencil_caches[eval_idx]

            Δw = extract_stencil_cotangent_mooncake(ΔW_nzval, W, eval_idx, neighbors, k)

            if sum(abs, Δw) > 0
                local_data = [pts[i] for i in neighbors]
                Δlocal_data = [zeros(TD, dim_space) for _ in 1:k]
                Δeval_pt = zeros(TD, dim_space)

                backward_stencil_partial_with_ε!(
                    Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache,
                    collect(1:k), eval_point, local_data, bas, mon, k, ℒ.dim,
                    grad_Lφ_x, grad_Lφ_xi
                )

                for (local_idx, global_idx) in enumerate(neighbors)
                    Δdata_raw[global_idx] .+= Δlocal_data[local_idx]
                end
                Δeval_raw[eval_idx] .+= Δeval_pt
            end
        end

        for i in 1:N_data
            for d in 1:dim_space
                Δdata_fdata[i][d] += Δdata_raw[i][d]
            end
        end
        for i in 1:N_eval
            for d in 1:dim_space
                Δeval_fdata[i][d] += Δeval_raw[i][d]
            end
        end

        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return W_codual, _build_weights_partial_pb!!
end

# Generic implementation for Laplacian with IMQ/Gaussian basis
function Mooncake.rrule!!(
        ::CoDual{typeof(_build_weights)},
        op::CoDual{Laplacian},
        data::CoDual{<:AbstractVector},
        eval_points::CoDual{<:AbstractVector},
        adjl::CoDual{<:AbstractVector},
        basis::CoDual{<:Union{IMQ, Gaussian}},
    )
    ℒ = primal(op)
    pts = primal(data)
    eval_pts = primal(eval_points)
    adj = primal(adjl)
    bas = primal(basis)

    dim_space = length(first(pts))
    mon = MonomialBasis(dim_space, bas.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(bas)

    W, cache = _forward_with_cache(pts, eval_pts, adj, bas, ℒrbf, ℒmon, mon, Laplacian)

    W_codual = zero_fcodual(W)
    W_fdata = W_codual.dx

    grad_Lφ_x = grad_applied_laplacian_wrt_x(bas)
    grad_Lφ_xi = grad_applied_laplacian_wrt_xi(bas)

    function _build_weights_laplacian_eps_pb!!(ΔW_rdata)
        ΔW_nzval = W_fdata.data.nzval

        TD = eltype(first(pts))
        N_data = length(pts)
        N_eval = length(eval_pts)
        k = cache.k

        Δdata_fdata = data.dx
        Δeval_fdata = eval_points.dx

        Δdata_raw = [zeros(TD, dim_space) for _ in 1:N_data]
        Δeval_raw = [zeros(TD, dim_space) for _ in 1:N_eval]
        Δε_acc = Ref(zero(TD))

        for eval_idx in 1:N_eval
            neighbors = adj[eval_idx]
            eval_point = eval_pts[eval_idx]
            stencil_cache = cache.stencil_caches[eval_idx]

            Δw = extract_stencil_cotangent_mooncake(ΔW_nzval, W, eval_idx, neighbors, k)

            if sum(abs, Δw) > 0
                local_data = [pts[i] for i in neighbors]
                Δlocal_data = [zeros(TD, dim_space) for _ in 1:k]
                Δeval_pt = zeros(TD, dim_space)

                backward_stencil_laplacian_with_ε!(
                    Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache,
                    collect(1:k), eval_point, local_data, bas, mon, k,
                    grad_Lφ_x, grad_Lφ_xi
                )

                for (local_idx, global_idx) in enumerate(neighbors)
                    Δdata_raw[global_idx] .+= Δlocal_data[local_idx]
                end
                Δeval_raw[eval_idx] .+= Δeval_pt
            end
        end

        for i in 1:N_data
            for d in 1:dim_space
                Δdata_fdata[i][d] += Δdata_raw[i][d]
            end
        end
        for i in 1:N_eval
            for d in 1:dim_space
                Δeval_fdata[i][d] += Δeval_raw[i][d]
            end
        end

        # For IMQ/Gaussian, accumulate ε gradient into basis rdata
        basis_rdata = Mooncake.RData{@NamedTuple{ε::TD, poly_deg::Mooncake.NoRData}}(
            (ε = Δε_acc[], poly_deg = NoRData())
        )

        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), basis_rdata
    end

    return W_codual, _build_weights_laplacian_eps_pb!!
end

# Generic implementation for Partial with IMQ/Gaussian basis
function Mooncake.rrule!!(
        ::CoDual{typeof(_build_weights)},
        op::CoDual{<:Partial},
        data::CoDual{<:AbstractVector},
        eval_points::CoDual{<:AbstractVector},
        adjl::CoDual{<:AbstractVector},
        basis::CoDual{<:Union{IMQ, Gaussian}},
    )
    ℒ = primal(op)
    pts = primal(data)
    eval_pts = primal(eval_points)
    adj = primal(adjl)
    bas = primal(basis)

    dim_space = length(first(pts))
    mon = MonomialBasis(dim_space, bas.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(bas)

    W, cache = _forward_with_cache(pts, eval_pts, adj, bas, ℒrbf, ℒmon, mon, Partial)

    W_codual = zero_fcodual(W)
    W_fdata = W_codual.dx

    grad_Lφ_x = grad_applied_partial_wrt_x(bas, ℒ.dim)
    grad_Lφ_xi = grad_applied_partial_wrt_xi(bas, ℒ.dim)

    function _build_weights_partial_eps_pb!!(ΔW_rdata)
        ΔW_nzval = W_fdata.data.nzval

        TD = eltype(first(pts))
        N_data = length(pts)
        N_eval = length(eval_pts)
        k = cache.k

        Δdata_fdata = data.dx
        Δeval_fdata = eval_points.dx

        Δdata_raw = [zeros(TD, dim_space) for _ in 1:N_data]
        Δeval_raw = [zeros(TD, dim_space) for _ in 1:N_eval]
        Δε_acc = Ref(zero(TD))

        for eval_idx in 1:N_eval
            neighbors = adj[eval_idx]
            eval_point = eval_pts[eval_idx]
            stencil_cache = cache.stencil_caches[eval_idx]

            Δw = extract_stencil_cotangent_mooncake(ΔW_nzval, W, eval_idx, neighbors, k)

            if sum(abs, Δw) > 0
                local_data = [pts[i] for i in neighbors]
                Δlocal_data = [zeros(TD, dim_space) for _ in 1:k]
                Δeval_pt = zeros(TD, dim_space)

                backward_stencil_partial_with_ε!(
                    Δlocal_data, Δeval_pt, Δε_acc, Δw, stencil_cache,
                    collect(1:k), eval_point, local_data, bas, mon, k, ℒ.dim,
                    grad_Lφ_x, grad_Lφ_xi
                )

                for (local_idx, global_idx) in enumerate(neighbors)
                    Δdata_raw[global_idx] .+= Δlocal_data[local_idx]
                end
                Δeval_raw[eval_idx] .+= Δeval_pt
            end
        end

        for i in 1:N_data
            for d in 1:dim_space
                Δdata_fdata[i][d] += Δdata_raw[i][d]
            end
        end
        for i in 1:N_eval
            for d in 1:dim_space
                Δeval_fdata[i][d] += Δeval_raw[i][d]
            end
        end

        basis_rdata = Mooncake.RData{@NamedTuple{ε::TD, poly_deg::Mooncake.NoRData}}(
            (ε = Δε_acc[], poly_deg = NoRData())
        )

        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), basis_rdata
    end

    return W_codual, _build_weights_partial_eps_pb!!
end

end # module
