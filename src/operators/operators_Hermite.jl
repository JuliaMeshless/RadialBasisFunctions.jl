abstract type AbstractOperator end
abstract type ScalarValuedOperator <: AbstractOperator end
abstract type VectorValuedOperator{Dim} <: AbstractOperator end

struct OperatorMatrices{SM}
    lhs::SM
    rhs::SM
    function OperatorMatrices(lhs::SM, rhs::SM) where {SM}
        return new{SM}(lhs, rhs)
    end
end

function (op::OperatorMatrices)(u::AbstractVector, bc_vals::AbstractVector)
    # Case 1: Single operator (op.lhs is a matrix)
    if op.lhs isa AbstractMatrix
        return op.lhs * u - op.rhs * bc_vals
        # Case 2: Multiple operators (op.lhs is a tuple of matrices)
    else
        # Create a tuple of results, one for each operator
        return ntuple(i -> op.lhs[i] * u - op.rhs[i] * bc_vals, length(op.lhs))
    end
end

struct HermiteRadialBasisOperator{}
    ℒ::L
    op_mat::OM
    data::D
    normals::D
    boundary_flag::AbstractVector{Bool}
    is_Neumann::AbstractVector{Bool}
    adjl::A
    basis::B
    valid_cache::Base.RefValue{Bool}
    function HermiteRadialBasisOperator(
        ℒ::L,
        op_mat::OM,
        data::D,
        normals::D,
        boundary_flag::AbstractVector{Bool},
        is_Neumann::AbstractVector{Bool},
        adjl::A,
        basis::B,
        cache_status::Bool=false,
    ) where {L,OM,D,A,B<:AbstractRadialBasis}
        return new{L,OM,D,A,B}(
            ℒ,
            op_mat,
            data,
            normals,
            boundary_flag,
            is_Neumann,
            adjl,
            basis,
            Ref(cache_status),
        )
    end
end

function HermiteRadialBasisOperator(
    ℒ,
    data::AbstractVector,
    normals::AbstractVector,
    boundary_flag::AbstractVector{Bool},
    is_Neumann::AbstractVector{Bool},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {T<:Int,B<:AbstractRadialBasis}
    dim = length(first(data))  # Missing dimension calculation
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)
    lhs, rhs = _build_weights(
        data, normals, boundary_flag, is_Neumann, adjl, basis, ℒrbf, ℒmon, mon
    )
    op_mat = OperatorMatrices(lhs, rhs)  # Wrap the matrices in OperatorMatrices
    return HermiteRadialBasisOperator(
        ℒ, op_mat, data, normals, boundary_flag, is_Neumann, adjl, basis, true
    )
end