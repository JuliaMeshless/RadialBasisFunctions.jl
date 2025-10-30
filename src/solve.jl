function _build_weights(ℒ, op)
    data = op.data
    eval_points = op.eval_points
    adjl = op.adjl
    basis = op.basis
    return _build_weights(ℒ, data, eval_points, adjl, basis)
end

function _build_weights(ℒ, data, eval_points, adjl, basis)
    dim = length(first(data)) # dimension of data

    # build monomial basis and operator
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    return _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon)
end

function _build_weights(
    data, eval_points, adjl, basis, ℒrbf, ℒmon, mon; batch_size=10, device=CPU()
)
    # Use the unified kernel infrastructure with standard allocation strategy
    return _build_weights_unified(
        StandardAllocation(),
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        nothing;
        batch_size=batch_size,
        device=device,
    )
end

function _build_stencil!(
    A::Symmetric,
    b,
    ℒrbf,
    ℒmon,
    data::AbstractVector{TD},
    eval_point::TE,
    basis::B,
    mon::MonomialBasis,
    k::Int,
) where {TD,TE,B<:AbstractRadialBasis}
    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)
    return (A \ b)[1:k, :]
end

function _build_collocation_matrix!(
    A::Symmetric, data::AbstractVector, basis::B, mon::MonomialBasis{Dim,Deg}, k::K
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg}
    # radial basis section
    AA = parent(A)
    N = size(A, 2)
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = basis(data[i], data[j])
    end

    # monomial augmentation
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            mon(a, data[i])
        end
    end

    return nothing
end

# Unified RHS building core - handles both single and tuple operators
function _build_rhs_core!(
    b, ℒrbf, ℒmon, data::AbstractVector, eval_point, basis, k, num_ops,
    get_rbf_op, get_mon_op, set_rbf_value!, get_mono_view
)
    # RBF section
    for op_idx in 1:num_ops
        ℒ = get_rbf_op(ℒrbf, op_idx)
        @inbounds for i in eachindex(data)
            set_rbf_value!(b, i, op_idx, ℒ(eval_point, data[i]))
        end
    end

    # Monomial augmentation
    if basis.poly_deg > -1
        for op_idx in 1:num_ops
            ℒ = get_mon_op(ℒmon, op_idx)
            bmono = get_mono_view(b, k, op_idx)
            ℒ(bmono, eval_point)
        end
    end

    return nothing
end

# Single operator version - delegates to unified core
function _build_rhs!(
    b::AbstractVector, ℒrbf, ℒmon, data::AbstractVector{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    _build_rhs_core!(
        b, ℒrbf, ℒmon, data, eval_point, basis, k, 1,
        (ℒ, _) -> ℒ,  # get_rbf_op: always return the single operator
        (ℒ, _) -> ℒ,  # get_mon_op: always return the single operator
        (b, i, _, val) -> (b[i] = val),  # set_rbf_value!: 1D indexing
        (b, k, _) -> view(b, (k + 1):length(b))  # get_mono_view: 1D slice
    )
end

# Tuple operator version - delegates to unified core
function _build_rhs!(
    b::AbstractMatrix, ℒrbf::Tuple, ℒmon::Tuple, data::AbstractVector{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, ℒmon must have the same length"
    _build_rhs_core!(
        b, ℒrbf, ℒmon, data, eval_point, basis, k, length(ℒrbf),
        (ℒ, j) -> ℒ[j],  # get_rbf_op: index into tuple
        (ℒ, j) -> ℒ[j],  # get_mon_op: index into tuple
        (b, i, j, val) -> (b[i, j] = val),  # set_rbf_value!: 2D indexing
        (b, k, j) -> view(b, (k + 1):size(b, 1), j)  # get_mono_view: 2D slice
    )
end

# Compatibility method: Matrix with single operators (stores result in first column)
function _build_rhs!(
    b::AbstractMatrix, ℒrbf, ℒmon, data::AbstractVector{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    # Treat as single operator writing to first column of matrix
    _build_rhs_core!(
        b, ℒrbf, ℒmon, data, eval_point, basis, k, 1,
        (ℒ, _) -> ℒ,  # get_rbf_op: always return the single operator
        (ℒ, _) -> ℒ,  # get_mon_op: always return the single operator
        (b, i, _, val) -> (b[i, 1] = val),  # set_rbf_value!: write to first column
        (b, k, _) -> view(b, (k + 1):size(b, 1), 1)  # get_mono_view: view first column
    )
end
