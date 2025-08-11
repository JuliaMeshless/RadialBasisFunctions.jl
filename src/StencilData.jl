struct StencilData{T}
    local_adjl::Vector{Int}                     # Local adjacency list indices
    eval_point::AbstractVector{T}                # Evaluation point for the RHS
    A::AbstractMatrix{T}            # Local system matrix
    b::AbstractMatrix{T}                          # Local RHS matrix (one column per operator)
    d::Vector{AbstractVector{T}}                  # Local data points (now using the same type T)
    is_boundary::AbstractVector{Bool}             # Whether nodes are on boundary
    is_Neumann::AbstractVector{Bool}              # Whether nodes have Neumann boundary conditions
    normal::Vector{AbstractVector{T}}             # Normal vectors (meaningful for Neumann points)
    lhs_v::AbstractMatrix{T}                      # Local coefficients for internal nodes
    rhs_v::AbstractMatrix{T}                      # Local coefficients for boundary nodes
    weights::AbstractMatrix{T}                    # Weights for the local system
    function StencilData(all_coords, adjl, functional_data)
        TD = eltype(first(all_coords))

        dim = length(first(all_coords))
        nmon = binomial(dim + functional_data.basis.poly_deg, functional_data.mon.poly_deg)
        k = length(first(adjl))
        n = k + nmon

        num_ops = _num_ops(functional_data.ℒrbf)

        local_adjl = zeros(eltype(first(adjl)[1]), k)
        eval_point = zeros(TD, dim)
        A = Symmetric(zeros(TD, n, n), :U)
        b = zeros(TD, n, num_ops)
        d = [zeros(TD, dim) for _ in 1:k]
        is_boundary = zeros(Bool, k)
        is_Neumann = zeros(Bool, k)
        normal = [zeros(TD, dim) for _ in 1:k]
        lhs_v = zeros(TD, k, num_ops)
        rhs_v = zeros(TD, k, num_ops)
        weights = zeros(TD, n, num_ops)

        return new(
            local_adjl,
            eval_point,
            A,
            b,
            d,
            is_boundary,
            is_Neumann,
            normal,
            lhs_v,
            rhs_v,
            weights,
        )
    end
end

function _set_stencil_eval_point!(
    region_data::RegionData, chunk_index::Int, point::AbstractVector{T}
) where {T}
    stencil_data = region_data.stencil_data[chunk_index]
    return stencil_data.eval_point .= point
end

function _update_stencil!(region_data::RegionData, global_index::Int, chunk_index::Int)
    stencil_data = region_data.stencil_data[chunk_index]
    stencil_data.local_adjl .= region_data.adjl[global_index]

    fill!(parent(stencil_data.A), 0)
    fill!(stencil_data.b, 0)
    fill!(stencil_data.lhs_v, 0)
    fill!(stencil_data.rhs_v, 0)

    for (idx, j) in enumerate(stencil_data.local_adjl)
        stencil_data.d[idx] = data[j]
    end

    #TODO: calculate j_boundary
    for (j_local, j_global) in enumerate(stencil_data.local_adjl)
        stencil_data.is_boundary[j_local] = boundary_flag[j_global]
        stencil_data.is_Neumann[j_local] = is_Neumann[j_global]
        if is_Neumann[j_global]
            copyto!(stencil_data.normal[j_local], convert.(T, normals[j_global]))
        else
            stencil_data.normal[j_local] .= zero(T)
        end
    end

    _build_collocation_matrix_Hermite!(stencil_data, region_data.functional_data)
    eval_point = stencil_data.eval_point
    _build_rhs!(stencil_data, region_data.functional_data, eval_point)

    fill!(stencil_data.weights, 0)
    stencil_data.weights .= bunchkaufman!(stencil_data.A) \ stencil_data.b

    # Store weights in appropriate matrices
    for j in 1:k
        if !stencil_data.is_boundary[j]
            stencil_data.lhs_v[j, :] .= view(stencil_data.weights, j, :)
        else
            stencil_data.rhs_v[j, :] .= view(stencil_data.weights, j, :)
        end
    end

    return nothing
end

function _build_collocation_matrix_Hermite!(stencil::StencilData, functional_data)
    basis = functional_data.basis
    mon = functional_data.mon
    k = length(stencil.local_adjl)
    A = parent(stencil.A)
    n = size(A, 2)
    @inbounds for j in 1:k, i in 1:j
        _calculate_matrix_entry_RBF!(i, j, stencil, basis)
    end

    if Deg > -1
        @inbounds for i in 1:k
            _calculate_matrix_entry_poly!(
                A, i, k + 1, n, stencil.d[i], stencil.is_Neumann[i], stencil.normal[i], mon
            )
        end
    end

    return nothing
end

function _calculate_matrix_entry_RBF!(i, j, stencil::StencilData, basis)
    A = parent(stencil.A)
    data = stencil.d
    is_Neumann_i = stencil.is_Neumann[i]
    is_Neumann_j = stencil.is_Neumann[j]
    if !is_Neumann_i && !is_Neumann_j
        A[i, j] = basis(data[i], data[j])
    elseif is_Neumann_i && !is_Neumann_j
        n = stencil.normal[i]
        A[i, j] = LinearAlgebra.dot(n, ∇(basis)(data[i], data[j]))
    elseif !is_Neumann_i && is_Neumann_j
        n = stencil.normal[j]
        A[i, j] = LinearAlgebra.dot(n, -∇(basis)(data[i], data[j]))
    elseif is_Neumann_i && is_Neumann_j
        ni = stencil.normal[i]
        nj = stencil.normal[j]
        A[i, j] = directional∂²(basis, ni, nj)(data[i], data[j])
    end
    return nothing
end

function _calculate_matrix_entry_poly!(
    A, row, col_start, col_end, data_point, is_Neumann, normal, mon
)
    a = view(A, row, col_start:col_end)
    if is_Neumann
        ∂_normal(mon, normal)(a, data_point)
    else
        mon(a, data_point)
    end

    return nothing
end

function _build_rhs!(stencil, functional_data, eval_point)
    ℒrbf = functional_data.ℒrbf
    ℒmon = functional_data.ℒmon
    basis = functional_data.basis
    k = length(stencil.local_adjl)
    b = stencil.b
    data = stencil.d

    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, and ℒmon must have the same length"

    # radial basis section
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in eachindex(data)
            if stencil.is_Neumann[i]
                b[i, j] = ℒ(eval_point, data[i], stencil.normal[i])
            else
                b[i, j] = ℒ(eval_point, data[i])
            end
        end
    end

    # monomial augmentation
    if basis.poly_deg > -1
        N = size(b, 1)
        for (j, ℒ) in enumerate(ℒmon)
            bmono = view(b, (k + 1):N, j)
            ℒ(bmono, eval_point)
        end
    end

    return nothing
end