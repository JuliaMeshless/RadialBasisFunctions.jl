struct StencilData{T}
    local_adjl::Vector{Int}                     # Local adjacency list indices
    eval_point::AbstractVector{T}                # Evaluation point for the RHS
    A::Symmetric{T}                              # Local system matrix (will be treated as symmetric)
    b::AbstractMatrix{T}                          # Local RHS matrix (one column per operator)
    local_coords::Vector{AbstractVector{T}}                  # Local data points (now using the same type T)
    is_boundary::AbstractVector{Bool}             # Whether nodes are on boundary
    boundary_types::AbstractVector{BoundaryType}              # Boundary type (Dirichlet / Neumann / Robin)
    normals::Vector{AbstractVector{T}}             # Normal vectors (meaningful for Neumann points)
    lhs_v::AbstractMatrix{T}                      # Local coefficients for internal nodes
    rhs_v::AbstractMatrix{T}                      # Local coefficients for boundary nodes
    weights::AbstractMatrix{T}                    # Weights for the local system
    function StencilData(region_data::RegionData)
        all_coords = region_data.all_coords
        adjl = region_data.adjl
        functional_data = region_data.functional_data
        TD = eltype(first(all_coords))
        dim = length(first(all_coords))
        nmon = binomial(
            dim + functional_data.basis.poly_deg, functional_data.basis.poly_deg
        )
        k = length(first(adjl))
        n = k + nmon

        num_ops = _num_ops(functional_data.ℒrbf)

        # adjacency entries are Int-like; take eltype of inner adjacency vector
        local_adjl = zeros(eltype(first(adjl)), k)
        # println("local_adjl = $local_adjl")
        eval_point = zeros(TD, dim)
        A = Symmetric(zeros(TD, n, n), :U)  # Regular matrix, will use Symmetric wrapper when solving
        b = zeros(TD, n, num_ops)
        local_coords = [zeros(TD, dim) for _ in 1:k]
        is_boundary = zeros(Bool, k)
        boundary_types = _init_boundary_types(TD, k)
        normals = [zeros(TD, dim) for _ in 1:k]
        lhs_v = zeros(TD, k, num_ops)
        rhs_v = zeros(TD, k, num_ops)
        weights = zeros(TD, n, num_ops)

        return new{TD}(
            local_adjl,
            eval_point,
            A,
            b,
            local_coords,
            is_boundary,
            boundary_types,
            normals,
            lhs_v,
            rhs_v,
            weights,
        )
    end
end

function _set_stencil_eval_point!(
    stencil_data::StencilData, point::AbstractVector{T}
) where {T}
    stencil_data.eval_point .= point
    return nothing
end

function _reset_local_arrays!(stencil_data::StencilData)
    fill!(parent(stencil_data.A), 0)
    fill!(stencil_data.b, 0)
    fill!(stencil_data.lhs_v, 0)
    fill!(stencil_data.rhs_v, 0)
    fill!(stencil_data.weights, 0)
    _set_to_zero!(stencil_data.boundary_types)
    for n in stencil_data.normals
        fill!(n, 0)
    end
    return nothing
end

function _set_stencil_local_coords!(stencil_data::StencilData, region_data::RegionData)
    @inbounds for i in eachindex(stencil_data.local_adjl)
        stencil_data.local_coords[i] = region_data.all_coords[stencil_data.local_adjl[i]]
    end
    return nothing
end

function _set_stencil_boundary_data!(stencil_data::StencilData, region_data::RegionData)
    @inbounds for i in eachindex(stencil_data.local_adjl)
        global_index = stencil_data.local_adjl[i]
        boundary_index = region_data.global_to_boundary[global_index]
        stencil_data.is_boundary[i] = region_data.is_boundary[global_index]
        if stencil_data.is_boundary[i]
            stencil_data.boundary_types[i].coefficients .=
                region_data.boundary_types[boundary_index].coefficients
            if is_Neumann(stencil_data.boundary_types[i]) ||
                is_Robin(stencil_data.boundary_types[i])
                stencil_data.normals[i] .= region_data.normals[boundary_index]
            end
        end
    end
    return nothing
end

function _update_stencil!(
    stencil_data::StencilData, region_data::RegionData, global_index::Int
)
    _reset_local_arrays!(stencil_data)

    stencil_data.local_adjl .= region_data.adjl[global_index]

    _set_stencil_local_coords!(stencil_data, region_data)

    _set_stencil_boundary_data!(stencil_data, region_data)

    _build_collocation_matrix_Hermite!(stencil_data, region_data.functional_data)
    _build_rhs!(stencil_data, region_data.functional_data)

    try
        F = bunchkaufman(stencil_data.A)
        stencil_data.weights .= F \ stencil_data.b
    catch err
        if err isa LinearAlgebra.SingularException
            Aparent = parent(stencil_data.A)  # Get underlying matrix from Symmetric wrapper
            n = size(Aparent, 1)
            k = length(stencil_data.local_adjl)
            nmon = n - k
            r = rank(Aparent)  # Check rank of underlying matrix
            println(
                "Singular local stencil_data matrix (rank=" *
                string(r) *
                " < n=" *
                string(n) *
                "). " *
                "Stencil size k=" *
                string(k) *
                ", polynomial terms nmon=" *
                string(nmon) *
                ".",
            )
        end
    end

    # Store weights in appropriate matrices
    # The first k entries of weights correspond to the k local nodes
    k = length(stencil_data.local_adjl)
    num_ops = size(stencil_data.weights, 2)

    for j in 1:k
        for op in 1:num_ops
            if !stencil_data.is_boundary[j]
                stencil_data.lhs_v[j, op] = stencil_data.weights[j, op]
            else
                stencil_data.rhs_v[j, op] = stencil_data.weights[j, op]
            end
        end
    end

    return nothing
end

function _build_collocation_matrix_Hermite!(stencil_data::StencilData, functional_data)
    basis = functional_data.basis
    mon = functional_data.mon
    k = length(stencil_data.local_adjl)
    # Only fill upper triangle for symetric matrix
    @inbounds for j in 1:k, i in 1:j  # Changed to j >= i for upper triangle
        _calculate_matrix_entry_RBF!(i, j, stencil_data, basis)
    end

    if degree(mon) > -1
        @inbounds for i in 1:k
            _calculate_matrix_entry_poly!(stencil_data, i, k + 1, mon)
        end
    end

    return nothing
end

function _calculate_matrix_entry_RBF!(i, j, stencil_data::StencilData, basis)
    A = parent(stencil_data.A)
    data = stencil_data.local_coords

    bt_i = stencil_data.boundary_types[i]
    bt_j = stencil_data.boundary_types[j]
    is_int_i = !stencil_data.is_boundary[i]
    is_int_j = !stencil_data.is_boundary[j]
    is_dir_i = is_Dirichlet(bt_i)
    is_dir_j = is_Dirichlet(bt_j)
    is_nr_i = !is_int_i && !is_dir_i # boundary and not Dirichlet => Neumann or Robin
    is_nr_j = !is_int_j && !is_dir_j

    φ = basis(data[i], data[j])

    if (is_int_i || is_dir_i) && (is_int_j || is_dir_j)
        A[i, j] = φ
        return nothing
    end

    g = ∇(basis)(data[i], data[j])

    if is_nr_i && (is_int_j || is_dir_j)
        n_i = stencil_data.normals[i]
        A[i, j] = α(bt_i) * φ + β(bt_i) * LinearAlgebra.dot(n_i, g)
    elseif (is_int_i || is_dir_i) && is_nr_j
        n_j = stencil_data.normals[j]
        A[i, j] = α(bt_j) * φ + β(bt_j) * LinearAlgebra.dot(n_j, -g)
    elseif is_nr_i && is_nr_j
        n_i = stencil_data.normals[i]
        n_j = stencil_data.normals[j]
        # Mixed Robin/Neumann interaction (expand (α+β∂_n)_i (α+β∂_n)_j applied to kernel)
        # Terms: α_i α_j φ + α_i β_j (∂_{n_j} wrt j) + β_i α_j (∂_{n_i} wrt i) + β_i β_j (∂_{n_i}∂_{n_j})
        term_∂i = LinearAlgebra.dot(n_i, g)                  # ∂/∂n_i (first arg)
        term_∂j = LinearAlgebra.dot(n_j, -g)                 # ∂/∂n_j (second arg)
        term_∂i∂j = directional∂²(basis, n_i, n_j)(data[i], data[j])
        A[i, j] =
            α(bt_i) * α(bt_j) * φ +
            α(bt_i) * β(bt_j) * term_∂j +
            β(bt_i) * α(bt_j) * term_∂i +
            β(bt_i) * β(bt_j) * term_∂i∂j
    end
    return nothing
end

function _calculate_matrix_entry_poly!(stencil_data::StencilData, row, col_start, mon)
    A = parent(stencil_data.A)
    col_end = size(A, 2)
    data_point = stencil_data.local_coords[row]
    bt = stencil_data.boundary_types[row]

    # Internal or Dirichlet nodes: only polynomial values
    if !stencil_data.is_boundary[row] || is_Dirichlet(bt)
        a = view(A, row, col_start:col_end)
        mon(a, data_point)
        return nothing
    end

    nvec = stencil_data.normals[row]
    if is_Neumann(bt)
        a = view(A, row, col_start:col_end)
        ∂_normal(mon, nvec)(a, data_point)
        return nothing
    end

    # Robin: α * P + β * ∂_n P
    nmon = col_end - col_start + 1
    polyvals = zeros(eltype(A), nmon)
    derivvals = zeros(eltype(A), nmon)
    mon(polyvals, data_point)
    ∂_normal(mon, nvec)(derivvals, data_point)
    @inbounds for (k, c) in enumerate(col_start:col_end)
        A[row, c] = α(bt) * polyvals[k] + β(bt) * derivvals[k]
    end
    return nothing
end

function _build_rhs!(stencil_data::StencilData, functional_data)
    ℒrbf = functional_data.ℒrbf
    ℒmon = functional_data.ℒmon
    basis = functional_data.basis
    k = length(stencil_data.local_adjl)
    b = stencil_data.b
    data = stencil_data.local_coords
    eval_point = stencil_data.eval_point

    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, and ℒmon must have the same length"

    # radial basis section
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in eachindex(data)
            if stencil_data.is_boundary[i]
                bt = stencil_data.boundary_types[i]
                αv = α(bt)
                βv = β(bt)
                b[i, j] =
                    αv * ℒ(eval_point, data[i]) +
                    βv * ℒ(eval_point, data[i], stencil_data.normals[i])
            else # internal
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