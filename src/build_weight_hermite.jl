using LinearAlgebra: dot, Symmetric
using SparseArrays: sparse
using KernelAbstractions

# Import required types and functions from existing codebase
include("boundary_types.jl")
include("solve.jl")

"""
Count the actual number of non-zero elements for optimized sparse matrix allocation.

Returns:
- total_nnz: Total number of non-zero elements across all stencils
- nnz_per_row: Vector containing number of non-zeros for each evaluation point
- row_offsets: Cumulative offsets for filling I,J,V arrays
"""
function _count_nonzeros(
    adjl, is_boundary::Vector{Bool}, boundary_conditions::Vector{BoundaryCondition}
)
    N_eval = length(adjl)
    nnz_per_row = Vector{Int}(undef, N_eval)
    row_offsets = Vector{Int}(undef, N_eval + 1)

    total_nnz = 0
    row_offsets[1] = 1  # 1-based indexing for first row

    for eval_idx in 1:N_eval
        if is_boundary[eval_idx] && is_dirichlet(boundary_conditions[eval_idx])
            # Dirichlet: only diagonal element is non-zero
            nnz_per_row[eval_idx] = 1
        else
            # Interior, Neumann, or Robin: full stencil
            nnz_per_row[eval_idx] = length(adjl[eval_idx])
        end

        total_nnz += nnz_per_row[eval_idx]
        row_offsets[eval_idx + 1] = total_nnz + 1
    end

    return total_nnz, nnz_per_row, row_offsets
end

"""
Construct global_to_boundary index mapping using is_boundary global vector.
is_boundary is N_tot x 1 and global_to_boundary is N_tot x 1
this structure ensures we can use normals and boundary_conditions which are only defined on boundary points.
"""
function _construct_global_to_boundary(is_boundary::Vector{Bool})
    N_tot = length(is_boundary)
    global_to_boundary = Vector{Int}(undef, N_tot)

    boundary_counter = 0
    for i in 1:N_tot
        if is_boundary[i]
            boundary_counter += 1
            global_to_boundary[i] = boundary_counter
        else
            global_to_boundary[i] = 0  # Non-boundary points map to 0
        end
    end

    return global_to_boundary
end

"""
Main unified Hermite weight construction function with optimized two-pass approach.
"""
function _build_weights(
    data,
    eval_points,
    adjl,
    basis,
    ℒrbf,
    ℒmon,
    mon,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{BoundaryCondition},
    normals::Vector{Vector{T}},
    batch_size=10,
    device=CPU(),
) where {T}
    TD = eltype(first(data))
    dim = length(first(data))
    k = length(first(adjl))
    nmon = binomial(dim + basis.poly_deg, basis.poly_deg)
    num_ops = _num_ops(ℒrbf)

    # Pass 1: Count non-zero elements for exact allocation
    total_nnz, nnz_per_row, row_offsets = _count_nonzeros(
        adjl, is_boundary, boundary_conditions
    )

    # Allocate exact memory for sparse matrix
    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Matrix{TD}(undef, total_nnz, num_ops)

    # Pass 2: Pre-allocate boundary info structures (one per batch)
    N_eval = length(data)
    n_batches = ceil(Int, N_eval / batch_size)

    # Pre-allocate boundary info for each batch
    batch_hermite_datas = [HermiteStencilData{TD}(k, dim) for _ in 1:n_batches]
    global_to_boundary = _construct_global_to_boundary(is_boundary)

    @kernel function fill_sparse_arrays_kernel(
        I,
        J,
        V,
        data,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        is_boundary,
        boundary_conditions,
        normals,
        batch_hermite_datas,
        global_to_boundary,
        row_offsets,
        batch_size,
        N_eval,
        nmon,
    )
        batch_idx = @index(Global)
        batch_hermite_data = batch_hermite_datas[batch_idx]

        # Calculate range for this batch
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, N_eval)

        # Pre-allocate work arrays for this thread
        n = length(first(adjl)) + nmon  # max possible stencil size
        A = Symmetric(zeros(TD, n, n), :U)
        b = _prepare_b(ℒrbf, TD, n)

        for eval_idx in start_idx:end_idx
            start_pos = row_offsets[eval_idx]

            neighbors = adjl[eval_idx]
            stencil_type_result = stencil_type(
                is_boundary, boundary_conditions, eval_idx, neighbors, global_to_boundary
            )

            if stencil_type_result isa DirichletStencil
                I[start_pos] = eval_idx
                J[start_pos] = eval_idx
                V[start_pos, :] .= 1.0
            else
                if stencil_type_result isa StandardStencil
                    local_data = view(data, adjl[eval_idx])
                elseif stencil_type_result isa HermiteStencil
                    update_stencil_data!(
                        batch_hermite_data,
                        data,
                        adjl[eval_idx],
                        is_boundary,
                        boundary_conditions,
                        normals,
                        global_to_boundary,
                    )
                    local_data = batch_hermite_data
                end

                eval_point = data[eval_idx]
                weights = _build_stencil!(
                    A, b, ℒrbf, ℒmon, local_data, eval_point, basis, mon, k
                )
                for local_idx in 1:k
                    pos = start_pos + local_idx - 1
                    I[pos] = eval_idx
                    J[pos] = neighbors[local_idx]
                    V[pos, :] = weights[local_idx, :]
                end
            end
        end
    end

    # Launch kernel
    kernel = fill_sparse_arrays_kernel(device)
    kernel(
        I,
        J,
        V,
        data,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        is_boundary,
        boundary_conditions,
        normals,
        batch_hermite_datas,
        global_to_boundary,
        row_offsets,
        batch_size,
        N_eval,
        nmon;
        ndrange=n_batches,
        workgroupsize=1,
    )

    # Wait for completion
    KernelAbstractions.synchronize(device)

    # Create and return sparse matrix/matrices
    nrows = length(data)
    ncols = length(data)

    if num_ops == 1
        return sparse(I, J, V[:, 1], nrows, ncols)
    else
        return ntuple(i -> sparse(I, J, V[:, i], nrows, ncols), num_ops)
    end
end

"""
Multiple dispatch for _build_stencil! when local_data is HermiteStencilData (HermiteStencil).
Handles boundary conditions within the stencil using Hermite interpolation.
"""
function _build_stencil!(
    A::Symmetric, b, ℒrbf, ℒmon, data::HermiteStencilData, eval_point, basis, mon, k
)
    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)
    return (A \ b)[1:k, :]
end

"""
Build Hermite collocation matrix with boundary condition modifications.
"""
function _build_collocation_matrix!(
    A::Symmetric, hermite_data::HermiteStencilData, basis, mon, k
)
    AA = parent(A)
    N = size(A, 2)

    # Build RBF matrix entries with Hermite modifications
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = _hermite_rbf_entry(i, j, hermite_data, basis)
    end

    # Polynomial augmentation with boundary operator modifications
    if basis.poly_deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            _hermite_poly_entry!(a, i, hermite_data, mon)
        end
    end

    return nothing
end

"""
Compute single RBF matrix entry for Hermite interpolation.
"""
function _hermite_rbf_entry(i::Int, j::Int, hermite_data::HermiteStencilData, basis)
    xi, xj = hermite_data.data[i], hermite_data.data[j]
    bt_i = hermite_data.boundary_conditions[i]
    bt_j = hermite_data.boundary_conditions[j]
    is_int_i = !hermite_data.is_boundary[i]
    is_int_j = !hermite_data.is_boundary[j]
    is_dir_i = is_dirichlet(bt_i)
    is_dir_j = is_dirichlet(bt_j)
    is_nr_i = !is_int_i && !is_dir_i # boundary and not Dirichlet => Neumann or Robin
    is_nr_j = !is_int_j && !is_dir_j

    φ = basis(xi, xj)

    if (is_int_i || is_dir_i) && (is_int_j || is_dir_j)
        return φ
    end

    g = ∇(basis)(xi, xj)

    if is_nr_i && (is_int_j || is_dir_j)
        n_i = hermite_data.normals[i]
        return α(bt_i) * φ + β(bt_i) * dot(n_i, g)
    elseif (is_int_i || is_dir_i) && is_nr_j
        n_j = hermite_data.normals[j]
        return α(bt_j) * φ + β(bt_j) * dot(n_j, -g)
    elseif is_nr_i && is_nr_j
        n_i = hermite_data.normals[i]
        n_j = hermite_data.normals[j]
        # Mixed Robin/Neumann interaction
        term_∂i = dot(n_i, g)                  # ∂/∂n_i (first arg)
        term_∂j = dot(n_j, -g)                 # ∂/∂n_j (second arg)
        term_∂i∂j = directional∂²(basis, n_i, n_j)(xi, xj)
        return α(bt_i) * α(bt_j) * φ +
               α(bt_i) * β(bt_j) * term_∂j +
               β(bt_i) * α(bt_j) * term_∂i +
               β(bt_i) * β(bt_j) * term_∂i∂j
    end
end

"""
Compute polynomial entries for Hermite interpolation.
"""
function _hermite_poly_entry!(
    a::AbstractVector, i::Int, hermite_data::HermiteStencilData, mon
)
    xi = hermite_data.data[i]
    bt = hermite_data.boundary_conditions[i]

    # Internal or Dirichlet nodes: only polynomial values
    if !hermite_data.is_boundary[i] || is_dirichlet(bt)
        mon(a, xi)
        return nothing
    end

    nvec = hermite_data.normals[i]
    if is_neumann(bt)
        ∂_normal(mon, nvec)(a, xi)
        return nothing
    end

    # Robin: α * P + β * ∂_n P
    nmon = length(a)
    polyvals = zeros(eltype(a), nmon)
    derivvals = zeros(eltype(a), nmon)
    mon(polyvals, xi)
    ∂_normal(mon, nvec)(derivvals, xi)
    @inbounds for k in 1:nmon
        a[k] = α(bt) * polyvals[k] + β(bt) * derivvals[k]
    end
    return nothing
end

"""
Build RHS for Hermite stencil with boundary conditions.
"""
function _build_rhs!(b, ℒrbf, ℒmon, hermite_data::HermiteStencilData, eval_point, basis, k)
    # Handle multiple operators case
    num_ops = isa(ℒrbf, Tuple) ? length(ℒrbf) : 1
    ℒrbf_tuple = isa(ℒrbf, Tuple) ? ℒrbf : (ℒrbf,)
    ℒmon_tuple = isa(ℒmon, Tuple) ? ℒmon : (ℒmon,)

    # RBF section with Hermite modifications
    for j in 1:num_ops
        ℒ = ℒrbf_tuple[j]
        @inbounds for i in 1:k
            if hermite_data.is_boundary[i]
                bt = hermite_data.boundary_conditions[i]
                αv = α(bt)
                βv = β(bt)
                if num_ops == 1
                    b[i] = αv * ℒ(eval_point, hermite_data.data[i]) +
                           βv * ℒ(eval_point, hermite_data.data[i], hermite_data.normals[i])
                else
                    b[i, j] = αv * ℒ(eval_point, hermite_data.data[i]) +
                              βv * ℒ(eval_point, hermite_data.data[i], hermite_data.normals[i])
                end
            else # internal
                if num_ops == 1
                    b[i] = ℒ(eval_point, hermite_data.data[i])
                else
                    b[i, j] = ℒ(eval_point, hermite_data.data[i])
                end
            end
        end
    end

    # Monomial augmentation
    if basis.poly_deg > -1
        N = size(b, 1)
        for j in 1:num_ops
            ℒ = ℒmon_tuple[j]
            if num_ops == 1
                bmono = view(b, (k + 1):N)
            else
                bmono = view(b, (k + 1):N, j)
            end
            ℒ(bmono, eval_point)
        end
    end

    return nothing
end

"""
Helper functions
"""
function _num_ops(ℒrbf)
    return isa(ℒrbf, Tuple) ? length(ℒrbf) : 1
end

function _prepare_b(ℒrbf, TD, n)
    num_ops = _num_ops(ℒrbf)
    return num_ops == 1 ? zeros(TD, n) : zeros(TD, n, num_ops)
end