"""
Hermite interpolation scheme for handling boundary conditions in RBF stencils.

This module provides Hermite variants of the core stencil building functions
that properly handle Neumann and Robin boundary conditions by modifying
both the collocation matrix and RHS to maintain symmetry and non-singularity.
"""

using LinearAlgebra: dot
using SparseArrays: sparse
using KernelAbstractions
using KernelAbstractions: @kernel, @index, CPU

"""
Build collocation matrix for Hermite interpolation with boundary conditions.
This is the Hermite variant of _build_collocation_matrix! from solve.jl.

For boundary points with Neumann/Robin conditions, the basis functions are modified:
- Instead of Φ(·,xⱼ), we use B₂Φ(·,xⱼ) where B is the boundary operator
- This maintains matrix symmetry by applying the same operator to rows and columns
"""
function _build_collocation_matrix!(
    A::Symmetric, data::HermiteStencilData, basis::B, mon::MonomialBasis{Dim,Deg}, k::K
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg}
    AA = parent(A)
    N = size(A, 2)

    # Build RBF matrix entries with Hermite modifications
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = _hermite_rbf_entry(i, j, data, basis)
    end

    # Polynomial augmentation with boundary operator modifications
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            _hermite_poly_entry!(a, i, data, mon)
        end
    end

    return nothing
end

"""
Compute single RBF matrix entry for Hermite interpolation.
Handles all combinations of interior/boundary points with appropriate operators.
"""
function _hermite_rbf_entry(
    i::Int, j::Int, data::HermiteStencilData{T}, basis::B
) where {B<:AbstractRadialBasis,T}
    xi, xj = data.data[i], data.data[j]
    is_bound_i = data.is_boundary[i]
    is_bound_j = data.is_boundary[j]

    # Standard case: both interior points
    if !is_bound_i && !is_bound_j
        return basis(xi, xj)
    end

    # Get basis value and gradient
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)

    bc_i = data.boundary_conditions[i]
    bc_j = data.boundary_conditions[j]

    # Cases involving boundary points
    if is_bound_i && !is_bound_j
        # Boundary-Interior: Apply boundary operator to first argument
        ni = data.normals[i]
        if is_dirichlet(bc_i)
            return φ
        else
            # Neumann/Robin: α*φ + β*∂ₙφ
            return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
        end

    elseif !is_bound_i && is_bound_j
        # Interior-Boundary: Apply boundary operator to second argument
        nj = data.normals[j]
        if is_dirichlet(bc_j)
            return φ
        else
            # Neumann/Robin: α*φ + β*∂ₙφ (note sign flip for gradient w.r.t. second arg)
            return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
        end

    else # is_bound_i && is_bound_j
        # Boundary-Boundary: Apply boundary operators to both arguments
        ni = data.normals[i]
        nj = data.normals[j]

        if is_dirichlet(bc_i) && is_dirichlet(bc_j)
            return φ
        elseif is_dirichlet(bc_i) && !is_dirichlet(bc_j)
            return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
        elseif !is_dirichlet(bc_i) && is_dirichlet(bc_j)
            return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
        else
            # Both Neumann/Robin: mixed derivative term
            ∂i∂j_φ = directional∂²(basis, ni, nj)(xi, xj)
            return (
                α(bc_i) * α(bc_j) * φ +
                α(bc_i) * β(bc_j) * dot(nj, -∇φ) +
                β(bc_i) * α(bc_j) * dot(ni, ∇φ) +
                β(bc_i) * β(bc_j) * ∂i∂j_φ
            )
        end
    end
end

"""
Compute polynomial entries for Hermite interpolation.
Applies boundary operators to polynomial basis functions at boundary points.
"""
function _hermite_poly_entry!(
    a::AbstractVector, i::Int, data::HermiteStencilData, mon::MonomialBasis
)
    xi = data.data[i]
    is_bound_i = data.is_boundary[i]

    if !is_bound_i
        # Interior point: standard polynomial evaluation
        mon(a, xi)
    else
        bc_i = data.boundary_conditions[i]
        if is_dirichlet(bc_i)
            # Dirichlet boundary: standard polynomial evaluation
            mon(a, xi)
        else
            # Neumann/Robin: α*P + β*∂ₙP
            ni = data.normals[i]
            nmon = length(a)

            # Evaluate polynomial and its normal derivative
            T = eltype(a)
            poly_vals = zeros(T, nmon)
            deriv_vals = zeros(T, nmon)

            mon(poly_vals, xi)
            ∂_normal(mon, ni)(deriv_vals, xi)

            # Apply boundary condition
            @inbounds for k in 1:nmon
                a[k] = α(bc_i) * poly_vals[k] + β(bc_i) * deriv_vals[k]
            end
        end
    end

    return nothing
end

"""
Build RHS for Hermite interpolation with boundary conditions.
This is the Hermite variant of _build_rhs! from solve.jl.

When the evaluation point (center of stencil) has Neumann/Robin conditions,
the differential operator must be modified according to the boundary operator.
"""
function _build_rhs!(
    b, ℒrbf, ℒmon, data::HermiteStencilData{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}

    # RBF section with Hermite modifications for stencil points
    @inbounds for i in 1:k
        if data.is_boundary[i]
            bc_i = data.boundary_conditions[i]
            if is_dirichlet(bc_i)
                b[i] = ℒrbf(eval_point, data.data[i])
            else
                # Neumann/Robin: Apply boundary operator to RBF operator
                ni = data.normals[i]
                b[i] = (
                    α(bc_i) * ℒrbf(eval_point, data.data[i]) +
                    β(bc_i) * ℒrbf(eval_point, data.data[i], ni)
                )
            end
        else
            # Interior point: standard evaluation
            b[i] = ℒrbf(eval_point, data.data[i])
        end
    end

    # Monomial augmentation
    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)
        ℒmon(bmono, eval_point)
    end

    return nothing
end

"""
Multi-operator version of Hermite RHS building.
"""
function _build_rhs!(
    b,
    ℒrbf::Tuple,
    ℒmon::Tuple,
    data::HermiteStencilData{TD},
    eval_point::TE,
    basis::B,
    k::K,
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, ℒmon must have the same length"

    # RBF section with Hermite modifications for stencil points
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in 1:k
            if data.is_boundary[i]
                bc_i = data.boundary_conditions[i]
                if is_dirichlet(bc_i)
                    b[i, j] = ℒ(eval_point, data.data[i])
                else
                    # Neumann/Robin: Apply boundary operator to RBF operator
                    ni = data.normals[i]
                    b[i, j] = (
                        α(bc_i) * ℒ(eval_point, data.data[i]) +
                        β(bc_i) * ℒ(eval_point, data.data[i], ni)
                    )
                end
            else
                # Interior point: standard evaluation
                b[i, j] = ℒ(eval_point, data.data[i])
            end
        end
    end

    # Monomial augmentation
    if basis.poly_deg > -1
        N = size(b, 1)
        for (j, ℒ) in enumerate(ℒmon)
            bmono = view(b, (k + 1):N, j)
            ℒ(bmono, eval_point)
        end
    end

    return nothing
end

"""
Build complete Hermite stencil with boundary conditions.
This is the Hermite variant of _build_stencil! from solve.jl.
"""
function _build_stencil!(
    A::Symmetric,
    b,
    ℒrbf,
    ℒmon,
    data::HermiteStencilData{TD},
    eval_point::TE,
    basis::B,
    mon::MonomialBasis,
    k::Int,
) where {TD,TE,B<:AbstractRadialBasis}
    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)

    return (A \ b)[1:k, :]
end

"""
Standard stencil building - delegates to solve.jl implementation.
This method handles the case when local_data is a simple view of coordinate vectors.
"""
function _build_stencil!(
    A::Symmetric,
    b,
    ℒrbf,
    ℒmon,
    local_data::SubArray{<:AbstractVector},  # This is what view(data, neighbors) returns
    eval_point,
    basis::AbstractRadialBasis,
    mon::MonomialBasis,
    k::Int,
)
    # Use the standard solve.jl approach for interior stencils
    # Build standard collocation matrix
    AA = parent(A)
    n = k + length(mon)

    # RBF part
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = basis(local_data[i], local_data[j])
    end

    # Polynomial part
    if mon.poly_deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):n)
            mon(a, local_data[i])
        end
    end

    # Build RHS
    _build_rhs_standard!(b, ℒrbf, ℒmon, local_data, eval_point, mon)

    return (A \ b)[1:k, :]
end

"""
Standard RHS building for interior stencils.
"""
function _build_rhs_standard!(b, ℒrbf, ℒmon, local_data, eval_point, mon)
    k = length(local_data)

    # Handle single or multiple operators
    if isa(ℒrbf, Tuple)
        # Multiple operators
        for (j, ℒ) in enumerate(ℒrbf)
            @inbounds for i in 1:k
                b[i, j] = ℒ(eval_point, local_data[i])
            end
        end

        # Monomial augmentation
        if mon.poly_deg > -1
            N = size(b, 1)
            for (j, ℒ) in enumerate(ℒmon)
                bmono = view(b, (k + 1):N, j)
                ℒ(bmono, eval_point)
            end
        end
    else
        # Single operator
        @inbounds for i in 1:k
            b[i] = ℒrbf(eval_point, local_data[i])
        end

        # Monomial augmentation
        if mon.poly_deg > -1
            N = length(b)
            bmono = view(b, (k + 1):N)
            ℒmon(bmono, eval_point)
        end
    end

    return nothing
end

"""
Build weights for Hermite interpolation with sparse matrix construction.
This function follows the solve_hermite.jl philosophy by extending the core 
_build_weights function to handle boundary conditions on both stencil points 
AND evaluation points.

This is the Hermite variant of _build_weights from solve.jl, optimized for
sparse matrix construction with proper boundary condition handling.
"""
function _build_weights(
    data::Vector{<:AbstractVector},
    eval_points,
    adjl,
    basis::AbstractRadialBasis,
    ℒrbf,
    ℒmon,
    mon::MonomialBasis,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    batch_size::Int=10,
    device=CPU(),
)
    TD = eltype(first(data))
    dim = length(first(data))
    k = length(first(adjl))
    nmon = binomial(dim + basis.poly_deg, basis.poly_deg)
    num_ops = _num_ops(ℒrbf)

    # Pass 1: Count non-zero elements for exact allocation
    total_nnz, _, row_offsets = _count_nonzeros(adjl, is_boundary, boundary_conditions)

    # Allocate exact memory for sparse matrix
    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Matrix{TD}(undef, total_nnz, num_ops)

    # Pass 2: Pre-allocate boundary info structures (one per batch)
    N_eval = length(eval_points)
    n_batches = ceil(Int, N_eval / batch_size)

    # Pre-allocate Hermite stencil data for each batch
    batch_hermite_datas = [HermiteStencilData{TD}(k, dim) for _ in 1:n_batches]
    global_to_boundary = _construct_global_to_boundary(is_boundary)

    @kernel function fill_sparse_arrays_kernel(
        I,
        J,
        V,
        data,
        eval_points,
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
        hermite_data = batch_hermite_datas[batch_idx]

        # Calculate range for this batch
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, N_eval)

        # Pre-allocate work arrays for this thread
        n = k + nmon  # max possible stencil size
        A = Symmetric(zeros(TD, n, n), :U)
        b = _prepare_b(ℒrbf, TD, n)

        for eval_idx in start_idx:end_idx
            start_pos = row_offsets[eval_idx]
            neighbors = adjl[eval_idx]
            eval_point = eval_points[eval_idx]

            # Handle different stencil types
            stencil_type_result = stencil_type(
                is_boundary, boundary_conditions, eval_idx, neighbors, global_to_boundary
            )

            if isa(stencil_type_result, DirichletStencil)
                # Dirichlet center: only diagonal element is non-zero
                I[start_pos] = eval_idx
                J[start_pos] = eval_idx
                V[start_pos, :] .= 1.0
            else
                # Standard or Hermite stencil: determine local_data and dispatch
                local_data = if isa(stencil_type_result, StandardStencil)
                    view(data, neighbors)
                elseif isa(stencil_type_result, HermiteStencil)
                    update_stencil_data!(
                        hermite_data,
                        data,
                        neighbors,
                        is_boundary,
                        boundary_conditions,
                        normals,
                        global_to_boundary,
                    )
                    hermite_data
                end

                weights = _build_stencil!(
                    A, b, ℒrbf, ℒmon, local_data, eval_point, basis, mon, k
                )

                # Fill sparse matrix entries
                for local_idx in 1:k
                    pos = start_pos + local_idx - 1
                    I[pos] = eval_idx
                    J[pos] = neighbors[local_idx]
                    if num_ops == 1
                        V[pos, 1] = weights[local_idx]
                    else
                        V[pos, :] = weights[local_idx, :]
                    end
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
        eval_points,
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
    nrows = length(eval_points)
    ncols = length(data)

    if num_ops == 1
        return sparse(I, J, V[:, 1], nrows, ncols)
    else
        return ntuple(i -> sparse(I, J, V[:, i], nrows, ncols), num_ops)
    end
end

"""
Helper function to count non-zero elements for optimized sparse matrix allocation.
"""
function _count_nonzeros(
    adjl, is_boundary::Vector{Bool}, boundary_conditions::Vector{<:BoundaryCondition}
)
    N_eval = length(adjl)
    nnz_per_row = Vector{Int}(undef, N_eval)
    row_offsets = Vector{Int}(undef, N_eval + 1)
    global_to_boundary = _construct_global_to_boundary(is_boundary)

    total_nnz = 0
    row_offsets[1] = 1  # 1-based indexing for first row

    for eval_idx in 1:N_eval
        if is_boundary[eval_idx]
            boundary_idx = global_to_boundary[eval_idx]
            bc = boundary_conditions[boundary_idx]
            if is_dirichlet(bc)
                # Dirichlet: only diagonal element is non-zero
                nnz_per_row[eval_idx] = 1
            else
                # Neumann/Robin center: full stencil
                nnz_per_row[eval_idx] = length(adjl[eval_idx])
            end
        else
            # Interior: full stencil
            nnz_per_row[eval_idx] = length(adjl[eval_idx])
        end

        total_nnz += nnz_per_row[eval_idx]
        row_offsets[eval_idx + 1] = total_nnz + 1
    end

    return total_nnz, nnz_per_row, row_offsets
end

"""
Construct global_to_boundary index mapping.
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