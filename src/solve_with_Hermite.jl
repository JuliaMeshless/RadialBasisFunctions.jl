function _build_weights(data, normals, boundary_flag, is_Neumann, adjl, basis, ℒrbf, ℒmon, mon)

    nchunks = Threads.nthreads()
    TD = eltype(first(data))
    dim = length(first(data))
    nmon = binomial(dim + basis.poly_deg, basis.poly_deg)
    k = length(first(adjl))

    (I_lhs, J_lhs, V_lhs),(I_rhs, J_rhs, V_rhs) = _preallocate_IJV_matrices(
        adjl, data, boundary_flag, ℒrbf)
    
    stencil_data = [StencilData(TD, dim, k+nmon, k, _num_ops(ℒrbf)) for _ in 1:nchunks]
    
    # Get thread offsets
    lhs_offsets, rhs_offsets = _calculate_thread_offsets(adjl, boundary_flag, nchunks)
    
    # Build stencil for each data point and store in global weight matrices
    Threads.@threads for (ichunk, xrange) in enumerate(index_chunks(adjl; n=nchunks))
        lhs_idx = lhs_offsets[ichunk] + 1
        rhs_idx = rhs_offsets[ichunk] + 1
        
        for i in xrange
            if !boundary_flag[i]
                
                # Update stencil and compute weights in a single call
                _update_stencil!(
                    stencil_data[ichunk],
                    adjl[i],
                    data,
                    boundary_flag,
                    is_Neumann,
                    normals,
                    ℒrbf,
                    ℒmon,
                    data[i],
                    basis,
                    mon,
                    k
                )
                
                # Copy results from stencil_data to global matrices
                lhs_idx, rhs_idx = _write_coefficients_to_global_matrices!(
                    V_lhs, V_rhs, stencil_data[ichunk], adjl[i], boundary_flag, lhs_idx, rhs_idx
                )
            end
        end
    end
    
    return _return_global_matrices(I_lhs, J_lhs, V_lhs, I_rhs, J_rhs, V_rhs, boundary_flag)
end

function _preallocate_IJV_matrices(adjl, data, boundary_flag, ℒrbf)
    TD = eltype(first(data))
    Na = length(adjl)
    num_ops = _num_ops(ℒrbf)
    
    # Count entries in one pass while also collecting the I, J pairs
    lhs_pairs = Tuple{Int, Int}[]
    rhs_pairs = Tuple{Int, Int}[]
    
    for i in 1:Na
        if !boundary_flag[i]  # internal node
            for j in adjl[i]
                if !boundary_flag[j]  # internal neighbor
                    push!(lhs_pairs, (i, j))
                else  # boundary neighbor
                    push!(rhs_pairs, (i, j))
                end
            end
        end
    end
    
    # Create arrays with exact sizes
    lhs_count = length(lhs_pairs)
    rhs_count = length(rhs_pairs)
    
    I_lhs = zeros(Int, lhs_count)
    J_lhs = zeros(Int, lhs_count)
    V_lhs = zeros(TD, lhs_count, num_ops)
    
    I_rhs = zeros(Int, rhs_count)
    J_rhs = zeros(Int, rhs_count)
    V_rhs = zeros(TD, rhs_count, num_ops)
    
    # Fill arrays directly
    for (idx, (i, j)) in enumerate(lhs_pairs)
        I_lhs[idx] = i
        J_lhs[idx] = j
    end
    
    for (idx, (i, j)) in enumerate(rhs_pairs)
        I_rhs[idx] = i
        J_rhs[idx] = j
    end
    
    return (I_lhs, J_lhs, V_lhs), (I_rhs, J_rhs, V_rhs)
end

"""
    _calculate_thread_offsets(adjl, boundary_flag, nchunks)

Calculate the starting offsets for each thread when filling LHS and RHS matrices.
- lhs_offsets: Starting indices for internal-to-internal connections
- rhs_offsets: Starting indices for internal-to-boundary connections

Returns a tuple of (lhs_offsets, rhs_offsets).
"""
function _calculate_thread_offsets(adjl, boundary_flag, nchunks)
    thread_lhs_counts = zeros(Int, nchunks)
    thread_rhs_counts = zeros(Int, nchunks)
    
    # Count elements per thread
    for (ichunk, xrange) in enumerate(index_chunks(adjl; n=nchunks))
        for i in xrange
            if !boundary_flag[i]  # Only internal nodes generate equations
                local_adjl = adjl[i]
                for j_global in local_adjl
                    if !boundary_flag[j_global]  # Internal neighbor -> LHS
                        thread_lhs_counts[ichunk] += 1
                    else  # Boundary neighbor -> RHS
                        thread_rhs_counts[ichunk] += 1
                    end
                end
            end
        end
    end
    
    # Calculate starting indices for each thread
    lhs_offsets = cumsum([0; thread_lhs_counts[1:end-1]])
    rhs_offsets = cumsum([0; thread_rhs_counts[1:end-1]])
    
    return lhs_offsets, rhs_offsets
end

function _write_coefficients_to_global_matrices!(
    V_lhs, V_rhs, stencil, local_adjl, boundary_flag, lhs_idx, rhs_idx
)
    for (j_local, j_global) in enumerate(local_adjl)
        if !boundary_flag[j_global]  # Internal node -> goes to LHS
            V_lhs[lhs_idx, :] = stencil.lhs_v[j_local, :]
            lhs_idx += 1
        else  # Boundary node -> goes to RHS
            V_rhs[rhs_idx, :] = stencil.rhs_v[j_local, :]
            rhs_idx += 1
        end
    end
    
    return lhs_idx, rhs_idx
end

"""
    _return_global_matrices(I_lhs, J_lhs, V_lhs, I_rhs, J_rhs, V_rhs, boundary_flag)

Constructs sparse matrix representation of the global linear system.

# Arguments
- `I_lhs`, `J_lhs`, `V_lhs`: COO format components for LHS matrix
- `I_rhs`, `J_rhs`, `V_rhs`: COO format components for RHS matrix
- `boundary_flag`: Boolean array indicating boundary nodes

# Returns
- For single operators: tuple of (lhs_matrix, rhs_matrix)
- For multiple operators: tuple of (lhs_matrices, rhs_matrices)
"""
function _return_global_matrices(I_lhs, J_lhs, V_lhs, I_rhs, J_rhs, V_rhs, boundary_flag)
    nrows = count(.!boundary_flag)
    ncols_lhs = count(.!boundary_flag)
    ncols_rhs = count(boundary_flag)
    
    if size(V_lhs, 2) == 1
        lhs_matrix = sparse(I_lhs, J_lhs, V_lhs[:, 1], nrows, ncols_lhs)
        rhs_matrix = sparse(I_rhs, J_rhs, V_rhs[:, 1], nrows, ncols_rhs)
        return lhs_matrix, rhs_matrix
    else
        lhs_matrices = ntuple(i -> sparse(I_lhs, J_lhs, V_lhs[:, i], nrows, ncols_lhs), size(V_lhs, 2))
        rhs_matrices = ntuple(i -> sparse(I_rhs, J_rhs, V_rhs[:, i], nrows, ncols_rhs), size(V_rhs, 2))
        return lhs_matrices, rhs_matrices
    end
end
