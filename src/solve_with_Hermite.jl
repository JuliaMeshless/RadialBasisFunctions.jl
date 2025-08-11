#here I want to define the RegionData stuct and want to optimize all the functions for it
#domain data constructors are of responsability of MeshlessMultiphysics.jl

struct FunctionalData
    basis
    mon
    ℒrbf::Tuple
    ℒmon::Tuple
    function FunctionalData(basis, mon, ℒrbf, ℒmon)
        if type(ℒrbf) != Tuple
            ℒrbf = (ℒrbf,)
        end
        if type(ℒmon) != Tuple
            ℒmon = (ℒmon,)
        end
        return new(basis, mon, ℒrbf, ℒmon)
    end
end

struct RegionData
    all_coords # N_points x dim
    is_boundary # N_points
    boundary_type # N_boundary
    boundary_normals # N_boundary x dim
    adjl
    functional_data::FunctionalData # Contains basis, monomial basis, and linear operators
    stencil_data # N_points x threads
    global_to_boundary::Vector{Int} # Maps global indices to boundary indices
    boundary_to_global::Vector{Int} # Maps boundary indices to global indices

    function RegionData(
        all_coords, is_boundary, boundary_type, boundary_normals, adjl, functional_data
    )
        nchunks = Threads.nthreads()
        stencil_data = [StencilData(all_coords, adjl, functional_data) for _ in 1:nchunks]
        global_to_boundary, boundary_to_global = set_global_and_boundary_indices!(
            region_data
        )
        return new(
            all_coords,
            is_boundary,
            boundary_type,
            boundary_normals,
            adjl,
            functional_data,
            stencil_data,
            global_to_boundary,
            boundary_to_global,
        )
    end
end

function set_global_and_boundary_indices!(region_data::RegionData)
    #I want this function to provide a mapping from global to boundary indices and vice versa
    global_to_boundary = zeros(Int, length(region_data.all_coords))
    boundary_to_global = zeros(Int, sum(region_data.is_boundary))

    global_idx = 1
    for (i, is_bnd) in enumerate(region_data.is_boundary)
        if is_bnd
            boundary_to_global[global_idx] = i
            global_idx += 1
        end
    end

    return global_to_boundary, boundary_to_global
end

function _build_weights(region_data::RegionData)
    lhs, rhs = _preallocate_IJV_matrices(region_data)

    lhs_offsets, rhs_offsets = _calculate_thread_offsets(region_data, nchunks)

    # Build stencil for each data point and store in global weight matrices
    Threads.@threads for (ichunk, xrange) in
                         enumerate(index_chunks(region_data.adjl; n=nchunks))
        for i in xrange
            if !region_data.is_boundary[i]

                # Update stencil and compute weights in a single call
                _set_stencil_eval_point!(region_data, ichunk, region_data.all_coords[i])
                _update_stencil!(region_data, i, ichunk)

                # Copy results from stencil_data to global matrices
                _write_coefficients_to_global_matrices!(
                    lhs,
                    rhs,
                    region_data.stencil_data[ichunk],
                    lhs_offsets[ichunk],
                    rhs_offsets[ichunk],
                )
            end
        end
    end

    return _return_global_matrices(lhs, rhs, region_data.is_boundary)
end

function _preallocate_IJV_matrices(region_data)
    TD = eltype(first(region_data.all_coords))
    adjl = region_data.adjl
    Na = length(adjl)
    num_ops = _num_ops(region_data.functional_data.ℒrbf)
    boundary_flag = region_data.is_boundary

    lhs_count = 0
    rhs_count = 0

    for i in 1:Na
        if !boundary_flag[i]  # internal node
            # Count neighbors directly without creating intermediate arrays
            for j in adjl[i]
                if !boundary_flag[j]  # internal neighbor
                    lhs_count += 1
                else  # boundary neighbor
                    rhs_count += 1
                end
            end
        end
    end

    I_lhs = zeros(Int, lhs_count)
    J_lhs = zeros(Int, lhs_count)
    V_lhs = zeros(TD, lhs_count, num_ops)

    I_rhs = zeros(Int, rhs_count)
    J_rhs = zeros(Int, rhs_count)
    V_rhs = zeros(TD, rhs_count, num_ops)

    # Create mapping from global indices to internal/boundary-specific indices using cumsum
    internal_idx = cumsum(.!boundary_flag) .* (.!boundary_flag)
    boundary_idx = cumsum(boundary_flag) .* boundary_flag

    # Fill indices
    lhs_idx = 1
    rhs_idx = 1

    for i in eachindex(adjl)
        if !boundary_flag[i]  # If node i is internal
            i_internal = internal_idx[i]  # Remap to internal index
            for j in adjl[i]  # For each neighbor
                if !boundary_flag[j]  # Internal neighbor
                    j_internal = internal_idx[j]  # Remap to internal index
                    I_lhs[lhs_idx] = i_internal
                    J_lhs[lhs_idx] = j_internal
                    lhs_idx += 1
                else  # Boundary neighbor
                    j_boundary = boundary_idx[j]  # Remap to boundary index
                    I_rhs[rhs_idx] = i_internal
                    J_rhs[rhs_idx] = j_boundary
                    rhs_idx += 1
                end
            end
        end
    end

    return (lhs=(I=I_lhs, J=J_lhs, V=V_lhs), rhs=(I=I_rhs, J=J_rhs, V=V_rhs))
end

"""
    _calculate_thread_offsets(adjl, boundary_flag, nchunks)

Calculate the starting offsets for each thread when filling LHS and RHS matrices.
- lhs_offsets: Starting indices for internal-to-internal connections
- rhs_offsets: Starting indices for internal-to-boundary connections

Returns a tuple of (lhs_offsets, rhs_offsets).
"""
function _calculate_thread_offsets(region_data, nchunks)
    adjl = region_data.adjl
    boundary_flag = region_data.is_boundary
    thread_lhs_counts = zeros(Int, nchunks)
    thread_rhs_counts = zeros(Int, nchunks)

    # Count elements per thread
    for (ichunk, xrange) in enumerate(index_chunks(adjl; n=nchunks))
        for i in xrange
            if !boundary_flag[i]  # Only internal nodes generate equations
                # Count neighbors directly without creating intermediate arrays
                for j in adjl[i]
                    if !boundary_flag[j]  # internal neighbor
                        thread_lhs_counts[ichunk] += 1
                    else  # boundary neighbor
                        thread_rhs_counts[ichunk] += 1
                    end
                end
            end
        end
    end

    # Calculate starting indices for each thread
    lhs_offsets = cumsum([0; thread_lhs_counts[1:(end - 1)]]) .+ 1
    rhs_offsets = cumsum([0; thread_rhs_counts[1:(end - 1)]]) .+ 1

    return lhs_offsets, rhs_offsets
end

function _write_coefficients_to_global_matrices!(lhs, rhs, stencil, lhs_idx, rhs_idx)
    for j_local in eachindex(stencil.local_adjl)
        if !stencil.is_boundary[j_local]  # Internal node -> goes to LHS
            lhs.V[lhs_idx, :] = stencil.lhs_v[j_local, :]
            lhs_idx += 1
        else  # Boundary node -> goes to RHS
            rhs.V[rhs_idx, :] = stencil.rhs_v[j_local, :]
            rhs_idx += 1
        end
    end

    return nothing
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
function _return_global_matrices(lhs, rhs, boundary_flag)
    nrows = count(.!boundary_flag)
    ncols_lhs = count(.!boundary_flag)
    ncols_rhs = count(boundary_flag)

    if size(lhs.V, 2) == 1
        lhs_matrix = sparse(lhs.I, lhs.J, lhs.V[:, 1], nrows, ncols_lhs)
        rhs_matrix = sparse(rhs.I, rhs.J, rhs.V[:, 1], nrows, ncols_rhs)
        return lhs_matrix, rhs_matrix
    else
        lhs_matrices = ntuple(
            i -> sparse(lhs.I, lhs.J, lhs.V[:, i], nrows, ncols_lhs), size(lhs.V, 2)
        )
        rhs_matrices = ntuple(
            i -> sparse(rhs.I, rhs.J, rhs.V[:, i], nrows, ncols_rhs), size(rhs.V, 2)
        )
        return lhs_matrices, rhs_matrices
    end
end
