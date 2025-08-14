#here I want to define the RegionData stuct and want to optimize all the functions for it
#domain data constructors are of responsability of MeshlessMultiphysics.jl

struct FunctionalData
    basis
    mon
    ℒrbf::Tuple
    ℒmon::Tuple
    function FunctionalData(basis, mon, ℒrbf, ℒmon)
        ℒrbf_tuple = ℒrbf isa Tuple ? ℒrbf : (ℒrbf,)
        ℒmon_tuple = ℒmon isa Tuple ? ℒmon : (ℒmon,)
        return new(basis, mon, ℒrbf_tuple, ℒmon_tuple)
    end
end

struct RegionData
    all_coords                 # Vector of coordinate vectors (N_points)
    is_boundary::Vector{Bool}  # Length N_points
    boundary_types             # Vector length N_boundary (BoundaryType objects)
    normals                    # Vector length N_boundary (normal vectors)
    adjl                       # Adjacency list (Vector of neighbor index vectors)
    functional_data::FunctionalData
    global_to_boundary::Vector{Int}  # Maps global index -> boundary index (0 if internal)
    boundary_to_global::Vector{Int}  # Maps boundary index -> global index
    function RegionData(
        all_coords,
        is_boundary::AbstractVector{Bool},
        boundary_types,
        normals,
        adjl,
        functional_data::FunctionalData,
    )
        global_to_boundary, boundary_to_global = _compute_boundary_mappings(is_boundary)
        return new(
            all_coords,
            collect(is_boundary),
            boundary_types,
            normals,
            adjl,
            functional_data,
            global_to_boundary,
            boundary_to_global,
        )
    end
end

function _compute_boundary_mappings(is_boundary::AbstractVector{Bool})
    N = length(is_boundary)
    n_bnd = count(is_boundary)
    global_to_boundary = zeros(Int, N)
    boundary_to_global = Vector{Int}(undef, n_bnd)
    bidx = 0
    @inbounds for i in 1:N
        if is_boundary[i]
            bidx += 1
            global_to_boundary[i] = bidx
            boundary_to_global[bidx] = i
        end
    end
    return global_to_boundary, boundary_to_global
end