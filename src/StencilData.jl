struct StencilData{T}
    A::Symmetric{T,Matrix{T}}            # Local system matrix
    b::Matrix{T}                          # Local RHS matrix (one column per operator)
    d::Vector{Vector{T}}                  # Local data points (now using the same type T)
    is_boundary::Vector{Bool}             # Whether nodes are on boundary
    is_Neumann::Vector{Bool}              # Whether nodes have Neumann boundary conditions
    normal::Vector{Vector{T}}             # Normal vectors (meaningful for Neumann points)
    lhs_v::Matrix{T}                      # Local coefficients for internal nodes
    rhs_v::Matrix{T}                      # Local coefficients for boundary nodes

    # Constructor to initialize all fields
    function StencilData{T}(n::Int, k::Int, num_ops::Int, dim::Int) where {T}

        A = Symmetric(zeros(T, n, n), :U)
        b = zeros(T, n, num_ops)
        d = [zeros(T, dim) for _ in 1:k]
        is_boundary = zeros(Bool, k)
        is_Neumann = zeros(Bool, k)
        normal = [zeros(T, dim) for _ in 1:k]
        lhs_v = zeros(T, k, num_ops)
        rhs_v = zeros(T, k, num_ops)

        return new(A, b, d, is_boundary, is_Neumann, normal, lhs_v, rhs_v)
    end
end

#convenience constructor
function StencilData(T::Type, data_dim::Int, n::Int, k::Int, num_ops::Int)
    return StencilData{T}(n, k, num_ops, data_dim)
end

"""
    _update_stencil!(
        stencil::StencilData{T},
        local_adjl,
        data,
        boundary_flag,
        is_Neumann,
        normals,
        ℒrbf,
        ℒmon,
        eval_point,
        basis::B,
        mon::MonomialBasis{Dim,Deg},
        k::Int
    ) where {T, B<:AbstractRadialBasis, Dim, Deg}

Updates stencil data and computes weights in a single operation.
Modifies the stencil in-place.
"""
function _update_stencil!(
    stencil::StencilData{T},
    local_adjl,
    data,
    boundary_flag,
    is_Neumann,
    normals,
    ℒrbf,
    ℒmon,
    eval_point,
    basis::B,
    mon::MonomialBasis{Dim,Deg},
    k::Int,
) where {T,B<:AbstractRadialBasis,Dim,Deg}
    fill!(stencil.lhs_v, 0)
    fill!(stencil.rhs_v, 0)
    fill!(parent(stencil.A), 0)
    fill!(stencil.b, 0)
    
    # Update stencil data
    for (idx, j) in enumerate(local_adjl)
        stencil.d[idx] = data[j]
    end
    
    for (j_local, j_global) in enumerate(local_adjl)
        stencil.is_boundary[j_local] = boundary_flag[j_global]
        stencil.is_Neumann[j_local] = is_Neumann[j_global]
        if is_Neumann[j_global]
            copyto!(stencil.normal[j_local], convert.(T, normals[j_global]))
        else
            stencil.normal[j_local] .= zero(T)
        end
    end
    
    # Build collocation matrix and RHS
    _build_collocation_matrix_Hermite!(stencil.A, stencil.d, stencil, basis, mon, k)
    _build_rhs!(stencil.b, ℒrbf, ℒmon, stencil.d, stencil, eval_point, basis, k)
    
    # Solve system for each operator (this can easily be improved by using LDL algo)
    weights = (stencil.A \ stencil.b)[1:k, :]
    
    # Store weights in appropriate matrices
    for j in 1:k
        if !stencil.is_boundary[j]
            stencil.lhs_v[j, :] .= view(weights, j, :)
        else
            stencil.rhs_v[j, :] .= view(weights, j, :)
        end
    end
    
    return nothing
end

function _build_collocation_matrix_Hermite!(
    A::Symmetric, 
    data::AbstractVector, 
    stencil::StencilData, 
    basis::B, 
    mon::MonomialBasis{Dim,Deg}, 
    k::K
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg}
    # radial basis section
    AA = parent(A)
    n = size(A, 2)
    @inbounds for j in 1:k, i in 1:j
        _calculate_matrix_entry_RBF!(AA, i, j, data, stencil, basis)
    end

    # monomial augmentation
    if Deg > -1
        @inbounds for i in 1:k
            _calculate_matrix_entry_poly!(AA,i,k + 1,n,data[i],stencil.is_Neumann[i],stencil.normal[i],mon)
        end
    end

    return nothing
end

function _calculate_matrix_entry_RBF!(A, i, j, data, stencil::StencilData, basis)
    is_Neumann_i = stencil.is_Neumann[i]
    is_Neumann_j = stencil.is_Neumann[j]
    if !is_Neumann_i && !is_Neumann_j
        A[i, j] = basis(data[i], data[j])
    elseif is_Neumann_i && !is_Neumann_j
        n = stencil.normal[i]
        A[i,j] = LinearAlgebra.dot(n, ∇(basis)(data[i], data[j]))
    elseif !is_Neumann_i && is_Neumann_j
        n = stencil.normal[j]
        A[i,j] = LinearAlgebra.dot(n, -∇(basis)(data[i], data[j]))
    elseif is_Neumann_i && is_Neumann_j
        ni = stencil.normal[i]
        nj = stencil.normal[j]
        A[i, j] = directional∂²(basis, ni, nj)(data[i], data[j])
    end
    return nothing
end

function _calculate_matrix_entry_poly!(A, row, col_start, col_end, data_point, is_Neumann, normal, mon)
    # Get view of the polynomial part for this row
    a = view(A, row, col_start:col_end)
    
    if is_Neumann
        # For Neumann boundary points, use normal derivative
        # This uses the ∂_normal function we created earlier
        ∂_normal(mon, normal)(a, data_point)
    else
        # For regular points, use standard polynomial evaluation
        mon(a, data_point)
    end
    
    return nothing
end

function _build_rhs!(
    b::Matrix{T}, 
    ℒrbf::Tuple, 
    ℒmon::Tuple, 
    data::AbstractVector, 
    stencil::StencilData, 
    eval_point, 
    basis::B, 
    k::Int,
) where {T, B<:AbstractRadialBasis}
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

# Handle the case when ℒrbf and ℒmon are single operators (not tuples)
function _build_rhs!(
    b::Matrix{T}, 
    ℒrbf, 
    ℒmon, 
    data::AbstractVector, 
    stencil::StencilData, 
    eval_point, 
    basis::B, 
    k::Int,
) where {T, B<:AbstractRadialBasis}
    # Wrap single operators in a tuple and call the tuple version
    return _build_rhs!(b, (ℒrbf,), (ℒmon,), data, stencil, eval_point, basis, k)
end
