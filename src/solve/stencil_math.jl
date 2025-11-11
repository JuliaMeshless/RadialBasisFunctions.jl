"""
Layer: Pure Mathematical Operations

This file contains the core RBF mathematics for stencil construction:
- Collocation matrix building (A)
- RHS vector building (b)
- Stencil assembly (solving A \\ b)
- Hermite boundary condition modifications

All functions are pure mathematical operations with no I/O or parallelism.
Fully testable and GPU-agnostic.

Called by: kernel_exec.jl (within parallel kernels)
Calls: basis functions, operators
Dependencies: LinearAlgebra, types.jl

Note: Functions use underscore prefix (_build_*) for backward compatibility
with existing code (e.g., interpolation.jl).
"""

using LinearAlgebra: Symmetric, dot

# ============================================================================
# Collocation Matrix Construction
# ============================================================================

"""
    _build_collocation_matrix!(A, data, basis, mon, k)

Build RBF collocation matrix for interior stencil (no boundary conditions).

Matrix structure:
```
┌─────────────────┬─────────┐
│  Φ(xᵢ, xⱼ)      │ P(xᵢ)   │  k×k RBF + k×nmon polynomial
├─────────────────┼─────────┤
│  P(xⱼ)ᵀ         │   0     │  nmon×k poly + nmon×nmon zero
└─────────────────┴─────────┘
```
"""
function _build_collocation_matrix!(
    A::Symmetric,
    data::AbstractVector,
    basis::AbstractRadialBasis,
    mon::MonomialBasis{Dim,Deg},
    k::Int,
) where {Dim,Deg}
    AA = parent(A)
    N = size(A, 2)

    # RBF block (upper triangular, symmetric)
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = basis(data[i], data[j])
    end

    # Polynomial augmentation block
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            mon(a, data[i])
        end
    end

    return nothing
end

"""
    _build_collocation_matrix!(A, data::HermiteStencilData, basis, mon, k)

Build RBF collocation matrix for Hermite stencil (with boundary conditions).

For boundary points with Neumann/Robin conditions, basis functions are modified:
- Instead of Φ(·,xⱼ), we use B₂Φ(·,xⱼ) where B is the boundary operator
- This maintains matrix symmetry
"""
function _build_collocation_matrix!(
    A::Symmetric,
    data::HermiteStencilData,
    basis::AbstractRadialBasis,
    mon::MonomialBasis{Dim,Deg},
    k::Int,
) where {Dim,Deg}
    AA = parent(A)
    N = size(A, 2)

    # RBF block with Hermite modifications
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = compute_hermite_rbf_entry(i, j, data, basis)
    end

    # Polynomial block with boundary modifications
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            compute_hermite_poly_entry!(a, i, data, mon)
        end
    end

    return nothing
end

# ============================================================================
# RHS Vector Construction
# ============================================================================

"""
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)

Build RHS vector for interior stencil, single operator.
"""
function _build_rhs!(
    b::AbstractVector,
    ℒrbf,
    ℒmon,
    data::AbstractVector,
    eval_point,
    basis::AbstractRadialBasis,
    k::Int,
)
    # RBF section: apply operator at eval_point
    @inbounds for i in eachindex(data)
        b[i] = ℒrbf(eval_point, data[i])
    end

    # Polynomial section
    if basis.poly_deg > -1
        bmono = view(b, (k + 1):length(b))
        ℒmon(bmono, eval_point)
    end

    return nothing
end

"""
    _build_rhs!(b, ℒrbf::Tuple, ℒmon::Tuple, data, eval_point, basis, k)

Build RHS vector for interior stencil, multiple operators.
"""
function _build_rhs!(
    b::AbstractMatrix,
    ℒrbf::Tuple,
    ℒmon::Tuple,
    data::AbstractVector,
    eval_point,
    basis::AbstractRadialBasis,
    k::Int,
)
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon)

    # RBF section - each operator
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in eachindex(data)
            b[i, j] = ℒ(eval_point, data[i])
        end
    end

    # Polynomial section - each operator
    if basis.poly_deg > -1
        for (j, ℒ_op) in enumerate(ℒmon)
            bmono = view(b, (k + 1):size(b, 1), j)
            ℒ_op(bmono, eval_point)
        end
    end

    return nothing
end

"""
    _build_rhs!(b, ℒrbf, ℒmon, data::HermiteStencilData, eval_point, basis, mon, k)

Build RHS vector for Hermite stencil, single operator.
Applies boundary conditions to both stencil points and evaluation point.
"""
function _build_rhs!(
    b::AbstractVector,
    ℒrbf,
    ℒmon,
    data::HermiteStencilData,
    eval_point,
    basis::AbstractRadialBasis,
    mon::MonomialBasis,
    k::Int,
)
    # RBF section with boundary modifications for stencil points
    @inbounds for i in 1:k
        b[i] = hermite_rbf_rhs(
            ℒrbf,
            eval_point,
            data.data[i],
            data.is_boundary[i],
            data.boundary_conditions[i],
            data.normals[i],
        )
    end

    # Monomial section
    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)

        # Find evaluation point index in stencil
        eval_idx = findfirst(i -> data.data[i] == eval_point, 1:k)
        @assert eval_idx !== nothing "Evaluation point not found in stencil"
        @assert !is_dirichlet(data.boundary_conditions[eval_idx]) "Dirichlet eval nodes handled at higher level"

        # Apply boundary conditions to monomial evaluation
        hermite_mono_rhs!(
            bmono,
            ℒmon,
            mon,
            eval_point,
            data.is_boundary[eval_idx],
            data.boundary_conditions[eval_idx],
            data.normals[eval_idx],
            eltype(data.data[1]),
        )
    end

    return nothing
end

"""
    _build_rhs!(b, ℒrbf::Tuple, ℒmon::Tuple, data::HermiteStencilData, eval_point, basis, mon, k)

Build RHS vector for Hermite stencil, multiple operators.
"""
function _build_rhs!(
    b::AbstractMatrix,
    ℒrbf::Tuple,
    ℒmon::Tuple,
    data::HermiteStencilData,
    eval_point,
    basis::AbstractRadialBasis,
    mon::MonomialBasis,
    k::Int,
)
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon)

    # RBF section with boundary modifications - each operator
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in 1:k
            b[i, j] = hermite_rbf_rhs(
                ℒ,
                eval_point,
                data.data[i],
                data.is_boundary[i],
                data.boundary_conditions[i],
                data.normals[i],
            )
        end
    end

    # Monomial section
    if basis.poly_deg > -1
        N = size(b, 1)

        # Find evaluation point index
        eval_idx = findfirst(i -> data.data[i] == eval_point, 1:k)
        @assert eval_idx !== nothing "Evaluation point not found in stencil"
        @assert !is_dirichlet(data.boundary_conditions[eval_idx]) "Dirichlet eval nodes handled at higher level"

        # Apply boundary conditions to each operator
        for (j, ℒ_op) in enumerate(ℒmon)
            bmono = view(b, (k + 1):N, j)
            hermite_mono_rhs!(
                bmono,
                ℒ_op,
                mon,
                eval_point,
                data.is_boundary[eval_idx],
                data.boundary_conditions[eval_idx],
                data.normals[eval_idx],
                eltype(data.data[1]),
            )
        end
    end

    return nothing
end

# ============================================================================
# Stencil Assembly
# ============================================================================

"""
    _build_stencil!(A, b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)

Assemble complete stencil: build collocation matrix, build RHS, solve for weights.
Works for both interior and Hermite stencils via multiple dispatch on `data` type.

Returns: weights (first k rows of solution, size k×num_ops)
"""
function _build_stencil!(
    A::Symmetric,
    b,
    ℒrbf,
    ℒmon,
    data::AbstractVector,  # For interior stencils
    eval_point,
    basis::AbstractRadialBasis,
    mon::MonomialBasis,
    k::Int,
)
    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)
    return (A \ b)[1:k, :]
end

# Specialized version for HermiteStencilData
function _build_stencil!(
    A::Symmetric,
    b,
    ℒrbf,
    ℒmon,
    data::HermiteStencilData,
    eval_point,
    basis::AbstractRadialBasis,
    mon::MonomialBasis,
    k::Int,
)
    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)
    return (A \ b)[1:k, :]
end

# ============================================================================
# Hermite-Specific Computations
# ============================================================================

"""
    compute_hermite_rbf_entry(i, j, data, basis)

Compute single RBF matrix entry with Hermite boundary modifications.
Dispatches based on point types (Interior/Dirichlet/NeumannRobin).
"""
function compute_hermite_rbf_entry(
    i::Int, j::Int, data::HermiteStencilData, basis::AbstractRadialBasis
)
    xi, xj = data.data[i], data.data[j]
    type_i = point_type(data.is_boundary[i], data.boundary_conditions[i])
    type_j = point_type(data.is_boundary[j], data.boundary_conditions[j])

    return hermite_rbf_dispatch(type_i, type_j, i, j, xi, xj, data, basis)
end

# Interior-Interior: Standard RBF evaluation
function hermite_rbf_dispatch(::InteriorPoint, ::InteriorPoint, i, j, xi, xj, data, basis)
    return basis(xi, xj)
end

# Interior-Dirichlet: Standard RBF evaluation
function hermite_rbf_dispatch(::InteriorPoint, ::DirichletPoint, i, j, xi, xj, data, basis)
    return basis(xi, xj)
end

# Interior-NeumannRobin: Apply boundary operator to second argument
function hermite_rbf_dispatch(
    ::InteriorPoint, ::NeumannRobinPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_j = data.boundary_conditions[j]
    nj = data.normals[j]
    return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
end

# Dirichlet-Interior: Standard RBF evaluation
function hermite_rbf_dispatch(::DirichletPoint, ::InteriorPoint, i, j, xi, xj, data, basis)
    return basis(xi, xj)
end

# Dirichlet-Dirichlet: Standard RBF evaluation
function hermite_rbf_dispatch(::DirichletPoint, ::DirichletPoint, i, j, xi, xj, data, basis)
    return basis(xi, xj)
end

# Dirichlet-NeumannRobin: Apply boundary operator to second argument
function hermite_rbf_dispatch(
    ::DirichletPoint, ::NeumannRobinPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_j = data.boundary_conditions[j]
    nj = data.normals[j]
    return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
end

# NeumannRobin-Interior: Apply boundary operator to first argument
function hermite_rbf_dispatch(
    ::NeumannRobinPoint, ::InteriorPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_i = data.boundary_conditions[i]
    ni = data.normals[i]
    return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
end

# NeumannRobin-Dirichlet: Apply boundary operator to first argument
function hermite_rbf_dispatch(
    ::NeumannRobinPoint, ::DirichletPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_i = data.boundary_conditions[i]
    ni = data.normals[i]
    return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
end

# NeumannRobin-NeumannRobin: Apply boundary operators to both arguments
function hermite_rbf_dispatch(
    ::NeumannRobinPoint, ::NeumannRobinPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_i = data.boundary_conditions[i]
    bc_j = data.boundary_conditions[j]
    ni = data.normals[i]
    nj = data.normals[j]
    ∂i∂j_φ = directional∂²(basis, ni, nj)(xi, xj)

    return (
        α(bc_i) * α(bc_j) * φ +
        α(bc_i) * β(bc_j) * dot(nj, -∇φ) +
        β(bc_i) * α(bc_j) * dot(ni, ∇φ) +
        β(bc_i) * β(bc_j) * ∂i∂j_φ
    )
end

"""
    compute_hermite_poly_entry!(a, i, data, mon)

Compute polynomial entries for Hermite interpolation.
Applies boundary operators to polynomial basis at boundary points.
"""
function compute_hermite_poly_entry!(
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

            T = eltype(a)
            poly_vals = zeros(T, nmon)
            deriv_vals = zeros(T, nmon)

            mon(poly_vals, xi)
            ∂_normal(mon, ni)(deriv_vals, xi)

            @inbounds for k in 1:nmon
                a[k] = α(bc_i) * poly_vals[k] + β(bc_i) * deriv_vals[k]
            end
        end
    end

    return nothing
end

# ============================================================================
# Boundary Condition Helpers
# ============================================================================

"""
    hermite_rbf_rhs(ℒrbf, eval_point, data_point, is_bound, bc, normal)

Apply boundary conditions to RBF operator evaluation.
- Interior/Dirichlet: standard evaluation ℒΦ(x_eval, x_data)
- Neumann/Robin: α*ℒΦ + β*ℒ(∂ₙΦ)
"""
@inline function hermite_rbf_rhs(
    ℒrbf, eval_point, data_point, is_bound::Bool, bc::BoundaryCondition, normal
)
    if !is_bound || is_dirichlet(bc)
        return ℒrbf(eval_point, data_point)
    else
        # Neumann/Robin: α*ℒφ + β*ℒ(∂ₙφ)
        return α(bc) * ℒrbf(eval_point, data_point) +
               β(bc) * ℒrbf(eval_point, data_point, normal)
    end
end

"""
    hermite_mono_rhs!(bmono, ℒmon, mon, eval_point, is_bound, bc, normal, T)

Apply boundary conditions to monomial operator evaluation.
- Interior/Dirichlet: apply operator ℒ to monomials
- Neumann/Robin: α*ℒ(P) + β*ℒ(∂ₙP)
"""
@inline function hermite_mono_rhs!(
    bmono::AbstractVector,
    ℒmon,
    mon::MonomialBasis,
    eval_point,
    is_bound::Bool,
    bc::BoundaryCondition,
    normal,
    T::Type,
)
    if !is_bound || is_dirichlet(bc)
        # Interior or Dirichlet: apply operator to standard monomial basis
        ℒmon(bmono, eval_point)
        return nothing
    end

    # Neumann/Robin: α*ℒ(P) + β*ℒ(∂ₙP)
    nmon = length(bmono)
    poly_vals = zeros(T, nmon)
    deriv_vals = zeros(T, nmon)

    mon(poly_vals, eval_point)
    ∂_normal(mon, normal)(deriv_vals, eval_point)

    @inbounds for idx in 1:nmon
        bmono[idx] = α(bc) * poly_vals[idx] + β(bc) * deriv_vals[idx]
    end
end
