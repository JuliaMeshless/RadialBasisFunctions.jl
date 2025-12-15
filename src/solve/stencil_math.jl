using LinearAlgebra: Symmetric, dot

# ============================================================================
# Dispatch Helpers for Data Access
# ============================================================================

# Get point at index i - dispatch on data type
_get_point(data::AbstractVector, i) = data[i]
_get_point(data::HermiteStencilData, i) = data.data[i]

# Compute RBF matrix entry - dispatch on data type
_rbf_entry(i, j, data::AbstractVector, basis) = basis(data[i], data[j])
function _rbf_entry(i, j, data::HermiteStencilData, basis)
    return compute_hermite_rbf_entry(i, j, data, basis)
end

# Compute polynomial entry - dispatch on data type
_poly_entry!(a, i, data::AbstractVector, mon) = mon(a, data[i])
function _poly_entry!(a, i, data::HermiteStencilData, mon)
    return compute_hermite_poly_entry!(a, i, data, mon)
end

# Compute RBF RHS entry - dispatch on data type
_rbf_rhs(ℒrbf, eval_point, data::AbstractVector, i) = ℒrbf(eval_point, data[i])
function _rbf_rhs(ℒrbf, eval_point, data::HermiteStencilData, i)
    return hermite_rbf_rhs(
        ℒrbf,
        eval_point,
        data.data[i],
        data.is_boundary[i],
        data.boundary_conditions[i],
        data.normals[i],
    )
end

# Compute monomial RHS - dispatch on data type
_mono_rhs!(bmono, ℒmon, mon, eval_point, data::AbstractVector, k) = ℒmon(bmono, eval_point)
function _mono_rhs!(bmono, ℒmon, mon, eval_point, data::HermiteStencilData, k)
    eval_idx = findfirst(i -> data.data[i] == eval_point, 1:k)
    @assert eval_idx !== nothing "Evaluation point not found in stencil"
    @assert !is_dirichlet(data.boundary_conditions[eval_idx]) "Dirichlet eval nodes handled at higher level"
    return hermite_mono_rhs!(
        bmono,
        ℒmon,
        mon,
        eval_point,
        data.is_boundary[eval_idx],
        data.boundary_conditions[eval_idx],
        data.normals[eval_idx],
        data.poly_workspace,
    )
end

# ============================================================================
# Collocation Matrix Construction
# ============================================================================

"""
    _build_collocation_matrix!(A, data, basis, mon, k)

Build RBF collocation matrix. Works for both interior stencils (AbstractVector data)
and Hermite stencils (HermiteStencilData) via dispatch helpers.

Matrix structure:
```
┌─────────────────┬─────────┐
│  Φ(xᵢ, xⱼ)      │ P(xᵢ)   │  k×k RBF + k×nmon polynomial
├─────────────────┼─────────┤
│  P(xⱼ)ᵀ         │   0     │  nmon×k poly + nmon×nmon zero
└─────────────────┴─────────┘
```

For Hermite stencils with Neumann/Robin conditions, basis functions are modified
via `_rbf_entry` and `_poly_entry!` dispatch to maintain matrix symmetry.
"""
function _build_collocation_matrix!(
    A::Symmetric, data, basis::AbstractRadialBasis, mon::MonomialBasis{Dim,Deg}, k::Int
) where {Dim,Deg}
    AA = parent(A)
    N = size(A, 2)

    # RBF block (upper triangular, symmetric) - dispatches on data type
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = _rbf_entry(i, j, data, basis)
    end

    # Polynomial augmentation block - dispatches on data type
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            _poly_entry!(a, i, data, mon)
        end
    end

    return nothing
end

# ============================================================================
# RHS Vector Construction
# ============================================================================

"""
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)

Build RHS vector for single operator. Works for both interior stencils (AbstractVector)
and Hermite stencils (HermiteStencilData) via dispatch helpers.
"""
function _build_rhs!(
    b::AbstractVector,
    ℒrbf,
    ℒmon,
    data,
    eval_point,
    basis::AbstractRadialBasis,
    mon::MonomialBasis,
    k::Int,
)
    # RBF section - dispatches on data type
    @inbounds for i in 1:k
        b[i] = _rbf_rhs(ℒrbf, eval_point, data, i)
    end

    # Polynomial section - dispatches on data type
    if basis.poly_deg > -1
        bmono = view(b, (k + 1):length(b))
        _mono_rhs!(bmono, ℒmon, mon, eval_point, data, k)
    end

    return nothing
end

"""
    _build_rhs!(b, ℒrbf::Tuple, ℒmon::Tuple, data, eval_point, basis, mon, k)

Build RHS vector for multiple operators. Works for both interior stencils (AbstractVector)
and Hermite stencils (HermiteStencilData) via dispatch helpers.
"""
function _build_rhs!(
    b::AbstractMatrix,
    ℒrbf::Tuple,
    ℒmon::Tuple,
    data,
    eval_point,
    basis::AbstractRadialBasis,
    mon::MonomialBasis,
    k::Int,
)
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon)

    # RBF section - each operator, dispatches on data type
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in 1:k
            b[i, j] = _rbf_rhs(ℒ, eval_point, data, i)
        end
    end

    # Polynomial section - each operator, dispatches on data type
    if basis.poly_deg > -1
        for (j, ℒ_op) in enumerate(ℒmon)
            bmono = view(b, (k + 1):size(b, 1), j)
            _mono_rhs!(bmono, ℒ_op, mon, eval_point, data, k)
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
Works for both interior stencils (AbstractVector) and Hermite stencils (HermiteStencilData)
via dispatch helpers in `_build_collocation_matrix!` and `_build_rhs!`.

Returns: weights (first k rows of solution, size k×num_ops)
"""
function _build_stencil!(
    A::Symmetric,
    b,
    ℒrbf,
    ℒmon,
    data,
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
Dispatches based on boundary point type for type stability.
"""
function compute_hermite_poly_entry!(
    a::AbstractVector, i::Int, data::HermiteStencilData, mon::MonomialBasis
)
    xi = data.data[i]
    pt = point_type(data.is_boundary[i], data.boundary_conditions[i])
    return hermite_poly_dispatch!(a, pt, xi, data, i, mon)
end

# Interior/Dirichlet: standard polynomial evaluation
function hermite_poly_dispatch!(
    a::AbstractVector,
    ::Union{InteriorPoint,DirichletPoint},
    xi,
    data,
    i,
    mon::MonomialBasis,
)
    mon(a, xi)
    return nothing
end

# NeumannRobin: α*P + β*∂ₙP using pre-allocated workspace
function hermite_poly_dispatch!(
    a::AbstractVector,
    ::NeumannRobinPoint,
    xi,
    data::HermiteStencilData,
    i,
    mon::MonomialBasis,
)
    bc = data.boundary_conditions[i]
    ni = data.normals[i]
    workspace = data.poly_workspace

    # Compute P(xi) into a
    mon(a, xi)

    # Compute ∂ₙP(xi) into workspace
    ∂_normal(mon, ni)(workspace, xi)

    # Combine: a = α*P + β*∂ₙP
    α_val, β_val = α(bc), β(bc)
    @inbounds for k in eachindex(a)
        a[k] = α_val * a[k] + β_val * workspace[k]
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
    hermite_mono_rhs!(bmono, ℒmon, mon, eval_point, is_bound, bc, normal, workspace)

Apply boundary conditions to monomial operator evaluation.
Dispatches based on boundary point type for type stability.
"""
@inline function hermite_mono_rhs!(
    bmono::AbstractVector,
    ℒmon,
    mon::MonomialBasis,
    eval_point,
    is_bound::Bool,
    bc::BoundaryCondition,
    normal,
    workspace::AbstractVector,
)
    pt = point_type(is_bound, bc)
    return hermite_mono_rhs_dispatch!(
        bmono, pt, ℒmon, mon, eval_point, bc, normal, workspace
    )
end

# Interior/Dirichlet: apply operator to standard monomial basis
function hermite_mono_rhs_dispatch!(
    bmono,
    ::Union{InteriorPoint,DirichletPoint},
    ℒmon,
    mon,
    eval_point,
    bc,
    normal,
    workspace,
)
    ℒmon(bmono, eval_point)
    return nothing
end

# NeumannRobin: α*ℒ(P) + β*ℒ(∂ₙP) using pre-allocated workspace
function hermite_mono_rhs_dispatch!(
    bmono, ::NeumannRobinPoint, ℒmon, mon::MonomialBasis, eval_point, bc, normal, workspace
)
    # Compute P(eval_point) into bmono
    mon(bmono, eval_point)

    # Compute ∂ₙP(eval_point) into workspace
    ∂_normal(mon, normal)(workspace, eval_point)

    # Combine: bmono = α*P + β*∂ₙP
    α_val, β_val = α(bc), β(bc)
    @inbounds for idx in eachindex(bmono)
        bmono[idx] = α_val * bmono[idx] + β_val * workspace[idx]
    end
    return nothing
end
