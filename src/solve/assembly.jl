using LinearAlgebra: Symmetric, bunchkaufman!, ldiv!

# Helper to get writable matrix from Symmetric or plain matrix
_get_writable(A::Symmetric) = parent(A)
_get_writable(A) = A

# ============================================================================
# Collocation Matrix Construction
# ============================================================================

"""
    _build_collocation_matrix!(A, data, basis, mon, k)

Build the RBF collocation matrix for one stencil.

Matrix structure:
```
┌─────────────────┬─────────┐
│  Φ(xᵢ, xⱼ)      │ P(xᵢ)   │  k×k RBF + k×nmon polynomial
├─────────────────┼─────────┤
│  P(xⱼ)ᵀ         │   0     │  nmon×k poly + nmon×nmon zero
└─────────────────┴─────────┘
```

Only the upper triangle is written; `A` is wrapped `Symmetric(_, :U)` by the caller.
"""
function _build_collocation_matrix!(
        A, data, basis::AbstractRadialBasis, mon::MonomialBasis{Dim, Deg}, k::Int
    ) where {Dim, Deg}
    AA = _get_writable(A)
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

# ============================================================================
# RHS Vector Construction
# ============================================================================

"""
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)

Build the RHS vector for a single operator.
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
    @inbounds for i in 1:k
        b[i] = ℒrbf(eval_point, data[i])
    end

    if basis.poly_deg > -1
        bmono = view(b, (k + 1):length(b))
        ℒmon(bmono, eval_point)
    end

    return nothing
end

"""
    _build_rhs!(b, ℒrbf::Tuple, ℒmon::Tuple, data, eval_point, basis, mon, k)

Build the RHS matrix for multiple operators, one column each.
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

    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in 1:k
            b[i, j] = ℒ(eval_point, data[i])
        end
    end

    if basis.poly_deg > -1
        for (j, ℒ_op) in enumerate(ℒmon)
            bmono = view(b, (k + 1):size(b, 1), j)
            ℒ_op(bmono, eval_point)
        end
    end

    return nothing
end

# ============================================================================
# Stencil Assembly
# ============================================================================

"""
    _build_stencil!(λ, A, b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)

Assemble one complete stencil: build the collocation matrix, build the RHS, and solve
for the weights into the pre-allocated `λ` buffer.

Returns: view of the first k rows of `λ` (size k×num_ops).
"""
function _build_stencil!(
        λ,
        A,
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
    _solve_system!(λ, A, b)
    return _weight_view(λ, k)
end

# CPU: Symmetric → bunchkaufman (optimal for symmetric indefinite)
function _solve_system!(λ, A::Symmetric, b)
    return ldiv!(λ, bunchkaufman!(A, true), b)
end

# Dispatch helpers: Vector gets 1D view, Matrix gets 2D slice view
_weight_view(λ::AbstractVector, k) = view(λ, 1:k)
_weight_view(λ::AbstractMatrix, k) = view(λ, 1:k, :)
