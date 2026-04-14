# ============================================================================
# Operator Traits
# ============================================================================
#
# Trait functions that describe operator properties for dispatch, validation,
# and introspection. These enable better error messages at construction time
# and allow generic code to query operator capabilities.

"""
    output_rank(op::AbstractOperator{N}) -> Int

Tensor rank added to the output by this operator. Returns `N` from the type parameter.
"""
output_rank(::AbstractOperator{N}) where {N} = N

"""
    requires_vector_input(op) -> Bool

Whether the operator requires a vector field (matrix) as input rather than a scalar field (vector).
Operators like [`Divergence`](@ref), [`Curl`](@ref), [`StrainRate`](@ref), and
[`RotationRate`](@ref) act on vector fields and will error on scalar input.
"""
requires_vector_input(::AbstractOperator) = false
requires_vector_input(::Divergence) = true
requires_vector_input(::Curl) = true
requires_vector_input(::StrainRate) = true
requires_vector_input(::RotationRate) = true

"""
    is_symmetric(op) -> Bool

Whether the operator produces a symmetric output tensor. Symmetric operators can
exploit storage savings (e.g., only storing upper-triangular entries).
"""
is_symmetric(::AbstractOperator) = false
is_symmetric(::Hessian) = true
is_symmetric(::StrainRate) = true

"""
    is_antisymmetric(op) -> Bool

Whether the operator produces an anti-symmetric output tensor (Aᵢⱼ = −Aⱼᵢ).
"""
is_antisymmetric(::AbstractOperator) = false
is_antisymmetric(::RotationRate) = true

"""
    is_self_adjoint(op) -> Bool

Whether the operator is self-adjoint (⟨ℒu, v⟩ = ⟨u, ℒv⟩). Self-adjoint operators
produce symmetric weight matrices, which matters for solver selection and
eigenvalue problems.
"""
is_self_adjoint(::AbstractOperator) = false
is_self_adjoint(::Laplacian) = true
is_self_adjoint(::Identity) = true
is_self_adjoint(::Regrid) = false

"""
    derivative_order(op) -> Int

Total order of differentiation. Useful for estimating required polynomial degree
and stencil size.
"""
derivative_order(::Identity) = 0
derivative_order(::Regrid) = 0
derivative_order(op::Partial) = op.order
derivative_order(::MixedPartial) = 2
derivative_order(::Laplacian) = 2
derivative_order(::Jacobian) = 1
derivative_order(::Hessian) = 2
derivative_order(::Directional) = 1
derivative_order(::Divergence) = 1
derivative_order(::Curl) = 1
derivative_order(::StrainRate) = 1
derivative_order(::RotationRate) = 1
derivative_order(s::ScaledOperator) = derivative_order(s.op)
derivative_order(::Custom) = -1  # unknown
