# ------------------------------------------------------------------------------------------
# Algebra — in-type whenever both operands share the sparsity structure, so families of
# same-pattern matrices combine without densifying and the result aliases the same
# structure object. Mixing with CSC stays sparse (the generic AbstractMatrix fallback
# would silently densify to m × n).
# ------------------------------------------------------------------------------------------

function _check_same_structure(A::SellMatrix, B::SellMatrix)
    same_structure(A, B) || throw(
        ArgumentError(
            "SellMatrix operands have different sparsity structures; convert with `sparse` for general sparse arithmetic."
        )
    )
    return nothing
end

function Base.:+(A::SellMatrix, B::SellMatrix)
    _check_same_structure(A, B)
    return SellMatrix(A.vals + B.vals, A.structure)
end

function Base.:-(A::SellMatrix, B::SellMatrix)
    _check_same_structure(A, B)
    return SellMatrix(A.vals - B.vals, A.structure)
end

Base.:-(A::SellMatrix) = SellMatrix(-A.vals, A.structure)

Base.:+(A::SellMatrix, B::AbstractSparseMatrixCSC) = sparse(A) + B
Base.:+(A::AbstractSparseMatrixCSC, B::SellMatrix) = A + sparse(B)
Base.:-(A::SellMatrix, B::AbstractSparseMatrixCSC) = sparse(A) - B
Base.:-(A::AbstractSparseMatrixCSC, B::SellMatrix) = A - sparse(B)

Base.:*(α::Number, A::SellMatrix) = SellMatrix(α * A.vals, A.structure)
Base.:*(A::SellMatrix, α::Number) = α * A
Base.:/(A::SellMatrix, α::Number) = SellMatrix(A.vals / α, A.structure)

# Diagonal(d) * A scales logical row i by d[i]. At uniform C == 1 that is a column
# scaling of the k × m values matrix (broadcast); the general path scales row slots
# in place on a copy, so ghost-row padding stays zero.
function Base.:*(D::Diagonal, A::SellMatrix{T, 1}) where {T}
    length(D.diag) == size(A, 1) || throw(
        DimensionMismatch("Diagonal has $(length(D.diag)) rows, A has $(size(A, 1))")
    )
    A.structure.width > 0 || return _scale_rows(D.diag, A)
    return SellMatrix(values_matrix(A) .* reshape(D.diag, 1, :), A.structure)
end

function Base.:*(D::Diagonal, A::SellMatrix)
    length(D.diag) == size(A, 1) || throw(
        DimensionMismatch("Diagonal has $(length(D.diag)) rows, A has $(size(A, 1))")
    )
    return _scale_rows(D.diag, A)
end

@kernel function _scale_rows_kernel!(vals, @Const(sliceptr), @Const(d), C)
    i = @index(Global)
    base, w = _row_base(sliceptr, i, C)
    @inbounds di = d[i]
    @inbounds for l in 1:w
        vals[base + (l - 1) * C] *= di
    end
end

function _scale_rows(d::AbstractVector, A::SellMatrix{T, C}) where {T, C}
    S = A.structure
    P = promote_type(T, eltype(d))
    out = similar(A.vals, P)
    copyto!(out, A.vals)  # ghost-row and sentinel slots stay zero
    backend = KernelAbstractions.get_backend(A)
    if backend isa CPU
        @inbounds for i in 1:S.m
            base, w = _row_base(S.sliceptr, i, C)
            di = d[i]
            for l in 1:w
                out[base + (l - 1) * C] *= di
            end
        end
    else
        kernel! = _scale_rows_kernel!(backend)
        kernel!(out, S.sliceptr, d, C; ndrange = S.m)
        KernelAbstractions.synchronize(backend)
    end
    return SellMatrix(out, A.structure)
end

# A * Diagonal(d) scales the entry in column j by d[j] — elementwise over the storage,
# sentinel slots (column 0) kept at zero.
function Base.:*(A::SellMatrix{T, C}, D::Diagonal) where {T, C}
    length(D.diag) == size(A, 2) || throw(
        DimensionMismatch("Diagonal has $(length(D.diag)) columns, A has $(size(A, 2))")
    )
    S = A.structure
    d = D.diag
    P = promote_type(T, eltype(d))
    out = similar(A.vals, P)
    backend = KernelAbstractions.get_backend(A)
    if backend isa CPU
        Ti = eltype(S.colind)
        @inbounds for q in eachindex(S.colind)
            j = S.colind[q]
            out[q] = j == zero(Ti) ? zero(P) : A.vals[q] * d[Int(j)]
        end
    else
        kernel! = _scale_cols_kernel!(backend)
        kernel!(out, A.vals, S.colind, d; ndrange = length(out))
        KernelAbstractions.synchronize(backend)
    end
    return SellMatrix(out, A.structure)
end

@kernel function _scale_cols_kernel!(out, @Const(vals), @Const(colind), @Const(d))
    q = @index(Global)
    @inbounds j = Int(colind[q])
    @inbounds out[q] = j == 0 ? zero(eltype(out)) : vals[q] * d[j]
end

# Generic AbstractMatrix `\` would densify and LU-factorize; route through sparse instead
Base.:\(A::SellMatrix, b::AbstractVecOrMat) = sparse(A) \ b

Base.copy(A::SellMatrix) = SellMatrix(copy(A.vals), A.structure)

# The structure is frozen, so copyto! transfers values only — and refuses sources with
# a different sparsity structure rather than rewriting column indices (which may be
# aliased by other members of a same-structure family).
function Base.copyto!(dest::SellMatrix, src::SellMatrix)
    if size(dest) != size(src)
        throw(DimensionMismatch("destination and source SellMatrix differ in shape"))
    end
    _check_same_structure(dest, src)
    copyto!(dest.vals, src.vals)
    return dest
end

# Fast path valid because padding slots always store zero. Values are compared through
# `vec` so matrix- and vector-backed storage of the same structure compare equal.
function Base.:(==)(A::SellMatrix, B::SellMatrix)
    size(A) == size(B) || return false
    same_structure(A, B) && return vec(A.vals) == vec(B.vals)
    return sparse(A) == sparse(B)
end

function Base.isapprox(A::SellMatrix, B::SellMatrix; kwargs...)
    size(A) == size(B) || return false
    same_structure(A, B) && return isapprox(vec(A.vals), vec(B.vals); kwargs...)
    return isapprox(sparse(A), sparse(B); kwargs...)
end
