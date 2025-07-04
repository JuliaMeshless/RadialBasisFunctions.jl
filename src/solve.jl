function _build_weights(ℒ, op)
    data = op.data
    eval_points = op.eval_points
    adjl = op.adjl
    basis = op.basis
    return _build_weights(ℒ, data, eval_points, adjl, basis)
end

function _build_weights(ℒ, data, eval_points, adjl, basis)
    dim = length(first(data)) # dimension of data

    # build monomial basis and operator
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    return _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon)
end

function _build_weights(
    data, eval_points, adjl, basis, ℒrbf, ℒmon, mon; batch_size=10, device=CPU()
)
    TD = eltype(first(data))
    dim = length(first(data)) # dimension of data
    nmon = binomial(dim + basis.poly_deg, basis.poly_deg)
    k = length(first(adjl))  # number of data in influence/support domain
    sizes = (k, nmon)

    # allocate arrays to build sparse matrix
    Na = length(adjl)
    I = zeros(Int, k * Na)
    J = reduce(vcat, adjl)
    V = zeros(TD, k * Na, _num_ops(ℒrbf))

    # Calculate number of batches
    n_batches = ceil(Int, Na / batch_size)

    # Create kernel for building stencils in batches
    @kernel function build_stencils_kernel(
        I, J, V, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, k, batch_size, Na
    )
        # Get the batch index for this thread
        batch_idx = @index(Global)

        # Calculate the range of points for this batch
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, Na)

        # Pre-allocate work arrays for this thread
        n = k + nmon
        A = Symmetric(zeros(TD, n, n), :U)
        b = _prepare_b(ℒrbf, TD, n)

        # Process each point in the batch sequentially
        for i in start_idx:end_idx
            # Set row indices for sparse matrix
            for idx in 1:k
                I[(i - 1) * k + idx] = i
            end

            # Get data points in the influence domain
            local_data = [data[j] for j in adjl[i]]

            # Build stencil and store in global weight matrix
            stencil = _build_stencil!(
                A, b, ℒrbf, ℒmon, local_data, eval_points[i], basis, mon, k
            )

            # Store the stencil weights in the value array
            for op in axes(V, 2)
                for idx in 1:k
                    V[(i - 1) * k + idx, op] = stencil[idx, op]
                end
            end
        end
    end

    # Launch kernel with one thread per batch
    kernel = build_stencils_kernel(device)
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
        k,
        batch_size,
        Na;
        ndrange=n_batches,
        workgroupsize=1,
    )

    # Wait for kernel to complete
    KernelAbstractions.synchronize(device)

    # Create and return sparse matrix/matrices
    nrows = length(adjl)
    ncols = length(data)
    if size(V, 2) == 1
        return sparse(I, J, V[:, 1], nrows, ncols)
    else
        return ntuple(i -> sparse(I, J, V[:, i], nrows, ncols), size(V, 2))
    end
end

function _build_stencil!(
    A::Symmetric,
    b,
    ℒrbf,
    ℒmon,
    data::AbstractVector{TD},
    eval_point::TE,
    basis::B,
    mon::MonomialBasis,
    k::Int,
) where {TD,TE,B<:AbstractRadialBasis}
    # Reset matrices to zero
    #=
    fill!(parent(A), zero(TD))
    if b isa AbstractMatrix
        fill!(b, zero(TD))
    else
        fill!.(b, zero(TD))
    end
    =#

    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)
    return (A \ b)[1:k, :]
end

function _build_collocation_matrix!(
    A::Symmetric, data::AbstractVector, basis::B, mon::MonomialBasis{Dim,Deg}, k::K
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg}
    # radial basis section
    AA = parent(A)
    N = size(A, 2)
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = basis(data[i], data[j])
    end

    # monomial augmentation
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            mon(a, data[i])
        end
    end

    return nothing
end

function _build_rhs!(
    b, ℒrbf, ℒmon, data::AbstractVector{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    # radial basis section
    @inbounds for i in eachindex(data)
        b[i] = ℒrbf(eval_point, data[i])
    end

    # monomial augmentation
    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)
        ℒmon(bmono, eval_point)
    end

    return nothing
end

function _build_rhs!(
    b, ℒrbf::Tuple, ℒmon::Tuple, data::AbstractVector{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, ℒmon must have the same length"
    # radial basis section
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in eachindex(data)
            b[i, j] = ℒ(eval_point, data[i])
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
