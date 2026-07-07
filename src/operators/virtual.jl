"""
    VirtualPartial{T<:Real} <: AbstractOperator{0}

Operator for the first partial derivative with respect to `dim`, computed virtually:
the data is interpolated (regridded) to points shifted by `Δ` along `dim` and a
one-sided finite difference is taken between the shifted and unshifted interpolants.
"""
struct VirtualPartial{T <: Real} <: AbstractOperator{0}
    dim::Int
    Δ::T
    backward::Bool
end

"""
    function ∂virtual(data, eval_points, dim, Δ, basis; backward=false, k=autoselect_k(data, basis))

Builds a virtual `RadialBasisOperator` which will be evaluated at `eval_points` where the
operator is the partial derivative with respect to `dim`. Virtual operators interpolate the
data to structured points at a distance `Δ` for which standard finite difference formulas
can be applied. The result is a standard [`RadialBasisOperator`](@ref), so weight caching
and [`update_weights!`](@ref) come for free.

Defaults to a forward difference (`backward=false`). Note the convenience form without
`eval_points` instead defaults to a backward difference (`backward=true`).
"""
function ∂virtual(
        data::AbstractVector,
        eval_points::AbstractVector,
        dim,
        Δ,
        basis::B = PHS(3; poly_deg = 2);
        backward = false,
        k::T = autoselect_k(data, basis),
    ) where {T <: Int, B <: AbstractRadialBasis}
    return RadialBasisOperator(
        VirtualPartial(dim, Δ, backward), data; eval_points = eval_points, basis = basis, k = k
    )
end

"""
    function ∂virtual(data, dim, Δ, basis; backward=true, k=autoselect_k(data, basis))

Builds a virtual `RadialBasisOperator` which will be evaluated at the input points (`data`)
where the operator is the partial derivative with respect to `dim`. Virtual operators
interpolate the data to structured points at a distance `Δ` for which standard finite
difference formulas can be applied. The result is a standard [`RadialBasisOperator`](@ref),
so weight caching and [`update_weights!`](@ref) come for free.

Defaults to a backward difference (`backward=true`), unlike the form with explicit
`eval_points`, which defaults to forward (`backward=false`).
"""
function ∂virtual(
        data::AbstractVector,
        dim,
        Δ,
        basis::B = PHS(3; poly_deg = 2);
        backward = true,
        k::T = autoselect_k(data, basis),
    ) where {T <: Int, B <: AbstractRadialBasis}
    return ∂virtual(data, data, dim, Δ, basis; backward = backward, k = k)
end

# Custom _build_weights: difference of two Regrid weight builds. Both point sets find
# their neighbors from the current data (adjl only supplies the stencil size) so that
# rebuilds via update_weights! stay consistent after in-place data mutation.
function _build_weights(ℒ::VirtualPartial, data, eval_points, adjl, basis; device = CPU())
    N = length(first(data))
    dx = zeros(eltype(first(data)), N)
    dx[ℒ.dim] = ℒ.Δ

    k = length(first(adjl))
    self_adjl = find_neighbors(data, eval_points, k)
    self = _build_weights(Regrid(), data, eval_points, self_adjl, basis; device = device)
    shifted = ℒ.backward ? eval_points .- Ref(dx) : eval_points .+ Ref(dx)
    shifted_adjl = find_neighbors(data, shifted, k)
    other = _build_weights(Regrid(), data, shifted, shifted_adjl, basis; device = device)

    return if ℒ.backward
        (self .- other) ./ ℒ.Δ
    else # forward difference
        (other .- self) ./ ℒ.Δ
    end
end

function _build_weights(
        ::VirtualPartial,
        data::AbstractVector,
        eval_points::AbstractVector,
        adjl::AbstractVector,
        basis::AbstractRadialBasis,
        is_boundary::Vector{Bool},
        boundary_conditions::Vector{<:BoundaryCondition},
        normals::Vector{<:AbstractVector};
        device = CPU(),
    )
    throw(
        ArgumentError(
            "VirtualPartial does not support Hermite boundary conditions. Use partial " *
                "with hermite=... instead.",
        ),
    )
end

# pretty printing
function print_op(op::VirtualPartial)
    return "virtual ∂f/∂x$(op.dim) ($(op.backward ? "backward" : "forward") difference, Δ = $(op.Δ))"
end
