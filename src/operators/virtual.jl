# convienience constructors
"""
    function ∂virtual(data, eval_points, dim, Δ, basis; k=autoselect_k(data, basis))

Builds a virtual `RadialBasisOperator` whichi will be evaluated at `eval_points` where the
operator is the partial derivative with respect to `dim`. Virtual operators interpolate the
data to structured points at a distance `Δ` for which standard finite difference formulas
can be applied.
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
    N = length(first(data))
    dx = zeros(N)
    dx[dim] = Δ

    self = regrid(data, eval_points, basis; k = k)
    update_weights!(self)

    return if backward
        li = regrid(data, eval_points .- Ref(dx), basis; k = k)
        update_weights!(li)
        w = (self.weights .- li.weights) ./ Δ
        x -> w * x
    else # forward difference
        ri = regrid(data, eval_points .+ Ref(dx), basis; k = k)
        update_weights!(ri)
        w = (ri.weights .- self.weights) ./ Δ
        x -> w * x
    end
end

"""
    function ∂virtual(data, dim, Δ, basis; k=autoselect_k(data, basis))

Builds a virtual `RadialBasisOperator` whichi will be evaluated at the input points (`data`)
where the operator is the partial derivative with respect to `dim`. Virtual operators
interpolate the data to structured points at a distance `Δ` for which standard finite
difference formulas can be applied.
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
