using RadialBasisFunctions, StaticArrays, Mooncake, Test
include("test/extensions/ad_test_utils.jl")

N = 9
points = [SVector{2}(0.1 + 0.8 * i / 3, 0.1 + 0.8 * j / 3) for i in 1:3 for j in 1:3]
adjl = RadialBasisFunctions.find_neighbors(points, 8)
basis = PHS(3; poly_deg = 2)
pts_flat = vcat([collect(p) for p in points]...)

function loss_elasticity_assembly(pts)
    p = [SVector{2}(pts[2i - 1], pts[2i]) for i in 1:N]
    W_d2x = RadialBasisFunctions._build_weights(Partial(2, 1), p, p, adjl, basis)
    W_d2y = RadialBasisFunctions._build_weights(Partial(2, 2), p, p, adjl, basis)
    W_d2xy = RadialBasisFunctions._build_weights(MixedPartial(1, 2), p, p, adjl, basis)
    W_dx = RadialBasisFunctions._build_weights(Partial(1, 1), p, p, adjl, basis)
    W_dy = RadialBasisFunctions._build_weights(Partial(1, 2), p, p, adjl, basis)
    return sum(W_d2x.nzval .^ 2) + sum(W_d2y.nzval .^ 2) + sum(W_d2xy.nzval .^ 2) +
           sum(W_dx.nzval .^ 2) + sum(W_dy.nzval .^ 2)
end

try
    rule = Mooncake.build_rrule(loss_elasticity_assembly, pts_flat)
    _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_elasticity_assembly, pts_flat)
    validate_gradient(dpts, loss_elasticity_assembly, pts_flat; rtol = 1.0e-3)
    println("SUCCESS")
catch e
    println("ERROR:")
    showerror(stdout, e)
    println()
    # Print first 30 lines of backtrace
    bt = catch_backtrace()
    for (i, frame) in enumerate(stacktrace(bt))
        i > 30 && break
        println(frame)
    end
end
