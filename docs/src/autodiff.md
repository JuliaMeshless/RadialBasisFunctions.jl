# Automatic Differentiation

RadialBasisFunctions.jl supports automatic differentiation (AD) through two package extensions:

- **Mooncake.jl** - Reverse-mode AD with support for mutation
- **Enzyme.jl** - Native EnzymeRules for high-performance reverse-mode AD

All examples use [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl) which provides a unified API over different AD backends.

## Compatibility

Enzyme.jl currently has known issues with Julia 1.12+. If you encounter problems, use Julia < 1.12 or switch to Mooncake.jl.

See: [Enzyme.jl#2699](https://github.com/EnzymeAD/Enzyme.jl/issues/2699)

## Implementation Status

Both backends have native AD rule implementations. Enzyme.jl uses `EnzymeRules` (`augmented_primal`/`reverse`) and Mooncake.jl uses native `rrule!!` with `@is_primitive`.

## Differentiating Through Operators

The most common use case is differentiating a loss function with respect to field values while keeping the operator fixed. Create the operator once outside the loss function, then differentiate through its application.

```@example autodiff
using RadialBasisFunctions
using StaticArrays
import DifferentiationInterface as DI
import Mooncake

# Create points and operator (outside loss function)
N = 49
points = [SVector{2}(0.1 + 0.8 * i / 7, 0.1 + 0.8 * j / 7) for i in 1:7 for j in 1:7]
values = sin.(getindex.(points, 1)) .+ cos.(getindex.(points, 2))

lap = laplacian(points)

# Loss function: minimize squared Laplacian
function loss(v)
    result = lap(v)
    return sum(result .^ 2)
end

# Compute gradient using DifferentiationInterface
backend = DI.AutoMooncake(; config=nothing)  # also supports DI.AutoEnzyme()
grad = DI.gradient(loss, backend, values)
grad[1:5]  # Show first 5 gradient values
```

This works with any operator type:

```@example autodiff
# Gradient operator (vector-valued)
∇f = gradient(points)

function loss_grad(v)
    result = ∇f(v)
    return sum(result .^ 2)
end

grad = DI.gradient(loss_grad, backend, values)
grad[1:5]
```

```@example autodiff
# Partial derivative operator
∂x = partial(points, 1, 1)

function loss_partial(v)
    result = ∂x(v)
    return sum(result .^ 2)
end

grad = DI.gradient(loss_partial, backend, values)
grad[1:5]
```

## Differentiating Through Interpolators

When differentiating through interpolation, the `Interpolator` must be constructed inside the loss function since changing the input values changes the interpolation weights.

!!! note
    Differentiating through `Interpolator` construction currently requires Mooncake.
    Enzyme does not yet support this code path via DifferentiationInterface.

```@example autodiff
N_interp = 30
points_interp = [SVector{2}(0.5 + 0.4 * cos(2π * i / N_interp), 0.5 + 0.4 * sin(2π * i / N_interp)) for i in 1:N_interp]
values_interp = sin.(getindex.(points_interp, 1))
eval_points = [SVector{2}(0.5, 0.5), SVector{2}(0.6, 0.6)]

# Loss function - must rebuild interpolator inside
function loss_interp(v)
    interp = Interpolator(points_interp, v)
    result = interp(eval_points)
    return sum(result .^ 2)
end

grad = DI.gradient(loss_interp, backend, values_interp)
grad[1:5]
```

## Differentiating Basis Functions Directly

For low-level control, you can differentiate basis function evaluations directly. This is useful for custom applications or understanding the underlying derivatives.

```@example autodiff
x = [0.5, 0.5]
xi = [0.3, 0.4]

# PHS basis
phs = PHS(3)
function loss_phs(xv)
    return phs(xv, xi)^2
end

grad = DI.gradient(loss_phs, backend, x)
```

All basis types are supported:

```@example autodiff
# IMQ basis
imq = IMQ(1.0)
function loss_imq(xv)
    return imq(xv, xi)^2
end

grad = DI.gradient(loss_imq, backend, x)
```

```@example autodiff
# Gaussian basis
gauss = Gaussian(1.0)
function loss_gauss(xv)
    return gauss(xv, xi)^2
end

grad = DI.gradient(loss_gauss, backend, x)
```

## Differentiating Weight Construction

For advanced use cases like mesh optimization or shape parameter tuning, you can differentiate through the weight construction process using the internal `_build_weights` function.

```@example autodiff
# Using Mooncake for weight construction
N_weights = 25
points_weights = [SVector{2}(0.1 + 0.8 * i / 5, 0.1 + 0.8 * j / 5) for i in 1:5 for j in 1:5]
adjl = RadialBasisFunctions.find_neighbors(points_weights, 10)
basis = PHS(3; poly_deg=2)
ℒ = Partial(1, 1)  # First derivative in x

# Loss function w.r.t. point positions
function loss_weights(pts)
    pts_vec = [SVector{2}(pts[2*i-1], pts[2*i]) for i in 1:N_weights]
    W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
    return sum(W.nzval .^ 2)
end

pts_flat = vcat([collect(p) for p in points_weights]...)
grad = DI.gradient(loss_weights, backend, pts_flat)
grad[1:6]  # Gradients for first 3 points (x,y pairs)
```

This also works with the Laplacian operator and different basis types:

```@example autodiff
ℒ_lap = Laplacian()
basis_imq = IMQ(1.0; poly_deg=2)

function loss_weights_lap(pts)
    pts_vec = [SVector{2}(pts[2*i-1], pts[2*i]) for i in 1:N_weights]
    W = RadialBasisFunctions._build_weights(ℒ_lap, pts_vec, pts_vec, adjl, basis_imq)
    return sum(W.nzval .^ 2)
end

grad = DI.gradient(loss_weights_lap, backend, pts_flat)
grad[1:6]
```

## Supported Components

| Component | Enzyme | Mooncake |
|-----------|:------:|:--------:|
| Operator evaluation (`op(values)`) | ✓ | ✓ |
| Interpolator construction | ✓* | ✓ |
| Interpolator evaluation | ✓ | ✓ |
| Basis functions (PHS, IMQ, Gaussian) | ✓ | ✓ |
| Weight construction (`_build_weights`) | ✓ | ✓ |
| Shape parameter (ε) differentiation | ✓ | ✓ |

*Enzyme supports Interpolator construction via native `autodiff` but may fail through DifferentiationInterface due to `factorize` limitations. Use Mooncake for this use case.

## Using Enzyme Backend

Switch to Enzyme by changing the backend (requires Julia < 1.12):

```julia
import DifferentiationInterface as DI
import Enzyme

backend = DI.AutoEnzyme()
grad = DI.gradient(loss, backend, values)
```
