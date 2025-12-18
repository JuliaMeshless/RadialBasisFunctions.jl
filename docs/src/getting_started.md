# Getting Started

First, let's load the package along with the [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) package which we use for each data point

```@example overview
using RadialBasisFunctions
using StaticArrays
```

## Interpolation

Suppose we have a set of data ``\mathbf{x}`` where ``\mathbf{x}_i \in \mathbb{R}^2``, and we want to interpolate a function ``f:\mathbb{R}^2 \rightarrow \mathbb{R}``

```@example overview
f(x) = 2*x[1]^2 + 3*x[2]
x = [SVector{2}(rand(2)) for _ in 1:1000]
y = f.(x)
```

and now we can build the interpolator

```@example overview
interp = Interpolator(x, y)
```

and evaluate it at a new point

```@example overview
x_new = [rand(2) for _ in 1:5]
y_new = interp(x_new)
y_true = f.(x_new)
```

and compare the error

```@example overview
abs.(y_true .- y_new)
```

Wow! The error is numerically zero! Well... we set ourselves up for success here. `Interpolator` (along with `RadialBasisOperator`) has an optional argument to provide the type of radial basis including the degree of polynomial augmentation. The default basis is a cubic polyharmonic spline with 2nd degree polynomial augmentation (which the constructor is `PHS(3, poly_deg=2)`) and given the underlying function we are interpolating is a 2nd order polynomial itself, we are able to represent it exactly (up to machine precision). Let's see what happens when we only use 1st order polynomial augmentation

```@example overview
interp = Interpolator(x, y, PHS(3, poly_deg=1))
y_new = interp(x_new)
abs.(y_true .- y_new)
```

## Operators

This package also provides an API for operators. There is support for several built-in operators along with support for user-defined operators. Currently, we have implementations for

- partial derivative (1st and 2nd order)
- laplacian
- gradient

but we plan to add more in the future. Please make and issue or pull request for additional operators.

### Partial Derivative

We can take the same data as above and build a partial derivative operator with a similar construction as the interpolator. For the partial we need to specify the order of differentiation we want along with the dimension for which to take the partial. We can also supply some optional arguments such as the basis and number of points in the stencil. The function inputs order is `partial(x, order, dim, basis; k=5)`

```@example overview
df_x_rbf = partial(x, 1, 1)

# define exact
df_x(x) = 4*x[1]

# error
all(abs.(df_x.(x) .- df_x_rbf(y)) .< 1e-10)
```

### Laplacian

Building a laplacian operator is as easy as

```@example overview
lap_rbf = laplacian(x)

# define exact
lap(x) = 4

# error
all(abs.(lap.(x) .- lap_rbf(y)) .< 1e-8)
```

### Gradient

We can also retrieve the gradient. This is really just a convenience wrapper around `Partial`.

```@example overview
grad = gradient(x)

# define exacts
df_x(x) = 4*x[1]
df_y(x) = 3

# error
all(df_x.(x) .≈ grad(y)[1])
```

```@example overview
all(df_y.(x) .≈ grad(y)[2])
```

## Current Limitations

1. **Data format**: The package requires `Vector{AbstractVector}` input (not matrices). Each point must have inferrable dimension, e.g., `SVector{2,Float64}` from StaticArrays.jl. Matrix input support is planned.

2. **Global interpolation**: `Interpolator` currently uses all points globally. Local collocation support (like the operators use) is planned for future releases.
