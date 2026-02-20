# Getting Started

Data must be an `AbstractVector` of point vectors — each point needs an inferrable dimension (e.g., `SVector{2,Float64}` from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)).

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
x_new = [SVector{2}(rand(2)) for _ in 1:5]
y_new = interp(x_new)
y_true = f.(x_new)
```

and compare the error

```@example overview
abs.(y_true .- y_new)
```

The error is numerically zero because the default basis — `PHS(3; poly_deg=2)` — includes quadratic polynomial augmentation, which can represent our 2nd-order polynomial `f` exactly. Reducing the polynomial degree shows the effect:

```@example overview
interp = Interpolator(x, y, PHS(3; poly_deg=1))
y_new = interp(x_new)
abs.(y_true .- y_new)
```

## Operators

Operators compute RBF-FD weights for differentiation on scattered data. They're lazy — weights are built on first evaluation and cached.

### Partial Derivative

```@example overview
df_x_rbf = partial(x, 1, 1)

# define exact
df_x(x) = 4*x[1]

# error
all(abs.(df_x.(x) .- df_x_rbf(y)) .< 1e-10)
```

### Laplacian

```@example overview
lap_rbf = laplacian(x)

# define exact
lap(x) = 4

# error
all(abs.(lap.(x) .- lap_rbf(y)) .< 1e-8)
```

### Gradient / Jacobian

The `jacobian` function computes all partial derivatives. For scalar fields, this is the gradient.
The `gradient` function is a convenience alias for `jacobian`.

```@example overview
op = jacobian(x)  # or equivalently: gradient(x)
result = op(y)    # Matrix of size (N, dim)

# define exacts
df_x(x) = 4*x[1]
df_y(x) = 3

# error - access columns for each partial derivative
all(df_x.(x) .≈ result[:, 1])
```

```@example overview
all(df_y.(x) .≈ result[:, 2])
```

### Directional Derivative

Compute derivatives in any direction using `directional`. The direction can be constant or vary spatially:

```@example overview
using LinearAlgebra: normalize

# Constant direction (same for all points)
v = normalize([1.0, 1.0])
dir_op = directional(x, v)
result = dir_op(y)
typeof(result)
```

The direction can also vary per-point, useful for computing normal derivatives:

```@example overview
# Spatially-varying direction (e.g., radial directions)
normals = [normalize(collect(p)) for p in x]
normal_deriv = directional(x, normals)
typeof(normal_deriv(y))
```

### Custom Operators

Define your own differential operators using `custom`. The function should accept a basis and return a callable `(x, xc) -> value`. Here's an example that creates an interpolation-like operator:

```@example overview
# Custom operator that evaluates the basis function
op = custom(x, basis -> (x, xc) -> basis(x, xc))
typeof(op)
```

For more complex differential operators, use `operator algebra` (see below) to combine built-in operators.

### Regridding

Interpolate field values from one set of points to another using `regrid`:

```@example overview
# Target points (fine grid, different from original x)
x_fine = [SVector{2}(rand(2)) for _ in 1:500]

# Build regridding operator from x to x_fine
rg = regrid(x, x_fine)
y_fine = rg(y)
length(y_fine)
```

### Operator Algebra

Operators can be combined using `+` and `-`:

```@example overview
# Create individual operators
∂x = partial(x, 1, 1)
∂y = partial(x, 1, 2)

# Combine them: ∂f/∂x + ∂f/∂y
combined = ∂x + ∂y
result = combined(y)
typeof(result)
```

## Boundary Conditions (Hermite Interpolation)

For PDE applications, operators support Hermite interpolation with boundary conditions. This is useful when you need to enforce Dirichlet, Neumann, or Robin conditions at boundary nodes.

### Boundary Condition Types

- `Dirichlet()` - Value specified: ``u = g``
- `Neumann()` - Normal derivative specified: ``\partial u/\partial n = g``
- `Robin(α, β)` - Mixed condition: ``\alpha u + \beta \partial u/\partial n = g``
- `Internal()` - Interior point (no boundary condition)

### Example with Hermite Interpolation

```@example overview
using LinearAlgebra: norm

# Define boundary information
is_boundary = [norm(p) > 0.9 for p in x]  # Points near unit circle boundary
boundary_indices = findall(is_boundary)

# Create boundary conditions (Dirichlet on boundary)
bcs = [Dirichlet() for _ in boundary_indices]

# Normal vectors at boundary points
normals = [normalize(collect(x[i])) for i in boundary_indices]

# Build operator with Hermite interpolation
lap_hermite = laplacian(x; hermite=(
    is_boundary=is_boundary,
    bc=bcs,
    normals=normals
))
typeof(lap_hermite)
```

## Advanced: Virtual Operators

Virtual operators (`∂virtual`) use finite difference formulas on interpolated values at offset points. This can be useful for certain numerical schemes:

```@example overview
# Virtual partial derivative in x-direction with spacing Δ=0.01
virtual_dx = ∂virtual(x, 1, 0.01)
result = virtual_dx(y)
typeof(result)
```

## Current Limitations

1. **Data format**: The package requires `AbstractVector{<:AbstractVector}` input (not matrices). Each point must have inferrable dimension, e.g., `SVector{2,Float64}` from StaticArrays.jl. Matrix input support is planned.

2. **Global interpolation**: `Interpolator` currently uses all points globally. Local collocation support (like the operators use) is planned for future releases.

3. **GPU weight computation**: Operators and interpolators can be moved to GPU via `Adapt.jl` (e.g., `cu(operator)`) for evaluation, but weight computation (stencil assembly and solve) currently runs on CPU only. A GPU-compatible dense solver is needed for full GPU support ([#88](https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/88)).
