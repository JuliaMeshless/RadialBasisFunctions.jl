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
x = rand(SVector{2,Float64}, 1000)
y = f.(x)
nothing # hide
```

and now we can build the interpolator

```@example overview
interp = Interpolator(x, y)
```

and evaluate it at a new point

```@example overview
x_new = rand(SVector{2,Float64}, 5)
y_new = interp(x_new)
y_true = f.(x_new)
nothing # hide
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

Operators compute RBF-FD weights for differentiation on scattered data. Weights are built at construction and cached; invalidate the cache to trigger recomputation.

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
normals = map(normalize, x)
normal_deriv = directional(x, normals)
typeof(normal_deriv(y))
```

For the common case of differentiating along outward normals, [`normal_derivative`](@ref)
wraps `directional` and normalizes the vectors you pass it.

### Custom & PDE Operators

Beyond the built-ins, the [`@operator`](@ref) macro lets you write PDE operators in
mathematical notation — Helmholtz, diffusion, advection-diffusion — and call them directly
with data points. See [Building PDE Operators](@ref) for the recipes, and
[Operators & Type Hierarchy](@ref) for an in-depth guide to the operator system.

### Regridding

Interpolate field values from one set of points to another using `regrid`:

```@example overview
# Target points (fine grid, different from original x)
x_fine = rand(SVector{2,Float64}, 500)

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

## Enforcing Boundary Conditions

For PDE applications, operators support Hermite interpolation to enforce Dirichlet,
Neumann, or Robin conditions at boundary nodes. See [Boundary Conditions](@ref) for the
condition types and the `hermite` keyword.

## Where to Next

- [Operators & Type Hierarchy](@ref) — the operator system, rank semantics, and virtual operators
- [Building PDE Operators](@ref) — assemble Helmholtz, diffusion, and advection-diffusion operators
- [Quick Reference](@ref) — data formats, basis options, and operator constructors at a glance
- [Convergence & Parameter Selection](@ref) — how to pick a basis, `poly_deg`, and stencil size

## Current Limitations

1. **Global interpolation**: `Interpolator` currently uses all points globally. Local collocation support (like the operators use) is planned for future releases.

2. **GPU weight computation**: weight computation (stencil assembly and solve) currently runs on CPU only; a GPU-compatible dense solver is needed for full GPU support ([#88](https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/88)). Built operators *can* be moved to GPU for evaluation — see [GPU Evaluation](@ref) in the Quick Reference.
