# Quick Reference

## Data Format

Data **must** be an `AbstractVector` of point vectors, not a `Matrix`:

```julia
using StaticArrays

# CORRECT: Vector of static vectors
points = [SVector{2}(rand(2)) for _ in 1:100]

# Converting from matrix
matrix = rand(100, 2)
points = [SVector{2}(row) for row in eachrow(matrix)]

# 3D points
points_3d = [SVector{3}(rand(3)) for _ in 1:100]
```

## Basis Functions

| Type | Constructor | Formula | Shape Parameter |
|------|-------------|---------|-----------------|
| Polyharmonic Spline | `PHS(n)` | $r^n$ | None (scale-free) |
| Inverse Multiquadric | `IMQ(ε)` | $\frac{1}{\sqrt{(\varepsilon r)^2 + 1}}$ | Optional (default: 1) |
| Gaussian | `Gaussian(ε)` | $e^{-(\varepsilon r)^2}$ | Optional (default: 1) |

### PHS Orders

| Order | Constructor | Smoothness | Use Case |
|-------|-------------|------------|----------|
| 1 | `PHS(1)` | C⁰ | Rough data |
| 3 | `PHS(3)` | C² | General purpose (default) |
| 5 | `PHS(5)` | C⁴ | Smooth functions |
| 7 | `PHS(7)` | C⁶ | Very smooth functions |

### Polynomial Augmentation

```julia
# Default: quadratic (poly_deg=2)
basis = PHS(3)

# Custom polynomial degree
basis = PHS(3; poly_deg=0)   # Constant only
basis = PHS(3; poly_deg=1)   # Linear
basis = PHS(3; poly_deg=3)   # Cubic
basis = PHS(3; poly_deg=-1)  # No polynomial (not recommended)
```

## Operators

### Creating Operators

```julia
using RadialBasisFunctions, StaticArrays

points = [SVector{2}(rand(2)) for _ in 1:100]

# Scalar field operators
lap = laplacian(points)           # ∇²f (scalar output)
grad = gradient(points)           # ∇f (vector output)
∂x = partial(points, 1, 1)        # ∂f/∂x (1st order, dim 1)
∂²y = partial(points, 2, 2)       # ∂²f/∂y² (2nd order, dim 2)
∂xy = mixed_partial(points, 1, 2) # ∂²f/∂x∂y
H = hessian(points)               # Full Hessian matrix
∂v = directional(points, v)       # ∇f·v
∂n = normal_derivative(points, normals)  # ∇f·n̂
op = @operator ∇ ⋅ (κ * ∇)
diff = custom(points, op)                      # ∇⋅(κ∇f)

# Vector field operators (input: N×D matrix)
div_op = divergence(points)       # ∇⋅u (scalar output)
curl_op = curl(points)            # ∇×u (2D: scalar, 3D: vector)

# Interpolation operators
rg = regrid(source, target)       # Local interpolation
```

### Applying Operators

```julia
values = sin.(getindex.(points, 1))

# Apply to data
lap_values = lap(values)          # Vector of scalars
grad_values = grad(values)        # Matrix (N × dim)
```

### Common Options

```julia
# Custom basis
lap = laplacian(points; basis=PHS(5; poly_deg=3))

# Custom stencil size
lap = laplacian(points; k=30)

# Different evaluation points
lap = laplacian(points; eval_points=other_points)

# Precomputed neighbors
adjl = find_neighbors(points, k)
lap = laplacian(points; adjl=adjl)
```

## Global Interpolation

```julia
# Create interpolator (uses all points)
interp = Interpolator(points, values)

# Evaluate at single point
result = interp(SVector(0.5, 0.5))

# Evaluate at multiple points
results = interp(new_points)
```

## Hermite Boundary Conditions

For PDE problems with boundary conditions:

```julia
# Prepare boundary data
is_boundary = [is_on_boundary(p) for p in points]
bc = [Dirichlet() for _ in points]  # or Neumann(), Robin(α, β)
normals = [compute_normal(p) for p in points]

# Create operator with Hermite interpolation
lap = laplacian(
    points;
    hermite=(is_boundary=is_boundary, bc=bc, normals=normals)
)
```

### Boundary Condition Types

| Type | Constructor | Meaning |
|------|-------------|---------|
| Dirichlet | `Dirichlet()` | Fixed value |
| Neumann | `Neumann()` | Fixed normal derivative |
| Robin | `Robin(α, β)` | $\alpha u + \beta \frac{\partial u}{\partial n}$ |

## GPU Evaluation

Weight computation currently runs on CPU. Once built, operators can be moved to GPU for fast evaluation:

```julia
using CUDA, Adapt

# Build operator on CPU
lap = laplacian(points)

# Move to GPU for evaluation
lap_gpu = cu(lap)
values_gpu = cu(values)
result_gpu = lap_gpu(values_gpu)
```

Full GPU weight computation is planned — see [#88](https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/88).

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `MethodError: no method matching` | Matrix data | Use `Vector{SVector}` |
| `n must be 1, 3, 5, or 7` | Invalid PHS order | Use odd integer ≤ 7 |
| `Shape parameter should be > 0` | Negative ε | Use positive ε (0.1-10.0) |
| `SingularException` | Duplicate points or bad stencil | Remove duplicates, adjust k |

## Performance Tips

1. **Reuse adjacency lists** for multiple operators on same points
2. **Use StaticArrays** (`SVector`) for best performance
3. **Batch operations** - create operator once, apply many times
4. **GPU for large problems** — build operators on CPU, then `cu(operator)` for GPU evaluation
