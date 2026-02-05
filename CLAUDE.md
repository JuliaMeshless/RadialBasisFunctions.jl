# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RadialBasisFunctions.jl is a Julia package for radial basis function (RBF) interpolation and differential operators. It supports various RBF types (Polyharmonic Splines, Inverse Multiquadric, Gaussian) with polynomial augmentation for solving PDEs and data interpolation problems.

## Development Commands

### Testing
```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run specific test files
julia --project=test test/runtests.jl

# Run tests for specific components
julia --project=test test/basis/polyharmonic_spline.jl
julia --project=test test/operators/gradient.jl

# Run tests in development mode (activating the main project)
julia --project=. test/runtests.jl
```

### Documentation
```bash
# Build documentation locally
julia --project=docs docs/make.jl

# Run doctests only
julia --project=docs -e "using Documenter: DocMeta, doctest; using RadialBasisFunctions; DocMeta.setdocmeta!(RadialBasisFunctions, :DocTestSetup, :(using RadialBasisFunctions); recursive=true); doctest(RadialBasisFunctions)"
```

### Benchmarking
```bash
# Run benchmarks
julia --project=benchmark benchmark/benchmarks.jl
```

## Architecture Overview

### Core Components

1. **Basis Functions** (`src/basis/`): Abstract types and concrete implementations
   - `AbstractRadialBasis` - Base type for all RBF implementations
   - Specific RBFs: PHS (Polyharmonic Splines), IMQ (Inverse Multiquadric), Gaussian
   - `MonomialBasis` - Polynomial augmentation support

2. **Operators** (`src/operators/`): Differential operators built on RBFs
   - `RadialBasisOperator` - Main operator type with lazy weight computation
   - Specific operators: `Partial`, `Jacobian`, `Laplacian`, `Directional`, `Custom`
   - `operator_algebra.jl` - Composition and algebraic operations on operators
   - Virtual operators for performance optimization

3. **Solve System** (`src/solve/`): Core weight computation organized in 4 layers
   - `api.jl` - Entry points and routing (`_build_weights()` functions)
   - `execution.jl` - Parallel execution via KernelAbstractions.jl, memory allocation, batch processing
   - `assembly.jl` - Pure mathematical operations (collocation matrix, RHS, stencil assembly)
   - `types.jl` - Shared data structures (boundary conditions, stencil classification, operator traits)
   - Hermite interpolation with boundary conditions via multiple dispatch

4. **Interpolation** (`src/interpolation.jl`): 
   - `Interpolator` type for global interpolation (uses all data points)
   - Supports scattered data interpolation

5. **Utilities** (`src/utils.jl`):
   - `find_neighbors()` - k-nearest neighbor search using NearestNeighbors.jl
   - `reorder_points!()` - Point ordering utilities

### Key Design Patterns

- **Lazy Evaluation**: Operators compute weights only when needed, with caching
- **GPU Support**: KernelAbstractions.jl enables GPU computation for weight building
- **Modular Design**: Basis functions, operators, and solvers are decoupled
- **Type System**: Heavy use of parametric types for performance
- **Neighbor Search**: Local support domains via k-nearest neighbors for efficiency

### Data Requirements

The package requires `Vector{AbstractVector}` input format (not matrices). Each point must be a vector type with inferrable dimension (e.g., `SVector{2,Float64}` from StaticArrays.jl).

### Performance Considerations

- Stencil weights are computed in batches to manage memory
- Operators cache weights and check validity before recomputation
- GPU kernels are used for parallel weight computation
- Local neighborhoods (k-nearest neighbors) reduce computational complexity

## Key Files for Understanding

- `src/RadialBasisFunctions.jl` - Main module with exports and precompilation
- `src/solve/assembly.jl` - Pure mathematical operations for weight computation
- `src/solve/execution.jl` - GPU/CPU parallel execution kernels
- `src/solve/api.jl` - Entry points for weight building
- `src/operators/operators.jl:10-31` - Main operator type definition
- `src/basis/basis.jl` - Abstract basis type hierarchy
- `docs/src/internals.md` - Detailed solve system architecture and pseudocode

## Important Development Notes

- The package requires Julia 1.10+ (see Project.toml compatibility)
- Uses KernelAbstractions.jl for GPU/CPU parallelization
- Data must be in `Vector{AbstractVector}` format (not matrices) - each point needs inferrable dimension (e.g., `SVector{2,Float64}`)
- **Autodiff examples**: Always use DifferentiationInterface.jl for AD examples in docs and tests. This provides a unified interface over Enzyme.jl and Mooncake.jl backends.

---

## Common Usage Patterns

### Basic Interpolation
```julia
using RadialBasisFunctions, StaticArrays

# Data MUST be Vector{AbstractVector}, not a Matrix
points = [SVector{2}(rand(2)) for _ in 1:100]
values = sin.(getindex.(points, 1))  # scalar values at each point

# Create interpolator (uses all points globally)
interp = Interpolator(points, values)

# Evaluate at new points
new_point = SVector(0.5, 0.5)
result = interp(new_point)

# Evaluate at multiple points
new_points = [SVector{2}(rand(2)) for _ in 1:50]
results = interp(new_points)
```

### Differential Operators
```julia
# Create operators (weights computed lazily on first use)
lap = laplacian(points)           # ∇²f
grad = gradient(points)           # ∇f (returns Nx2 matrix)
∂x = partial(points, 1, 1)        # ∂f/∂x₁ (first derivative, dimension 1)
∂²y = partial(points, 2, 2)       # ∂²f/∂x₂² (second derivative, dimension 2)

# Apply to data
lap_values = lap(values)
grad_values = grad(values)        # Each row is [∂f/∂x, ∂f/∂y]
∂x_values = ∂x(values)
```

### Custom Basis Selection
```julia
# PHS - no shape parameter needed (most robust)
basis = PHS(3)                    # r³ (cubic, default)
basis = PHS(5; poly_deg=3)        # r⁵ with cubic polynomial augmentation

# IMQ - requires shape parameter tuning
basis = IMQ(1.0)                  # ε=1.0, good starting point
basis = IMQ(0.5; poly_deg=3)      # smaller ε = flatter basis

# Gaussian - requires shape parameter tuning
basis = Gaussian(1.0)             # ε=1.0

# Use custom basis in operators
lap = laplacian(points; basis=PHS(5; poly_deg=3))
```

### Stencil Size Control
```julia
# k = number of nearest neighbors used per stencil
# Default: auto-selected based on polynomial degree

# Manual override
lap = laplacian(points; k=30)     # Use 30 nearest neighbors

# Precompute neighbors for reuse
using NearestNeighbors
adjl = find_neighbors(points, 30)
lap = laplacian(points; adjl=adjl)
grad = gradient(points; adjl=adjl)  # Reuse same neighbors
```

### Regridding (Interpolation Between Point Sets)
```julia
source_points = [SVector{2}(rand(2)) for _ in 1:100]
target_points = [SVector{2}(rand(2)) for _ in 1:500]
source_values = sin.(getindex.(source_points, 1))

# Build regrid operator
rg = regrid(source_points, target_points)

# Transfer data from source to target
target_values = rg(source_values)
```

---

## Troubleshooting FAQ

### Data Format Errors

**Problem:** `MethodError: no method matching...` when creating operators

**Cause:** Data is in matrix format instead of `Vector{AbstractVector}`

**Fix:**
```julia
# WRONG
points = rand(100, 2)  # Matrix

# CORRECT
using StaticArrays
points = [SVector{2}(rand(2)) for _ in 1:100]

# Converting from matrix
matrix_data = rand(100, 2)
points = [SVector{2}(row) for row in eachrow(matrix_data)]
```

### Invalid PHS Order

**Problem:** `ArgumentError: n must be 1, 3, 5, or 7`

**Cause:** PHS order must be odd and ≤ 7

**Fix:**
```julia
# WRONG
PHS(2)  # even number
PHS(9)  # too high

# CORRECT
PHS(1)  # linear (least smooth)
PHS(3)  # cubic (default, good balance)
PHS(5)  # quintic (smoother)
PHS(7)  # septic (smoothest)
```

### Shape Parameter Errors (IMQ/Gaussian)

**Problem:** `ArgumentError: Shape parameter should be > 0`

**Cause:** ε must be positive

**Fix:**
```julia
# WRONG
IMQ(-1.0)
Gaussian(0.0)

# CORRECT
IMQ(1.0)      # Typical range: 0.1 to 10.0
Gaussian(0.5) # Smaller ε = wider basis function
```

### Poor Accuracy / Oscillations

**Causes and Fixes:**

1. **Stencil too small:** Increase `k`
   ```julia
   lap = laplacian(points; k=50)  # Try larger stencils
   ```

2. **Polynomial degree too low:** Increase `poly_deg`
   ```julia
   basis = PHS(3; poly_deg=4)  # Higher polynomial degree
   ```

3. **Wrong basis for problem:** Try different basis
   ```julia
   # For very smooth functions, try higher-order PHS
   basis = PHS(5; poly_deg=4)
   ```

4. **Shape parameter ill-suited (IMQ/Gaussian):** Tune ε
   ```julia
   # Smaller ε for smoother interpolation
   basis = IMQ(0.1)
   ```

### Singular System / Ill-Conditioned Matrix

**Problem:** `SingularException` or poor numerical results

**Causes and Fixes:**

1. **Duplicate or near-duplicate points:** Remove duplicates
   ```julia
   using NearestNeighbors
   # Check for points closer than tolerance
   tree = KDTree(points)
   # ... remove near-duplicates
   ```

2. **Stencil too large for local point density:** Reduce `k`
   ```julia
   lap = laplacian(points; k=20)
   ```

3. **Polynomial degree too high for stencil size:** Reduce `poly_deg`
   ```julia
   basis = PHS(3; poly_deg=1)  # Linear polynomial
   ```

---

## When to Use What

### Basis Function Selection

| Scenario | Recommended Basis | Why |
|----------|-------------------|-----|
| General purpose | `PHS(3)` | No parameter tuning, robust |
| Very smooth data | `PHS(5)` or `PHS(7)` | Higher order = smoother |
| Need fine-tuned accuracy | `IMQ(ε)` or `Gaussian(ε)` | Shape parameter allows tuning |
| Rough/noisy data | `PHS(1)` or `PHS(3)` | Lower order, less overfitting |
| GPU computation | Any | All bases support GPU via KernelAbstractions |

### Polynomial Degree Selection

| `poly_deg` | Effect | Use When |
|------------|--------|----------|
| -1 | No polynomial | Edge cases, pure RBF |
| 0 | Constant | Minimal augmentation |
| 1 | Linear | Simple problems, small stencils |
| 2 (default) | Quadratic | Most applications |
| 3+ | Cubic+ | High accuracy needs, large stencils |

**Rule of thumb:** `poly_deg` should be ≤ `(k - 1) / dim` to avoid underdetermined systems

### Stencil Size (k) Guidelines

| Problem Type | Suggested k | Notes |
|--------------|-------------|-------|
| Quick test | Auto (default) | Uses `autoselect_k()` |
| 2D Laplacian | 20-40 | Balance accuracy/speed |
| 2D Gradient | 15-30 | First derivatives need fewer |
| 3D operators | 40-80 | Higher dimension needs more |
| High accuracy | 50+ | More neighbors = smoother |

### Operator Selection

| Task | Operator | Returns |
|------|----------|---------|
| Sum of second derivatives | `laplacian(points)` | Vector (scalar per point) |
| All partial derivatives | `gradient(points)` | Matrix (N × dim) |
| Single partial derivative | `partial(points, order, dim)` | Vector |
| Directional derivative | `directional(points, v)` | Vector |
| Interpolate to new points | `regrid(src, dst)` | Vector |
| Global interpolation | `Interpolator(points, values)` | Callable object |

---

## Extension Patterns

### Adding a New RBF Type

1. Create file `src/basis/my_rbf.jl`
2. Define struct inheriting from `AbstractRadialBasis`
3. Implement the basis function callable and derivative operators

```julia
# Minimal example: Wendland C2 (compactly supported)
struct WendlandC2{T, D<:Int} <: AbstractRadialBasis
    ε::T        # shape parameter (support radius)
    poly_deg::D
    function WendlandC2(ε::T=1.0; poly_deg::D=2) where {T, D<:Int}
        ε > 0 || throw(ArgumentError("ε must be > 0"))
        new{T,D}(ε, poly_deg)
    end
end

# Basis function: (1 - εr)₊⁴(4εr + 1) where (·)₊ = max(·, 0)
function (rbf::WendlandC2)(x, xᵢ)
    r = euclidean(x, xᵢ) * rbf.ε
    r >= 1 && return zero(r)
    return (1 - r)^4 * (4r + 1)
end

# Implement derivative operators: ∂, ∇, ∂², ∇², D, D², H
# See src/basis/polyharmonic_spline.jl for full examples
```

4. Include in `src/basis/basis.jl`
5. Export in `src/RadialBasisFunctions.jl`
6. Add tests in `test/basis/`

### Adding a New Operator

1. Create file `src/operators/my_operator.jl`
2. Define operator struct inheriting from `ScalarValuedOperator` or `VectorValuedOperator`
3. Implement the operator-basis interaction

```julia
# Example: Mixed partial derivative ∂²f/∂x∂y
struct MixedPartial{T<:Int} <: ScalarValuedOperator
    dim1::T
    dim2::T
end

# Define how operator acts on basis (returns callable)
(op::MixedPartial)(basis) = MixedPartialOp(basis, op.dim1, op.dim2)

# Constructor function
function mixed_partial(data::AbstractVector, dim1::Int, dim2::Int; kw...)
    return RadialBasisOperator(MixedPartial(dim1, dim2), data; kw...)
end

# Implement MixedPartialOp(basis, dim1, dim2)(x, xᵢ) for each basis type
```

4. Include in `src/operators/operators.jl`
5. Export in `src/RadialBasisFunctions.jl`
6. Add tests in `test/operators/`