# Custom Distance Metrics Implementation Summary

**Date**: 2025-11-13
**Status**: Phase 4 - Partially Complete (Gradient operator done, 4 operators remaining)

## Overview

Implementation of custom distance metric support for RadialBasisFunctions.jl, addressing the limitation where Euclidean distance was hardcoded throughout the package.

## Design Decisions

### 1. **Non-Breaking Changes**
- All new metric parameters default to `Euclidean()`
- Existing code works without modification
- Metric parameter added as keyword argument: `metric=Euclidean()`

### 2. **Performance Preservation**
- **Euclidean metric**: Uses existing fast analytical derivatives
- **Non-Euclidean metrics**: Uses automatic differentiation (ForwardDiff.jl)
- Method dispatch based on metric type: `AbstractRadialBasis{M<:Metric}` where `M` determines which implementation is used

### 3. **Type System Architecture**
- Parametric types: `AbstractRadialBasis{M<:Metric}`
- Metrics from Distances.jl (already a dependency)
- Each basis stores its metric in a field and as a type parameter

### 4. **Safety & Validation**
- Hermite boundary conditions restricted to Euclidean metric only
- Clear error messages when non-Euclidean used with Hermite interpolation
- Validation at operator construction (early failure)

---

## Implementation Details

### Phase 1: Type System & Infrastructure ✅

**Files Modified:**
- `Project.toml`: Added ForwardDiff.jl dependency
- `src/RadialBasisFunctions.jl`: Added `using ForwardDiff`
- `src/basis/basis.jl`: Updated abstract type hierarchy

**Changes:**
```julia
# Before
abstract type AbstractBasis end
abstract type AbstractRadialBasis <: AbstractBasis end

# After
abstract type AbstractBasis{M<:Metric} end
abstract type AbstractRadialBasis{M<:Metric} <: AbstractBasis{M} end
```

---

### Phase 2: Basis Functions ✅

All basis functions updated with dual implementation pattern:

#### Pattern Used (Example: PHS1)

**Struct Definition:**
```julia
# Before
struct PHS1{T<:Int} <: AbstractPHS
    poly_deg::T
    function PHS1(poly_deg::T) where {T<:Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end

# After
struct PHS1{T<:Int, M<:Metric} <: AbstractPHS{M}
    poly_deg::T
    metric::M
    function PHS1(poly_deg::T; metric::M=Euclidean()) where {T<:Int, M<:Metric}
        check_poly_deg(poly_deg)
        return new{T, M}(poly_deg, metric)
    end
end
```

**Basis Evaluation:**
```julia
# Before
(phs::PHS1)(x, xᵢ) = euclidean(x, xᵢ)

# After
(phs::PHS1)(x, xᵢ) = phs.metric(x, xᵢ)
```

**Derivative Functions (Dual Implementation):**
```julia
# Fast path: Analytical derivative for Euclidean metric
function ∂(phs::PHS1{T, Euclidean}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where T
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        r = phs.metric(x, xᵢ)
        if normal_arg === nothing
            return (x[dim] - xᵢ[dim]) / (r + AVOID_INF)
        else
            # Analytical formula for Hermite case
            ...
        end
    end
    return ∂ℒ
end

# Slow path: AD-based derivative for non-Euclidean metrics
function ∂(phs::PHS1{T, M}, dim::Int, normal::Union{AbstractVector,Nothing}=nothing) where {T, M<:Metric}
    function ∂ℒ(x, xᵢ, normal_arg=normal)
        return ForwardDiff.derivative(t -> phs(x + t * _unit_vector(x, dim), xᵢ), 0.0)
    end
    return ∂ℒ
end
```

**Helper Function:**
```julia
# Helper for AD-based derivatives (added to each basis file)
function _unit_vector(x::AbstractVector, dim::Int)
    e = zero(x)
    e = setindex(e, one(eltype(x)), dim)
    return e
end
```

**Files Completed:**
1. ✅ `src/basis/polyharmonic_spline.jl`
   - PHS1, PHS3, PHS5, PHS7 (all updated)
   - Updated convenience constructor `PHS(n; poly_deg, metric)`
   - Removed obsolete keyword constructor loop
   - Each has 6 derivative functions: `∂`, `∇`, `directional∂²`, `∂²`, `∇²`

2. ✅ `src/basis/gaussian.jl`
   - Struct updated with metric parameter
   - 4 derivative functions: `∂`, `∇`, `∂²`, `∇²`
   - Fixed missing return statements

3. ✅ `src/basis/inverse_multiquadric.jl`
   - Struct updated with metric parameter
   - 4 derivative functions: `∂`, `∇`, `∂²`, `∇²`
   - Fixed missing return statements

4. ✅ `src/basis/monomial.jl`
   - Added metric parameter for type consistency
   - Metric not used in computation (polynomials aren't distance-based)
   - Added docstring note explaining this

---

### Phase 3: Validation & Safety ✅

**File: `src/solve/stencil_math.jl`**

Added validation to `compute_hermite_rbf_entry`:
```julia
function compute_hermite_rbf_entry(
    i::Int, j::Int, data::HermiteStencilData, basis::AbstractRadialBasis{M}
) where M
    # Validate that Hermite interpolation only uses Euclidean metric
    if !(M <: Euclidean)
        throw(ArgumentError(
            "Hermite interpolation with boundary conditions requires Euclidean metric. " *
            "Normal derivatives used in Neumann/Robin boundary conditions are not well-defined " *
            "for non-Euclidean metrics. Current metric: $(M)"
        ))
    end
    # ... rest of function
end
```

**Reasoning:** Hermite interpolation with boundary conditions uses normal derivatives (`∂ₙu = n·∇u`), which assume Euclidean geometry. The gradient formula changes for non-Euclidean metrics, making these operations mathematically incorrect.

---

### Phase 4: Neighbor Search & Operators 🔄

#### Neighbor Search ✅

**File: `src/utils.jl`**

Updated both `find_neighbors` functions:
```julia
# Before
function find_neighbors(data::AbstractVector, k::Int)
    tree = KDTree(data)
    adjl, _ = knn(tree, data, k, true)
    return adjl
end

# After
function find_neighbors(data::AbstractVector, k::Int; metric::M=Euclidean()) where {M<:Metric}
    tree = KDTree(data, metric)
    adjl, _ = knn(tree, data, k, true)
    return adjl
end
```

Added comprehensive docstrings documenting the metric parameter.

#### Gradient Operator ✅

**File: `src/operators/gradient.jl`**

Updated all 3 constructor variants to extract metric from basis:
```julia
# Pattern used for all variants
function gradient(
    data::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k; metric=basis.metric),  # ← Key change
) where {B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Gradient{Dim}()
    return RadialBasisOperator(ℒ, data, basis; k=k, adjl=adjl)
end
```

**Three variants updated:**
1. ✅ `gradient(data, basis; k, adjl)`
2. ✅ `gradient(data, eval_points, basis; k, adjl)`
3. ✅ `gradient(data, eval_points, basis, is_boundary, boundary_conditions, normals; k, adjl)` (Hermite variant)

---

## Remaining Work

### Operators to Update (4 remaining)

Same pattern as gradient - extract `metric=basis.metric` in `find_neighbors` call:

1. **`src/operators/laplacian.jl`** ❌
   - Update `laplacian(data, basis; k, adjl)`
   - Update `laplacian(data, eval_points, basis; k, adjl)`
   - Update Hermite variant

2. **`src/operators/partial.jl`** ❌
   - Update `partial(data, dim, order, basis; k, adjl)`
   - Update `partial(data, eval_points, dim, order, basis; k, adjl)`

3. **`src/operators/directional.jl`** ❌
   - Update `directional(data, v1, v2, basis; k, adjl)`
   - Update `directional(data, eval_points, v1, v2, basis; k, adjl)`

4. **`src/operators/custom.jl`** ❌
   - Update `custom(ℒ, data, basis; k, adjl)`
   - Update `custom(ℒ, data, eval_points, basis; k, adjl)`

**Pattern for each:**
```julia
# Change this line in each constructor:
adjl=find_neighbors(data, k),
# To this:
adjl=find_neighbors(data, k; metric=basis.metric),
```

---

### Testing (Not Started)

Need to add comprehensive tests:

1. **Backward Compatibility** ❌
   ```julia
   # Test that existing code works unchanged
   @testset "Non-breaking defaults" begin
       data = [SVector(rand(2)...) for _ in 1:100]
       basis = PHS(3; poly_deg=2)  # No metric specified
       op = gradient(data, basis)
       # Should work exactly as before
   end
   ```

2. **Custom Metrics** ❌
   ```julia
   @testset "Custom metrics" begin
       # Test with Manhattan distance
       basis_manhattan = PHS(3; poly_deg=2, metric=Cityblock())
       op = gradient(data, basis_manhattan)

       # Test with Chebyshev distance
       basis_cheby = PHS(3; poly_deg=2, metric=Chebyshev())
       op = gradient(data, basis_cheby)

       # Test with user-defined metric
       struct MyMetric <: Metric end
       (::MyMetric)(x, y) = sum((x .- y).^4)
       basis_custom = PHS(3; poly_deg=2, metric=MyMetric())
       op = gradient(data, basis_custom)
   end
   ```

3. **Hermite Validation** ❌
   ```julia
   @testset "Hermite validation errors" begin
       basis_manhattan = PHS(3; poly_deg=2, metric=Cityblock())
       # Should throw error when used with Hermite
       @test_throws ArgumentError gradient(
           data, eval_points, basis_manhattan,
           is_boundary, boundary_conditions, normals
       )
   end
   ```

4. **AD Correctness** ❌
   ```julia
   @testset "AD derivatives match analytical for Euclidean" begin
       # Verify AD gives same results as analytical for Euclidean
       # This validates our AD implementation
   end
   ```

---

### Documentation (Not Started)

Need to add:

1. **Tutorial** ❌: `docs/src/custom_metrics.md`
   - How to use built-in metrics (Manhattan, Chebyshev, etc.)
   - How to define custom metrics
   - Performance implications (AD overhead)
   - Mathematical limitations (Hermite, polynomial augmentation)

2. **Update Internals** ❌: `docs/src/internals.md`
   - Document the dual implementation approach
   - Explain when AD vs analytical is used

3. **API Documentation** ❌: Update docstrings in `docs/src/api.md`
   - Document metric parameter for all constructors

---

## Key Technical Notes

### 1. Duplicate Helper Function
The `_unit_vector` helper function is currently duplicated in:
- `src/basis/polyharmonic_spline.jl`
- `src/basis/gaussian.jl`
- `src/basis/inverse_multiquadric.jl`

**TODO**: Move this to `src/basis/basis.jl` to avoid duplication.

### 2. Squared Euclidean Distance
Some code uses `sqeuclidean(x, xᵢ)` for optimization. This has been updated to:
```julia
evaluate(SqEuclidean(), x, xᵢ)
```
for consistency with Distances.jl API.

### 3. Method Dispatch Ambiguity
The dual implementation uses method dispatch on the metric type parameter. Julia's method dispatch automatically selects:
- `func(::Type{T, Euclidean}, ...)` for Euclidean (fast analytical path)
- `func(::Type{T, M}, ...) where {M<:Metric}` for others (AD path)

No runtime overhead - selection happens at compile time.

### 4. KDTree Compatibility
`NearestNeighbors.jl`'s `KDTree` supports many metrics from Distances.jl out of the box:
- Euclidean, Chebyshev, Minkowski, Cityblock, etc.

For exotic custom metrics that KDTree doesn't support, users may need to use `BallTree` instead (slower but works with any metric).

### 5. GPU Considerations
Custom metrics may not work with GPU-based neighbor search (PointNeighbors.jl) unless they're GPU-compatible. This is a limitation to document.

---

## Usage Examples (After Completion)

### Using Built-in Metrics
```julia
using RadialBasisFunctions
using Distances

# Euclidean (default - no change from before)
basis = PHS(3; poly_deg=2)
op = gradient(data, basis)

# Manhattan distance (L1 norm)
basis_l1 = PHS(3; poly_deg=2, metric=Cityblock())
op_l1 = gradient(data, basis_l1)

# Chebyshev distance (L∞ norm)
basis_linf = PHS(3; poly_deg=2, metric=Chebyshev())
op_linf = gradient(data, basis_linf)

# Minkowski-p distance
basis_lp = PHS(3; poly_deg=2, metric=Minkowski(1.5))
op_lp = gradient(data, basis_lp)
```

### Defining Custom Metrics
```julia
using Distances

# Define custom metric type
struct PeriodicDistance{T<:Real} <: Metric
    period::T
end

# Make it callable
function (d::PeriodicDistance)(x, y)
    diff = abs.(x .- y)
    diff = min.(diff, d.period .- diff)  # Handle periodicity
    return sqrt(sum(diff.^2))
end

# Use it
basis_periodic = PHS(3; poly_deg=2, metric=PeriodicDistance(2π))
op_periodic = gradient(data, basis_periodic)
```

### Performance Comparison
```julia
# Fast: Euclidean with analytical derivatives
@time op_euclidean = gradient(data, PHS(3; poly_deg=2))

# Slower: Manhattan with AD derivatives (2-10x overhead)
@time op_manhattan = gradient(data, PHS(3; poly_deg=2, metric=Cityblock()))
```

---

## Next Steps for Resuming Work

1. **Update remaining 4 operators** (straightforward - same pattern as gradient)
   - Laplacian, Partial, Directional, Custom
   - Change: `adjl=find_neighbors(data, k)` → `adjl=find_neighbors(data, k; metric=basis.metric)`

2. **Run existing tests** to ensure backward compatibility
   ```bash
   julia --project=. -e "using Pkg; Pkg.test()"
   ```

3. **Add new tests** for custom metrics and validation

4. **Write documentation tutorial**

5. **Optional optimizations:**
   - Move `_unit_vector` to basis.jl
   - Consider analytical derivatives for common non-Euclidean metrics (Manhattan, Chebyshev)
   - Add BallTree fallback for unsupported metrics

---

## Files Modified Summary

### Completed (13 files)
1. ✅ `Project.toml`
2. ✅ `src/RadialBasisFunctions.jl`
3. ✅ `src/basis/basis.jl`
4. ✅ `src/basis/polyharmonic_spline.jl`
5. ✅ `src/basis/gaussian.jl`
6. ✅ `src/basis/inverse_multiquadric.jl`
7. ✅ `src/basis/monomial.jl`
8. ✅ `src/solve/stencil_math.jl`
9. ✅ `src/utils.jl`
10. ✅ `src/operators/gradient.jl`

### Remaining (4 files)
11. ❌ `src/operators/laplacian.jl`
12. ❌ `src/operators/partial.jl`
13. ❌ `src/operators/directional.jl`
14. ❌ `src/operators/custom.jl`

### To Create
15. ❌ `test/custom_metrics_test.jl`
16. ❌ `docs/src/custom_metrics.md`

---

## Estimated Remaining Effort

- **Operators** (4 files): ~15 minutes (simple pattern repetition)
- **Testing**: ~30 minutes (write comprehensive tests)
- **Documentation**: ~30 minutes (tutorial + API updates)
- **Testing & Bug Fixes**: ~30 minutes (run tests, fix any issues)

**Total**: ~2 hours to completion

---

## Contact Points for Questions

1. **Hermite interpolation**: See `src/solve/stencil_math.jl:336-353` for validation logic
2. **Dual dispatch pattern**: See `src/basis/polyharmonic_spline.jl:53-81` (PHS1 ∂ function)
3. **Metric extraction**: See `src/operators/gradient.jl:22` for pattern
4. **User-defined metrics**: See Distances.jl documentation

---

*Generated: 2025-11-13*
*Implementation Status: 72% Complete (10/14 code files, 0/2 new files)*
