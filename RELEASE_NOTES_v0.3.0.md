# v0.3.0 Release Notes

## Breaking Changes

- **`Gradient` type removed** — `gradient(...)` is now a convenience alias for `jacobian(...)`. Code that used `Gradient` as a type directly will break.
- **Basis operator closures replaced with functor structs** — `∂`, `∇`, `∂²`, `∇²`, `D`, `D²` are now callable structs instead of closure-returning functions. The public API (`∂(basis, dim)`, etc.) is unchanged, but code dispatching on closure types will need updating.
- **`RadialBasisOperator` gained a `device` type parameter** — struct now has 7 type parameters (was 6). Code pattern-matching on the full parametric type may need updating.

## New Features

### Automatic Differentiation Support (Enzyme.jl & Mooncake.jl)
- Native reverse-mode AD via package extensions (auto-activated when Enzyme or Mooncake is loaded)
- **Enzyme extension** — `EnzymeRules` (`augmented_primal` + `reverse`) for operator application, `_build_weights`, and basis evaluation
- **Mooncake extension** — `rrule!!` with `@is_primitive` for operator application, `_build_weights`, `Interpolator` construction, and basis evaluation
- Shared backward pass (`src/solve/backward.jl`) using the implicit function theorem to differentiate through the stencil linear solve
- Differentiable w.r.t. data point locations and basis shape parameter `ε`
- Analytic second derivatives for all PHS orders, Gaussian, and IMQ (needed for chain rule through RHS assembly)
- Shape parameter derivatives (`∂φ/∂ε`, `∂∇²φ/∂ε`, `∂∂φ/∂ε`) for Gaussian and IMQ bases
- BLAS-optimized backward pass using `mul!` rank-1 updates
- Compatible with `DifferentiationInterface.jl` for a unified AD interface

### Jacobian Operator
- New `Jacobian` operator replaces the old `Gradient` type
- Generalizes beyond scalar fields: scalar input → `(N_eval × D)` gradient; vector input → `(N_eval × D × D)` Jacobian; higher-rank tensor support
- `gradient(...)` is preserved as a convenience alias

### Hessian Basis Functor
- New `H` (Hessian) functor for all basis types (PHS 1/3/5/7, Gaussian, IMQ)
- Returns `SMatrix{N,N,T}` for second-derivative matrix of basis functions
- Used internally by Hermite dispatch for optimized second-derivative computation

### GPU Improvements
- `device` keyword on all operator constructors — auto-detects backend via `KernelAbstractions.get_backend(data)` or explicit override (e.g., `device=CUDABackend()`)
- `Adapt.jl` support — `Adapt.adapt_structure` for `RadialBasisOperator` and `Interpolator`, enabling `adapt(CUDABackend(), op)`
- GPU array types preserved in operator algebra (`+`/`-`)
- `VectorValuedOperator` evaluation uses `mul!` with views to eliminate temporaries
- `Interpolator` batch evaluation accepts `AbstractVector{<:AbstractVector}` (not just `Vector`)
- `show` methods use `eltype()` instead of `typeof(first())` for GPU compatibility

### Unified Keyword-Based Constructor API
- Primary `RadialBasisOperator` constructor rewritten with keyword arguments:
  ```julia
  RadialBasisOperator(ℒ, data;
      eval_points=data, basis=PHS(3; poly_deg=2),
      k=autoselect_k(data, basis), adjl=...,
      hermite=nothing, device=get_backend(data))
  ```
- `hermite` keyword accepts a `NamedTuple` `(is_boundary=..., bc=..., normals=...)` replacing old positional arguments
- All operator constructors (`laplacian`, `partial`, `gradient`, `jacobian`, `directional`, `custom`, `regrid`) forward `kw...` to unified constructor

## Refactoring

### Basis Functions: Closures to Functors
- All differential operator factories replaced with typed callable structs (`∂{B}`, `∇{B}`, `∂²{B}`, `∇²{B}`, `D{B,V}`, `D²{B,V1,V2}`, `H{B}`)
- Enables multiple dispatch in AD rules and across the codebase
- PHS normal dispatch refactored to use 3-argument call instead of `Union{AbstractVector,Nothing}`

### Solve System Reorganization
- `stencil_math.jl` → `assembly.jl` (pure math: collocation matrix, RHS, stencil assembly)
- `kernel_exec.jl` → `execution.jl` (memory allocation, kernel launching, batch processing)
- New `BasisOperators` struct packages basis + gradient + Hessian functors, computed once per stencil
- 6 new files for AD support (`ad_shared.jl`, `backward.jl`, `backward_cache.jl`, `forward_cache.jl`, `operator_second_derivatives.jl`, `shape_parameter_derivatives.jl`)

## Performance

- Pre-allocated `λ` buffer reused across stencils in solve loop (fewer allocations)
- `view(data, neighbors)` instead of `[data[i] for i in neighbors]` per stencil
- `dot()` instead of intermediate allocations in polynomial evaluation
- `mul!` with views in `VectorValuedOperator` evaluation (zero-allocation GPU-friendly)

## Testing

- Comprehensive AD test suite (`test/extensions/autodiff_di.jl`) testing Laplacian, Gradient, Partial, Interpolator (construction + evaluation), basis functions, shape parameter differentiation, and 3D operators against finite differences
- All `rand`/`randn` calls seeded with `MersenneTwister` for deterministic CI
- New test files for Jacobian operator, device keyword, and individual extension tests
- Enzyme tests skipped on Julia 1.12+ (`VERSION < v"1.12"`)

## Documentation

- Switched to DocumenterVitepress.jl frontend
- New autodiff guide (`docs/src/guides/autodiff.md`) with runnable examples using `DI.AutoMooncake`
- New quick reference page (`docs/src/guides/quickref.md`)
- Restructured navigation with Guides and Reference sections
- Refreshed landing page and README

## Infrastructure

- Formatter switched from JuliaFormatter.jl to Runic.jl
- New dependencies: `Adapt.jl`, `StaticArrays.jl` (direct); `Enzyme.jl`, `EnzymeCore.jl`, `Mooncake.jl` (weak deps)
- CI action bumps (checkout v6, upload-artifact v6, etc.)
- `ChainRulesCore` dependency removed
