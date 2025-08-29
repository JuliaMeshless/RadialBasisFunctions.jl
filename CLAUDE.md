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
   - Specific operators: `Partial`, `Gradient`, `Laplacian`, `Directional`, `Custom`
   - `operator_algebra.jl` - Composition and algebraic operations on operators
   - Virtual operators for performance optimization

3. **Solve System** (`src/solve.jl`, `src/solve_utils.jl`): Core weight computation
   - `_build_weights()` - Main function for computing RBF stencil weights
   - Uses KernelAbstractions.jl for GPU/CPU parallelization
   - Batch processing for memory efficiency
   - `solve_hermite.jl` - Clean Hermite interpolation implementation with multiple dispatch

4. **Interpolation** (`src/interpolation.jl`): 
   - `Interpolator` type for global interpolation (uses all data points)
   - Supports scattered data interpolation

5. **Utilities** (`src/utils.jl`):
   - `find_neighbors()` - k-nearest neighbor search using NearestNeighbors.jl
   - `find_neighbors_pn()` - k-nearest neighbor search using PointNeighbors.jl (alternative)
     - Uses multiple dispatch to automatically select optimal backend
     - GPU arrays (CuArray): FullGridCellList with DynamicVectorOfVectors (via CUDA extension)
     - CPU arrays: DictionaryCellList (default)
   - `reorder_points!()` - Point ordering utilities

### Key Design Patterns

- **Lazy Evaluation**: Operators compute weights only when needed, with caching
- **GPU Support**: KernelAbstractions.jl enables GPU computation for weight building
- **Modular Design**: Basis functions, operators, and solvers are decoupled
- **Type System**: Heavy use of parametric types for performance
- **Neighbor Search**: Local support domains via k-nearest neighbors for efficiency
  - Two implementations available: NearestNeighbors.jl (default) and PointNeighbors.jl
  - Use `use_pointneighbors=true` in operator constructors to enable PointNeighbors.jl
  - PointNeighbors.jl automatically detects GPU arrays (CuArray) and uses GPU-compatible backends

### Data Requirements

The package requires `Vector{AbstractVector}` input format (not matrices). Each point must be a vector type with inferrable dimension (e.g., `SVector{2,Float64}` from StaticArrays.jl).

### Performance Considerations

- Stencil weights are computed in batches to manage memory
- Operators cache weights and check validity before recomputation
- GPU kernels are used for parallel weight computation
- Local neighborhoods (k-nearest neighbors) reduce computational complexity

## Key Files for Understanding

- `src/RadialBasisFunctions.jl` - Main module with exports and precompilation
- `src/solve.jl:21-102` - Core weight computation with GPU kernels
- `src/operators/operators.jl:10-31` - Main operator type definition
- `src/basis/basis.jl` - Abstract basis type hierarchy

## Important Development Notes

- The package requires Julia 1.8+ (see Project.toml compatibility)
- Uses KernelAbstractions.jl for GPU/CPU parallelization
- Data must be in `Vector{AbstractVector}` format (not matrices) - each point needs inferrable dimension (e.g., `SVector{2,Float64}`)
- Two neighbor search backends available: NearestNeighbors.jl (default) and PointNeighbors.jl (`use_pointneighbors=true`)
- PointNeighbors.jl automatically handles GPU arrays with appropriate backends