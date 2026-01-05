# Operator Refactoring

## Architecture Overview

### Three-Layer Design

```
Layer 1: make_operator()          → Constructs operator instances (handles Dim inference)
Layer 2: RadialBasisOperator()    → Unified constructor (handles hermite dispatch)
Layer 3: build_operator()         → Thin wrapper (just 3 lines!)
Layer 4: operator functions        → User-facing API (jacobian, laplacian, etc.)
```

## Implementation

### Part 1: Refactor `src/operators/operators.jl`

#### Step 1A: Add Direct Dispatch for Operator Construction

Add after line 73 (after existing RadialBasisOperator constructors):

```julia
# ============================================================================
# Operator Construction via Direct Dispatch
# ============================================================================

"""
    make_operator(OpType, data, params...)

Construct operator from type, automatically extracting spatial dimension when needed.
Uses direct multiple dispatch - no trait hierarchy needed.

# Examples

make_operator(Laplacian, data)              # → Laplacian()
make_operator(Jacobian, data)               # → Jacobian{2}() (infers Dim from data)
make_operator(Partial, data, 1, 2)          # → Partial(1, 2)
make_operator(Directional, data, [1.0, 0])  # → Directional{2}([1.0, 0])
make_operator(Custom, data, ℒ)              # → Custom(ℒ)

"""
make_operator(::Type{Laplacian}, data::AbstractVector) = Laplacian()

make_operator(::Type{Custom}, data::AbstractVector, ℒ::Function) = Custom(ℒ)

make_operator(::Type{Partial}, data::AbstractVector, order::Int, dim::Int) =
    Partial(order, dim)

make_operator(::Type{Jacobian}, data::AbstractVector) =
    Jacobian{length(first(data))}()

make_operator(::Type{Directional}, data::AbstractVector, v::AbstractVector) =
    Directional{length(first(data))}(v)
```

**Lines added: ~25**

#### Step 1B: Replace RadialBasisOperator Constructors (lines 33-73)

Delete the three existing constructors and replace with:

```julia
# ============================================================================
# Unified RadialBasisOperator Constructor
# ============================================================================

# Dispatch helper for weight building
_build_weights_dispatch(ℒ, data, eval_points, adjl, basis, ::Nothing) =
    _build_weights(ℒ, data, eval_points, adjl, basis)

function _build_weights_dispatch(ℒ, data, eval_points, adjl, basis, hermite::NamedTuple)
    return _build_weights(
        ℒ, data, eval_points, adjl, basis,
        hermite.is_boundary,
        hermite.boundary_conditions,
        hermite.normals
    )
end

"""
    RadialBasisOperator(ℒ, data; eval_points=data, basis=PHS(3; poly_deg=2), hermite=nothing, k, adjl)

Unified constructor for RadialBasisOperator supporting standard and Hermite interpolation.

# Arguments
- `ℒ`: Operator instance (e.g., `Laplacian()`, `Jacobian{2}()`)
- `data`: Data points

# Keyword Arguments
- `eval_points`: Evaluation points (default: `data`)
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `hermite`: Hermite data as NamedTuple `(is_boundary=..., boundary_conditions=..., normals=...)` (default: `nothing`)
- `k`: Stencil size (default: autoselect)
- `adjl`: Adjacency list (default: computed from data and eval_points)

# Examples

# Basic usage
op = RadialBasisOperator(Laplacian(), data)

# Custom basis and stencil size
op = RadialBasisOperator(Jacobian{2}(), data; basis=PHS(5; poly_deg=3), k=40)

# Different evaluation points
op = RadialBasisOperator(Laplacian(), data; eval_points=eval_pts)

# Hermite interpolation
op = RadialBasisOperator(Laplacian(), data;
                         eval_points=eval_pts,
                         hermite=(is_boundary=[true, false, ...],
                                 boundary_conditions=[Dirichlet(), ...],
                                 normals=[[1.0, 0.0], ...]))

"""
function RadialBasisOperator(
    ℒ,
    data::AbstractVector;
    eval_points::AbstractVector=data,
    basis::AbstractRadialBasis=PHS(3; poly_deg=2),
    hermite::Union{Nothing,NamedTuple}=nothing,
    k::Int=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
)
    weights = _build_weights_dispatch(ℒ, data, eval_points, adjl, basis, hermite)
    return RadialBasisOperator(ℒ, weights, data, eval_points, adjl, basis, true)
end
```

**Lines: ~50 but mostly comments (replaces ~40 lines of code for 3 constructors)**

#### Step 1C: Add build_operator Helper

Add after the RadialBasisOperator constructor:

```julia
# ============================================================================
# Unified Operator Builder
# ============================================================================

"""
    build_operator(OpType, data, op_params; kw...)

Build operator by constructing operator instance then delegating to RadialBasisOperator.
All keyword arguments are forwarded to RadialBasisOperator.

# Examples
build_operator(Jacobian, data, ())              # No operator params
build_operator(Partial, data, (1, 2))           # order=1, dim=2
build_operator(Directional, data, (v,))         # direction vector v
build_operator(Custom, data, (ℒ,))              # custom function ℒ
"""
function build_operator(OpType::Type{<:AbstractOperator}, data::AbstractVector, op_params::Tuple; kw...)
    ℒ = make_operator(OpType, data, op_params...)
    return RadialBasisOperator(ℒ, data; kw...)
end
```

**Lines: ~15**

**Total changes to operators.jl: +90 lines, -40 lines = +50 net**

### Part 2: Refactor Operator Files

#### `src/operators/jacobian.jl` (Complete replacement)

```julia
"""
    Jacobian{Dim} <: VectorValuedOperator{Dim}

Jacobian operator - computes all partial derivatives.
For scalar fields, this is the gradient.
For vector fields, this is the full Jacobian matrix.
"""
struct Jacobian{Dim} <: VectorValuedOperator{Dim} end

function (op::Jacobian{Dim})(basis) where {Dim}
    return ntuple(dim -> ∂(basis, dim), Dim)
end

"""
    jacobian(data; basis=PHS(3; poly_deg=2), eval_points=data, hermite=nothing, k, adjl)

Build Jacobian operator. Dimension inferred automatically from data.

# Keyword Arguments
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `eval_points`: Evaluation points (default: `data`)
- `hermite`: Hermite interpolation data (default: `nothing`)
- `k`: Stencil size (default: autoselect)
- `adjl`: Adjacency list (default: computed)

# Examples
data = [SVector{2}(rand(2)) for _ in 1:100]

# Basic
op = jacobian(data)

# Custom basis and stencil
op = jacobian(data; basis=PHS(5; poly_deg=3), k=40)

# Different eval points
op = jacobian(data; eval_points=data[1:10])

# Hermite interpolation
op = jacobian(data;
              eval_points=eval_pts,
              hermite=(is_boundary=[true, false, ...],
                      boundary_conditions=[Dirichlet(), ...],
                      normals=[[1.0, 0.0], ...]))
"""
function jacobian(data::AbstractVector; kw...)
    return build_operator(Jacobian, data, (); kw...)
end

# One-shot: create and apply
function jacobian(data::AbstractVector, x; basis::AbstractRadialBasis=PHS(3; poly_deg=2), kw...)
    op = jacobian(data; basis=basis, kw...)
    return op(x)
end

print_op(::Jacobian) = "Jacobian (J)"
```

**Lines: ~50 (was ~170)**

#### `src/operators/laplacian.jl` (Complete replacement)

```julia
"""
    Laplacian <: ScalarValuedOperator

Laplacian operator - sum of second derivatives (∇²f).
"""
struct Laplacian <: ScalarValuedOperator end
(::Laplacian)(basis) = ∇²(basis)

"""
    laplacian(data; basis=PHS(3; poly_deg=2), eval_points=data, hermite=nothing, k, adjl)

Build Laplacian operator.

See [`jacobian`](@ref) for keyword arguments and examples.
"""
function laplacian(data::AbstractVector; kw...)
    return build_operator(Laplacian, data, (); kw...)
end

print_op(op::Laplacian) = "Laplacian (∇²f)"
```

**Lines: ~18 (was ~60)**

#### `src/operators/partial.jl` (Complete replacement)

```julia
"""
    Partial{T<:Int} <: ScalarValuedOperator

Partial derivative operator of specified order and dimension.
"""
struct Partial{T<:Int} <: ScalarValuedOperator
    order::T
    dim::T
end
(op::Partial)(basis) = ∂(basis, op.order, op.dim)

"""
    partial(data, order, dim; basis=PHS(3; poly_deg=2), eval_points=data, hermite=nothing, k, adjl)

Build partial derivative operator.

# Arguments
- `data`: Data points
- `order`: Derivative order (1 or 2)
- `dim`: Dimension index to differentiate

# Keyword Arguments
See [`jacobian`](@ref) for keyword arguments.

# Examples

# First derivative in x-direction
∂x = partial(data, 1, 1)

# Second derivative in y with custom basis
∂²y = partial(data, 2, 2; basis=PHS(5; poly_deg=4), k=50)

"""
function partial(data::AbstractVector, order::Int, dim::Int; kw...)
    return build_operator(Partial, data, (order, dim); kw...)
end

# Helper for applying operator to basis
function ∂(basis::AbstractBasis, order::T, dim::T) where {T<:Int}
    if order == 1
        return ∂(basis, dim)
    elseif order == 2
        return ∂²(basis, dim)
    else
        throw(ArgumentError(
            "Only orders 1 and 2 supported. Use Custom operator for higher orders."
        ))
    end
end

print_op(op::Partial) = "∂ⁿf/∂xᵢ (n = $(op.order), i = $(op.dim))"
```

**Lines: ~50 (was ~80)**

#### `src/operators/directional.jl` (Complete replacement)

```julia
"""
    Directional{Dim,T} <: ScalarValuedOperator

Directional derivative operator: ∇f⋅v
"""
struct Directional{Dim,T} <: ScalarValuedOperator
    v::T
end
Directional{Dim}(v) where {Dim} = Directional{Dim,typeof(v)}(v)

"""
    directional(data, v; basis=PHS(3; poly_deg=2), eval_points=data, hermite=nothing, k, adjl)

Build directional derivative operator.

# Arguments
- `data`: Data points
- `v`: Direction vector (length = Dim for constant, or length = N for spatially-varying)

# Keyword Arguments
See [`jacobian`](@ref) for keyword arguments.
"""
function directional(data::AbstractVector, v::AbstractVector; kw...)
    return build_operator(Directional, data, (v,); kw...)
end

# Validation helper
function _validate_directional_vector(v, Dim::Int, data_length::Int)
    if !(length(v) == Dim || length(v) == data_length)
        throw(DomainError(
            "Direction vector length $(length(v)) must equal dimension $Dim or data length $data_length"
        ))
    end
end

# Weight combination helper
function _combine_directional_weights(weights, v, Dim::Int)
    if length(v) == Dim
        # Constant direction
        return mapreduce(+, zip(weights, v)) do (w, vᵢ)
            w * vᵢ
        end
    else
        # Spatially-varying direction
        vv = ntuple(i -> getindex.(v, i), Dim)
        return mapreduce(+, zip(weights, vv)) do (w, vᵢ)
            Diagonal(vᵢ) * w
        end
    end
end

# Weight building: standard
function _build_weights(ℒ::Directional{Dim}, data, eval_points, adjl, basis) where {Dim}
    v = ℒ.v
    _validate_directional_vector(v, Dim, length(data))
    weights = _build_weights(Jacobian{Dim}(), data, eval_points, adjl, basis)
    return _combine_directional_weights(weights, v, Dim)
end

# Weight building: Hermite
function _build_weights(
    ℒ::Directional{Dim}, data, eval_points, adjl, basis,
    is_boundary, boundary_conditions, normals
) where {Dim}
    v = ℒ.v
    _validate_directional_vector(v, Dim, length(data))

    dim = length(first(data))
    mon = MonomialBasis(dim, basis.poly_deg)
    jacobian_op = Jacobian{Dim}()
    ℒmon = jacobian_op(mon)
    ℒrbf = jacobian_op(basis)

    weights = _build_weights(
        data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
        is_boundary, boundary_conditions, normals
    )

    return _combine_directional_weights(weights, v, Dim)
end

print_op(op::Directional) = "Directional Derivative (∇f⋅v)"
```

**Lines: ~80 (was ~150)**

#### `src/operators/custom.jl` (Complete replacement)

```julia
"""
    Custom{F<:Function} <: AbstractOperator

Custom operator with user-defined function.

The function should accept a basis and return a callable `(x, xᵢ) -> value`.
"""
struct Custom{F<:Function} <: AbstractOperator
    ℒ::F
end
(op::Custom)(basis) = op.ℒ(basis)

"""
    custom(data, ℒ; basis=PHS(3; poly_deg=2), eval_points=data, hermite=nothing, k, adjl)

Build custom operator.

# Arguments
- `data`: Data points
- `ℒ`: Function `ℒ(basis) -> (x, xᵢ) -> value`

# Keyword Arguments
See [`jacobian`](@ref) for keyword arguments.
"""
function custom(data::AbstractVector, ℒ::Function; kw...)
    return build_operator(Custom, data, (ℒ,); kw...)
end

print_op(op::Custom) = "Custom Operator"
```

**Lines: ~25 (was ~75)**

#### `src/operators/gradient.jl` (Complete replacement)

```julia
"""
    gradient(data; kw...)

Build gradient operator.

This is an alias for [`jacobian`](@ref). For scalar fields, the Jacobian IS the gradient.
See [`jacobian`](@ref) for full documentation and examples.
"""
gradient(data::AbstractVector; kw...) = jacobian(data; kw...)

# One-shot application
gradient(data::AbstractVector, x; kw...) = jacobian(data, x; kw...)
```

**Lines: ~12 (was ~100)**

## Summary of Changes

### Line Counts by File

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| `operators.jl` | 73 | 123 | +50 (infrastructure) |
| `jacobian.jl` | 170 | 50 | **-120** |
| `laplacian.jl` | 60 | 18 | **-42** |
| `partial.jl` | 80 | 50 | **-30** |
| `directional.jl` | 150 | 80 | **-70** |
| `custom.jl` | 75 | 25 | **-50** |
| `gradient.jl` | 100 | 12 | **-88** |
| **TOTAL** | **708** | **358** | **-350 lines (49% reduction)** |

### Architectural Benefits

1. **Single RadialBasisOperator constructor**
   - Was: 3 constructors with duplicated logic
   - Now: 1 constructor with hermite dispatch

2. **Type-based dispatch replaces conditionals**
   - `_build_weights_dispatch(::Nothing)` vs `_build_weights_dispatch(::NamedTuple)`
   - No `if isnothing(hermite)` branches needed

3. **3-line build_operator**
   - All complexity pushed to RadialBasisOperator
   - Just constructs operator and delegates

4. **Consistent API across all operators**
   - All use `build_operator(OpType, data, params; kw...)`
   - Keyword arguments everywhere (except required params)

## Migration Guide

### Test File Updates

**Pattern 1: Basic constructor**
```julia
# Before
op = jacobian(data, PHS(3; poly_deg=2))

# After
op = jacobian(data; basis=PHS(3; poly_deg=2))
```

**Pattern 2: With eval_points**
```julia
# Before
op = jacobian(data, eval_points, PHS(3; poly_deg=2))

# After
op = jacobian(data; eval_points=eval_points, basis=PHS(3; poly_deg=2))
```

**Pattern 3: Hermite**
```julia
# Before
op = jacobian(data, eval_points, basis, is_boundary, boundary_conditions, normals)

# After
op = jacobian(data;
              eval_points=eval_points,
              basis=basis,
              hermite=(is_boundary=is_boundary,
                      boundary_conditions=boundary_conditions,
                      normals=normals))
```

### Automated Migration Script

```julia
# scripts/migrate_operators.jl
using TOML

function migrate_operator_call(line, operator)
    # Pattern 1: op(data, basis) → op(data; basis=basis)
    pattern1 = Regex("($operator)\\(([^,]+),\\s*([^;\\)]+)\\)")
    if occursin(pattern1, line)
        return replace(line, pattern1 => s"\1(\2; basis=\3)")
    end

    # Pattern 2: op(data, eval_points, basis) → op(data; eval_points=eval_points, basis=basis)
    pattern2 = Regex("($operator)\\(([^,]+),\\s*([^,]+),\\s*([^;\\)]+)\\)")
    if occursin(pattern2, line)
        return replace(line, pattern2 => s"\1(\2; eval_points=\3, basis=\4)")
    end

    return line
end

function migrate_file(filepath)
    lines = readlines(filepath)
    operators = ["jacobian", "laplacian", "partial", "directional", "custom", "gradient"]

    modified = false
    for (i, line) in enumerate(lines)
        for op in operators
            new_line = migrate_operator_call(line, op)
            if new_line != line
                lines[i] = new_line
                modified = true
                break
            end
        end
    end

    if modified
        println("Migrated: $filepath")
        write(filepath, join(lines, "\n") * "\n")
    end
end

# Run on all test files
for (root, dirs, files) in walkdir("test")
    for file in files
        endswith(file, ".jl") && migrate_file(joinpath(root, file))
    end
end
```

## Implementation Checklist

### Phase 1: Infrastructure (operators.jl)
- [ ] Add `make_operator` methods (5 methods)
- [ ] Add `_build_weights_dispatch` helper
- [ ] Replace 3 RadialBasisOperator constructors with 1 unified version
- [ ] Add `build_operator` helper
- [ ] Test: Create operators directly with new RadialBasisOperator API

### Phase 2: Refactor Operators (one at a time)
- [ ] Replace `jacobian.jl`
- [ ] Run `julia --project=test test/operators/jacobian.jl`
- [ ] Migrate jacobian test calls
- [ ] Replace `laplacian.jl` + migrate tests
- [ ] Replace `partial.jl` + migrate tests
- [ ] Replace `directional.jl` + migrate tests
- [ ] Replace `custom.jl` + migrate tests
- [ ] Replace `gradient.jl` + migrate tests

### Phase 3: Full Integration
- [ ] Run full test suite: `julia --project=test test/runtests.jl`
- [ ] Fix integration test failures
- [ ] Update documentation examples
- [ ] Update CLAUDE.md if needed

## Testing Strategy

### Unit Tests for New Infrastructure

```julia
# test/operators/operator_construction.jl
using RadialBasisFunctions
using StaticArrays
using Test

@testset "make_operator dispatch" begin
    data = [SVector{2}(rand(2)) for _ in 1:10]

    # Dimension inference
    ℒ = make_operator(Jacobian, data)
    @test ℒ isa Jacobian{2}

    # With parameters
    ℒ = make_operator(Partial, data, 1, 2)
    @test ℒ.order == 1
    @test ℒ.dim == 2

    # Direction vector
    v = [1.0, 0.0]
    ℒ = make_operator(Directional, data, v)
    @test ℒ isa Directional{2}
    @test ℒ.v === v
end

@testset "RadialBasisOperator unified constructor" begin
    data = [SVector{2}(rand(2)) for _ in 1:100]

    # Basic
    op = RadialBasisOperator(Laplacian(), data)
    @test op isa RadialBasisOperator

    # With eval_points
    eval_pts = data[1:10]
    op = RadialBasisOperator(Laplacian(), data; eval_points=eval_pts)
    @test length(op.eval_points) == 10

    # Hermite
    is_boundary = fill(false, 10)
    is_boundary[1] = true
    boundary_conditions = [Dirichlet{Float64}() for _ in 1:10]
    normals = [SVector{2}(1.0, 0.0) for _ in 1:10]

    op = RadialBasisOperator(Laplacian(), data;
                             eval_points=eval_pts,
                             hermite=(is_boundary=is_boundary,
                                     boundary_conditions=boundary_conditions,
                                     normals=normals))
    @test op isa RadialBasisOperator
end

@testset "build_operator convenience" begin
    data = [SVector{2}(rand(2)) for _ in 1:100]

    op = build_operator(Jacobian, data, ())
    @test op.ℒ isa Jacobian{2}

    op = build_operator(Partial, data, (1, 2); basis=PHS(5; poly_deg=3))
    @test op.ℒ.order == 1
    @test op.ℒ.dim == 2
    @test op.basis isa PHS{5}
end
```

## Validation

After implementation, verify:

1. **No nested ternaries** - grep for `? .* :` in operator files
2. **Single RadialBasisOperator constructor** - check operators.jl
3. **All operators use build_operator** - check each operator file
4. **No positional basis argument** - grep for `(data,.*basis` patterns
5. **Hermite uses NamedTuple** - grep for `hermite=` in tests
6. **All tests pass** - run full suite

## Final Architecture

```
User calls: jacobian(data; basis=PHS(3))
     ↓
build_operator(Jacobian, data, ())
     ↓
make_operator(Jacobian, data) → Jacobian{2}()  [Dim inferred]
     ↓
RadialBasisOperator(Jacobian{2}(), data; basis=PHS(3))
     ↓
_build_weights_dispatch(..., ::Nothing)  [No hermite]
     ↓
_build_weights(Jacobian{2}(), data, data, adjl, basis)
     ↓
Returns: RadialBasisOperator{Jacobian{2}, ...}
```

Clean, single-path execution with type-based dispatch at each decision point.
