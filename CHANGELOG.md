# Changelog

All notable changes to RadialBasisFunctions.jl are documented here.

This project follows [Semantic Versioning](https://semver.org/). While the package is pre-1.0, minor version
bumps may contain breaking changes, and breaking changes are made without deprecation shims.

## [0.6.0] — unreleased

Breaking release. The operator constructor surface was simplified and several exported types changed shape.

### Breaking

#### Positional `eval_points` and positional Hermite constructor tiers removed

Removed from `partial`, `laplacian`, `gradient`, `jacobian`, `directional`, `mixed_partial`, and `hessian`
([#141], `8f374bd`). `divergence`, `curl`, `strain_rate`, `rotation_rate`, `custom`, and `regrid` were not
affected. The keyword constructors and the short trailing-basis forms (`partial(data, order, dim, basis)` etc.)
are unchanged, as are the positional and Hermite constructors on `RadialBasisOperator` itself.

```julia
# before (0.5.x)                                    # after (0.6.0)
laplacian(data, eval_points, basis)                 laplacian(data; eval_points, basis)
partial(data, eval_points, order, dim, basis)       partial(data, order, dim; eval_points, basis)
directional(data, eval_points, v, basis)            directional(data, v; eval_points, basis)
mixed_partial(data, eval_points, i, j, basis)       mixed_partial(data, i, j; eval_points, basis)
gradient(data, eval_points, basis)                  gradient(data; eval_points, basis)

# Hermite (6–8 positional args) → the `hermite` keyword
laplacian(data, eval_points, basis,                 laplacian(data; eval_points, basis,
          is_boundary, bcs, normals)                    hermite = (is_boundary = is_boundary,
                                                                   bc = bcs, normals = normals))
```

**On the rationale.** [#141]'s description attributes this removal to a dispatch collision between
`op(data, eval_points)` and the one-shot `op(data, x)`. That is not accurate, and the record is corrected here:
at v0.5.0 the eval tier already typed `eval_points::AbstractVector{<:AbstractVector}`, which is strictly more
specific than the one-shot's untyped `x`, so `jacobian(data, eval_points)` resolved correctly and no ambiguity
existed. The silent misdispatch described in that PR was introduced by the removal itself and contained two
commits later by constraining the one-shots (see below).

The tiers remain removed for API clarity, and because leaving the second positional slot free preserves the
option to support `Vector{SVector{D,T}}` as a vector-field representation in a future release — which would
otherwise make a field and a point set indistinguishable by type.

#### One-shot convenience forms narrowed

`gradient`, `jacobian`, and `hessian` one-shots now require the field argument to be an
`AbstractArray{<:Number}` (`fa6e2ca`; `src/operators/gradient.jl:54`, `jacobian.jl:89`, `hessian.jl:70`).
Previously the argument was untyped. Calls passing a non-numeric-eltype array now raise a `MethodError` at the
call site instead of failing inside sparse matrix multiplication. `mixed_partial`'s one-shot was already typed
and is unchanged.

#### `∂virtual` returns a `RadialBasisOperator`

`∂virtual` previously returned a bare closure `x -> w * x`; it now returns a lazy
`RadialBasisOperator{<:VirtualPartial}` (`f0c9174`; `src/operators/virtual.jl`). Applying the result with
`op(u)` still works and the numerics are unchanged. Code that stored, type-annotated, or dispatched on the
result as a `Function` must be updated.

#### `HermiteStencilData` field layout changed

Two fields were added (`normal_workspace`, `eval_local_idx`) and the `data` field was narrowed from
`AbstractVector{Vector{T}}` to `Vector{Vector{T}}` (`9c332d0`, `c933dde`; `src/solve/types.jl`). The documented
constructor signature is unchanged, so idiomatic construction is unaffected; code relying on the internal field
count or ordering (positional `new`, destructuring, field iteration) must be updated. A length-mismatch check
also changed from an `AssertionError` to a `DimensionMismatch`.

`update_hermite_stencil_data!` gained a trailing optional positional argument `eval_point = nothing`
(backward compatible).

#### `Regrid` is now a fieldless singleton

`struct Regrid; ℒ::typeof(identity); end` became `struct Regrid end`, and its call method changed from
`(op::Regrid)(x) = op.ℒ(x)` to `(::Regrid)(basis) = basis` (`src/operators/regridding.jl:6-7`). `Regrid()` still
constructs, and the exported `regrid(...)` constructors are unchanged. Only `Regrid().ℒ` field access breaks;
`Regrid()(x)` already raised a `MethodError` before this change (`0990419`).

#### `reorder_points!` returns the permutation

The three-argument method now returns the SymRCM permutation vector instead of `nothing` (`3c9986f`;
`src/utils.jl:45`). Its type parameter was fixed in the same commit
(`AbstractVector{AbstractVector{T}}` → `AbstractVector{<:AbstractVector{T}}`), meaning the exported method had
always raised a `MethodError` and now dispatches correctly.

#### Error types changed

Unsupported Hermite Neumann/Robin normal forms and unsupported backward polynomial dimensions now raise
`ArgumentError` at a single choke point instead of a deep `MethodError` (`9c332d0`, `5cca8f6`). This affects
only code catching specific exception types.

### Added

Nine new exports, all additive:

- `autoselect_k` — the stencil-size heuristic used as the default for `k`.
- `VirtualPartial`, `SumOperator` — new operator types.
- `output_rank`, `requires_vector_input`, `is_symmetric`, `is_antisymmetric`, `is_self_adjoint`,
  `derivative_order` — operator trait predicates.

No exports were removed or renamed.

### Changed (non-breaking)

- `StaticArrays` and `ChunkSplitters` were removed from `[deps]`; `StaticArraysCore` remains. **This has no
  user-visible effect.** A package's `[deps]` are not transitively loadable by downstream users — `using
  StaticArrays` has always required `StaticArrays` in the downstream project's own `[deps]` — and
  RadialBasisFunctions has never re-exported `SVector`.
- `[compat]` entries were added for `Random` and `SparseArrays`, which were previously listed in `[deps]` with
  no bounds.
- Internal: the `AbstractGradientOperator` family was introduced and renamed to `AbstractJacobianOperator`
  (`442bfca`, `d67a959`). Both are unexported; no API change.

### Fixed

- Hot-loop allocations in the weight-building and Hermite paths.
- Float32 type genericity throughout the operator and solve layers.
- AD backward pass reuses a per-stencil scratch workspace (`221aa53`).

### Known issues

- The Enzyme.jl autodiff extension has failing tests on Julia 1.10 and 1.11, marked `@test_broken` and tracked
  in [#150]. The Mooncake backend is unaffected.

[#141]: https://github.com/JuliaMeshless/RadialBasisFunctions.jl/pull/141
[#150]: https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/150
