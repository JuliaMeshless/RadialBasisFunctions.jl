# Changelog

All notable changes to RadialBasisFunctions.jl are documented here.

This project follows [Semantic Versioning](https://semver.org/). While the package is pre-1.0, minor version
bumps may contain breaking changes, and breaking changes are made without deprecation shims.

## [0.9.0] — unreleased

Breaking release: the Hermite (symmetric-collocation) boundary-condition path is removed. RadialBasisFunctions is now purely an operator/stencil library — it no longer carries a boundary-condition vocabulary, and it no longer takes surface normals as an operator input. Boundary conditions and their discretization belong to the consuming physics package (e.g. Macchiato.jl); geometry belongs to WhatsThePoint.jl.

### Breaking

- **The `hermite` keyword is removed from every operator constructor.** `laplacian(x; hermite=(is_boundary=…, bc=…, normals=…))` and the equivalent on `partial`, `mixed_partial`, `gradient`, `jacobian`, `hessian`, `directional`, `normal_derivative`, `divergence`, `curl`, `strain_rate`, `rotation_rate`, and `custom` no longer exist. Enforce boundary conditions at the assembly layer instead.
- **The boundary-condition types are removed**: `BoundaryCondition`, `Dirichlet`, `Neumann`, `Robin`, `Internal`, and the predicates `α`, `β`, `is_dirichlet`, `is_neumann`, `is_robin`, `is_internal`. This also retires the `α`/`β` export collision with downstream physics packages.
- **The Hermite machinery is removed**: `HermiteStencilData`, `update_hermite_stencil_data!`, `BoundaryData`, `classify_stencil`, `InteriorStencil`, `DirichletStencil`, `HermiteStencil`, and `construct_global_to_boundary`. `src/solve/types.jl` is deleted; its two surviving arity helpers moved to `src/solve/execution.jl`.
- **The three-argument normal-form basis functors are removed** — `(op::∂{<:PHS})(x, xᵢ, normal)` and the `∇`/`∂²`/`∇²` equivalents across PHS1/3/5/7 (16 methods), along with `∂_Hermite` on the monomial basis and the `(x, xᵢ, normal)` methods on `SumKernel`/`ScaledKernel`. `∂_normal` / `∂_normal!` are kept: a normal is a direction, not a boundary condition.

### Changed

- `_build_collocation_matrix!`, `_build_rhs!`, and the weight kernel lost their Hermite dispatch layer, which collapsed each of `_rbf_entry`/`_poly_entry!`/`_rbf_rhs`/`_mono_rhs!` to a single method and let them inline. `src/solve/assembly.jl` goes 468 → 143 lines with no behavioral change to interior stencils.
- Numerics are unchanged: identical weights on every non-Hermite path, verified by the full test suite.

## [0.8.0] — 2026-08-24

Breaking release: operator weights moved from `SparseMatrixCSC` to stencil-wise (ELL) storage, making evaluation multithreaded/GPU-capable and ~7.6× faster at N = 100k, k = 50 on 13 threads (~1.5× single-threaded). See [#156].

### Breaking

- **The Mooncake extension is removed; Enzyme is the sole supported AD backend.** The evaluation kernels are multithreaded and cannot be traced generically by reverse-mode backends, so AD support means maintained rules — and those are Enzyme's (`EnzymeRules`, loaded via the package extension). Native rules now also cover the bare weight matvec (`W * x`, `weights(op) * x`), including weight cotangents when the weights are built inside the differentiated region. In-place evaluation (`op(y, x)`, `mul!(y, op, x)`) is not differentiable — use the out-of-place forms in losses.
- **Operator weights are now `StencilWeights`, not `SparseMatrixCSC`.** The new exported type stores a dense k × N_eval value matrix (`parent(W)`, column i = eval point i's stencil in `adjl` order) plus a shared k × N_eval `Int32` neighbor-index matrix; the logical size stays `(N_eval, N_data)`. `weights(op)` returns it (or an `NTuple` of them); use the new `sparse(op)` / `SparseMatrixCSC(op)` for global system assembly and implicit solves (`sparse(op) \ rhs`). `VirtualPartial` operators keep sparse weights internally (their two stencil sets don't share one ELL structure).
- **In-place weight mutation goes through `parent`.** `op.weights .= …` no longer works (`StencilWeights` has a fixed stencil structure and no `setindex!`); mutate `parent(op.weights)` instead. `update_weights!` now rewrites the value matrix in place with no sparse reassembly.
- **Single-eval-point operators keep the leading eval dimension.** With `N_eval == 1`, gradient-family and Hessian-family results are `1×D` / `1×D×D` arrays (and divergence/2D-curl a length-1 vector) instead of the previously collapsed `Vector`/`Matrix`/scalar shapes, and their weight components are `1×N` `StencilWeights` instead of `SparseVector`s.
- **AD losses over built weights read `parent`.** `sum(W.nzval .^ 2)` becomes `sum(parent(W) .^ 2)` — the cotangent lands directly in the stencil-major value matrix, which also simplified both AD extensions.
- **Stricter input validation.** A user-supplied ragged `adjl` (stencils of differing lengths) now throws `ArgumentError` instead of silently corrupting the weight buffers, and Hermite (boundary-condition) operators require `eval_points` to be the same point set as `data` — previously that combination produced silently wrong identity rows or a `BoundsError`.

### Added

- `StencilWeights` ELL weight storage with a row-parallel apply kernel — `Threads.@threads` + SIMD on CPU (with a serial fast path below ~4k rows), a KernelAbstractions kernel on GPU backends — plus a deterministic transpose-map adjoint apply (`W' * x` gathers through a precomputed `EllTransposeMap`; no atomics, GPU-capable, also the AD pullback path), `Matrix`/`sparse` conversions, stencil-preserving algebra (`+`, `-`, scalar scaling, `Diagonal *`; mixing with sparse matrices stays sparse), and `Adapt` support that moves the value, index, and transpose-map arrays to the device — shared once across gradient-family components — so `cu(op)` evaluation now genuinely runs on the GPU (weight *building* remains CPU-only, [#88]).
- `sparse(op)` / `SparseMatrixCSC(op)` interop conversions. Dirichlet identity rows convert back to single-entry rows exactly as the old storage stored them.
- Benchmarks: a `Laplacian` group comparing the ELL apply kernel against the former CSC matvec.

[#88]: https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/88
[#156]: https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/156

## [0.7.1] — 2026-08-11

### Fixed

- **2D `laplacian()` operators were wrong for every PHS basis.** The fused ∇² functors for
  PHS1/3/5/7 (both the 2-arg and the 3-arg Hermite variants) hardcoded the 3D constants of
  Δrᵏ = k(k+d−2)·r^(k−2) — e.g. 12r instead of 9r for PHS3 in 2D. The per-dimension ∂²
  functors were already correct, so summing `partial(…, 2, dim)` operators or taking the
  trace of `hessian` was unaffected; only the fused `laplacian()` path was wrong in 2D.
  Found by cross-verifying RBF-FD Poisson weights against KernelInterpolation.jl (ours came
  out exactly 4/3 × theirs for PHS3). Regression tests now pin ∇² to
  tr(ForwardDiff.hessian) and Σ∂² in both 2D and 3D.

- **AD gradients through 2D `laplacian()` weights used the same 3D-hardcoded constants.**
  The hand-written backward-pass gradients of ∇²φ for PHS1/3/5/7 in
  `grad_applied_laplacian_wrt_x` — shared by the Enzyme and Mooncake rules — were not
  updated alongside the primal fix above, so differentiating `_build_weights` with a 2D
  Laplacian gave gradients off by the constant ratio (e.g. 4/3 for PHS3). They are now
  dimension-aware, and a regression test pins them (and the Partial-operator gradients)
  against ForwardDiff of the basis functors in both 2D and 3D.

## [0.7.0] — 2026-08-03

### Breaking

- `Dirichlet`, `Neumann`, and `Robin` are no longer exported (they remain public API).
  Downstream physics packages (e.g. Macchiato.jl) export their own boundary-condition
  vocabularies using these names, so co-loading both packages made the bare names
  ambiguous. Import them explicitly: `using RadialBasisFunctions: Dirichlet, Neumann, Robin`.
  `BoundaryCondition` and `Internal` remain exported.

### Added

- Public `weights(op)` accessor for `RadialBasisOperator` — returns the stencil weight
  matrix (or tuple of matrices for gradient-family operators), rebuilding first when the
  cache is stale. Downstream assemblers should use it instead of reaching into the
  `weights` field.
- Scalar algebra on built operators: `α * op`, `op * α`, `op / α`, and `-op` now work on
  `RadialBasisOperator`, scaling the existing weights without re-collocation. Previously
  scalars combined only with symbolic operators (`α * Laplacian()`), which pushed users to
  manipulate `op.weights` by hand.

- Enzyme rule for the `Interpolator` constructor, plus `Duplicated`/`MixedDuplicated` evaluation rules
  for interpolators built inside a differentiated region — Enzyme now matches Mooncake on every AD
  path ([#147]). Enzyme is the recommended default backend; the autodiff guide leads with
  `DI.AutoEnzyme(; function_annotation=Enzyme.Const)`.

### Fixed

- **Mixed partials were silently wrong at the default `poly_deg = 2` in 2D and 3D.** The
  hand-coded monomial evaluators (`MonomialBasis{2,2}`, `MonomialBasis{3,2}`) order terms
  differently from the generic multiexponents pipeline that `_∂mixed` relied on, so the
  ∂²/(∂xᵢ∂xⱼ) monomial action landed in the wrong slot — `mixed_partial(x, 1, 2)` mapped
  f = xy to ≈ 0 instead of 1. Affected `mixed_partial`, `hessian` off-diagonal terms, and
  `@operator ∂(i,j)` for any basis with degree-2 polynomial augmentation in 2 or 3
  dimensions; degree ≥ 3 was unaffected (evaluator and derivative pipeline share the
  generic ordering there), which is why the existing tests — all at `poly_deg = 4` — never
  caught it. Hand-coded `_∂mixed` specializations now mirror the evaluator orderings, and
  regression tests cover the default degree. Found by comparing against Macchiato.jl's
  workaround (`_ℒ_mixed_partial`), which existed precisely because of this ordering
  mismatch.
- The Enzyme test suite is fully green on Julia 1.10/1.11 (previously `@test_broken`, [#150]) and runs
  un-gated on Julia 1.12 (requires Enzyme ≥ 0.13.190 in the test/docs environments).
- The `Interpolator` constructor factorizes the collocation matrix with Bunch-Kaufman instead of the
  Union-typed generic `factorize`, which broke Enzyme's type analysis and cost dynamic dispatch. The
  Mooncake constructor pullback reuses the cached factorization (O(n²) solve instead of O(n³)).
- Enzyme shape-parameter tangents are constructed without the `ε > 0` validation, which threw for
  negative gradients on the Active-basis path.

## [0.6.0] — 2026-07-17

Breaking release. The operator constructor surface was simplified and several exported types changed shape.

### Breaking

#### Positional `eval_points` and positional Hermite constructor tiers removed

Removed from `partial`, `laplacian`, `gradient`, `jacobian`, `directional`, `mixed_partial`, and `hessian`
([#141], `8f374bd`), and later in the same release from `RadialBasisOperator` itself and `custom`, so the
`eval_points` keyword is now the only way to supply evaluation points. `divergence`, `curl`, `strain_rate`,
and `rotation_rate` were not affected. `regrid(data, eval_points)` keeps its positional form — there it is
the primary source→target signature, not a back-compat tier. The keyword constructors and the short
trailing-basis forms (`partial(data, order, dim, basis)`, `RadialBasisOperator(ℒ, data, basis)`,
`custom(data, ℒ, basis)`, etc.) are unchanged.

```julia
# before (0.5.x)                                    # after (0.6.0)
laplacian(data, eval_points, basis)                 laplacian(data; eval_points, basis)
partial(data, eval_points, order, dim, basis)       partial(data, order, dim; eval_points, basis)
directional(data, eval_points, v, basis)            directional(data, v; eval_points, basis)
mixed_partial(data, eval_points, i, j, basis)       mixed_partial(data, i, j; eval_points, basis)
gradient(data, eval_points, basis)                  gradient(data; eval_points, basis)
RadialBasisOperator(ℒ, data, eval_points, basis)    RadialBasisOperator(ℒ, data; eval_points, basis)
custom(data, eval_points, ℒ, basis)                 custom(data, ℒ; eval_points, basis)

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
[#147]: https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/147
[#150]: https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/150
