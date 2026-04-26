# AD Extension: MixedPartial and Directional

This document specifies the two changes needed in RadialBasisFunctions.jl before the
shape optimization pipeline in Macchiato.jl can be implemented. It includes the full
mathematical derivations, the exact files to modify, and the tests that must pass before
the work is considered done.

Context: Macchiato.jl needs to differentiate through its PDE assembly, which for 2D
linear elasticity uses five `_build_weights` calls: `Partial(2,1)`, `Partial(2,2)`,
`MixedPartial(1,2)`, `Partial(1,1)`, and `Partial(1,2)`. The first four are also the
correct operators for the stress computation. `Partial` and `Laplacian` already have
Mooncake rrule!!s. `MixedPartial` and `Directional` do not.

---

## Change 1: `MixedPartial` — add backward infrastructure

### Why it is needed

`_build_weights(MixedPartial(1, 2), pts, pts, adjl, basis)` has no `@is_primitive`
registration and no `rrule!!`. When Mooncake traces the loss, it falls through to the
generic `_build_weights` implementation in `assembly.jl`, which calls `bunchkaufman!`
— a LAPACK foreign call that Mooncake cannot trace. The trace crashes at the first
call to `_build_weights(MixedPartial(...), ...)`.

### How the existing stencil IFT backward works

For reference, look at the existing `Partial` backward. The stencil solve is:

```
A_stencil · λ = b(ℒ, eval_point, neighbors)
```

`A_stencil` is the RBF collocation matrix — the same for all operators. `b` is the
vector of operator-applied basis values at the eval point:

```
b[j] = ℒφ(eval_point, data[neighbor_j])
```

The IFT backward (`backward_linear_solve!` in `solve/backward.jl`) computes `η =
A^{-T} · ∂L/∂λ`, then:

- `∂L/∂b = η` — this chains through the RHS into `∂L/∂pts` via `grad_Lφ_x` and
  `grad_Lφ_xi` (the derivatives of `ℒφ(x, xi)` w.r.t. `x` and `xi`)
- `∂L/∂A` chains through the collocation matrix via `backward_collocation!` —
  this is already general and requires no per-operator change

For `MixedPartial`, the operator applied to the basis is `∂²φ/(∂x_{d1}∂x_{d2})`. The
backward needs its derivative w.r.t. the evaluation and data point positions — these are
`grad_applied_mixed_partial_wrt_x` and `grad_applied_mixed_partial_wrt_xi`.

### Mathematical derivations: `grad_applied_mixed_partial_wrt_x`

For all derivations, `dim1 ≠ dim2` (when equal, `MixedPartial` reduces to `Partial(2,d)`
and is already handled). Let `δ = x - xi`, `δ_d = x[d] - xi[d]`, `r = ‖x - xi‖`.

The function to differentiate in the backward is `∂²φ/(∂x_{d1}∂x_{d2})(x, xi)` with
respect to `x[j]`. The formula for each PHS order and smooth basis follows.

**PHS3** — φ(r) = r³

`∂²φ/(∂x_{d1}∂x_{d2})` for `d1 ≠ d2`:
```
∂φ/∂x_{d1} = 3·δ_{d1}·r
∂²φ/(∂x_{d1}∂x_{d2}) = 3·δ_{d1}·δ_{d2}/r
```

Derivative w.r.t. `x[j]` of `3·δ_{d1}·δ_{d2}/r`:
```
j == d1:  3·δ_{d2}·(1/r - δ_{d1}²/r³)
j == d2:  3·δ_{d1}·(1/r - δ_{d2}²/r³)
else:    -3·δ_{d1}·δ_{d2}·δ_{j}/r³
```

**PHS1** — φ(r) = r

```
∂²φ/(∂x_{d1}∂x_{d2}) = -δ_{d1}·δ_{d2}/r³
```

Derivative w.r.t. `x[j]`:
```
j == d1: -δ_{d2}/r³ + 3·δ_{d1}²·δ_{d2}/r⁵
j == d2: -δ_{d1}/r³ + 3·δ_{d1}·δ_{d2}²/r⁵
else:     3·δ_{d1}·δ_{d2}·δ_{j}/r⁵
```

Note: return `zero(x)` when `r < 1e-12` (same guard as `grad_partial_phs1_wrt_x`).

**PHS5** — φ(r) = r⁵

```
∂²φ/(∂x_{d1}∂x_{d2}) = 15·δ_{d1}·δ_{d2}·r
```

Derivative w.r.t. `x[j]`:
```
j == d1:  15·δ_{d2}·(r + δ_{d1}²/r)
j == d2:  15·δ_{d1}·(r + δ_{d2}²/r)
else:     15·δ_{d1}·δ_{d2}·δ_{j}/r
```

**PHS7** — φ(r) = r⁷

```
∂²φ/(∂x_{d1}∂x_{d2}) = 35·δ_{d1}·δ_{d2}·r³
```

Derivative w.r.t. `x[j]`:
```
j == d1:  35·δ_{d2}·r·(r² + 3·δ_{d1}²)
j == d2:  35·δ_{d1}·r·(r² + 3·δ_{d2}²)
else:     105·δ_{d1}·δ_{d2}·δ_{j}·r
```

**IMQ** — φ(r) = 1/√(ε²r²+1), let `s = ε²r² + 1`

```
∂²φ/(∂x_{d1}∂x_{d2}) = 3ε⁴·δ_{d1}·δ_{d2}/s^(5/2)
```

Derivative w.r.t. `x[j]`:
```
j == d1:  3ε⁴·δ_{d2}·(1/s^(5/2) - 5ε²·δ_{d1}²/s^(7/2))
j == d2:  3ε⁴·δ_{d1}·(1/s^(5/2) - 5ε²·δ_{d2}²/s^(7/2))
else:    -15ε⁶·δ_{d1}·δ_{d2}·δ_{j}/s^(7/2)
```

**Gaussian** — φ(r) = exp(-ε²r²)

```
∂²φ/(∂x_{d1}∂x_{d2}) = 4ε⁴·δ_{d1}·δ_{d2}·φ
```

Derivative w.r.t. `x[j]`:
```
j == d1:  4ε⁴·φ·δ_{d2}·(1 - 2ε²·δ_{d1}²)
j == d2:  4ε⁴·φ·δ_{d1}·(1 - 2ε²·δ_{d2}²)
else:    -8ε⁶·φ·δ_{d1}·δ_{d2}·δ_{j}
```

**`grad_applied_mixed_partial_wrt_xi`**: by the same antisymmetry used for `Partial`,
this is always the negation of `grad_applied_mixed_partial_wrt_x`:
```julia
grad_applied_mixed_partial_wrt_xi(b, d1, d2) = negate_grad(grad_applied_mixed_partial_wrt_x(b, d1, d2))
```

### Files to modify

#### `src/solve/operator_second_derivatives.jl`

Add functions:
- `grad_mixed_partial_phs1_wrt_x(dim1, dim2)`
- `grad_mixed_partial_phs3_wrt_x(dim1, dim2)`
- `grad_mixed_partial_phs5_wrt_x(dim1, dim2)`
- `grad_mixed_partial_phs7_wrt_x(dim1, dim2)`
- `grad_mixed_partial_imq_wrt_x(ε, dim1, dim2)`
- `grad_mixed_partial_gaussian_wrt_x(ε, dim1, dim2)`

Then the dispatch function:
```julia
grad_applied_mixed_partial_wrt_x(::PHS1, d1, d2) = grad_mixed_partial_phs1_wrt_x(d1, d2)
grad_applied_mixed_partial_wrt_x(::PHS3, d1, d2) = grad_mixed_partial_phs3_wrt_x(d1, d2)
grad_applied_mixed_partial_wrt_x(::PHS5, d1, d2) = grad_mixed_partial_phs5_wrt_x(d1, d2)
grad_applied_mixed_partial_wrt_x(::PHS7, d1, d2) = grad_mixed_partial_phs7_wrt_x(d1, d2)
grad_applied_mixed_partial_wrt_x(b::IMQ, d1, d2) = grad_mixed_partial_imq_wrt_x(b.ε, d1, d2)
grad_applied_mixed_partial_wrt_x(b::Gaussian, d1, d2) = grad_mixed_partial_gaussian_wrt_x(b.ε, d1, d2)
grad_applied_mixed_partial_wrt_xi(b, d1, d2) = negate_grad(grad_applied_mixed_partial_wrt_x(b, d1, d2))
```

#### `src/solve/ad_shared.jl`

Add three dispatch methods, following the exact pattern of the existing `Partial` entries:

```julia
_optype(::MixedPartial) = MixedPartial

_get_grad_funcs(::Type{<:MixedPartial}, basis, ℒ) = (
    grad_applied_mixed_partial_wrt_x(basis, ℒ.dim1, ℒ.dim2),
    grad_applied_mixed_partial_wrt_xi(basis, ℒ.dim1, ℒ.dim2),
)

function _get_rhs_closures(::Type{<:MixedPartial}, ℒ, basis)
    d1, d2 = ℒ.dim1, ℒ.dim2
    poly_backward! = (Δeval_point, Δb, k, nmon, num_ops) ->
        _backward_mixed_partial_polynomial_section!(Δeval_point, Δb, k, nmon, d1, d2, num_ops)
    ∂Lφ_∂ε_fn = (x, xi) -> ∂MixedPartial_φ_∂ε(basis, d1, d2, x, xi)
    return poly_backward!, ∂Lφ_∂ε_fn
end
```

`_backward_mixed_partial_polynomial_section!` chains the cotangent of the mixed-partial
monomial RHS back to the eval point. This is the third-order polynomial backward and
follows the same structure as `_backward_partial_polynomial_section!` but applies the
mixed second derivative to the monomial gradient. For PHS bases (where ε is not a free
parameter), `∂Lφ_∂ε_fn` can return `zero(eltype(x))`.

#### `ext/RadialBasisFunctionsMooncakeExt/RadialBasisFunctionsMooncakeExt.jl`

Add import at the top:
```julia
import RadialBasisFunctions: MixedPartial
```

Register the primitive for both PHS and smooth bases, and add the `rrule!!`. The
`rrule!!` is identical in structure to the existing `Partial`/`Laplacian` ones — only
the `OpType` passed to `_mooncake_build_weights_forward` and
`_mooncake_build_weights_pullback` changes:

```julia
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(_build_weights), <:MixedPartial, AbstractVector, AbstractVector, AbstractVector, <:AbstractRadialBasis}

function Mooncake.rrule!!(
    ::CoDual{typeof(_build_weights)},
    op::CoDual{<:MixedPartial},
    data::CoDual{<:AbstractVector},
    eval_points::CoDual{<:AbstractVector},
    adjl::CoDual{<:AbstractVector},
    basis::CoDual{<:AbstractPHS},
)
    ℒ, pts, eval_pts, adj, bas = primal(op), primal(data), primal(eval_points), primal(adjl), primal(basis)
    OpType = _optype(ℒ)   # returns MixedPartial
    W, W_codual, cache, mon, dim_space = _mooncake_build_weights_forward(ℒ, pts, eval_pts, adj, bas, OpType)
    shared_pb!! = _mooncake_build_weights_pullback(W_codual.dx, W, data, eval_points, cache, adj, pts, eval_pts, bas, mon, ℒ, OpType, dim_space)

    function pb!!(ΔW_rdata)
        shared_pb!!(ΔW_rdata)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return W_codual, pb!!
end
```

A separate `rrule!!` overload for IMQ/Gaussian bases is needed if the shape parameter ε
must be differentiable (same split as for `Partial` — see existing code). For PHS bases
the single `rrule!!` above suffices.

---

## Change 2: `Directional` — rewrite `_build_weights` to compose `Partial` calls

### Design rationale

The current implementation of `_build_weights(ℒ::Directional, ...)` in
`src/operators/directional.jl` routes through `_build_weights(Jacobian{Dim}(), ...)` to
build all `Dim` first-derivative weight matrices in one stencil solve, then combines
them with the direction vector:

```julia
# current implementation
function _build_weights(ℒ::Directional{Dim}, data, eval_points, adjl, basis) where {Dim}
    weights = _build_weights(Jacobian{Dim}(), data, eval_points, adjl, basis)
    return _combine_directional_weights(weights, v, Dim)
end
```

`Jacobian` has no Mooncake `rrule!!`. When Mooncake encounters this call it cannot trace
through the stencil solver, so **any loss that assembles a Neumann BC row via
`normal_derivative` or `directional` produces no gradient w.r.t. point positions**.
The gradient silently terminates.

The fix replaces the Jacobian call with `Dim` explicit calls to `_build_weights(Partial(1,
d), ...)`, each of which already has a Mooncake rrule!!:

```julia
# new implementation
function _build_weights(ℒ::Directional{Dim}, data, eval_points, adjl, basis; device=CPU()) where {Dim}
    v = ℒ.v
    _validate_directional_vector(v, Dim, length(data))
    return _combine_partial_weights(v, data, eval_points, adjl, basis, Dim; device=device)
end

function _combine_partial_weights(v, data, eval_points, adjl, basis, Dim; device=CPU())
    if length(v) == Dim
        # constant direction: weighted sum of Partial weight matrices
        return sum(1:Dim) do d
            v[d] * _build_weights(Partial(1, d), data, eval_points, adjl, basis; device=device)
        end
    else
        # spatially varying direction: per-row diagonal scaling
        return sum(1:Dim) do d
            vd = Diagonal(getindex.(v, d))
            vd * _build_weights(Partial(1, d), data, eval_points, adjl, basis; device=device)
        end
    end
end
```

### Why this is correct

`W_{∂/∂v} f = (∇f) · v = sum_d v[d] · (∂f/∂x_d)`, so the directional derivative weight
matrix is exactly `sum_d v[d] · W_{∂x_d}` (or, for spatially varying `v`, with
`v_i[d]` scaling row `i`). This is exactly what `_combine_directional_weights` already
computes — the change is only in how the per-dimension weight matrices are obtained.

The existing Jacobian path solves one stencil system per point to obtain all `Dim`
partial derivative vectors simultaneously, which is more efficient. The new path solves
`Dim` independent stencil systems. For 2D this is 2× the work; for 3D it is 3×.
The stencils are identical across the `Dim` calls (same collocation matrix), so in
principle the LU factorization could be shared — but this optimisation is not required
for correctness and can be deferred.

### What Mooncake now traces

With the new implementation, Mooncake sees `Dim` calls to `_build_weights(Partial(1,d),
...)`, each of which is a registered `@is_primitive` with a complete rrule!!. The linear
combination `v[d] * W_∂d` is traced automatically.

This produces two correct gradient flows:

1. `∂L/∂pts_stencil` — through the existing `Partial` stencil IFT, covering how moving
   a point changes the weights of the stencils centred on or near it.

2. `∂L/∂v[d]` — through the `v[d] * W_∂d` multiplication, flowing back to the
   direction vector coefficients. For shape optimisation this feeds the chain
   `∂L/∂v → ∂L/∂n(pts) → ∂L/∂pts_geometry`, which is handled in Macchiato.jl (not here).

### File to modify

`src/operators/directional.jl` — replace the body of the non-Hermite
`_build_weights(ℒ::Directional, ...)` overload (lines 125–130). The Hermite overload
(lines 132–169) is a separate code path and is not used in the Macchiato shape
optimization pipeline; leave it unchanged for now.

---

## Tests

Add all new tests to `test/extensions/mooncake_ext.jl` inside a new top-level
`@testset` block, and mirror them in `test/extensions/autodiff_di.jl` using the
`MOONCAKE_BACKEND` constant already defined there. Also add the new loss generators to
`test/extensions/ad_test_utils.jl`.

### New test data helper

Add to `ad_test_utils.jl`:

```julia
"""
    make_directional_test_data(; N=25, k=10)

Returns (points, N, adjl, pts_flat, normals_flat, normals_varying).
`normals_flat` is a flat Vector{Float64} of length 2N for testing ∂L/∂v.
"""
function make_directional_test_data(; N=25, k=10)
    points = [SVector{2}(0.1 + 0.8*i/5, 0.1 + 0.8*j/5) for i in 1:5 for j in 1:5]
    adjl = RadialBasisFunctions.find_neighbors(points, k)
    pts_flat = vcat([collect(p) for p in points]...)
    # Constant direction (normalised)
    n_const = [1.0/√2, 1.0/√2]
    # Spatially varying directions (each row normalised)
    n_varying = [normalize(SVector{2}(sin(i*0.3), cos(i*0.4))) for i in 1:N]
    return points, N, adjl, pts_flat, n_const, n_varying
end
```

### Test 1: `MixedPartial` gradient w.r.t. point positions

```julia
@testset "Mooncake Extension - MixedPartial _build_weights" begin
    points, N, adjl, pts_flat = make_build_weights_test_data()

    @testset "MixedPartial(1,2) with PHS3" begin
        loss = make_build_weights_loss(MixedPartial(1, 2), adjl, PHS(3; poly_deg=2), N)
        rule = Mooncake.build_rrule(loss, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
        validate_gradient(dpts, loss, pts_flat; rtol=1e-3)
    end

    @testset "MixedPartial(1,2) with PHS1/5/7" begin
        for n in [1, 5, 7]
            loss = make_build_weights_loss(MixedPartial(1, 2), adjl, PHS(n; poly_deg=2), N)
            rule = Mooncake.build_rrule(loss, pts_flat)
            _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
            validate_gradient(dpts, loss, pts_flat; rtol=1e-2)
        end
    end

    @testset "MixedPartial(1,2) with IMQ and Gaussian" begin
        for (name, basis) in [("IMQ", IMQ(1.0; poly_deg=2)), ("Gaussian", Gaussian(1.0; poly_deg=2))]
            loss = make_build_weights_loss(MixedPartial(1, 2), adjl, basis, N)
            rule = Mooncake.build_rrule(loss, pts_flat)
            _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
            validate_gradient(dpts, loss, pts_flat; rtol=1e-3)
        end
    end

    @testset "MixedPartial in 3D: (1,2), (1,3), (2,3)" begin
        N3 = 64
        pts3 = [SVector{3}(0.1+0.8*((i*7+3)%N3)/N3,
                           0.1+0.8*((i*11+5)%N3)/N3,
                           0.1+0.8*((i*13+7)%N3)/N3) for i in 1:N3]
        adj3 = RadialBasisFunctions.find_neighbors(pts3, 20)
        flat3 = vcat([collect(p) for p in pts3]...)
        for (d1, d2) in [(1,2), (1,3), (2,3)]
            loss = make_build_weights_loss(MixedPartial(d1, d2), adj3, PHS(3; poly_deg=2), N3)
            rule = Mooncake.build_rrule(loss, flat3)
            _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, flat3)
            validate_gradient(dpts, loss, flat3; rtol=1e-3)
        end
    end
end
```

### Test 2: `Directional` gradient w.r.t. point positions (constant direction)

```julia
@testset "Mooncake Extension - Directional _build_weights" begin
    points, N, adjl, pts_flat, n_const, n_varying = make_directional_test_data()

    @testset "Constant direction - gradient w.r.t. pts" begin
        function loss_dir_const(pts)
            pts_vec = [SVector{2}(pts[2i-1], pts[2i]) for i in 1:N]
            W = RadialBasisFunctions._build_weights(
                    RadialBasisFunctions.Directional{2}(n_const),
                    pts_vec, pts_vec, adjl, PHS(3; poly_deg=2))
            return sum(W.nzval .^ 2)
        end
        rule = Mooncake.build_rrule(loss_dir_const, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_dir_const, pts_flat)
        validate_gradient(dpts, loss_dir_const, pts_flat; rtol=1e-3)
    end

    @testset "Spatially varying direction - gradient w.r.t. pts" begin
        function loss_dir_varying(pts)
            pts_vec = [SVector{2}(pts[2i-1], pts[2i]) for i in 1:N]
            W = RadialBasisFunctions._build_weights(
                    RadialBasisFunctions.Directional{2}(n_varying),
                    pts_vec, pts_vec, adjl, PHS(3; poly_deg=2))
            return sum(W.nzval .^ 2)
        end
        rule = Mooncake.build_rrule(loss_dir_varying, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_dir_varying, pts_flat)
        validate_gradient(dpts, loss_dir_varying, pts_flat; rtol=1e-3)
    end

    @testset "Constant direction - gradient flows into direction vector" begin
        # Flatten the direction vector and differentiate w.r.t. it
        # (verifies the v[d]*W_∂d chain works)
        pts_fixed = [SVector{2}(0.1+0.8*i/5, 0.1+0.8*j/5) for i in 1:5 for j in 1:5]
        adj_fixed = RadialBasisFunctions.find_neighbors(pts_fixed, 10)
        function loss_wrt_direction(n_flat)
            n = [n_flat[1], n_flat[2]]
            W = RadialBasisFunctions._build_weights(
                    RadialBasisFunctions.Directional{2}(n),
                    pts_fixed, pts_fixed, adj_fixed, PHS(3; poly_deg=2))
            return sum(W.nzval .^ 2)
        end
        n0 = [1.0/√2, 1.0/√2]
        rule = Mooncake.build_rrule(loss_wrt_direction, n0)
        _, (_, dn) = Mooncake.value_and_gradient!!(rule, loss_wrt_direction, n0)
        validate_gradient(dn, loss_wrt_direction, n0; rtol=1e-3)
    end
end
```

### Test 3: Integration — elasticity assembly gradient (all five operators)

This is the closest proxy to Phase 1 of Macchiato without requiring Macchiato to be
present. It chains all five `_build_weights` calls used in `make_system_differentiable`
and differentiates the sum of all resulting weight entries w.r.t. point positions.

```julia
@testset "Mooncake Extension - Elasticity assembly gradient (all 5 operators)" begin
    N = 25
    points = [SVector{2}(0.1+0.8*i/5, 0.1+0.8*j/5) for i in 1:5 for j in 1:5]
    adjl = RadialBasisFunctions.find_neighbors(points, 14)
    basis = PHS(3; poly_deg=3)   # poly_deg=3 required for elasticity — see Macchiato plan
    pts_flat = vcat([collect(p) for p in points]...)

    # Mimics make_system_differentiable for LinearElasticity
    function loss_elasticity_assembly(pts)
        p = [SVector{2}(pts[2i-1], pts[2i]) for i in 1:N]
        W_d2x  = RadialBasisFunctions._build_weights(Partial(2, 1),      p, p, adjl, basis)
        W_d2y  = RadialBasisFunctions._build_weights(Partial(2, 2),      p, p, adjl, basis)
        W_d2xy = RadialBasisFunctions._build_weights(MixedPartial(1, 2), p, p, adjl, basis)
        W_dx   = RadialBasisFunctions._build_weights(Partial(1, 1),      p, p, adjl, basis)
        W_dy   = RadialBasisFunctions._build_weights(Partial(1, 2),      p, p, adjl, basis)
        return sum(W_d2x.nzval.^2) + sum(W_d2y.nzval.^2) + sum(W_d2xy.nzval.^2) +
               sum(W_dx.nzval.^2)  + sum(W_dy.nzval.^2)
    end

    rule = Mooncake.build_rrule(loss_elasticity_assembly, pts_flat)
    _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_elasticity_assembly, pts_flat)
    validate_gradient(dpts, loss_elasticity_assembly, pts_flat; rtol=1e-3)
end
```

---

## Verification checklist

Before marking this work complete and proceeding to Macchiato.jl Phase 1, all of the
following must hold:

- [ ] `julia test/runtests.jl` passes with no regressions (all existing tests green)
- [ ] `@testset "MixedPartial _build_weights"` passes for PHS1/3/5/7, IMQ, Gaussian, 2D and 3D
- [ ] `@testset "Directional _build_weights"` passes for constant direction, spatially varying direction, and gradient w.r.t. direction vector
- [ ] `@testset "Elasticity assembly gradient"` passes (all five operators in one loss)
- [ ] The `autodiff_di.jl` counterparts of the above tests also pass via `DI.AutoMooncake`

Once all boxes are checked, the `MixedPartial` and `Directional` entries in the
"Responsibility split" table in `Macchiato.jl/plan_shape_optimization_AD.md` can be
updated from "Blocked" to "Done".
