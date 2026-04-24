# Vector & Higher-Rank Operators

Convergence for operators that produce tensor output or act on vector fields:
`hessian`, `jacobian`, `divergence`, `curl`. The test vector field is
`u(x) = [sin(πx₁) + 0.5cos(2πx₂), exp(x₁x₂)]` on `[0,1]²` with all components having
nonzero partials.

"Which PHS order?" plots below show PHS3/5/7 at fixed `poly_deg = 3`. Holding the
polynomial augmentation constant isolates the PHS-order effect: curves should be
parallel on log-log, differing only in the error constant.

IMQ and Gaussian h-refinement is covered on its own page
([Shape-Parameter Bases](shape-parameter-bases.md)) — shape-parameter behavior is
dominated by the ε-vs-spacing interaction and is discussed there for a
representative subset of operators.

## Hessian

The Hessian assembles second partials `∂²u/∂xᵢ∂xⱼ` into an `N × D × D` tensor. Error is
reported as a Frobenius NRMSE. Because one of the four entries is a mixed partial, the
Hessian inherits the mixed-partial caveat — the non-converging combos are the same as
for the [mixed partial](scalar-operators.md#mixed-partial-xxj) operator.

!!! warning "Excluded combinations"
    Same PHS set as mixed partial: `PHS1/p=1`, `PHS3/p=1`, `PHS3/p=2`, `PHS5/p=2` are
    omitted. **Use PHS5/p=3 or higher for the full Hessian.** Shape-parameter bases:
    see [Shape-Parameter Bases](shape-parameter-bases.md).

### Which PHS order?

![Hessian h-refinement, PHS orders at polynomial degree 3](../../assets/convergence/plots/hessian_phs_matched.png)

At `poly_deg = 3`, all three PHS orders converge at `O(h²)` (the second-derivative rate).
PHS7 has the smallest error constant.

### How much polynomial degree?

![Hessian h-refinement, PHS polynomial degree sweeps](../../assets/convergence/plots/hessian_phs_polydeg_sweep.png)

## Jacobian

The Jacobian of a vector field is an `N × Dᵢₙ × Dₒᵤₜ` tensor of first partials — no
second derivatives, no mixed-partial issue. Convergence matches the gradient cleanly.

### Which PHS order?

![Jacobian h-refinement, PHS orders at polynomial degree 3](../../assets/convergence/plots/jacobian_phs_matched.png)

Parallel `O(h³)` curves across PHS3/5/7.

### How much polynomial degree?

![Jacobian h-refinement, PHS polynomial degree sweeps](../../assets/convergence/plots/jacobian_phs_polydeg_sweep.png)

## Divergence (∇·)

`∇·u = ∂u₁/∂x₁ + ∂u₂/∂x₂` — a sum of first partials. Convergence rates match the
gradient; all PHS orders at matched poly_deg are well-behaved.

### Which PHS order?

![Divergence h-refinement, PHS orders at polynomial degree 3](../../assets/convergence/plots/divergence_phs_matched.png)

Parallel `O(h³)` curves across PHS3/5/7.

### How much polynomial degree?

![Divergence h-refinement, PHS polynomial degree sweeps](../../assets/convergence/plots/divergence_phs_polydeg_sweep.png)

## Curl (∇×, 2D)

In 2D, `∇×u = ∂u₂/∂x₁ − ∂u₁/∂x₂` (the scalar z-component). Like divergence, this is a
first-derivative operator and follows the same convergence profile.

### Which PHS order?

![Curl h-refinement, PHS orders at polynomial degree 3](../../assets/convergence/plots/curl2d_phs_matched.png)

Parallel `O(h³)` curves across PHS3/5/7.

### How much polynomial degree?

![Curl h-refinement, PHS polynomial degree sweeps](../../assets/convergence/plots/curl2d_phs_polydeg_sweep.png)
