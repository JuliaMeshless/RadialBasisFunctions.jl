# Design Document: Learnable Shape Parameters via RBF Splatting

## 1. Motivation

In the current RadialBasisFunctions.jl architecture, shape parameters (used in IMQ and Gaussian basis functions) are fixed scalars chosen by the user before any computation begins. Weights are then computed by solving a linear system. This raises a natural question: **what if we also optimized the shape parameters?**

## 2. Evolution of the Idea

### 2.1 Initial Observation: Solving for Shape Parameters

The standard RBF operator weight computation solves:

```
A(epsilon) * lambda = b(epsilon)
```

where `A` is the collocation matrix, `b` is the RHS encoding the operator applied to each basis function, and `lambda` are the stencil weights. Both `A` and `b` depend on the shape parameter `epsilon`, but `epsilon` is fixed beforehand.

If we tried to also solve for `epsilon`, the system becomes **underdetermined**:

| Scenario | Unknowns | Equations | Deficit |
|----------|----------|-----------|---------|
| Single global epsilon | N + M + 1 | N + M | 1 |
| Per-center epsilon_i | 2N + M | N + M | N |

Moreover, `epsilon` appears **nonlinearly** inside the basis functions (e.g., `1/sqrt(1 + (epsilon*r)^2)` for IMQ), so the problem can't be solved as a simple linear system extension.

### 2.2 Inspiration from Gaussian Splatting

3D Gaussian Splatting (Kerbl et al., 2023) represents scenes as collections of Gaussians where **all** parameters — positions, covariances (shape), opacities, colors — are optimized jointly via gradient descent through a differentiable renderer. The key insight: don't solve a constrained system, just define a loss and optimize everything end-to-end.

The analogy to RBFs is direct:

| Gaussian Splatting | RBF "Splatting" |
|---|---|
| Gaussian primitives | RBF basis functions |
| Position, covariance, color | Center, shape parameter, weight |
| Differentiable renderer | Differentiable operator evaluation |
| Photometric loss | Operator reproduction loss |
| Optimize all params jointly | Optimize lambda_i and epsilon_i jointly |

### 2.3 Why Not a Hybrid Approach?

A natural first thought is a hybrid: solve for weights `lambda` via the linear system (guaranteeing exact interpolation), then optimize `epsilon` via gradient descent on some outer objective. This is a bilevel optimization:

```
min_epsilon  L(lambda*(epsilon), epsilon)
subject to   lambda*(epsilon) = A(epsilon)^{-1} * b(epsilon)
```

But this constrains weights to always produce exact interpolation for any `epsilon`. The true joint optimum `min_{lambda, epsilon} L(lambda, epsilon)` may have weights that **don't** exactly interpolate — slightly relaxed interpolation with better-tuned shape parameters could yield a superior solution overall.

This matters especially for **differential operators**, where we don't care about interpolating `f` perfectly — we care about accurately computing derivatives. The weights that best reproduce derivatives are not necessarily the interpolation weights.

### 2.4 The Training Data Problem

For interpolation, ground truth is obvious: the user provides function values. But for operators (the more interesting case), where does "ground truth" come from?

The answer is already embedded in the current weight computation. The linear system `A * lambda = b` enforces that operator weights **exactly reproduce the operator applied to each basis function and polynomial**. The right-hand side values are computed analytically:

- `b[i] = L_rbf(x_center, x_i)` — the operator applied to each RBF, known in closed form
- `b[k+1:end] = L_mon(x_center)` — the operator applied to each monomial, known in closed form

These polynomial reproduction conditions are what guarantee convergence order. **Polynomials are the training data.** We know `Lp` analytically for any polynomial `p`, and we want the weights to reproduce these values. The current system enforces this as hard constraints; the splatting approach enforces it as a loss function.

## 3. Proposed Approach: RBF Splatting for Operators

### 3.1 Core Idea

Replace the per-stencil linear solve with a gradient-based optimization that jointly finds optimal weights `lambda_i` and per-center shape parameters `epsilon_i` by minimizing a polynomial reproduction loss.

### 3.2 Loss Function

For a single stencil centered at `x_0` with neighbors `{x_1, ..., x_k}`:

```
L(lambda, epsilon) = sum_{m=1}^{M} | sum_{i=1}^{k} lambda_i * p_m(x_i) - L[p_m](x_0) |^2
```

where:
- `p_m` are monomials up to degree `d` (from `MonomialBasis`)
- `L[p_m](x_0)` is the analytically computed operator applied to `p_m`, evaluated at the stencil center
- `lambda_i` are the stencil weights (optimized)
- `epsilon_i` are the per-center shape parameters (optimized), which affect basis function evaluation

The operator-on-polynomial values `L[p_m]` are already computed by the existing code path: `L_mon(eval_point)` in `_build_rhs!` (`src/solve/assembly.jl:116-138`).

Optional regularization terms:
- **Smoothness**: penalize large variations in `epsilon_i` across neighbors
- **Conditioning**: penalize extreme `epsilon` values that cause numerical issues
- **Basis reproduction**: optionally also reproduce the operator on the RBF basis functions themselves

### 3.3 Forward Pass (Differentiable)

For a given `{lambda, epsilon}`, the operator application is:

```
(Lf)_approx = sum_{i=1}^{k} lambda_i * f(x_i)
```

The loss evaluation requires:
1. Evaluate monomials at neighbor points: `p_m(x_i)` — no dependence on `epsilon`
2. Compute analytic operator-on-monomial: `L[p_m](x_0)` — no dependence on `epsilon`
3. Compute weighted sum: `sum_i lambda_i * p_m(x_i)` — depends on `lambda`
4. Compute residual and loss

Note: `epsilon` enters through an implicit coupling — the "good" weights depend on the basis function shapes, which depend on `epsilon`. To make `epsilon` participate in the gradient, we can add a basis function reproduction term:

```
L_basis(lambda, epsilon) = sum_{j=1}^{k} | sum_{i=1}^{k} lambda_i * phi(x_i, x_j; epsilon_i) - L[phi](x_0, x_j; epsilon_j) |^2
```

This term directly couples `epsilon` into the loss, since both `phi` and `L[phi]` depend on the shape parameters.

### 3.4 Full Loss

```
L_total = L_poly(lambda) + alpha * L_basis(lambda, epsilon) + beta * R(epsilon)
```

where:
- `L_poly`: polynomial reproduction (convergence guarantee)
- `L_basis`: basis function reproduction (shape parameter coupling)
- `R(epsilon)`: regularization on shape parameters
- `alpha, beta`: tunable hyperparameters

### 3.5 Optimization

Per-stencil optimization via gradient descent (or L-BFGS):
1. Initialize `lambda` from the current linear solve (warm start)
2. Initialize `epsilon_i` from the global user-provided value
3. Differentiate `L_total` w.r.t. `{lambda, epsilon}` using AD (Enzyme or Mooncake, already supported)
4. Iterate until convergence

## 4. Integration with Existing Codebase

### 4.1 Key Files to Modify or Extend

| File | Change |
|------|--------|
| `src/basis/inverse_multiquadric.jl` | Support per-center `epsilon` (vector instead of scalar) |
| `src/basis/gaussian.jl` | Support per-center `epsilon` (vector instead of scalar) |
| `src/solve/assembly.jl` | New `_build_stencil_splatting!` alongside existing `_build_stencil!` |
| `src/solve/api.jl` | New dispatch path for splatting-based weight computation |
| `src/solve/execution.jl` | Integration of optimization loop into kernel |
| `src/operators/operators.jl` | Store per-center `epsilon` in operator struct; new kwarg to opt-in |

### 4.2 Existing Code to Reuse

- **`MonomialBasis`** (`src/basis/monomial.jl`): Already generates monomials and evaluates them. Use directly for polynomial test functions.
- **Operator-on-monomial evaluation**: `L_mon = L(mon)` already computed in `_build_weights` (`src/solve/api.jl:31-32`). Reuse for ground truth.
- **Operator-on-basis evaluation**: `L_rbf = L(basis)` already computed (`src/solve/api.jl:31`). Reuse for basis reproduction loss.
- **`_build_collocation_matrix!`** (`src/solve/assembly.jl:72`): Reuse for warm-start initialization (solve linear system first, then optimize).
- **`_build_rhs!`** (`src/solve/assembly.jl:116`): Reuse for computing ground truth operator values.
- **`find_neighbors`** (`src/utils.jl:5`): Unchanged — stencil topology stays the same.
- **Enzyme/Mooncake extensions** (`ext/`): AD infrastructure already exists for differentiating through RBF evaluations.

### 4.3 User-Facing API

```julia
# Current API (unchanged)
lap = laplacian(points; basis=IMQ(1.0))

# New: opt-in to learnable shape parameters
lap = laplacian(points; basis=IMQ(1.0), optimize_shape=true)

# With tuning options
lap = laplacian(points;
    basis=IMQ(1.0),
    optimize_shape=true,
    optimize_maxiter=100,
    optimize_poly_deg=4,    # degree of polynomial test functions
    optimize_alpha=0.1,     # basis reproduction weight
)
```

### 4.4 Architecture Sketch

```
Operator Constructor (optimize_shape=true)
    |
    v
_build_weights(...; optimize_shape=true)  [api.jl]
    |
    +-- Warm start: solve linear system (existing path)
    |       -> initial lambda_0, global epsilon_0
    |
    +-- Compute ground truth:
    |       -> L[p_m](x_center) for each monomial  (reuse _build_rhs!)
    |       -> L[phi_j](x_center) for each basis fn (reuse _build_rhs!)
    |
    +-- Optimization loop (per stencil):
    |       for iter in 1:maxiter
    |           loss = L_poly(lambda) + alpha * L_basis(lambda, epsilon)
    |           d_lambda, d_epsilon = AD_gradient(loss)
    |           lambda -= lr * d_lambda
    |           epsilon -= lr * d_epsilon
    |       end
    |
    +-- Store optimized lambda as sparse weights
    +-- Store optimized epsilon in operator struct
    |
    v
RadialBasisOperator (with per-center epsilon)
```

## 5. Open Questions

1. **Scope of optimization**: Per-stencil independent optimization, or global optimization across all stencils simultaneously? Per-stencil is simpler and parallelizable, but neighboring stencils share points and could benefit from consistency.

2. **Optimizer choice**: Simple gradient descent, Adam, or L-BFGS? L-BFGS is natural for small-dimensional problems (k weights + k shape params per stencil).

3. **Convergence guarantees**: The polynomial reproduction loss should recover at least the same convergence order as the linear solve. Need to verify this theoretically or empirically.

4. **Cost vs. benefit**: The linear solve is O(k^3) per stencil. Optimization with AD will be more expensive. Is the accuracy improvement worth the cost? This is an empirical question — likely most valuable for problems where shape parameter sensitivity is high (IMQ, Gaussian with suboptimal global epsilon).

5. **AD backend**: Enzyme has known limitations with this codebase (can't differentiate through `factorize`). The optimization loop itself doesn't need factorization, but the warm start does. Mooncake may be more straightforward here.

## 6. Verification Plan

1. **Unit tests**: Verify polynomial reproduction to machine precision for known analytic cases
2. **Convergence study**: Compare convergence rates (h-refinement) between linear-solve weights and splatting weights on standard test problems (e.g., Laplacian of `sin(x)*cos(y)`)
3. **Shape parameter visualization**: Plot optimized `epsilon_i` distributions — they should adapt to local point density and solution smoothness
4. **Regression tests**: Ensure `optimize_shape=false` (default) produces identical results to current implementation
5. **Benchmark**: Measure wall-clock overhead of optimization vs. linear solve
