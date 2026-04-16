# RBF-WENO: WENO-Style Operator on Unstructured Points

## Motivation

Classical WENO (Weighted Essentially Non-Oscillatory) schemes achieve high-order accuracy in smooth regions while suppressing oscillations near discontinuities — but they require structured grids. By combining WENO's sub-stencil machinery with RBF interpolation, we can get the same shock-capturing behavior on **scattered, unstructured points**.

The nonlinear weights ARE the limiter — no separate limiting mechanism is needed. In smooth regions, the weights recover optimal (high-order) accuracy. Near discontinuities, they automatically shift away from contaminated sub-stencils.

## Overview

Solve 1D advection ∂u/∂t + c·∂u/∂x = 0 on scattered points, comparing:
1. **Central RBF** (full stencil, no limiting) — oscillates near discontinuities
2. **Upwind-biased RBF** (left sub-stencil only) — stable but diffusive
3. **RBF-WENO** (classical nonlinear weights) — sharp and stable

Start with a 1D demo (`dev/rbf_weno_demo.jl`) using analytically derived WENO weights. A learned (NN-based) version can follow as a second phase.

## Sub-Stencil Construction

For each point, the k=7 nearest neighbors are sorted by position. Three overlapping sub-stencils of size 5 are extracted:
- **Left-biased:** sorted neighbors [1:5]
- **Central:** sorted neighbors [2:6]
- **Right-biased:** sorted neighbors [3:7]

This mirrors WENO5's 3-sub-stencil structure. Each sub-stencil gets its own RBF ∂/∂x operator via `partial(x_pts, 1, 1; adjl=sub_adjl, k=5)`, fully reusing existing infrastructure.

**2D extension:** Replace "sort by x" with angular/directional partitioning of neighbors (e.g., 120° sectors for 3 sub-stencils, quadrant-based for 4).

## Classical WENO Weight Derivation

### Step 1 — Optimal (linear) weights d_s

For each point i, solve for d_s such that `Σ_s d_s * w_s[i,:] ≈ w_full[i,:]` where:
- `w_s` is sub-stencil s's sparse weight row (from its RBF ∂/∂x operator)
- `w_full` is the full-stencil weight row (from the k=7 RBF ∂/∂x operator)

This is a small least-squares problem (3 unknowns, up to 7 equations) per point. The d_s represent the ideal blending in smooth regions that recovers full-stencil accuracy.

Since sub-stencils are subsets of the sorted full-stencil neighbors, their column indices map naturally to the full-stencil's index space.

Negative d_s can occur — clamp to positive and renormalize (or use WENO-Z formulation).

Computed once at setup (geometry-dependent, not solution-dependent).

### Step 2 — Smoothness indicators β_s

For each sub-stencil s, use RBF operators to compute first and second derivatives, then:
```
β_s(u)[i] = h_i * (∂u/∂x)_s[i]² + h_i³ * (∂²u/∂x²)_s[i]²
```
where h_i is local spacing (distance to nearest neighbor). This is the Jiang-Shu smoothness indicator adapted to RBF stencils.

β_s is large when the sub-stencil crosses a discontinuity.

### Step 3 — Nonlinear weights

```
α_s = d_s / (ε + β_s)^p     (ε = 1e-6, p = 2)
ω_s = α_s / Σ_s α_s
```

- Smooth regions: β_s ≈ same for all s → ω_s ≈ d_s (optimal order recovered)
- Near discontinuity: β_s large for contaminated stencil → ω_s → 0 for that stencil

### Evaluation

The WENO derivative is solution-adaptive (nonlinear):
```
(∂u/∂x)_WENO[i] = Σ_s ω_s(u)[i] * (∂u/∂x)_s[i]
```

## Implementation Plan

### Required operators per sub-stencil (3 sub-stencils)
- ∂/∂x via `partial(x_pts, 1, 1; adjl=sub_adjl, k=5, basis=PHS(3; poly_deg=2))`
- ∂²/∂x² via `partial(x_pts, 2, 1; adjl=sub_adjl, k=5, basis=PHS(3; poly_deg=2))`
- 1 full-stencil ∂/∂x operator (k=7) for computing optimal weights

### Demo structure (`dev/rbf_weno_demo.jl`)

1. **Setup:** Generate N=200 scattered 1D points (jittered uniform, 30% jitter)
2. **Build sub-stencils:** Sort k=7 neighbors by position, extract 3 overlapping subsets of 5
3. **Build operators:** 3×(∂/∂x + ∂²/∂x²) sub-stencil ops + 1 full-stencil ∂/∂x
4. **Compute optimal weights d_s:** Least-squares per point
5. **Time-stepping:** Forward Euler with CFL ≈ 0.4, compare central/upwind/WENO
6. **Visualization:**
   - Animated GIF: exact vs central vs upwind vs WENO
   - Weight snapshot: ω₁, ω₂, ω₃ vs x showing adaptation near discontinuity

### Dependencies
- `RadialBasisFunctions`, `StaticArrays`, `Random`, `Statistics`, `CairoMakie`
- No Lux, no AD — pure analytical weights

## Tricky Parts

1. **Optimal weight mapping:** Sub-stencil weight rows live in subsets of the full stencil's column space. Need index mapping — but since sub-stencils are sorted subsets of full-stencil neighbors, this is straightforward.

2. **Negative optimal weights:** Least-squares may produce negative d_s. Clamp and renormalize, or consider WENO-Z which is more robust to this.

3. **Non-uniform smoothness indicators:** The h scaling in β_s should use local spacing (nearest-neighbor distance), not global average.

4. **Boundary points:** First/last points have one-sided stencils. Pin boundary values and exclude from error metrics.

## Future: Learned RBF-WENO

Once classical RBF-WENO works, the natural follow-up is replacing the analytical weight formula with a neural network:
- NN input: local solution features (u values, sub-stencil derivatives, smoothness indicators)
- NN output: sub-stencil weights via softmax (sum to 1)
- Train on exact advection solutions with sharp fronts
- Potential advantages: better adaptation, less sensitivity to ε/p tuning, generalizes to problems where classical smoothness indicators fail
