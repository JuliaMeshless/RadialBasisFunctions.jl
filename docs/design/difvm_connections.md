# DiFVM Connections: Graph-Based FVM and RBF-FD Operators

Reference: Du et al. (2026), "DiFVM: A Vectorized Graph-Based Finite Volume Solver for Differentiable CFD on Unstructured Meshes," arXiv:2603.15920v1.

## The Core Isomorphism

DiFVM's central insight is that finite-volume discretization on unstructured meshes is structurally isomorphic to graph neural network message passing. Every FVM operator decomposes into three graph primitives:

1. **Message construction** -- compute fluxes at each face from the two endpoint cell states
2. **Aggregation** -- scatter-add those face fluxes back to each cell (`segment_sum`)
3. **Node update** -- advance each cell's state using the aggregated residual

RBF-FD operators follow the same pattern with a different discretization:

1. **Gather** -- for each evaluation point, pull field values from k-nearest neighbors (the stencil) via the sparsity pattern of `W`
2. **Weighted aggregation** -- multiply by precomputed stencil weights (the sparse mat-vec `W * x`)
3. **Output** -- the result at each eval point is the sum of weighted neighbor contributions

Both systems encode irregular connectivity as **static index arrays** built once at initialization. DiFVM stores `owner`/`neighbor` edge-index arrays; RadialBasisFunctions.jl stores `adjl::Vector{Vector{Int}}` from k-NN. Both become the sparsity pattern for all subsequent computation.

## Key Differences

| | DiFVM (FVM) | RadialBasisFunctions.jl (RBF-FD) |
|---|---|---|
| **Graph source** | Mesh topology (faces connect cells) | k-NN search (geometric proximity) |
| **Edge weights** | Geometric (face areas, distances) + physics (viscosity, upwind) | Precomputed via local collocation solves (solve k x k linear systems) |
| **Conservation** | Built-in -- flux through a face is equal and opposite for its two cells | Not inherently conservative (but accuracy is high-order) |
| **Stencil uniformity** | Variable (each cell has different face count) | Uniform k for all stencils (simplifies batching) |
| **Operator composition** | Separate operators for convection, diffusion, pressure gradient | Algebraic composition via `@operator` macro |

## What RadialBasisFunctions.jl Already Has That DiFVM Needs

### Operator Algebra

DiFVM hard-codes each PDE operator (convective flux, diffusive flux, pressure gradient). RadialBasisFunctions.jl's `@operator` macro and `operator_algebra.jl` let users compose arbitrary differential operators symbolically -- `nabla^2`, `nabla cdot (kappa nabla)`, etc. -- and the weights are computed automatically. This is more general and would let a CFD solver express the same physics without hand-deriving each flux operator.

### Meshfree Geometry Handling

DiFVM needs non-orthogonality corrections (their over-relaxed correction in Eq. 15) because FVM's two-point stencil can't resolve gradients on skewed meshes. RBF-FD operators sidestep this entirely -- the k-neighbor stencil naturally handles arbitrary point distributions without special geometric corrections.

### AD Backward Pass

RadialBasisFunctions.jl already has hand-written Enzyme/Mooncake rules for the weight-building and operator-application paths (`src/solve/backward.jl`, the Enzyme/Mooncake extensions). DiFVM relies on JAX's built-in VJP for everything, which works because JAX primitives are all differentiable. The custom-rule approach is more work but avoids differentiating through the stencil solve (the `factorize` problem).

## What DiFVM Has That's Relevant for an RBF-FD CFD Solver

### 1. PISO as Constrained Message Passing

The pressure-velocity coupling is decomposed into three graph "layers" -- momentum MP, graph Laplacian projection, correction MP -- all using the same `owner`/`neighbor` scatter primitives. If we build a CFD solver on RBF operators, we'd need an analogous pressure projection step. The insight that the pressure Poisson equation becomes a **weighted graph Laplacian** (DiFVM Eq. 21) is directly usable -- our `laplacian()` operator already computes this, just with RBF weights instead of FVM geometric weights.

### 2. Implicit Differentiation Through Iterative Solves

This is the most architecturally important technique for building a differentiable CFD solver on top of RBF-FD operators. The full treatment follows in the next section.

### 3. Gradient Checkpointing for Time Stepping

For transient CFD, storing all intermediate states for backprop is O(n_steps x N_cells). DiFVM uses `jax.lax.scan` with binomial checkpointing to get O(sqrt(n_steps) x N_cells) memory. We'd need the Julia equivalent (likely via Checkpointing.jl or custom Enzyme rules) when differentiating through time loops.

---

## Deep Dive: Implicit Differentiation Through Iterative Solves

### The Problem

In a CFD solver, there is at least one iterative linear solve per time step -- the pressure Poisson equation. Naively, if you want end-to-end gradients through the simulation, AD would try to differentiate through every CG/BiCGStab iteration. That means:

- **Memory**: storing all intermediate Krylov vectors for the backward pass -- O(n_iterations x N_cells) per time step
- **Compute**: backpropagating through the iterative solver's entire convergence history
- **Fragility**: the number of iterations varies between time steps, and AD through iterative loops with dynamic termination is painful in most frameworks

### DiFVM's Solution (Their Eq. 25)

Given a converged pressure solve F(p*, u*) = 0 where p* is the pressure field satisfying the Poisson equation for a given intermediate velocity u*, the implicit function theorem gives:

```
dp*/du* = -(dF/dp*)^{-1} * (dF/du*)
```

For the adjoint (reverse mode), you need `v^T * dp*/du*`, which becomes solving the **adjoint linear system**:

```
(dF/dp*)^T * lambda = dL/dp*
```

then computing `dL/du* += -lambda^T * (dF/du*)`.

The key insight: the adjoint system has the **same matrix** as the forward pressure solve (since the pressure Laplacian is symmetric, the transpose is free). You solve it with the same CG solver, same preconditioner, same cost as one forward solve -- regardless of how many iterations the forward solve took.

### We Already Do Exactly This for Stencil Weights

The `backward_linear_solve!` function in `src/solve/backward.jl` implements the identical mathematical pattern for the collocation solve `A * lambda = b`:

```
Forward:  A * lambda = b        (collocation solve for stencil weights)
Adjoint:  eta = A^{-T} * Delta_lambda   (solve adjoint system with same matrix)
Then:     Delta_A = -eta * lambda^T      (outer product)
          Delta_b = eta
```

The difference is scale. The stencil matrices are small (k x k, where k ~ 20-80), so we store the dense matrix `A_mat` in `StencilForwardCache` and solve with a direct `\`. A pressure Poisson system is N_cells x N_cells (potentially millions), so an iterative solver would be needed for both forward and adjoint.

### What This Looks Like Concretely for an RBF-FD CFD Solver

Suppose we build a pressure Poisson step using the existing operators:

```julia
# Forward pass
L = laplacian(points)           # RBF-FD Laplacian operator (sparse matrix)
rhs = divergence(u_star)        # RHS from intermediate velocity
p = cg(L.weights, rhs)          # Iterative solve -- CG since L is SPD-like
u_new = u_star - dt * gradient(points)(p)  # Velocity correction
```

For AD, we'd need a custom rule around the `cg` call that:

1. **Forward**: runs CG normally, caches `L.weights` and the converged `p`
2. **Backward**: given `Delta_p` (the cotangent of the pressure), solves `L.weights^T * lambda = Delta_p` using the same CG solver, then propagates `Delta_rhs = lambda` backward

In Julia with the existing patterns, the Enzyme rule structure would mirror what we already have:

```julia
function EnzymeRules.augmented_primal(
    config, func::Const{typeof(cg)}, ::Type{RT},
    A::Const, b::Duplicated
) where {RT}
    p = cg(A.val, b.val)                          # normal forward solve
    cache = (A.val, copy(p))                       # cache for backward
    return EnzymeRules.AugmentedReturn(p, shadow, cache)
end

function EnzymeRules.reverse(
    config, func::Const{typeof(cg)}, dret, cache, A, b
)
    A_val, p = cache
    Delta_p = dret                                 # cotangent of output
    lambda = cg(A_val', Delta_p)                   # adjoint solve -- same cost as forward
    b.dval .+= lambda                              # propagate Delta_b = lambda
    return (nothing, nothing)
end
```

### Why This Matters Beyond the Pressure Solve

Any time the CFD solver has an **implicit step** -- pressure projection, implicit viscosity, implicit time integration -- the same choice arises:

1. **Differentiate through the iterations** -- memory-hungry, fragile, slow
2. **Implicit function theorem** -- one extra solve of the same system, exact gradients

The codebase already chose option 2 for the stencil solves. The pattern generalizes directly. The only new ingredient is that the "matrix" is now the sparse RBF-FD operator (`L.weights`) rather than a small dense collocation matrix, so the adjoint solve is iterative rather than direct.

### Connection to DiFVM's Architecture

DiFVM gets this "for free" from JAX because `jax.lax.scan` + `jax.custom_vjp` handle the plumbing. In Julia, we'd wire it up explicitly via EnzymeRules or Mooncake's `rrule!!`, but the math is identical. The existing `build_weights_pullback_loop!` abstraction -- which already parameterizes the cotangent extraction pattern across both Enzyme and Mooncake -- is exactly the kind of shared infrastructure that would be extended for this.

The implicit differentiation approach also composes cleanly with **gradient checkpointing** for time stepping (DiFVM's other trick). At each checkpointed time step, you'd:

1. Rerun the forward (including the pressure solve)
2. Use the IFT rule for the pressure solve's backward -- no need to store CG iterates even during recomputation

This means the memory cost of differentiating through an entire transient simulation is O(sqrt(n_steps) x N_cells) for time checkpointing + O(N_cells) for each pressure adjoint solve -- exactly what DiFVM achieves.

---

## The Big Picture: Path to a Differentiable RBF-FD CFD Solver

RBF operators are already "graph operators" -- sparse matrices whose sparsity pattern encodes a static connectivity graph. The path to a differentiable RBF-FD CFD solver:

1. **Spatial discretization**: Already done -- `laplacian()`, `gradient()`, `partial()` operators discretize the PDE terms
2. **Pressure projection**: Build a pressure Poisson solve using `laplacian()`, with implicit-function-theorem AD as described above
3. **Time integration**: `u^{n+1} = u^n + dt * (convection + diffusion + pressure)` where each term is an RBF operator application
4. **End-to-end AD**: Already partially there via the Enzyme/Mooncake extensions; the missing pieces are differentiating through the time loop and pressure solve

The advantage over DiFVM's approach: no mesh is needed. No mesh topology, no face areas, no non-orthogonality corrections. Just scattered points and k-NN neighborhoods. The tradeoff is that strict local conservation (FVM's main selling point) is lost, but geometric flexibility and high-order accuracy on arbitrary point clouds are gained.
