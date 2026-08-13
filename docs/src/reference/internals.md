# Internals: RBF Weight Building System

This document explains the architecture of the solve system, which builds sparse weight matrices for RBF operators. It is intended for developers who want to understand or extend the package.

## Call Graph

```@raw html
<div style="overflow-x:auto;">
<svg viewBox="0 0 720 590" width="100%" role="img" aria-labelledby="cg-title cg-desc"
     style="max-width:720px;height:auto;display:block;margin:1.5rem auto;font-family:var(--vp-font-family-base);">
  <title id="cg-title">RBF weight-building call graph</title>
  <desc id="cg-desc">From constructing an operator down to returning the sparse weight matrix, colored by architecture layer: api.jl, execution.jl, and assembly.jl.</desc>
  <defs>
    <marker id="cg-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--vp-c-text-3)"/>
    </marker>
  </defs>

  <!-- edges -->
  <g fill="none" stroke="var(--vp-c-text-3)" stroke-width="1.6" marker-end="url(#cg-arrow)">
    <line x1="360" y1="60"  x2="360" y2="86"/>
    <line x1="360" y1="142" x2="360" y2="168"/>
    <line x1="360" y1="224" x2="360" y2="250"/>
    <path d="M360,296 V318 H170 V336"/>
    <path d="M360,296 V318 H550 V336"/>
    <path d="M170,392 L332,432"/>
    <path d="M550,392 L388,432"/>
    <line x1="360" y1="488" x2="360" y2="528"/>
  </g>

  <!-- nodes -->
  <g stroke-width="1.6" font-size="14">
    <!-- User -->
    <rect x="210" y="14" width="300" height="46" rx="23" fill="var(--vp-c-default-soft)" stroke="var(--vp-c-text-3)"/>
    <text x="360" y="42" text-anchor="middle" fill="var(--vp-c-text-1)">User constructs operator</text>

    <!-- api.jl -->
    <rect x="210" y="86" width="300" height="56" rx="8" fill="var(--vp-c-indigo-soft)" stroke="var(--vp-c-indigo-1)"/>
    <text x="360" y="110" text-anchor="middle" fill="var(--vp-c-text-1)" font-family="var(--vp-font-family-mono)" font-size="13.5">_build_weights(ℒ, op)</text>
    <text x="360" y="128" text-anchor="middle" fill="var(--vp-c-text-2)" font-size="10.5">apply ℒ to basis + monomial</text>

    <!-- execution.jl -->
    <rect x="210" y="168" width="300" height="56" rx="8" fill="var(--vp-c-yellow-soft)" stroke="var(--vp-c-yellow-1)"/>
    <text x="360" y="192" text-anchor="middle" fill="var(--vp-c-text-1)" font-family="var(--vp-font-family-mono)" font-size="13.5">build_weights_kernel</text>
    <text x="360" y="210" text-anchor="middle" fill="var(--vp-c-text-2)" font-size="10.5">allocate sparse arrays · parallel kernel per point</text>

    <!-- assembly.jl -->
    <rect x="210" y="250" width="300" height="46" rx="8" fill="var(--vp-c-green-soft)" stroke="var(--vp-c-green-1)"/>
    <text x="360" y="278" text-anchor="middle" fill="var(--vp-c-text-1)" font-family="var(--vp-font-family-mono)" font-size="13.5">_build_stencil!</text>

    <rect x="45" y="336" width="250" height="56" rx="8" fill="var(--vp-c-green-soft)" stroke="var(--vp-c-green-1)"/>
    <text x="170" y="360" text-anchor="middle" fill="var(--vp-c-text-1)" font-family="var(--vp-font-family-mono)" font-size="12">_build_collocation_matrix!</text>
    <text x="170" y="378" text-anchor="middle" fill="var(--vp-c-text-2)" font-size="10.5">assemble A</text>

    <rect x="425" y="336" width="250" height="56" rx="8" fill="var(--vp-c-green-soft)" stroke="var(--vp-c-green-1)"/>
    <text x="550" y="360" text-anchor="middle" fill="var(--vp-c-text-1)" font-family="var(--vp-font-family-mono)" font-size="12">_build_rhs!</text>
    <text x="550" y="378" text-anchor="middle" fill="var(--vp-c-text-2)" font-size="10.5">assemble b</text>

    <rect x="235" y="432" width="250" height="56" rx="8" fill="var(--vp-c-green-soft)" stroke="var(--vp-c-green-1)"/>
    <text x="360" y="456" text-anchor="middle" fill="var(--vp-c-text-1)" font-family="var(--vp-font-family-mono)" font-size="12.5">w = A \ b</text>
    <text x="360" y="474" text-anchor="middle" fill="var(--vp-c-text-2)" font-size="10.5">local solve → weights</text>

    <!-- return -->
    <rect x="210" y="528" width="300" height="46" rx="23" fill="var(--vp-c-default-soft)" stroke="var(--vp-c-text-3)"/>
    <text x="360" y="556" text-anchor="middle" fill="var(--vp-c-text-1)">pack sparse I, J, V  →  return W</text>
  </g>
</svg>
</div>
```

Nodes are colored by architecture layer — indigo `api.jl` (routing and entry points), amber `execution.jl` (allocation and parallel kernel execution), and green `assembly.jl` (the per-stencil mathematics: build **A**, build **b**, and the local solve). The execution layer runs one `_build_stencil!` per evaluation point; once every batch completes, the collected `(I, J, V)` triples are packed into the sparse weight matrix `W`. See the layer breakdown below.

## Architecture Overview

The solve system is organized into **four layers**, each with a clear responsibility:

```text
src/solve/
├── types.jl      # Layer 0: Shared data structures
├── assembly.jl   # Layer 1: Pure mathematical operations
├── execution.jl  # Layer 2: Parallel execution & allocation
└── api.jl        # Layer 3: Entry points & routing
```

| Layer | File | Purpose |
|-------|------|---------|
| 3 | `api.jl` | Entry points from operators |
| 2 | `execution.jl` | Memory allocation, parallel kernel execution, sparse matrix construction |
| 1 | `assembly.jl` | Pure mathematics: collocation matrix, RHS, stencil assembly |
| 0 | `types.jl` | Shared types and arity helpers |

---

## System Flow

The system builds sparse weight matrices by computing stencil weights in parallel for each evaluation point:

```text
┌───────────┐    ┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌────────────────────┐    ┌──────────────┐
│ User Code │───>│    api.jl     │───>│  execution.jl   │───>│   assembly.jl   │───>│   execution.jl     │───>│Return to user│
│           │    │ Route request │    │Allocate & launch│    │Build A,b, solve │    │Construct sparse    │    │              │
│           │    │               │    │     kernel      │    │      A\b        │    │      matrix        │    │              │
└───────────┘    └───────────────┘    └─────────────────┘    └─────────────────┘    └────────────────────┘    └──────────────┘
```

**Key steps:**
1. **Route** — Operator calls `_build_weights`, which extracts data and applies the operator to the basis
2. **Allocate** — Pre-allocate sparse arrays with exact non-zero counts (k entries per evaluation point)
3. **Build A** — Construct the collocation matrix for each stencil's k-nearest neighbors
4. **Build b** — Construct the RHS by applying the differential operator to the basis at the evaluation point
5. **Solve** — Compute weights via `A \ b`
6. **Assemble** — Pack weights into global sparse matrix

---

## Layer Details

### Layer 0: `types.jl` — Data Structures

Defines shared types used throughout the solve system.

**Key types:**
- `_num_ops`, `_prepare_buffer` — Operator arity helpers for pre-allocating RHS buffers (dispatch on `Tuple` vs non-Tuple)

**When to modify:** Adding new shared data structures or arity helpers.

---

### Layer 1: `assembly.jl` — Pure Mathematics

Contains all mathematical operations for building stencils. **No I/O, no parallelism** — fully testable in isolation.

**Key functions:**
- `_build_stencil!(A, b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)` — Assembles and solves local system
- `_build_collocation_matrix!(A, data, basis, mon, k)` — Fills the collocation matrix
- `_build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)` — Builds RHS vector

**When to modify:** Changing stencil mathematics or adding new RBF formulations.

---

### Layer 2: `execution.jl` — Parallel Execution

Handles memory management and parallel kernel execution via KernelAbstractions.jl.

**Key functions:**
- `build_weights_kernel(...)` — Main orchestrator: allocates, launches kernel, returns sparse matrix
- `allocate_sparse_arrays(TD, k, N_eval, num_ops, adjl)` — Exact non-zero allocation
- `launch_kernel!(...)` — Dispatches parallel kernel over batches
- `@kernel weight_kernel(...)` — Per-batch kernel that builds weights for each evaluation point

**When to modify:** Improving parallelism, changing batch strategy, or optimizing memory allocation.

---

### Layer 3: `api.jl` — Entry Points

Public-facing entry points that operators call to build weights.

**Key functions:**
- `_build_weights(ℒ, op)` — Entry from operator, extracts configuration
- `_build_weights(ℒ, data, eval_points, adjl, basis)` — Applies operator to basis
- `_build_weights(...; batch_size, device)` — Routes to kernel execution

**When to modify:** Adding new operator entry points or changing routing logic.

---

## Key Concepts

### Stencils

A **stencil** approximates a differential operator at a point using its k nearest neighbors:

$$\mathcal{L}u(\mathbf{x}_0) \approx \sum_{i=1}^{k} w_i \cdot u(\mathbf{x}_i)$$

The weights $w_i$ are computed by solving the local collocation system $\mathbf{A}\mathbf{w} = \mathbf{b}$.

### Collocation Matrix Structure

The collocation matrix has a block structure:

$$\begin{bmatrix} \mathbf{\Phi} & \mathbf{P} \\ \mathbf{P}^\top & \mathbf{0} \end{bmatrix}$$

where $\mathbf{\Phi}$ is the RBF kernel matrix and $\mathbf{P}$ is the polynomial augmentation matrix. The system is solved to find weights that exactly reproduce polynomials up to the specified degree.

---

## Performance Notes

**Memory allocation:**
- Exact non-zero counting before allocation (k entries per evaluation point)
- COO format construction with pre-computed row offsets

**Parallelization:**
- Batch processing to control memory usage
- Work arrays reused within each batch
- KernelAbstractions.jl enables CPU/GPU execution

---

## Advanced: Hermite Interpolation

For problems with boundary conditions, the package supports Hermite interpolation. This is triggered by providing boundary information (`is_boundary`, `boundary_conditions`, `normals`) to the operator constructor.

Hermite interpolation modifies the collocation matrix to incorporate boundary operators (Dirichlet, Neumann, Robin) at boundary nodes. This is an advanced feature primarily used for solving PDEs with explicit boundary conditions.

---

## Navigation Guide

| Want to... | Look in... |
|------------|------------|
| Modify stencil mathematics | `assembly.jl` |
| Improve parallelism or batching | `execution.jl` |
| Add a new operator entry point | `api.jl` |
| Understand sparse allocation | `execution.jl` → `allocate_sparse_arrays` |
| Debug a specific stencil | `assembly.jl` → `_build_stencil!` |
