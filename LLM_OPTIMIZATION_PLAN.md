# Plan: LLM Optimization for RadialBasisFunctions.jl

**Goal:** Make this package more discoverable and usable by AI assistants (Claude, etc.) so they can better help users with RBF/meshless methods.

---

## Phase 1: CLAUDE.md Enhancements (High Impact)

**File:** `CLAUDE.md`

### 1.1 Add Common Usage Patterns Section
Copy-paste snippets for:
- Basic interpolation with correct data format
- Differential operators (laplacian, gradient, partial)
- Custom basis selection (PHS, IMQ, Gaussian)
- Stencil size control

### 1.2 Add Troubleshooting FAQ
- Data format errors (Matrix vs Vector{AbstractVector})
- Invalid PHS order (must be odd)
- Shape parameter errors (must be > 0)
- Poor accuracy / oscillations fixes
- Singular system debugging

### 1.3 Add "When to Use What" Decision Guide
Tables for:
- Choosing basis function by scenario
- Polynomial degree selection
- Stencil size (k) guidelines
- Operator selection by task

### 1.4 Add Extension Patterns
How to:
- Add a new RBF type
- Add a new operator

---

## Phase 2: Create llms.txt (High Impact)

**File:** `llms.txt` (new, at repository root)

Emerging convention (like robots.txt) for LLM-friendly sites. Include:
- Package purpose (1-2 sentences)
- Alternative names/keywords (RBF-FD, meshfree, kernel methods, etc.)
- Quick start code
- Common pitfalls
- Key types and API overview
- Links to docs

---

## Phase 3: README Terminology (Medium Impact)

**File:** `README.md`

Add keyword synonyms for discoverability:
- "meshfree methods" / "meshless methods"
- "RBF-FD" (Radial Basis Function Finite Differences)
- "scattered data interpolation"
- "kernel methods"
- "polyharmonic splines"

---

## Phase 4: Docstring Cross-References (Medium Impact)

Add "See also" sections linking related functions:

| File | Add Cross-References To |
|------|------------------------|
| `src/operators/laplacian.jl` | `partial`, `gradient` |
| `src/operators/partial.jl` | `laplacian`, `gradient` |
| `src/operators/directional.jl` | `gradient`, `partial` |
| `src/interpolation.jl` | `regrid` |
| `src/operators/regridding.jl` | `Interpolator` |
| `src/basis/polyharmonic_spline.jl` | `IMQ`, `Gaussian` |
| `src/basis/inverse_multiquadric.jl` | `PHS`, `Gaussian` |
| `src/basis/gaussian.jl` | `PHS`, `IMQ` |

---

## Phase 5: Improved Error Messages (Medium Impact)

Make errors actionable with fix suggestions:

| File | Current | Improved |
|------|---------|----------|
| `src/basis/polyharmonic_spline.jl:18` | "n must be 1, 3, 5, or 7" | Add: "Use PHS(3) for cubic or PHS(5) for quintic" |
| `src/utils.jl:47` | "degree must be at least 0" | Add: "Use poly_deg=2 (default) for better accuracy" |
| `src/operators/partial.jl:93` | "Only first and second order supported" | Add: "For higher-order, use custom(...)" |
| `src/operators/directional.jl:98` | Direction vector length error | Add examples of correct formats |

---

## Phase 6: Quick Reference Cheat Sheet (Medium Impact)

**Files:**
- `docs/src/quickref.md` (new)
- `docs/make.jl` (update pages list)

Include:
- Data format requirements with conversion examples
- Basis function table with formulas
- Operators at a glance
- Common options
- Hermite boundary conditions

---

## Implementation Order

| Priority | Phase | Effort |
|----------|-------|--------|
| 1st | Phase 1: CLAUDE.md | Low |
| 1st | Phase 2: llms.txt | Low |
| 2nd | Phase 3: README keywords | Low |
| 2nd | Phase 6: Quick reference | Low |
| 3rd | Phase 4: Cross-references | Medium |
| 3rd | Phase 5: Error messages | Medium |

---

## Verification

1. **CLAUDE.md:** Ask Claude about the package in a new session - does it find the troubleshooting info?
2. **llms.txt:** Check it's accessible at docs URL
3. **README:** Search engines/LLMs should surface package for "julia meshfree methods" queries
4. **Docstrings:** Run `julia --project=docs docs/make.jl` - verify cross-refs render
5. **Error messages:** Trigger each error intentionally, verify helpful output
6. **Quick ref:** Build docs and verify new page appears in navigation

---

## Files to Modify

- `CLAUDE.md` - Major additions (4 new sections)
- `llms.txt` - New file (repository root)
- `README.md` - Add keywords section
- `docs/src/quickref.md` - New file
- `docs/make.jl` - Add quickref to pages
- `src/operators/laplacian.jl` - Add "See also"
- `src/operators/partial.jl` - Add "See also" + improve error
- `src/operators/directional.jl` - Add "See also" + improve error
- `src/operators/regridding.jl` - Add "See also"
- `src/interpolation.jl` - Add "See also"
- `src/basis/polyharmonic_spline.jl` - Add "See also" + improve error
- `src/basis/inverse_multiquadric.jl` - Add "See also"
- `src/basis/gaussian.jl` - Add "See also"
- `src/utils.jl` - Improve error message
