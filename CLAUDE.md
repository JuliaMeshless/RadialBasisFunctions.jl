# RadialBasisFunctions.jl

## Key files
- weight computation: `src/solve/api.jl` (routing) → `src/solve/assembly.jl` (math) → `src/solve/execution.jl` (KernelAbstractions CPU/GPU kernels)
- AD backward pass: `src/solve/backward.jl`
- operator types and hierarchy: `src/operators/operators.jl`; `@operator` macro in `src/operators/operator_macro.jl`
- basis hierarchy: `src/basis/basis.jl`
- exports and precompilation: `src/RadialBasisFunctions.jl`

## Gotchas
- Input points must be `Vector{<:AbstractVector}` with a compile-time-inferrable dimension (e.g. `SVector{2,Float64}`) — never a `Matrix`. This is the single most common user error.
- Operators compute weights **eagerly at construction** and cache them; use `invalidate_cache!` / `update_weights!` to force recomputation.
- `hermite` and `eval_points` are keyword-only since v0.6.
- Use DifferentiationInterface.jl for all AD examples in docs and tests — it unifies the Enzyme and Mooncake backends, which are loaded via extensions in `ext/`.

## Docs
User-facing material lives in `docs/src/` — add examples there, not here:
`docs/src/guides/quickref.md` (data formats, basis and operator options),
`docs/src/guides/custom_operators.md` (extending with new operators),
`docs/src/guides/troubleshooting.md`, `docs/src/reference/theory.md`.
