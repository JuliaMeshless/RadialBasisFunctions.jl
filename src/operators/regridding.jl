"""
    Regrid

Operator for interpolating from one set of points to another.
"""
struct Regrid
    ℒ
    Regrid() = new(identity)
end
(op::Regrid)(x) = op.ℒ(x)

# Primary interface using unified keyword constructor
"""
    regrid(data, eval_points; basis=PHS(3; poly_deg=2), k, adjl)

Build a `RadialBasisOperator` for interpolating from `data` points to `eval_points`.

# Arguments
- `data`: Source data points
- `eval_points`: Target evaluation points

# Keyword Arguments
- `basis`: RBF basis (default: `PHS(3; poly_deg=2)`)
- `k`: Stencil size (default: `autoselect_k(data, basis)`)
- `adjl`: Adjacency list (default: computed via `find_neighbors`)

# Examples
```julia
# Interpolate from coarse grid to fine grid
coarse = [SVector{2}(rand(2)) for _ in 1:100]
fine = [SVector{2}(rand(2)) for _ in 1:1000]
op = regrid(coarse, fine)

# Apply to field values
u_coarse = sin.(getindex.(coarse, 1))
u_fine = op(u_coarse)
```
"""
regrid(data::AbstractVector, eval_points::AbstractVector; kw...) =
    RadialBasisOperator(Regrid(), data; eval_points=eval_points, kw...)

# Backward compatible positional signature
regrid(data::AbstractVector, eval_points::AbstractVector, basis::AbstractRadialBasis; kw...) =
    RadialBasisOperator(Regrid(), data; eval_points=eval_points, basis=basis, kw...)

# pretty printing
print_op(::Regrid) = "regrid"
