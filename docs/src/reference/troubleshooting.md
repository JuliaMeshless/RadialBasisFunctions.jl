# Troubleshooting

Common errors, what causes them, and how to fix them.

## `MethodError: no method matching...` when creating an operator

Your data is a `Matrix` instead of a `Vector` of vectors. Every point must be its own
vector with a compile-time-inferrable dimension.

```julia
# WRONG — a Matrix
points = rand(100, 2)

# CORRECT
using StaticArrays
points = rand(SVector{2,Float64}, 100)

# Converting from a Matrix
matrix_data = rand(100, 2)
points = map(SVector{2}, eachrow(matrix_data))
```

## `ArgumentError: n must be 1, 3, 5, or 7`

Polyharmonic spline order must be odd and ≤ 7.

```julia
PHS(2)  # ✗ even
PHS(9)  # ✗ too high

PHS(1)  # linear   (least smooth)
PHS(3)  # cubic    (default, good balance)
PHS(5)  # quintic  (smoother)
PHS(7)  # septic   (smoothest)
```

## `ArgumentError: Shape parameter should be > 0`

The shape parameter ε of `IMQ` and `Gaussian` must be positive.

```julia
IMQ(-1.0)      # ✗
Gaussian(0.0)  # ✗

IMQ(1.0)       # typical range 0.1 – 10.0
Gaussian(0.5)  # smaller ε ⇒ wider basis function
```

## Poor accuracy or oscillations

Work through these in order:

1. **Stencil too small** — increase `k`:
   ```julia
   lap = laplacian(points; k = 50)
   ```
2. **Polynomial degree too low** — increase `poly_deg`:
   ```julia
   basis = PHS(3; poly_deg = 4)
   ```
3. **Wrong basis for the problem** — for very smooth functions, try a higher-order PHS:
   ```julia
   basis = PHS(5; poly_deg = 4)
   ```
4. **Shape parameter ill-suited** (`IMQ` / `Gaussian`) — tune ε; smaller is smoother:
   ```julia
   basis = IMQ(0.1)
   ```

## `SingularException` or an ill-conditioned system

1. **Duplicate or near-duplicate points.** Two points at (nearly) the same location make
   the collocation matrix singular. Detect them with a `KDTree` and remove them.
2. **Stencil too large for the local point density** — reduce `k`:
   ```julia
   lap = laplacian(points; k = 20)
   ```
3. **Polynomial degree too high for the stencil size** — reduce `poly_deg`:
   ```julia
   basis = PHS(3; poly_deg = 1)
   ```

   As a rule of thumb, keep `poly_deg ≤ (k - 1) / dim` so the polynomial block stays
   determined.
