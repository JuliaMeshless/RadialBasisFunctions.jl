# Convergence & Parameter Selection

This guide shows how accuracy depends on point density, polynomial degree, stencil
size, and basis type — with plots generated from live convergence sweeps. Use these
results to choose parameters for your own problems.

For mathematical background, see [Radial Basis Functions Theory](@ref). For a quick lookup table, see [Quick Reference](@ref).

```@example convergence
using RadialBasisFunctions
using StaticArrays
using CairoMakie
using Random

# Jittered grid on [0,1]² — quasi-uniform scattered points
function scattered_points(n_side; seed=42)
    Random.seed!(seed)
    h = 1.0 / n_side
    [SVector(
        clamp(h * (i - 0.5) + 0.2h * randn(), 0.001, 0.999),
        clamp(h * (j - 0.5) + 0.2h * randn(), 0.001, 0.999),
    ) for i in 1:n_side for j in 1:n_side]
end

# Test function with multiple length scales (Franke 1979)
function franke(x)
    a = 0.75 * exp(-(9x[1] - 2)^2 / 4 - (9x[2] - 2)^2 / 4)
    b = 0.75 * exp(-(9x[1] + 1)^2 / 49 - (9x[2] + 1) / 10)
    c = 0.5 * exp(-(9x[1] - 7)^2 / 4 - (9x[2] - 3)^2 / 4)
    d = 0.2 * exp(-(9x[1] - 4)^2 - (9x[2] - 7)^2)
    return a + b + c - d
end

# Trig function with known Laplacian
g(x) = 1 + sin(4x[1]) + cos(3x[1]) + sin(2x[2])
∇²g(x) = -16sin(4x[1]) - 9cos(3x[1]) - 4sin(2x[2])

# Normalized RMSE
nrmse(computed, exact) = sqrt(sum((computed .- exact).^2) / sum(exact.^2))

# Resolution sweep
n_sides = [10, 15, 20, 30, 45, 70]
Ns = n_sides .^ 2
nothing # hide
```

## h-Refinement: Error vs. Number of Points

The most fundamental convergence test: fix all parameters and refine the point cloud.
With polynomial augmentation of degree ``m``, interpolation converges at
``O(h^{m+1})`` and second derivatives at ``O(h^{m-1})`` in 2D, where
``h \sim 1/\sqrt{N}`` is the typical point spacing.

### Interpolation

```@example convergence
# Fixed evaluation points for measuring interpolation error
Random.seed!(99)
eval_pts = [SVector{2}(rand(2) .* 0.98 .+ 0.01) for _ in 1:500]
f_exact = franke.(eval_pts)

configs = [
    (PHS(3; poly_deg=2), "PHS3, p=2"),
    (PHS(3; poly_deg=3), "PHS3, p=3"),
    (PHS(5; poly_deg=3), "PHS5, p=3"),
    (PHS(5; poly_deg=4), "PHS5, p=4"),
]

interp_errors = Dict(label => Float64[] for (_, label) in configs)

for n_side in n_sides
    pts = scattered_points(n_side)
    vals = franke.(pts)
    for (basis, label) in configs
        rg = regrid(pts, eval_pts; basis=basis)
        push!(interp_errors[label], nrmse(rg(vals), f_exact))
    end
end

fig = Figure(; size=(600, 400))
ax = Axis(fig[1, 1];
    xlabel="N (number of points)", ylabel="Normalized RMSE",
    xscale=log10, yscale=log10,
    title="Interpolation Convergence (Franke's function)")
for (i, (_, label)) in enumerate(configs)
    scatterlines!(ax, Ns, interp_errors[label]; label=label, linewidth=2, markersize=8)
end
axislegend(ax; position=:lb)
fig
```

Higher polynomial degree produces steeper slopes — the convergence *rate* improves,
not just the constant. Increasing the PHS order (3 → 5) helps slightly at the same
polynomial degree, but the polynomial degree is the dominant factor.

### Laplacian

```@example convergence
lap_errors = Dict(label => Float64[] for (_, label) in configs)

for n_side in n_sides
    pts = scattered_points(n_side)
    vals = g.(pts)
    exact = ∇²g.(pts)
    for (basis, label) in configs
        lap = laplacian(pts; basis=basis)
        push!(lap_errors[label], nrmse(lap(vals), exact))
    end
end

fig = Figure(; size=(600, 400))
ax = Axis(fig[1, 1];
    xlabel="N (number of points)", ylabel="Normalized RMSE",
    xscale=log10, yscale=log10,
    title="Laplacian Convergence")
for (i, (_, label)) in enumerate(configs)
    scatterlines!(ax, Ns, lap_errors[label]; label=label, linewidth=2, markersize=8)
end
axislegend(ax; position=:lb)
fig
```

Differential operators converge more slowly than interpolation — each order of
differentiation costs roughly one order of polynomial accuracy. This is why
`poly_deg=2` (quadratic) is the minimum recommended for second derivatives.

## p-Refinement: Error vs. Polynomial Degree

How much does increasing `poly_deg` help at a fixed resolution?

```@example convergence
pts_fixed = scattered_points(30)  # N = 900
vals_franke = franke.(pts_fixed)
vals_g = g.(pts_fixed)
exact_lap = ∇²g.(pts_fixed)

poly_degs = 0:5
interp_err_p = Float64[]
lap_err_p = Float64[]

for pd in poly_degs
    basis = PHS(3; poly_deg=pd)
    rg = regrid(pts_fixed, eval_pts; basis=basis)
    push!(interp_err_p, nrmse(rg(vals_franke), f_exact))
    lap = laplacian(pts_fixed; basis=basis)
    push!(lap_err_p, nrmse(lap(vals_g), exact_lap))
end

fig = Figure(; size=(600, 400))
ax = Axis(fig[1, 1];
    xlabel="Polynomial degree", ylabel="Normalized RMSE",
    yscale=log10, xticks=collect(poly_degs),
    title="p-Refinement at N = 900")
scatterlines!(ax, collect(poly_degs), interp_err_p;
    label="Interpolation", linewidth=2, markersize=10)
scatterlines!(ax, collect(poly_degs), lap_err_p;
    label="Laplacian", linewidth=2, markersize=10)
axislegend(ax; position=:rt)
fig
```

The jump from `poly_deg=0` to `poly_deg=2` is dramatic — often several orders of
magnitude. Beyond `poly_deg=3`, returns diminish unless the function is very smooth
and ``N`` is large. The default `poly_deg=2` is a good starting point for most
applications.

**Recommended minimum `poly_deg` by application:**

| Application | Minimum `poly_deg` | Why |
|---|---|---|
| Interpolation | 2 | Exact for quadratic fields |
| First derivatives (∂, ∇) | 2 | One order lost to differentiation |
| Laplacian (∇²) | 2–3 | Two orders lost; 3 if accuracy matters |
| High-order PDEs | 3–4 | Need headroom for multiple differentiations |

## k-Refinement: Error vs. Stencil Size

The stencil size ``k`` (number of nearest neighbors) controls the local system size.
`autoselect_k` uses the formula from Bayona (2017):
``k = \max\!\big(2\binom{m+d}{d},\; 2d+1\big)`` where ``m`` is the polynomial degree
and ``d`` is the spatial dimension.

```@example convergence
pts_k = scattered_points(30)  # N = 900
vals_k = g.(pts_k)
exact_k = ∇²g.(pts_k)

k_range = 6:5:81
auto_k_2 = RadialBasisFunctions.autoselect_k(pts_k, PHS(3; poly_deg=2))
auto_k_3 = RadialBasisFunctions.autoselect_k(pts_k, PHS(3; poly_deg=3))

err_k_pd2 = Float64[]
err_k_pd3 = Float64[]

for k in k_range
    basis2 = PHS(3; poly_deg=2)
    basis3 = PHS(3; poly_deg=3)
    # Skip k values too small for the polynomial system
    if k >= RadialBasisFunctions.autoselect_k(pts_k, basis2)
        lap2 = laplacian(pts_k; basis=basis2, k=k)
        push!(err_k_pd2, nrmse(lap2(vals_k), exact_k))
    else
        push!(err_k_pd2, NaN)
    end
    if k >= RadialBasisFunctions.autoselect_k(pts_k, basis3)
        lap3 = laplacian(pts_k; basis=basis3, k=k)
        push!(err_k_pd3, nrmse(lap3(vals_k), exact_k))
    else
        push!(err_k_pd3, NaN)
    end
end

fig = Figure(; size=(600, 400))
ax = Axis(fig[1, 1];
    xlabel="Stencil size k", ylabel="Normalized RMSE",
    yscale=log10,
    title="Laplacian Error vs. Stencil Size (N = 900)")
scatterlines!(ax, collect(k_range), err_k_pd2;
    label="poly_deg=2", linewidth=2, markersize=8)
scatterlines!(ax, collect(k_range), err_k_pd3;
    label="poly_deg=3", linewidth=2, markersize=8)
vlines!(ax, [auto_k_2]; color=Cycled(1), linestyle=:dash, linewidth=1.5,
    label="auto k (p=2) = $auto_k_2")
vlines!(ax, [auto_k_3]; color=Cycled(2), linestyle=:dash, linewidth=1.5,
    label="auto k (p=3) = $auto_k_3")
axislegend(ax; position=:rt)
fig
```

Error is minimized somewhat above the `autoselect_k` value (dashed lines) — not
exactly at it — and then grows monotonically with ``k``. Two effects combine to
produce this: distant points degrade the local approximation (a polynomial of
fixed degree cannot represent function variation across an enlarged
neighborhood), and near-boundary stencils become increasingly one-sided. The
`autoselect_k` default is near-optimal, and increasing ``k`` beyond ~2× the
default wastes computation and hurts accuracy.

**`autoselect_k` values by polynomial degree and dimension:**

| `poly_deg` | 2D | 3D |
|---|---|---|
| 0 | 5 | 7 |
| 1 | 6 | 8 |
| 2 | 12 | 20 |
| 3 | 20 | 40 |
| 4 | 30 | 70 |

## Basis Comparison

### PHS Order

Each PHS order is paired with a matching polynomial degree
(``\text{poly\_deg} = \lceil n/2 \rceil`` for ``\text{PHS}(n)``).
Under-augmented high-order PHS is ill-posed — PHS(7) with `poly_deg=2`, for
example, fails to converge — and that is a separate issue from which PHS order
is better.

```@example convergence
phs_configs = [
    (PHS(1; poly_deg=1), "PHS1"),
    (PHS(3; poly_deg=2), "PHS3"),
    (PHS(5; poly_deg=3), "PHS5"),
    (PHS(7; poly_deg=4), "PHS7"),
]

phs_errors = Dict(label => Float64[] for (_, label) in phs_configs)

for n_side in n_sides
    pts = scattered_points(n_side)
    vals = g.(pts)
    exact = ∇²g.(pts)
    for (basis, label) in phs_configs
        lap = laplacian(pts; basis=basis)
        push!(phs_errors[label], nrmse(lap(vals), exact))
    end
end

fig = Figure(; size=(600, 400))
ax = Axis(fig[1, 1];
    xlabel="N (number of points)", ylabel="Normalized RMSE",
    xscale=log10, yscale=log10,
    title="PHS Order Comparison (scaled poly_deg)")
for (_, label) in phs_configs
    scatterlines!(ax, Ns, phs_errors[label]; label=label, linewidth=2, markersize=8)
end
axislegend(ax; position=:lb)
fig
```

PHS(1) (``r^1``) is unsuitable for second-order operators — its second derivative
is singular at ``r=0``, producing very large errors. Among the smooth PHS
kernels (3, 5, 7), each achieves its theoretical rate when paired with a matching
polynomial degree, and higher orders reach lower absolute error. The cost is
larger stencils (`autoselect_k` grows with `poly_deg`) and more computation per
stencil. PHS(3) with `poly_deg=2` remains the best default for most problems.

### PHS vs. IMQ vs. Gaussian

```@example convergence
family_configs = [
    (PHS(3; poly_deg=2), "PHS(3)"),
    (IMQ(1.0; poly_deg=2), "IMQ(1.0)"),
    (Gaussian(1.0; poly_deg=2), "Gaussian(1.0)"),
]

family_errors = Dict(label => Float64[] for (_, label) in family_configs)

for n_side in n_sides
    pts = scattered_points(n_side)
    vals = g.(pts)
    exact = ∇²g.(pts)
    for (basis, label) in family_configs
        lap = laplacian(pts; basis=basis)
        push!(family_errors[label], nrmse(lap(vals), exact))
    end
end

fig = Figure(; size=(600, 400))
ax = Axis(fig[1, 1];
    xlabel="N (number of points)", ylabel="Normalized RMSE",
    xscale=log10, yscale=log10,
    title="Basis Family Comparison (poly_deg=2)")
for (_, label) in family_configs
    scatterlines!(ax, Ns, family_errors[label]; label=label, linewidth=2, markersize=8)
end
axislegend(ax; position=:lb)
fig
```

PHS converges algebraically at the rate set by `poly_deg`. IMQ and Gaussian, with
a well-chosen ``\varepsilon``, can converge faster on smooth problems — their
infinite smoothness means convergence is not capped by the polynomial degree. On
this test function, IMQ and Gaussian both reach ~100× error reduction over the
sweep versus ~10× for PHS3. The catch is that this advantage is
``\varepsilon``-dependent, and choosing ``\varepsilon`` well is the main
difficulty with IMQ and Gaussian — covered next.

## Shape Parameter Sensitivity

IMQ and Gaussian have a shape parameter ``\varepsilon`` that controls the basis
function width. Too small (flat) causes ill-conditioning; too large (peaked) reduces
accuracy.

```@example convergence
pts_eps = scattered_points(30)  # N = 900
vals_eps = g.(pts_eps)
exact_eps = ∇²g.(pts_eps)

epsilons = 10.0 .^ range(-1, 1.3; length=20)
imq_err = Float64[]
gauss_err = Float64[]

for ε in epsilons
    lap_imq = laplacian(pts_eps; basis=IMQ(ε; poly_deg=2))
    push!(imq_err, nrmse(lap_imq(vals_eps), exact_eps))

    lap_g = laplacian(pts_eps; basis=Gaussian(ε; poly_deg=2))
    push!(gauss_err, nrmse(lap_g(vals_eps), exact_eps))
end

fig = Figure(; size=(600, 400))
ax = Axis(fig[1, 1];
    xlabel="Shape parameter ε", ylabel="Normalized RMSE",
    xscale=log10, yscale=log10,
    title="Shape Parameter Sensitivity (N = 900, poly_deg=2)")
scatterlines!(ax, epsilons, imq_err; label="IMQ", linewidth=2, markersize=8)
scatterlines!(ax, epsilons, gauss_err; label="Gaussian", linewidth=2, markersize=8)
axislegend(ax; position=:rt)
fig
```

The error curves are U-shaped: there is an optimal ``\varepsilon`` range — on
this test function it sits around ``\varepsilon \approx 0.3\text{–}1`` — but the
optimum is problem- and scale-dependent, and finding it requires experimentation.
This is the main disadvantage of IMQ and Gaussian compared to PHS, which has no
shape parameter.

## Practical Guidelines

**Default recipe:** `PHS(3; poly_deg=2)` with default `k`. This works for most problems.

**When to deviate:**

| Need | Change | Why |
|---|---|---|
| Higher accuracy | Increase `poly_deg` to 3 | Steeper convergence rate |
| Even higher accuracy | `PHS(5; poly_deg=4)` | Better kernel smoothness + higher polynomial |
| Rough or noisy data | `PHS(1; poly_deg=1)` | Lower order avoids oscillations |
| Fine-tuned accuracy | `IMQ(ε)` or `Gaussian(ε)` | Shape parameter allows tuning, but requires effort |
| Faster computation | Decrease `poly_deg` or `k` | Smaller local systems |

**Parameter summary:**

| Parameter | Default | Typical range | Effect |
|---|---|---|---|
| PHS order | 3 | 1, 3, 5, 7 | Modest effect with polynomial augmentation |
| `poly_deg` | 2 | 0–4 | **Dominant factor** in convergence rate |
| `k` | `autoselect_k` | auto to ~2× auto | Saturates quickly; default is near-optimal |
| ``\varepsilon`` (IMQ/Gaussian) | — | 0.1–10 | Problem-dependent; PHS avoids this entirely |

## References

- Bayona, V. (2017). An insight into RBF-FD approximations augmented with polynomials. *Computers & Mathematics with Applications*, 74(5), 1095-1110. [DOI](https://doi.org/10.1016/j.camwa.2017.05.025)
- Flyer, N., Fornberg, B., Bayona, V., & Barnett, G. A. (2016). On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy. *Journal of Computational Physics*, 321, 21-38. [DOI](https://doi.org/10.1016/j.jcp.2016.05.026)
- Franke, R. (1979). A critical comparison of some methods for interpolation of scattered data. *NPS Technical Report* NPS53-79-003.
