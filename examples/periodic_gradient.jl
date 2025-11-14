using RadialBasisFunctions
using StaticArraysCore
using Distances
using LinearAlgebra
using Random
using CairoMakie
using NearestNeighbors

# Define periodic metric: periodic in x-direction, regular in y-direction
struct PeriodicXMetric <: Metric
    period::Float64
end

function (d::PeriodicXMetric)(p1, p2)
    dx = abs(p1[1] - p2[1])
    dx = min(dx, d.period - dx)  # Shortest distance through periodicity
    dy = p1[2] - p2[2]
    return sqrt(dx^2 + dy^2)
end

# Setup domain: [0, 2π] × [0, 1]
Random.seed!(42)
period = 2π
n_points = 10_000

# Training data: randomly scattered points
data = [SVector(rand() * period, rand() * period) for _ in 1:n_points]

# Test function: periodic in x
f_test(p) = sin(p[1]) * (1 + p[2])

# Analytical gradient
grad_f_analytical(p) = SVector(cos(p[1]) * (1 + p[2]), sin(p[1]))

# Evaluate function at data points
f_values = f_test.(data)

# Test points for gradient evaluation
n_test = 100
test_points = [SVector(rand() * period, rand()) for _ in 1:n_test]

# Build gradient operator with periodic metric
metric_periodic = PeriodicXMetric(period)
basis = PHS(3; metric=metric_periodic)  # Polyharmonic spline with periodic metric
k_neighbors = 40
grad_op = gradient(data, test_points, basis; k=k_neighbors)

# Compute gradient using RBF operator
grad_tuple = grad_op(f_values)  # Returns (∂f/∂x, ∂f/∂y)
grad_computed = [SVector(grad_tuple[1][i], grad_tuple[2][i]) for i in 1:n_test]

# Compare to analytical gradient
grad_analytical = grad_f_analytical.(test_points)

# Compute error
errors = [norm(grad_computed[i] - grad_analytical[i]) for i in 1:n_test]
max_error = maximum(errors)
mean_error = sum(errors) / n_test

println("Gradient Computation with Periodic Boundary Conditions")
println("="^60)
println(
    "Domain: [0, $(round(period, digits=2))] × [0, $(round(period, digits=2))] (periodic in x)",
)
println("Training points: $n_points")
println("Test points: $n_test")
println("k-neighbors: $k_neighbors")
println()
println("Error Statistics:")
println("  Maximum error: $(max_error)")
println("  Mean error:    $(mean_error)")
println()

# Demonstrate periodic neighbor finding
println("Periodic Neighbor Demonstration:")
println("-"^60)
# Find a point near the right boundary (x ≈ 2π)
right_boundary_idx = argmax([p[1] for p in data])
boundary_point = data[right_boundary_idx]
println(
    "Boundary point: ($(round(boundary_point[1], digits=3)), $(round(boundary_point[2], digits=3)))",
)

# Find its neighbors using the periodic metric
tree = BallTree(data, metric_periodic)
neighbor_indices, distances = knn(tree, [boundary_point], k_neighbors, true)

# Check how many neighbors are on the "other side" (x < π/2)
left_side_neighbors = [data[idx][1] < period / 4 for idx in neighbor_indices[1]]
n_wrapped = sum(left_side_neighbors)

println(
    "Found $n_wrapped neighbors with x < $(round(period/4, digits=2)) (wrapping around)"
)
println("Sample neighbors:")
for i in 1:min(10, k_neighbors)
    idx = neighbor_indices[1][i]
    p = data[idx]
    dist = distances[1][i]
    marker = p[1] < period / 4 ? " ← wrapped!" : ""
    println(
        "  $(i). ($(round(p[1], digits=3)), $(round(p[2], digits=3)))  dist=$(round(dist, digits=3))$marker",
    )
end
println()

println("Periodic gradient computation successful ✓")
println()

## Visualization
println("Creating plots...")

# Create regular grid for plotting functions
nx, ny = 100, 100
x_plot = range(0, period; length=nx)
y_plot = range(0, period; length=ny)
grid_points = [SVector(x, y) for x in x_plot for y in y_plot]

# Compute analytical function on grid
f_analytical_grid = f_test.(grid_points)
f_analytical_matrix = reshape(f_analytical_grid, nx, ny)

# Interpolate function values on grid using RBF
interp = Interpolator(data, f_values, basis)
f_interp_grid = interp(grid_points)
f_interp_matrix = reshape(f_interp_grid, nx, ny)

# Create a coarser grid for gradient computation
nx_grad, ny_grad = 100, 100
x_grad = range(0, period; length=nx_grad)
y_grad = range(0, period; length=ny_grad)
grad_grid_points = [SVector(x, y) for x in x_grad for y in y_grad]

# Compute analytical gradient on coarse grid
grad_analytical_grid = grad_f_analytical.(grad_grid_points)
u_analytical = [g[1] for g in grad_analytical_grid]
v_analytical = [g[2] for g in grad_analytical_grid]
u_analytical_matrix = reshape(u_analytical, nx_grad, ny_grad)
v_analytical_matrix = reshape(v_analytical, nx_grad, ny_grad)

# Compute RBF gradient on coarse grid
grad_op_grid = gradient(data, grad_grid_points, basis; k=k_neighbors)
grad_rbf_tuple = grad_op_grid(f_values)
u_rbf = grad_rbf_tuple[1]
v_rbf = grad_rbf_tuple[2]
u_rbf_matrix = reshape(u_rbf, nx_grad, ny_grad)
v_rbf_matrix = reshape(v_rbf, nx_grad, ny_grad)

## Create 3x2 figure layout

fig = Figure(; size=(1200, 1400))

# Row 1, Left: Analytical function
ax1 = Axis(
    fig[1, 1];
    xlabel="x",
    ylabel="y",
    title="Analytical Function: f(x,y) = sin(x)(1+y)",
    aspect=DataAspect(),
)
hm1 = heatmap!(ax1, x_plot, y_plot, f_analytical_matrix'; colormap=:viridis)
Colorbar(fig[1, 2], hm1; label="f")

# Row 1, Right: Interpolated function
ax2 = Axis(
    fig[1, 3];
    xlabel="x",
    ylabel="y",
    title="RBF Interpolated Function",
    aspect=DataAspect(),
)
hm2 = heatmap!(ax2, x_plot, y_plot, f_interp_matrix'; colormap=:viridis)
Colorbar(fig[1, 4], hm2; label="f")

# Row 2, Left: Analytical ∂f/∂x
ax3 = Axis(fig[2, 1]; xlabel="x", ylabel="y", title="Analytical ∂f/∂x", aspect=DataAspect())
hm3 = heatmap!(ax3, x_grad, y_grad, u_analytical_matrix'; colormap=:RdBu)
Colorbar(fig[2, 2], hm3; label="∂f/∂x")

# Row 2, Right: RBF ∂f/∂x
ax4 = Axis(fig[2, 3]; xlabel="x", ylabel="y", title="RBF ∂f/∂x", aspect=DataAspect())
hm4 = heatmap!(ax4, x_grad, y_grad, u_rbf_matrix'; colormap=:RdBu)
Colorbar(fig[2, 4], hm4; label="∂f/∂x")

# Row 3, Left: Analytical ∂f/∂y
ax5 = Axis(fig[3, 1]; xlabel="x", ylabel="y", title="Analytical ∂f/∂y", aspect=DataAspect())
hm5 = heatmap!(ax5, x_grad, y_grad, v_analytical_matrix'; colormap=:RdBu)
Colorbar(fig[3, 2], hm5; label="∂f/∂y")

# Row 3, Right: RBF ∂f/∂y
ax6 = Axis(fig[3, 3]; xlabel="x", ylabel="y", title="RBF ∂f/∂y", aspect=DataAspect())
hm6 = heatmap!(ax6, x_grad, y_grad, v_rbf_matrix'; colormap=:RdBu)
Colorbar(fig[3, 4], hm6; label="∂f/∂y")

save("periodic_gradient.png", fig)
println("Plot saved as periodic_gradient.png")

fig
