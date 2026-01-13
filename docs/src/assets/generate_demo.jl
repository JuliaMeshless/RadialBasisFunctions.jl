# Script to generate README demo visualization
# Run with: julia --project=. docs/src/assets/generate_demo.jl

using RadialBasisFunctions
using StaticArrays
using CairoMakie
using Random

# Set up the figure theme
set_theme!(theme_light())

# Generate scattered data points
Random.seed!(42)
n_points = 300
points = [SVector{2}(rand(2)) for _ in 1:n_points]

# Target function - interesting pattern
f(x) = sin(3π * x[1]) * cos(3π * x[2]) + 0.5 * sin(5π * x[1] * x[2])
values = f.(points)

# Build interpolator
interp = Interpolator(points, values, PHS(3))

# Create evaluation grid
nx, ny = 100, 100
xs = range(0, 1, length = nx)
ys = range(0, 1, length = ny)
grid_points = [SVector(x, y) for y in ys, x in xs]
z_interp = [interp(p) for p in grid_points]

# Also compute true function for comparison
z_true = [f(p) for p in grid_points]

# Create figure
fig = Figure(size = (1000, 400), fontsize = 14)

# Panel 1: Scattered data points
ax1 = Axis3(
    fig[1, 1],
    title = "Scattered Data (300 points)",
    xlabel = "x", ylabel = "y", zlabel = "f(x,y)",
    azimuth = 0.4π,
    elevation = 0.15π
)
scatter!(
    ax1,
    [p[1] for p in points],
    [p[2] for p in points],
    values,
    color = values,
    colormap = :viridis,
    markersize = 8
)

# Panel 2: RBF Interpolated surface
ax2 = Axis3(
    fig[1, 2],
    title = "RBF Interpolation",
    xlabel = "x", ylabel = "y", zlabel = "f(x,y)",
    azimuth = 0.4π,
    elevation = 0.15π
)
surface!(
    ax2, xs, ys, z_interp,
    colormap = :viridis,
    shading = true
)

# Save the figure
save(joinpath(@__DIR__, "interpolation_demo.png"), fig, px_per_unit = 2)

println("Saved interpolation_demo.png")
