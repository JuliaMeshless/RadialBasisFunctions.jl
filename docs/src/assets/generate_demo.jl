# Script to generate README demo visualization
# Run with: julia --project=docs docs/src/assets/generate_demo.jl

using RadialBasisFunctions
using StaticArrays
using CairoMakie
using Random
using DelimitedFiles

# Set up the figure theme
set_theme!(theme_light())

# Load real Grand Canyon terrain data (from USGS 3DEP)
terrain_file = joinpath(@__DIR__, "terrain_data.csv")
raw_data = readdlm(terrain_file, ',', skipstart = 1)
xs_data = raw_data[:, 1]
ys_data = raw_data[:, 2]
elevations = raw_data[:, 3]

# Get grid dimensions
xs_unique = sort(unique(xs_data))
ys_unique = sort(unique(ys_data))
nx, ny = length(xs_unique), length(ys_unique)

# Create elevation grid (handle missing points)
z_full = fill(NaN, ny, nx)
for idx in eachindex(xs_data)
    i = findfirst(==(xs_data[idx]), xs_unique)
    j = findfirst(==(ys_data[idx]), ys_unique)
    if !isnothing(i) && !isnothing(j)
        z_full[j, i] = elevations[idx]
    end
end

# Sample scattered points from terrain (simulate sparse measurements)
Random.seed!(42)
n_points = 400
total_points = length(xs_data)
sample_indices = unique(rand(1:total_points, n_points * 2))[1:n_points]  # Ensure unique samples

points = [SVector(xs_data[i], ys_data[i]) for i in sample_indices]
values = elevations[sample_indices]

# Build interpolator with sampled data (use PHS5 for smoother interpolation)
interp = Interpolator(points, values, PHS(5))

# Interpolate to full grid
eval_points = [SVector(x, y) for y in ys_unique, x in xs_unique]
z_interp = [interp(p) for p in eval_points]

## Create figure with 3 panels
fig = Figure(size = (1500, 450), fontsize = 14);

azimuth = 0.8π
elevation = 0.2π

# Shared axis limits for consistent 3D views
xlims = (0, 1)
ylims = (0, 1)
zlims = (minimum(elevations), maximum(elevations))

# Panel 1: Scattered data points (sparse measurements)
ax1 = Axis3(
    fig[1, 1],
    title = "Survey Points",
    xlabel = "x",
    ylabel = "y",
    zlabel = "Elevation (m)",
    azimuth = azimuth,
    elevation = elevation,
    limits = (xlims, ylims, zlims),
)
scatter!(
    ax1,
    [p[1] for p in points],
    [p[2] for p in points],
    values;
    color = values,
    colormap = :terrain,
    markersize = 6,
)

# Panel 2: RBF Interpolated surface (reconstruction)
ax2 = Axis3(
    fig[1, 2],
    title = "RBF Reconstruction",
    xlabel = "x",
    ylabel = "y",
    zlabel = "Elevation (m)",
    azimuth = azimuth,
    elevation = elevation,
    limits = (xlims, ylims, zlims),
)
surface!(ax2, xs_unique, ys_unique, z_interp; colormap = :terrain)

# Panel 3: Gradient field with streamlines
ax3 = Axis(fig[1, 3], title = "Gradient Field (∇f)", xlabel = "x", ylabel = "y", aspect = 1)

# Background contour of terrain
contourf!(ax3, xs_unique, ys_unique, z_interp; colormap = :terrain, levels = 20)

# Compute gradient at scattered points
grad_op = gradient(points; k = 30)
grad_vals = grad_op(values)

# Build RBF interpolators for gradient components (positive = uphill)
interp_gx = Interpolator(points, grad_vals[:, 1], PHS(3))
interp_gy = Interpolator(points, grad_vals[:, 2], PHS(3))

# Streamplot function
f(x, y) = Point2f(interp_gx(SVector(x, y)), interp_gy(SVector(x, y)))

streamplot!(
    ax3, f, 0.0 .. 1.0, 0.0 .. 1.0;
    colormap = :grays,
    arrow_size = 8,
    linewidth = 1.5,
    stepsize = 0.01,
)

xlims!(ax3, 0, 1)
ylims!(ax3, 0, 1)

# Save the figure
save(joinpath(@__DIR__, "interpolation_demo.png"), fig; px_per_unit = 2)

println("Saved interpolation_demo.png")
println("Terrain: USGS 3DEP elevation data")
println("Elevation range: $(round(minimum(elevations), digits = 1)) - $(round(maximum(elevations), digits = 1)) m")
