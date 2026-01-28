# Script to generate README demo visualization
# Run with: julia --project=docs docs/src/assets/generate_demo.jl

using RadialBasisFunctions
using StaticArrays
using CairoMakie
using DelimitedFiles
using Random

set_theme!(theme_light())

# Load Makie's volcano dataset
volcano = readdlm(Makie.assetpath("volcano.csv"), ',', Float64)'  # transpose for (x, y) layout
nx, ny = size(volcano)

# Create normalized grid coordinates [0, 1]
xs = range(0, 1, nx)
ys = range(0, 1, ny)

# Create all grid points and subsample randomly
all_points = [SVector(x, y) for x in xs, y in ys] |> vec
all_values = vec(volcano)

Random.seed!(42)
n_samples = 500
sample_idx = randperm(length(all_points))[1:n_samples]
points = all_points[sample_idx]
values = all_values[sample_idx]

# Build interpolator
interp = Interpolator(points, values)

# Interpolate to grid for plotting
eval_points = [SVector(x, y) for x in xs, y in ys]
z_interp = [interp(p) for p in eval_points]

# Compute gradient
grad_op = gradient(points; k = 30)
grad_vals = grad_op(values)

# Build RBF interpolators for gradient components
interp_gx = Interpolator(points, grad_vals[:, 1])
interp_gy = Interpolator(points, grad_vals[:, 2])

grad_mag = sqrt.(grad_vals[:, 1] .^ 2 .+ grad_vals[:, 2] .^ 2)
interp_mag = Interpolator(points, grad_mag)
mag_interp = [interp_mag(p) for p in eval_points]

azimuth = 0.8π
elev = 0.2π

## Main figure with 3 panels in a row
fig = Figure(size = (1500, 450), fontsize = 14)

# Panel 1: Original data scatter (3D)
ax1 = Axis3(
    fig[1, 1],
    title = "Survey Points",
    xlabel = "x",
    ylabel = "y",
    zlabel = "Elevation (m)",
    azimuth = azimuth,
    elevation = elev,
)
scatter!(ax1, [p[1] for p in points], [p[2] for p in points], values; color = values, colormap = :terrain, markersize = 8)

# Panel 2: RBF Reconstruction surface
ax2 = Axis3(
    fig[1, 2],
    title = "RBF Reconstruction",
    xlabel = "x",
    ylabel = "y",
    zlabel = "Elevation (m)",
    azimuth = azimuth,
    elevation = elev,
)
surface!(ax2, xs, ys, z_interp; colormap = :terrain)

# Panel 3: Surface colored by gradient magnitude
ax3 = Axis3(
    fig[1, 3],
    title = "Gradient Magnitude (|∇f|)",
    xlabel = "x",
    ylabel = "y",
    zlabel = "Elevation (m)",
    azimuth = azimuth,
    elevation = elev,
)

surface!(ax3, xs, ys, z_interp; color = mag_interp, colormap = :plasma)

save(joinpath(@__DIR__, "interpolation_demo.png"), fig; px_per_unit = 4)

## Second figure: Gradient field and ascent path
fig2 = Figure(size = (1000, 450), fontsize = 14);

# Panel 1: Gradient field with streamlines
ax_grad = Axis(fig2[1, 1], title = "Gradient Field (∇f)", xlabel = "x", ylabel = "y", aspect = 1)

contourf!(ax_grad, xs, ys, z_interp; colormap = :terrain, levels = 20)

f(x, y) = Point2f(-interp_gx(SVector(x, y)), -interp_gy(SVector(x, y)))

streamplot!(
    ax_grad, f, 0.0 .. 1.0, 0.0 .. 1.0;
    colormap = :grays,
    arrow_size = 10,
    linewidth = 1.5,
    stepsize = 0.01,
)

# Panel 2: Gradient descent path
ax_descent = Axis(fig2[1, 2], title = "Gradient Descent Path", xlabel = "x", ylabel = "y", aspect = 1)

contourf!(ax_descent, xs, ys, z_interp; colormap = :terrain, levels = 20)

function gradient_descent(start; step_size = 0.005, max_iters = 500, tol = 1.0)
    path = [start]
    pos = start
    for _ in 1:max_iters
        gx = interp_gx(SVector(pos...))
        gy = interp_gy(SVector(pos...))
        grad_norm = sqrt(gx^2 + gy^2)
        grad_norm < tol && break  # Stop when gradient flattens out
        # Normalize and step downhill (negative gradient)
        pos = (pos[1] - step_size * gx / grad_norm, pos[2] - step_size * gy / grad_norm)
        # Stay in bounds
        pos = (clamp(pos[1], 0.02, 0.98), clamp(pos[2], 0.02, 0.98))
        push!(path, pos)
    end
    return path
end

function plot_descent_path!(ax, start_point; color = :black)
    path = gradient_descent(start_point)
    lines!(ax, [p[1] for p in path], [p[2] for p in path]; color = color, linewidth = 2)
    scatter!(ax, [start_point[1]], [start_point[2]]; color = color, markersize = 10, marker = :circle)
    return scatter!(ax, [path[end][1]], [path[end][2]]; color = color, markersize = 12, marker = :star5)
end

starting_points = [(0.3 + 0.4 * rand(), 0.2 + 0.4 * rand()) for _ in 1:20]
for pt in starting_points
    plot_descent_path!(ax_descent, pt)
end

save(joinpath(@__DIR__, "gradient_field.png"), fig2; px_per_unit = 4)
