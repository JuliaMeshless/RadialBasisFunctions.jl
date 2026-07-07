# Generate a Julia 3-dot RBF logo variant using Cairo.jl.
#
# Three filled circles in the canonical Julia triangular arrangement
# with concentric rings radiating outward, representing radial basis
# functions centered at each point. Ring opacity decays with distance.
#
# Run:  julia --project=docs docs/src/assets/generate_logo.jl

using Cairo

# ── parameters ──────────────────────────────────────────────────────

const W = 400          # SVG canvas width
const H = 400          # SVG canvas height (square for icon use)

const COLORS = [
    "#389826",  # Julia green  (top-center)
    "#CB3C33",  # Julia red    (bottom-left)
    "#9558B2",  # Julia purple (bottom-right)
]

const DOT_RADIUS = 10.0
const NUM_RINGS = 4
const RING_MIN_R = 25.0
const RING_MAX_R = 90.0
const RING_STROKE = 3.5
const RING_OPACITY_START = 0.85
const RING_OPACITY_END = 0.25

# ── geometry ────────────────────────────────────────────────────────

# Equilateral triangle centered in the canvas, matching Julia logo layout.
# The Julia logo has two dots on top and one on the bottom-center.
const CX = W / 2
const CY = H / 2
const TRI_SIDE = 140.0  # side length of the equilateral triangle
const TRI_HEIGHT = TRI_SIDE * sqrt(3) / 2

# Vertical offset so the triangle is optically centered
const Y_TOP = CY - 2 * TRI_HEIGHT / 3
const Y_BOT = CY + TRI_HEIGHT / 3

const CENTERS = [
    (CX, Y_TOP),                   # top-center   (green)
    (CX - TRI_SIDE / 2, Y_BOT),   # bottom-left  (red)
    (CX + TRI_SIDE / 2, Y_BOT),   # bottom-right (purple)
]

# Ring radii and opacities
const RING_RADII = range(RING_MIN_R, RING_MAX_R; length = NUM_RINGS)
const RING_OPACITIES = [
    RING_OPACITY_START * exp(-2.0 * (i - 1) / (NUM_RINGS - 1))
        for i in 1:NUM_RINGS
]

# ── helpers ─────────────────────────────────────────────────────────

function hex_to_rgb(hex::AbstractString)
    h = lstrip(hex, '#')
    r = parse(UInt8, h[1:2]; base = 16) / 255.0
    g = parse(UInt8, h[3:4]; base = 16) / 255.0
    b = parse(UInt8, h[5:6]; base = 16) / 255.0
    return (r, g, b)
end

# ── draw ────────────────────────────────────────────────────────────

const OUT = joinpath(@__DIR__, "logo.svg")

let
    surface = CairoSVGSurface(OUT, W, H)
    ctx = CairoContext(surface)

    # Concentric rings for each center (drawn first so dots sit on top)
    for (idx, (cx, cy)) in enumerate(CENTERS)
        r, g, b = hex_to_rgb(COLORS[idx])
        for j in NUM_RINGS:-1:1  # largest ring first (painter's order)
            set_source_rgba(ctx, r, g, b, RING_OPACITIES[j])
            set_line_width(ctx, RING_STROKE)
            arc(ctx, cx, cy, RING_RADII[j], 0, 2π)
            stroke(ctx)
        end
    end

    # Filled dots at each center
    for (idx, (cx, cy)) in enumerate(CENTERS)
        r, g, b = hex_to_rgb(COLORS[idx])
        set_source_rgba(ctx, r, g, b, 1.0)
        arc(ctx, cx, cy, DOT_RADIUS, 0, 2π)
        fill(ctx)
    end

    finish(surface)
end

println("Wrote $OUT")
