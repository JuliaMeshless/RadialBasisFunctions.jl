"""
Complete AD pipeline for RBF-based PDE solvers.

Demonstrates end-to-end differentiation through:
  pts_flat → _build_weights → W → assemble system → solve A\\b → u → loss

The backward pass through the linear solve uses the implicit function theorem (IFT):
  given ∂L/∂u:   η = Aᵀ\\(∂L/∂u),   ∂L/∂b = η,   ∂L/∂W[i,j] = -η[i]·u[j]  (interior i)

The gradient w.r.t. point positions then flows through the existing _build_weights
Mooncake rule, which converts ∂L/∂W.nzval → ∂L/∂pts.

Physical setup: Laplace equation ∇²u = f on a perturbed 2D grid with Dirichlet BCs.
Test function: f(x,y) = x² + y², ∇²f = 4. Exactly reproduced by PHS(3; poly_deg=2),
so the solver gives machine-precision solution at the nominal configuration.
The RHS b is FIXED at original point positions. As pts move, A(pts) changes but b
stays constant, giving a non-trivial loss gradient w.r.t. pts.

Run with:
    julia --project=. examples/test_AD.jl
"""

using RadialBasisFunctions
import RadialBasisFunctions: _build_weights, Laplacian
using StaticArrays
using LinearAlgebra
using SparseArrays
using Mooncake: Mooncake
using FiniteDifferences: FiniteDifferences, central_fdm, grad
using Random: MersenneTwister
import DifferentiationInterface as DI

# ==============================================================================
# 1. Point cloud setup
# ==============================================================================

rng = MersenneTwister(42)

n_side = 7                          # grid size: n_side × n_side points
N = n_side^2
h = 1.0 / (n_side + 1)             # interior spacing
noise_scale = 0.015                 # small random perturbation

points = [
    SVector{2}(
        i * h + noise_scale * randn(rng),
        j * h + noise_scale * randn(rng),
    )
    for i in 1:n_side for j in 1:n_side
]

println("Point cloud: $(N) points on a perturbed $(n_side)×$(n_side) grid in [0,1]²")

# ==============================================================================
# 2. Topology: adjacency list (fixed — does not change as pts move)
# ==============================================================================

K_NEIGHBORS = 18
adjl = find_neighbors(points, K_NEIGHBORS)
basis = PHS(3; poly_deg = 2)

println("Adjacency: k=$(K_NEIGHBORS) nearest neighbors (fixed topology)")

# ==============================================================================
# 3. Boundary / interior classification
# ==============================================================================

boundary_mask = [
    i == 1 || i == n_side || j == 1 || j == n_side
    for i in 1:n_side for j in 1:n_side
]
boundary_idx = findall(boundary_mask)
interior_idx = findall(.!boundary_mask)

println("Boundary points: $(length(boundary_idx)), interior: $(length(interior_idx))")

# ==============================================================================
# 4. Test function: f(x,y) = x² + y²,  ∇²f = 4  (constant)
#
# PHS(3; poly_deg=2) reproduces quadratics EXACTLY, so the Laplacian operator
# will give W * f_exact = 4 to machine precision at every point.
# ==============================================================================

f_exact(x, y) = x^2 + y^2
lap_f_exact    = 4.0                # ∇²f = 2 + 2 (constant)

f_vals = [f_exact(p[1], p[2]) for p in points]

# ==============================================================================
# 5. RHS vector (FIXED at original point positions)
#
# Interior: b[i] = ∇²f = 4
# Boundary: b[i] = f_exact(x_i, y_i)   (Dirichlet values)
#
# Fixing b is the key design choice: as pts move, only A(pts) changes.
# This makes u = A(pts)⁻¹ b genuinely sensitive to pts positions,
# giving a non-trivial gradient even at the exact solution.
# ==============================================================================

b_rhs = zeros(N)
b_rhs[interior_idx] .= lap_f_exact
b_rhs[boundary_idx]  = f_vals[boundary_idx]

# ==============================================================================
# 6. IFT-based differentiable PDE solve
#
# PDESolveIFT assembles the system matrix from W:
#   A[interior,:] = W[interior,:]       (Laplacian stencil)
#   A[boundary, boundary] = I           (Dirichlet: zero row, set diagonal)
# then solves A u = b.
#
# Mooncake rrule!! uses IFT to avoid differentiating through lu():
#   η    = Aᵀ \ (∂L/∂u)              adjoint solve
#   ∂L/∂b     += η
#   ∂L/∂W[i,j] = -η[i]·u[j]         interior rows only
# ==============================================================================

struct PDESolveIFT
    interior_idx::Vector{Int}
    boundary_idx::Vector{Int}
    is_interior::BitVector
end

function PDESolveIFT(interior_idx::Vector{Int}, boundary_idx::Vector{Int}, N::Int)
    is_int = falses(N)
    is_int[interior_idx] .= true
    return PDESolveIFT(interior_idx, boundary_idx, is_int)
end

function (solver::PDESolveIFT)(W::SparseMatrixCSC{Float64,Int}, b::Vector{Float64})
    A = Matrix(W)
    for i in solver.boundary_idx
        A[i, :] .= 0.0
        A[i, i] = 1.0
    end
    return A \ b
end

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{PDESolveIFT, SparseMatrixCSC{Float64,Int}, Vector{Float64}}

function Mooncake.rrule!!(
        solver_cd::Mooncake.CoDual{PDESolveIFT},
        W::Mooncake.CoDual{SparseMatrixCSC{Float64,Int}},
        b::Mooncake.CoDual{Vector{Float64}},
    )
    solver = Mooncake.primal(solver_cd)
    W_val  = Mooncake.primal(W)
    b_val  = Mooncake.primal(b)

    A = Matrix(W_val)
    for i in solver.boundary_idx
        A[i, :] .= 0.0
        A[i, i] = 1.0
    end
    u = A \ b_val
    u_codual = Mooncake.zero_fcodual(u)

    function pde_solve_pb!!(::Mooncake.NoRData)
        Δu = u_codual.dx
        η  = A' \ Δu              # adjoint solve
        b.dx .+= η                # ∂L/∂b
        ΔW_nzval = W.dx.data.nzval
        rows = rowvals(W_val)
        for j in 1:size(W_val, 2)
            for idx in nzrange(W_val, j)
                i = rows[idx]
                if solver.is_interior[i]
                    ΔW_nzval[idx] -= η[i] * u[j]
                end
            end
        end
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end

    return u_codual, pde_solve_pb!!
end

pde_solver = PDESolveIFT(interior_idx, boundary_idx, N)

# ==============================================================================
# 7. Sanity check: exact polynomial reproduction at nominal configuration
# ==============================================================================

W_nom = _build_weights(Laplacian(), points, points, adjl, basis)
u_nom = pde_solver(W_nom, b_rhs)

residual_rel = norm(u_nom - f_vals) / norm(f_vals)
println("\n--- Sanity check (exact polynomial reproduction) ---")
println("  Relative residual ||u - f_exact|| / ||f_exact|| = $(residual_rel)")
@assert residual_rel < 1e-10 "Solver sanity check failed — residual too large"
println("  ✓ Solver recovers f_exact to machine precision")

# ==============================================================================
# 8. Loss function
#
# loss(pts_flat) = sum(u(pts)²)
#
# At nominal pts: u = f_exact, so loss = ||f_exact||² > 0.
# As pts move (b fixed): A changes, u changes, loss changes non-trivially.
# The gradient is nonzero: ∂loss/∂pts = 2uᵀ·(-A⁻¹·∂A/∂pts·u) ≠ 0.
# ==============================================================================

function loss(pts_flat)
    pts = [SVector{2}(pts_flat[2 * i - 1], pts_flat[2 * i]) for i in 1:N]
    W   = _build_weights(Laplacian(), pts, pts, adjl, basis)
    u   = pde_solver(W, b_rhs)
    return sum(u .^ 2)
end

pts_flat = vcat([collect(p) for p in points]...)

println("\n--- Loss at nominal configuration ---")
println("  loss = $(loss(pts_flat))")

# ==============================================================================
# 9. Gradient via Mooncake (full chain, IFT through linear solve)
# ==============================================================================

println("\n--- Computing Mooncake gradient (first call includes tracing) ---")
backend = DI.AutoMooncake(; config = nothing)
t_mooncake = @elapsed grad_mooncake = DI.gradient(loss, backend, pts_flat)
println("  Elapsed: $(round(t_mooncake; digits=2))s")
println("  Gradient norm: $(norm(grad_mooncake))")
@assert !all(iszero, grad_mooncake) "Mooncake returned zero gradient — something is wrong"
println("  ✓ Gradient is non-zero")

# ==============================================================================
# 10. Validation against finite differences
# ==============================================================================

println("\n--- Finite difference validation ---")
t_fd = @elapsed grad_fd = grad(central_fdm(5, 1), loss, pts_flat)[1]
println("  Elapsed: $(round(t_fd; digits=2))s")

rel_err = norm(grad_mooncake - grad_fd) / norm(grad_fd)
println("  Relative error ||∇_mooncake - ∇_fd|| / ||∇_fd|| = $(rel_err)")
@assert rel_err < 1e-3 "Gradient mismatch: relative error = $(rel_err)"
println("  ✓ Mooncake gradient matches finite differences (rtol < 1e-3)")

println("\nAll checks passed.")
