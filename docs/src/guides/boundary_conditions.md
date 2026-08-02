# Boundary Conditions

For PDE applications, operators support **Hermite interpolation** with boundary conditions.
Use this when you need to enforce Dirichlet, Neumann, or Robin conditions at boundary nodes
while solving the PDE only at interior nodes.

For the mathematics — why standard collocation breaks down near boundaries and how the
Hermite formulation restores symmetry — see
[Hermite Approach for Boundary Stencils](@ref) in the theory reference.

## Boundary Condition Types

| Type | Constructor | Meaning |
|------|-------------|---------|
| Dirichlet | `Dirichlet()` | Value specified: ``u = g`` |
| Neumann | `Neumann()` | Normal derivative specified: ``\partial u/\partial n = g`` |
| Robin | `Robin(α, β)` | Mixed: ``\alpha u + \beta \, \partial u/\partial n = g`` |
| Internal | `Internal()` | Interior point (no boundary condition) |

`Dirichlet`, `Neumann`, and `Robin` are public but **not exported** — downstream physics
packages (e.g. Macchiato.jl) export their own boundary-condition types with these names.
Import them explicitly:

```julia
using RadialBasisFunctions: Dirichlet, Neumann, Robin
```

## Setting Up a Domain

Hermite stencils need a *real* boundary with meaningful outward normals. Below is a unit
disk: interior nodes scattered inside, boundary nodes placed on the circle itself.

```@example bcs
using RadialBasisFunctions
using RadialBasisFunctions: Dirichlet, Neumann, Robin
using StaticArrays
using LinearAlgebra: norm, normalize
using Random

Random.seed!(42)

# Boundary nodes on the unit circle
n_boundary = 120
θ = range(0, 2π; length=n_boundary + 1)[1:(end - 1)]
boundary_pts = map(t -> SVector(cos(t), sin(t)), θ)

# Interior nodes scattered inside the disk
interior_pts = SVector{2,Float64}[]
while length(interior_pts) < 700
    p = 2 * rand(SVector{2,Float64}) .- 1
    norm(p) < 0.95 && push!(interior_pts, p)
end

points = vcat(interior_pts, boundary_pts)
is_boundary = [i > length(interior_pts) for i in eachindex(points)]
boundary_indices = findall(is_boundary)

# Outward unit normals — radial on a disk
normals = normalize.(points[boundary_indices])
length(points), count(is_boundary)
```

## The `hermite` Keyword

Pass a named tuple with three fields to any operator constructor. `hermite` is
**keyword-only** (since v0.6):

- `is_boundary` — a `Bool` per data point, marking which points lie on the boundary
- `bc` — a boundary condition per boundary point, ordered as `findall(is_boundary)`
- `normals` — an outward unit normal per boundary point, same order

```@example bcs
bcs = fill(Dirichlet(), length(boundary_indices))

lap = laplacian(points; hermite=(
    is_boundary=is_boundary,
    bc=bcs,
    normals=normals
))
typeof(lap)
```

We can check it against a harmonic function, ``u = x^2 - y^2``, for which ``\nabla^2 u = 0``:

```@example bcs
u(p) = p[1]^2 - p[2]^2

result = lap(u.(points))
maximum(abs, result[.!is_boundary])   # ≈ 0 at interior nodes
```

## What Goes Into the Input Vector

This is the part that trips people up. A Hermite operator assembles as

```math
\mathcal{L}u^h(\mathbf{x}_i) = \sum_{j \in \mathcal{X}_{i,I}} w_j\, u(\mathbf{x}_j)
                             + \sum_{j \in \mathcal{X}_{i,B}} w_j\, g(\mathbf{x}_j)
```

so entries of the vector you pass mean different things depending on the node:

| Node | What that entry must hold |
|:---|:---|
| Interior | the field value ``u`` |
| Dirichlet boundary | ``g = u`` — same thing, so passing `u` everywhere just works |
| Neumann boundary | ``g = \partial u/\partial n`` — **not** ``u`` |
| Robin boundary | ``g = \alpha u + \beta\, \partial u/\partial n`` |

Dirichlet is forgiving because its boundary data *is* the field value. Neumann is not. On
the unit circle, ``\partial u/\partial n = \nabla u \cdot \hat{n} = 2x^2 - 2y^2``:

```@example bcs
∂u∂n(p) = 2 * p[1]^2 - 2 * p[2]^2

lap_neumann = laplacian(points; hermite=(
    is_boundary=is_boundary,
    bc=fill(Neumann(), length(boundary_indices)),
    normals=normals
))

v = map((b, p) -> b ? ∂u∂n(p) : u(p), is_boundary, points)
maximum(abs, lap_neumann(v)[.!is_boundary])
```

Passing `u` at the Neumann nodes instead gives an error many orders of magnitude larger —
the operator is doing exactly what it was built to do, just with the wrong data.

Condition types can be mixed freely; build the `bc` vector however the geometry requires,
and populate the input vector per the table above.

## Where It Applies

The Hermite treatment is applied **only to stencils that include boundary nodes**. Interior
stencils far from the boundary use the standard RBF-FD formulation unchanged, so there is
no cost for problems where boundary effects don't reach.

The same `hermite` keyword works on operators built with [`@operator`](@ref) — see
[Building PDE Operators](@ref) — and on [Custom Operators](@ref "Custom Operators").

## Alternative: Boundary Nodes as Unknowns

For multi-region or coupled problems it can be preferable to solve the governing equation
at boundary nodes too, treating all nodes as unknowns. That approach keeps the standard RBF
basis everywhere and modifies only the right-hand side at boundary evaluation points — it
is simpler than the Hermite method. See
[Constructing an Operator Treating Boundary Nodes as Unknowns](@ref) for the formulation.
