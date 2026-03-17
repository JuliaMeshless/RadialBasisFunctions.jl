# PDE Operators Cookbook

Recipes for assembling PDE-specific differential operators via [`custom`](@ref), producing a
**single weight matrix** that applies the full PDE operator in one matrix-vector multiply.

Each example uses the dual-dispatch pattern explained in [Composing Functors](@ref). Refer to
that section for the motivation and mechanics.

```@example pde
using RadialBasisFunctions
using RadialBasisFunctions: ∂, ∂², ∇², MonomialBasis
using StaticArrays

x = [SVector{2}(rand(2)) for _ in 1:100]
f(p) = sin(p[1]) * cos(p[2])
u = f.(x)
```

## Helmholtz Operator

The Helmholtz equation ``\nabla^2 f + k^2 f = 0`` appears in acoustics, electromagnetics,
and quantum mechanics. The operator combines a Laplacian with a scaled identity.

```@example pde
k² = 4.0

function helmholtz(basis)
    lap = ∇²(basis)
    (x, xc) -> lap(x, xc) + k² * basis(x, xc)
end
function helmholtz(basis::MonomialBasis)
    lap = ∇²(basis)
    function (b, x)
        b .= lap(x) .+ k² .* basis(x)
        return nothing
    end
end

helm_op = custom(x, helmholtz; rank=0)

# Verify against separate built-in operators
expected = laplacian(x)(u) .+ k² .* u
maximum(abs, helm_op(u) .- expected)
```

## Anisotropic Diffusion

Anisotropic diffusion ``\kappa_x \, \partial^2 f / \partial x^2 + \kappa_y \, \partial^2 f / \partial y^2``
models heat conduction in materials with direction-dependent conductivity.

```@example pde
κ_x = 2.0
κ_y = 0.5

function aniso_diffusion(basis)
    d²x = ∂²(basis, 1)
    d²y = ∂²(basis, 2)
    (x, xc) -> κ_x * d²x(x, xc) + κ_y * d²y(x, xc)
end
function aniso_diffusion(basis::MonomialBasis)
    d²x = ∂²(basis, 1)
    d²y = ∂²(basis, 2)
    function (b, x)
        b .= κ_x .* d²x(x) .+ κ_y .* d²y(x)
        return nothing
    end
end

aniso_op = custom(x, aniso_diffusion; rank=0)

# Verify against separate built-in operators
expected = κ_x .* partial(x, 2, 1)(u) .+ κ_y .* partial(x, 2, 2)(u)
maximum(abs, aniso_op(u) .- expected)
```

When ``\kappa_x = \kappa_y``, this reduces to a scaled Laplacian.

## Advection-Diffusion

The steady advection-diffusion equation ``\nu \nabla^2 f - \mathbf{c} \cdot \nabla f = 0``
balances viscous diffusion against transport by a velocity field. It appears in fluid dynamics,
pollutant transport, and thermal convection.

```@example pde
ν = 0.01
c = SVector(1.0, 0.5)

function advdiff(basis)
    lap = ∇²(basis)
    dx = ∂(basis, 1)
    dy = ∂(basis, 2)
    (x, xc) -> ν * lap(x, xc) - c[1] * dx(x, xc) - c[2] * dy(x, xc)
end
function advdiff(basis::MonomialBasis)
    lap = ∇²(basis)
    dx = ∂(basis, 1)
    dy = ∂(basis, 2)
    function (b, x)
        b .= ν .* lap(x) .- c[1] .* dx(x) .- c[2] .* dy(x)
        return nothing
    end
end

advdiff_op = custom(x, advdiff; rank=0)

# Verify against separate built-in operators
expected = ν .* laplacian(x)(u) .- c[1] .* partial(x, 1, 1)(u) .- c[2] .* partial(x, 1, 2)(u)
maximum(abs, advdiff_op(u) .- expected)
```

## Sharing Stencils

When multiple `custom` operators act on the same point set, precompute the neighbor list
once and pass it to avoid redundant nearest-neighbor searches:

```@example pde
adjl = find_neighbors(x, 30)

helm_op  = custom(x, helmholtz; rank=0, adjl=adjl)
aniso_op = custom(x, aniso_diffusion; rank=0, adjl=adjl)
```
