# API Reference

Full docstrings for every exported name, plus internals for developers extending the
package.

## Macros

```@docs
@operator
```

## Exported Functions

```@autodocs
Modules = [RadialBasisFunctions]
Private = false
Order   = [:function, :type]
```

## Private

```@autodocs
Modules = [RadialBasisFunctions]
Public = false
Order   = [:function, :type, :constant]
```

## EllSparse

Self-contained SELL-C/ELL sparse storage backing the stencil weight matrices. Not re-exported — reach these names as `RadialBasisFunctions.EllSparse.<name>`.

```@autodocs
Modules = [RadialBasisFunctions.EllSparse]
Order   = [:module, :type, :function]
```
