# ------------------------------------------------------------------------------------------
# Adapt: data movement only, never a layout change. `adapt` moves the same flat storage
# to another backend through the trusted constructors (no revalidation), preserving the
# transpose map's position sequence — and therefore adjoint summation order. Change
# layout explicitly with `reslice`, on the host, before uploading.
# ------------------------------------------------------------------------------------------

Adapt.adapt_structure(to, tm::TransposeMap) = TransposeMap(
    Adapt.adapt(to, tm.offsets), Adapt.adapt(to, tm.positions),
    tm.rows === nothing ? nothing : Adapt.adapt(to, tm.rows),
)

Adapt.adapt_structure(to, S::SellStructure{C}) where {C} = SellStructure{C}(
    S.m, S.n, Adapt.adapt(to, S.colind), Adapt.adapt(to, S.sliceptr), S.width,
    S.padded, S.tmap === nothing ? nothing : Adapt.adapt(to, S.tmap),
)

Adapt.adapt_structure(to, A::SellMatrix) =
    SellMatrix(Adapt.adapt(to, A.vals), Adapt.adapt(to, A.structure))

"""
    adapt_family(to, family::NTuple{N, SellMatrix}) -> NTuple{N, SellMatrix}

Adapt a family of same-structure matrices (e.g. gradient components) to a backend,
adapting the shared structure **once** and rebinding each member's values onto it — so
the `===` structure aliasing survives the move. All members must alias one structure
object.
"""
function adapt_family(to, family::NTuple{N, SellMatrix}) where {N}
    N == 0 && return family
    S = first(family).structure
    all(A -> A.structure === S, family) || throw(
        ArgumentError(
            "adapt_family requires all members to alias one structure object; adapt them individually otherwise"
        )
    )
    S_adapted = Adapt.adapt(to, S)
    return map(A -> SellMatrix(Adapt.adapt(to, A.vals), S_adapted), family)
end
