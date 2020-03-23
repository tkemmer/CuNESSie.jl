function rfenergy(bem::R) where {T, R <: NESSie.BEM.BEMResult{T}}
    cuΞ     = PositionVector([c.pos for c in bem.model.charges])
    cuelms  = TriangleVector(bem.model.elements)

    Ṽ  = LaplacePotentialMatrix{SingleLayer}(cuΞ, cuelms)
    W  = LaplacePotentialMatrix{DoubleLayer}(cuΞ, cuelms)

    (Ṽ * bem.q .- W * bem.u) ⋅ [c.val for c in bem.model.charges] * T(55.27935812569474)
end
