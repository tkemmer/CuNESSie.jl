function rfenergy(bem::NESSie.BEM.NonlocalBEMResult{T}) where T
    cuΞ     = PositionVector([c.pos for c in bem.model.charges])
    cuelms  = TriangleVector(bem.model.elements)

    V  = LaplacePotentialMatrix{SingleLayer}(cuΞ, cuelms)
    K  = LaplacePotentialMatrix{DoubleLayer}(cuΞ, cuelms)

    (V * bem.q .- K * bem.u) ⋅ [c.val for c in bem.model.charges] * T(110.56123849403735)
end
