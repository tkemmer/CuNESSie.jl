struct PostProcessor{T, R <: NESSie.BEM.BEMResult{T}}
    bem     ::R
    elements::TriangleVector{T}
end

@inline function PostProcessor(bem::R) where {T, R <: NESSie.BEM.BEMResult{T}}
    PostProcessor(bem, TriangleVector(bem.model.elements))
end

function rfenergy(proc::PostProcessor{T}) where T
    cuΞ   = PositionVector([c.pos for c in proc.bem.model.charges])
    qvals = [c.val for c in proc.bem.model.charges]

    Ṽ  = LaplacePotentialMatrix{SingleLayer}(cuΞ, proc.elements)
    W  = LaplacePotentialMatrix{DoubleLayer}(cuΞ, proc.elements)

    (Ṽ * proc.bem.q .- W * proc.bem.u) ⋅ qvals * T(55.27935812569474)
end

@inline rfenergy(
    bem::R
) where {T, R <: NESSie.BEM.BEMResult{T}} = rfenergy(PostProcessor(bem))

function φΩ(Ξ::Vector{Vector{T}}, proc::PostProcessor{T}) where T
    cuΞ = PositionVector(Ξ)
    φm  = φmol(Ξ, proc.bem.model.charges)

    Ṽ = LaplacePotentialMatrix{SingleLayer}(cuΞ, proc.elements)
    W = LaplacePotentialMatrix{DoubleLayer}(cuΞ, proc.elements)

    εΩ = proc.bem.model.params.εΩ
    NESSie.potprefactor(T) .*
        (T(4π) .\ (Ṽ * proc.bem.q .- W * proc.bem.u) .+ εΩ .\ φm)
end

@inline φΩ(
    Ξ  ::Vector{Vector{T}},
    bem::R
) where {T, R <: NESSie.BEM.BEMResult{T}} = φΩ(Ξ, PostProcessor(bem))
