struct PostProcessor{T, R <: BEMResult{T}}
    bem     ::R
    elements::TriangleVector{T}
end

@inline function PostProcessor(bem::R) where {T, R <: BEMResult{T}}
    PostProcessor(bem, TriangleVector(bem.model.elements))
end

function φ⃰Ω(Ξ::Vector{Vector{T}}, proc::PostProcessor{T}) where T
    Ξ = PositionVector(Ξ)
    Ṽ = LaplacePotentialMatrix{SingleLayer}(Ξ, proc.elements)
    W = LaplacePotentialMatrix{DoubleLayer}(Ξ, proc.elements)

    (NESSie.potprefactor(T) / T(4π)) .* (Ṽ * proc.bem.q .- W * proc.bem.u)
end

function φΩ(Ξ::Vector{Vector{T}}, proc::PostProcessor{T}) where T
    εΩ = proc.bem.model.params.εΩ
    φm  = φmol(Ξ, proc.bem.model.charges)

    φ⃰Ω(Ξ, proc) .+ (NESSie.potprefactor(T) / εΩ .* φm)
end

@inline φΩ(
    Ξ  ::Vector{Vector{T}},
    bem::R
) where {T, R <: BEMResult{T}} = φΩ(Ξ, PostProcessor(bem))

function rfenergy(proc::PostProcessor{T}) where T
    qposs = [c.pos for c in proc.bem.model.charges]
    qvals = [c.val for c in proc.bem.model.charges]

    φ⃰Ω(qposs, proc) ⋅ qvals * T(48.242647748524156)
end

@inline rfenergy(
    bem::R
) where {T, R <: BEMResult{T}} = rfenergy(PostProcessor(bem))

function φΣ(
    Ξ::Vector{Vector{T}},
    proc::PostProcessor{T, LocalBEMResult{T, Triangle{T}}}
) where T
    εΩ = proc.bem.model.params.εΩ
    εΣ = proc.bem.model.params.εΣ
    Ξ  = PositionVector(Ξ)
    Ṽ  = LaplacePotentialMatrix{SingleLayer}(Ξ, proc.elements)
    W  = LaplacePotentialMatrix{DoubleLayer}(Ξ, proc.elements)

    (NESSie.potprefactor(T) / T(4π)) .*
        (W * (proc.bem.u .+ proc.bem.umol) .-
        ((εΩ / εΣ) .* (Ṽ * (proc.bem.q .+ proc.bem.qmol))))
end

function φΣ(
    Ξ::Vector{Vector{T}},
    proc::PostProcessor{T, NonlocalBEMResult{T, Triangle{T}}}
) where T
    εΩ   = proc.bem.model.params.εΩ
    εΣ   = proc.bem.model.params.εΣ
    ε∞   = proc.bem.model.params.ε∞
    yuk  = yukawa(proc.bem.model.params)
    Ξ    = PositionVector(Ξ)
    Ṽ    = LaplacePotentialMatrix{SingleLayer}(Ξ, proc.elements)
    W    = LaplacePotentialMatrix{DoubleLayer}(Ξ, proc.elements)
    ṼʸmṼ = ReYukawaPotentialMatrix{SingleLayer}(Ξ, proc.elements, yuk)
    WʸmW = ReYukawaPotentialMatrix{DoubleLayer}(Ξ, proc.elements, yuk)

    (NESSie.potprefactor(T) / T(4π)) .* (
        (Ṽ * ((-εΩ/ε∞) .* (proc.bem.q .+ proc.bem.qmol)))
        .+ (W * (proc.bem.u .+ proc.bem.umol))
        .+ (ṼʸmṼ * ((εΩ * (1/εΣ - 1/ε∞)) .* (proc.bem.q .+ proc.bem.qmol)))
        .+ (WʸmW * (proc.bem.u .- ((ε∞/εΣ) * proc.bem.w) .+ ((1 - εΩ/εΣ) * proc.bem.umol)))
    )
end

@inline φΣ(
    Ξ  ::Vector{Vector{T}},
    bem::R
) where {T, R <: BEMResult{T}} = φΣ(Ξ, PostProcessor(bem))
