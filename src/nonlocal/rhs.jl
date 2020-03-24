# TODO
struct NonlocalSystemOutputs{T} <: AbstractArray{T, 1}
    β      ::Vector{T}
    numelem::Int
end

function NonlocalSystemOutputs(
    A   ::NonlocalSystemMatrix{T},
    umol::Vector{T},
    qmol::Vector{T}
) where T
    εΩ   = A.params.εΩ
    εΣ   = A.params.εΣ
    ε∞   = A.params.ε∞
    yuk  = yukawa(A.params)
    Ṽ    = LaplacePotentialMatrix{SingleLayer}(A.Ξ, A.elements)
    W    = LaplacePotentialMatrix{DoubleLayer}(A.Ξ, A.elements)
    ṼʸmṼ = ReYukawaPotentialMatrix{SingleLayer}(A.Ξ, A.elements, yuk)
    WʸmW = ReYukawaPotentialMatrix{DoubleLayer}(A.Ξ, A.elements, yuk)

    NonlocalSystemOutputs(
        W * umol .+ ((1 - εΩ/εΣ) .* (WʸmW * umol)) .- (T(2π) .* umol) .-
        ((εΩ/ε∞) .* (Ṽ * qmol)) .+ ((εΩ * (1/εΣ - 1/ε∞)) .* (ṼʸmṼ * qmol)),
        length(A.elements)
    )
end

@inline Base.size(v::NonlocalSystemOutputs{T}) where T = (3 * v.numelem, )

@inline Base.getindex(
    v::NonlocalSystemOutputs{T},
    i::Int
) where T = i ∈ (v.numelem+1):(3*v.numelem) ? zero(T) : Base.getindex(v.β, i)
# FIXME checkounds message

@inline Base.setindex(
    v::NonlocalSystemOutputs{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for ", typeof(v))
