struct NonlocalSystemOutputs{T} <: AbstractArray{T, 1}
    β      ::Vector{T}
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
        ((εΩ/ε∞) .* (Ṽ * qmol)) .+ ((εΩ * (1/εΣ - 1/ε∞)) .* (ṼʸmṼ * qmol))
    )
end

@inline Base.size(v::NonlocalSystemOutputs{T}) where T = 3 .* Base.size(v.β)

@inline function Base.getindex(v::NonlocalSystemOutputs{T}, i::Int) where T
    @boundscheck checkbounds(v, i)
    @inbounds i ∈ 1:size(v.β, 1) ? Base.getindex(v.β, i) : zero(T)
end

@inline Base.setindex(
    v::NonlocalSystemOutputs{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for ", typeof(v))

struct NonlocalSystem{T}
    model::Model{T, Triangle{T}}
    A    ::NonlocalSystemMatrix{T}
    b    ::NonlocalSystemOutputs{T}
    umol ::Vector{T}
    qmol ::Vector{T}
end

function NonlocalSystem(model::Model{T, Triangle{T}}) where T
    cuΞ    = PositionVector([e.center for e in model.elements])
    cuelms = TriangleVector(model.elements)

    A    = NonlocalSystemMatrix(cuΞ, cuelms, model.params)
    umol = model.params.εΩ .\   φmol(model, tolerance=_etol(T))
    qmol = model.params.εΩ .\ ∂ₙφmol(model)
    b    = NonlocalSystemOutputs(A, umol, qmol)

    NonlocalSystem(model, A, b, umol, qmol)
end

function solve(sys::NonlocalSystem{T}) where T
    numξ = length(sys.A.Ξ)
    cauchy  = _solve_linear_system(sys.A, sys.b)
    NonlocalBEMResult(
        sys.model,
        view(cauchy, 1:numξ),
        view(cauchy, 1+numξ:2numξ),
        view(cauchy, 1+2numξ:3numξ),
        sys.umol,
        sys.qmol
    )
end

@inline solve(
         ::Type{NonlocalES},
    model::Model{T, Triangle{T}}
) where T = solve(NonlocalSystem(model))
