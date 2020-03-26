struct NonlocalSystemMatrix{T} <: AbstractArray{T, 2}
    Ξ       ::PositionVector{T} # soa
    elements::TriangleVector{T} # aso
    params  ::Option{T}
end

@inline function NonlocalSystemMatrix(model::Model{T, Triangle{T}}) where T
    NonlocalSystemMatrix(
        PositionVector([e.center for e in model.elements]),
        TriangleVector(model.elements),
        model.params
    )
end

@inline Base.size(A::NonlocalSystemMatrix{T}) where T = (3 * length(A.Ξ), 3 * length(A.elements))

@inline Base.getindex(
    A::NonlocalSystemMatrix{T},
     ::Int
) where T = error("getindex not defined for ", typeof(A))

@inline Base.setindex!(
    A::NonlocalSystemMatrix{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for ", typeof(A))

@inline function Base.show(io::IO, ::MIME"text/plain", A::NonlocalSystemMatrix{T}) where T
    print(io, join(size(A), "×"), " ", typeof(A), "($(repr(A.params)))")
end

@inline function Base.show(io::IO, A::NonlocalSystemMatrix{T}) where T
    print(io, typeof(A))
end

function LinearAlgebra.diag(A::NonlocalSystemMatrix{T}, k::Int = 0) where T
    k != 0 && error("diag not defined for k != 0")
    Kʸ = ReYukawaPotentialMatrix{DoubleLayer}(A.Ξ, A.elements, yukawa(A.params))
    V  = LaplacePotentialMatrix{SingleLayer}(A.Ξ, A.elements)
    σ  = T(2π) .* ones(T, size(A.Ξ, 1))
    [ σ .- diag(Kʸ); diag(V) ; σ ]
end

@inline function LinearAlgebra.mul!(
    Y::AbstractArray{T, 1},
    A::NonlocalSystemMatrix{T},
    v::AbstractArray{T, 1}
) where T
    Y .= A * v
end

@inline function Base.:*(A::NonlocalSystemMatrix{T}, x::AbstractArray{T, 1}) where T
    Array(_mul(A.Ξ, A.elements, CuArray(x), A.params))
end

function _mul(
    Ξ       ::PositionVector{T},
    elements::TriangleVector{T},
    x       ::CuVector{T},
    params  ::Option{T}
) where T
    numelem = length(elements)
    cfg = _kcfg(numelem)
    εΩ  = params.εΩ
    εΣ  = params.εΣ
    ε∞  = params.ε∞
    yuk = yukawa(params)

    V  = LaplacePotentialMatrix{SingleLayer}(Ξ, elements)
    Vʸ = ReYukawaPotentialMatrix{SingleLayer}(Ξ, elements, yuk)

    u  = x[1:numelem]
    q  = x[numelem+1:2numelem]
    w  = x[2numelem+1:3numelem]

    ls = V * q
    ys = (εΩ * (1/ε∞ - 1/εΣ)) .* (Vʸ * q)

    ld = CuArray{T}(undef, 3numelem)
    @cuda config=cfg _mul_ld_kernel!(ld, Ξ, elements, x)

    yd = CuArray{T}(undef, numelem)
    @cuda config=cfg _mul_yd_kernel!(yd, Ξ, elements, x, ε∞/εΣ, yuk)

    [
        T(2π) .* u .+ (εΩ/ε∞) .* ls .+ ld[1:numelem] .+ ys .+ yd;
        T(2π) .* u .- ls .+ ld[numelem+1:2numelem];
        T(2π) .* w .+ (εΩ/ε∞) .* ls .+ ld[2numelem+1:end]
    ]
end

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
