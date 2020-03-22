struct PositionVector{T} <: AbstractArray{T, 1}
    dim::Int
    vec::CuVector{T}
end

PositionVector(
    pos::Vector{Vector{T}}
) where T = PositionVector(
    length(pos),
    CuArray([
        [p[1] for p in pos];
        [p[2] for p in pos];
        [p[3] for p in pos]
    ])
)

@inline Base.size(v::PositionVector{T}) where T = (v.dim,)

@inline Base.getindex(
    v::PositionVector{T},
     ::Int
) where T = error("getindex not defined for ", typeof(v))

@inline Base.setindex!(
    v::PositionVector{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for ", typeof(v))

@inline Base.convert(
     ::Type{CuPositionVector{T}},
    v::PositionVector{T}
) where T = (dim = v.dim, vec = convert(CuDeviceVector{T, AS.Global}, v.vec))

@inline Adapt.adapt_storage(
    a::CUDAnative.Adaptor,
    v::PositionVector{T}
) where T = CuPositionVector(v.dim, Adapt.adapt_storage(a, v.vec))

# deprecated
@inline function Ξ2device(pos::Vector{Vector{T}}) where T
    CuArray([
        [p[1] for p in pos];
        [p[2] for p in pos];
        [p[3] for p in pos]
    ])
end

@inline function elements2device(elements::Vector{Triangle{T}}) where T
    CuArray(NESSie.unpack(
        [[e.v1; e.v2; e.v3; e.normal; e.distorig; e.area] for e in elements]
    ))
end
