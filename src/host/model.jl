struct PositionVector{T} <: NoAccessArray{T, 1}
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

@inline Adapt.adapt_structure(
    a::CUDA.Adaptor,
    v::PositionVector{T}
) where T = CuPositionVector(v.dim, Adapt.adapt_storage(a, v.vec))

struct TriangleVector{T} <: NoAccessArray{T, 1}
    dim::Int
    vec::CuVector{T}
end

function TriangleVector(
    elements::Vector{Triangle{T}}
) where T
    vec = Vector{T}(undef, 14 * length(elements))
    for i in eachindex(elements)
        elm = elements[i]
        i = 14 * (i - 1)
        view(vec,  i+1:i+3)  .= elm.v1
        view(vec,  i+4:i+6)  .= elm.v2
        view(vec,  i+7:i+9)  .= elm.v3
        view(vec, i+10:i+12) .= elm.normal
        vec[i+13] = elm.distorig
        vec[i+14] = elm.area
    end
    TriangleVector(length(elements), CuArray(vec))
end

@inline Base.size(v::TriangleVector{T}) where T = (v.dim,)

@inline Adapt.adapt_structure(
    a::CUDA.Adaptor,
    v::TriangleVector{T}
) where T = CuTriangleVector(v.dim, Adapt.adapt_storage(a, v.vec))
