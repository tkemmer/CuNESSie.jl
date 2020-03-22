const CuPosition{T} = NamedTuple{(:x, :y, :z), NTuple{3, T}}

struct CuPositionVector{T} <: AbstractArray{T, 1}
    dim::Int
    vec::CuDeviceVector{T, AS.Global}
end

@inline Base.size(v::CuPositionVector{T}) where T = (v.dim,)

@inline Base.getindex(
    v::CuPositionVector{T},
    i::Int
) where T = (x = v.vec[i], y = v.vec[v.dim + i], z = v.vec[2 * v.dim + i])

@inline Base.setindex!(
    v::CuPositionVector{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for ", typeof(v))

const CuTriangle{T} = NamedTuple{
    (:v1, :v2, :v3, :normal, :distorig, :area),
    Tuple{CuPosition{T}, CuPosition{T}, CuPosition{T}, CuPosition{T}, T, T}
}

struct CuTriangleVector{T} <: AbstractArray{T, 1}
    dim::Int
    vec::CuDeviceVector{T, AS.Global}
end

@inline Base.size(v::CuTriangleVector{T}) where T = (v.dim,)

@inline function Base.getindex(
    v::CuTriangleVector{T},
    i::Int
) where T
    i = (i - 1) * 14 + 1
    (
        v1 = (x = v.vec[i], y = v.vec[i + 1], z = v.vec[i + 2]),
        v2 = (x = v.vec[i + 3], y = v.vec[i + 4], z = v.vec[i + 5]),
        v3 = (x = v.vec[i + 6], y = v.vec[i + 7], z = v.vec[i + 8]),
        normal = (x = v.vec[i + 9], y = v.vec[i + 10], z = v.vec[i + 11]),
        distorig = v.vec[i + 12],
        area = v.vec[i + 13]
    )
end

@inline Base.setindex!(
    v::CuTriangleVector{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for ", typeof(v))
