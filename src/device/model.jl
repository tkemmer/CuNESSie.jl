const CuPosition{T} = NamedTuple{(:x, :y, :z), NTuple{3, T}}

# deprecated
@inline function CuPosition(
    vec::CuDeviceVector{T},
    idx::Int
) where T
    stride = fld(size(vec, 1), 3)
    (x = vec[idx], y = vec[stride + idx], z = vec[2stride + idx])
end

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

# deprecated
@inline function CuTriangle(
    vec::CuDeviceVector{T},
    idx::Int
) where T
    idx = (idx - 1) * 14 + 1
    (
        v1 = (x = vec[idx], y = vec[idx + 1], z = vec[idx + 2]),
        v2 = (x = vec[idx + 3], y = vec[idx + 4], z = vec[idx + 5]),
        v3 = (x = vec[idx + 6], y = vec[idx + 7], z = vec[idx + 8]),
        normal = (x = vec[idx + 9], y = vec[idx + 10], z = vec[idx + 11]),
        distorig = vec[idx + 12],
        area = vec[idx + 13]
    )
end

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
