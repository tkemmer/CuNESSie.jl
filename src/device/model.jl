const CuPosition{T} = NamedTuple{(:x, :y, :z), NTuple{3, T}}

# deprecated
@inline function CuPosition(
    vec::CuDeviceVector{T},
    idx::Int
) where T
    stride = fld(size(vec, 1), 3)
    (x = vec[idx], y = vec[stride + idx], z = vec[2stride + idx])
end

const CuPositionVector{T} = NamedTuple{(:dim, :vec), Tuple{Int, CuDeviceVector{T, AS.Global}}}

@inline Base.size(v::CuPositionVector{T}) where T = (v.dim,)

@inline Base.getindex(
    v::CuPositionVector{T},
    i::Int
) where T = (x = v.vec[i], y = v.vec[v.dim + i], z = v.vec[2 * v.dim + i])

@inline Base.setindex!(
    v::CuPositionVector{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for CuPositionVector{$T}")

const CuTriangle{T} = NamedTuple{
    (:v1, :v2, :v3, :normal, :distorig, :area),
    Tuple{CuPosition{T}, CuPosition{T}, CuPosition{T}, CuPosition{T}, T, T}
}

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
