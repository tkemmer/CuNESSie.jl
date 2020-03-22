const CuPosition{T} = NamedTuple{(:x, :y, :z), NTuple{3, T}}

@inline function CuPosition(
    vec::CuDeviceVector{T},
    idx::Int
) where T
    stride = fld(size(vec, 1), 3)
    (x = vec[idx], y = vec[stride + idx], z = vec[2stride + idx])
end

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
