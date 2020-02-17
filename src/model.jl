const CuPosition{T} = NamedTuple{(:x, :y, :z), NTuple{3, T}}

@inline function CuPosition(
    vec::CuDeviceVector{T},
    idx::Int
) where T
    (x = vec[idx], y = vec[idx + 1], z = vec[idx + 2])
end

const CuTriangle{T} = NamedTuple{
    (:v1, :v2, :v3, :normal, :distorig, :area),
    Tuple{CuPosition{T}, CuPosition{T}, CuPosition{T}, CuPosition{T}, T, T}
}

@inline function CuTriangle(
    vec::CuDeviceVector{T},
    idx::Int
) where T
    (
        v1 = CuPosition(vec, idx),
        v2 = CuPosition(vec, idx + 3),
        v3 = CuPosition(vec, idx + 6),
        normal = CuPosition(vec, idx + 9),
        distorig = vec[idx + 12],
        area = vec[idx + 13]
    )
end
