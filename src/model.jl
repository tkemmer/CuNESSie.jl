const CuPosition{T} = NamedTuple{(:x, :y, :z), NTuple{3, T}}

@inline function CuPosition(
    vec::CuDeviceVector{T},
    idx::Int,
    stride::Int
) where T
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
    (
        v1 = (x = vec[idx], y = vec[idx + 1], z = vec[idx + 2]),
        v2 = (x = vec[idx + 3], y = vec[idx + 4], z = vec[idx + 5]),
        v3 = (x = vec[idx + 6], y = vec[idx + 7], z = vec[idx + 8]),
        normal = (x = vec[idx + 9], y = vec[idx + 10], z = vec[idx + 11]),
        distorig = vec[idx + 12],
        area = vec[idx + 13]
    )
end

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
