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
