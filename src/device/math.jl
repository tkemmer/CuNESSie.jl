@inline function _abs(x::T) where T
    CUDA.abs(x)
end

@inline function _asin(x::T) where T
    CUDA.asin(x)
end

@inline function _cathethus(hyp::T, cos_theta::T) where T
    CUDA.sqrt(hyp * hyp * (1 - cos_theta * cos_theta))
end

@inline function _clamp(x::T, lo::T, hi::T) where T
    CUDA.max(lo, CUDA.min(hi, x))
end

@inline function _cos(u::CuPosition{T}, unorm::T, v::CuPosition{T}, vnorm::T) where T
    _dot(u, v) / (unorm * vnorm)
end

@inline function _cross(u::CuPosition{T}, v::CuPosition{T}) where T
    (
        x = u.y * v.z - u.z * v.y,
        y = u.z * v.x - u.x * v.z,
        z = u.x * v.y - u.y * v.x
    )
end

@inline function _dot(u::CuPosition{T}, v::CuPosition{T}) where T
    CUDA.fma(u.x, v.x, CUDA.fma(u.y, v.y, u.z * v.z))
end

@inline function _log(x::T) where T
    CUDA.log(x)
end

@inline function _norm(v::CuPosition{T}) where T
    CUDA.sqrt(_dot(v, v))
end

@inline function _sign(x::T) where T
    x == 0 ? zero(x) : x / _abs(x)
end

@inline function _sqrt(x::T) where T
    CUDA.sqrt(x)
end

@inline function _smul(v::CuPosition{T}, s::T) where T
    (x = s * v.x, y = s * v.y, z = s * v.z)
end

@inline function _sub(u::CuPosition{T}, v::CuPosition{T}) where T
    (x = u.x - v.x, y = u.y - v.y, z = u.z - v.z)
end
